// io_manager.cpp
#include "io_manager.h"
#include "friday_types.h"
#include <cstdio>
#include <stdexcept>
#include <cstring>  // strerror
#include <algorithm> // std::find_if

#ifdef _WIN32
#include <windows.h>
// Sayfa boyutunu almak için
SYSTEM_INFO g_sys_info; // Bir kez alınacak
static bool g_sys_info_initialized = false;
#else // Linux
#include <unistd.h> // open, close, sysconf, aio_read, aio_error, aio_return
#include <fcntl.h>  // O_RDONLY, O_DIRECT
#include <errno.h>  // errno, strerror
#endif

// Statik yardımcı fonksiyon: Sistem sayfa boyutunu bir kez alır.
static size_t get_system_page_size_io() {
    static size_t page_size = 0;
    if (page_size == 0) {
#ifdef _WIN32
        if (!g_sys_info_initialized) {
            GetSystemInfo(&g_sys_info);
            g_sys_info_initialized = true;
        }
        page_size = g_sys_info.dwPageSize; // allocation granularity yerine page size
#else // Linux
        long ps = sysconf(_SC_PAGESIZE);
        if (ps == -1) {
            fprintf(stderr, "WARNING: Could not get system page size for IOManager, defaulting to 4096. Error: %s\n", strerror(errno));
            page_size = 4096; // Fallback
        } else {
            page_size = static_cast<size_t>(ps);
        }
#endif
        if (page_size != IO_ALIGNMENT) {
            fprintf(stderr, "WARNING: IOManager detected system page size (%zu) differs from FRIDAY_TYPES_H::IO_ALIGNMENT (%zu). Ensure O_DIRECT compatibility.\n", page_size, IO_ALIGNMENT);
        }
    }
    return page_size;
}

IOManager::IOManager()
#ifdef _WIN32
    : _file_handle(INVALID_HANDLE_VALUE)
#else // Linux
    : _file_descriptor(-1)
#endif
{
    get_system_page_size_io();
    fprintf(stdout, "IOManager initialized. System page size: %zu bytes.\n", get_system_page_size_io());
}

IOManager::~IOManager() {
    close_model_file();

#ifdef _WIN32
    fprintf(stdout, "IOManager Windows: No explicit pending AIO requests to clear, relying on system.\n");
#else // Linux
    std::lock_guard<std::mutex> lock(this->_callback_mutex);
    for (auto& req_ptr : _pending_requests) {
        if (req_ptr->status == AsyncIOStatus::PENDING) {
            int cancel_res = aio_cancel(req_ptr->fd, &req_ptr->cb);
            if (cancel_res == AIO_CANCELED) {
                fprintf(stderr, "WARNING: Pending AIO request for %p (offset %ld) was canceled during shutdown.\n", req_ptr->buffer, req_ptr->offset);
            } else if (cancel_res == AIO_NOTCANCELED) {
                fprintf(stderr, "WARNING: Pending AIO request for %p (offset %ld) could not be canceled during shutdown.\n", req_ptr->buffer, req_ptr->offset);
            } else if (cancel_res == -1) {
                fprintf(stderr, "ERROR: aio_cancel failed for %p (offset %ld): %s\n", req_ptr->buffer, req_ptr->offset, strerror(errno));
            }
        }
        _callbacks.erase(req_ptr.get());
    }
    _pending_requests.clear();
    fprintf(stdout, "IOManager destroyed, all pending AIO requests cleared.\n");
#endif
}

bool IOManager::open_model_file(const std::string& filepath) {
#ifdef _WIN32
    if (_file_handle != INVALID_HANDLE_VALUE) {
        fprintf(stderr, "WARNING: Model file already open. Closing and reopening.\n");
        close_model_file();
    }

    _file_handle = CreateFileA(
        filepath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, // OS Bypass ve asenkron I/O
        NULL
    );

    if (_file_handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "ERROR: Failed to open model file '%s' with FILE_FLAG_NO_BUFFERING: %lu\n", filepath.c_str(), GetLastError());
        return false;
    }

    fprintf(stdout, "Successfully opened model file '%s' with FILE_FLAG_NO_BUFFERING. Handle: %p\n", filepath.c_str(), _file_handle);
    return true;
#else // Linux
    if (_file_descriptor != -1) {
        fprintf(stderr, "WARNING: Model file already open. Closing and reopening.\n");
        close_model_file();
    }

    _file_descriptor = open(filepath.c_str(), O_RDONLY | O_DIRECT);
    if (_file_descriptor == -1) {
        fprintf(stderr, "ERROR: Failed to open model file '%s' with O_DIRECT: %s\n", filepath.c_str(), strerror(errno));
        return false;
    }

    fprintf(stdout, "Successfully opened model file '%s' with O_DIRECT. FD: %d\n", filepath.c_str(), _file_descriptor);
    return true;
#endif
}

void IOManager::close_model_file() {
#ifdef _WIN32
    if (_file_handle != INVALID_HANDLE_VALUE) {
        if (!CloseHandle(_file_handle)) {
            fprintf(stderr, "ERROR: Failed to close model file handle %p: %lu\n", _file_handle, GetLastError());
        } else {
            fprintf(stdout, "Successfully closed model file handle %p.\n", _file_handle);
        }
        _file_handle = INVALID_HANDLE_VALUE;
    }
#else // Linux
    if (_file_descriptor != -1) {
        if (close(_file_descriptor) == -1) {
            fprintf(stderr, "ERROR: Failed to close model file FD %d: %s\n", _file_descriptor, strerror(errno));
        } else {
            fprintf(stdout, "Successfully closed model file FD %d.\n", _file_descriptor);
        }
        _file_descriptor = -1;
    }
#endif
}

#ifdef _WIN32
VOID CALLBACK IOManager::FileIOCompletionRoutine(
    DWORD dwErrorCode,
    DWORD dwNumberOfBytesTransfered,
    LPOVERLAPPED lpOverlapped
) {
    // lpOverlapped'den AsyncIORequest'i geri al
    // AsyncIORequest'in ilk üyesi OVERLAPPED olmalı ki bu dönüşüm mümkün olsun.
    // (Alternatif: OVERLAPPED'in içinde bir pointer saklayabiliriz.)
    AsyncIORequest* req = reinterpret_cast<AsyncIORequest*>(lpOverlapped);
    IOManager* self = reinterpret_cast<IOManager*>(req->file_handle); // file_handle'ı IOManager pointer'ı olarak kullanıyoruz (PoC basitleştirmesi)

    if (self == nullptr) {
        fprintf(stderr, "ERROR: IOManager::FileIOCompletionRoutine: self pointer is null.\n");
        return;
    }

    AsyncIOStatus final_status = AsyncIOStatus::COMPLETED;
    if (dwErrorCode != 0) {
        fprintf(stderr, "ERROR: Windows AIO read failed for buffer %p, offset %ld, size %zu: %lu\n",
                req->buffer, req->offset, req->size, dwErrorCode);
        final_status = AsyncIOStatus::FAILED;
    } else if (dwNumberOfBytesTransfered != req->size) {
        fprintf(stderr, "WARNING: Windows AIO read requested %zu bytes, but read %lu bytes for buffer %p, offset %ld.\n",
                req->size, dwNumberOfBytesTransfered, req->buffer, req->offset);
    }

    // Callback'i bul ve çağır
    std::lock_guard<std::mutex> lock(self->_callback_mutex_win);
    auto callback_it = self->_callbacks_win.find(lpOverlapped);
    if (callback_it != self->_callbacks_win.end()) {
        callback_it->second(final_status, std::unique_ptr<AsyncIORequest>(req)); // unique_ptr'ı callback'e taşı
        self->_callbacks_win.erase(callback_it);
    } else {
        fprintf(stderr, "WARNING: Completed Windows AIO request %p has no associated callback.\n", lpOverlapped);
        delete req; // Eğer callback yoksa, kendimiz serbest bırakmalıyız.
    }
}
#endif

std::unique_ptr<AsyncIORequest> IOManager::async_read_ssd_block(
    void* buffer,
    size_t size,
    off_t offset,
    std::function<void(AsyncIOStatus, std::unique_ptr<AsyncIORequest>)> callback
) {
#ifdef _WIN32
    if (_file_handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "ERROR: async_read_ssd_block called but no model file is open (Windows).\n");
        return nullptr;
    }
    if (buffer == nullptr || size == 0) {
        throw std::invalid_argument("async_read_ssd_block: buffer cannot be null and size cannot be zero.");
    }

    size_t page_size = get_system_page_size_io();

    // FILE_FLAG_NO_BUFFERING için hizalama ve boyut kontrolü
    if (reinterpret_cast<uintptr_t>(buffer) % page_size != 0) {
        throw std::invalid_argument(
            "async_read_ssd_block (Windows): Buffer address must be page-aligned (" +
            std::to_string(page_size) + " bytes). Buffer: " + std::to_string(reinterpret_cast<uintptr_t>(buffer))
        );
    }
    if (offset % page_size != 0) {
        throw std::invalid_argument(
            "async_read_ssd_block (Windows): Offset must be a multiple of page size (" +
            std::to_string(page_size) + " bytes). Offset: " + std::to_string(offset)
        );
    }
    if (size % page_size != 0) {
        throw std::invalid_argument(
            "async_read_ssd_block (Windows): Size must be a multiple of page size (" +
            std::to_string(page_size) + " bytes). Size: " + std::to_string(size)
        );
    }

    // AsyncIORequest'i ve OVERLAPPED'i oluştur
    // OVERLAPPED'in ilk üye olması önemli, böylece reinterpret_cast güvenli olur.
    auto request_ptr = std::make_unique<AsyncIORequest>();
    request_ptr->buffer = buffer;
    request_ptr->size = size;
    request_ptr->offset = offset;
    request_ptr->status = AsyncIOStatus::PENDING;
    request_ptr->file_handle = reinterpret_cast<HANDLE>(this); // Callback içinde IOManager'a geri dönmek için (PoC basitleştirmesi)

    // OVERLAPPED yapısını doldur
    std::memset(&request_ptr->overlapped, 0, sizeof(OVERLAPPED));
    request_ptr->overlapped.Offset = static_cast<DWORD>(offset);
    request_ptr->overlapped.OffsetHigh = static_cast<DWORD>((uint64_t)offset >> 32);

    // Callback'i takip etmek için sakla
    {
        std::lock_guard<std::mutex> lock(this->_callback_mutex_win);
        _callbacks_win[&request_ptr->overlapped] = callback;
    }

    // Asenkron okuma işlemini başlat
    // ReadFileEx, I/O tamamlandığında FileIOCompletionRoutine'i çağırır.
    if (!ReadFileEx(_file_handle, request_ptr->buffer, static_cast<DWORD>(request_ptr->size), &request_ptr->overlapped, FileIOCompletionRoutine)) {
        DWORD error = GetLastError();
        fprintf(stderr, "ERROR: ReadFileEx failed for buffer %p, offset %ld, size %zu: %lu\n",
                buffer, offset, size, error);

        // Hata durumunda callback'i hemen çağır (veya doğrudan temizle)
        std::lock_guard<std::mutex> lock(this->_callback_mutex_win);
        auto callback_it = _callbacks_win.find(&request_ptr->overlapped);
        if (callback_it != _callbacks_win.end()) {
            callback_it->second(AsyncIOStatus::FAILED, std::move(request_ptr));
            _callbacks_win.erase(callback_it);
        }
        return nullptr;
    }

    fprintf(stdout, "Started Windows AIO read for buffer %p, offset %ld, size %zu.\n", buffer, offset, size);
    return request_ptr;

#else // Linux
    if (_file_descriptor == -1) {
        fprintf(stderr, "ERROR: async_read_ssd_block called but no model file is open.\n");
        return nullptr;
    }
    if (buffer == nullptr || size == 0) {
        throw std::invalid_argument("async_read_ssd_block: buffer cannot be null and size cannot be zero.");
    }

    size_t page_size = get_system_page_size_io();

    // O_DIRECT için hizalama ve boyut kontrolü
    if (reinterpret_cast<uintptr_t>(buffer) % page_size != 0) {
        throw std::invalid_argument(
            "async_read_ssd_block: Buffer address must be page-aligned (" +
            std::to_string(page_size) + " bytes). Buffer: " + std::to_string(reinterpret_cast<uintptr_t>(buffer))
        );
    }
    if (offset % page_size != 0) {
        throw std::invalid_argument(
            "async_read_ssd_block: Offset must be a multiple of page size (" +
            std::to_string(page_size) + " bytes). Offset: " + std::to_string(offset)
        );
    }
    if (size % page_size != 0) {
        throw std::invalid_argument(
            "async_read_ssd_block: Size must be a multiple of page size (" +
            std::to_string(page_size) + " bytes). Size: " + std::to_string(size)
        );
    }

    auto request = std::make_unique<AsyncIORequest>();
    request->buffer = buffer;
    request->size = size;
    request->offset = offset;
    request->status = AsyncIOStatus::PENDING;
    request->fd = _file_descriptor;

    std::memset(&request->cb, 0, sizeof(aiocb));
    request->cb.aio_fildes = request->fd;
    request->cb.aio_buf = request->buffer;
    request->cb.aio_nbytes = request->size;
    request->cb.aio_offset = request->offset;
    request->cb.aio_sigevent.sigev_notify = SIGEV_NONE;

    int ret = aio_read(&request->cb);
    if (ret == -1) {
        fprintf(stderr, "ERROR: aio_read failed for buffer %p, offset %ld, size %zu: %s\n",
                buffer, offset, size, strerror(errno));
        request->status = AsyncIOStatus::FAILED;
        if (callback) {
            callback(AsyncIOStatus::FAILED, std::move(request));
        }
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(this->_callback_mutex);
        AsyncIORequest* raw_req_ptr = request.get();
        _pending_requests.push_back(std::move(request));
        _callbacks[raw_req_ptr] = callback;
    }
    fprintf(stdout, "Started AIO read for buffer %p, offset %ld, size %zu.\n", buffer, offset, size);
    return std::unique_ptr<AsyncIORequest>(_pending_requests.back().get());
#endif
}

void IOManager::poll_io_events() {
#ifdef _WIN32
    // Windows'ta ReadFileEx kullanıldığında, I/O Completion Routines
    // GetQueuedCompletionStatusEx (IOCP) veya SleepEx ile çağrılır.
    // PoC için, sadece SleepEx ile bu thread'in callback'leri işlemesini bekleyeceğiz.
    SleepEx(INFINITE, TRUE); // Infinite bekle, alertable state
#else // Linux
    std::lock_guard<std::mutex> lock(this->_callback_mutex);
    auto it = _pending_requests.begin();
    while (it != _pending_requests.end()) {
        AsyncIORequest* req = it->get();

        int error_code = aio_error(&req->cb);

        if (error_code == EINPROGRESS) {
            ++it;
            continue;
        }

        AsyncIOStatus final_status = AsyncIOStatus::COMPLETED;
        if (error_code != 0) {
            fprintf(stderr, "ERROR: AIO request for buffer %p, offset %ld failed with error: %s\n",
                    req->buffer, req->offset, strerror(error_code));
            final_status = AsyncIOStatus::FAILED;
        } else {
            ssize_t bytes_read = aio_return(&req->cb);
            if (bytes_read == -1) {
                 fprintf(stderr, "ERROR: aio_return failed for buffer %p, offset %ld: %s\n",
                    req->buffer, req->offset, strerror(errno));
                final_status = AsyncIOStatus::FAILED;
            } else if (static_cast<size_t>(bytes_read) != req->size) {
                fprintf(stderr, "WARNING: AIO request for buffer %p, offset %ld requested %zu bytes, but read %zd bytes.\n",
                        req->buffer, req->offset, req->size, bytes_read);
            }
        }

        auto callback_it = _callbacks.find(req);
        if (callback_it != _callbacks.end()) {
            callback_it->second(final_status, std::move(*it));
            _callbacks.erase(callback_it);
        } else {
            fprintf(stderr, "WARNING: Completed AIO request %p has no associated callback.\n", req);
        }

        it = _pending_requests.erase(it);
    }
#endif
}
