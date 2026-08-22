#ifndef FRIDAY_IO_MANAGER_H
#define FRIDAY_IO_MANAGER_H

#include "friday_types.h"
#include <functional>
#include <string>
#include <memory> // For std::unique_ptr
#include <future> // For std::future
#include <mutex>  // _callback_mutex için
#include <unordered_map> // _callbacks için
#include <list> // _pending_requests için (Windows'ta listeye gerek kalmayabilir, I/O completion portları daha farklı)

// I/OManager sınıfı, asenkron ve O_DIRECT/FILE_FLAG_NO_BUFFERING özellikli disk okumalarını yönetir.
class IOManager {
public:
    IOManager();
    ~IOManager();

    bool open_model_file(const std::string& filepath);
    void close_model_file();

    std::unique_ptr<AsyncIORequest> async_read_ssd_block(
        void* buffer,
        size_t size,
        off_t offset,
        std::function<void(AsyncIOStatus, std::unique_ptr<AsyncIORequest>)> callback
    );

    void poll_io_events();

private:
    #ifdef _WIN32
    HANDLE _file_handle;
    static VOID CALLBACK FileIOCompletionRoutine(
        DWORD dwErrorCode,
        DWORD dwNumberOfBytesTransfered,
        LPOVERLAPPED lpOverlapped
    );
    std::unordered_map<OVERLAPPED*, std::function<void(AsyncIOStatus, std::unique_ptr<AsyncIORequest>)>> _callbacks_win;
    std::mutex _callback_mutex_win;
    #else // Linux
    int _file_descriptor;
    std::list<std::unique_ptr<AsyncIORequest>> _pending_requests;
    std::unordered_map<AsyncIORequest*, std::function<void(AsyncIOStatus, std::unique_ptr<AsyncIORequest>)>> _callbacks;
    std::mutex _callback_mutex;
    #endif
};

#endif // FRIDAY_IO_MANAGER_H
