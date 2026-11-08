#include "cpu_threadpool.h"

#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define FRIDAY_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define FRIDAY_CPU_RELAX() asm volatile("yield")
#else
#define FRIDAY_CPU_RELAX() ((void)0)
#endif

namespace friday_cpu {

ThreadPool::ThreadPool(int num_threads) {
    if (num_threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        num_threads = hw > 0 ? (int)hw : 4;
    }
    for (int i = 1; i < num_threads; ++i) workers_.emplace_back([this] { workerLoop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_.store(true);
        generation_.fetch_add(1);
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
}

void ThreadPool::runChunks() {
    for (;;) {
        const int b = next_.fetch_add(chunk_);
        if (b >= n_) break;
        const int e = b + chunk_ < n_ ? b + chunk_ : n_;
        (*fn_)(b, e);
    }
}

void ThreadPool::parallelFor(int n, int chunk, const std::function<void(int, int)>& fn) {
    if (n <= 0) return;
    if (workers_.empty() || n <= chunk) { fn(0, n); return; }
    {
        std::lock_guard<std::mutex> lk(mu_);
        fn_ = &fn;
        n_ = n;
        chunk_ = chunk < 1 ? 1 : chunk;
        next_.store(0);
        active_.store((int)workers_.size());
        generation_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    runChunks();
    while (active_.load(std::memory_order_acquire) != 0) FRIDAY_CPU_RELAX();
}

void ThreadPool::workerLoop() {
    uint64_t seen = 0;
    for (;;) {
        // Spin for a short while (~100 us) before sleeping: decode jobs arrive back to back.
        bool got = false;
        for (int i = 0; i < 20000; ++i) {
            if (generation_.load(std::memory_order_acquire) != seen) { got = true; break; }
            FRIDAY_CPU_RELAX();
        }
        if (!got) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return generation_.load() != seen; });
        }
        seen = generation_.load(std::memory_order_acquire);
        if (stop_.load()) return;
        runChunks();
        active_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

}  // namespace friday_cpu
