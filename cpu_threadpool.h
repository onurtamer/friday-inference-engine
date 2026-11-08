#ifndef FRIDAY_CPU_THREADPOOL_H
#define FRIDAY_CPU_THREADPOOL_H

// Small persistent thread pool for decode-time CPU kernels. parallelFor splits
// [0, n) into chunks handed out through an atomic counter; the calling thread
// participates. Workers spin briefly between jobs (decode issues many short
// jobs back to back) and then sleep on a condition variable.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace friday_cpu {

class ThreadPool {
public:
    explicit ThreadPool(int num_threads);  // total threads including the caller; <= 0: hardware default
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return (int)workers_.size() + 1; }

    // fn(begin, end) over chunks of `chunk` items. Blocks until all chunks are done.
    void parallelFor(int n, int chunk, const std::function<void(int, int)>& fn);

private:
    void workerLoop();
    void runChunks();

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<int> next_{0};
    std::atomic<int> active_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int, int)>* fn_ = nullptr;
    int n_ = 0, chunk_ = 1;
};

}  // namespace friday_cpu

#endif  // FRIDAY_CPU_THREADPOOL_H
