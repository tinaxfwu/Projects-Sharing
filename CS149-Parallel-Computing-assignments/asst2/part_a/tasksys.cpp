#include "tasksys.h"
#include <stdio.h>

IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads), num_threads_(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    workers_ = new std::thread[num_threads];
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {
    delete[] workers_;
}

void TaskSystemParallelSpawn::runThread(IRunnable* runnable,
                                        int num_total_tasks,
                                        std::atomic<int>* task_counter,
                                        std::mutex* mutex) {
    // See lecture 5, page 10 for inspiration
    while (1) {
        int i;
        mutex->lock();
        i = *task_counter;
        *task_counter += 1;
        mutex->unlock();
        if (i >= num_total_tasks) {
            break;
        }
        runnable->runTask(i, num_total_tasks);
    }
}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    std::mutex* mutex = new std::mutex();
    std::atomic<int> task_counter;
    task_counter = 0;
    for (int i = 0; i < num_threads_; i++) {
        workers_[i] = std::thread(TaskSystemParallelSpawn::runThread, 
                                    runnable, num_total_tasks, 
                                    &task_counter, mutex);
    }
    for (int i = 0; i < num_threads_; i++) {
        workers_[i].join();
    }
    delete mutex;
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads),
                                                                                             num_threads_(num_threads),
                                                                                             active_(true) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    thread_state_ = new ThreadState();
    thread_state_->mutex = new std::mutex();
    thread_state_->runnable = nullptr;
    thread_state_->task_counter = -1;
    thread_state_->num_total_tasks = -1;
    thread_state_->num_tasks_done = -1;

    thread_pool_ = new std::thread[num_threads];
    for (int i = 0; i < num_threads_; i++) {
        thread_pool_[i] = std::thread(&TaskSystemParallelThreadPoolSpinning::runSpinningThread, this);
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    active_ = false;
    for (int i = 0; i < num_threads_; i++) {
        thread_pool_[i].join();
    }
    delete[] thread_pool_;
    delete thread_state_->mutex;
    delete thread_state_;
}

void TaskSystemParallelThreadPoolSpinning::runSpinningThread() {
    while (active_) {
        thread_state_->mutex->lock();
        int i = thread_state_->task_counter;
        thread_state_->task_counter += 1;
        thread_state_->mutex->unlock();
        if (i >= thread_state_->num_total_tasks) {
            continue;
        }
        
        thread_state_->runnable->runTask(i, thread_state_->num_total_tasks);

        thread_state_->num_tasks_done += 1;
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    thread_state_->mutex->lock();
    thread_state_->runnable = runnable;
    thread_state_->num_total_tasks = num_total_tasks;
    thread_state_->num_tasks_done = 0;
    thread_state_->task_counter = 0;
    thread_state_->mutex->unlock();
    
    // Check that all tasks have been completed before returning
    // to the caller.
    while (1) {
        thread_state_->mutex->lock();
        if (thread_state_->num_tasks_done == num_total_tasks) {
            thread_state_->mutex->unlock();
            break;
        }
        thread_state_->mutex->unlock();
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads),
                                                                                             num_threads_(num_threads),
                                                                                             active_(true) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    // printf("entering constructor\n");
    thread_state_ = new ThreadState();
    thread_state_->mutex = new std::mutex();
    thread_state_->runnable = nullptr;
    thread_state_->task_counter = -1;
    thread_state_->num_total_tasks = -1;
    thread_state_->num_tasks_done = -1;
    thread_state_->has_tasks_condition_variable = new std::condition_variable();
    thread_state_->has_tasks_mutex = new std::mutex();
    thread_state_->tasks_done_condition_variable = new std::condition_variable();

    thread_pool_ = new std::thread[num_threads];
    for (int i = 0; i < num_threads_; i++) {
        thread_pool_[i] = std::thread(&TaskSystemParallelThreadPoolSleeping::runSleepingThread, this, i);
    }
    // printf("exiting constructor\n");
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    // printf("~~~~~~~~DESTRUCTOR: setting active to false\n");
    active_ = false;
    // printf("~~~~~~~~DESTRUCTOR: calling notify all\n");
    thread_state_->has_tasks_condition_variable->notify_all();
    
    // printf("~~~~~~~~DESTRUCTOR: joining threads\n");
    for (int i = 0; i < num_threads_; i++) {
        thread_pool_[i].join();
    }
    delete[] thread_pool_;
    delete thread_state_->mutex;
    delete thread_state_->has_tasks_condition_variable;
    delete thread_state_->has_tasks_mutex;
    delete thread_state_->tasks_done_condition_variable;
    delete thread_state_;
}

void TaskSystemParallelThreadPoolSleeping::runSleepingThread(int thread_id) {
    // printf("entering runSleepingThread: %d\n", thread_id);
    while (active_) {
        thread_state_->mutex->lock();
        int i = thread_state_->task_counter;
        thread_state_->task_counter += 1;
        thread_state_->mutex->unlock();

        // If there are no available tasks, sleep and wait for tasks once run() has been called.
        if (active_ && i >= thread_state_->num_total_tasks) {
            std::unique_lock<std::mutex> lk(*thread_state_->has_tasks_mutex);
            // printf("thread %d about to wait for tasks. task_counter: %d, num_total_tasks: %d\n", thread_id, thread_state_->task_counter, thread_state_->num_total_tasks);
            thread_state_->has_tasks_condition_variable->wait(lk, [this]() {
                return !active_ || thread_state_->task_counter < thread_state_->num_total_tasks;
            });
            // printf("thread %d wait has returned\n", thread_id);
            lk.unlock();
            continue;
        }
        // printf("thread %d running task %d \n", thread_id, i);
        thread_state_->runnable->runTask(i, thread_state_->num_total_tasks);

        
        thread_state_->num_tasks_done += 1;
        // printf("thread %d: num_tasks_done %d\n", thread_id, thread_state_->num_tasks_done);
        thread_state_->mutex->lock();
        // If all tasks have been completed, notify main application thread
        if (thread_state_->num_tasks_done == thread_state_->num_total_tasks) {
            // printf("thread %d: ALL TASKS COMPLETED\n", thread_id);
            thread_state_->mutex->unlock();
            thread_state_->tasks_done_condition_variable->notify_all();
        }
        else {
            thread_state_->mutex->unlock();
        }
    }
    // printf("thread %d: exiting while loop\n", thread_id);
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    // printf("=======CALL RUN\n");
    thread_state_->mutex->lock();
    thread_state_->runnable = runnable;
    thread_state_->num_total_tasks = num_total_tasks;
    thread_state_->task_counter = 0;
    thread_state_->num_tasks_done = 0;
    thread_state_->mutex->unlock();
    // printf("=========RUN ABOUT TO NOTIFY\n");
    thread_state_->has_tasks_condition_variable->notify_all();
    
    // Sleep until all tasks have been completed before returning
    // to the caller.
    std::unique_lock<std::mutex> lk(*thread_state_->mutex);
    thread_state_->tasks_done_condition_variable->wait(lk, [this]() {
        return thread_state_->num_tasks_done >= thread_state_->num_total_tasks;
    });
    lk.unlock();
    // printf("=====DONE\n");
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
