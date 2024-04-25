#include "tasksys.h"

#include <algorithm>

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
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
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

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
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

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
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
                                                                                             active_(true),
                                                                                             curr_bulk_task_id_(-1),
                                                                                             max_completed_bulk_task_id_(-1) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //    
    // printf("entering constructor\n");
    thread_state_ = new ThreadState();
    thread_state_->mutex = new std::mutex();
    thread_state_->has_tasks_condition_variable = new std::condition_variable();
    thread_state_->tasks_done_condition_variable = new std::condition_variable();
    
    wait_queue_mutex_ = new std::mutex();
    ready_queue_mutex_ = new std::mutex();

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
    //
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
    delete thread_state_->tasks_done_condition_variable;
    delete thread_state_;
    delete wait_queue_mutex_;
    delete ready_queue_mutex_;
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    std::vector<TaskID> deps;
    runAsyncWithDeps(runnable, num_total_tasks, deps);
    sync();
}

/*
Example of Part B implementation
================================
e.g. BulkTask = (bulk task id, max dep id, [task counter], [num total tasks])
- ready: 
- wait:  
- runAsyncWithDeps: (0,), (1,0), (2,1), (3,0), (4,1), (5,0)
- MAX completed bulk task:

- current ("next") bulk task id: 3
- completed_tasks: {} ==> once all bulk tasks launched have been completed, we can return from sync()
- num tasks done per bulk task: {
    0: ..., 1: ..., 
}
*/
void TaskSystemParallelThreadPoolSleeping::runSleepingThread(int thread_id) {
    // printf("entering runSleepingThread: %d\n", thread_id);
    while (active_) {
        ready_queue_mutex_->lock();
        if (!ready_queue_.empty()) {
            // printf("thread %d: checking ready queue\n", thread_id);

            // If all tasks for the first bulk task in the ready queue have been completed,
            // then remove that bulk task from the queue.
            if (ready_queue_.front().task_counter >= ready_queue_.front().num_total_tasks) {
                // printf("thread %d: removing bulk task %d from ready queue, num tasks done: %d, num total tasks: %d\n", thread_id, ready_queue_.front().id, num_tasks_done_per_bulk_task_[ready_queue_.front().id], ready_queue_.front().num_total_tasks);
                ready_queue_.pop();
                ready_queue_mutex_->unlock();
            }
            else {
                // Get a bulk task from the ready queue and run it.
                //  printf("thread %d: getting bulk task from ready queue\n", thread_id);
                BulkTask bulk_task = ready_queue_.front();
                ready_queue_.front().task_counter += 1;
                ready_queue_mutex_->unlock();
                bulk_task.runnable->runTask(bulk_task.task_counter, bulk_task.num_total_tasks);
                
                // printf("thread %d: ran bulk task %d and task %d, num tasks done: %d\n", thread_id, bulk_task.id, bulk_task.task_counter, num_tasks_done_per_bulk_task_[bulk_task.id]);
                thread_state_->mutex->lock();
                // ready_queue_.front().num_tasks_done += 1;
                num_tasks_done_per_bulk_task_[bulk_task.id] += 1;
                // Check if all tasks have been completed for this bulk task. 
                if (num_tasks_done_per_bulk_task_[bulk_task.id] >= bulk_task.num_total_tasks) {
                    max_completed_bulk_task_id_ = max_completed_bulk_task_id_ < bulk_task.id ? bulk_task.id : max_completed_bulk_task_id_;

                    completed_bulk_tasks_.insert(bulk_task.id);
                    // printf("thread %d: bulk task id %d, ALL TASKS COMPLETED, completed_bulk_tasks_.size() %d\n", thread_id, bulk_task.id, (int) completed_bulk_tasks_.size());
                    // If number of bulk tasks completed is the same as the number of bulk tasks launched,
                    // then notify main app thread so that it can return from sync().
                    if ((int)completed_bulk_tasks_.size() - 1 == curr_bulk_task_id_) {
                        thread_state_->mutex->unlock();
                        // printf("thread %d: bulk task id %d, ALL BULK TASKS COMPLETED, last completed task id %d\n", thread_id, bulk_task.id, max_completed_bulk_task_id_);
                        thread_state_->tasks_done_condition_variable->notify_all();
                        continue;
                    }

                    // For all bulk tasks (i.e. dependants) that depend upon this bulk task that just completed,
                    // decrement the number of dependencies for the dependant bulk tasks.
                    auto iter = inverse_deps_.find(bulk_task.id);
                    if (iter != inverse_deps_.end()) {
                        // printf("found deps for bulk task id: %d\n", bulk_task.id);
                        for (auto dependent_bulk_task_id : iter->second) {
                            num_deps_per_bulk_task_[dependent_bulk_task_id] -= 1;
                            // printf("dep id %d for bulk task id: %d, num deps %d\n", dependent_bulk_task_id, bulk_task.id, num_deps_per_bulk_task_[dependent_bulk_task_id]);
                        }
                    }
                }
                thread_state_->mutex->unlock();
            }
        }
        else {
            // printf("thread %d: ready queue empty\n", thread_id);
            ready_queue_mutex_->unlock();

            // If ready and wait queues are empty, then sleep.
            wait_queue_mutex_->lock();
            if (active_ && wait_queue_.empty()) {
                wait_queue_mutex_->unlock();

                std::unique_lock<std::mutex> lk(*thread_state_->mutex);
                // printf("thread %d: no tasks, going to sleep...\n", thread_id);

                thread_state_->has_tasks_condition_variable->wait(lk, [this]() {
                    return !active_ || !wait_queue_.empty() || !ready_queue_.empty() ;
                });
                lk.unlock();
                // printf("thread %d: woken up\n", thread_id);
            }
            else {
                // Move bulk tasks from wait queue to ready queue only if the max dep id of these tasks
                // is at most the max id of bulk tasks completed so far.
                while (active_ && !wait_queue_.empty() && wait_queue_.top().max_dep_id <= max_completed_bulk_task_id_) {
                    // printf("thread %d: moving task from wait queue\n", thread_id);

                    // Ensure that all dependencies have been completed before moving the bulk task from
                    // the wait queue to the ready queue.
                    // bool ready_to_move = true;
                    // for (auto dep : wait_queue_.top().deps) {
                    //     if (std::find(completed_bulk_tasks_.begin(), completed_bulk_tasks_.end(), dep) == completed_bulk_tasks_.end()) {
                    //         ready_to_move = false;
                    //         break;
                    //     }
                    // }
                    // if (ready_to_move) {
                    //     ready_queue_mutex_->lock();
                    //     ready_queue_.push(wait_queue_.top());
                    //     ready_queue_mutex_->unlock();
                    //     wait_queue_.pop();
                    // }
                    thread_state_->mutex->lock();
                    if (num_deps_per_bulk_task_[wait_queue_.top().id] == 0) {
                        thread_state_->mutex->unlock();
                        ready_queue_mutex_->lock();
                        ready_queue_.push(wait_queue_.top());
                        ready_queue_mutex_->unlock();
                        wait_queue_.pop();
                    }
                    else {
                        thread_state_->mutex->unlock();
                        break;
                    }
                }
                wait_queue_mutex_->unlock();
            }
        }
    }
    // printf("thread %d: exiting while loop\n", thread_id);
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {
    curr_bulk_task_id_ += 1;
    TaskID max_dep_id = deps.empty() ? -1 : *std::max_element(deps.begin(), deps.end());
    // printf("===================CALL RUN ASYNC WITH DEPS, TASK ID: %d, MAX DEP ID: %d\n", curr_bulk_task_id_, max_dep_id);
    // BulkTask bulk_task(curr_bulk_task_id_, max_dep_id, runnable, num_total_tasks);
    
    // Initialize number of tasks done for this bulk task to 0.
    thread_state_->mutex->lock();
    num_tasks_done_per_bulk_task_[curr_bulk_task_id_] = 0;
    // Keep track of number of dependencies per bulk task ID.
    num_deps_per_bulk_task_[curr_bulk_task_id_] = 0;
    for (auto dep : deps) {
        // If a dependency bulk task has not yet been completed, then
        // add this bulk task ID to the set of dependants for the dependency bulk task,
        // and increment the number of dependencies for this bulk task ID.
        if (completed_bulk_tasks_.find(dep) == completed_bulk_tasks_.end()) {
            inverse_deps_[dep].insert(curr_bulk_task_id_);
            num_deps_per_bulk_task_[curr_bulk_task_id_] += 1;
        }
    }
    thread_state_->mutex->unlock();

    // If this bulk task has no dependencies, add it directly to ready queue
    if (max_dep_id == -1) {
        ready_queue_mutex_->lock();
        ready_queue_.emplace(curr_bulk_task_id_, max_dep_id, runnable, num_total_tasks, deps);
        ready_queue_mutex_->unlock();
    }
    else {
        wait_queue_mutex_->lock();
        wait_queue_.emplace(curr_bulk_task_id_, max_dep_id, runnable, num_total_tasks, deps);
        wait_queue_mutex_->unlock();
    }
    
    thread_state_->has_tasks_condition_variable->notify_all();
    // printf("===================RETURNING FROM RUN ASYNC WITH DEPS\n");
    
    return curr_bulk_task_id_;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //
    // printf("=========CALL SYNC completed_bulk_tasks_ size %d, curr bulk task id %d\n", (int)completed_bulk_tasks_.size(), curr_bulk_task_id_);
    
    // Sleep until all tasks have been completed before returning
    // to the caller.
    std::unique_lock<std::mutex> lk(*thread_state_->mutex);
    if ((int)completed_bulk_tasks_.size() - 1 < curr_bulk_task_id_) {
    // printf("=========SYNC ABOUT TO SLEEP, completed_bulk_tasks_ size %d, curr bulk task id %d\n", (int)completed_bulk_tasks_.size(), curr_bulk_task_id_);
        thread_state_->tasks_done_condition_variable->wait(lk, [this](){
            return (int)completed_bulk_tasks_.size() - 1 >= curr_bulk_task_id_;
        });
    }
    lk.unlock();

    // printf("===================RETURN SYNC\n");
    return;
}