#ifndef _TASKSYS_H
#define _TASKSYS_H

#include "itasksys.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <set>

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
    public:
        TaskSystemSerial(int num_threads);
        ~TaskSystemSerial();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
    public:
        TaskSystemParallelSpawn(int num_threads);
        ~TaskSystemParallelSpawn();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSpinning(int num_threads);
        ~TaskSystemParallelThreadPoolSpinning();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSleeping(int num_threads);
        ~TaskSystemParallelThreadPoolSleeping();
        const char* name();
        void runSleepingThread(int thread_id);
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
    private:
        int num_threads_;
        bool active_; // true if task system is active (and therefore, threads should be too)
        std::thread* thread_pool_;
        TaskID curr_bulk_task_id_; // current bulk task ID being processed
        TaskID max_completed_bulk_task_id_; // maximum bulk task ID that has been completed
        std::set<TaskID> completed_bulk_tasks_; // set containing all bulk task IDs that have been completed
    
        // map from bulk task ID to number of tasks completed for that bulk task.
        // We use a map member variable instead of keeping track of the number of tasks done 
        // per BulkTask instance in the ready queue, to ensure that the number of tasks 
        // completed is up-to-date for each bulk task. Otherwise, it's possible that we 
        // update number tasks done for the incorrect bulk task if we always read the BulkTask
        // instance from the queue.
        std::unordered_map<TaskID, int> num_tasks_done_per_bulk_task_;

        std::unordered_map<TaskID, std::set<TaskID>> inverse_deps_;
        std::unordered_map<TaskID, int> num_deps_per_bulk_task_;

        typedef struct {
            std::mutex* mutex; 

            // used to notify sleeping worker threads that there are bulk tasks available
            // to process
            std::condition_variable* has_tasks_condition_variable;
            
            // used to check if all bulk tasks launched so far are done
            // to notify main app thread to wake up and return from sync()
            std::condition_variable* tasks_done_condition_variable;
            std::mutex* tasks_done_mutex;
        } ThreadState;
        ThreadState* thread_state_;

        struct BulkTask {
            TaskID id;
            TaskID max_dep_id; // max ID of this task's dependencies
            IRunnable* runnable;
            int num_total_tasks;
            TaskID task_counter; // current task of this bulk task that's being processed

            std::vector<TaskID> deps;

            BulkTask(TaskID id, TaskID max_dep_id, IRunnable* runnable, 
                     int num_total_tasks, const std::vector<TaskID>& deps) 
                : id(id), max_dep_id(max_dep_id), runnable(runnable), 
                num_total_tasks(num_total_tasks), task_counter(0), deps(deps) {}
        };

        std::queue<BulkTask> ready_queue_;
        std::mutex* ready_queue_mutex_;

        // Determine priority order of BulkTasks in the wait queue: if bulk task 1
        // has a max dependency with lower id # than bulk task 2, than bulk task 1
        // has higher priority than 2 (i.e. a min priority queue by max_dep_id)
        struct CompareWaitTasks {
            bool operator()(const BulkTask& task1, const BulkTask& task2) {
                return task1.max_dep_id > task2.max_dep_id;
            }
        };
        std::priority_queue<BulkTask, std::vector<BulkTask>, CompareWaitTasks> wait_queue_;
        std::mutex* wait_queue_mutex_;
};

#endif
