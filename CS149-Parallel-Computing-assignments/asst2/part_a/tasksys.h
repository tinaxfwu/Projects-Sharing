#ifndef _TASKSYS_H
#define _TASKSYS_H

#include "itasksys.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

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
        static void runThread(IRunnable* runnable,
                        int num_total_tasks,
                        std::atomic<int>* task_counter,
                        std::mutex* mutex);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
    private:
        int num_threads_;
        std::thread* workers_;
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
        void runSpinningThread();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
    private:
        int num_threads_;
        bool active_; // true if task system is active (and therefore, spinning threads should be too)
        std::thread* thread_pool_;
    
        typedef struct {
            std::mutex* mutex;
            IRunnable* runnable;
            int task_counter; // aka current task id being processed by a worker thread
            // int num_total_tasks;
            std::atomic<int> num_total_tasks;
            // int num_tasks_done;  // used to check if batch of tasks passed to run() are done
            std::atomic<int> num_tasks_done;
        } ThreadState;
        ThreadState* thread_state_;
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

        typedef struct {
            std::mutex* mutex;
            IRunnable* runnable;
            int task_counter; // aka current task id being processed by a worker thread
            int num_total_tasks;
            // int num_tasks_done; 
            std::atomic<int> num_tasks_done; 

            // used to check if there are available tasks for worker/sleeping threads
            // to process
            std::condition_variable* has_tasks_condition_variable;
            std::mutex* has_tasks_mutex;
            
            // used to check if batch of tasks passed to run() are done
            // to notify main app thread to wake up
            std::condition_variable* tasks_done_condition_variable;
        } ThreadState;
        ThreadState* thread_state_;
};

#endif
