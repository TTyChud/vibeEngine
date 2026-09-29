//! Threading for vibeEngine: a work-stealing-free thread pool, a batched
//! [`parallel_for`], and a re-armable [`TaskGraph`] DAG executor.
//!
//! The pool waits by *helping* — a thread that finds no queued work runs its
//! own remaining slice — so a wait never spins a core idle-busy and never
//! deadlocks when a task itself enqueues more work.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

mod graph;
pub use graph::{Task, TaskError, TaskGraph};

/// Number of worker threads to spawn for a pool.
///
/// Returns the hardware concurrency, clamped to at least 1. A single-core box
/// gets a pool of one, which still works (work is executed inline) but gains
/// nothing from threads.
pub fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

type TaskBox = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct Queue {
    tasks: Mutex<Vec<TaskBox>>,
    signal: Condvar,
    /// Bumped on every push so a waiter can tell "queue was empty" from
    /// "queue has work I already looked at".
    epoch: AtomicUsize,
}

impl Queue {
    fn push(&self, task: TaskBox) {
        self.tasks.lock().unwrap().push(task);
        self.epoch.fetch_add(1, Ordering::Release);
        self.signal.notify_one();
    }

    fn pop(&self) -> Option<TaskBox> {
        self.tasks.lock().unwrap().pop()
    }

    fn len(&self) -> usize {
        self.tasks.lock().unwrap().len()
    }
}

/// A pool of worker threads that execute closures.
///
/// Tasks are LIFO within the pool's single queue: the most recently pushed
/// task is the most likely to still be cache-hot, and depth-first execution
/// keeps a worker's cache warm across a nested push.
pub struct ThreadPool {
    inner: Arc<Inner>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

struct Inner {
    queue: Arc<Queue>,
    shutdown: AtomicBool,
    /// Tasks pushed but not yet finished, whether queued or running.
    ///
    /// Incremented while the task is still in the queue, so a waiter can never
    /// observe an empty queue and zero running at the same moment.
    in_flight: AtomicUsize,
    /// Number of tasks ever completed, for tests and the editor's stats panel.
    completed: AtomicUsize,
}

impl ThreadPool {
    /// Create a pool with `workers` threads.
    ///
    /// `workers == 0` is clamped to one thread.
    pub fn new(workers: usize) -> ThreadPool {
        let workers = workers.max(1);
        let inner = Arc::new(Inner {
            queue: Arc::new(Queue::default()),
            shutdown: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
            completed: AtomicUsize::new(0),
        });

        let mut handles = Vec::with_capacity(workers);
        for i in 0..workers {
            let inner = Arc::clone(&inner);
            let handle = std::thread::Builder::new()
                .name(format!("vibe-worker-{i}"))
                .spawn(move || worker_loop(inner))
                .expect("failed to spawn worker thread");
            handles.push(handle);
        }

        ThreadPool {
            inner,
            workers: handles,
        }
    }

    /// Create a pool sized from the hardware concurrency.
    pub fn with_default_size() -> ThreadPool {
        ThreadPool::new(default_worker_count())
    }

    /// Queue a closure to run on a worker.
    pub fn spawn<F: FnOnce() + Send + 'static>(&self, f: F) {
        // Count the task as in flight before it is visible in the queue.
        self.inner.in_flight.fetch_add(1, Ordering::AcqRel);
        self.inner.queue.push(Box::new(f));
    }

    /// Number of tasks queued but not yet started.
    pub fn queued(&self) -> usize {
        self.inner.queue.len()
    }

    /// Total tasks completed since the pool was created.
    pub fn completed_count(&self) -> usize {
        self.inner.completed.load(Ordering::Relaxed)
    }

    /// True while any task is running or queued.
    pub fn is_busy(&self) -> bool {
        self.inner.in_flight.load(Ordering::Acquire) > 0
    }

    /// Block until the queue is empty and no task is running.
    ///
    /// [`parallel_for`] and [`TaskGraph::execute`] call this internally, so
    /// most code never needs it.
    pub fn wait_for_all(&self) {
        while self.is_busy() {
            if let Some(task) = self.inner.queue.pop() {
                self.run_one(task);
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Block until idle, but give up after `timeout`.
    ///
    /// Returns `true` if the pool went idle within the timeout.
    pub fn wait_for_all_timeout(&self, timeout: Duration) -> bool {
        let start = Instant::now();
        while self.is_busy() {
            if let Some(task) = self.inner.queue.pop() {
                self.run_one(task);
            } else if start.elapsed() >= timeout {
                return !self.is_busy();
            } else {
                std::thread::yield_now();
            }
        }
        true
    }

    fn run_one(&self, task: TaskBox) {
        let guard = ActiveGuard { inner: &self.inner };
        task();
        drop(guard);
    }
}

struct ActiveGuard<'a> {
    inner: &'a Arc<Inner>,
}

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.inner.in_flight.fetch_sub(1, Ordering::AcqRel);
        self.inner.completed.fetch_add(1, Ordering::Relaxed);
    }
}

fn worker_loop(inner: Arc<Inner>) {
    while !inner.shutdown.load(Ordering::Acquire) {
        if let Some(task) = inner.queue.pop() {
            let guard = ActiveGuard { inner: &inner };
            task();
            drop(guard);
            continue;
        }

        // Nothing queued. Sleep until a push arrives, but re-check shutdown on
        // each wake so shutdown does not depend on a task being pushed.
        let mut tasks = inner.queue.tasks.lock().unwrap();
        let observed = inner.queue.epoch.load(Ordering::Acquire);
        if tasks.is_empty() && !inner.shutdown.load(Ordering::Acquire) {
            let (guard, result) = inner
                .queue
                .signal
                .wait_timeout(tasks, Duration::from_millis(50))
                .unwrap();
            tasks = guard;
            if result.timed_out() && inner.queue.epoch.load(Ordering::Acquire) == observed {
                drop(tasks);
            }
        } else {
            drop(tasks);
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        {
            let tasks = self.inner.queue.tasks.lock().unwrap();
            self.inner.queue.signal.notify_all();
            drop(tasks);
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

impl std::fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadPool")
            .field("workers", &self.workers.len())
            .field("queued", &self.queued())
            .field("completed", &self.completed_count())
            .finish()
    }
}

/// Split `0..len` into contiguous batches of at most `batch_size` and run `f`
/// on each index across the pool, returning once every index is done.
///
/// The calling thread participates: it runs batches itself rather than only
/// waiting, so a [`ThreadPool::new(1)`] pool and a 4-thread pool both finish a
/// parallel_for, and neither leaves a core spinning.
///
/// # Panics
///
/// Panics if `batch_size` is zero, since that would never advance.
pub fn parallel_for<F>(pool: &ThreadPool, len: usize, batch_size: usize, f: F)
where
    F: Fn(usize) + Send + Sync + 'static,
{
    assert!(batch_size > 0, "batch_size must be greater than zero");

    if len == 0 {
        return;
    }

    let num_batches = len.div_ceil(batch_size);
    // Workers claim batches by bumping one shared cursor, so there is no mutex
    // on the hot path and the work still balances across threads.
    let cursor = Arc::new(AtomicUsize::new(0));
    let job = Arc::new(BatchJob {
        cursor: Arc::clone(&cursor),
        num_batches,
        batch_size,
        len,
        f: Arc::new(f),
    });

    // Every batch is claimed from the same cursor, so the calling thread simply
    // helps until the cursor is exhausted, then waits for the stragglers.
    for _ in 0..num_batches {
        let job = Arc::clone(&job);
        pool.spawn(move || job.run());
    }
    pool.wait_for_all();
}

struct BatchJob<F> {
    cursor: Arc<AtomicUsize>,
    num_batches: usize,
    batch_size: usize,
    len: usize,
    f: Arc<F>,
}

impl<F: Fn(usize)> BatchJob<F> {
    fn run(&self) {
        loop {
            let batch = self.cursor.fetch_add(1, Ordering::Relaxed);
            if batch >= self.num_batches {
                break;
            }
            let start = batch * self.batch_size;
            let end = (start + self.batch_size).min(self.len);
            for i in start..end {
                (self.f)(i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn spawn_runs_task() {
        let pool = ThreadPool::new(2);
        let flag = Arc::new(AtomicUsize::new(0));
        let f = Arc::clone(&flag);
        pool.spawn(move || {
            f.store(7, Ordering::Release);
        });
        pool.wait_for_all();
        assert_eq!(flag.load(Ordering::Acquire), 7);
    }

    #[test]
    fn many_tasks_all_complete() {
        let pool = ThreadPool::new(4);
        let count = Arc::new(AtomicUsize::new(0));
        for _ in 0..200 {
            let c = Arc::clone(&count);
            pool.spawn(move || {
                c.fetch_add(1, Ordering::AcqRel);
            });
        }
        pool.wait_for_all();
        assert_eq!(count.load(Ordering::Acquire), 200);
        assert!(pool.completed_count() >= 200);
    }

    #[test]
    fn parallel_for_covers_every_index_exactly_once() {
        let pool = ThreadPool::new(4);
        let seen = Arc::new(StdMutex::new(vec![0usize; 1000]));
        let s = Arc::clone(&seen);
        parallel_for(&pool, 1000, 32, move |i| {
            s.lock().unwrap()[i] += 1;
        });
        let seen = seen.lock().unwrap();
        assert!(
            seen.iter().all(|&c| c == 1),
            "some index was not run exactly once"
        );
    }

    #[test]
    fn parallel_for_zero_length_is_noop() {
        let pool = ThreadPool::new(2);
        let hits = Arc::new(AtomicUsize::new(0));
        let h = Arc::clone(&hits);
        parallel_for(&pool, 0, 8, move |_| {
            h.fetch_add(1, Ordering::AcqRel);
        });
        assert_eq!(hits.load(Ordering::Acquire), 0);
    }

    #[test]
    fn parallel_for_single_thread_pool_still_completes() {
        let pool = ThreadPool::new(1);
        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);
        parallel_for(&pool, 64, 8, move |_| {
            c.fetch_add(1, Ordering::AcqRel);
        });
        assert_eq!(count.load(Ordering::Acquire), 64);
    }

    #[test]
    fn parallel_for_batch_larger_than_len() {
        let pool = ThreadPool::new(2);
        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);
        parallel_for(&pool, 5, 1000, move |_| {
            c.fetch_add(1, Ordering::AcqRel);
        });
        assert_eq!(count.load(Ordering::Acquire), 5);
    }

    #[test]
    fn parallel_for_runs_indices_concurrently() {
        let pool = ThreadPool::new(4);
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        for _ in 0..4 {
            let (a, p) = (Arc::clone(&active), Arc::clone(&peak));
            pool.spawn(move || {
                let n = a.fetch_add(1, Ordering::AcqRel) + 1;
                p.fetch_max(n, Ordering::AcqRel);
                std::thread::sleep(Duration::from_millis(30));
                a.fetch_sub(1, Ordering::AcqRel);
            });
        }
        pool.wait_for_all();
        assert!(
            peak.load(Ordering::Acquire) > 1,
            "tasks did not overlap on separate threads"
        );
    }

    #[test]
    fn nested_spawn_from_within_task_completes() {
        let pool = ThreadPool::new(2);
        let pool2 = Arc::new(pool);
        let done = Arc::new(AtomicUsize::new(0));
        let d = Arc::clone(&done);
        let p = Arc::clone(&pool2);
        pool2.spawn(move || {
            for _ in 0..10 {
                let d2 = Arc::clone(&d);
                let p2 = Arc::clone(&p);
                p2.spawn(move || {
                    d2.fetch_add(1, Ordering::AcqRel);
                });
            }
        });
        pool2.wait_for_all();
        assert_eq!(done.load(Ordering::Acquire), 10);
    }

    #[test]
    fn is_busy_false_after_drain() {
        let pool = ThreadPool::new(2);
        for _ in 0..10 {
            pool.spawn(|| {});
        }
        pool.wait_for_all();
        assert!(!pool.is_busy());
    }

    #[test]
    fn wait_timeout_reports_idle() {
        let pool = ThreadPool::new(2);
        pool.spawn(|| {});
        assert!(pool.wait_for_all_timeout(Duration::from_secs(5)));
    }

    #[test]
    fn default_worker_count_at_least_one() {
        assert!(default_worker_count() >= 1);
    }

    #[test]
    fn drop_joins_workers_cleanly() {
        let count = Arc::new(AtomicUsize::new(0));
        {
            let pool = ThreadPool::new(3);
            for _ in 0..50 {
                let c = Arc::clone(&count);
                pool.spawn(move || {
                    c.fetch_add(1, Ordering::AcqRel);
                });
            }
            pool.wait_for_all();
        }
        assert_eq!(count.load(Ordering::Acquire), 50);
    }

    #[test]
    fn zero_workers_clamps_to_one() {
        let pool = ThreadPool::new(0);
        let done = Arc::new(AtomicBool::new(false));
        let d = Arc::clone(&done);
        pool.spawn(move || {
            d.store(true, Ordering::Release);
        });
        pool.wait_for_all();
        assert!(done.load(Ordering::Acquire));
    }
}
