//! A re-armable DAG of tasks, for per-frame work with dependencies.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::ThreadPool;

/// A node in a [`TaskGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Task(pub usize);

type TaskFn = Arc<dyn Fn() + Send + Sync>;

struct Node {
    name: String,
    run: TaskFn,
    /// Number of unmet dependencies; re-armed on every execute.
    pending: Mutex<usize>,
    /// Tasks released when this one finishes.
    dependents: Vec<Task>,
}

/// Errors from graph construction and execution.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TaskError {
    /// A dependency referenced a task that does not exist.
    #[error("task {name} depends on unknown task {dep}")]
    UnknownDependency {
        /// The task with the bad edge.
        name: String,
        /// The index it referenced.
        dep: usize,
    },
    /// A dependency cycle makes the graph unschedulable.
    #[error("dependency cycle involving task {0}")]
    Cycle(String),
    /// The same name was added twice.
    #[error("duplicate task name {0}")]
    DuplicateName(String),
}

struct Shared {
    nodes: Mutex<Vec<Arc<Node>>>,
    pool: Arc<ThreadPool>,
    in_flight: AtomicUsize,
}

impl Shared {
    fn node(&self, task: Task) -> Arc<Node> {
        Arc::clone(&self.nodes.lock().unwrap()[task.0])
    }

    fn len(&self) -> usize {
        self.nodes.lock().unwrap().len()
    }
}

impl Shared {
    fn schedule(self: &Arc<Self>, task: Task) {
        let node = self.node(task);
        let dependents = node.dependents.clone();
        let me = Arc::clone(self);
        self.pool.spawn(move || {
            (node.run)();
            for dep in dependents {
                let next = Arc::clone(&me.node(dep));
                let ready = {
                    let mut pending = next.pending.lock().unwrap();
                    *pending -= 1;
                    *pending == 0
                };
                if ready {
                    me.schedule(dep);
                }
            }
            me.in_flight.fetch_sub(1, Ordering::AcqRel);
        });
    }
}

/// A DAG of tasks executed across a [`ThreadPool`].
///
/// Dependency counts are re-armed at the start of every
/// [`TaskGraph::execute`], so one graph can be run every frame.
pub struct TaskGraph {
    shared: Arc<Shared>,
    by_name: HashMap<String, Task>,
    order: Vec<Task>,
}

impl TaskGraph {
    /// An empty graph that will run on `pool`.
    pub fn new(pool: Arc<ThreadPool>) -> TaskGraph {
        TaskGraph {
            shared: Arc::new(Shared {
                nodes: Mutex::new(Vec::new()),
                pool,
                in_flight: AtomicUsize::new(0),
            }),
            by_name: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Add a task, returning its handle.
    ///
    /// # Panics
    ///
    /// Panics if the name is already used; use [`TaskGraph::try_add`] to
    /// handle that case.
    pub fn add<F>(&mut self, name: impl Into<String>, f: F) -> Task
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.try_add(name, f).expect("duplicate task name")
    }

    /// Add a task, returning `None` if the name is already used.
    pub fn try_add<F>(&mut self, name: impl Into<String>, f: F) -> Option<Task>
    where
        F: Fn() + Send + Sync + 'static,
    {
        let name = name.into();
        if self.by_name.contains_key(&name) {
            return None;
        }
        let id = Task(self.shared.len());
        self.by_name.insert(name.clone(), id);
        self.shared.nodes.lock().unwrap().push(Arc::new(Node {
            name,
            run: Arc::new(f),
            pending: Mutex::new(0),
            dependents: Vec::new(),
        }));
        self.order.clear();
        Some(id)
    }

    /// Declare that `task` must run after `dependency`.
    pub fn add_dependency(&mut self, task: Task, dependency: Task) -> Result<(), TaskError> {
        let mut nodes = self.shared.nodes.lock().unwrap();
        if task.0 >= nodes.len() || dependency.0 >= nodes.len() {
            return Err(TaskError::UnknownDependency {
                name: nodes
                    .get(task.0)
                    .map(|n| n.name.clone())
                    .unwrap_or_default(),
                dep: dependency.0,
            });
        }
        if task == dependency {
            return Err(TaskError::Cycle(nodes[task.0].name.clone()));
        }
        *nodes[task.0].pending.lock().unwrap() += 1;
        Arc::get_mut(&mut nodes[dependency.0])
            .expect("no worker holds this node")
            .dependents
            .push(task);
        self.order.clear();
        Ok(())
    }

    /// Look up a task by name.
    pub fn task(&self, name: &str) -> Option<Task> {
        self.by_name.get(name).copied()
    }

    /// Number of tasks.
    pub fn len(&self) -> usize {
        self.shared.len()
    }

    /// True when the graph has no tasks.
    pub fn is_empty(&self) -> bool {
        self.shared.len() == 0
    }

    /// Compute a topological order, detecting cycles.
    pub fn topological_order(&mut self) -> Result<&[Task], TaskError> {
        if !self.order.is_empty() {
            return Ok(&self.order);
        }
        let nodes = self.shared.nodes.lock().unwrap();
        let n = nodes.len();
        // Same edge-derived indegree as is_acyclic, so neither check disturbs
        // the live pending counters.
        let mut indegree = vec![0usize; n];
        for node in nodes.iter() {
            for dep in &node.dependents {
                indegree[dep.0] += 1;
            }
        }
        let mut queue: Vec<usize> = (0..n).filter(|i| indegree[*i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(i) = queue.pop() {
            order.push(Task(i));
            for dep in &nodes[i].dependents {
                indegree[dep.0] -= 1;
                if indegree[dep.0] == 0 {
                    queue.push(dep.0);
                }
            }
        }
        if order.len() != n {
            let stuck = (0..n)
                .find(|i| indegree[*i] > 0)
                .map(|i| nodes[i].name.clone())
                .unwrap_or_default();
            return Err(TaskError::Cycle(stuck));
        }
        drop(nodes);
        self.order = order;
        Ok(&self.order)
    }

    /// True when the graph has no dependency cycle.
    ///
    /// Reads the dependency edges rather than the live pending counters, so
    /// calling it never perturbs a graph that is about to run.
    pub fn is_acyclic(&self) -> bool {
        let nodes = self.shared.nodes.lock().unwrap();
        let n = nodes.len();
        // A task is blocked by every task that lists it as a dependent, so
        // indegree is recomputed from the edges instead of the live counters,
        // which keeps this check side-effect free.
        let mut indegree = vec![0usize; n];
        for node in nodes.iter() {
            for dep in &node.dependents {
                indegree[dep.0] += 1;
            }
        }
        let mut queue: Vec<usize> = (0..n).filter(|i| indegree[*i] == 0).collect();
        let mut visited = 0;
        while let Some(i) = queue.pop() {
            visited += 1;
            for dep in &nodes[i].dependents {
                indegree[dep.0] -= 1;
                if indegree[dep.0] == 0 {
                    queue.push(dep.0);
                }
            }
        }
        visited == n
    }

    /// Run every task, honouring dependencies, and return when all are done.
    ///
    /// Re-arms dependency counts, so the same graph can run repeatedly.
    ///
    /// # Panics
    ///
    /// Panics if the graph has a cycle, since no task could ever run.
    pub fn execute(&mut self) {
        assert!(
            self.is_acyclic(),
            "cannot execute a graph with a dependency cycle"
        );

        let nodes = self.shared.nodes.lock().unwrap();
        for node in nodes.iter() {
            *node.pending.lock().unwrap() = 0;
        }
        for node in nodes.iter() {
            for dep in &node.dependents {
                *nodes[dep.0].pending.lock().unwrap() += 1;
            }
        }
        let roots: Vec<Task> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| *n.pending.lock().unwrap() == 0)
            .map(|(i, _)| Task(i))
            .collect();
        drop(nodes);

        self.shared.in_flight.store(roots.len(), Ordering::Release);
        for task in roots {
            self.shared.schedule(task);
        }
        self.shared.pool.wait_for_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    fn pool() -> Arc<ThreadPool> {
        Arc::new(ThreadPool::new(4))
    }

    #[test]
    fn empty_graph_executes() {
        let mut g = TaskGraph::new(pool());
        g.execute();
        assert!(g.is_empty());
    }

    #[test]
    fn single_task_runs() {
        let p = pool();
        let count = Arc::new(AtomicUsize::new(0));
        let mut g = TaskGraph::new(Arc::clone(&p));
        let c = Arc::clone(&count);
        g.add("one", move || {
            c.fetch_add(1, Ordering::AcqRel);
        });
        g.execute();
        assert_eq!(count.load(Ordering::Acquire), 1);
    }

    #[test]
    fn independent_tasks_all_run() {
        let p = pool();
        let count = Arc::new(AtomicUsize::new(0));
        let mut g = TaskGraph::new(Arc::clone(&p));
        for i in 0..10 {
            let c = Arc::clone(&count);
            g.add(format!("t{i}"), move || {
                c.fetch_add(1, Ordering::AcqRel);
            });
        }
        g.execute();
        assert_eq!(count.load(Ordering::Acquire), 10);
    }

    #[test]
    fn dependencies_run_in_order() {
        let p = pool();
        let log = Arc::new(StdMutex::new(Vec::new()));
        let mut g = TaskGraph::new(Arc::clone(&p));
        let l = Arc::clone(&log);
        let a = g.add("a", move || l.lock().unwrap().push("a"));
        let l = Arc::clone(&log);
        let b = g.add("b", move || l.lock().unwrap().push("b"));
        let l = Arc::clone(&log);
        let c = g.add("c", move || l.lock().unwrap().push("c"));
        g.add_dependency(b, a).unwrap();
        g.add_dependency(c, b).unwrap();
        g.execute();
        assert_eq!(*log.lock().unwrap(), vec!["a", "b", "c"]);
    }

    #[test]
    fn diamond_dependencies_all_run() {
        let p = pool();
        let count = Arc::new(AtomicUsize::new(0));
        let mut g = TaskGraph::new(Arc::clone(&p));
        let mk = |g: &mut TaskGraph, name: &str, c: Arc<AtomicUsize>| {
            g.add(name, move || {
                c.fetch_add(1, Ordering::AcqRel);
            })
        };
        let root = mk(&mut g, "root", Arc::clone(&count));
        let left = mk(&mut g, "left", Arc::clone(&count));
        let right = mk(&mut g, "right", Arc::clone(&count));
        let join = mk(&mut g, "join", Arc::clone(&count));
        g.add_dependency(left, root).unwrap();
        g.add_dependency(right, root).unwrap();
        g.add_dependency(join, left).unwrap();
        g.add_dependency(join, right).unwrap();
        g.execute();
        assert_eq!(count.load(Ordering::Acquire), 4);
    }

    #[test]
    fn graph_is_rearmable_across_frames() {
        let p = pool();
        let count = Arc::new(AtomicUsize::new(0));
        let mut g = TaskGraph::new(Arc::clone(&p));
        let l = Arc::clone(&count);
        let a = g.add("a", move || {
            l.fetch_add(1, Ordering::AcqRel);
        });
        let l = Arc::clone(&count);
        let b = g.add("b", move || {
            l.fetch_add(1, Ordering::AcqRel);
        });
        g.add_dependency(b, a).unwrap();
        g.execute();
        g.execute();
        g.execute();
        assert_eq!(count.load(Ordering::Acquire), 6, "3 frames x 2 tasks");
    }

    #[test]
    fn topological_order_respects_dependencies() {
        let mut g = TaskGraph::new(pool());
        let a = g.add("a", || {});
        let b = g.add("b", || {});
        g.add_dependency(b, a).unwrap();
        let order = g.topological_order().unwrap().to_vec();
        let pos_a = order.iter().position(|t| *t == a).unwrap();
        let pos_b = order.iter().position(|t| *t == b).unwrap();
        assert!(pos_a < pos_b);
    }

    #[test]
    fn cycle_is_reported_by_topological_order() {
        let mut g = TaskGraph::new(pool());
        let a = g.add("a", || {});
        let b = g.add("b", || {});
        g.add_dependency(b, a).unwrap();
        g.add_dependency(a, b).unwrap();
        assert!(!g.is_acyclic());
        assert!(matches!(g.topological_order(), Err(TaskError::Cycle(_))));
    }

    #[test]
    fn self_dependency_is_rejected() {
        let mut g = TaskGraph::new(pool());
        let a = g.add("a", || {});
        assert!(matches!(g.add_dependency(a, a), Err(TaskError::Cycle(_))));
    }

    #[test]
    fn unknown_dependency_is_rejected() {
        let mut g = TaskGraph::new(pool());
        let a = g.add("a", || {});
        assert!(matches!(
            g.add_dependency(a, Task(99)),
            Err(TaskError::UnknownDependency { .. })
        ));
    }

    #[test]
    fn duplicate_name_returns_none() {
        let mut g = TaskGraph::new(pool());
        assert!(g.try_add("x", || {}).is_some());
        assert!(g.try_add("x", || {}).is_none());
    }

    #[test]
    fn lookup_by_name_works() {
        let mut g = TaskGraph::new(pool());
        let t = g.add("findable", || {});
        assert_eq!(g.task("findable"), Some(t));
        assert_eq!(g.task("missing"), None);
    }

    #[test]
    fn independent_graph_is_acyclic() {
        let mut g = TaskGraph::new(pool());
        g.add("a", || {});
        g.add("b", || {});
        assert!(g.is_acyclic());
        assert_eq!(g.topological_order().unwrap().len(), 2);
    }

    #[test]
    fn len_reflects_task_count() {
        let mut g = TaskGraph::new(pool());
        g.add("a", || {});
        g.add("b", || {});
        assert_eq!(g.len(), 2);
        assert!(!g.is_empty());
    }

    #[test]
    #[should_panic(expected = "dependency cycle")]
    fn executing_a_cycle_panics() {
        let mut g = TaskGraph::new(pool());
        let a = g.add("a", || {});
        let b = g.add("b", || {});
        g.add_dependency(b, a).unwrap();
        g.add_dependency(a, b).unwrap();
        g.execute();
    }

    #[test]
    fn deep_chain_runs_in_order() {
        let p = pool();
        let log = Arc::new(StdMutex::new(Vec::new()));
        let mut g = TaskGraph::new(Arc::clone(&p));
        let mut prev = None;
        let mut tasks = Vec::new();
        for i in 0..20 {
            let l = Arc::clone(&log);
            let t = g.add(format!("t{i}"), move || l.lock().unwrap().push(i));
            if let Some(p) = prev {
                g.add_dependency(t, p).unwrap();
            }
            prev = Some(t);
            tasks.push(t);
        }
        assert!(g.is_acyclic());
        g.execute();
        let got = log.lock().unwrap().clone();
        assert_eq!(got, (0..20).collect::<Vec<_>>());
        assert_eq!(tasks.len(), 20);
    }
}
