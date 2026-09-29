//! Executes a compiled graph, mapping graph handles to live backend objects.

use std::collections::HashMap;
use std::sync::Arc;

use vibe_jobs::TaskGraph;
use vibe_rhi::ResourceHandle;

use crate::error::GraphError;
use crate::graph::{CompiledGraph, PassId};

/// Resolves a graph handle to the backend object that is live right now.
///
/// The backend implements this; the executor only needs to know that a lookup
/// can fail, so a stale handle in a graph is reported rather than dereferenced.
pub trait ResourceResolver {
    /// The backend object id behind a graph handle.
    ///
    /// Returns `None` when the handle is dead or was never created.
    fn resolve(&self, resource: ResourceHandle) -> Option<u64>;

    /// Called before a pass runs, so the backend can bind and emit barriers.
    fn begin_pass(
        &mut self,
        pass: PassId,
        barriers: &[crate::graph::Barrier],
    ) -> Result<(), GraphError>;

    /// Called after a pass runs.
    fn end_pass(&mut self, pass: PassId) -> Result<(), GraphError>;
}

/// What one frame of graph execution cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExecutionStats {
    /// Passes that ran.
    pub passes_executed: u32,
    /// Barriers emitted.
    pub barriers_emitted: u32,
    /// Handles that failed to resolve.
    pub unresolved: u32,
    /// Passes whose body was submitted to the job pool rather than run inline.
    pub passes_dispatched: u32,
}

/// Runs a [`CompiledGraph`] against a backend.
///
/// Pass bodies go through [`GraphExecutor::with_pass_body`], which lets a
/// caller record a pass in parallel on the job system. Without a body, a pass
/// only emits its begin/end notifications, which is what a compile-only pass
/// needs when the caller wants the graph for inspection rather than execution.
pub struct GraphExecutor {
    pool: Arc<vibe_jobs::ThreadPool>,
    stats: ExecutionStats,
}

impl GraphExecutor {
    /// Create an executor that dispatches to `pool`.
    pub fn new(pool: Arc<vibe_jobs::ThreadPool>) -> GraphExecutor {
        GraphExecutor {
            pool,
            stats: ExecutionStats::default(),
        }
    }

    /// The pool this executor dispatches to.
    pub fn pool(&self) -> &Arc<vibe_jobs::ThreadPool> {
        &self.pool
    }

    /// Stats from the last [`GraphExecutor::execute`].
    pub fn stats(&self) -> ExecutionStats {
        self.stats
    }

    /// Check every handle the graph touches resolves, before any GPU work.
    ///
    /// Catching a dead handle here means the frame fails with a clear error
    /// instead of recording a command buffer that dereferences nothing.
    pub fn validate<R: ResourceResolver>(
        &self,
        graph: &CompiledGraph,
        resolver: &R,
    ) -> Result<(), GraphError> {
        for pass in &graph.passes {
            for h in pass
                .reads
                .iter()
                .chain(pass.writes.iter())
                .chain(pass.discards.iter())
            {
                if resolver.resolve(*h).is_none() {
                    return Err(GraphError::DeadResource {
                        pass: pass.name.clone(),
                        resource: *h,
                    });
                }
            }
        }
        Ok(())
    }

    /// Execute a graph, calling `begin_pass`/`end_pass` around each pass.
    pub fn execute<R: ResourceResolver>(
        &mut self,
        graph: &CompiledGraph,
        resolver: &mut R,
    ) -> Result<ExecutionStats, GraphError> {
        self.validate(graph, resolver)?;
        self.stats = ExecutionStats::default();

        for pass in &graph.passes {
            resolver.begin_pass(pass.id, &pass.barriers)?;
            resolver.end_pass(pass.id)?;
            self.stats.passes_executed += 1;
            self.stats.barriers_emitted += pass.barriers.len() as u32;
        }
        Ok(self.stats)
    }

    /// Execute a graph with a per-pass body, dispatching independent passes to
    /// the job pool.
    ///
    /// A pass is dispatched when every pass it depends on has already run, so
    /// the recorded order still matches the compiled order. Because a body
    /// records into shared GPU state, the pool work is a *pre-record* step: the
    /// backend's `begin_pass` is what actually makes the recording order safe.
    pub fn execute_with_bodies<R, F>(
        &mut self,
        graph: &CompiledGraph,
        resolver: &mut R,
        body: F,
    ) -> Result<ExecutionStats, GraphError>
    where
        R: ResourceResolver,
        F: FnMut(PassId) + Send + Sync + 'static,
    {
        self.validate(graph, resolver)?;
        self.stats = ExecutionStats::default();

        let body = Arc::new(std::sync::Mutex::new(body));
        let mut tasks = TaskGraph::new(Arc::clone(&self.pool));
        let mut handles = HashMap::new();

        for pass in &graph.passes {
            let pass_id = pass.id;
            let body = Arc::clone(&body);
            let task = tasks.add(pass.name.clone(), move || {
                let mut guard = body.lock().unwrap();
                guard(pass_id);
            });
            for dep in &pass.depends_on {
                if let Some(&d) = handles.get(dep) {
                    tasks.add_dependency(task, d).ok();
                }
            }
            handles.insert(pass.id, task);
        }

        // Record order stays on this thread; the pool runs the bodies.
        for pass in &graph.passes {
            resolver.begin_pass(pass.id, &pass.barriers)?;
            self.stats.passes_executed += 1;
            self.stats.barriers_emitted += pass.barriers.len() as u32;
            self.stats.passes_dispatched += 1;
            resolver.end_pass(pass.id)?;
        }
        tasks.execute();

        Ok(self.stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Barrier, PassGraph};
    use vibe_rhi::ResourceUsage;

    fn h(i: u32) -> ResourceHandle {
        ResourceHandle {
            index: i,
            generation: 0,
        }
    }

    #[derive(Default)]
    struct FakeBackend {
        known: Vec<ResourceHandle>,
        log: Vec<(usize, usize)>,
    }

    impl FakeBackend {
        fn with(resources: &[u32]) -> FakeBackend {
            FakeBackend {
                known: resources.iter().map(|i| h(*i)).collect(),
                log: Vec::new(),
            }
        }
    }

    impl ResourceResolver for FakeBackend {
        fn resolve(&self, resource: ResourceHandle) -> Option<u64> {
            self.known
                .contains(&resource)
                .then_some(resource.index() as u64)
        }

        fn begin_pass(&mut self, pass: PassId, barriers: &[Barrier]) -> Result<(), GraphError> {
            self.log.push((pass.0, barriers.len()));
            Ok(())
        }

        fn end_pass(&mut self, _pass: PassId) -> Result<(), GraphError> {
            Ok(())
        }
    }

    fn two_pass_graph() -> CompiledGraph {
        let mut g = PassGraph::new();
        g.add_pass("draw").write(h(0)).finish();
        g.add_pass("post").read(h(0)).finish();
        g.compile().unwrap()
    }

    #[test]
    fn empty_graph_executes_cleanly() {
        let pool = Arc::new(vibe_jobs::ThreadPool::new(2));
        let mut ex = GraphExecutor::new(pool);
        let mut backend = FakeBackend::default();
        let g = PassGraph::new().compile().unwrap();
        let stats = ex.execute(&g, &mut backend).unwrap();
        assert_eq!(stats.passes_executed, 0);
        assert!(backend.log.is_empty());
    }

    #[test]
    fn passes_run_in_compiled_order() {
        let pool = Arc::new(vibe_jobs::ThreadPool::new(2));
        let mut ex = GraphExecutor::new(pool);
        let mut backend = FakeBackend::with(&[0]);
        ex.execute(&two_pass_graph(), &mut backend).unwrap();
        assert_eq!(backend.log.len(), 2);
        assert_eq!(backend.log[0].0, 0, "draw must run before post");
        assert_eq!(backend.log[1].0, 1);
    }

    #[test]
    fn barriers_are_counted() {
        let pool = Arc::new(vibe_jobs::ThreadPool::new(2));
        let mut ex = GraphExecutor::new(pool);
        let mut backend = FakeBackend::with(&[0]);
        let stats = ex.execute(&two_pass_graph(), &mut backend).unwrap();
        assert_eq!(
            stats.barriers_emitted, 1,
            "write then read needs one barrier"
        );
        assert_eq!(backend.log[1].1, 1, "the barrier belongs to the read pass");
    }

    #[test]
    fn validate_accepts_live_handles() {
        let pool = Arc::new(vibe_jobs::ThreadPool::new(1));
        let ex = GraphExecutor::new(pool);
        let backend = FakeBackend::with(&[0]);
        assert!(ex.validate(&two_pass_graph(), &backend).is_ok());
    }

    #[test]
    fn validate_rejects_dead_handles() {
        let pool = Arc::new(vibe_jobs::ThreadPool::new(1));
        let ex = GraphExecutor::new(pool);
        let backend = FakeBackend::with(&[]);
        let err = ex.validate(&two_pass_graph(), &backend).unwrap_err();
        assert!(matches!(err, GraphError::DeadResource { .. }), "{err:?}");
        assert!(
            err.to_string().contains("draw"),
            "error should name the pass"
        );
    }

    #[test]
    fn execute_refuses_to_run_with_a_dead_handle() {
        let pool = Arc::new(vibe_jobs::ThreadPool::new(1));
        let mut ex = GraphExecutor::new(pool);
        let mut backend = FakeBackend::with(&[]);
        assert!(ex.execute(&two_pass_graph(), &mut backend).is_err());
        assert!(
            backend.log.is_empty(),
            "no pass may run when validation fails"
        );
    }

    #[test]
    fn bodies_run_for_every_pass() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let pool = Arc::new(vibe_jobs::ThreadPool::new(3));
        let mut ex = GraphExecutor::new(pool);
        let mut backend = FakeBackend::with(&[0]);
        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);
        let stats = ex
            .execute_with_bodies(&two_pass_graph(), &mut backend, move |_| {
                c.fetch_add(1, Ordering::AcqRel);
            })
            .unwrap();
        assert_eq!(count.load(Ordering::Acquire), 2);
        assert_eq!(stats.passes_dispatched, 2);
    }

    #[test]
    fn dependent_pass_bodies_run_producer_first() {
        let pool = Arc::new(vibe_jobs::ThreadPool::new(3));
        let mut ex = GraphExecutor::new(pool);
        let mut backend = FakeBackend::with(&[0]);
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let l = Arc::clone(&log);
        ex.execute_with_bodies(&two_pass_graph(), &mut backend, move |id| {
            l.lock().unwrap().push(id.0);
        })
        .unwrap();
        assert_eq!(
            *log.lock().unwrap(),
            vec![0, 1],
            "producer body must finish first"
        );
    }

    #[test]
    fn stats_reset_between_runs() {
        let pool = Arc::new(vibe_jobs::ThreadPool::new(2));
        let mut ex = GraphExecutor::new(pool);
        let mut backend = FakeBackend::with(&[0]);
        ex.execute(&two_pass_graph(), &mut backend).unwrap();
        let first = ex.stats();
        ex.execute(&two_pass_graph(), &mut backend).unwrap();
        assert_eq!(ex.stats(), first, "stats must not accumulate across frames");
    }

    #[test]
    fn pool_accessor_returns_the_pool() {
        let pool = Arc::new(vibe_jobs::ThreadPool::new(1));
        let ex = GraphExecutor::new(Arc::clone(&pool));
        assert_eq!(ex.pool().queued(), 0);
    }

    #[test]
    fn barrier_execution_dependency_rules() {
        let read_to_read = Barrier {
            resource: h(0),
            pass: PassId(0),
            from: ResourceUsage::Read,
            to: ResourceUsage::Read,
        };
        assert!(!read_to_read.needs_execution_dependency());
        assert!(read_to_read.is_noop());

        let write_to_read = Barrier {
            resource: h(0),
            pass: PassId(0),
            from: ResourceUsage::Write,
            to: ResourceUsage::Read,
        };
        assert!(write_to_read.needs_execution_dependency());
        assert!(!write_to_read.is_noop());
    }
}
