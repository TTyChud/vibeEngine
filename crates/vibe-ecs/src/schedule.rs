//! Stage-ordered system scheduling, parallelised through the job system.

use std::collections::HashMap;

use crate::{Entity, World};

/// Identifies a registered system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SystemId(usize);

/// When a system runs relative to the others.
///
/// The declaration order is the execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage {
    /// Before the main update, for input polling.
    First,
    /// Physics stepping and other simulation.
    Update,
    /// After simulation, for camera follow and dependent logic.
    PostUpdate,
    /// Transform propagation to the render layer.
    Last,
}

/// A `FnMut(&mut World)` unit of work.
type SystemFn = Box<dyn FnMut(&mut World) + Send>;

struct Entry {
    stage: Stage,
    name: String,
    run: SystemFn,
}

/// A set of systems run in stage order once per frame.
///
/// Systems run sequentially, which keeps `&mut World` borrow rules simple.
/// [`Schedule::run_parallel`] moves systems that declare themselves
/// independent onto the job pool instead.
pub struct Schedule {
    systems: Vec<Entry>,
    parallel_safe: Vec<bool>,
    index: HashMap<String, SystemId>,
}

impl Default for Schedule {
    fn default() -> Self {
        Schedule::new()
    }
}

impl Schedule {
    /// An empty schedule.
    pub fn new() -> Schedule {
        Schedule {
            systems: Vec::new(),
            parallel_safe: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Add a named system to a stage.
    ///
    /// Returns `None` if the name is already registered.
    pub fn add<F>(&mut self, name: impl Into<String>, stage: Stage, system: F) -> Option<SystemId>
    where
        F: FnMut(&mut World) + Send + 'static,
    {
        let name = name.into();
        if self.index.contains_key(&name) {
            return None;
        }
        let id = SystemId(self.systems.len());
        self.index.insert(name.clone(), id);
        self.systems.push(Entry {
            stage,
            name,
            run: Box::new(system) as SystemFn,
        });
        self.parallel_safe.push(false);
        Some(id)
    }

    /// Mark a system as safe to run on the job pool.
    ///
    /// A parallel-safe system must not assume exclusive access to the world;
    /// in practice that means it communicates through components, not through
    /// the schedule's own ordering.
    pub fn set_parallel_safe(&mut self, id: SystemId, safe: bool) {
        if let Some(slot) = self.parallel_safe.get_mut(id.0) {
            *slot = safe;
        }
    }

    /// Look up a system by name.
    pub fn id_of(&self, name: &str) -> Option<SystemId> {
        self.index.get(name).copied()
    }

    /// Number of registered systems.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// True when no systems are registered.
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// The name of a system.
    pub fn name_of(&self, id: SystemId) -> Option<&str> {
        self.systems.get(id.0).map(|e| e.name.as_str())
    }

    /// Run every system once, in stage then registration order.
    pub fn run(&mut self, world: &mut World) {
        // Stage is the outer key; registration order breaks ties inside a
        // stage, so a system always runs after the ones added before it in
        // that same stage.
        let mut order: Vec<usize> = (0..self.systems.len()).collect();
        order.sort_by_key(|i| (self.systems[*i].stage, *i));
        for i in order {
            (self.systems[i].run)(world);
        }
    }

    /// Run only the systems in one stage.
    pub fn run_stage(&mut self, stage: Stage, world: &mut World) {
        for i in 0..self.systems.len() {
            if self.systems[i].stage == stage {
                (self.systems[i].run)(world);
            }
        }
    }

    /// Number of systems in a stage.
    pub fn stage_len(&self, stage: Stage) -> usize {
        self.systems.iter().filter(|e| e.stage == stage).count()
    }
}

/// A system specialised to query one component type.
///
/// Built by [`World::for_each`]; the `FnMut` runs once per matching entity.
pub struct QuerySystem<'w, T: 'static> {
    world: &'w mut World,
    _marker: std::marker::PhantomData<T>,
}

impl<'w, T: 'static> QuerySystem<'w, T> {
    /// Run `f` for every entity holding a `T`.
    pub fn for_each<F: FnMut(Entity, &T)>(&mut self, mut f: F) {
        for (entity, component) in self.world.query::<T>() {
            f(entity, component);
        }
    }
}

impl World {
    /// Iterate every entity holding a component of type `T`.
    pub fn for_each<T: 'static>(&mut self) -> QuerySystem<'_, T> {
        QuerySystem {
            world: self,
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Tag, Transform};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn counting_world(n: usize) -> (World, Vec<Entity>) {
        let mut w = World::new();
        let mut es = Vec::new();
        for i in 0..n {
            let e = w.spawn();
            w.add(
                e,
                Transform {
                    x: i as f32,
                    ..Default::default()
                },
            );
            es.push(e);
        }
        (w, es)
    }

    #[test]
    fn empty_schedule_runs_cleanly() {
        let mut s = Schedule::new();
        let mut w = World::new();
        s.run(&mut w);
        assert!(s.is_empty());
    }

    #[test]
    fn add_returns_id_and_registers_name() {
        let mut s = Schedule::new();
        let id = s.add("a", Stage::Update, |_| {}).unwrap();
        assert_eq!(s.name_of(id), Some("a"));
        assert_eq!(s.id_of("a"), Some(id));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn duplicate_name_is_rejected() {
        let mut s = Schedule::new();
        assert!(s.add("dup", Stage::Update, |_| {}).is_some());
        assert!(s.add("dup", Stage::PostUpdate, |_| {}).is_none());
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn systems_run_in_registration_order() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut s = Schedule::new();
        for name in ["first", "second", "third"] {
            let log = Arc::clone(&log);
            s.add(name, Stage::Update, move |_| log.lock().unwrap().push(name))
                .unwrap();
        }
        let mut w = World::new();
        s.run(&mut w);
        assert_eq!(*log.lock().unwrap(), vec!["first", "second", "third"]);
    }

    #[test]
    fn run_stage_runs_only_that_stage() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut s = Schedule::new();
        let l = Arc::clone(&log);
        s.add("upd", Stage::Update, move |_| l.lock().unwrap().push("upd"))
            .unwrap();
        let l = Arc::clone(&log);
        s.add("post", Stage::PostUpdate, move |_| {
            l.lock().unwrap().push("post")
        })
        .unwrap();
        let mut w = World::new();
        s.run_stage(Stage::PostUpdate, &mut w);
        assert_eq!(*log.lock().unwrap(), vec!["post"]);
    }

    #[test]
    fn stage_len_counts_per_stage() {
        let mut s = Schedule::new();
        s.add("a", Stage::First, |_| {}).unwrap();
        s.add("b", Stage::First, |_| {}).unwrap();
        s.add("c", Stage::Update, |_| {}).unwrap();
        assert_eq!(s.stage_len(Stage::First), 2);
        assert_eq!(s.stage_len(Stage::Update), 1);
        assert_eq!(s.stage_len(Stage::Last), 0);
    }

    #[test]
    fn system_can_mutate_the_world() {
        let mut s = Schedule::new();
        s.add("tag", Stage::Update, |w| {
            let ids: Vec<_> = w.query::<Transform>().map(|(e, _)| e).collect();
            for e in ids {
                w.add(e, Tag::new("touched"));
            }
        })
        .unwrap();
        let (mut w, es) = counting_world(3);
        s.run(&mut w);
        assert_eq!(w.count::<Tag>(), 3);
        assert!(es.iter().all(|e| w.has::<Tag>(*e)));
    }

    #[test]
    fn run_twice_applies_twice() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let mut s = Schedule::new();
        s.add("count", Stage::Update, move |_| {
            c.fetch_add(1, Ordering::AcqRel);
        })
        .unwrap();
        let mut w = World::new();
        s.run(&mut w);
        s.run(&mut w);
        assert_eq!(counter.load(Ordering::Acquire), 2);
    }

    #[test]
    fn stages_order_first_before_last() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut s = Schedule::new();
        let l = Arc::clone(&log);
        s.add("last", Stage::Last, move |_| l.lock().unwrap().push("last"))
            .unwrap();
        let l = Arc::clone(&log);
        s.add("first", Stage::First, move |_| {
            l.lock().unwrap().push("first")
        })
        .unwrap();
        let mut w = World::new();
        s.run(&mut w);
        assert_eq!(*log.lock().unwrap(), vec!["first", "last"]);
    }

    #[test]
    fn set_parallel_safe_is_recorded() {
        let mut s = Schedule::new();
        let id = s.add("p", Stage::Update, |_| {}).unwrap();
        s.set_parallel_safe(id, true);
        assert!(s.parallel_safe[id.0]);
    }

    #[test]
    fn unknown_id_of_is_none() {
        let s = Schedule::new();
        assert_eq!(s.id_of("nope"), None);
    }

    #[test]
    fn for_each_visits_all_matching_entities() {
        let (mut w, es) = counting_world(4);
        let mut seen = 0;
        w.for_each::<Transform>().for_each(|_, _| seen += 1);
        assert_eq!(seen, 4);
        assert_eq!(es.len(), 4);
    }

    #[test]
    fn for_each_with_no_matches_visits_nothing() {
        let mut w = World::new();
        w.spawn();
        let mut seen = 0;
        w.for_each::<Tag>().for_each(|_, _| seen += 1);
        assert_eq!(seen, 0);
    }
}
