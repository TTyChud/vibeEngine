//! The ECS `World`: entity allocation, type-erased component storage, queries.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::entity::{Entity, Uuid};
use crate::erased::ErasedSet;
use crate::sparse_set::SparseSet;

type Storage = Box<dyn ErasedSet>;

/// Stores entities and their components.
///
/// Components live in one [`SparseSet`] per type, so iteration is cache-local
/// and adding a component costs an O(1) insert regardless of how many
/// entities exist.
pub struct World {
    generations: Vec<u32>,
    free: Vec<u32>,
    alive: Vec<bool>,
    uuids: HashMap<Entity, Uuid>,
    components: HashMap<TypeId, Storage>,
}

impl Default for World {
    fn default() -> Self {
        World::new()
    }
}

impl World {
    /// An empty world.
    pub fn new() -> World {
        World {
            generations: Vec::new(),
            free: Vec::new(),
            alive: Vec::new(),
            uuids: HashMap::new(),
            components: HashMap::new(),
        }
    }

    /// Create an entity with a fresh UUID.
    pub fn spawn(&mut self) -> Entity {
        let entity = self.allocate();
        self.uuids.insert(entity, Uuid::new_v4());
        entity
    }

    /// Create an entity with a chosen UUID, for scene loading.
    ///
    /// Returns the handle, which may differ from a previously despawned entity
    /// that held the same UUID.
    pub fn spawn_with_uuid(&mut self, uuid: Uuid) -> Entity {
        let entity = self.allocate();
        self.uuids.insert(entity, uuid);
        entity
    }

    fn allocate(&mut self) -> Entity {
        if let Some(index) = self.free.pop() {
            self.alive[index as usize] = true;
            let generation = self.generations[index as usize];
            return Entity::from_parts(index, generation);
        }
        let index = self.generations.len() as u32;
        self.generations.push(0);
        self.alive.push(true);
        Entity::from_parts(index, 0)
    }

    /// Destroy an entity and every component on it.
    ///
    /// Returns `false` if the handle was already invalid or stale.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        self.remove_all_components(entity);
        let index = entity.index() as usize;
        self.alive[index] = false;
        self.uuids.remove(&entity);
        self.generations[index] = self.generations[index].wrapping_add(1);
        self.free.push(index as u32);
        true
    }

    /// True when `entity` is a live handle with a matching generation.
    pub fn is_alive(&self, entity: Entity) -> bool {
        let index = entity.index() as usize;
        index < self.alive.len()
            && self.alive[index]
            && self.generations[index] == entity.generation()
    }

    /// The UUID assigned to `entity`, if it is alive.
    pub fn uuid(&self, entity: Entity) -> Option<Uuid> {
        self.uuids.get(&entity).copied()
    }

    /// Find the live entity carrying `uuid`.
    pub fn entity_by_uuid(&self, uuid: Uuid) -> Option<Entity> {
        self.uuids
            .iter()
            .find(|(_, u)| **u == uuid)
            .map(|(e, _)| *e)
    }

    /// Despawn every entity and drop every component.
    pub fn clear(&mut self) {
        for storage in self.components.values_mut() {
            storage.clear_all();
        }
        self.generations.clear();
        self.alive.clear();
        self.free.clear();
        self.uuids.clear();
    }

    /// Number of live entities.
    pub fn len(&self) -> usize {
        self.alive.iter().filter(|a| **a).count()
    }

    /// True when no entities are alive.
    pub fn is_empty(&self) -> bool {
        !self.alive.iter().any(|a| *a)
    }

    /// Every live entity.
    pub fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.alive
            .iter()
            .enumerate()
            .filter(|(_, a)| **a)
            .map(|(i, _)| Entity::from_parts(i as u32, self.generations[i]))
    }

    fn set<T: 'static>(&mut self, entity: Entity, value: T) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.storage_mut::<T>().insert(entity, value)
    }

    fn storage_mut<T: 'static>(&mut self) -> &mut SparseSet<T> {
        let id = TypeId::of::<T>();
        let storage = self
            .components
            .entry(id)
            .or_insert_with(|| Box::new(SparseSet::<T>::new()));
        let erased: &mut dyn Any = &mut **storage;
        erased
            .downcast_mut::<SparseSet<T>>()
            .expect("component storage type mismatch")
    }

    fn storage_ref<T: 'static>(&self) -> Option<&SparseSet<T>> {
        let storage: &dyn Any = &**self.components.get(&TypeId::of::<T>())?;
        storage.downcast_ref::<SparseSet<T>>()
    }

    fn remove<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        let storage: &mut dyn Any = &mut **self.components.get_mut(&TypeId::of::<T>())?;
        storage.downcast_mut::<SparseSet<T>>()?.remove(entity)
    }

    fn remove_all_components(&mut self, entity: Entity) {
        for storage in self.components.values_mut() {
            storage.remove_entity(entity);
        }
    }
}

impl World {
    /// Add or replace a component, returning the previous value.
    ///
    /// Returns `None` if the entity is not alive.
    pub fn add<T: 'static>(&mut self, entity: Entity, component: T) -> Option<T> {
        self.set(entity, component)
    }

    /// Read a component.
    pub fn get<T: 'static>(&self, entity: Entity) -> Option<&T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.storage_ref::<T>()?.get(entity)
    }

    /// Mutably read a component.
    pub fn get_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.storage_mut::<T>().get_mut(entity)
    }

    /// True when the entity has a component of this type.
    pub fn has<T: 'static>(&self, entity: Entity) -> bool {
        self.is_alive(entity) && self.storage_ref::<T>().is_some_and(|s| s.contains(entity))
    }

    /// Remove a component, returning it if present.
    pub fn remove_component<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        self.remove::<T>(entity)
    }

    /// Every `(entity, component)` pair of type `T`.
    pub fn query<T: 'static>(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.storage_ref::<T>().into_iter().flat_map(|s| s.iter())
    }

    /// Every `(entity, component)` pair of type `T` with mutable access.
    ///
    /// Borrows all of `self` mutably, so it cannot be combined with another
    /// `&self` borrow in the same expression.
    pub fn query_mut<T: 'static>(&mut self) -> impl Iterator<Item = (Entity, &mut T)> {
        self.storage_mut::<T>().iter_mut()
    }

    /// Number of entities holding a component of type `T`.
    pub fn count<T: 'static>(&self) -> usize {
        self.storage_ref::<T>().map_or(0, |s| s.len())
    }

    /// True when no entity anywhere holds a component of type `T`.
    ///
    /// Cheaper than [`World::count`] when the answer is usually "no", since an
    /// unregistered component type returns immediately.
    pub fn is_component_type_empty<T: 'static>(&self) -> bool {
        self.components
            .get(&TypeId::of::<T>())
            .is_none_or(|s| s.is_empty_set())
    }
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let components: usize = self.components.values().map(|s| s.count()).sum();
        f.debug_struct("World")
            .field("entities", &self.len())
            .field("component_types", &self.components.len())
            .field("components", &components)
            .finish()
    }
}

/// Errors from world operations.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum WorldError {
    /// A handle referred to a despawned or recycled entity.
    #[error("entity {0} is not alive")]
    DeadEntity(String),
    /// A UUID was not present in the world.
    #[error("uuid {0} not found in world")]
    UnknownUuid(String),
    /// A required component was missing.
    #[error("entity {0} is missing component {1}")]
    MissingComponent(String, &'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{SpriteRenderer, Tag, Transform};

    #[test]
    fn spawn_creates_alive_entity_with_uuid() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.is_alive(e));
        assert!(w.uuid(e).is_some_and(|u| !u.is_nil()));
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn uuids_are_unique_across_spawns() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        assert_ne!(w.uuid(a), w.uuid(b));
    }

    #[test]
    fn add_get_round_trip() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.add(e, Transform::default()).is_none());
        assert_eq!(w.get::<Transform>(e), Some(&Transform::default()));
        assert!(w.has::<Transform>(e));
    }

    #[test]
    fn add_replaces_previous_value() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            Transform {
                x: 1.0,
                ..Default::default()
            },
        );
        let old = w.add(
            e,
            Transform {
                x: 2.0,
                ..Default::default()
            },
        );
        assert_eq!(old.map(|t| t.x), Some(1.0));
        assert_eq!(w.get::<Transform>(e).map(|t| t.x), Some(2.0));
    }

    #[test]
    fn get_mut_mutates_in_place() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.get_mut::<Transform>(e).unwrap().x = 42.0;
        assert_eq!(w.get::<Transform>(e).map(|t| t.x), Some(42.0));
    }

    #[test]
    fn get_on_dead_entity_is_none() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.despawn(e);
        assert_eq!(w.get::<Transform>(e), None);
        assert!(!w.has::<Transform>(e));
    }

    #[test]
    fn remove_component_returns_it() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Tag::new("player"));
        assert_eq!(w.remove_component::<Tag>(e), Some(Tag::new("player")));
        assert!(!w.has::<Tag>(e));
        assert_eq!(w.remove_component::<Tag>(e), None);
    }

    #[test]
    fn despawn_clears_all_components() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.add(e, Tag::new("x"));
        w.add(e, SpriteRenderer::default());
        assert!(w.despawn(e));
        assert!(!w.has::<Transform>(e));
        assert!(!w.has::<Tag>(e));
        assert!(!w.has::<SpriteRenderer>(e));
        assert_eq!(w.len(), 0);
    }

    #[test]
    fn despawn_twice_returns_false() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.despawn(e));
        assert!(!w.despawn(e));
    }

    #[test]
    fn recycled_slot_gets_new_generation() {
        let mut w = World::new();
        let a = w.spawn();
        w.add(a, Tag::new("old"));
        w.despawn(a);
        let b = w.spawn();
        assert_eq!(a.index(), b.index(), "slot should be reused");
        assert_ne!(a.generation(), b.generation());
        assert!(
            w.has::<Tag>(a) == false,
            "stale handle must not see the new entity's data"
        );
    }

    #[test]
    fn stale_handle_cannot_despawn_new_entity() {
        let mut w = World::new();
        let a = w.spawn();
        w.despawn(a);
        let b = w.spawn();
        assert!(
            !w.despawn(a),
            "a stale handle must not despawn the recycled entity"
        );
        assert!(w.is_alive(b));
    }

    #[test]
    fn query_visits_every_entity_with_component() {
        let mut w = World::new();
        let mut entities = Vec::new();
        for i in 0..5 {
            let e = w.spawn();
            w.add(
                e,
                Transform {
                    x: i as f32,
                    ..Default::default()
                },
            );
            entities.push(e);
        }
        let tagged: Vec<_> = w.query::<Transform>().map(|(e, _)| e).collect();
        assert_eq!(tagged.len(), 5);
        for e in entities {
            assert!(tagged.contains(&e));
        }
    }

    #[test]
    fn query_mut_updates_all() {
        let mut w = World::new();
        for _ in 0..3 {
            let e = w.spawn();
            w.add(e, Transform::default());
        }
        for (_, t) in w.query_mut::<Transform>() {
            t.x = 7.0;
        }
        assert!(w.query::<Transform>().all(|(_, t)| t.x == 7.0));
    }

    #[test]
    fn count_tracks_component_population() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        w.add(a, Tag::new("a"));
        assert_eq!(w.count::<Tag>(), 1);
        w.add(b, Tag::new("b"));
        assert_eq!(w.count::<Tag>(), 2);
        w.despawn(b);
        assert_eq!(w.count::<Tag>(), 1);
    }

    #[test]
    fn entities_lists_only_alive() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        w.despawn(a);
        let all: Vec<_> = w.entities().collect();
        assert_eq!(all, vec![b]);
    }

    #[test]
    fn spawn_with_uuid_then_lookup_by_uuid() {
        let mut w = World::new();
        let uuid = Uuid::new_v4();
        let e = w.spawn_with_uuid(uuid);
        assert_eq!(w.uuid(e), Some(uuid));
        assert_eq!(w.entity_by_uuid(uuid), Some(e));
    }

    #[test]
    fn entity_by_uuid_missing_is_none() {
        let w = World::new();
        assert_eq!(w.entity_by_uuid(Uuid::new_v4()), None);
    }

    #[test]
    fn add_to_dead_entity_is_ignored() {
        let mut w = World::new();
        let e = w.spawn();
        w.despawn(e);
        assert!(w.add(e, Tag::new("x")).is_none());
        assert!(!w.has::<Tag>(e));
    }

    #[test]
    fn many_entities_spawn_and_despawn() {
        let mut w = World::new();
        let mut all = Vec::new();
        for i in 0..1000 {
            let e = w.spawn();
            w.add(
                e,
                Transform {
                    x: i as f32,
                    ..Default::default()
                },
            );
            all.push(e);
        }
        assert_eq!(w.len(), 1000);
        for e in all.iter() {
            assert!(w.despawn(*e));
        }
        assert!(w.is_empty());
        assert_eq!(w.count::<Transform>(), 0);
    }

    #[test]
    fn clear_empties_the_world() {
        let mut w = World::new();
        for i in 0..10 {
            let e = w.spawn();
            w.add(
                e,
                Transform {
                    x: i as f32,
                    ..Default::default()
                },
            );
            w.add(e, Tag::new("t"));
        }
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.count::<Transform>(), 0);
        assert_eq!(w.count::<Tag>(), 0);
        assert_eq!(w.entities().count(), 0);
    }

    #[test]
    fn world_is_reusable_after_clear() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Tag::new("old"));
        w.clear();
        let e2 = w.spawn();
        assert!(w.is_alive(e2));
        assert!(
            !w.has::<Tag>(e2),
            "component storage must be empty after clear"
        );
    }

    #[test]
    fn is_component_type_empty_tracks_population() {
        let mut w = World::new();
        assert!(
            w.is_component_type_empty::<Tag>(),
            "unregistered type is empty"
        );
        let e = w.spawn();
        w.add(e, Tag::new("a"));
        assert!(!w.is_component_type_empty::<Tag>());
        assert_eq!(w.count::<Tag>(), 1);
        w.despawn(e);
        assert!(
            w.is_component_type_empty::<Tag>(),
            "set survives despawn but is empty"
        );
    }

    #[test]
    fn debug_impl_is_usable() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        let s = format!("{w:?}");
        assert!(s.contains("entities: 1"), "{s}");
    }

    #[test]
    fn empty_world_reports_empty() {
        let w = World::new();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
    }
}
