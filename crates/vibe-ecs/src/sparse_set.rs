//! Sparse-set storage for one component type.

use std::marker::PhantomData;

use crate::Entity;

pub(crate) const NONE: u32 = u32::MAX;

pub(crate) struct SparseSet<T> {
    dense: Vec<(Entity, T)>,
    /// Dense index per slot, tagged with the generation that wrote it, so a
    /// handle from before a slot was recycled misses instead of aliasing.
    sparse: Vec<(u32, u32)>,
    _marker: PhantomData<T>,
}

impl<T> SparseSet<T> {
    pub(crate) fn new() -> Self {
        SparseSet {
            dense: Vec::new(),
            sparse: Vec::new(),
            _marker: PhantomData,
        }
    }

    fn slot(&self, entity: Entity) -> Option<usize> {
        let idx = entity.index() as usize;
        let (dense_index, generation) = *self.sparse.get(idx)?;
        if dense_index == NONE || generation != entity.generation() {
            return None;
        }
        Some(dense_index as usize)
    }

    pub(crate) fn insert(&mut self, entity: Entity, value: T) -> Option<T> {
        let idx = entity.index() as usize;
        if idx >= self.sparse.len() {
            self.sparse.resize(idx + 1, (NONE, 0));
        }
        if let Some(dense_index) = self.slot(entity) {
            return Some(std::mem::replace(&mut self.dense[dense_index].1, value));
        }
        self.sparse[idx] = (self.dense.len() as u32, entity.generation());
        self.dense.push((entity, value));
        None
    }

    pub(crate) fn get(&self, entity: Entity) -> Option<&T> {
        Some(&self.dense[self.slot(entity)?].1)
    }

    pub(crate) fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        let dense_index = self.slot(entity)?;
        Some(&mut self.dense[dense_index].1)
    }

    pub(crate) fn contains(&self, entity: Entity) -> bool {
        self.slot(entity).is_some()
    }

    pub(crate) fn remove(&mut self, entity: Entity) -> Option<T> {
        let dense_index = self.slot(entity)?;
        self.sparse[entity.index() as usize] = (NONE, 0);
        let (_, value) = self.dense.swap_remove(dense_index);
        if dense_index < self.dense.len() {
            let moved = self.dense[dense_index].0;
            self.sparse[moved.index() as usize] = (dense_index as u32, moved.generation());
        }
        Some(value)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.dense.iter().map(|(e, v)| (*e, v))
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (Entity, &mut T)> {
        self.dense.iter_mut().map(|(e, v)| (*e, v))
    }

    pub(crate) fn len(&self) -> usize {
        self.dense.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        for slot in self.sparse.iter_mut() {
            *slot = (NONE, 0);
        }
        self.dense.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut set = SparseSet::new();
        let e = Entity::from_parts(0, 1);
        assert!(set.insert(e, 42i32).is_none());
        assert_eq!(set.get(e), Some(&42));
        assert!(set.contains(e));
    }

    #[test]
    fn insert_replaces_and_returns_old() {
        let mut set = SparseSet::new();
        let e = Entity::from_parts(3, 1);
        set.insert(e, 1i32);
        assert_eq!(set.insert(e, 2i32), Some(1));
        assert_eq!(set.get(e), Some(&2));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn get_missing_returns_none() {
        let set: SparseSet<i32> = SparseSet::new();
        assert_eq!(set.get(Entity::from_parts(0, 1)), None);
        assert!(!set.contains(Entity::from_parts(9, 1)));
    }

    #[test]
    fn get_out_of_sparse_range() {
        let mut set = SparseSet::new();
        set.insert(Entity::from_parts(1, 1), 7i32);
        assert_eq!(set.get(Entity::from_parts(1000, 1)), None);
    }

    #[test]
    fn remove_returns_value_and_clears_membership() {
        let mut set = SparseSet::new();
        let e = Entity::from_parts(2, 1);
        set.insert(e, 5i32);
        assert_eq!(set.remove(e), Some(5));
        assert!(!set.contains(e));
        assert_eq!(set.get(e), None);
        assert_eq!(set.remove(e), None);
    }

    #[test]
    fn remove_repairs_sparse_index_of_swapped_entity() {
        let mut set = SparseSet::new();
        let a = Entity::from_parts(0, 1);
        let b = Entity::from_parts(1, 1);
        let c = Entity::from_parts(2, 1);
        set.insert(a, 10i32);
        set.insert(b, 20i32);
        set.insert(c, 30i32);
        set.remove(a);
        assert_eq!(set.get(c), Some(&30));
        assert_eq!(set.get(b), Some(&20));
        assert!(!set.contains(a));
    }

    #[test]
    fn get_mut_allows_in_place_update() {
        let mut set = SparseSet::new();
        let e = Entity::from_parts(4, 1);
        set.insert(e, 1i32);
        *set.get_mut(e).unwrap() = 99;
        assert_eq!(set.get(e), Some(&99));
    }

    #[test]
    fn iter_yields_all_inserted_pairs() {
        let mut set = SparseSet::new();
        set.insert(Entity::from_parts(0, 1), "a");
        set.insert(Entity::from_parts(1, 1), "b");
        let mut found: Vec<_> = set.iter().map(|(_, v)| *v).collect();
        found.sort();
        assert_eq!(found, vec!["a", "b"]);
    }

    #[test]
    fn iter_mut_updates_all() {
        let mut set = SparseSet::new();
        set.insert(Entity::from_parts(0, 1), 1i32);
        set.insert(Entity::from_parts(1, 1), 2i32);
        for (_, v) in set.iter_mut() {
            *v *= 10;
        }
        let mut vals: Vec<_> = set.iter().map(|(_, v)| *v).collect();
        vals.sort();
        assert_eq!(vals, vec![10, 20]);
    }

    #[test]
    fn clear_empties_both_structures() {
        let mut set = SparseSet::new();
        set.insert(Entity::from_parts(0, 1), 1i32);
        set.insert(Entity::from_parts(5, 1), 2i32);
        set.clear();
        assert!(set.is_empty());
        assert!(!set.contains(Entity::from_parts(5, 1)));
    }

    #[test]
    fn entity_version_does_not_affect_membership() {
        let mut set = SparseSet::new();
        set.insert(Entity::from_parts(3, 1), 8i32);
        assert!(
            !set.contains(Entity::from_parts(3, 2)),
            "a stale generation must miss"
        );
        assert!(set.contains(Entity::from_parts(3, 1)));
    }

    #[test]
    fn sparse_grows_to_high_index() {
        let mut set = SparseSet::new();
        let e = Entity::from_parts(50_000, 1);
        set.insert(e, 1i32);
        assert_eq!(set.get(e), Some(&1));
    }
}
