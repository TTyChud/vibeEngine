//! Type-erased operations on a component set, so despawn can clear an entity
//! from every set without knowing their types.

use std::any::Any;

use crate::entity::Entity;
use crate::sparse_set::SparseSet;

/// A component set seen through a fixed interface.
///
/// The `Any` supertrait is what lets a caller downcast back to a concrete
/// `SparseSet<T>` when it does know the type.
pub(crate) trait ErasedSet: Any {
    fn remove_entity(&mut self, entity: Entity) -> bool;
    /// Total components held in this set, for world-level statistics.
    fn count(&self) -> usize;
    fn clear_all(&mut self);
    /// True when the set holds no components.
    fn is_empty_set(&self) -> bool;
}

impl<T: 'static> ErasedSet for SparseSet<T> {
    fn remove_entity(&mut self, entity: Entity) -> bool {
        self.remove(entity).is_some()
    }

    fn count(&self) -> usize {
        self.len()
    }

    fn clear_all(&mut self) {
        self.clear();
    }

    fn is_empty_set(&self) -> bool {
        self.is_empty()
    }
}
