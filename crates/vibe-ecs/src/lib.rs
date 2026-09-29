//! Sparse-set entity component system for vibeEngine.
//!
//! Storage is one sparse set per component type: a dense `Vec` iterated in
//! order for cache locality, plus a sparse index from entity slot to dense
//! position. No external ECS dependency.

pub mod components;
mod entity;
mod erased;
mod schedule;
mod sparse_set;
mod world;

pub use components::*;
pub use entity::{Entity, Uuid};
pub use schedule::{Schedule, Stage, SystemId};
pub use world::{World, WorldError};

/// Marker for a component type.
///
/// Blanket-implemented for every `'static` type, so it is only useful as a
/// bound when you want to accept any component.
pub trait Component: 'static {}

impl<T: 'static> Component for T {}
