//! 2D and 3D rigid body physics for vibeEngine.
//!
//! Backed by Rapier, which is the Rust-native equivalent of Box2D (2D) and Jolt
//! (3D). The solver lives behind [`sim2d`] and [`sim3d`], so the rest of the
//! engine only sees ECS components and opaque handles.

pub mod error;
pub mod registry;
pub mod sim2d;
pub mod sim3d;

pub use error::PhysicsError;
pub use registry::{
    BodyCounts, BodyHandle, BodyTransform, Dimension, PhysicsRegistry, Shape3D, effective_mass_2d,
    effective_mass_3d, has_any_collider, shape_from_world, summarise, transform_from_world,
    write_transform,
};
