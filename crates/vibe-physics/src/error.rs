//! Errors from physics operations.

/// A physics operation could not be completed.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PhysicsError {
    /// The entity has no body handle, so it was never registered or has since
    /// been removed.
    #[error("entity {entity} has no {kind} body")]
    NoBody {
        /// The entity that was asked about.
        entity: String,
        /// Which body kind was requested.
        kind: &'static str,
    },
    /// The handle refers to a body the world has already destroyed.
    #[error("stale {kind} handle for entity {entity}")]
    StaleBody {
        /// The entity that was asked about.
        entity: String,
        /// Which body kind was requested.
        kind: &'static str,
    },
    /// The entity has a body component but no matching collider, so the body
    /// would have no shape.
    #[error("entity {0} has a body but no collider")]
    NoCollider(String),
    /// More than one collider component on one entity, which is ambiguous.
    #[error("entity {0} has more than one collider")]
    AmbiguousCollider(String),
    /// A collider was created with a non-positive dimension, which the solver
    /// would reject.
    #[error("{shape} collider on entity {entity} has a non-positive size")]
    DegenerateShape {
        /// The entity that was asked about.
        entity: String,
        /// Which shape was bad.
        shape: &'static str,
    },
    /// The timestep was zero or negative.
    #[error("invalid timestep {0}")]
    InvalidTimestep(f32),
}
