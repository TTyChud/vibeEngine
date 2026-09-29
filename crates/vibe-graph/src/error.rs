//! Typed errors from graph construction and compilation.

use vibe_rhi::ResourceHandle;

/// A graph could not be built or compiled.
#[derive(Debug, thiserror::Error, PartialEq, Eq, Clone)]
pub enum GraphError {
    /// A pass read a resource that no earlier pass writes and that was not
    /// declared as an external write, so the read would be uninitialized.
    #[error("pass {pass} reads resource {resource:?} before any pass writes it")]
    ReadBeforeWrite {
        /// The offending pass.
        pass: String,
        /// The resource it read.
        resource: ResourceHandle,
    },
    /// The dependency edges form a cycle, so some pass can never run.
    #[error("pass dependency cycle, no pass can start at {pass}")]
    Cycle {
        /// A pass stuck in the cycle.
        pass: String,
    },
    /// A pass referenced a handle that is no longer live.
    #[error("pass {pass} references dead resource {resource:?}")]
    DeadResource {
        /// The offending pass.
        pass: String,
        /// The stale handle.
        resource: ResourceHandle,
    },
    /// The same pass name was registered twice.
    #[error("duplicate pass name {0}")]
    DuplicatePass(String),
    /// A pass id was out of range.
    #[error("pass id {0} is out of range")]
    UnknownPass(usize),
}
