//! Frame graph for vibeEngine.
//!
//! A pass declares what it reads and writes; the graph infers execution order,
//! synthesizes the barriers, drops passes that produce nothing, and reports each
//! resource's lifetime. The same shape as Godot's RenderingDevice dependency
//! tracking and Unreal's RDG, without either framework's assumptions.

pub mod aliasing;
pub mod dump;
pub mod error;
pub mod executor;
pub mod graph;
pub mod registry;

pub use aliasing::{AliasAllocator, AliasGroup, MemoryPlan};
pub use dump::to_json;
pub use error::GraphError;
pub use executor::{ExecutionStats, GraphExecutor, ResourceResolver};
pub use graph::{
    Barrier, CompiledGraph, CompiledPass, PassBuilder, PassGraph, PassId, ResourceLifetime,
    topology_hash,
};
pub use registry::{ResourceDesc, ResourceRegistry};
