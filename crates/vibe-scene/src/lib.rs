//! Scene serialization and the virtual filesystem for vibeEngine.

pub mod error;
pub mod scene;
pub mod vfs;

pub use error::SceneError;
pub use scene::{
    EntityRecord, SCENE_VERSION, SceneFile, scene_to_world, world_to_scene, world_to_yaml,
    yaml_to_world,
};
pub use vfs::{Vfs, VfsError};
