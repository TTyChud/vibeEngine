//! Window creation and Vulkan surface wiring.

pub mod error;
pub mod surface;
pub mod window;

pub use error::WindowError;
pub use surface::{
    SurfaceDesc, SurfaceKind, create_surface, required_instance_extensions, surface_kind,
};
pub use window::{Window, WindowDesc, WindowEvents};
