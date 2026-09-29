//! Window creation and Vulkan surface wiring.

pub mod error;
pub mod surface;
pub mod window;

pub use error::WindowError;
pub use surface::{SurfaceDesc, create_surface};
pub use window::{Window, WindowDesc, WindowEvents};
