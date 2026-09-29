//! Errors from window and surface creation.

/// A window or surface could not be created.
#[derive(Debug, thiserror::Error)]
pub enum WindowError {
    /// No display server was reachable.
    #[error("could not connect to a display server: {0}")]
    NoDisplay(String),
    /// The event loop could not be created.
    #[error("event loop: {0}")]
    EventLoop(String),
    /// The window could not be created.
    #[error("window: {0}")]
    Create(String),
    /// The window does not expose a usable raw handle.
    #[error("the window has no {0} handle")]
    NoHandle(&'static str),
    /// The surface could not be created for the platform.
    #[error("vulkan surface: {0}")]
    Surface(String),
    /// A Vulkan call failed.
    #[error("vulkan call failed: {0:?}")]
    Vk(ash::vk::Result),
}
