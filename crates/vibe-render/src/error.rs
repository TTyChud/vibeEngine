//! Errors from the renderer.

impl From<ash::vk::Result> for RenderError {
    fn from(e: ash::vk::Result) -> Self {
        RenderError::Vk(e)
    }
}

/// A frame could not be built or drawn.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// A batch would exceed the vertex buffer's capacity.
    #[error("batch of {quads} quads exceeds the {capacity}-quad buffer")]
    BatchTooLarge {
        /// Quads the caller tried to add.
        quads: usize,
        /// Quads the buffer holds.
        capacity: usize,
    },
    /// No camera was supplied, so there is no view-projection matrix.
    #[error("no camera")]
    NoCamera,
    /// A Vulkan call failed.
    #[error("vulkan call failed: {0:?}")]
    Vk(ash::vk::Result),
    /// A shader or pipeline could not be built.
    #[error("{0}")]
    Pipeline(String),
}
