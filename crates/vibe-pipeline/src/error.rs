//! Errors from pipeline creation.

use ash::vk;

/// A pipeline could not be built.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// The vertex layout has a gap, an overlap, or a byte size that does not
    /// match the stride.
    #[error("vertex layout is invalid: {0}")]
    BadLayout(String),
    /// Two attributes share a location or a shader binding.
    #[error("duplicate {kind} {value} in the pipeline description")]
    Duplicate {
        /// What collided: "location" or "binding".
        kind: &'static str,
        /// The value that collided.
        value: u32,
    },
    /// The description has no vertex or fragment stage.
    #[error("pipeline needs at least one shader stage")]
    NoStages,
    /// The driver rejected the pipeline.
    #[error("pipeline creation failed: {0:?}")]
    Vk(vk::Result),
    /// A shader was missing from the library.
    #[error("shader {0} is not in the library")]
    MissingShader(String),
}
