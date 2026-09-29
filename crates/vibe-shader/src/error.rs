//! Errors from shader compilation.

/// A shader could not be compiled.
#[derive(Debug, thiserror::Error)]
pub enum ShaderError {
    /// The source is not valid WGSL.
    #[error("wgsl parse error: {0}")]
    Parse(String),
    /// The module parsed but failed validation.
    ///
    /// Carries the message and the source line, which is what an author needs
    /// to fix it.
    #[error("shader validation failed at line {line}: {message}")]
    Validation {
        /// 1-based line the error points at, or 0 when unknown.
        line: usize,
        /// The diagnostic text.
        message: String,
    },
    /// The SPIR-V backend rejected the module.
    #[error("spirv generation failed: {0}")]
    Backend(String),
    /// A shader was requested that is not in the library.
    #[error("no shader named {0:?}")]
    NotFound(String),
}
