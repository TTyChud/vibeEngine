//! WGSL to SPIR-V compilation, via naga.
//!
//! Pure Rust, so there is no C++ toolchain to build and no external binary to
//! ship: the engine compiles its own shaders. The compiler sits behind a
//! [`ShaderCompiler`] trait so a precompiled-SPIR-V path or a future Slang
//! backend can replace it without touching callers.

pub mod compiler;
pub mod error;
pub mod library;

pub use compiler::{
    CompileOptions, SPIRV_MAGIC, ShaderCompiler, WgslCompiler, compile_entry_point, compile_wgsl,
    looks_like_spirv, words_to_bytes,
};
pub use error::ShaderError;
pub use library::{ShaderKind, ShaderLibrary};
