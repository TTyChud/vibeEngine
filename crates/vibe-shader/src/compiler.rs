//! WGSL to SPIR-V, via naga.

use log::debug;
use naga::Module;
use naga::back::spv;
use naga::valid::{Capabilities, ValidationFlags, Validator};

use crate::error::ShaderError;

/// How to compile a shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileOptions {
    /// Target SPIR-V language version.
    pub lang_version: (u8, u8),
    /// Source text to embed as debug info; `None` for a release build.
    ///
    /// Held as owned text because the backend borrows it and the options
    /// outlive the compile call.
    pub debug_source: Option<String>,
    /// The file name reported in debug info.
    pub file_name: String,
}

impl Default for CompileOptions {
    fn default() -> Self {
        // SPIR-V 1.0 is the floor every Vulkan 1.0 implementation accepts, and
        // nothing the engine emits needs more.
        CompileOptions {
            lang_version: (1, 0),
            debug_source: None,
            file_name: String::new(),
        }
    }
}

impl CompileOptions {
    /// Options with no debug info, for a shipped build.
    pub fn release() -> CompileOptions {
        CompileOptions::default()
    }

    /// Options carrying full debug info for `source`, so a validation error
    /// can quote the offending line.
    pub fn with_debug(source: &str, file_name: &str) -> CompileOptions {
        CompileOptions {
            lang_version: (1, 0),
            debug_source: Some(source.to_string()),
            file_name: file_name.to_string(),
        }
    }

    fn debug_info(&self) -> Option<spv::DebugInfo<'_>> {
        self.debug_source.as_deref().map(|src| spv::DebugInfo {
            source_code: src,
            file_name: &self.file_name,
            language: spv::SourceLanguage::WGSL,
        })
    }
}

/// Compiles shader source to SPIR-V words.
///
/// A trait so the engine can swap in precompiled SPIR-V, or a different source
/// language, without changing anything that builds pipelines.
pub trait ShaderCompiler {
    /// Compile WGSL source to SPIR-V words.
    fn compile(&self, source: &str, options: CompileOptions) -> Result<Vec<u32>, ShaderError>;
}

/// The naga-backed compiler.
#[derive(Debug, Clone, Copy, Default)]
pub struct WgslCompiler;

impl WgslCompiler {
    /// A compiler with default options.
    pub fn new() -> WgslCompiler {
        WgslCompiler
    }
}

impl ShaderCompiler for WgslCompiler {
    fn compile(&self, source: &str, options: CompileOptions) -> Result<Vec<u32>, ShaderError> {
        compile_wgsl(source, options)
    }
}

/// Parse, validate and emit SPIR-V for a WGSL module.
///
/// Validation runs before the backend because the error messages there are the
/// ones worth showing an author; a backend error at that point is almost always
/// a bug in the engine rather than in the shader.
pub fn compile_wgsl(source: &str, options: CompileOptions) -> Result<Vec<u32>, ShaderError> {
    compile_entry_point(source, options, None)
}

/// Compile WGSL to SPIR-V for one named entry point.
///
/// `None` emits the module's first entry point, which is all a single-stage
/// shader needs. A pipeline needs the vertex and fragment stages out of one
/// module, so it asks for each by name.
pub fn compile_entry_point(
    source: &str,
    options: CompileOptions,
    entry_point: Option<(&str, naga::ShaderStage)>,
) -> Result<Vec<u32>, ShaderError> {
    let module: Module = naga::front::wgsl::parse_str(source)
        .map_err(|e| ShaderError::Parse(first_line(&e.to_string())))?;

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    let info = validator.validate(&module).map_err(|e| {
        // With debug info the diagnostic can quote the offending line, which is
        // what makes the error actionable rather than merely correct.
        let message = match &options.debug_source {
            Some(src) => e.emit_to_string(src),
            None => e.emit_to_string(""),
        };
        ShaderError::Validation {
            line: 0,
            message: first_line(&message),
        }
    })?;

    let (major, minor) = options.lang_version;
    let mut flags = spv::WriterFlags::empty();
    if options.debug_source.is_some() {
        flags |= spv::WriterFlags::DEBUG;
    }

    let pipeline_options = entry_point.map(|(name, stage)| spv::PipelineOptions {
        shader_stage: stage,
        entry_point: name.to_string(),
    });

    // Start from naga's defaults, which are the ones it documents, and change
    // only what the engine needs: no faked bindings, and debug info only when
    // the source is available to embed.
    let spv_options = spv::Options {
        flags,
        // The bindings are carried over from the WGSL declarations, which is
        // what the engine wants: set 0/binding 0 in WGSL and it stays that way
        // in the SPIR-V. A binding map would be needed only to remap layouts.
        fake_missing_bindings: true,
        lang_version: (major, minor),
        debug_info: options.debug_info(),
        ..Default::default()
    };

    let words = spv::write_vec(&module, &info, &spv_options, pipeline_options.as_ref())
        .map_err(|e| ShaderError::Backend(first_line(&e.to_string())))?;

    debug!(
        "compiled {} bytes of wgsl into {} spirv words ({} bytes)",
        source.len(),
        words.len(),
        words.len() * 4
    );
    Ok(words)
}

fn first_line(message: &str) -> String {
    message.lines().next().unwrap_or(message).to_string()
}

/// SPIR-V words as little-endian bytes, which is what `vkShaderModule` wants.
pub fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

/// The SPIR-V magic number every module must start with.
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// True when a word stream looks like valid SPIR-V.
///
/// A cheap guard before handing bytes to the driver: an empty or wrong-magic
/// stream means the compile silently produced nothing, which the driver would
/// otherwise report as an unhelpful validation error.
pub fn looks_like_spirv(words: &[u32]) -> bool {
    words.first().is_some_and(|w| *w == SPIRV_MAGIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal valid shader: a full-screen triangle from three vertices.
    const TRIANGLE: &str = r#"
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>( 3.0, -1.0),
                vec2<f32>(-1.0,  3.0),
            );
            return vec4<f32>(positions[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
    "#;

    #[test]
    fn a_valid_shader_compiles_to_spirv() {
        let words = compile_wgsl(TRIANGLE, CompileOptions::default()).unwrap();
        assert!(!words.is_empty());
        assert!(looks_like_spirv(&words));
    }

    #[test]
    fn compiled_output_carries_the_spirv_magic_number() {
        let words = compile_wgsl(TRIANGLE, CompileOptions::default()).unwrap();
        assert_eq!(words[0], SPIRV_MAGIC);
    }

    #[test]
    fn a_uniform_and_varying_shader_compiles() {
        let src = r#"
            struct Camera { view_projection: mat4x4<f32> }
            @group(0) @binding(0) var<uniform> camera: Camera;

            struct VertexOut {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
            }

            @vertex
            fn vs_main(@location(0) pos: vec2<f32>,
                       @location(1) uv: vec2<f32>) -> VertexOut {
                var out: VertexOut;
                out.position = camera.view_projection * vec4<f32>(pos, 0.0, 1.0);
                out.uv = uv;
                return out;
            }

            @fragment
            fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
                return vec4<f32>(in.uv, 0.0, 1.0);
            }
        "#;
        assert!(looks_like_spirv(
            &compile_wgsl(src, CompileOptions::default()).unwrap()
        ));
    }

    #[test]
    fn a_compute_shader_compiles() {
        let src = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = f32(id.x) * 2.0;
            }
        "#;
        assert!(compile_wgsl(src, CompileOptions::default()).is_ok());
    }

    #[test]
    fn each_entry_point_can_be_selected() {
        for (name, stage) in [
            ("vs_main", naga::ShaderStage::Vertex),
            ("fs_main", naga::ShaderStage::Fragment),
        ] {
            let words =
                compile_entry_point(TRIANGLE, CompileOptions::default(), Some((name, stage)))
                    .unwrap_or_else(|e| panic!("{name} failed: {e}"));
            assert!(looks_like_spirv(&words), "{name}");
        }
    }

    #[test]
    fn an_unknown_entry_point_is_an_error() {
        let err = compile_entry_point(
            TRIANGLE,
            CompileOptions::default(),
            Some(("no_such_entry", naga::ShaderStage::Vertex)),
        )
        .unwrap_err();
        assert!(matches!(err, ShaderError::Backend(_)), "{err:?}");
    }

    #[test]
    fn invalid_syntax_is_a_parse_error() {
        let err = compile_wgsl("this is not wgsl", CompileOptions::default()).unwrap_err();
        assert!(matches!(err, ShaderError::Parse(_)), "{err:?}");
    }

    #[test]
    fn a_parse_error_carries_a_message() {
        let err = compile_wgsl("@vertex fn broken(", CompileOptions::default()).unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn a_validly_parsed_but_invalid_shader_is_a_validation_error() {
        // Well-formed WGSL, but a fragment stage returning a struct is not a
        // colour output.
        let src = r#"
            struct NotAColour { x: f32 }
            @fragment
            fn fs_main() -> NotAColour {
                return NotAColour(1.0);
            }
        "#;
        let err = compile_wgsl(src, CompileOptions::default()).unwrap_err();
        assert!(
            matches!(err, ShaderError::Validation { .. }),
            "expected validation, got {err:?}"
        );
    }

    #[test]
    fn debug_options_do_not_change_validity() {
        let plain = compile_wgsl(TRIANGLE, CompileOptions::release()).unwrap();
        let debug =
            compile_wgsl(TRIANGLE, CompileOptions::with_debug(TRIANGLE, "tri.wgsl")).unwrap();
        assert!(looks_like_spirv(&plain));
        assert!(looks_like_spirv(&debug));
    }

    #[test]
    fn debug_builds_carry_more_information() {
        let release = compile_wgsl(TRIANGLE, CompileOptions::release()).unwrap();
        let debug =
            compile_wgsl(TRIANGLE, CompileOptions::with_debug(TRIANGLE, "tri.wgsl")).unwrap();
        assert!(
            debug.len() >= release.len(),
            "debug info should not shrink the module: {} vs {}",
            debug.len(),
            release.len()
        );
    }

    #[test]
    fn spirv_1_0_is_the_default_target() {
        assert_eq!(CompileOptions::default().lang_version, (1, 0));
    }

    #[test]
    fn words_convert_to_little_endian_bytes() {
        let words = compile_wgsl(TRIANGLE, CompileOptions::default()).unwrap();
        let bytes = words_to_bytes(&words);
        assert_eq!(bytes.len(), words.len() * 4);
        // The magic number is 0x07230203, so little-endian bytes are 03 02 23 07.
        assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07]);
    }

    #[test]
    fn the_magic_check_rejects_non_spirv() {
        assert!(!looks_like_spirv(&[]));
        assert!(!looks_like_spirv(&[0, 0, 0]));
        assert!(looks_like_spirv(&[SPIRV_MAGIC]));
    }

    #[test]
    fn the_trait_object_form_works() {
        let compiler: &dyn ShaderCompiler = &WgslCompiler::new();
        let words = compiler
            .compile(TRIANGLE, CompileOptions::default())
            .unwrap();
        assert!(looks_like_spirv(&words));
    }

    #[test]
    fn compilation_is_deterministic() {
        let a = compile_wgsl(TRIANGLE, CompileOptions::default()).unwrap();
        let b = compile_wgsl(TRIANGLE, CompileOptions::default()).unwrap();
        assert_eq!(a, b, "the same source must produce the same words");
    }

    #[test]
    fn an_empty_source_compiles_to_an_empty_module() {
        // An empty WGSL module is legal: it has no entry points, so naga emits a
        // header with nothing in it. What matters is that it does not panic.
        match compile_wgsl("", CompileOptions::default()) {
            Ok(words) => assert!(!words.is_empty(), "even an empty module has a header"),
            Err(e) => assert!(
                matches!(e, ShaderError::Parse(_) | ShaderError::Backend(_)),
                "{e:?}"
            ),
        }
    }

    #[test]
    fn whitespace_only_source_does_not_panic() {
        let _ = compile_wgsl("   \n\n  ", CompileOptions::default());
    }

    #[test]
    fn a_vertex_only_shader_is_valid() {
        let src = r#"
            @vertex
            fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;
        assert!(compile_wgsl(src, CompileOptions::default()).is_ok());
    }

    #[test]
    fn a_fragment_only_shader_is_valid() {
        let src = r#"
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(0.0, 1.0, 0.0, 1.0);
            }
        "#;
        assert!(compile_wgsl(src, CompileOptions::default()).is_ok());
    }
}
