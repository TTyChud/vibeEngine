//! A cache of compiled shaders, so a pipeline is not rebuilt every frame.

use std::collections::HashMap;

use crate::compiler::{CompileOptions, ShaderCompiler, looks_like_spirv};
use crate::error::ShaderError;

/// Which pipeline stage a shader serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ShaderKind {
    /// The vertex stage.
    Vertex,
    /// The fragment stage.
    Fragment,
    /// The compute stage.
    Compute,
}

impl ShaderKind {
    /// The name used in logs and in shader file names.
    pub const fn name(self) -> &'static str {
        match self {
            ShaderKind::Vertex => "vertex",
            ShaderKind::Fragment => "fragment",
            ShaderKind::Compute => "compute",
        }
    }

    /// Parse a stage name.
    pub fn parse(name: &str) -> Option<ShaderKind> {
        match name {
            "vertex" | "vert" | "vs" => Some(ShaderKind::Vertex),
            "fragment" | "frag" | "fs" => Some(ShaderKind::Fragment),
            "compute" | "comp" | "cs" => Some(ShaderKind::Compute),
            _ => None,
        }
    }
}

/// A named set of compiled shaders, one per stage.
///
/// Compilation is the slow part of building a pipeline, so the library caches
/// by name *and* by the options it was compiled with: changing the debug flags
/// has to invalidate the entry rather than hand back a stale module.
#[derive(Debug, Default)]
pub struct ShaderLibrary {
    shaders: HashMap<(String, ShaderKind), Vec<u32>>,
    options: HashMap<(String, ShaderKind), CompileOptions>,
}

impl ShaderLibrary {
    /// An empty library.
    pub fn new() -> ShaderLibrary {
        ShaderLibrary::default()
    }

    /// Compile and cache a shader.
    ///
    /// Re-compiles only when the options differ from the cached ones, so
    /// calling this every frame is cheap.
    pub fn add<C: ShaderCompiler>(
        &mut self,
        compiler: &C,
        name: impl Into<String>,
        kind: ShaderKind,
        source: &str,
        options: CompileOptions,
    ) -> Result<(), ShaderError> {
        let name = name.into();
        let key = (name.clone(), kind);

        if self.options.get(&key) == Some(&options) && self.shaders.contains_key(&key) {
            return Ok(());
        }

        let words = compiler.compile(source, options.clone())?;
        if !looks_like_spirv(&words) {
            return Err(ShaderError::Backend(format!("{name:?} produced no spirv")));
        }
        self.shaders.insert(key.clone(), words);
        self.options.insert(key, options);
        Ok(())
    }

    /// The compiled words for a shader.
    pub fn get(&self, name: &str, kind: ShaderKind) -> Option<&[u32]> {
        self.shaders
            .get(&(name.to_string(), kind))
            .map(|v| v.as_slice())
    }

    /// True when a shader is cached.
    pub fn contains(&self, name: &str, kind: ShaderKind) -> bool {
        self.shaders.contains_key(&(name.to_string(), kind))
    }

    /// Drop a shader, e.g. after a hot reload.
    pub fn remove(&mut self, name: &str, kind: ShaderKind) -> bool {
        self.options.remove(&(name.to_string(), kind));
        self.shaders.remove(&(name.to_string(), kind)).is_some()
    }

    /// Number of cached shaders.
    pub fn len(&self) -> usize {
        self.shaders.len()
    }

    /// True when nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.shaders.is_empty()
    }

    /// Drop everything.
    pub fn clear(&mut self) {
        self.shaders.clear();
        self.options.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ShaderError;

    const SRC: &str = r#"
        @vertex
        fn vs_main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0);
        }
    "#;

    /// A compiler that counts calls, so caching can be observed.
    struct CountingCompiler {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ShaderCompiler for CountingCompiler {
        fn compile(&self, source: &str, _options: CompileOptions) -> Result<Vec<u32>, ShaderError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            if source.is_empty() {
                return Err(ShaderError::Parse("empty".to_string()));
            }
            Ok(vec![crate::compiler::SPIRV_MAGIC, 1, 2, 3])
        }
    }

    fn compiler() -> (
        CountingCompiler,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (
            CountingCompiler {
                calls: std::sync::Arc::clone(&calls),
            },
            calls,
        )
    }

    #[test]
    fn a_shader_is_cached_after_adding() {
        let (c, _) = compiler();
        let mut lib = ShaderLibrary::new();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        assert!(lib.contains("quad", ShaderKind::Vertex));
        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn adding_the_same_shader_twice_compiles_once() {
        let (c, calls) = compiler();
        let mut lib = ShaderLibrary::new();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::Acquire), 1);
        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn changing_options_recompiles() {
        let (c, calls) = compiler();
        let mut lib = ShaderLibrary::new();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::with_debug(SRC, "quad.wgsl"),
        )
        .unwrap();
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Acquire),
            2,
            "different options must invalidate the cache"
        );
    }

    #[test]
    fn the_same_name_in_two_stages_is_two_entries() {
        let (c, _) = compiler();
        let mut lib = ShaderLibrary::new();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        lib.add(
            &c,
            "quad",
            ShaderKind::Fragment,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        assert_eq!(lib.len(), 2);
        assert!(lib.contains("quad", ShaderKind::Vertex));
        assert!(lib.contains("quad", ShaderKind::Fragment));
    }

    #[test]
    fn get_returns_the_compiled_words() {
        let (c, _) = compiler();
        let mut lib = ShaderLibrary::new();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        let words = lib.get("quad", ShaderKind::Vertex).unwrap();
        assert_eq!(words[0], crate::compiler::SPIRV_MAGIC);
    }

    #[test]
    fn get_on_a_missing_shader_is_none() {
        let lib = ShaderLibrary::new();
        assert!(lib.get("nope", ShaderKind::Vertex).is_none());
    }

    #[test]
    fn the_wrong_stage_does_not_match() {
        let (c, _) = compiler();
        let mut lib = ShaderLibrary::new();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        assert!(lib.get("quad", ShaderKind::Fragment).is_none());
    }

    #[test]
    fn a_failed_compile_is_not_cached() {
        let (c, _) = compiler();
        let mut lib = ShaderLibrary::new();
        assert!(
            lib.add(&c, "bad", ShaderKind::Vertex, "", CompileOptions::default())
                .is_err()
        );
        assert!(
            !lib.contains("bad", ShaderKind::Vertex),
            "a failure must not poison the cache"
        );
    }

    #[test]
    fn output_without_the_magic_number_is_rejected() {
        struct BadCompiler;
        impl ShaderCompiler for BadCompiler {
            fn compile(&self, _: &str, _: CompileOptions) -> Result<Vec<u32>, ShaderError> {
                Ok(vec![0, 0, 0])
            }
        }
        let mut lib = ShaderLibrary::new();
        let err = lib.add(
            &BadCompiler,
            "bad",
            ShaderKind::Vertex,
            "",
            CompileOptions::default(),
        );
        assert!(matches!(err, Err(ShaderError::Backend(_))));
    }

    #[test]
    fn remove_drops_an_entry() {
        let (c, _) = compiler();
        let mut lib = ShaderLibrary::new();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        assert!(lib.remove("quad", ShaderKind::Vertex));
        assert!(!lib.contains("quad", ShaderKind::Vertex));
        assert!(
            !lib.remove("quad", ShaderKind::Vertex),
            "removing twice reports nothing removed"
        );
    }

    #[test]
    fn removing_one_stage_leaves_the_other() {
        let (c, _) = compiler();
        let mut lib = ShaderLibrary::new();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        lib.add(
            &c,
            "quad",
            ShaderKind::Fragment,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        lib.remove("quad", ShaderKind::Vertex);
        assert!(lib.contains("quad", ShaderKind::Fragment));
    }

    #[test]
    fn clear_empties_the_library() {
        let (c, _) = compiler();
        let mut lib = ShaderLibrary::new();
        lib.add(&c, "a", ShaderKind::Vertex, SRC, CompileOptions::default())
            .unwrap();
        lib.clear();
        assert!(lib.is_empty());
    }

    #[test]
    fn an_empty_library_reports_empty() {
        let lib = ShaderLibrary::new();
        assert!(lib.is_empty());
        assert_eq!(lib.len(), 0);
    }

    #[test]
    fn re_adding_after_removal_recompiles() {
        let (c, calls) = compiler();
        let mut lib = ShaderLibrary::new();
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        lib.remove("quad", ShaderKind::Vertex);
        lib.add(
            &c,
            "quad",
            ShaderKind::Vertex,
            SRC,
            CompileOptions::default(),
        )
        .unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::Acquire), 2);
    }

    #[test]
    fn stage_names_round_trip() {
        for kind in [
            ShaderKind::Vertex,
            ShaderKind::Fragment,
            ShaderKind::Compute,
        ] {
            assert_eq!(ShaderKind::parse(kind.name()), Some(kind), "{kind:?}");
        }
    }

    #[test]
    fn stage_aliases_parse() {
        assert_eq!(ShaderKind::parse("vs"), Some(ShaderKind::Vertex));
        assert_eq!(ShaderKind::parse("fs"), Some(ShaderKind::Fragment));
        assert_eq!(ShaderKind::parse("comp"), Some(ShaderKind::Compute));
    }

    #[test]
    fn an_unknown_stage_name_is_none() {
        assert_eq!(ShaderKind::parse("geometry"), None);
        assert_eq!(ShaderKind::parse(""), None);
    }

    #[test]
    fn stage_kinds_are_distinct() {
        assert_ne!(ShaderKind::Vertex, ShaderKind::Fragment);
        assert_ne!(ShaderKind::Fragment, ShaderKind::Compute);
    }
}
