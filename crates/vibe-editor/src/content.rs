//! The content browser: a directory tree of the project's assets.
//!
//! The browser lists what is on disk and classifies it by extension, because the
//! interesting question is not "what files are here" but "which of these can I
//! drop into the scene". Classification is by extension rather than by reading
//! each file: opening a 40 MB glTF to find out it is a glTF is a second of work
//! per asset and buys nothing.
//!
//! The listing is cached. Rescanning a large project on every frame turns the
//! browser into the editor's frame budget, and nothing under the cursor has
//! changed since the last scan in any normal use.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What kind of asset a file is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContentKind {
    /// A 3D model: glTF or GLB.
    Model,
    /// A texture: png, jpg, tga, bmp, ktx.
    Texture,
    /// A font: ttf or otf.
    Font,
    /// A scene: yaml.
    Scene,
    /// A shader: wgsl.
    Shader,
    /// An audio file: wav, ogg, mp3, flac.
    Audio,
    /// A file the engine has no importer for.
    Unknown,
}

impl ContentKind {
    /// The extension this kind is recognised by, for a filter box.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            ContentKind::Model => &["gltf", "glb"],
            ContentKind::Texture => &["png", "jpg", "jpeg", "tga", "bmp", "ktx"],
            ContentKind::Font => &["ttf", "otf"],
            ContentKind::Scene => &["yaml", "yml"],
            ContentKind::Shader => &["wgsl"],
            ContentKind::Audio => &["wav", "ogg", "mp3", "flac"],
            ContentKind::Unknown => &[],
        }
    }

    /// The label the browser groups by.
    pub fn label(self) -> &'static str {
        match self {
            ContentKind::Model => "Models",
            ContentKind::Texture => "Textures",
            ContentKind::Font => "Fonts",
            ContentKind::Scene => "Scenes",
            ContentKind::Shader => "Shaders",
            ContentKind::Audio => "Audio",
            ContentKind::Unknown => "Other",
        }
    }

    /// Classify a path by its extension, case-insensitively.
    pub fn of(path: &Path) -> ContentKind {
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            return ContentKind::Unknown;
        };
        let ext = ext.to_ascii_lowercase();
        // An extension with no name, as in a dotfile like `.gitignore`, is not
        // an asset: `Path::extension` returns None for those, so reaching here
        // means there really was a stem.
        ContentKind::ALL
            .iter()
            .copied()
            .find(|k| k.extensions().contains(&ext.as_str()))
            .unwrap_or(ContentKind::Unknown)
    }

    /// Every recognised kind, in a stable order.
    pub const ALL: [ContentKind; 7] = [
        ContentKind::Model,
        ContentKind::Texture,
        ContentKind::Font,
        ContentKind::Scene,
        ContentKind::Shader,
        ContentKind::Audio,
        ContentKind::Unknown,
    ];

    /// True when the engine can import this kind.
    pub fn is_importable(self) -> bool {
        self != ContentKind::Unknown
    }
}

/// One row of the browser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentEntry {
    /// The file's name, without directories.
    pub name: String,
    /// The file's full path.
    pub path: PathBuf,
    /// What kind of asset it is.
    pub kind: ContentKind,
    /// The file's size in bytes, or `None` when it could not be read.
    pub size: Option<u64>,
    /// True when the entry is a directory rather than a file.
    pub is_directory: bool,
}

impl ContentEntry {
    /// A human-readable size, or an empty string when unknown.
    pub fn size_display(&self) -> String {
        let Some(bytes) = self.size else {
            return String::new();
        };
        const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
        let mut value = bytes as f64;
        let mut unit = 0;
        while value >= 1024.0 && unit + 1 < UNITS.len() {
            value /= 1024.0;
            unit += 1;
        }
        if unit == 0 {
            format!("{bytes} B")
        } else {
            format!("{value:.1} {}", UNITS[unit])
        }
    }
}

/// The content browser's state.
pub struct ContentBrowser {
    /// The directory being listed.
    pub root: PathBuf,
    /// The entries, sorted by name.
    pub entries: Vec<ContentEntry>,
    /// A filter applied to names and kinds; empty shows everything.
    pub filter: String,
    /// The kind filter, or `None` for every kind.
    pub kind_filter: Option<ContentKind>,
    /// The entry the user has clicked.
    pub selected: Option<PathBuf>,
    /// How many times the directory has been listed.
    pub scan_count: u64,
    /// How many directories could not be read.
    pub unreadable: usize,
    /// How deep to descend.
    pub max_depth: usize,
}

impl Default for ContentBrowser {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ContentBrowser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentBrowser")
            .field("root", &self.root)
            .field("entries", &self.entries.len())
            .field("scan_count", &self.scan_count)
            .finish()
    }
}

impl ContentBrowser {
    /// A browser with no directory open.
    pub fn new() -> ContentBrowser {
        ContentBrowser {
            root: PathBuf::new(),
            entries: Vec::new(),
            filter: String::new(),
            kind_filter: None,
            selected: None,
            scan_count: 0,
            unreadable: 0,
            max_depth: 4,
        }
    }

    /// A browser rooted at a directory.
    pub fn at(root: impl Into<PathBuf>) -> ContentBrowser {
        let mut b = ContentBrowser::new();
        b.root = root.into();
        b
    }

    /// Re-list the directory.
    ///
    /// Returns how many entries were found. A directory that cannot be read is
    /// counted in `unreadable` rather than returning an error, because a
    /// project with one unreadable subdirectory should still show everything
    /// else rather than showing an error page.
    pub fn rescan(&mut self) -> usize {
        self.entries.clear();
        self.unreadable = 0;
        self.scan_count += 1;
        if self.root.as_os_str().is_empty() {
            return 0;
        }
        self.walk(&self.root.clone(), 0, &mut Vec::new());
        // Sorted by name so the listing does not shuffle between scans, which is
        // filesystem order and is not stable.
        self.entries.sort_by(|a, b| a.name.cmp(&b.name));
        self.entries.len()
    }

    /// Walk one directory level.
    fn walk(&mut self, dir: &Path, depth: usize, seen: &mut Vec<PathBuf>) {
        if depth > self.max_depth {
            return;
        }
        // A symlink loop makes the same directory reachable forever. The path is
        // compared before the contents are read, so a loop is cut rather than
        // filling the disk.
        let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        if seen.contains(&canonical) {
            self.unreadable += 1;
            return;
        }
        seen.push(canonical);

        let Ok(read) = std::fs::read_dir(dir) else {
            self.unreadable += 1;
            return;
        };
        let mut subdirectories = Vec::new();
        for item in read.flatten() {
            let path = item.path();
            let is_directory = path.is_dir();
            if is_directory {
                subdirectories.push(path.clone());
            }
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            self.entries.push(ContentEntry {
                name,
                kind: ContentKind::of(&path),
                size: (!is_directory)
                    .then(|| std::fs::metadata(&path).ok().map(|m| m.len()))
                    .flatten(),
                path,
                is_directory,
            });
        }
        for sub in subdirectories {
            self.walk(&sub, depth + 1, seen);
        }
    }

    /// The entries that pass the current filter.
    pub fn visible(&self) -> Vec<&ContentEntry> {
        self.entries.iter().filter(|e| self.matches(e)).collect()
    }

    /// True when an entry passes the name and kind filters.
    pub fn matches(&self, entry: &ContentEntry) -> bool {
        if let Some(kind) = self.kind_filter
            && entry.kind != kind
        {
            return false;
        }
        if self.filter.is_empty() {
            return true;
        }
        entry
            .name
            .to_lowercase()
            .contains(&self.filter.to_lowercase())
    }

    /// How many entries pass the filter.
    pub fn visible_count(&self) -> usize {
        self.visible().len()
    }

    /// The entries grouped by kind, in the kind order.
    pub fn by_kind(&self) -> BTreeMap<ContentKind, Vec<&ContentEntry>> {
        let mut out: BTreeMap<ContentKind, Vec<&ContentEntry>> = BTreeMap::new();
        for kind in ContentKind::ALL {
            out.insert(kind, Vec::new());
        }
        for entry in self.visible() {
            out.entry(entry.kind).or_default().push(entry);
        }
        out
    }

    /// Select an entry by path.
    pub fn select(&mut self, path: PathBuf) {
        self.selected = Some(path);
    }

    /// Clear the selection.
    pub fn clear_selection(&mut self) {
        self.selected = None;
    }

    /// The selected entry, if it is still listed.
    pub fn selected_entry(&self) -> Option<&ContentEntry> {
        let path = self.selected.as_ref()?;
        self.entries.iter().find(|e| &e.path == path)
    }

    /// Count the importable entries.
    pub fn importable_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.kind.is_importable() && !e.is_directory)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_glb_is_a_model() {
        assert_eq!(ContentKind::of(Path::new("a.glb")), ContentKind::Model);
        assert_eq!(ContentKind::of(Path::new("a.gltf")), ContentKind::Model);
    }

    #[test]
    fn a_png_is_a_texture() {
        assert_eq!(ContentKind::of(Path::new("a.png")), ContentKind::Texture);
        assert_eq!(ContentKind::of(Path::new("a.jpg")), ContentKind::Texture);
    }

    #[test]
    fn a_ttf_is_a_font() {
        assert_eq!(ContentKind::of(Path::new("a.ttf")), ContentKind::Font);
        assert_eq!(ContentKind::of(Path::new("a.otf")), ContentKind::Font);
    }

    #[test]
    fn a_wgsl_is_a_shader() {
        assert_eq!(ContentKind::of(Path::new("a.wgsl")), ContentKind::Shader);
    }

    #[test]
    fn a_yaml_is_a_scene() {
        assert_eq!(ContentKind::of(Path::new("a.yaml")), ContentKind::Scene);
    }

    #[test]
    fn classification_ignores_case() {
        assert_eq!(ContentKind::of(Path::new("A.PNG")), ContentKind::Texture);
        assert_eq!(ContentKind::of(Path::new("A.GLB")), ContentKind::Model);
    }

    #[test]
    fn an_unknown_extension_is_unknown() {
        assert_eq!(ContentKind::of(Path::new("a.exe")), ContentKind::Unknown);
    }

    #[test]
    fn a_file_with_no_extension_is_unknown() {
        assert_eq!(ContentKind::of(Path::new("README")), ContentKind::Unknown);
    }

    #[test]
    fn a_dotfile_is_unknown() {
        assert_eq!(
            ContentKind::of(Path::new(".gitignore")),
            ContentKind::Unknown
        );
    }

    #[test]
    fn kinds_are_importable_except_unknown() {
        for kind in ContentKind::ALL {
            assert_eq!(
                kind.is_importable(),
                kind != ContentKind::Unknown,
                "{kind:?}"
            );
        }
    }

    #[test]
    fn every_kind_has_a_distinct_label() {
        let labels: std::collections::HashSet<_> =
            ContentKind::ALL.iter().map(|k| k.label()).collect();
        assert_eq!(labels.len(), ContentKind::ALL.len());
    }

    #[test]
    fn an_empty_browser_has_no_entries() {
        let b = ContentBrowser::new();
        assert!(b.entries.is_empty());
        assert_eq!(b.visible_count(), 0);
    }

    #[test]
    fn rescanning_an_empty_root_is_zero() {
        let mut b = ContentBrowser::new();
        assert_eq!(b.rescan(), 0);
    }

    #[test]
    fn a_missing_directory_counts_as_unreadable() {
        let mut b = ContentBrowser::at("/nonexistent-directory-for-tests");
        assert_eq!(b.rescan(), 0);
        assert!(b.unreadable > 0, "a missing directory is reported");
    }

    #[test]
    fn rescanning_counts() {
        let mut b = ContentBrowser::new();
        b.rescan();
        b.rescan();
        assert_eq!(b.scan_count, 2);
    }

    #[test]
    fn an_empty_filter_passes_everything() {
        let b = ContentBrowser::new();
        let entry = ContentEntry {
            name: "a.png".to_string(),
            path: PathBuf::from("a.png"),
            kind: ContentKind::Texture,
            size: None,
            is_directory: false,
        };
        assert!(b.matches(&entry));
    }

    #[test]
    fn a_name_filter_matches_case_insensitively() {
        let mut b = ContentBrowser::new();
        b.filter = "HERO".to_string();
        let entry = ContentEntry {
            name: "hero.png".to_string(),
            path: PathBuf::from("hero.png"),
            kind: ContentKind::Texture,
            size: None,
            is_directory: false,
        };
        assert!(b.matches(&entry));
    }

    #[test]
    fn a_kind_filter_excludes_other_kinds() {
        let mut b = ContentBrowser::new();
        b.kind_filter = Some(ContentKind::Model);
        let texture = ContentEntry {
            name: "a.png".to_string(),
            path: PathBuf::from("a.png"),
            kind: ContentKind::Texture,
            size: None,
            is_directory: false,
        };
        assert!(!b.matches(&texture));
    }

    #[test]
    fn both_filters_apply_together() {
        let mut b = ContentBrowser::new();
        b.filter = "hero".to_string();
        b.kind_filter = Some(ContentKind::Model);
        let model = ContentEntry {
            name: "hero.glb".to_string(),
            path: PathBuf::from("hero.glb"),
            kind: ContentKind::Model,
            size: None,
            is_directory: false,
        };
        let texture = ContentEntry {
            name: "hero.png".to_string(),
            path: PathBuf::from("hero.png"),
            kind: ContentKind::Texture,
            size: None,
            is_directory: false,
        };
        let other = ContentEntry {
            name: "villain.glb".to_string(),
            path: PathBuf::from("villain.glb"),
            kind: ContentKind::Model,
            size: None,
            is_directory: false,
        };
        assert!(b.matches(&model));
        assert!(!b.matches(&texture), "the kind filter excludes it");
        assert!(!b.matches(&other), "the name filter excludes it");
    }

    #[test]
    fn selecting_records_the_path() {
        let mut b = ContentBrowser::new();
        b.select(PathBuf::from("a.png"));
        assert_eq!(b.selected, Some(PathBuf::from("a.png")));
    }

    #[test]
    fn clearing_removes_the_selection() {
        let mut b = ContentBrowser::new();
        b.select(PathBuf::from("a.png"));
        b.clear_selection();
        assert!(b.selected.is_none());
    }

    #[test]
    fn a_selected_entry_that_is_not_listed_is_none() {
        let b = ContentBrowser::at(".");
        let mut b2 = b;
        b2.select(PathBuf::from("/not/here.png"));
        assert!(b2.selected_entry().is_none());
    }

    #[test]
    fn byte_sizes_are_shown_plainly() {
        let e = ContentEntry {
            name: "a".to_string(),
            path: PathBuf::from("a"),
            kind: ContentKind::Unknown,
            size: Some(512),
            is_directory: false,
        };
        assert_eq!(e.size_display(), "512 B");
    }

    #[test]
    fn larger_sizes_are_shown_in_units() {
        let e = ContentEntry {
            name: "a".to_string(),
            path: PathBuf::from("a"),
            kind: ContentKind::Model,
            size: Some(2048),
            is_directory: false,
        };
        assert_eq!(e.size_display(), "2.0 KiB");
    }

    #[test]
    fn a_megabyte_scale_size_does_not_overflow_the_units() {
        let e = ContentEntry {
            name: "a".to_string(),
            path: PathBuf::from("a"),
            kind: ContentKind::Model,
            size: Some(u64::MAX),
            is_directory: false,
        };
        let s = e.size_display();
        assert!(!s.is_empty());
        assert!(s.contains("GiB") || s.contains("MiB"), "{s}");
    }

    #[test]
    fn an_unknown_size_shows_nothing() {
        let e = ContentEntry {
            name: "a".to_string(),
            path: PathBuf::from("a"),
            kind: ContentKind::Unknown,
            size: None,
            is_directory: false,
        };
        assert_eq!(e.size_display(), "");
    }

    #[test]
    fn grouping_covers_every_kind() {
        let mut b = ContentBrowser::new();
        b.entries.push(ContentEntry {
            name: "a.glb".to_string(),
            path: PathBuf::from("a.glb"),
            kind: ContentKind::Model,
            size: None,
            is_directory: false,
        });
        let grouped = b.by_kind();
        assert_eq!(grouped.len(), ContentKind::ALL.len());
        assert_eq!(grouped[&ContentKind::Model].len(), 1);
    }

    #[test]
    fn a_directory_is_not_an_importable_asset() {
        let mut b = ContentBrowser::new();
        b.entries.push(ContentEntry {
            name: "textures".to_string(),
            path: PathBuf::from("textures"),
            kind: ContentKind::Unknown,
            size: None,
            is_directory: true,
        });
        assert_eq!(b.importable_count(), 0);
    }

    #[test]
    fn an_importable_file_is_counted() {
        let mut b = ContentBrowser::new();
        b.entries.push(ContentEntry {
            name: "a.glb".to_string(),
            path: PathBuf::from("a.glb"),
            kind: ContentKind::Model,
            size: None,
            is_directory: false,
        });
        assert_eq!(b.importable_count(), 1);
    }

    #[test]
    fn a_real_directory_is_listed() {
        let dir = std::env::temp_dir().join("vibe-content-test");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("model.glb"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        let mut b = ContentBrowser::at(&dir);
        let n = b.rescan();
        assert_eq!(n, 2, "both files are listed: {b:?}");
        assert_eq!(b.importable_count(), 1, "only the glb is importable");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_subdirectory_is_listed_too() {
        let dir = std::env::temp_dir().join("vibe-content-nested");
        let sub = dir.join("textures");
        let _ = std::fs::create_dir_all(&sub);
        std::fs::write(sub.join("hero.png"), b"x").unwrap();
        let mut b = ContentBrowser::at(&dir);
        b.rescan();
        assert_eq!(b.importable_count(), 1, "the nested png is found: {b:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_scan_is_sorted_by_name() {
        let dir = std::env::temp_dir().join("vibe-content-sorted");
        let _ = std::fs::create_dir_all(&dir);
        for name in ["z.glb", "a.glb", "m.glb"] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        let mut b = ContentBrowser::at(&dir);
        b.rescan();
        let names: Vec<&str> = b.entries.iter().map(|e| e.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "the listing must not shuffle");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
