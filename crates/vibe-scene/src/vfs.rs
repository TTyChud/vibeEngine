//! A virtual filesystem rooted at the executable.
//!
//! Game content is addressed by paths like `assets/sprites/hero.png` regardless
//! of where the binary sits, which is what lets a built game run from a
//! different directory than it was developed in. Paths are resolved against the
//! executable's directory first, then an explicit mount point.

use std::path::{Path, PathBuf};

/// An error from a virtual path operation.
#[derive(Debug, thiserror::Error)]
pub enum VfsError {
    /// A path escaped the root, via `..` or an absolute path.
    #[error("path {0} escapes the virtual filesystem root")]
    Escapes(String),
    /// The path was empty or otherwise unusable.
    #[error("invalid virtual path {0:?}")]
    Invalid(String),
    /// The file was not found under any mount.
    #[error("virtual file {0} not found (searched {1})")]
    NotFound(String, String),
    /// A path was not valid UTF-8.
    #[error("virtual path is not valid utf-8")]
    NotUtf8,
}

/// A read-only virtual filesystem for game content.
#[derive(Debug, Clone)]
pub struct Vfs {
    root: PathBuf,
}

impl Vfs {
    /// A filesystem rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Vfs {
        Vfs { root: root.into() }
    }

    /// A filesystem rooted at the directory containing the running executable.
    ///
    /// Falls back to the current directory when the executable path cannot be
    /// read, which is what a `cargo test` binary needs.
    pub fn from_executable() -> Vfs {
        let root = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."));
        Vfs { root }
    }

    /// The root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Normalise a virtual path, rejecting anything that escapes the root.
    ///
    /// Rejects `..` outright rather than resolving it, because a virtual path
    /// that needs `..` is a bug in the content, not a legitimate reference.
    pub fn normalize(&self, path: &str) -> Result<PathBuf, VfsError> {
        if path.is_empty() {
            return Err(VfsError::Invalid(path.to_string()));
        }
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            return Err(VfsError::Escapes(path.to_string()));
        }
        let mut out = PathBuf::new();
        for component in candidate.components() {
            match component {
                std::path::Component::Normal(part) => out.push(part),
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                    return Err(VfsError::Escapes(path.to_string()));
                }
                _ => return Err(VfsError::Escapes(path.to_string())),
            }
        }
        if out.as_os_str().is_empty() {
            return Err(VfsError::Invalid(path.to_string()));
        }
        Ok(out)
    }

    /// Resolve a virtual path to a real one under the root.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, VfsError> {
        Ok(self.root.join(self.normalize(path)?))
    }

    /// Read a virtual file as bytes.
    pub fn read(&self, path: &str) -> Result<Vec<u8>, VfsError> {
        let real = self.resolve(path)?;
        std::fs::read(&real)
            .map_err(|_| VfsError::NotFound(path.to_string(), self.root.display().to_string()))
    }

    /// Read a virtual file as UTF-8 text.
    pub fn read_to_string(&self, path: &str) -> Result<String, VfsError> {
        let bytes = self.read(path)?;
        String::from_utf8(bytes).map_err(|_| VfsError::NotUtf8)
    }

    /// True when the virtual file exists.
    pub fn exists(&self, path: &str) -> bool {
        self.resolve(path).map(|p| p.is_file()).unwrap_or(false)
    }

    /// Every file under a virtual directory, relative and sorted.
    ///
    /// The sort makes scene load order deterministic, which matters because
    /// entity UUIDs are assigned in file order.
    pub fn list_dir(&self, path: &str) -> Result<Vec<String>, VfsError> {
        let real = self.resolve(path)?;
        let mut out = Vec::new();
        collect(&real, &self.root, &mut out);
        out.sort();
        Ok(out)
    }
}

fn collect(dir: &Path, root: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, root, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            if let Some(s) = rel.to_str() {
                out.push(s.replace('\\', "/"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Removes its directory on drop, so a failed assertion does not leave
    /// temp files behind.
    struct TempDir(PathBuf);

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A private temp directory per call, so tests running in parallel cannot
    /// delete each other's files.
    fn temp_vfs() -> (Vfs, TempDir) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        // pid alone collides when the counter restarts, so mix in the clock.
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut base = std::env::temp_dir();
        base.push("vibe-engine-tests");
        let _ = std::fs::create_dir_all(&base);
        let mut dir = base.clone();
        dir.push(format!("vfs-{}-{n}-{stamp}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("assets/sub")).unwrap();
        std::fs::write(dir.join("assets/hero.txt"), "hello").unwrap();
        std::fs::write(dir.join("assets/sub/a.txt"), "a").unwrap();
        std::fs::write(dir.join("assets/b.txt"), "b").unwrap();
        (Vfs::new(dir.clone()), TempDir(dir))
    }

    #[test]
    fn normalize_accepts_a_plain_path() {
        let (vfs, _dir) = temp_vfs();
        assert_eq!(
            vfs.normalize("assets/hero.txt").unwrap(),
            PathBuf::from("assets/hero.txt")
        );
    }

    #[test]
    fn normalize_strips_current_dir() {
        let (vfs, _dir) = temp_vfs();
        assert_eq!(
            vfs.normalize("./assets/./hero.txt").unwrap(),
            PathBuf::from("assets/hero.txt")
        );
    }

    #[test]
    fn normalize_rejects_parent_dir() {
        let (vfs, _dir) = temp_vfs();
        assert!(matches!(
            vfs.normalize("../secrets"),
            Err(VfsError::Escapes(_))
        ));
        assert!(matches!(
            vfs.normalize("assets/../../etc/passwd"),
            Err(VfsError::Escapes(_))
        ));
    }

    #[test]
    fn normalize_rejects_absolute_paths() {
        let (vfs, _dir) = temp_vfs();
        assert!(matches!(
            vfs.normalize("/etc/passwd"),
            Err(VfsError::Escapes(_))
        ));
    }

    #[test]
    fn normalize_rejects_empty_path() {
        let (vfs, _dir) = temp_vfs();
        assert!(matches!(vfs.normalize(""), Err(VfsError::Invalid(_))));
    }

    #[test]
    fn resolve_stays_under_the_root() {
        let (vfs, dir) = temp_vfs();
        let resolved = vfs.resolve("assets/hero.txt").unwrap();
        assert!(resolved.starts_with(&dir.0));
    }

    #[test]
    fn read_returns_contents() {
        let (vfs, _dir) = temp_vfs();
        assert_eq!(vfs.read_to_string("assets/hero.txt").unwrap(), "hello");
    }

    #[test]
    fn read_missing_file_reports_not_found_with_root() {
        let (vfs, dir) = temp_vfs();
        let err = vfs.read("assets/nope.txt").unwrap_err();
        assert!(matches!(err, VfsError::NotFound(..)));
        assert!(err.to_string().contains(&dir.0.display().to_string()));
    }

    #[test]
    fn exists_reflects_the_filesystem() {
        let (vfs, _dir) = temp_vfs();
        assert!(vfs.exists("assets/hero.txt"));
        assert!(!vfs.exists("assets/nope.txt"));
        assert!(!vfs.exists("../escape"));
    }

    #[test]
    fn list_dir_is_recursive_relative_and_sorted() {
        let (vfs, _dir) = temp_vfs();
        let files = vfs.list_dir("assets").unwrap();
        assert_eq!(
            files,
            vec!["assets/b.txt", "assets/hero.txt", "assets/sub/a.txt"]
        );
    }

    #[test]
    fn list_dir_on_missing_directory_is_empty_not_an_error() {
        let (vfs, _dir) = temp_vfs();
        assert!(vfs.list_dir("nope").unwrap().is_empty());
    }

    #[test]
    fn from_executable_gives_a_usable_root() {
        let vfs = Vfs::from_executable();
        assert!(!vfs.root().as_os_str().is_empty());
        assert!(vfs.resolve("a/b.txt").is_ok());
    }

    #[test]
    fn root_is_reported() {
        let (vfs, dir) = temp_vfs();
        assert_eq!(vfs.root(), dir.0.as_path());
    }
}
