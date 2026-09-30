//! The editing session: which mode the editor is in, and what it is editing.
//!
//! Play/stop is a mode rather than a flag because the two states have genuinely
//! different rules: in edit mode the editor may move entities and rewrite
//! components, and in play mode the running game owns that. A boolean would let
//! a stray inspector drag silently corrupt a running simulation, so the mode
//! gates the edit operations instead.

use std::path::{Path, PathBuf};

use vibe_scene::SceneError;

/// Whether the editor is editing or running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EditorMode {
    /// The scene can be inspected and changed; nothing is simulating.
    #[default]
    Edit,
    /// The game is running. The editor may look but not change the scene.
    Play,
    /// The game is running but paused at a breakpoint.
    Pause,
}

impl EditorMode {
    /// True when the simulation is advancing.
    pub fn is_running(self) -> bool {
        matches!(self, EditorMode::Play)
    }

    /// True when the editor may change the scene.
    ///
    /// A paused game is still a running game: the point of a pause is to inspect
    /// and step, so letting the editor rewrite the scene under it would make the
    /// paused state a lie.
    pub fn allows_editing(self) -> bool {
        matches!(self, EditorMode::Edit)
    }

    /// The label the toolbar shows.
    pub fn label(self) -> &'static str {
        match self {
            EditorMode::Edit => "Edit",
            EditorMode::Play => "Play",
            EditorMode::Pause => "Pause",
        }
    }
}

/// A file operation the user asked for.
#[derive(Debug, Clone, PartialEq)]
pub enum FileAction {
    /// Start editing a new, empty scene.
    New,
    /// Open an existing scene.
    Open(PathBuf),
    /// Save to the current path.
    Save,
    /// Save to a new path and adopt it.
    SaveAs(PathBuf),
    /// Close the scene.
    Close,
}

/// What went wrong touching a scene file.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// The scene file could not be read or written.
    #[error("scene file: {0}")]
    Scene(#[from] SceneError),
    /// The file has no parent directory, so a relative name cannot be resolved.
    #[error("path {0:?} has no parent directory")]
    NoParent(PathBuf),
    /// Something was asked to save before anything was opened.
    #[error("there is no scene to save yet; use Save As first")]
    NothingToSave,
    /// The scene has changes and the caller did not say to discard them.
    #[error("the scene {path:?} has unsaved changes")]
    Unsaved {
        /// The scene that would be closed.
        path: PathBuf,
    },
}

/// The editor's scene state.
pub struct EditorSession {
    /// The current mode.
    pub mode: EditorMode,
    /// The open scene's path, once it has one.
    pub path: Option<PathBuf>,
    /// True when the scene has changes not yet written.
    pub dirty: bool,
    /// The entity the inspector is showing.
    pub selection: Option<vibe_ecs::Entity>,
    /// The number of times play has been entered, for the status bar.
    pub play_count: u64,
}

impl Default for EditorSession {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for EditorSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditorSession")
            .field("mode", &self.mode)
            .field("path", &self.path)
            .field("dirty", &self.dirty)
            .field("selection", &self.selection)
            .finish()
    }
}

impl EditorSession {
    /// A session with no scene open.
    pub fn new() -> EditorSession {
        EditorSession {
            mode: EditorMode::Edit,
            path: None,
            dirty: false,
            selection: None,
            play_count: 0,
        }
    }

    /// True when a scene is open.
    pub fn is_open(&self) -> bool {
        self.path.is_some()
    }

    /// Enter play mode.
    ///
    /// Doing this while already playing is a no-op rather than an error, because
    /// the toolbar's play button is a toggle and a double click should not
    /// produce a message.
    pub fn play(&mut self) {
        if self.mode != EditorMode::Play {
            self.mode = EditorMode::Play;
            self.play_count += 1;
        }
    }

    /// Pause a running game.
    pub fn pause(&mut self) {
        if self.mode == EditorMode::Play {
            self.mode = EditorMode::Pause;
        }
    }

    /// Return to editing, discarding the running state.
    pub fn stop(&mut self) {
        if self.mode != EditorMode::Edit {
            self.mode = EditorMode::Edit;
            self.selection = None;
        }
    }

    /// Toggle between playing and editing.
    pub fn toggle_play(&mut self) {
        if self.mode == EditorMode::Edit {
            self.play();
        } else {
            self.stop();
        }
    }

    /// Record that the scene has changed.
    ///
    /// Does nothing while a game is running: the simulation changing a scene is
    /// not the same as the user editing it, and marking the file dirty from a
    /// running game would prompt to save state the developer never made.
    pub fn mark_dirty(&mut self) {
        if self.mode.allows_editing() {
            self.dirty = true;
        }
    }

    /// Record that the scene has been written.
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Adopt a path, as after a successful save.
    pub fn set_path(&mut self, path: PathBuf) {
        self.path = Some(path);
        self.mark_clean();
    }

    /// Close the scene, refusing when there are unsaved changes unless
    /// `discard` says otherwise.
    pub fn close(&mut self, discard: bool) -> Result<(), SessionError> {
        if self.dirty && !discard {
            return Err(SessionError::Unsaved {
                path: self
                    .path
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("<unsaved>")),
            });
        }
        self.path = None;
        self.dirty = false;
        self.selection = None;
        self.stop();
        Ok(())
    }

    /// The scene's name for a window title.
    pub fn title(&self) -> String {
        let name = self
            .path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "untitled".to_string());
        let star = if self.dirty { "*" } else { "" };
        format!("{name}{star} — vibeEngine")
    }

    /// Resolve a possibly-relative path against the open scene's directory.
    ///
    /// A relative asset path in a scene file is relative to that file, so
    /// resolving against the process's working directory instead silently
    /// misses every asset in a project opened from elsewhere.
    pub fn resolve(&self, relative: &Path) -> Result<PathBuf, SessionError> {
        if relative.is_absolute() {
            return Ok(relative.to_path_buf());
        }
        let base = self
            .path
            .as_ref()
            .and_then(|p| p.parent())
            .ok_or_else(|| SessionError::NoParent(relative.to_path_buf()))?;
        Ok(base.join(relative))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_session_is_editing() {
        let s = EditorSession::new();
        assert_eq!(s.mode, EditorMode::Edit);
        assert!(s.mode.allows_editing());
        assert!(!s.mode.is_running());
    }

    #[test]
    fn a_new_session_has_no_scene() {
        assert!(!EditorSession::new().is_open());
    }

    #[test]
    fn playing_switches_the_mode() {
        let mut s = EditorSession::new();
        s.play();
        assert_eq!(s.mode, EditorMode::Play);
        assert!(s.mode.is_running());
        assert!(!s.mode.allows_editing());
    }

    #[test]
    fn a_paused_game_still_refuses_edits() {
        let mut s = EditorSession::new();
        s.play();
        s.pause();
        assert_eq!(s.mode, EditorMode::Pause);
        assert!(
            !s.mode.allows_editing(),
            "a pause is for inspecting, not for writing"
        );
        assert!(!s.mode.is_running());
    }

    #[test]
    fn stopping_returns_to_editing() {
        let mut s = EditorSession::new();
        s.play();
        s.stop();
        assert_eq!(s.mode, EditorMode::Edit);
    }

    #[test]
    fn toggling_play_flips_the_mode() {
        let mut s = EditorSession::new();
        s.toggle_play();
        assert_eq!(s.mode, EditorMode::Play);
        s.toggle_play();
        assert_eq!(s.mode, EditorMode::Edit);
    }

    #[test]
    fn playing_twice_counts_once() {
        let mut s = EditorSession::new();
        s.play();
        s.play();
        assert_eq!(s.play_count, 1, "a second click is not a new session");
    }

    #[test]
    fn an_edit_marks_the_scene_dirty() {
        let mut s = EditorSession::new();
        s.mark_dirty();
        assert!(s.dirty);
    }

    #[test]
    fn a_running_game_does_not_mark_the_scene_dirty() {
        let mut s = EditorSession::new();
        s.play();
        s.mark_dirty();
        assert!(!s.dirty, "a simulation is not a user edit");
    }

    #[test]
    fn saving_clears_the_dirty_flag() {
        let mut s = EditorSession::new();
        s.mark_dirty();
        s.set_path(PathBuf::from("/tmp/a.yaml"));
        assert!(!s.dirty);
        assert!(s.is_open());
    }

    #[test]
    fn a_dirty_scene_will_not_close_without_discard() {
        let mut s = EditorSession::new();
        s.set_path(PathBuf::from("/tmp/a.yaml"));
        s.mark_dirty();
        assert!(s.close(false).is_err());
        assert!(s.is_open(), "the scene stays open");
    }

    #[test]
    fn a_dirty_scene_closes_when_discarded() {
        let mut s = EditorSession::new();
        s.set_path(PathBuf::from("/tmp/a.yaml"));
        s.mark_dirty();
        s.close(true).unwrap();
        assert!(!s.is_open());
    }

    #[test]
    fn a_clean_scene_closes_without_discard() {
        let mut s = EditorSession::new();
        s.set_path(PathBuf::from("/tmp/a.yaml"));
        s.close(false).unwrap();
        assert!(!s.is_open());
    }

    #[test]
    fn closing_stops_a_running_game() {
        let mut s = EditorSession::new();
        s.set_path(PathBuf::from("/tmp/a.yaml"));
        s.play();
        s.close(false).unwrap();
        assert_eq!(s.mode, EditorMode::Edit);
    }

    #[test]
    fn closing_clears_the_selection() {
        let mut s = EditorSession::new();
        s.set_path(PathBuf::from("/tmp/a.yaml"));
        s.close(false).unwrap();
        assert!(s.selection.is_none());
    }

    #[test]
    fn a_dirty_title_carries_an_asterisk() {
        let mut s = EditorSession::new();
        s.set_path(PathBuf::from("/tmp/scene.yaml"));
        assert!(!s.title().starts_with('*'));
        s.mark_dirty();
        assert!(s.title().contains("*"), "{}", s.title());
    }

    #[test]
    fn an_unnamed_scene_says_untitled() {
        assert!(EditorSession::new().title().contains("untitled"));
    }

    #[test]
    fn a_relative_path_resolves_against_the_scenes_directory() {
        let mut s = EditorSession::new();
        s.set_path(PathBuf::from("/project/scenes/main.yaml"));
        let p = s.resolve(Path::new("textures/hero.png")).unwrap();
        assert_eq!(p, PathBuf::from("/project/scenes/textures/hero.png"));
    }

    #[test]
    fn an_absolute_path_is_left_alone() {
        let mut s = EditorSession::new();
        s.set_path(PathBuf::from("/project/main.yaml"));
        let p = s.resolve(Path::new("/elsewhere/hero.png")).unwrap();
        assert_eq!(p, PathBuf::from("/elsewhere/hero.png"));
    }

    #[test]
    fn resolving_with_no_scene_open_is_an_error() {
        let s = EditorSession::new();
        assert!(s.resolve(Path::new("hero.png")).is_err());
    }

    #[test]
    fn mode_labels_are_distinct() {
        let all = [EditorMode::Edit, EditorMode::Play, EditorMode::Pause];
        let labels: std::collections::HashSet<_> = all.iter().map(|m| m.label()).collect();
        assert_eq!(labels.len(), 3);
    }
}
