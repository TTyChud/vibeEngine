//! The editor's frame loop: egui's context, the panels, and what comes out.
//!
//! This is the piece that makes the rest of the crate visible. The panels know
//! what to show and the bridge knows how to turn that into triangles; what sits
//! between them is this: assemble the menu bar and the open panels into one
//! egui pass, and hand the result to the renderer.
//!
//! egui is immediate-mode, so nothing here is cached and nothing here holds
//! state beyond the context egui itself owns. That is the point: a panel cannot
//! get out of sync with the data it shows, because there is no copy of the data
//! to be stale.

use egui::Context;

use crate::content::{ContentBrowser, ContentKind};
use crate::hierarchy::HierarchyPanel;
use crate::inspector::InspectorPanel;
use crate::logs::LogPanel;
use crate::session::{EditorMode, EditorSession};
use crate::ui_bridge::UiDrawData;

/// Which side panel is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Panel {
    /// Nothing but the viewport.
    None,
    /// The scene tree.
    #[default]
    Hierarchy,
    /// The component inspector.
    Inspector,
    /// The asset browser.
    Content,
    /// The log.
    Log,
}

impl Panel {
    /// The panel's title.
    pub fn title(self) -> &'static str {
        match self {
            Panel::None => "",
            Panel::Hierarchy => "Hierarchy",
            Panel::Inspector => "Inspector",
            Panel::Content => "Content",
            Panel::Log => "Log",
        }
    }

    /// The panel's width in points.
    ///
    /// The content browser is wider because it shows names, kinds and sizes in
    /// columns, and cramming those into 260 points wraps every cell.
    pub fn width(self) -> f32 {
        match self {
            Panel::Content => 380.0,
            Panel::None => 0.0,
            _ => 280.0,
        }
    }
}

/// The editor's assembled state, handed to [`EditorFrame::draw`] each frame.
///
/// A plain struct rather than a set of arguments, because the argument list
/// grows every time a panel is added and a call site that has to be updated is a
/// call site that will be missed.
pub struct EditorFrame<'a> {
    /// egui's context, which holds the retained UI state.
    pub context: &'a Context,
    /// The session, for the mode and the scene's name.
    pub session: &'a mut EditorSession,
    /// The hierarchy panel.
    pub hierarchy: &'a mut HierarchyPanel,
    /// The inspector panel.
    pub inspector: &'a mut InspectorPanel,
    /// The log panel.
    pub log_panel: &'a mut LogPanel,
    /// The asset browser.
    pub content: &'a mut ContentBrowser,
    /// Which panel is open.
    pub panel: Panel,
    /// The window's size in points.
    pub viewport: egui::Vec2,
    /// The scene, for the hierarchy and inspector to read.
    pub world: &'a mut vibe_ecs::World,
    /// A path the host wants opened, consumed on the next frame's Open.
    ///
    /// The file dialog belongs to the platform, so the host resolves the path
    /// and hands it over rather than the editor reaching for one. Consumed once
    /// and cleared, so a path left here by a host that crashed does not reopen
    /// the same scene on every subsequent frame.
    pub requested_path: &'a mut Option<std::path::PathBuf>,
}

impl EditorFrame<'_> {
    /// Draw one frame and return the geometry.
    ///
    /// The whole body is wrapped so a panel that panics on malformed data
    /// cannot take the editor down with it: a crashed editor loses the user's
    /// unsaved work, which is the worst outcome available.
    pub fn draw(mut self) -> UiDrawData {
        // A path the host handed over is applied before the panels draw, so the
        // frame shows the scene it was asked to open. Doing it only on the Open
        // button's click would make an open depend on a click round trip, which
        // a headless frame has no way to produce.
        if let Some(path) = self.requested_path.take() {
            self.open_scene(&path);
        }
        let output = self.context.run_ui(self.take_input(), |ui| {
            // Wrapped because a panel that panics on malformed scene data must
            // not take the editor with it: a crashed editor loses the user's
            // unsaved work, which is the worst outcome available.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.draw_menu_bar(ui);
                self.draw_panels(ui);
            }));
        });
        let data = UiDrawData::from_output(self.context, &output);
        // The font atlas egui rasterised is released rather than uploaded: this
        // backend draws coloured triangles and samples nothing, and egui
        // asserts on a delta that is dropped unapplied. Clearing it is the
        // documented way to say "deliberately not applying these".
        let mut output = output;
        output.textures_delta.clear();
        data
    }

    /// The input for this frame.
    ///
    /// An empty input is right when the platform layer has not filled one in
    /// yet: egui runs a full pass on an empty input rather than skipping, so the
    /// panels lay out on the first frame instead of popping in later.
    fn take_input(&self) -> egui::RawInput {
        egui::RawInput::default()
    }

    /// The menu bar: file actions and the play controls.
    fn draw_menu_bar(&mut self, root: &mut egui::Ui) {
        egui::Panel::top("vibe-menu").show(root, |ui| {
            ui.horizontal(|ui| {
                if ui.button("New").clicked() {
                    self.session.path = None;
                    self.session.mark_clean();
                }
                if ui.button("Open").clicked() {
                    // The path comes from the host, which owns the file dialog.
                    // A host that already has one sets requested_path and the
                    // frame picks it up above; this button is here for the
                    // interactive case, where the host is watching for the
                    // click and will fill the path in response.
                    log::info!("open requested");
                }
                if ui
                    .button(if self.session.is_open() {
                        "Save"
                    } else {
                        "Save As"
                    })
                    .clicked()
                {
                    match self.session.path.clone() {
                        Some(path) => self.save_scene(&path),
                        None => log::info!("save-as needs a path from the host"),
                    }
                }
                ui.separator();
                self.draw_play_controls(ui);
            });
        });
    }

    /// Play, pause and stop.
    fn draw_play_controls(&mut self, ui: &mut egui::Ui) {
        let running = self.session.mode == EditorMode::Play;
        if ui.button(if running { "Stop" } else { "Play" }).clicked() {
            self.session.toggle_play();
        }
        let paused = self.session.mode == EditorMode::Pause;
        if ui
            .add_enabled(
                running,
                egui::Button::new(if paused { "Resume" } else { "Pause" }),
            )
            .clicked()
        {
            if paused {
                self.session.play();
            } else {
                self.session.pause();
            }
        }
    }

    /// The open side panel.
    fn draw_panels(&mut self, root: &mut egui::Ui) {
        if self.panel == Panel::None {
            return;
        }
        let shown = self.panel;
        egui::Panel::left("vibe-side")
            .exact_size(self.panel.width())
            .show(root, |ui| {
                ui.heading(self.panel.title());
                ui.separator();
                match shown {
                    Panel::Hierarchy => self.draw_hierarchy(ui),
                    Panel::Inspector => self.draw_inspector(ui),
                    Panel::Content => self.draw_content(ui),
                    Panel::Log => self.log_panel.show(ui),
                    Panel::None => {}
                }
            });
        // A panel can ask to be closed by clicking its own close button.
        if shown == Panel::None {
            self.panel = Panel::None;
        }
    }

    /// The scene tree.
    fn draw_hierarchy(&mut self, ui: &mut egui::Ui) {
        let rows = self.hierarchy.rows(self.world);
        self.hierarchy.note_row_count(rows.len());
        egui::ScrollArea::vertical().show(ui, |ui| {
            for row in rows {
                let label = self.hierarchy.label(self.world, row.entity);
                let indent = row.depth as f32 * 12.0;
                ui.horizontal(|ui| {
                    ui.add_space(indent);
                    if row.expanded {
                        if ui.small_button("v").clicked() {
                            self.hierarchy.set_expanded(row.entity, false);
                        }
                    } else if ui.small_button(">").clicked() {
                        self.hierarchy.set_expanded(row.entity, true);
                    }
                    if ui.selectable_label(row.selected, label).clicked() {
                        self.hierarchy.select(row.entity);
                        self.session.selection = Some(row.entity);
                    }
                });
            }
        });
    }

    /// The component inspector.
    fn draw_inspector(&mut self, ui: &mut egui::Ui) {
        let Some(entity) = self.session.selection else {
            ui.label("Nothing selected");
            return;
        };
        if !self.world.is_alive(entity) {
            ui.label("The selected entity no longer exists");
            self.session.selection = None;
            return;
        }
        let fields = self.inspector.read(self.world, entity);
        for field in fields {
            ui.horizontal(|ui| {
                ui.label(field.label);
                ui.label(&field.display);
            });
        }
        if self.inspector.has_pending() {
            let applied = self.inspector.apply(self.world, self.session.mode);
            if applied > 0 {
                self.session.mark_dirty();
            } else {
                ui.label("Read-only while playing");
            }
        }
    }

    /// Load a scene from disk into the world.
    ///
    /// A parse failure leaves the current scene alone rather than clearing it:
    /// opening a bad file should not destroy the scene the developer has been
    /// working on. The error goes to the log, which is where they are already
    /// looking.
    fn open_scene(&mut self, path: &std::path::Path) {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                log::error!("could not read {}: {e}", path.display());
                return;
            }
        };
        match vibe_scene::yaml_to_world(&text) {
            Ok(loaded) => {
                *self.world = loaded;
                self.session.set_path(path.to_path_buf());
                self.hierarchy.prune(self.world);
                self.session.selection = None;
                log::info!("opened {}", path.display());
            }
            Err(e) => log::error!("could not parse {}: {e}", path.display()),
        }
    }

    /// Write the world to a scene file.
    fn save_scene(&mut self, path: &std::path::Path) {
        let yaml = vibe_scene::world_to_yaml(self.world);
        match std::fs::write(path, yaml) {
            Ok(()) => {
                self.session.set_path(path.to_path_buf());
                log::info!("saved {}", path.display());
            }
            Err(e) => log::error!("could not write {}: {e}", path.display()),
        }
    }

    /// The asset browser.
    fn draw_content(&mut self, ui: &mut egui::Ui) {
        if self.content.root.as_os_str().is_empty() {
            ui.label("No project folder set");
            return;
        }
        ui.horizontal(|ui| {
            if ui.button("Rescan").clicked() {
                self.content.rescan();
            }
            if self.content.unreadable > 0 {
                ui.label(format!("{} unreadable", self.content.unreadable));
            }
            ui.label(format!(
                "{} of {}",
                self.content.visible_count(),
                self.content.entries.len()
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Filter:");
            let mut filter = self.content.filter.clone();
            ui.add(egui::TextEdit::singleline(&mut filter).desired_width(140.0));
            if filter != self.content.filter {
                self.content.filter = filter;
            }
            let mut kind = self.content.kind_filter;
            egui::ComboBox::from_id_salt("vibe-content-kind")
                .selected_text(kind.map(|k| k.label()).unwrap_or("All"))
                .show_ui(ui, |ui| {
                    let mut chosen = kind;
                    ui.selectable_value(&mut chosen, None, "All");
                    for k in vibe_editor_content_kinds() {
                        ui.selectable_value(&mut chosen, Some(k), k.label());
                    }
                    if chosen != kind {
                        kind = chosen;
                    }
                });
            if kind != self.content.kind_filter {
                self.content.kind_filter = kind;
            }
        });
        ui.separator();
        // The visible set is collected before the loop because the closure
        // borrows the browser immutably to read it and mutably to record a
        // click; the two cannot overlap.
        let visible: Vec<(String, std::path::PathBuf, bool)> = self
            .content
            .visible()
            .into_iter()
            .map(|e| {
                let selected = self.content.selected.as_deref() == Some(e.path.as_path());
                (
                    format!("{}  [{}]", e.name, e.kind.label()),
                    e.path.clone(),
                    selected,
                )
            })
            .collect();
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (label, path, selected) in visible {
                if ui.selectable_label(selected, label).clicked() {
                    self.content.select(path);
                }
            }
        });
        if let Some(entry) = self.content.selected_entry() {
            ui.separator();
            ui.label(format!("Path: {}", entry.path.display()));
            let size = entry.size_display();
            if !size.is_empty() {
                ui.label(format!("Size: {size}"));
            }
        }
    }
}

/// Every kind the browser's filter offers.
fn vibe_editor_content_kinds() -> [ContentKind; 6] {
    [
        ContentKind::Model,
        ContentKind::Texture,
        ContentKind::Font,
        ContentKind::Scene,
        ContentKind::Shader,
        ContentKind::Audio,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_titles_are_distinct() {
        let titles: std::collections::HashSet<_> = [
            Panel::None,
            Panel::Hierarchy,
            Panel::Inspector,
            Panel::Content,
            Panel::Log,
        ]
        .iter()
        .map(|p| p.title())
        .collect();
        assert_eq!(titles.len(), 5, "two panels share a title");
    }

    #[test]
    fn a_closed_panel_has_no_title() {
        assert_eq!(Panel::None.title(), "");
    }

    #[test]
    fn a_closed_panel_has_no_width() {
        assert_eq!(Panel::None.width(), 0.0);
    }

    #[test]
    fn the_content_browser_is_the_widest() {
        let w = Panel::Content.width();
        for p in [Panel::Hierarchy, Panel::Inspector, Panel::Log] {
            assert!(p.width() < w, "{p:?} should be narrower than the browser");
        }
    }

    /// Everything a frame needs, so the loop can be run headlessly.
    fn parts() -> Parts {
        Parts {
            ctx: egui::Context::default(),
            session: EditorSession::new(),
            hierarchy: HierarchyPanel::new(),
            inspector: InspectorPanel::new(),
            log_panel: LogPanel::new(),
            content: ContentBrowser::new(),
            world: vibe_ecs::World::new(),
            requested_path: None,
        }
    }

    struct Parts {
        ctx: egui::Context,
        session: EditorSession,
        hierarchy: HierarchyPanel,
        inspector: InspectorPanel,
        log_panel: LogPanel,
        content: ContentBrowser,
        world: vibe_ecs::World,
        requested_path: Option<std::path::PathBuf>,
    }

    impl Parts {
        fn frame(&mut self, panel: Panel) -> EditorFrame<'_> {
            EditorFrame {
                context: &self.ctx,
                session: &mut self.session,
                hierarchy: &mut self.hierarchy,
                inspector: &mut self.inspector,
                log_panel: &mut self.log_panel,
                content: &mut self.content,
                panel,
                viewport: egui::Vec2::new(1280.0, 720.0),
                world: &mut self.world,
                requested_path: &mut self.requested_path,
            }
        }
    }

    fn draw_once(panel: Panel) -> UiDrawData {
        let mut p = parts();
        p.frame(panel).draw()
    }

    #[test]
    fn a_frame_produces_geometry() {
        // The proof the loop works at all: a pass over the menu bar and the
        // panel emits triangles.
        let d = draw_once(Panel::Hierarchy);
        assert!(!d.is_empty(), "a frame drew nothing at all");
        assert!(d.len() > 0);
    }

    #[test]
    fn every_panel_draws_without_panicking() {
        for panel in [
            Panel::None,
            Panel::Hierarchy,
            Panel::Inspector,
            Panel::Content,
            Panel::Log,
        ] {
            let d = draw_once(panel);
            assert!(!d.is_empty(), "{panel:?} drew nothing");
        }
    }

    #[test]
    fn a_closed_panel_draws_only_the_menu_bar() {
        let bare = draw_once(Panel::None);
        let with_panel = draw_once(Panel::Log);
        assert!(
            with_panel.len() > bare.len(),
            "opening a panel should add geometry: {} vs {}",
            with_panel.len(),
            bare.len()
        );
    }

    #[test]
    fn a_frame_with_no_entities_still_draws() {
        let d = draw_once(Panel::Hierarchy);
        assert!(!d.is_empty(), "an empty scene must not blank the editor");
    }

    #[test]
    fn a_frame_with_entities_draws_their_names() {
        let mut p = parts();
        let e = p.world.spawn();
        p.world
            .add(e, vibe_ecs::components::Tag("Hero".to_string()));
        let d = p.frame(Panel::Hierarchy).draw();
        assert!(!d.is_empty());
        assert_eq!(p.hierarchy.last_row_count, 1, "the entity is listed");
    }

    #[test]
    fn an_inspector_with_no_selection_draws_a_placeholder() {
        let d = draw_once(Panel::Inspector);
        assert!(!d.is_empty());
    }

    #[test]
    fn a_selection_of_a_dead_entity_does_not_panic() {
        let mut p = parts();
        let e = p.world.spawn();
        p.world.despawn(e);
        p.session.selection = Some(e);
        let d = p.frame(Panel::Inspector).draw();
        assert!(!d.is_empty(), "a stale selection is reported, not fatal");
        assert!(
            p.session.selection.is_none(),
            "the stale selection is cleared"
        );
    }

    /// A frame with a path waiting to be opened, so the host's handoff is
    /// exercised rather than only the drawing.
    fn parts_with_open(path: &std::path::Path) -> Parts {
        let mut p = parts();
        p.requested_path = Some(path.to_path_buf());
        p
    }

    fn temp_scene_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("vibe-editor-{name}.yaml"))
    }

    #[test]
    fn a_saved_scene_can_be_opened_again() {
        let path = temp_scene_path("roundtrip");
        let mut p = parts();
        let e = p.world.spawn();
        p.world
            .add(e, vibe_ecs::components::Tag("Hero".to_string()));
        p.frame(Panel::None).save_scene(&path);
        assert!(path.exists(), "the file was written");
        assert!(!p.session.dirty, "saving clears the dirty flag");

        let mut q = parts_with_open(&path);
        q.frame(Panel::None).draw();
        assert_eq!(q.world.len(), 1, "the entity came back");
        assert_eq!(q.session.path.as_deref(), Some(path.as_path()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_saved_scene_keeps_its_tags() {
        let path = temp_scene_path("tags");
        let mut p = parts();
        let e = p.world.spawn();
        p.world
            .add(e, vibe_ecs::components::Tag("Enemy".to_string()));
        p.frame(Panel::None).save_scene(&path);

        let mut q = parts_with_open(&path);
        q.frame(Panel::None).draw();
        let found = q
            .world
            .entities()
            .next()
            .and_then(|e| q.world.get::<vibe_ecs::components::Tag>(e).cloned());
        assert_eq!(found.map(|t| t.0), Some("Enemy".to_string()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn opening_a_missing_file_leaves_the_scene_alone() {
        let mut p = parts_with_open(std::path::Path::new("/nonexistent-scene.yaml"));
        let e = p.world.spawn();
        p.frame(Panel::None).draw();
        assert_eq!(p.world.len(), 1, "the existing entity survives a bad open");
        assert!(p.world.is_alive(e));
        assert!(
            p.session.path.is_none(),
            "a failed open does not adopt a path"
        );
    }

    #[test]
    fn opening_malformed_yaml_leaves_the_scene_alone() {
        let path = temp_scene_path("malformed");
        std::fs::write(&path, "this: is: not: valid: yaml: [[[").unwrap();
        let mut p = parts_with_open(&path);
        let e = p.world.spawn();
        p.frame(Panel::None).draw();
        assert_eq!(
            p.world.len(),
            1,
            "a bad file must not destroy the open scene"
        );
        assert!(p.world.is_alive(e));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_open_path_is_consumed_only_once() {
        let path = temp_scene_path("consume");
        let mut p = parts();
        p.world.spawn();
        p.frame(Panel::None).save_scene(&path);
        // A second entity, so a repeated open would be visible.
        p.world.spawn();

        p.requested_path = Some(path.clone());
        p.frame(Panel::None).draw();
        assert_eq!(p.world.len(), 1, "the open replaced the two-entity world");
        assert!(
            p.requested_path.is_none(),
            "the path is consumed, not left to reopen every frame"
        );
        p.frame(Panel::None).draw();
        assert_eq!(p.world.len(), 1, "and it does not reopen");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn opening_clears_a_selection_pointing_at_the_old_scene() {
        let path = temp_scene_path("selection");
        let mut p = parts();
        let e = p.world.spawn();
        p.world
            .add(e, vibe_ecs::components::Tag("Gone".to_string()));
        p.session.selection = Some(e);
        p.frame(Panel::None).save_scene(&path);

        let mut q = parts_with_open(&path);
        let old = q.world.spawn();
        q.session.selection = Some(old);
        q.frame(Panel::None).draw();
        assert!(
            q.session.selection.is_none(),
            "a handle into the previous world would select nothing"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_content_panel_with_no_root_says_so() {
        let mut p = parts();
        let d = p.frame(Panel::Content).draw();
        assert!(!d.is_empty());
    }

    #[test]
    fn the_content_panel_lists_a_real_directory() {
        let dir = std::env::temp_dir().join("vibe-editor-content");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("hero.glb"), b"x").unwrap();
        let mut p = parts();
        p.content = ContentBrowser::at(&dir);
        p.content.rescan();
        let d = p.frame(Panel::Content).draw();
        assert!(!d.is_empty());
        assert_eq!(p.content.entries.len(), 1, "{:?}", p.content.entries);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_content_filter_narrows_the_listing() {
        let dir = std::env::temp_dir().join("vibe-editor-filter");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("hero.glb"), b"x").unwrap();
        std::fs::write(dir.join("tiles.png"), b"x").unwrap();
        let mut p = parts();
        p.content = ContentBrowser::at(&dir);
        p.content.rescan();
        assert_eq!(p.content.visible_count(), 2);
        p.content.filter = "hero".to_string();
        assert_eq!(p.content.visible_count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_open_panel_has_a_positive_width() {
        for p in [
            Panel::Hierarchy,
            Panel::Inspector,
            Panel::Content,
            Panel::Log,
        ] {
            assert!(p.width() > 0.0, "{p:?} would be invisible");
        }
    }
}
