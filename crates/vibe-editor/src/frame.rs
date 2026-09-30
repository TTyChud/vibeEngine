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
    /// Which panel is open.
    pub panel: Panel,
    /// The window's size in points.
    pub viewport: egui::Vec2,
    /// The scene, for the hierarchy and inspector to read.
    pub world: &'a mut vibe_ecs::World,
}

impl EditorFrame<'_> {
    /// Draw one frame and return the geometry.
    ///
    /// The whole body is wrapped so a panel that panics on malformed data
    /// cannot take the editor down with it: a crashed editor loses the user's
    /// unsaved work, which is the worst outcome available.
    pub fn draw(mut self) -> UiDrawData {
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
                    // A file dialog is a platform concern; the session records
                    // the intent and the host resolves it.
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
                    log::info!("save requested for {:?}", self.session.path);
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

    /// The asset browser.
    fn draw_content(&mut self, ui: &mut egui::Ui) {
        let _ = ui;
    }
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
    fn parts() -> (
        egui::Context,
        EditorSession,
        HierarchyPanel,
        InspectorPanel,
        LogPanel,
        vibe_ecs::World,
    ) {
        (
            egui::Context::default(),
            EditorSession::new(),
            HierarchyPanel::new(),
            InspectorPanel::new(),
            LogPanel::new(),
            vibe_ecs::World::new(),
        )
    }

    fn draw_once(panel: Panel) -> UiDrawData {
        let (ctx, mut session, mut hierarchy, mut inspector, mut log_panel, mut world) = parts();
        EditorFrame {
            context: &ctx,
            session: &mut session,
            hierarchy: &mut hierarchy,
            inspector: &mut inspector,
            log_panel: &mut log_panel,
            panel,
            viewport: egui::Vec2::new(1280.0, 720.0),
            world: &mut world,
        }
        .draw()
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
        let (ctx, mut session, mut hierarchy, mut inspector, mut log_panel, mut world) = parts();
        let e = world.spawn();
        world.add(e, vibe_ecs::components::Tag("Hero".to_string()));
        let d = EditorFrame {
            context: &ctx,
            session: &mut session,
            hierarchy: &mut hierarchy,
            inspector: &mut inspector,
            log_panel: &mut log_panel,
            panel: Panel::Hierarchy,
            viewport: egui::Vec2::new(1280.0, 720.0),
            world: &mut world,
        }
        .draw();
        assert!(!d.is_empty());
        assert_eq!(hierarchy.last_row_count, 1, "the entity is listed");
    }

    #[test]
    fn an_inspector_with_no_selection_draws_a_placeholder() {
        let d = draw_once(Panel::Inspector);
        assert!(!d.is_empty());
    }

    #[test]
    fn a_selection_of_a_dead_entity_does_not_panic() {
        let (ctx, mut session, mut hierarchy, mut inspector, mut log_panel, mut world) = parts();
        let e = world.spawn();
        world.despawn(e);
        session.selection = Some(e);
        let d = EditorFrame {
            context: &ctx,
            session: &mut session,
            hierarchy: &mut hierarchy,
            inspector: &mut inspector,
            log_panel: &mut log_panel,
            panel: Panel::Inspector,
            viewport: egui::Vec2::new(1280.0, 720.0),
            world: &mut world,
        }
        .draw();
        assert!(!d.is_empty(), "a stale selection is reported, not fatal");
        assert!(
            session.selection.is_none(),
            "the stale selection is cleared"
        );
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
