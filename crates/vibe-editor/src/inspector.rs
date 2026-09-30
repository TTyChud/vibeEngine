//! The component inspector: what is on the selected entity, and how to change it.
//!
//! The panel reads the entity's components and reports edits as a diff rather
//! than writing them straight in. That is what makes a running game safe: the
//! diff is only applied when the editor is in edit mode, so a stray drag on a
//! paused frame cannot rewrite the scene under the simulation.

use glam::{Quat, Vec3, Vec4};
use vibe_ecs::{Entity, World, components};

use crate::session::EditorMode;

/// One changed field, ready to apply.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldEdit {
    /// The entity's position.
    Translation(Vec3),
    /// The entity's rotation.
    Rotation(Quat),
    /// The entity's scale.
    Scale(Vec3),
    /// The entity's tint.
    Tint(Vec4),
    /// The entity's sprite size.
    Size(glam::Vec2),
    /// The entity's text.
    Text(String),
    /// The entity's name.
    Name(String),
}

/// A value read off an entity, for drawing.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldValue {
    /// The field's name, as shown.
    pub label: &'static str,
    /// The value, as a string.
    pub display: String,
}

impl FieldValue {
    /// A named value.
    pub fn new(label: &'static str, display: impl Into<String>) -> FieldValue {
        FieldValue {
            label,
            display: display.into(),
        }
    }
}

/// The inspector's state.
pub struct InspectorPanel {
    /// The edits the user has made but not applied.
    pub pending: Vec<FieldEdit>,
    /// The entity the fields were read from.
    pub inspected: Option<Entity>,
    /// True when a field is being dragged.
    pub editing: bool,
    /// How many fields the last read found.
    pub last_field_count: usize,
}

impl Default for InspectorPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for InspectorPanel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InspectorPanel")
            .field("inspected", &self.inspected)
            .field("pending", &self.pending.len())
            .finish()
    }
}

impl InspectorPanel {
    /// A panel with nothing to show.
    pub fn new() -> InspectorPanel {
        InspectorPanel {
            pending: Vec::new(),
            inspected: None,
            editing: false,
            last_field_count: 0,
        }
    }

    /// Read an entity's components into displayable fields.
    ///
    /// The order is fixed rather than derived from the world's component set, so
    /// a panel does not rearrange itself between frames as components are added
    /// and removed — a control that moves under the cursor mid-drag is how a
    /// value gets set to the wrong thing.
    pub fn read(&mut self, world: &World, entity: Entity) -> Vec<FieldValue> {
        self.inspected = Some(entity);
        let mut out = Vec::new();

        if let Some(t) = world.get::<components::Transform>(entity) {
            out.push(FieldValue::new(
                "Position",
                format!("{:.2}, {:.2}, {:.2}", t.x, t.y, t.z),
            ));
            out.push(FieldValue::new(
                "Rotation",
                format!(
                    "{:.1}, {:.1}, {:.1}, {:.1}",
                    t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
                ),
            ));
            out.push(FieldValue::new(
                "Scale",
                format!("{:.2}, {:.2}, {:.2}", t.scale.x, t.scale.y, t.scale.z),
            ));
        }
        if let Some(s) = world.get::<components::SpriteRenderer>(entity) {
            out.push(FieldValue::new(
                "Size",
                format!("{:.0} x {:.0}", s.size.x, s.size.y),
            ));
            out.push(FieldValue::new(
                "Rotation",
                format!("{:.1} deg", s.rotation_radians.to_degrees()),
            ));
        }
        if let Some(m) = world.get::<components::Model3D>(entity) {
            out.push(FieldValue::new("Mesh", m.mesh_path.clone()));
            out.push(FieldValue::new("Skin", format!("{}", m.skin_index)));
        }
        if let Some(t) = world.get::<components::Text>(entity) {
            out.push(FieldValue::new("Text", t.content.clone()));
            out.push(FieldValue::new("Font size", format!("{:.0}", t.size)));
        }
        if let Some(a) = world.get::<components::Animator>(entity) {
            out.push(FieldValue::new("Clip", format!("{}", a.clip_index)));
            out.push(FieldValue::new("Speed", format!("{:.2}", a.speed)));
        }
        if let Some(l) = world.get::<components::DirectionalLight>(entity) {
            out.push(FieldValue::new(
                "Light colour",
                format!("{:.2}, {:.2}, {:.2}", l.color.x, l.color.y, l.color.z),
            ));
            out.push(FieldValue::new("Intensity", format!("{:.2}", l.intensity)));
        }
        if let Some(b) = world.get::<components::Rigidbody2D>(entity) {
            out.push(FieldValue::new("Mass", format!("{:.2}", b.mass)));
        }
        if let Some(b) = world.get::<components::Rigidbody3D>(entity) {
            out.push(FieldValue::new("Mass", format!("{:.2}", b.mass)));
        }
        if let Some(t) = world.get::<components::Tag>(entity) {
            out.push(FieldValue::new("Name", t.0.clone()));
        }

        self.last_field_count = out.len();
        out
    }

    /// Read an entity without keeping it as the inspected one.
    pub fn peek(world: &World, entity: Entity) -> Vec<FieldValue> {
        let mut p = InspectorPanel::new();
        p.read(world, entity)
    }

    /// Queue an edit.
    pub fn stage(&mut self, edit: FieldEdit) {
        self.pending.push(edit);
    }

    /// Discard the queued edits.
    pub fn discard(&mut self) {
        self.pending.clear();
    }

    /// True when there is something to apply.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Apply the queued edits, if the mode allows editing.
    ///
    /// Returns how many were applied. A paused or playing game gets zero: the
    /// edits are kept so returning to edit mode does not silently lose them,
    /// which is the behaviour a developer expects after pausing to look.
    pub fn apply(&mut self, world: &mut World, mode: EditorMode) -> usize {
        if !mode.allows_editing() {
            return 0;
        }
        let Some(entity) = self.inspected else {
            return 0;
        };
        if !world.is_alive(entity) {
            self.pending.clear();
            return 0;
        }
        let count = self.pending.len();
        for edit in self.pending.drain(..) {
            apply_edit(world, entity, edit);
        }
        count
    }
}

/// Write one edit into the world.
fn apply_edit(world: &mut World, entity: Entity, edit: FieldEdit) {
    match edit {
        FieldEdit::Translation(v) => {
            if let Some(t) = world.get_mut::<components::Transform>(entity) {
                t.x = v.x;
                t.y = v.y;
                t.z = v.z;
            }
        }
        FieldEdit::Rotation(q) => {
            if let Some(t) = world.get_mut::<components::Transform>(entity) {
                t.rotation = q;
            }
        }
        FieldEdit::Scale(v) => {
            if let Some(t) = world.get_mut::<components::Transform>(entity) {
                t.scale = v;
            }
        }
        FieldEdit::Tint(v) => {
            if let Some(m) = world.get_mut::<components::Model3D>(entity) {
                m.tint = v;
            }
        }
        FieldEdit::Size(v) => {
            if let Some(s) = world.get_mut::<components::SpriteRenderer>(entity) {
                s.size = v;
            }
        }
        FieldEdit::Text(s) => {
            if let Some(t) = world.get_mut::<components::Text>(entity) {
                t.content = s;
            }
        }
        FieldEdit::Name(s) => {
            if let Some(t) = world.get_mut::<components::Tag>(entity) {
                t.0 = s;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tagged_world() -> (World, Entity) {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, components::Tag("Player".to_string()));
        (w, e)
    }

    #[test]
    fn a_new_panel_has_nothing_pending() {
        assert!(!InspectorPanel::new().has_pending());
    }

    #[test]
    fn an_entity_with_no_components_has_no_fields() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(InspectorPanel::peek(&w, e).is_empty());
    }

    #[test]
    fn a_transform_reports_three_fields() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, components::Transform::default());
        let fields = InspectorPanel::peek(&w, e);
        assert_eq!(fields.len(), 3);
    }

    #[test]
    fn a_tag_is_reported() {
        let (w, e) = tagged_world();
        let fields = InspectorPanel::peek(&w, e);
        assert!(fields.iter().any(|f| f.label == "Name"), "{fields:?}");
    }

    #[test]
    fn a_tagged_entity_reports_its_name() {
        let (w, e) = tagged_world();
        let fields = InspectorPanel::peek(&w, e);
        let name = fields.iter().find(|f| f.label == "Name").unwrap();
        assert_eq!(name.display, "Player");
    }

    #[test]
    fn a_transform_reports_its_position() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            components::Transform {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                ..Default::default()
            },
        );
        let fields = InspectorPanel::peek(&w, e);
        let p = fields.iter().find(|f| f.label == "Position").unwrap();
        assert_eq!(p.display, "1.00, 2.00, 3.00");
    }

    #[test]
    fn fields_come_back_in_a_stable_order() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, components::Transform::default());
        w.add(e, components::Tag("A".to_string()));
        let first = InspectorPanel::peek(&w, e);
        let second = InspectorPanel::peek(&w, e);
        assert_eq!(first, second, "a control must not move between frames");
    }

    #[test]
    fn reading_records_the_entity() {
        let (w, e) = tagged_world();
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        assert_eq!(p.inspected, Some(e));
    }

    #[test]
    fn reading_counts_the_fields() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, components::Transform::default());
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        assert_eq!(p.last_field_count, 3);
    }

    #[test]
    fn staging_an_edit_marks_the_panel_dirty() {
        let mut p = InspectorPanel::new();
        p.stage(FieldEdit::Name("x".to_string()));
        assert!(p.has_pending());
    }

    #[test]
    fn discarding_clears_the_edits() {
        let mut p = InspectorPanel::new();
        p.stage(FieldEdit::Name("x".to_string()));
        p.discard();
        assert!(!p.has_pending());
    }

    #[test]
    fn applying_writes_the_translation() {
        let (mut w, e) = tagged_world();
        w.add(e, components::Transform::default());
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Translation(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(p.apply(&mut w, EditorMode::Edit), 1);
        let t = w.get::<components::Transform>(e).unwrap();
        assert_eq!(t.translation(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn applying_writes_the_name() {
        let (mut w, e) = tagged_world();
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Name("Enemy".to_string()));
        p.apply(&mut w, EditorMode::Edit);
        assert_eq!(w.get::<components::Tag>(e).unwrap().0, "Enemy");
    }

    #[test]
    fn applying_writes_the_scale() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, components::Transform::default());
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Scale(Vec3::splat(3.0)));
        p.apply(&mut w, EditorMode::Edit);
        assert_eq!(
            w.get::<components::Transform>(e).unwrap().scale,
            Vec3::splat(3.0)
        );
    }

    #[test]
    fn applying_empties_the_queue() {
        let (mut w, e) = tagged_world();
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Name("x".to_string()));
        p.apply(&mut w, EditorMode::Edit);
        assert!(!p.has_pending());
    }

    #[test]
    fn a_running_game_refuses_to_apply() {
        let (mut w, e) = tagged_world();
        w.add(e, components::Transform::default());
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Translation(Vec3::splat(9.0)));
        assert_eq!(p.apply(&mut w, EditorMode::Play), 0);
        assert_eq!(w.get::<components::Transform>(e).unwrap().x, 0.0);
    }

    #[test]
    fn a_refused_edit_is_kept_for_later() {
        let (mut w, e) = tagged_world();
        w.add(e, components::Transform::default());
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Translation(Vec3::splat(9.0)));
        p.apply(&mut w, EditorMode::Pause);
        assert!(
            p.has_pending(),
            "pausing to edit and resuming must not lose the edit"
        );
    }

    #[test]
    fn a_kept_edit_applies_once_editing_resumes() {
        let (mut w, e) = tagged_world();
        w.add(e, components::Transform::default());
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Translation(Vec3::splat(9.0)));
        p.apply(&mut w, EditorMode::Pause);
        p.apply(&mut w, EditorMode::Edit);
        assert_eq!(w.get::<components::Transform>(e).unwrap().x, 9.0);
    }

    #[test]
    fn applying_to_a_dead_entity_applies_nothing() {
        let (mut w, e) = tagged_world();
        w.add(e, components::Transform::default());
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Name("x".to_string()));
        w.despawn(e);
        assert_eq!(p.apply(&mut w, EditorMode::Edit), 0);
    }

    #[test]
    fn applying_to_a_dead_entity_clears_the_queue() {
        let (mut w, e) = tagged_world();
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Name("x".to_string()));
        w.despawn(e);
        p.apply(&mut w, EditorMode::Edit);
        assert!(!p.has_pending());
    }

    #[test]
    fn applying_with_nothing_read_applies_nothing() {
        let (mut w, _) = tagged_world();
        let mut p = InspectorPanel::new();
        p.stage(FieldEdit::Name("x".to_string()));
        assert_eq!(p.apply(&mut w, EditorMode::Edit), 0);
    }

    #[test]
    fn an_edit_for_a_missing_component_is_ignored() {
        // A Translation edit on an entity with no Transform must not add one:
        // an inspector that invents components hides the mistake instead of
        // showing it.
        let (mut w, e) = tagged_world();
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Translation(Vec3::splat(5.0)));
        p.apply(&mut w, EditorMode::Edit);
        assert!(w.get::<components::Transform>(e).is_none());
    }

    #[test]
    fn several_edits_apply_in_order() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, components::Transform::default());
        let mut p = InspectorPanel::new();
        p.read(&w, e);
        p.stage(FieldEdit::Translation(Vec3::new(1.0, 0.0, 0.0)));
        p.stage(FieldEdit::Translation(Vec3::new(2.0, 0.0, 0.0)));
        assert_eq!(p.apply(&mut w, EditorMode::Edit), 2);
        assert_eq!(
            w.get::<components::Transform>(e).unwrap().x,
            2.0,
            "the last wins"
        );
    }
}
