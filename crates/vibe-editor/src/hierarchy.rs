//! The scene hierarchy panel: the entity tree, and what is selected in it.
//!
//! The tree is derived from the ECS's own components rather than kept
//! alongside them. A separate tree needs its own sync, and every path that
//! spawns or despawns an entity has to remember to update it; a despawned entity
//! left in a separate list is a stale row that selects nothing and cannot be
//! removed, which is the bug that makes a hand-maintained tree unusable.

use std::collections::HashMap;

use vibe_ecs::{Entity, World, components::Transform};

/// Which entity the inspector is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Selection {
    /// The selected entity, if any.
    pub entity: Option<Entity>,
}

impl Selection {
    /// Nothing selected.
    pub fn none() -> Selection {
        Selection { entity: None }
    }

    /// True when something is selected.
    pub fn is_some(&self) -> bool {
        self.entity.is_some()
    }

    /// Select an entity, or clear the selection.
    pub fn select(&mut self, entity: Option<Entity>) {
        self.entity = entity;
    }

    /// Clear the selection.
    pub fn clear(&mut self) {
        self.entity = None;
    }

    /// True when the given entity is selected.
    pub fn is(&self, entity: Entity) -> bool {
        self.entity == Some(entity)
    }
}

/// One row of the tree, flattened for drawing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Row {
    /// The entity this row is.
    pub entity: Entity,
    /// How deep in the tree, for the indent.
    pub depth: u32,
    /// True when the row's children are shown.
    pub expanded: bool,
    /// True when this row is the selected one.
    pub selected: bool,
}

/// The hierarchy panel's state.
pub struct HierarchyPanel {
    /// What is selected.
    pub selection: Selection,
    /// The entities whose children are shown.
    pub expanded: std::collections::HashSet<Entity>,
    /// Each entity's children, which the ECS's `Transform` does not carry.
    ///
    /// The engine's transform is pure TRS with no parent, so parenting is the
    /// editor's own bookkeeping. It is rebuilt from the scene on load rather
    /// than persisted separately, which is why a parent that no longer exists
    /// is dropped rather than dangling.
    pub children: HashMap<Entity, Vec<Entity>>,
    /// A filter applied to entity names; empty shows everything.
    pub filter: String,
    /// How many rows the last draw produced.
    pub last_row_count: usize,
}

impl Default for HierarchyPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for HierarchyPanel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HierarchyPanel")
            .field("selection", &self.selection)
            .field("expanded", &self.expanded.len())
            .field("parented", &self.parented_count())
            .field("filter", &self.filter)
            .finish()
    }
}

impl HierarchyPanel {
    /// A panel with nothing selected.
    pub fn new() -> HierarchyPanel {
        HierarchyPanel {
            selection: Selection::none(),
            expanded: std::collections::HashSet::new(),
            children: HashMap::new(),
            filter: String::new(),
            last_row_count: 0,
        }
    }

    /// Select an entity.
    pub fn select(&mut self, entity: Entity) {
        self.selection.select(Some(entity));
    }

    /// Clear the selection.
    pub fn clear_selection(&mut self) {
        self.selection.clear();
    }

    /// True when an entity's children are shown.
    pub fn is_expanded(&self, entity: Entity) -> bool {
        self.expanded.contains(&entity)
    }

    /// Show or hide an entity's children.
    pub fn set_expanded(&mut self, entity: Entity, expanded: bool) {
        if expanded {
            self.expanded.insert(entity);
        } else {
            self.expanded.remove(&entity);
        }
    }

    /// Flip an entity's expansion.
    pub fn toggle_expanded(&mut self, entity: Entity) {
        if !self.expanded.remove(&entity) {
            self.expanded.insert(entity);
        }
    }

    /// Collapse everything, which is what Escape from a deep tree wants.
    pub fn collapse_all(&mut self) {
        self.expanded.clear();
    }

    /// Expand everything in a world.
    pub fn expand_all(&mut self, world: &World) {
        self.expanded = world.entities().collect();
    }

    /// True when a name passes the filter.
    ///
    /// An empty filter passes everything, and the comparison is
    /// case-insensitive because a filter box that is case-sensitive makes users
    /// retype what they already typed.
    pub fn matches_filter(&self, name: &str) -> bool {
        if self.filter.is_empty() {
            return true;
        }
        name.to_lowercase().contains(&self.filter.to_lowercase())
    }

    /// How many entities have a parent.
    pub fn parented_count(&self) -> usize {
        self.children.values().map(|c| c.len()).sum()
    }

    /// Make an entity a child of another.
    ///
    /// Re-parenting removes the entity from its previous parent, so a child
    /// appears in exactly one list; leaving it in both would draw it twice and
    /// make despawning it leave a row behind.
    pub fn set_parent(&mut self, child: Entity, parent: Option<Entity>) {
        self.detach(child);
        if let Some(parent) = parent {
            self.children.entry(parent).or_default().push(child);
        }
    }

    /// Remove an entity from its parent's child list.
    pub fn detach(&mut self, child: Entity) {
        for list in self.children.values_mut() {
            list.retain(|c| *c != child);
        }
        self.children.retain(|_, v| !v.is_empty());
    }

    /// An entity's parent, if it has one.
    pub fn parent_of(&self, entity: Entity) -> Option<Entity> {
        self.children
            .iter()
            .find(|(_, list)| list.contains(&entity))
            .map(|(p, _)| *p)
    }

    /// An entity's children, in insertion order.
    pub fn children_of(&self, entity: Entity) -> &[Entity] {
        self.children
            .get(&entity)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Drop every entity that is no longer alive.
    ///
    /// A child naming a dead parent would be invisible, because its row is only
    /// drawn when the parent's is.
    pub fn prune(&mut self, world: &World) {
        for list in self.children.values_mut() {
            list.retain(|c| world.is_alive(*c));
        }
        self.children
            .retain(|p, list| world.is_alive(*p) && !list.is_empty());
        if let Some(selected) = self.selection.entity
            && !world.is_alive(selected)
        {
            self.selection.clear();
        }
    }

    /// Flatten the world into the rows to draw, in draw order.
    ///
    /// An entity whose parent is not shown still gets a row: a filter that hid
    /// the parent would otherwise leave a matching child visible but
    /// unreachable, which is worse than an extra-looking root.
    pub fn rows(&self, world: &World) -> Vec<Row> {
        let mut out: Vec<Row> = Vec::new();
        let mut emitted: std::collections::HashSet<Entity> = std::collections::HashSet::new();

        for entity in world.entities() {
            if self.parent_of(entity).is_none() {
                self.push_row(world, entity, 0, &mut out, &mut emitted);
            }
        }
        // Anything not reached from a root is emitted as its own root, so a
        // cycle or a dead parent cannot make an entity vanish from the tree.
        for entity in world.entities() {
            if !emitted.contains(&entity) {
                out.push(Row {
                    entity,
                    depth: 0,
                    expanded: self.is_expanded(entity),
                    selected: self.selection.is(entity),
                });
            }
        }
        out
    }

    /// Push one entity and its visible descendants.
    fn push_row(
        &self,
        world: &World,
        entity: Entity,
        depth: u32,
        out: &mut Vec<Row>,
        emitted: &mut std::collections::HashSet<Entity>,
    ) {
        if emitted.contains(&entity) || !world.is_alive(entity) {
            // A cycle in the child links would otherwise recurse forever.
            return;
        }
        emitted.insert(entity);
        out.push(Row {
            entity,
            depth,
            expanded: self.is_expanded(entity),
            selected: self.selection.is(entity),
        });
        if !self.is_expanded(entity) {
            return;
        }
        for child in self.children_of(entity) {
            self.push_row(world, *child, depth + 1, out, emitted);
        }
    }

    /// Record how many rows were drawn, for the panel's status line.
    ///
    /// `rows` is a pure query — a read-only traversal has no business writing
    /// to the panel it reads from — so the count is set here, by whoever drew.
    pub fn note_row_count(&mut self, count: usize) {
        self.last_row_count = count;
    }

    /// The name shown for an entity.
    pub fn label(&self, world: &World, entity: Entity) -> String {
        named_label(world, entity)
    }
}

/// The label for an entity: its tag, or its id.
pub fn named_label(world: &World, entity: Entity) -> String {
    if let Some(tag) = world.get::<vibe_ecs::components::Tag>(entity) {
        if !tag.0.is_empty() {
            return tag.0.clone();
        }
    }
    format!("Entity {}", entity.index())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vibe_ecs::components::Tag;

    fn world_with_roots() -> World {
        let mut w = World::new();
        w.spawn();
        w.spawn();
        w
    }

    fn handle(index: u32) -> Entity {
        Entity::from_parts(index, 1)
    }

    #[test]
    fn a_new_panel_has_nothing_selected() {
        let p = HierarchyPanel::new();
        assert!(!p.selection.is_some());
    }

    #[test]
    fn selecting_records_the_entity() {
        let mut p = HierarchyPanel::new();
        let e = Entity::from_parts(0, 1);
        p.select(e);
        assert!(p.selection.is(e));
    }

    #[test]
    fn selecting_none_clears_it() {
        let mut p = HierarchyPanel::new();
        p.selection.select(Some(Entity::from_parts(0, 1)));
        p.selection.select(None);
        assert!(!p.selection.is_some());
    }

    #[test]
    fn clearing_removes_the_selection() {
        let mut p = HierarchyPanel::new();
        p.select(Entity::from_parts(0, 1));
        p.clear_selection();
        assert!(!p.selection.is_some());
    }

    #[test]
    fn an_unexpanded_entity_is_not_expanded() {
        assert!(!HierarchyPanel::new().is_expanded(Entity::from_parts(0, 1)));
    }

    #[test]
    fn setting_expanded_expands() {
        let mut p = HierarchyPanel::new();
        let e = Entity::from_parts(0, 1);
        p.set_expanded(e, true);
        assert!(p.is_expanded(e));
    }

    #[test]
    fn setting_not_expanded_collapses() {
        let mut p = HierarchyPanel::new();
        let e = Entity::from_parts(0, 1);
        p.set_expanded(e, true);
        p.set_expanded(e, false);
        assert!(!p.is_expanded(e));
    }

    #[test]
    fn toggling_flips_the_state() {
        let mut p = HierarchyPanel::new();
        let e = Entity::from_parts(0, 1);
        p.toggle_expanded(e);
        assert!(p.is_expanded(e));
        p.toggle_expanded(e);
        assert!(!p.is_expanded(e));
    }

    #[test]
    fn collapsing_all_empties_the_set() {
        let mut p = HierarchyPanel::new();
        p.set_expanded(Entity::from_parts(0, 1), true);
        p.set_expanded(Entity::from_parts(1, 1), true);
        p.collapse_all();
        assert!(p.expanded.is_empty());
    }

    #[test]
    fn an_empty_filter_passes_everything() {
        let p = HierarchyPanel::new();
        assert!(p.matches_filter("anything"));
    }

    #[test]
    fn a_filter_matches_a_substring() {
        let mut p = HierarchyPanel::new();
        p.filter = "play".to_string();
        assert!(p.matches_filter("Player"));
        assert!(!p.matches_filter("Enemy"));
    }

    #[test]
    fn a_filter_ignores_case() {
        let mut p = HierarchyPanel::new();
        p.filter = "PLAY".to_string();
        assert!(p.matches_filter("player"), "the user should not retype");
    }

    #[test]
    fn an_empty_world_has_no_rows() {
        let p = HierarchyPanel::new();
        assert!(p.rows(&World::new()).is_empty());
    }

    #[test]
    fn a_world_of_roots_yields_one_row_each() {
        let w = world_with_roots();
        let p = HierarchyPanel::new();
        assert_eq!(p.rows(&w).len(), 2);
    }

    #[test]
    fn a_parent_is_a_root_while_its_child_is_not() {
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn();
        let mut p = HierarchyPanel::new();
        p.set_parent(child, Some(parent));
        assert_eq!(p.parent_of(parent), None);
        assert_eq!(p.parent_of(child), Some(parent));
    }

    #[test]
    fn a_nested_child_is_indented_only_once_expanded() {
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn();
        let mut p = HierarchyPanel::new();
        p.set_parent(child, Some(parent));

        let collapsed = p.rows(&w);
        assert!(
            collapsed.iter().all(|r| r.depth == 0),
            "a collapsed child draws at the root depth: {collapsed:?}"
        );

        p.set_expanded(parent, true);
        let expanded = p.rows(&w);
        assert!(
            expanded.iter().any(|r| r.depth == 1),
            "expanding indents the child: {expanded:?}"
        );
    }

    #[test]
    fn re_parenting_moves_the_child_rather_than_copying_it() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        let child = w.spawn();
        let mut p = HierarchyPanel::new();
        p.set_parent(child, Some(a));
        p.set_parent(child, Some(b));
        assert_eq!(p.children_of(a).len(), 0, "the old parent let go");
        assert_eq!(p.children_of(b), &[child]);
    }

    #[test]
    fn detaching_makes_a_child_a_root_again() {
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn();
        let mut p = HierarchyPanel::new();
        p.set_parent(child, Some(parent));
        p.detach(child);
        assert_eq!(p.parent_of(child), None);
        assert!(p.rows(&w).iter().all(|r| r.depth == 0));
    }

    #[test]
    fn an_empty_child_list_is_not_kept() {
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn();
        let mut p = HierarchyPanel::new();
        p.set_parent(child, Some(parent));
        p.detach(child);
        assert!(
            !p.children.contains_key(&parent),
            "an empty entry would make a leaf look like it has state"
        );
    }

    #[test]
    fn every_entity_appears_exactly_once() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        w.add(b, Transform::default());
        let mut p = HierarchyPanel::new();
        p.set_parent(b, Some(a));
        p.expand_all(&w);
        let rows = p.rows(&w);
        let mut seen = std::collections::HashSet::new();
        for r in &rows {
            assert!(seen.insert(r.entity), "{} appeared twice", r.entity);
        }
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn a_parent_cycle_does_not_recurse_forever() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        let mut p = HierarchyPanel::new();
        p.set_parent(b, Some(a));
        p.set_parent(a, Some(b));
        p.expand_all(&w);
        let rows = p.rows(&w);
        assert!(rows.len() <= 4, "the cycle is cut, not followed: {rows:?}");
    }

    #[test]
    fn a_cycled_entity_is_still_shown_once() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        let mut p = HierarchyPanel::new();
        p.set_parent(b, Some(a));
        p.set_parent(a, Some(b));
        p.expand_all(&w);
        let rows = p.rows(&w);
        let mut seen = std::collections::HashSet::new();
        for r in &rows {
            assert!(seen.insert(r.entity), "{} appeared twice", r.entity);
        }
    }

    #[test]
    fn pruning_drops_a_dead_child() {
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn();
        let mut p = HierarchyPanel::new();
        p.set_parent(child, Some(parent));
        w.despawn(child);
        p.prune(&w);
        assert_eq!(p.children_of(parent).len(), 0);
    }

    #[test]
    fn pruning_clears_a_dead_selection() {
        let mut w = World::new();
        let e = w.spawn();
        let mut p = HierarchyPanel::new();
        p.select(e);
        w.despawn(e);
        p.prune(&w);
        assert!(
            !p.selection.is_some(),
            "the inspector must not show a corpse"
        );
    }

    #[test]
    fn pruning_keeps_a_live_selection() {
        let mut w = World::new();
        let e = w.spawn();
        let mut p = HierarchyPanel::new();
        p.select(e);
        p.prune(&w);
        assert!(p.selection.is_some());
    }

    #[test]
    fn the_parented_count_matches_the_map() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        let c = w.spawn();
        let mut p = HierarchyPanel::new();
        p.set_parent(b, Some(a));
        p.set_parent(c, Some(a));
        assert_eq!(p.parented_count(), 2);
    }

    #[test]
    fn the_selected_row_is_marked() {
        let w = world_with_roots();
        let mut p = HierarchyPanel::new();
        let e = w.entities().next().unwrap();
        p.select(e);
        let rows = p.rows(&w);
        assert_eq!(rows.iter().filter(|r| r.selected).count(), 1);
    }

    #[test]
    fn a_tag_becomes_the_label() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Tag("Player".to_string()));
        assert_eq!(named_label(&w, e), "Player");
    }

    #[test]
    fn an_untagged_entity_is_labelled_by_index() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(named_label(&w, e).contains("Entity"));
    }

    #[test]
    fn an_empty_tag_falls_back_to_the_index() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Tag(String::new()));
        assert!(named_label(&w, e).contains("Entity"));
    }

    #[test]
    fn expanding_all_covers_every_entity() {
        let w = world_with_roots();
        let mut p = HierarchyPanel::new();
        p.expand_all(&w);
        for e in w.entities() {
            assert!(p.is_expanded(e), "{} was not expanded", e);
        }
    }
}
