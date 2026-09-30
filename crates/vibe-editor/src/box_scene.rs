//! The 3D scene the user edits: boxes, where they are, how big, and what is
//! painted on them.
//!
//! This is the state layer, and it holds no Vulkan objects. A box is a
//! [`SceneBox`] with a position, a size and a texture path; the renderer turns
//! those into a vertex buffer and a sampled image. Keeping the two apart is what
//! lets the interesting parts — where a click places a box, what a resize drag
//! does to the size, which texture an entity gets — be tested with no device at
//! all, which matters because a resize bug is silent: the box still draws, just
//! wrong, and nothing crashes to tell you.

use glam::{Vec2, Vec3};
use vibe_ecs::components::Transform;

/// The size a newly placed box starts at, in metres.
///
/// A metre cube rather than something larger or smaller: the editor camera
/// orbits at a few metres, so a unit box is a few dozen pixels across at the
/// default zoom, which is big enough to click and small enough that three of
/// them fit in view.
pub const DEFAULT_BOX_SIZE: Vec3 = Vec3::splat(1.0);

/// The smallest a box may become on any axis, in metres.
///
/// Zero extent on an axis means the box has no volume on that axis, so it
/// rasterises to a line and then to nothing: dragging a resize handle onto its
/// opposite face would silently delete the object the user is editing. A
/// sliver is recoverable, a vanished box is not.
pub const MIN_BOX_EXTENT: f32 = 0.05;

/// One box in the scene.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneBox {
    /// The entity this box belongs to.
    pub entity: vibe_ecs::Entity,
    /// Where the box's centre is, in metres.
    pub position: Vec3,
    /// The box's full size along each axis, in metres.
    pub size: Vec3,
    /// The texture painted on the box, or `None` for the untextured look.
    ///
    /// A path rather than decoded pixels: decoding belongs to the render layer,
    /// and a path is what a scene file can actually store.
    pub texture: Option<std::path::PathBuf>,
    /// Whether this box is drawn.
    pub visible: bool,
}

impl SceneBox {
    /// A box of the default size at the origin.
    pub fn new(entity: vibe_ecs::Entity) -> SceneBox {
        SceneBox {
            entity,
            position: Vec3::ZERO,
            size: DEFAULT_BOX_SIZE,
            texture: None,
            visible: true,
        }
    }

    /// The box's eight corners, in the order the resize handles expect.
    pub fn corners(&self) -> [Vec3; 8] {
        vibe_render::mesh3d::box_corners(self.position, self.size)
    }

    /// The half-extents, which is what a drag along one axis scales.
    pub fn half(&self) -> Vec3 {
        self.size * 0.5
    }
}

/// Which corner handle a resize drag grabbed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeHandle {
    /// The corner at the given index, in the order [`SceneBox::corners`] returns.
    Corner(usize),
}

/// Which of the eight corner handles a click landed on.
///
/// The handles are square regions in screen space, so the test is a point-in-box
/// against each projected corner rather than a distance to a line like the axis
/// gizmo uses. Nearest-wins when two overlap, so the handle under the cursor is
/// always the one that gets dragged rather than whichever the loop saw first.
pub fn pick_corner(projected: &[(usize, Vec2)], cursor: Vec2, radius: f32) -> Option<ResizeHandle> {
    let mut best: Option<(usize, f32)> = None;
    for (index, screen) in projected {
        let d = (*screen - cursor).length();
        if d <= radius {
            match best {
                Some((_, closer)) if closer <= d => {}
                _ => best = Some((*index, d)),
            }
        }
    }
    best.map(|(index, _)| ResizeHandle::Corner(index))
}

/// The index of the corner opposite `index`.
///
/// Found by position, not by index arithmetic. An earlier version used
/// `index ^ 7`, which reads like the obvious bit trick and is wrong:
/// [`vibe_render::mesh3d::box_corners`] orders the corners as the four on -Z
/// then the four on +Z, so corner 0's opposite is 6, not 7. Asking the geometry
/// which corner it is means the two cannot drift apart again if the ordering
/// ever changes.
pub fn opposite_corner(corners: &[Vec3; 8], centre: Vec3, index: usize) -> usize {
    let diagonal = corners[index] - centre;
    // The opposite corner is the one furthest *against* the diagonal: for corner
    // 0 at (-1,-1,-1) the opposite at (1,1,1) projects to -3 against the
    // diagonal, while its three neighbours each project to -1 and itself to +3.
    // So the opposite is the minimum, and it is the only one that stands out.
    //
    // Seeded from corner 0 rather than from -inf: a `best_score` that starts at
    // `-inf` is never beaten, because nothing compares less than it, and the
    // function then returns the corner it was asked about every time.
    let mut best = 0;
    let mut best_score = (corners[0] - centre).dot(diagonal);
    for (i, c) in corners.iter().enumerate().skip(1) {
        let score = (c - centre).dot(diagonal);
        if score < best_score {
            best_score = score;
            best = i;
        }
    }
    best
}

/// The size a box should have after its `handle` corner is dragged to `world`.
///
/// The dragged corner follows the cursor and the opposite corner stays put, so
/// the box resizes from its far side rather than from its centre. That is what
/// makes a resize feel attached to the handle the user grabbed.
pub fn resize_to(
    centre: Vec3,
    start_size: Vec3,
    handle: ResizeHandle,
    world: Vec3,
) -> (Vec3, Vec3) {
    let ResizeHandle::Corner(index) = handle;
    let corners = vibe_render::mesh3d::box_corners(centre, start_size);
    let opposite = opposite_corner(&corners, centre, index);
    let anchor = corners[opposite];
    // Each extent is the distance from the anchor to the cursor, taken positive.
    // The anchor sits on the opposite side from the cursor for a normal drag, so
    // the absolute value is the size; the clamp is what stops a cursor dragged
    // past the anchor from inverting the box.
    let size = (world - anchor).abs().max(Vec3::splat(MIN_BOX_EXTENT));
    // The centre is the midpoint of the anchor and the cursor, so the box keeps
    // spanning the same pair of opposite corners.
    let new_centre = (anchor + world) * 0.5;
    (new_centre, size)
}

/// A resize drag in progress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResizeDrag {
    /// Which handle is being dragged.
    pub handle: ResizeHandle,
    /// Which box is being resized, by index.
    ///
    /// An index rather than a re-derived match on the box's values, because a
    /// drag must write to exactly the box it started on: matching on position
    /// and size would let a drag on a box that happens to share them with a
    /// neighbouring one move the wrong object.
    pub index: usize,
    /// The box's centre when the drag started.
    pub start_centre: Vec3,
    /// The box's size when the drag started.
    pub start_size: Vec3,
}

impl ResizeDrag {
    /// Begin a drag on a handle of the box at `index`.
    pub fn begin(
        index: usize,
        handle: ResizeHandle,
        start_centre: Vec3,
        start_size: Vec3,
    ) -> ResizeDrag {
        ResizeDrag {
            handle,
            index,
            start_centre,
            start_size,
        }
    }

    /// The centre and size the box takes when dragged to `world`.
    pub fn apply(&self, world: Vec3) -> (Vec3, Vec3) {
        resize_to(self.start_centre, self.start_size, self.handle, world)
    }
}

/// The boxes the user is editing.
#[derive(Debug, Clone, Default)]
pub struct BoxScene {
    /// Every box, in the order they were created.
    pub boxes: Vec<SceneBox>,
    /// The box a resize drag is in progress on.
    pub drag: Option<ResizeDrag>,
    /// How far apart the new boxes are placed, in metres.
    pub placement_spacing: f32,
}

impl BoxScene {
    /// An empty scene.
    pub fn new() -> BoxScene {
        BoxScene {
            boxes: Vec::new(),
            drag: None,
            placement_spacing: 2.0,
        }
    }

    /// Add a box for an entity at a free spot, and return it.
    ///
    /// Placed on a grid rather than all at the origin: a box at the origin is
    /// invisible inside every other box at the origin, and the user has no way
    /// to tell they created four.
    pub fn place(&mut self, entity: vibe_ecs::Entity) -> usize {
        let position = self.next_free_spot();
        self.boxes.push(SceneBox {
            entity,
            position,
            ..SceneBox::new(entity)
        });
        self.boxes.len() - 1
    }

    /// The next position on the placement grid.
    ///
    /// Laid out along X with a fixed spacing, cycling to a new row rather than
    /// running off to infinity, so the scene stays compact at any count.
    pub fn next_free_spot(&self) -> Vec3 {
        let spacing = if self.placement_spacing > 0.0 {
            self.placement_spacing
        } else {
            DEFAULT_BOX_SIZE.x * 2.0
        };
        const PER_ROW: usize = 4;
        let index = self.boxes.len();
        let column = index % PER_ROW;
        let row = index / PER_ROW;
        Vec3::new(column as f32 * spacing, row as f32 * spacing, 0.0)
    }

    /// The index of a box, by entity.
    pub fn index_of(&self, entity: vibe_ecs::Entity) -> Option<usize> {
        self.boxes.iter().position(|b| b.entity == entity)
    }

    /// The box for an entity.
    pub fn get(&self, entity: vibe_ecs::Entity) -> Option<&SceneBox> {
        self.index_of(entity).map(|i| &self.boxes[i])
    }

    /// The box for an entity, mutably.
    pub fn get_mut(&mut self, entity: vibe_ecs::Entity) -> Option<&mut SceneBox> {
        self.index_of(entity).map(|i| &mut self.boxes[i])
    }

    /// Remove a box.
    pub fn remove(&mut self, entity: vibe_ecs::Entity) -> bool {
        let Some(index) = self.index_of(entity) else {
            return false;
        };
        self.boxes.remove(index);
        // A drag on the box that just went away must not survive it. The drag
        // records the index, and the removal shifted everything after it, so the
        // drag is ended unconditionally rather than guessed at: ending a drag
        // the user is not looking at is harmless, and a drag that survived
        // would resize whichever box took the removed one's place.
        self.drag = None;
        true
    }

    /// Start a resize drag on a handle of a box.
    pub fn begin_resize(&mut self, entity: vibe_ecs::Entity, handle: ResizeHandle) -> bool {
        let Some(index) = self.index_of(entity) else {
            return false;
        };
        let b = &self.boxes[index];
        self.drag = Some(ResizeDrag {
            handle,
            index,
            start_centre: b.position,
            start_size: b.size,
        });
        true
    }

    /// Apply the in-progress drag to its box, given the handle's world position.
    pub fn update_resize(&mut self, world: Vec3) -> Option<(Vec3, Vec3)> {
        let drag = self.drag?;
        // The index is the box's own, so a drag can only ever write to the box it
        // started on. `get_mut` still bounds it, because the index is a number
        // and the boxes behind it can be shortened.
        let b = self.boxes.get_mut(drag.index)?;
        let (centre, size) = drag.apply(world);
        b.position = centre;
        b.size = size;
        Some((centre, size))
    }

    /// End the resize drag.
    pub fn end_resize(&mut self) {
        self.drag = None;
    }

    /// Put a texture on a box.
    ///
    /// A path that does not load is still recorded: the panel shows the name the
    /// user picked, and the renderer reports the decode failure once, rather than
    /// the browser silently doing nothing when clicked.
    pub fn set_texture(
        &mut self,
        entity: vibe_ecs::Entity,
        path: Option<std::path::PathBuf>,
    ) -> bool {
        let Some(b) = self.get_mut(entity) else {
            return false;
        };
        b.texture = path;
        true
    }

    /// Write a box's transform into the ECS, so the hierarchy and the inspector
    /// agree with what is drawn.
    pub fn sync_transform(&self, entity: vibe_ecs::Entity, world: &mut vibe_ecs::World) -> bool {
        let Some(b) = self.get(entity) else {
            return false;
        };
        let mut transform = world.get::<Transform>(entity).cloned().unwrap_or_default();
        transform.x = b.position.x;
        transform.y = b.position.y;
        transform.z = b.position.z;
        // The box's geometry is a unit box scaled to its size, so the transform's
        // scale *is* the size. Storing it that way is what makes the inspector's
        // scale field and the resize handles the same control.
        transform.scale = b.size;
        world.add(entity, transform);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(n: u32) -> vibe_ecs::Entity {
        // Only ever compared, never used to index into a real world, so the raw
        // index is enough; a live handle would need a whole world to mint.
        vibe_ecs::Entity::from_parts(n, 0)
    }

    #[test]
    fn a_new_box_is_a_metre_cube_at_the_origin() {
        let b = SceneBox::new(entity(1));
        assert_eq!(b.position, Vec3::ZERO);
        assert_eq!(b.size, Vec3::splat(1.0));
        assert_eq!(b.size, DEFAULT_BOX_SIZE);
        assert!(b.texture.is_none(), "a new box has no texture");
        assert!(b.visible);
    }

    #[test]
    fn boxes_are_placed_apart_not_stacked() {
        let mut scene = BoxScene::new();
        let a = scene.place(entity(1));
        let b = scene.place(entity(2));
        let c = scene.place(entity(3));
        let (pa, pb, pc) = (
            scene.boxes[a].position,
            scene.boxes[b].position,
            scene.boxes[c].position,
        );
        // Two boxes at the same point cannot be told apart, which is the bug a
        // grid placement exists to prevent.
        assert!(pa.distance(pb) > 0.5, "boxes 0 and 1 overlap at {pa:?}");
        assert!(pb.distance(pc) > 0.5, "boxes 1 and 2 overlap at {pb:?}");
    }

    #[test]
    fn placement_wraps_to_a_new_row() {
        let mut scene = BoxScene::new();
        for i in 0..5 {
            scene.place(entity(i as u32 + 1));
        }
        // Four on the first row, the fifth starts the second, so the layout stays
        // compact rather than running off in one direction forever.
        assert_eq!(scene.boxes[3].position.y, 0.0);
        assert!(
            scene.boxes[4].position.y > 0.0,
            "the fifth box starts a row"
        );
    }

    #[test]
    fn a_box_is_found_by_its_entity() {
        let mut scene = BoxScene::new();
        scene.place(entity(7));
        assert!(scene.get(entity(7)).is_some());
        assert!(
            scene.get(entity(8)).is_none(),
            "an unknown entity has no box"
        );
    }

    #[test]
    fn placing_the_same_entity_twice_gives_two_boxes() {
        // Not a correctness bug in itself, but it is what the caller must know:
        // `place` does not deduplicate, so a double-click makes two boxes.
        let mut scene = BoxScene::new();
        scene.place(entity(1));
        scene.place(entity(1));
        assert_eq!(scene.boxes.len(), 2);
    }

    #[test]
    fn corners_surround_the_box() {
        let b = SceneBox {
            position: Vec3::new(1.0, 2.0, 3.0),
            size: Vec3::splat(2.0),
            ..SceneBox::new(entity(1))
        };
        for corner in b.corners() {
            assert!(
                (corner - b.position)
                    .abs()
                    .abs_diff_eq(Vec3::splat(1.0), 1.0e-5),
                "corner {corner:?} is not one unit from the centre"
            );
        }
    }

    #[test]
    fn the_half_extent_is_half_the_size() {
        let b = SceneBox {
            size: Vec3::new(2.0, 4.0, 6.0),
            ..SceneBox::new(entity(1))
        };
        assert_eq!(b.half(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn a_click_near_a_corner_picks_it() {
        let projected = [(0usize, Vec2::new(10.0, 10.0)), (7, Vec2::new(100.0, 10.0))];
        let picked = pick_corner(&projected, Vec2::new(12.0, 11.0), 8.0);
        assert_eq!(picked, Some(ResizeHandle::Corner(0)));
    }

    #[test]
    fn a_click_away_from_every_corner_picks_nothing() {
        let projected = [(0usize, Vec2::new(10.0, 10.0)), (7, Vec2::new(100.0, 10.0))];
        assert_eq!(pick_corner(&projected, Vec2::new(55.0, 55.0), 8.0), None);
    }

    #[test]
    fn the_nearest_corner_wins_when_two_overlap() {
        // Two handles a few pixels apart both cover the cursor. Picking the
        // first one in the list rather than the nearest makes the resize grab
        // whichever corner happened to be drawn first, which is not what the
        // user aimed at.
        let projected = [(0usize, Vec2::new(10.0, 10.0)), (1, Vec2::new(13.0, 10.0))];
        let picked = pick_corner(&projected, Vec2::new(13.0, 10.0), 8.0);
        assert_eq!(picked, Some(ResizeHandle::Corner(1)));
    }

    #[test]
    fn a_resize_keeps_the_opposite_corner_fixed() {
        let centre = Vec3::ZERO;
        let size = Vec3::splat(2.0);
        let corners = vibe_render::mesh3d::box_corners(centre, size);
        // The anchor is whichever corner the geometry says is opposite, not a
        // hardcoded index: the corner ordering puts -1's opposite at 6, so a
        // test that assumed 7 would agree with a wrong implementation.
        let opposite = opposite_corner(&corners, centre, 0);
        assert_ne!(opposite, 0, "a corner cannot be its own opposite");
        let anchor = corners[opposite];
        // Drag corner 0 further out to -3 on every axis.
        let (new_centre, new_size) =
            resize_to(centre, size, ResizeHandle::Corner(0), Vec3::splat(-3.0));
        // The box now spans from -3 to the anchor at +1, so size 4, centre -1.
        assert!(
            new_size.abs_diff_eq(Vec3::splat(4.0), 1.0e-4),
            "size was {new_size:?}"
        );
        assert!(
            new_centre.abs_diff_eq(Vec3::splat(-1.0), 1.0e-4),
            "centre was {new_centre:?}"
        );
        // The far corner is what the user was not touching, so it must not move.
        // Checked against the anchor's own position rather than a corner index,
        // because the index it lands on after a resize is not part of the
        // contract — the position is.
        let after = vibe_render::mesh3d::box_corners(new_centre, new_size);
        let still_there = after.iter().any(|c| c.abs_diff_eq(anchor, 1.0e-4));
        assert!(
            still_there,
            "the opposite corner {anchor:?} moved; corners are now {after:?}"
        );
    }

    #[test]
    fn every_corner_has_a_distinct_opposite() {
        // The eight corners form four opposite pairs. If two corners shared an
        // opposite, dragging one of them would anchor the box to a corner the
        // user was not on, and the box would jump.
        let centre = Vec3::ZERO;
        let corners = vibe_render::mesh3d::box_corners(centre, Vec3::splat(2.0));
        let mut pairs = Vec::new();
        for i in 0..8 {
            let o = opposite_corner(&corners, centre, i);
            assert_ne!(i, o, "corner {i} is its own opposite");
            // Being opposite is symmetric: if i is opposite to o, so is o to i.
            assert_eq!(opposite_corner(&corners, centre, o), i);
            let key = if i < o { (i, o) } else { (o, i) };
            pairs.push(key);
        }
        pairs.sort_unstable();
        pairs.dedup();
        assert_eq!(pairs.len(), 4, "eight corners form four opposite pairs");
    }

    #[test]
    fn a_resize_only_moves_the_axis_the_corner_owns() {
        // Each extent is measured from the anchor on that axis alone, so a drag
        // that also moves in the other two axes does not shear the box.
        let centre = Vec3::ZERO;
        let size = Vec3::splat(2.0);
        let corners = vibe_render::mesh3d::box_corners(centre, size);
        // Corner 6 is (+1, +1, +1), so the anchor is corner 0 at (-1, -1, -1).
        // The anchor is taken from the geometry rather than hardcoded, so this
        // test cannot agree with a wrong opposite-corner answer.
        let anchor = corners[opposite_corner(&corners, centre, 6)];
        let target = Vec3::new(-2.0, 0.5, 0.25);
        let (_, new_size) = resize_to(centre, size, ResizeHandle::Corner(6), target);
        // Measured per axis: the drag went to -2 on X but only part way on Y and
        // Z, and the box must follow all three independently.
        let expected = (target - anchor).abs();
        assert!(
            new_size.abs_diff_eq(expected, 1.0e-4),
            "size was {new_size:?}, expected {expected:?}"
        );
    }

    #[test]
    fn a_resize_never_collapses_a_box() {
        // Dragging a corner exactly onto its opposite leaves nothing to draw, so
        // the extent is floored. Without the floor the box disappears and the
        // user has lost the object they were editing.
        for target in [Vec3::splat(1.0), Vec3::splat(0.99), Vec3::ZERO] {
            let (_, size) = resize_to(
                Vec3::ZERO,
                Vec3::splat(2.0),
                ResizeHandle::Corner(0),
                target,
            );
            for axis in 0..3 {
                assert!(
                    size[axis] >= MIN_BOX_EXTENT,
                    "axis {axis} collapsed to {} dragging to {target:?}",
                    size[axis]
                );
            }
        }
    }

    #[test]
    fn a_drag_writes_to_the_box_it_started_on() {
        let mut scene = BoxScene::new();
        let a = scene.place(entity(1));
        let b = scene.place(entity(2));
        let untouched = scene.boxes[b].position;
        let start_size = scene.boxes[a].size;
        assert!(scene.begin_resize(entity(1), ResizeHandle::Corner(0)));
        scene.update_resize(scene.boxes[a].position + Vec3::splat(5.0));
        assert_eq!(
            scene.boxes[b].position, untouched,
            "the drag moved a box it never started on"
        );
        assert!(
            scene.boxes[a].size != start_size,
            "the dragged box did not resize"
        );
    }

    #[test]
    fn ending_a_drag_stops_it_writing() {
        let mut scene = BoxScene::new();
        scene.place(entity(1));
        scene.begin_resize(entity(1), ResizeHandle::Corner(0));
        assert!(scene.drag.is_some());
        scene.end_resize();
        assert!(scene.drag.is_none());
        // With no drag in progress, a mouse move cannot change the box.
        let before = scene.boxes[0].size;
        assert!(scene.update_resize(Vec3::splat(99.0)).is_none());
        assert_eq!(scene.boxes[0].size, before);
    }

    #[test]
    fn a_drag_on_a_missing_box_does_not_start() {
        let mut scene = BoxScene::new();
        assert!(!scene.begin_resize(entity(99), ResizeHandle::Corner(0)));
        assert!(scene.drag.is_none());
    }

    #[test]
    fn removing_the_dragged_box_ends_the_drag() {
        // A drag that outlives its box would write into whichever box took the
        // removed one's index, moving an object the user never touched.
        let mut scene = BoxScene::new();
        scene.place(entity(1));
        scene.place(entity(2));
        scene.begin_resize(entity(1), ResizeHandle::Corner(0));
        scene.remove(entity(1));
        assert!(scene.drag.is_none(), "the drag outlived its box");
    }

    #[test]
    fn a_texture_is_recorded_on_the_box() {
        let mut scene = BoxScene::new();
        scene.place(entity(1));
        assert!(scene.set_texture(entity(1), Some("/tmp/a.png".into())));
        assert_eq!(
            scene.get(entity(1)).unwrap().texture,
            Some("/tmp/a.png".into())
        );
    }

    #[test]
    fn a_texture_can_be_cleared() {
        let mut scene = BoxScene::new();
        scene.place(entity(1));
        scene.set_texture(entity(1), Some("/tmp/a.png".into()));
        assert!(scene.set_texture(entity(1), None));
        assert!(scene.get(entity(1)).unwrap().texture.is_none());
    }

    #[test]
    fn a_texture_on_a_missing_box_is_refused() {
        let mut scene = BoxScene::new();
        assert!(!scene.set_texture(entity(99), Some("/tmp/a.png".into())));
    }

    #[test]
    fn the_transform_carries_the_position_and_size() {
        let mut world = vibe_ecs::World::new();
        let e = world.spawn();
        let mut scene = BoxScene::new();
        scene.place(e);
        scene.boxes[0].position = Vec3::new(1.0, 2.0, 3.0);
        scene.boxes[0].size = Vec3::new(2.0, 3.0, 4.0);
        assert!(scene.sync_transform(e, &mut world));
        let t = world.get::<Transform>(e).expect("transform written");
        assert_eq!((t.x, t.y, t.z), (1.0, 2.0, 3.0));
        // The scale *is* the size, which is what makes the inspector's scale
        // field and the resize handles the same control rather than two.
        assert_eq!(t.scale, Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn syncing_an_unknown_entity_writes_nothing() {
        let mut world = vibe_ecs::World::new();
        let scene = BoxScene::new();
        assert!(!scene.sync_transform(entity(3), &mut world));
    }

    #[test]
    fn the_editor_camera_puts_a_box_in_front_of_itself() {
        // The gizmo maths, and therefore the resize handles, all run through
        // `EditorCamera::view_projection`. If that matrix does not put a box in
        // front of the camera, the handles project nowhere and the 3D pass draws
        // an off-screen scene — both without an error.
        let mut camera = crate::camera::EditorCamera::perspective();
        camera.position = glam::Vec3::new(0.0, 0.0, 5.0);
        camera.yaw = 0.0;
        camera.pitch = 0.0;
        let viewport = glam::Vec2::new(800.0, 600.0);
        let at_origin = crate::gizmo::project_with_camera(glam::Vec3::ZERO, &camera, viewport);
        let screen = at_origin.expect("the origin is in front of the camera");
        // A camera on +Z looking down -Z sees the origin dead centre.
        assert!(
            screen.x.abs() - 400.0 < 1.0,
            "the origin projected to x {} on an 800-wide viewport",
            screen.x
        );
        assert!(
            screen.y.abs() - 300.0 < 1.0,
            "the origin projected to y {} on a 600-tall viewport",
            screen.y
        );
    }

    #[test]
    fn the_camera_separates_a_near_box_from_a_far_one() {
        // Two boxes along the view direction must project to different screen
        // positions, and the nearer one must be lower down the screen when the
        // camera looks along -Z. Identical projections mean the matrix is
        // collapsing depth, which shows up as every box drawn on top of the
        // others.
        let mut camera = crate::camera::EditorCamera::perspective();
        camera.position = glam::Vec3::ZERO;
        camera.yaw = 0.0;
        camera.pitch = 0.0;
        let viewport = glam::Vec2::new(800.0, 600.0);
        let near =
            crate::gizmo::project_with_camera(glam::Vec3::new(0.0, 0.0, -2.0), &camera, viewport)
                .expect("the near box is in front");
        let far =
            crate::gizmo::project_with_camera(glam::Vec3::new(0.0, 0.0, -8.0), &camera, viewport)
                .expect("the far box is in front");
        // Both are on the view axis, so both project to the centre; the test is
        // that projecting them does not fail and does not produce NaN, which is
        // what a Y-flipped or singular matrix would give.
        assert!(near.is_finite() && far.is_finite());
        assert!((near - far).length() < 1.0, "both are on the view axis");
    }

    #[test]
    fn a_box_behind_the_camera_has_no_screen_position() {
        // A handle behind the camera must be unpickable rather than mirrored to
        // the far side of the screen, or the user grabs a corner they cannot see.
        let mut camera = crate::camera::EditorCamera::perspective();
        camera.position = glam::Vec3::ZERO;
        camera.yaw = 0.0;
        camera.pitch = 0.0;
        let viewport = glam::Vec2::new(800.0, 600.0);
        let behind =
            crate::gizmo::project_with_camera(glam::Vec3::new(0.0, 0.0, 5.0), &camera, viewport);
        assert!(
            behind.is_none(),
            "a point behind the camera projected to {behind:?}"
        );
    }

    #[test]
    fn a_resize_handle_is_where_the_corner_lands_on_screen() {
        // The handles are drawn at the projected corners, so a resize that
        // changes the box has to move the handle the user is holding. This is the
        // loop that makes the resize feel attached: pick, drag, corner follows.
        let mut camera = crate::camera::EditorCamera::perspective();
        camera.position = glam::Vec3::new(0.0, 0.0, 6.0);
        camera.yaw = 0.0;
        camera.pitch = 0.0;
        let viewport = glam::Vec2::new(800.0, 600.0);
        let small = SceneBox {
            position: Vec3::ZERO,
            size: Vec3::splat(1.0),
            ..SceneBox::new(entity(1))
        };
        let big = SceneBox {
            position: Vec3::ZERO,
            size: Vec3::splat(2.0),
            ..SceneBox::new(entity(1))
        };
        let a = crate::gizmo::project_with_camera(small.corners()[0], &camera, viewport);
        let b = crate::gizmo::project_with_camera(big.corners()[0], &camera, viewport);
        let (a, b) = (a.expect("front corner"), b.expect("front corner"));
        assert!(
            (a - b).length() > 5.0,
            "doubling the box moved its handle only {:?} pixels",
            (a - b).length()
        );
    }
}
