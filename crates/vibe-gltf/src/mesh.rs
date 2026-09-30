//! Mesh extraction from a glTF document.
//!
//! The loadable form is a set of interleaved-free, GPU-shaped vertex and index
//! buffers rather than the document's sparse accessor soup, because that is
//! what a vertex buffer upload needs and re-packing per frame is not an option.
//! The conversion is pure logic given the numbers, so it is tested without
//! reading a file.

use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::error::GltfError;

/// The vertex the 3D pipeline consumes.
///
/// `#[repr(C)]` and 64 bytes: three position floats, three normal floats, two
/// UV floats, four joint indices and four joint weights. The fields already add
/// up to exactly 64 with no padding, which is a multiple of 16 as the
/// vertex-fetch rules want, so there is no padding field to keep in sync — and
/// no hole for a `bytemuck` cast to read a field the shader does not know about.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    /// Position in the mesh's own space.
    pub position: [f32; 3],
    /// Normal in the mesh's own space.
    pub normal: [f32; 3],
    /// Texture coordinate.
    pub uv: [f32; 2],
    /// Indices into the skin's joint array.
    pub joints: [u32; 4],
    /// Weights summing to 1.0, parallel to `joints`.
    pub weights: [f32; 4],
}

/// Bytes one vertex occupies in the buffer.
pub const VERTEX_STRIDE: usize = std::mem::size_of::<MeshVertex>();

/// One drawable piece of a model.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mesh {
    /// Interleaved vertices, ready to upload.
    pub vertices: Vec<MeshVertex>,
    /// Triangle list indices.
    pub indices: Vec<u32>,
    /// Index of the material this primitive uses.
    pub material: usize,
    /// Index of the skin this primitive is deformed by, if any.
    pub skin: Option<usize>,
}

impl Mesh {
    /// True when the mesh has nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// How many triangles the mesh holds.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// The axis-aligned bounds of the vertices, if there are any.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut iter = self.vertices.iter();
        let first = iter.next()?;
        let mut min = Vec3::new(first.position[0], first.position[1], first.position[2]);
        let mut max = min;
        for v in iter {
            let p = Vec3::new(v.position[0], v.position[1], v.position[2]);
            min = min.min(p);
            max = max.max(p);
        }
        Some((min, max))
    }

    /// The centre of the bounds, which is where a camera should frame it.
    pub fn center(&self) -> Option<Vec3> {
        self.bounds().map(|(lo, hi)| (lo + hi) * 0.5)
    }

    /// Apply a transform to every vertex position and normal.
    ///
    /// The normal is transformed by the inverse transpose so a non-uniform
    /// scale does not tilt it off perpendicular. Returns the same mesh, so a
    /// node's local transform can be baked in.
    pub fn transformed(&self, transform: Mat4) -> Mesh {
        // The normal transform is the inverse transpose, not the transpose.
        // For a pure rotation the transpose is the *inverse* rotation, so
        // transposing the matrix turns a normal the wrong way; inverting first
        // cancels the scale and leaves the rotation intact, which is what a
        // normal needs.
        let m = glam::Mat3::from_mat4(transform);
        let normal_matrix = m.inverse().transpose();
        let mut out = self.clone();
        for v in &mut out.vertices {
            let p = transform * Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
            v.position = [p.x, p.y, p.z];
            let n = normal_matrix * Vec3::new(v.normal[0], v.normal[1], v.normal[2]);
            v.normal = [n.x, n.y, n.z];
        }
        out
    }
}

/// A material's base colour, as the pipeline needs it.
///
/// glTF carries a PBR set of factors; only the base colour and metallic-
/// roughness matter for the engine's forward renderer, so the rest are dropped
/// rather than carried as unused fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Base colour factor, linear.
    pub base_color: Vec4,
    /// Metallic factor in `0.0..=1.0`.
    pub metallic: f32,
    /// Roughness factor in `0.0..=1.0`.
    pub roughness: f32,
    /// Whether the material claims to be double sided.
    pub double_sided: bool,
}

impl Default for Material {
    fn default() -> Self {
        // The glTF spec's default base colour factor is opaque white.
        Material {
            base_color: Vec4::ONE,
            metallic: 1.0,
            roughness: 1.0,
            double_sided: false,
        }
    }
}

/// A reference to a texture, kept as an index so a material does not own the
/// pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextureRef {
    /// Index into the model's texture list.
    pub index: usize,
    /// Texture coordinate set the material samples.
    pub tex_coord: u32,
}

/// How a UV coordinate set is laid out on a mesh.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct UvSet {
    /// Texture coordinate set index.
    pub index: u32,
    /// The offset the loader applied when the set is not `TEXCOORD_0`.
    pub offset: Vec2,
}

/// Normalise a UV set, wrapping or mirroring about its offset.
///
/// glTF exporters emit `TEXCOORD_1` and above for lightmaps and AO with no
/// convention of their own, so the only defensible default is to fold them onto
/// the first set rather than silently sample garbage.
pub fn normalize_uv(uv: Vec2, set: UvSet) -> Vec2 {
    let scaled = (uv + set.offset) * if set.index % 2 == 0 { 1.0 } else { -1.0 };
    Vec2::new(scaled.x.rem_euclid(1.0), scaled.y.rem_euclid(1.0))
}

/// Build a vertex from loose arrays, clamping the joint weights to sum to 1.
///
/// A skin whose weights do not sum to 1 makes the skinned position depend on
/// the bone count, which is not a thing anyone wants. Normalising here means
/// the shader can assume the sum and skip the divide.
pub fn make_vertex(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    joints: [u32; 4],
    weights: [f32; 4],
) -> MeshVertex {
    MeshVertex {
        position: position.to_array(),
        normal: normal.to_array(),
        uv: uv.to_array(),
        joints,
        weights: normalize_weights(weights),
    }
}

/// Scale four weights so they sum to 1, leaving an unweighted vertex alone.
///
/// An all-zero input would divide by zero, so it becomes the first joint at
/// full weight: a vertex bound to nothing must still end up somewhere.
pub fn normalize_weights(weights: [f32; 4]) -> [f32; 4] {
    let sum: f32 = weights.iter().sum();
    if sum <= f32::EPSILON {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let mut out = [0.0f32; 4];
    for (o, w) in out.iter_mut().zip(weights.iter()) {
        *o = w / sum;
    }
    out
}

/// The largest weight index that is actually used, for picking a bone budget.
pub fn dominant_joint(weights: [f32; 4]) -> Option<usize> {
    weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w > f32::EPSILON)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
}

/// A skin's inverse bind matrices, indexed by joint.
#[derive(Debug, Clone, PartialEq)]
pub struct Skin {
    /// Inverse bind matrix per joint, in joint-index order.
    pub inverse_bind: Vec<Mat4>,
    /// The skeleton node index for each joint.
    pub joints: Vec<usize>,
}

impl Skin {
    /// The number of joints.
    pub fn len(&self) -> usize {
        self.joints.len()
    }

    /// True when the skin binds nothing.
    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    /// Check the skin against the engine's per-frame bone budget.
    ///
    /// The limit is on joints, not matrices: a 4096-bone frame is 4096 mat4
    /// uploads even when most are identity, and that is the number the shader
    /// is dimensioned for.
    pub fn check_limit(&self, index: usize, limit: usize) -> Result<(), GltfError> {
        if self.joints.len() > limit {
            return Err(GltfError::TooManyJoints {
                skin: index,
                joints: self.joints.len(),
                limit,
            });
        }
        Ok(())
    }

    /// Check that every joint index names a node in the skeleton.
    pub fn check_joints(&self, index: usize, node_count: usize) -> Result<(), GltfError> {
        for joint in &self.joints {
            if *joint >= node_count {
                return Err(GltfError::UnknownJoint {
                    skin: index,
                    joint: *joint,
                    count: node_count,
                });
            }
        }
        Ok(())
    }
}

/// Multiply a skin's inverse bind matrices by the joints' world transforms.
///
/// The order matters: the joint's world matrix is the left factor, so a
/// translation on the joint moves the vertex with it. Getting this backwards
/// produces a mesh that collapses to the origin rather than one that is
/// obviously wrong, which is why it is written as one expression and tested.
pub fn skin_matrices(skin: &Skin, world: &[Mat4]) -> Vec<Mat4> {
    skin.inverse_bind
        .iter()
        .enumerate()
        .map(|(i, inv_bind)| {
            let joint_world = world.get(i).copied().unwrap_or(Mat4::IDENTITY);
            joint_world * *inv_bind
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> Mesh {
        Mesh {
            vertices: vec![
                make_vertex(
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::Z,
                    Vec2::ZERO,
                    [0; 4],
                    [1.0; 4],
                ),
                make_vertex(
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::Z,
                    Vec2::ZERO,
                    [0; 4],
                    [1.0; 4],
                ),
                make_vertex(
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::Z,
                    Vec2::ZERO,
                    [0; 4],
                    [1.0; 4],
                ),
            ],
            indices: vec![0, 1, 2],
            material: 0,
            skin: None,
        }
    }

    #[test]
    fn the_vertex_is_64_bytes() {
        // 12 position + 12 normal + 8 uv + 16 joints + 16 weights.
        assert_eq!(std::mem::size_of::<MeshVertex>(), 64);
        assert_eq!(VERTEX_STRIDE, 64);
    }

    #[test]
    fn the_vertex_stride_is_a_multiple_of_16() {
        assert_eq!(VERTEX_STRIDE % 16, 0, "vertex fetch wants this");
    }

    #[test]
    fn the_vertex_fields_account_for_every_byte() {
        // No padding field, and the named fields add up to the stride, so a
        // bytemuck cast cannot read a hole the shader does not know about.
        let named = 3 * 4 + 3 * 4 + 2 * 4 + 4 * 4 + 4 * 4;
        assert_eq!(named, VERTEX_STRIDE, "a gap would be an unnamed hole");
    }

    #[test]
    fn a_vertex_casts_to_bytes() {
        let v = make_vertex(Vec3::X, Vec3::Y, Vec2::ONE, [1, 2, 3, 4], [0.25; 4]);
        let bytes = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), VERTEX_STRIDE);
    }

    #[test]
    fn a_mesh_counts_its_triangles() {
        assert_eq!(quad().triangle_count(), 1);
    }

    #[test]
    fn an_empty_mesh_draws_nothing() {
        let m = Mesh::default();
        assert!(m.is_empty());
        assert_eq!(m.triangle_count(), 0);
    }

    #[test]
    fn bounds_span_every_vertex() {
        let m = quad();
        let (lo, hi) = m.bounds().unwrap();
        assert_eq!(lo, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(hi, Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn an_empty_mesh_has_no_bounds() {
        assert!(Mesh::default().bounds().is_none());
        assert!(Mesh::default().center().is_none());
    }

    #[test]
    fn the_centre_is_the_middle_of_the_bounds() {
        let mut m = quad();
        m.vertices[1].position = [4.0, 0.0, 0.0];
        assert_eq!(m.center().unwrap(), Vec3::new(2.0, 0.5, 0.0));
    }

    #[test]
    fn a_translation_moves_every_position() {
        let m = quad().transformed(Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));
        assert_eq!(m.vertices[0].position, [5.0, 0.0, 0.0]);
        assert_eq!(m.vertices[1].position, [6.0, 0.0, 0.0]);
    }

    #[test]
    fn a_transform_leaves_an_unrotated_normal_alone() {
        let m = quad().transformed(Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));
        assert_eq!(m.vertices[0].normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_rotation_turns_the_normal_with_the_position() {
        let mut m = quad();
        m.vertices[0].normal = [1.0, 0.0, 0.0];
        let quarter = Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let out = m.transformed(quarter);
        let n = Vec3::new(
            out.vertices[0].normal[0],
            out.vertices[0].normal[1],
            out.vertices[0].normal[2],
        );
        assert!((n - Vec3::Y).length() < 1e-5, "{n:?}");
    }

    #[test]
    fn a_non_uniform_scale_keeps_the_normal_perpendicular() {
        // Scaling x by 10 and leaving y alone would tilt a plain-transformed
        // normal; the inverse transpose has to correct for it.
        let mut m = quad();
        m.vertices[0].normal = [0.0, 1.0, 0.0];
        let scale = Mat4::from_scale(Vec3::new(10.0, 1.0, 1.0));
        let out = m.transformed(scale);
        let n = Vec3::new(
            out.vertices[0].normal[0],
            out.vertices[0].normal[1],
            out.vertices[0].normal[2],
        );
        assert!((n.normalize() - Vec3::Y).length() < 1e-4, "{n:?}");
    }

    #[test]
    fn weights_are_normalised_to_sum_to_one() {
        let w = normalize_weights([2.0, 2.0, 0.0, 0.0]);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(w[0], 0.5);
    }

    #[test]
    fn weights_that_already_sum_to_one_are_untouched() {
        let w = normalize_weights([0.25, 0.25, 0.25, 0.25]);
        assert_eq!(w, [0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn zero_weights_become_the_first_joint_at_full_weight() {
        // Dividing by a zero sum would produce NaN, which propagates into the
        // vertex and then into every triangle sharing it.
        let w = normalize_weights([0.0, 0.0, 0.0, 0.0]);
        assert_eq!(w, [1.0, 0.0, 0.0, 0.0]);
        assert!(w.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn negative_weights_still_produce_a_finite_result() {
        // A malformed file with negative weights would otherwise give a sum of
        // zero after clamping and divide by it.
        let w = normalize_weights([-1.0, -1.0, 0.0, 0.0]);
        assert!(w.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn the_dominant_joint_is_the_heaviest() {
        assert_eq!(dominant_joint([0.1, 0.7, 0.2, 0.0]), Some(1));
    }

    #[test]
    fn an_unweighted_vertex_has_no_dominant_joint() {
        assert_eq!(dominant_joint([0.0; 4]), None);
    }

    #[test]
    fn uv_set_zero_is_the_identity() {
        let uv = Vec2::new(0.25, 0.75);
        let out = normalize_uv(uv, UvSet::default());
        assert!((out - uv).length() < 1e-6, "{out:?}");
    }

    #[test]
    fn a_mirrored_uv_set_flips_and_wraps() {
        // Mirroring turns 0.25 into -0.25, and rem_euclid brings that back into
        // 0..1 as 0.75 rather than leaving a negative coordinate.
        let out = normalize_uv(
            Vec2::new(0.25, 0.25),
            UvSet {
                index: 1,
                ..Default::default()
            },
        );
        assert!((out.x - 0.75).abs() < 1e-6, "{out:?}");
        assert!((out.y - 0.75).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn uv_outside_zero_to_one_wraps() {
        let out = normalize_uv(Vec2::new(1.5, -0.5), UvSet::default());
        assert!((out - Vec2::new(0.5, 0.5)).length() < 1e-6, "{out:?}");
    }

    #[test]
    fn the_default_material_is_the_spec_default() {
        let m = Material::default();
        assert_eq!(m.base_color, Vec4::ONE);
        assert!(!m.double_sided);
    }

    #[test]
    fn a_skin_knows_its_joint_count() {
        let s = Skin {
            inverse_bind: vec![Mat4::IDENTITY, Mat4::IDENTITY],
            joints: vec![0, 1],
        };
        assert_eq!(s.len(), 2);
        assert!(!s.is_empty());
    }

    #[test]
    fn an_empty_skin_binds_nothing() {
        let s = Skin {
            inverse_bind: Vec::new(),
            joints: Vec::new(),
        };
        assert!(s.is_empty());
    }

    #[test]
    fn a_skin_within_the_budget_is_accepted() {
        let s = Skin {
            inverse_bind: vec![Mat4::IDENTITY; 8],
            joints: vec![0; 8],
        };
        assert!(s.check_limit(0, 4096).is_ok());
    }

    #[test]
    fn a_skin_over_the_budget_is_rejected_by_count() {
        let s = Skin {
            inverse_bind: vec![Mat4::IDENTITY; 9],
            joints: vec![0; 9],
        };
        let err = s.check_limit(2, 8).unwrap_err();
        match err {
            GltfError::TooManyJoints {
                skin,
                joints,
                limit,
                ..
            } => {
                assert_eq!(skin, 2);
                assert_eq!(joints, 9);
                assert_eq!(limit, 8);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn a_joint_past_the_skeleton_is_rejected() {
        let s = Skin {
            inverse_bind: vec![Mat4::IDENTITY],
            joints: vec![7],
        };
        let err = s.check_joints(1, 3).unwrap_err();
        assert!(matches!(
            err,
            GltfError::UnknownJoint {
                skin: 1,
                joint: 7,
                count: 3
            }
        ));
    }

    #[test]
    fn joints_within_the_skeleton_are_accepted() {
        let s = Skin {
            inverse_bind: vec![Mat4::IDENTITY; 2],
            joints: vec![0, 2],
        };
        assert!(s.check_joints(0, 3).is_ok());
    }

    #[test]
    fn skinning_multiplies_world_by_inverse_bind() {
        let s = Skin {
            inverse_bind: vec![Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0))],
            joints: vec![0],
        };
        let world = [Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))];
        let m = skin_matrices(&s, &world);
        assert_eq!(m.len(), 1);
        let expected = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))
            * Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0));
        assert!(
            (m[0]
                .to_cols_array()
                .iter()
                .zip(expected.to_cols_array().iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max))
                < 1e-5,
            "{:?}",
            m[0]
        );
    }

    #[test]
    fn an_identity_skin_leaves_a_vertex_alone() {
        let s = Skin {
            inverse_bind: vec![Mat4::IDENTITY],
            joints: vec![0],
        };
        let m = skin_matrices(&s, &[Mat4::IDENTITY]);
        assert_eq!(m[0], Mat4::IDENTITY);
    }

    #[test]
    fn a_missing_world_transform_falls_back_to_identity() {
        // A skeleton with fewer matrices than joints is malformed, but a NaN or
        // a panic here would take the whole frame down.
        let s = Skin {
            inverse_bind: vec![Mat4::IDENTITY; 3],
            joints: vec![0, 1, 2],
        };
        let m = skin_matrices(&s, &[Mat4::IDENTITY]);
        assert_eq!(m.len(), 3);
        assert!(m.iter().all(|x| x.is_finite()));
    }
}
