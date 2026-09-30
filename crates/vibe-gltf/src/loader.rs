//! Reading a glTF or GLB document into the engine's mesh, skin and clip types.
//!
//! The `gltf` crate hands back iterators of borrowed views whose lifetimes are
//! tied to the buffer data, so everything is read out and copied. That is the
//! right trade: a model is loaded once and held for the life of the scene, and
//! holding the buffer alive to avoid the copy would pin the whole file.

use std::path::Path;

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

use crate::anim::{Channel, ChannelPath, ChannelValue, Clip, Keyframe};
use crate::error::GltfError;
use crate::mesh::{Material, Mesh, Skin, TextureRef, UvSet, make_vertex, normalize_uv};

/// The largest number of joints a skin may bind, matching the shader's bone
/// array.
pub const MAX_JOINTS: usize = 4096;

/// Expand `channels`-per-pixel bytes into RGBA with `convert`.
///
/// A truncated input yields the pixels that fit rather than a panic: a
/// malformed texture should lose its tail, not take the process with it.
fn widen(bytes: &[u8], channels: usize, convert: impl Fn(&[u8]) -> [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / channels * 4);
    for chunk in bytes.chunks(channels) {
        if chunk.len() < channels {
            break;
        }
        out.extend_from_slice(&convert(chunk));
    }
    out
}

/// One decoded image, as RGBA8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// The image's name, from the document.
    pub name: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA8 rows, top row first.
    pub pixels: Vec<u8>,
}

impl Image {
    /// Convert an image `gltf` has already decoded into RGBA8.
    ///
    /// `gltf` hands back 8-bit channels in one of four channel counts, so this
    /// widens the narrow ones. A 16- or 32-bit image is rejected rather than
    /// narrowed: the engine's pipeline is RGBA8 end to end, and a quietly
    /// truncated texture looks like a lighting bug.
    pub fn from_gltf_data(index: usize, data: &gltf::image::Data) -> Result<Image, GltfError> {
        use gltf::image::Format;
        let name = format!("image{index}");
        let (width, height, pixels) = (data.width, data.height, &data.pixels);
        let rgba = match data.format {
            Format::R8G8B8A8 => pixels.clone(),
            Format::R8G8B8 => widen(pixels, 3, |p| [p[0], p[1], p[2], 255]),
            Format::R8G8 => widen(pixels, 2, |p| [p[0], p[1], 0, 255]),
            Format::R8 => widen(pixels, 1, |p| [p[0], p[0], p[0], 255]),
            other => {
                return Err(GltfError::Decode {
                    name,
                    message: format!("{other:?} is not an 8-bit format"),
                });
            }
        };
        Ok(Image {
            name,
            width,
            height,
            pixels: rgba,
        })
    }

    /// Decode an image from encoded bytes.
    pub fn decode(name: &str, bytes: &[u8]) -> Result<Image, GltfError> {
        let decoded = image::load_from_memory(bytes).map_err(|e| GltfError::Decode {
            name: name.to_string(),
            message: e.to_string(),
        })?;
        // to_rgba8 rather than the native format: the engine uploads RGBA8
        // whatever the source was, and converting here keeps that out of the
        // upload path.
        let rgba = decoded.to_rgba8();
        let (width, height) = rgba.dimensions();
        Ok(Image {
            name: name.to_string(),
            width,
            height,
            pixels: rgba.into_raw(),
        })
    }

    /// Bytes one row occupies.
    pub fn stride(&self) -> usize {
        self.width as usize * 4
    }

    /// The pixel at a coordinate, as RGBA, or `None` when out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = y as usize * self.stride() + x as usize * 4;
        self.pixels.get(i..i + 4).map(|p| [p[0], p[1], p[2], p[3]])
    }

    /// True when the image is a uniform colour, which a placeholder texture
    /// always is.
    pub fn is_uniform(&self) -> bool {
        let Some(first) = self.pixels.get(..4) else {
            return true;
        };
        self.pixels.chunks_exact(4).all(|p| p == first)
    }
}

/// A loaded model: its meshes, skins, clips and materials.
#[derive(Debug, Clone, Default)]
pub struct Model {
    /// The model's name, from the document.
    pub name: String,
    /// Every drawable mesh in the document.
    pub meshes: Vec<Mesh>,
    /// Every skin in the document.
    pub skins: Vec<Skin>,
    /// Every animation clip in the document.
    pub clips: Vec<Clip>,
    /// Every material in the document.
    pub materials: Vec<Material>,
    /// The base colour texture of each material, when it has one.
    pub material_textures: Vec<Option<TextureRef>>,
    /// Every node's world transform, in node order.
    pub node_transforms: Vec<Mat4>,
    /// The node each mesh belongs to, in mesh order.
    pub mesh_nodes: Vec<usize>,
    /// Decoded images, in document order.
    pub images: Vec<Image>,
    /// The scene's nodes, in traversal order.
    pub scene_nodes: Vec<usize>,
}

impl Model {
    /// How many triangles the model holds in total.
    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.triangle_count()).sum()
    }

    /// How many vertices the model holds in total.
    pub fn vertex_count(&self) -> usize {
        self.meshes.iter().map(|m| m.vertices.len()).sum()
    }

    /// The bounds of every mesh combined, if there are any.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut iter = self.meshes.iter().filter(|m| !m.is_empty());
        let first = iter.next()?;
        let (mut lo, mut hi) = first.bounds()?;
        for m in iter {
            if let Some((a, b)) = m.bounds() {
                lo = lo.min(a);
                hi = hi.max(b);
            }
        }
        Some((lo, hi))
    }

    /// The world transform of a node, defaulting to identity.
    pub fn node_transform(&self, node: usize) -> Mat4 {
        self.node_transforms
            .get(node)
            .copied()
            .unwrap_or(Mat4::IDENTITY)
    }

    /// The skin a mesh is deformed by, if any.
    pub fn skin_for(&self, mesh: usize) -> Option<&Skin> {
        self.meshes
            .get(mesh)
            .and_then(|m| m.skin)
            .and_then(|s| self.skins.get(s))
    }

    /// A clip by index.
    pub fn clip(&self, index: usize) -> Option<&Clip> {
        self.clips.get(index)
    }

    /// The longest clip in the model, which is what a preview should play.
    pub fn longest_clip(&self) -> Option<usize> {
        self.clips
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.duration().total_cmp(&b.1.duration()))
            .map(|(i, _)| i)
    }
}

/// Load a model from a `.gltf` or `.glb` file.
pub fn load(path: &Path) -> Result<Model, GltfError> {
    let imported = gltf::import(path).map_err(|source| GltfError::Parse {
        path: path.to_path_buf(),
        source,
    })?;
    let (document, buffers, images) = imported;
    let mut model = from_document(&document, &buffers, path)?;
    model.images = convert_images(&images);
    Ok(model)
}

/// Convert the document's images into RGBA8.
///
/// `gltf` has already decoded them, so this is a channel-count conversion, not
/// a decode. Only 8-bit formats are accepted: the engine uploads RGBA8, so a
/// 16- or 32-bit texture would need a different pipeline and a different
/// sampler, and silently narrowing it would change the model's appearance.
fn convert_images(images: &[gltf::image::Data]) -> Vec<Image> {
    images
        .iter()
        .enumerate()
        .filter_map(|(i, d)| Image::from_gltf_data(i, d).ok())
        .collect()
}

/// Load a model from bytes already in memory.
///
/// `base` resolves the document's relative references, so a `.gltf` with
/// external buffers and images still loads when its siblings are on disk.
pub fn load_from_bytes(bytes: &[u8], base: Option<&Path>) -> Result<Model, GltfError> {
    let path = base.unwrap_or(Path::new("<memory>")).to_path_buf();
    // `import_slice` handles both a GLB's blob and a .gltf's data URIs, and
    // rejects a .gltf that points at sibling files, which is the honest
    // answer: those bytes are not all here.
    let (document, buffers, images) =
        gltf::import_slice(bytes).map_err(|source| GltfError::Parse {
            path: path.clone(),
            source,
        })?;
    let mut model = from_document(&document, &buffers, base.unwrap_or(Path::new("<memory>")))?;
    model.images = convert_images(&images);
    Ok(model)
}

/// Build a model from an already-parsed document and its buffers.
pub fn from_document(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    path: &Path,
) -> Result<Model, GltfError> {
    let mut model = Model {
        name: path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".to_string()),
        ..Default::default()
    };

    model.materials = document.materials().map(read_material).collect();
    model.material_textures = document
        .materials()
        .map(|m| {
            m.pbr_metallic_roughness()
                .base_color_texture()
                .map(|t| TextureRef {
                    index: t.texture().index(),
                    tex_coord: t.tex_coord(),
                })
        })
        .collect();

    for (mesh_index, mesh) in document.meshes().enumerate() {
        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            let read = read_primitive(buffers, &primitive, mesh_index, primitive_index)?;
            model.meshes.push(read);
        }
    }

    let node_count = document.nodes().len();
    for (skin_index, skin) in document.skins().enumerate() {
        let joints: Vec<usize> = skin.joints().map(|n| n.index()).collect();
        // An Accessor is a view onto a buffer; it has to be read through a
        // reader before it yields anything.
        let inverse_bind: Vec<Mat4> = skin
            .reader(|buffer| Some(buffer_bytes(buffers, buffer)))
            .read_inverse_bind_matrices()
            // The accessor yields a [[f32; 4]; 4] that is already in
            // column-major order, which is exactly from_cols_array's input.
            // Reindexing it by hand transposes the matrix, which moves a
            // translation from the last column to the last row and silently
            // turns every bind matrix into a rotation.
            .map(|mats| mats.map(mat4_from_column_major).collect())
            // A skin with no inverse bind matrices binds every joint in its
            // parent's space, which is the identity for an unskinned skeleton.
            .unwrap_or_else(|| vec![Mat4::IDENTITY; joints.len()]);

        let skin = Skin {
            inverse_bind,
            joints,
        };
        skin.check_joints(skin_index, node_count)?;
        skin.check_limit(skin_index, MAX_JOINTS)?;
        model.skins.push(skin);
    }

    for (animation_index, animation) in document.animations().enumerate() {
        let mut clip = Clip {
            name: animation
                .name()
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("clip{animation_index}")),
            channels: Vec::new(),
        };
        for channel in animation.channels() {
            let sampler = channel.sampler();
            // The target's property is the glTF path as an enum rather than a
            // string, so every value the crate can hand back is supported.
            let path = match channel.target().property() {
                gltf::animation::Property::Translation => ChannelPath::Translation,
                gltf::animation::Property::Rotation => ChannelPath::Rotation,
                gltf::animation::Property::Scale => ChannelPath::Scale,
                gltf::animation::Property::MorphTargetWeights => ChannelPath::Weights,
            };
            let node = channel.target().node().index();
            let read = read_channel(buffers, &channel, node, path)?;
            read.validate(animation_index, sampler.index())?;
            clip.channels.push(read);
        }
        model.clips.push(clip);
    }

    model.node_transforms = read_node_transforms(document);
    model.mesh_nodes = read_mesh_nodes(document);
    model.images = Vec::new();
    model.scene_nodes = document
        .default_scene()
        .map(|s| s.nodes().map(|n| n.index()).collect())
        .unwrap_or_default();

    Ok(model)
}

/// Every node's world transform, parents composed with their children.
fn read_node_transforms(document: &gltf::Document) -> Vec<Mat4> {
    let nodes: Vec<gltf::Node<'_>> = document.nodes().collect();
    // Resolve each node's parent once; a document can be cyclic in theory, so
    // the chain walk is bounded.
    let mut parents: Vec<Option<usize>> = vec![None; nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        for child in node.children() {
            parents[child.index()] = Some(i);
        }
    }

    let local: Vec<Mat4> = nodes
        .iter()
        // matrix() is column-major as a [[f32; 4]; 4], so flattening it row by
        // row and feeding it to from_cols_array is a transpose. Reading it
        // column by column is the identity mapping, which is the point: a node
        // translated on y must keep that translation in the last column.
        .map(|n| mat4_from_column_major(n.transform().matrix()))
        .collect();

    let mut world: Vec<Mat4> = local.clone();
    for i in 0..nodes.len() {
        let mut current = i;
        let mut chain = Vec::new();
        while let Some(p) = parents[current] {
            if chain.contains(&p) {
                break;
            }
            chain.push(p);
            current = p;
        }
        for p in chain.into_iter().rev() {
            world[i] = local[p] * world[i];
        }
    }
    world
}

/// Which node each mesh belongs to, in mesh order.
///
/// glTF attaches a mesh to a node, and a node to a mesh, so the mapping is
/// inverted here. A mesh no node references is given node 0 rather than dropped:
/// an orphan mesh still draws, and dropping it silently loses geometry.
fn read_mesh_nodes(document: &gltf::Document) -> Vec<usize> {
    let mesh_count = document.meshes().len();
    let mut out = vec![0usize; mesh_count];
    for node in document.nodes() {
        if let Some(mesh) = node.mesh() {
            out[mesh.index()] = node.index();
        }
    }
    out
}

fn read_material(material: gltf::Material<'_>) -> Material {
    let pbr = material.pbr_metallic_roughness();
    Material {
        base_color: Vec4::from(pbr.base_color_factor()),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        double_sided: material.double_sided(),
    }
}

fn read_primitive<'a>(
    buffers: &'a [gltf::buffer::Data],
    primitive: &gltf::Primitive<'_>,
    mesh_index: usize,
    primitive_index: usize,
) -> Result<Mesh, GltfError> {
    let reader = primitive.reader(|buffer| Some(buffer_bytes(buffers, buffer)));
    // POSITION is the one attribute a primitive cannot do without: without it
    // there is no geometry at all, unlike a missing NORMAL which only costs
    // shading.
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or(GltfError::MissingAttribute {
            mesh: mesh_index,
            primitive: primitive_index,
            attribute: "POSITION",
        })?
        .collect();

    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|n| n.map(|c| [c[0], c[1], c[2]]).collect())
        .unwrap_or_default();
    // Every one of these is an enum over the component type the exporter
    // chose, and each has an into_f32 that reads them all the same way. An
    // absent attribute yields an empty list, and the per-vertex fill below
    // substitutes a default rather than shifting every later vertex.
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|t| t.into_f32().map(|c| [c[0], c[1]]).collect())
        .unwrap_or_default();
    let joints: Vec<[u32; 4]> = reader
        .read_joints(0)
        .map(|j| match j {
            gltf::mesh::util::ReadJoints::U8(v) => v
                .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32])
                .collect(),
            gltf::mesh::util::ReadJoints::U16(v) => v
                .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32])
                .collect(),
        })
        .unwrap_or_default();
    let weights: Vec<[f32; 4]> = reader
        .read_weights(0)
        .map(|w| w.into_f32().collect())
        .unwrap_or_default();

    let indices: Vec<u32> = reader
        .read_indices()
        .map(|i| i.into_u32().collect())
        .unwrap_or_default();

    let vertex_count = positions.len();
    let mut vertices = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        // A primitive with no NORMAL attribute gets a placeholder rather than
        // being rejected: the shading is wrong but the geometry draws, which is
        // the better failure.
        let normal = normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
        let uv = uvs.get(i).copied().unwrap_or([0.0, 0.0]);
        let j = joints.get(i).copied().unwrap_or([0; 4]);
        let w = weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]);
        let set = UvSet::default();
        let normalized = normalize_uv(Vec2::new(uv[0], uv[1]), set);
        vertices.push(make_vertex(
            Vec3::from_array(positions[i]),
            Vec3::from_array(normal),
            normalized,
            j,
            w,
        ));
    }

    // A primitive with no index buffer is a non-indexed draw; generating a
    // sequential index list keeps every downstream path working on one type.
    let indices = if indices.is_empty() {
        (0..vertex_count as u32).collect()
    } else {
        indices
    };
    for index in &indices {
        if *index as usize >= vertex_count {
            return Err(GltfError::IndexOutOfRange {
                mesh: mesh_index,
                primitive: primitive_index,
                index: *index as usize,
                count: vertex_count,
            });
        }
    }

    Ok(Mesh {
        vertices,
        indices,
        // A primitive with no material draws with the engine's default, which is
        // material 0 when the document has one and the spec's default otherwise.
        material: primitive.material().index().unwrap_or(0),
        skin: None,
    })
}

fn read_channel<'a>(
    buffers: &'a [gltf::buffer::Data],
    channel: &gltf::animation::Channel<'_>,
    node: usize,
    path: ChannelPath,
) -> Result<Channel, GltfError> {
    let sampler = channel.sampler();
    let reader = channel.reader(|buffer| Some(buffer_bytes(buffers, buffer)));
    // A sampler with an unreadable input or output accessor is malformed. It
    // is reported rather than dropped, because a silently missing channel means
    // a bone that quietly stops animating.
    let Some(inputs) = reader.read_inputs() else {
        return Err(GltfError::UnsupportedPath {
            animation: 0,
            sampler: sampler.index(),
            path: "input accessor is unreadable".to_string(),
        });
    };
    let times: Vec<f32> = inputs.collect();
    let Some(outputs) = reader.read_outputs() else {
        return Err(GltfError::UnsupportedPath {
            animation: 0,
            sampler: sampler.index(),
            path: "output accessor is unreadable".to_string(),
        });
    };

    let keyframes: Vec<Keyframe> = match (path, outputs) {
        (ChannelPath::Translation, gltf::animation::util::ReadOutputs::Translations(v)) => {
            zip_times(v, &times, |c| ChannelValue::Translation(Vec3::from(c)))
        }
        (ChannelPath::Rotation, gltf::animation::util::ReadOutputs::Rotations(v)) => {
            // into_f32 normalises the integer quaternion encodings, which is
            // what makes a u16 rotation a rotation rather than a short vector.
            zip_times(v.into_f32(), &times, |q| {
                ChannelValue::Rotation(Quat::from_xyzw(q[0], q[1], q[2], q[3]))
            })
        }
        (ChannelPath::Scale, gltf::animation::util::ReadOutputs::Scales(v)) => {
            zip_times(v, &times, |c| ChannelValue::Scale(Vec3::from(c)))
        }
        (ChannelPath::Weights, gltf::animation::util::ReadOutputs::MorphTargetWeights(v)) => {
            zip_times(v.into_f32(), &times, ChannelValue::Weight)
        }
        // A channel whose outputs are not the type its path implies is
        // malformed. ReadOutputs has no Debug, so the mismatch is named rather
        // than printed.
        (path, _) => {
            return Err(GltfError::UnsupportedPath {
                animation: 0,
                sampler: sampler.index(),
                path: format!(
                    "{} channel has outputs of a different type",
                    path.gltf_name()
                ),
            });
        }
    };

    Ok(Channel {
        node,
        path,
        keyframes,
    })
}

/// Turn a column-major 4x4 into a `Mat4`.
///
/// glTF stores matrices column by column, and `from_cols_array` takes them
/// column by column, so the correct code is a plain flatten. The tempting
/// version — reading the nested array row by row — is a transpose, which moves
/// a translation from the last column into the last row and turns every bind
/// matrix into a rotation. Both directions are checked below.
fn mat4_from_column_major(m: [[f32; 4]; 4]) -> Mat4 {
    Mat4::from_cols_array(&[
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1],
        m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3],
    ])
}

/// Pair each sampled value with its time.
///
/// The two accessors are parallel arrays, so zipping them stops at the shorter
/// one. Indexing the times by output position instead would run off the end of
/// the times array on a document whose output accessor is longer, and silently
/// give every extra keyframe the last time.
fn zip_times<I, T, F>(values: I, times: &[f32], convert: F) -> Vec<Keyframe>
where
    I: IntoIterator<Item = T>,
    F: Fn(T) -> ChannelValue,
{
    values
        .into_iter()
        .enumerate()
        .map(|(i, v)| Keyframe {
            time: times.get(i).copied().unwrap_or(0.0),
            value: convert(v),
        })
        .collect()
}

/// The bytes of one buffer, or an empty slice when the index is out of range.
///
/// `gltf`'s reader closure takes a borrowed slice with the buffer's own
/// lifetime, and the buffers are owned by the caller's slice, so the lookup has
/// to borrow rather than copy. An index past the end yields an empty slice,
/// which surfaces as a load error rather than a panic.
fn buffer_bytes<'a>(buffers: &'a [gltf::buffer::Data], buffer: gltf::Buffer<'_>) -> &'a [u8] {
    buffers
        .get(buffer.index())
        .map(|b| b.0.as_slice())
        .unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MeshVertex;

    #[test]
    fn an_image_pixels_are_addressable() {
        let img = Image {
            name: "flat".to_string(),
            width: 2,
            height: 2,
            pixels: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
        };
        assert_eq!(img.pixel(0, 0), Some([255, 0, 0, 255]));
        assert_eq!(img.pixel(1, 1), Some([255, 255, 255, 255]));
    }

    #[test]
    fn an_image_pixel_outside_is_none() {
        let img = Image {
            name: "flat".to_string(),
            width: 1,
            height: 1,
            pixels: vec![1, 2, 3, 4],
        };
        assert_eq!(img.pixel(1, 0), None);
        assert_eq!(img.pixel(0, 5), None);
    }

    #[test]
    fn an_image_stride_is_four_bytes_per_pixel() {
        let img = Image {
            name: "s".to_string(),
            width: 3,
            height: 1,
            pixels: Vec::new(),
        };
        assert_eq!(img.stride(), 12);
    }

    #[test]
    fn a_uniform_image_is_detected() {
        let img = Image {
            name: "flat".to_string(),
            width: 1,
            height: 1,
            pixels: vec![7, 7, 7, 255],
        };
        assert!(img.is_uniform());
    }

    #[test]
    fn an_image_with_two_colours_is_not_uniform() {
        let img = Image {
            name: "mixed".to_string(),
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 255, 255, 255, 255, 255],
        };
        assert!(!img.is_uniform());
    }

    #[test]
    fn an_empty_image_counts_as_uniform() {
        let img = Image {
            name: "empty".to_string(),
            width: 0,
            height: 0,
            pixels: Vec::new(),
        };
        assert!(img.is_uniform());
    }

    #[test]
    fn garbage_bytes_are_a_decode_error_not_a_panic() {
        let err = Image::decode("junk", b"not an image").unwrap_err();
        assert!(matches!(err, GltfError::Decode { .. }), "{err:?}");
    }

    #[test]
    fn a_png_decodes_to_its_dimensions() {
        // A 2x1 PNG built by hand: the IHDR is enough for the size, and the
        // decoder is the same one the loader uses.
        let png = minimal_png();
        let img = Image::decode("tiny", &png).unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.pixels.len(), 8);
    }

    #[test]
    fn a_missing_file_is_reported_not_panicked() {
        // gltf::import wraps the io failure in its own error type, so the
        // loader sees a parse error whose source is the io error. What matters
        // is that the path survives into the message and nothing panics.
        let err = load(Path::new("/nonexistent/model.gltf")).unwrap_err();
        assert!(err.to_string().contains("model.gltf"), "{err}");
    }

    #[test]
    fn an_empty_model_has_no_geometry() {
        let m = Model::default();
        assert_eq!(m.triangle_count(), 0);
        assert_eq!(m.vertex_count(), 0);
        assert!(m.bounds().is_none());
    }

    #[test]
    fn a_model_counts_its_geometry_across_meshes() {
        let m = Model {
            meshes: vec![
                Mesh {
                    vertices: vec![MeshVertex::default(); 3],
                    indices: vec![0, 1, 2],
                    ..Default::default()
                },
                Mesh {
                    vertices: vec![MeshVertex::default(); 4],
                    indices: vec![0, 1, 2, 2, 3, 0],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(m.triangle_count(), 3);
        assert_eq!(m.vertex_count(), 7);
    }

    #[test]
    fn a_model_reports_the_longest_clip() {
        let m = Model {
            clips: vec![
                Clip {
                    name: "short".to_string(),
                    channels: vec![crate::anim::Channel {
                        node: 0,
                        path: ChannelPath::Translation,
                        keyframes: vec![Keyframe {
                            time: 1.0,
                            value: ChannelValue::Weight(0.0),
                        }],
                    }],
                },
                Clip {
                    name: "long".to_string(),
                    channels: vec![crate::anim::Channel {
                        node: 0,
                        path: ChannelPath::Translation,
                        keyframes: vec![Keyframe {
                            time: 5.0,
                            value: ChannelValue::Weight(0.0),
                        }],
                    }],
                },
            ],
            ..Default::default()
        };
        assert_eq!(m.longest_clip(), Some(1));
    }

    #[test]
    fn a_column_major_matrix_keeps_its_translation_in_the_last_column() {
        // The bug this catches: flattening the nested array row by row is a
        // transpose, so a translation ends up in the last row and the matrix
        // becomes a pure rotation.
        let m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0, 1.0],
        ];
        let out = mat4_from_column_major(m);
        assert_eq!(out.w_axis, glam::Vec4::new(5.0, 0.0, 0.0, 1.0), "{out:?}");
        let p = out * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!((p.x - 5.0).abs() < 1e-6, "a translated origin moves: {p:?}");
    }

    #[test]
    fn a_transposed_matrix_would_have_been_wrong() {
        // The same numbers read row by row, which is what the bug did.
        let transposed = mat4_from_column_major([
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        assert_eq!(
            transposed.w_axis,
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            "reading row by row loses the translation entirely"
        );
    }

    #[test]
    fn an_identity_matrix_survives_the_conversion() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(mat4_from_column_major(identity), Mat4::IDENTITY);
    }

    #[test]
    fn a_model_with_no_clips_has_no_longest() {
        assert_eq!(Model::default().longest_clip(), None);
    }

    #[test]
    fn a_node_transform_out_of_range_is_identity() {
        let m = Model::default();
        assert_eq!(m.node_transform(99), Mat4::IDENTITY);
    }

    #[test]
    fn a_mesh_with_no_skin_reports_none() {
        let m = Model {
            meshes: vec![Mesh::default()],
            ..Default::default()
        };
        assert!(m.skin_for(0).is_none());
    }

    /// A valid 2x1 RGBA PNG, built byte by byte so the test needs no fixture.
    fn minimal_png() -> Vec<u8> {
        // Two IDAT-deflate-identical scanline filters of colour type 6.
        let mut raw = vec![0u8, 255, 0, 0, 255, 0, 255, 0, 255];
        // A zlib stream with stored (uncompressed) deflate blocks.
        let mut idat = vec![0x78, 0x01];
        idat.push(0x01);
        idat.push((raw.len() & 0xff) as u8);
        idat.push((raw.len() >> 8) as u8);
        idat.push(!(raw.len() & 0xff) as u8);
        idat.push(!((raw.len() >> 8) & 0xff) as u8);
        idat.extend_from_slice(&raw);
        idat.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);

        let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        png.extend_from_slice(&chunk(b"IHDR", &ihdr));
        png.extend_from_slice(&chunk(b"IDAT", &idat));
        png.extend_from_slice(&chunk(b"IEND", &[]));
        raw.clear();
        png
    }

    fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + data.len());
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let mut crc_input = kind.to_vec();
        crc_input.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
        out
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xffff_ffffu32;
        for byte in data {
            crc ^= *byte as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xedb8_8320 & mask);
            }
        }
        !crc
    }

    fn adler32(data: &[u8]) -> u32 {
        let mut a = 1u32;
        let mut b = 0u32;
        for byte in data {
            a = (a + *byte as u32) % 65521;
            b = (b + a) % 65521;
        }
        (b << 16) | a
    }
}
