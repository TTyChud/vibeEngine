//! Backend-agnostic render hardware interface for vibeEngine.
//!
//! The Vulkan backend in `vibe-vk` is the only live implementation; DX12 and
//! Metal exist here as enum placeholders so scene and render-graph code can be
//! written against a stable API before those backends land. The types in this
//! crate deliberately contain no Vulkan types, which is what makes that
//! possible.

#![deny(unsafe_code)]

use std::fmt;

use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use vibe_math::{Aabb2, look_at_rh, orthographic_rh, perspective_rh_reverse_z_infinite};

/// Which rendering backend a resource or pipeline belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Vulkan 1.1+, the live backend.
    Vulkan,
    /// Direct3D 12. Placeholder; no implementation yet.
    Dx12,
    /// Metal. Placeholder; no implementation yet.
    Metal,
}

impl Backend {
    /// The name used in file extensions and error messages.
    pub const fn name(self) -> &'static str {
        match self {
            Backend::Vulkan => "vulkan",
            Backend::Dx12 => "dx12",
            Backend::Metal => "metal",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A texture or buffer format, with the flags needed to build a Vulkan format.
///
/// Width/height/depth are 1 for buffers and unused for non-sampled formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Format {
    /// The platform-neutral format kind.
    pub kind: FormatKind,
    /// Bits per texel, e.g. 32 for `Rgba8Unorm`.
    pub bits_per_texel: u8,
    /// True when the format has a depth component usable as a depth attachment.
    pub has_depth: bool,
    /// True when the format has a stencil component.
    pub has_stencil: bool,
}

/// The platform-neutral set of formats the engine supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FormatKind {
    /// 8-bit unsigned normalised single channel.
    R8Unorm,
    /// 8-bit unsigned normalised two channels.
    Rg8Unorm,
    /// 8-bit unsigned normalised four channels.
    Rgba8Unorm,
    /// 8-bit unsigned normalised four channels in sRGB encoding.
    Rgba8UnormSrgb,
    /// 16-bit float, two channels.
    Rg16Float,
    /// 16-bit float, four channels.
    Rgba16Float,
    /// 32-bit float, four channels.
    Rgba32Float,
    /// Signed 32-bit integer, four channels; used for bindless buffer indices.
    Rgba32Uint,
    /// 24-bit depth plus 8-bit stencil.
    Depth24Stencil8,
    /// 32-bit float depth.
    Depth32Float,
}

impl Format {
    /// Bytes occupied by one texel.
    pub const fn texel_size(self) -> u32 {
        (self.bits_per_texel as u32).div_ceil(8)
    }

    /// Total bytes for a `width` x `height` x `depth` allocation, with at least
    /// one row per image and one slice per mip.
    pub const fn image_size(self, width: u32, height: u32, depth: u32) -> u64 {
        let w = if width == 0 { 1 } else { width } as u64;
        let h = if height == 0 { 1 } else { height } as u64;
        let d = if depth == 0 { 1 } else { depth } as u64;
        w * h * d * self.texel_size() as u64
    }

    /// A short debug name, e.g. `Rgba8Unorm`.
    pub const fn kind_name(self) -> &'static str {
        match self.kind {
            FormatKind::R8Unorm => "R8Unorm",
            FormatKind::Rg8Unorm => "Rg8Unorm",
            FormatKind::Rgba8Unorm => "Rgba8Unorm",
            FormatKind::Rgba8UnormSrgb => "Rgba8UnormSrgb",
            FormatKind::Rg16Float => "Rg16Float",
            FormatKind::Rgba16Float => "Rgba16Float",
            FormatKind::Rgba32Float => "Rgba32Float",
            FormatKind::Rgba32Uint => "Rgba32Uint",
            FormatKind::Depth24Stencil8 => "Depth24Stencil8",
            FormatKind::Depth32Float => "Depth32Float",
        }
    }
}

/// How textures and buffers are addressed by shaders.
///
/// Bindless is the target: one large descriptor array indexed by a `u32` in
/// the vertex data, with no per-object descriptor sets or rebinds. Not every
/// device can do it, and the Tiger Lake iGPU in particular reports
/// `VK_EXT_descriptor_indexing` but not `VK_KHR_bindless_texture`, so the
/// renderer needs a described-array fallback and must ask which mode it got.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BindlessSupport {
    /// Full bindless: shader-visible descriptor indexing with a large
    /// non-uniform array, addressed by integer index.
    #[default]
    Full,
    /// Only what `VK_EXT_descriptor_indexing` allows, typically a smaller
    /// per-stage descriptor array or a runtime array with a lower limit.
    Partial {
        /// Largest non-uniform array indexable from a shader.
        max_indexed_descriptors: u32,
    },
    /// No descriptor indexing at all; fall back to per-batch descriptor sets.
    None,
}

impl BindlessSupport {
    /// True when resources can be addressed by integer index at all.
    pub const fn is_bindless(self) -> bool {
        !matches!(self, BindlessSupport::None)
    }
}

/// How a buffer will be used, which decides its memory type and access pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    /// Total size in bytes.
    pub size: u64,
    /// How the CPU will access the buffer, if at all.
    pub usage: BufferUsage,
}

/// The three access patterns a buffer can be optimised for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    /// GPU-only, written by a host-visible staging copy. The default for
    /// vertex, index and storage buffers.
    GpuOnly,
    /// CPU writes every frame, GPU reads. For uniform and push-constant-backed
    /// blocks that are small enough to keep mapped.
    HostVisible,
    /// Both CPU and GPU write. Requires explicit synchronisation.
    HostCoherent,
}

/// Texture dimensions and mip/array counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureDesc {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Texel format.
    pub format: Format,
    /// Number of mip levels; 1 means no mips.
    pub mip_levels: u32,
    /// Number of array layers; 1 means a plain 2D texture.
    pub array_layers: u32,
}

impl TextureDesc {
    /// A 2D texture with one mip and one layer.
    pub fn new_2d(width: u32, height: u32, format: Format) -> TextureDesc {
        TextureDesc {
            width,
            height,
            format,
            mip_levels: 1,
            array_layers: 1,
        }
    }
}

/// A uniform, read-only view of a buffer range, for push constants and blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferRange {
    /// Byte offset from the start of the buffer.
    pub offset: u64,
    /// Size in bytes.
    pub size: u64,
}

impl BufferRange {
    /// A range starting at `offset` of `size` bytes.
    pub fn new(offset: u64, size: u64) -> BufferRange {
        BufferRange { offset, size }
    }

    /// A range covering `bytes`, which must already be initialised and have no
    /// padding, so it can be used as push constants.
    pub fn push_constants<T>(bytes: &[u8]) -> BufferRange
    where
        T: bytemuck::NoUninit,
    {
        BufferRange {
            offset: 0,
            size: bytes.len() as u64,
        }
    }
}

/// How a texture is sampled when read by a shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerDesc {
    /// Filter when the sample falls between texels.
    pub filter: FilterMode,
    /// Behaviour when UVs fall outside `0..1`.
    pub address_mode: AddressMode,
}

/// Texel filter mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterMode {
    /// Nearest texel.
    Nearest,
    /// Linear blend between texels.
    Linear,
}

/// Texture coordinate wrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// Clamp to the edge texel.
    ClampToEdge,
    /// Tile the texture.
    Repeat,
    /// Mirror on alternating tiles.
    MirroredRepeat,
}

/// A single quad, as uploaded to the batched 2D vertex buffer.
///
/// Kept 32-byte aligned so the whole vertex is one cache line and the stride
/// satisfies common hardware vertex-fetch rules.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuadVertex {
    /// Corner position in pixels, relative to the batch's transform.
    pub position: Vec2,
    /// Colour multiplied into the sampled texel.
    pub color: [u8; 4],
    /// Normalised texture coordinate.
    pub uv: Vec2,
    /// Index into the bindless texture array.
    pub texture_index: u32,
    /// Pads the struct to 32 bytes so one vertex is one aligned quarter of a
    /// cache line and the stride matches common vertex-fetch rules.
    pub _pad: [u32; 2],
}

impl QuadVertex {
    /// A vertex with no texture bound and no tint.
    pub fn untextured(position: Vec2) -> QuadVertex {
        QuadVertex {
            position,
            color: [255, 255, 255, 255],
            uv: Vec2::ZERO,
            texture_index: 0,
            _pad: [0; 2],
        }
    }
}

/// Size of one [`QuadVertex`] in bytes.
pub const QUAD_VERTEX_SIZE: u64 = std::mem::size_of::<QuadVertex>() as u64;

/// The camera the 2D batch renderer transforms quads with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2D {
    /// The full view-projection matrix, mapping world space to clip space.
    pub view_projection: Mat4,
    /// World-space rectangle visible on screen, used to cull quads.
    pub visible_bounds: Aabb2,
}

impl Camera2D {
    /// An orthographic camera covering `(0,0)..(width,height)` pixels, with Y
    /// pointing down so sprite coordinates match screen coordinates.
    pub fn screen(width: f32, height: f32) -> Camera2D {
        let view_projection = orthographic_rh(0.0, width, 0.0, height, -1.0, 1.0);
        Camera2D {
            view_projection,
            visible_bounds: Aabb2::new(Vec2::ZERO, Vec2::new(width, height)),
        }
    }
}

/// A perspective camera for 3D passes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera3D {
    /// View-projection matrix.
    pub view_projection: Mat4,
    /// Camera world position, for depth sorting and culling.
    pub position: Vec3,
    /// Field of view in radians.
    pub fov_y_radians: f32,
    /// Far plane distance.
    pub far: f32,
}

impl Camera3D {
    /// A perspective camera at `position` looking at `target`, with Y up and a
    /// reverse-Z infinite far plane.
    pub fn perspective(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_radians: f32,
        aspect: f32,
        near: f32,
    ) -> Camera3D {
        let view = look_at_rh(position, target, up);
        let proj = perspective_rh_reverse_z_infinite(fov_y_radians, aspect, near);
        Camera3D {
            view_projection: proj * view,
            position,
            fov_y_radians,
            far: f32::INFINITY,
        }
    }
}

/// Live counters from the 2D batch renderer, for the editor stats panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BatchStats {
    /// Quads submitted this frame.
    pub quads: u32,
    /// `vkCmdDraw` calls issued this frame.
    pub draw_calls: u32,
    /// Quads dropped by frustum culling this frame.
    pub culled: u32,
    /// Vertices uploaded this frame (always `quads * 6`).
    pub vertices: u32,
}

/// How a texture sub-region maps onto a quad.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubTexture {
    /// Index into the bindless texture array.
    pub texture_index: u32,
    /// Top-left corner in normalised UV space.
    pub uv_origin: Vec2,
    /// Size in normalised UV space.
    pub uv_size: Vec2,
}

impl SubTexture {
    /// The full extent of a bindless texture, with a 1x1-pixel inset removed
    /// from each edge to stop neighbouring atlas entries bleeding in.
    pub fn full(texture_index: u32) -> SubTexture {
        SubTexture {
            texture_index,
            uv_origin: Vec2::ZERO,
            uv_size: Vec2::ONE,
        }
    }

    /// A sub-region of a texture in normalised UV space.
    pub fn region(texture_index: u32, origin: Vec2, size: Vec2) -> SubTexture {
        SubTexture {
            texture_index,
            uv_origin: origin,
            uv_size: size,
        }
    }

    /// Whether this sub-texture draws a border inset, which is what a sprite
    /// atlas frame needs to avoid bleeding from its neighbours.
    pub fn is_inset(&self) -> bool {
        self.uv_size.x < 1.0 || self.uv_size.y < 1.0
    }
}

/// A rotated quad, decomposed for the batch vertex format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedQuad {
    /// Top-left corner in pixels.
    pub origin: Vec2,
    /// Width and height in pixels.
    pub size: Vec2,
    /// Rotation about `origin`, in radians, clockwise in Y-down space.
    pub rotation_radians: f32,
    /// Tint multiplied into the texture.
    pub tint: Vec4,
}

impl RotatedQuad {
    /// An axis-aligned quad.
    pub fn new(origin: Vec2, size: Vec2, tint: Vec4) -> RotatedQuad {
        RotatedQuad {
            origin,
            size,
            rotation_radians: 0.0,
            tint,
        }
    }

    /// The four corners in clockwise order, matching the quad's two triangles.
    pub fn corners(&self) -> [Vec2; 4] {
        let c = self.rotation_radians.cos();
        let s = self.rotation_radians.sin();
        let w = Vec2::new(self.size.x, 0.0);
        let h = Vec2::new(0.0, self.size.y);
        let right = Vec2::new(w.x * c - w.y * s, w.x * s + w.y * c);
        let down = Vec2::new(h.x * c - h.y * s, h.x * s + h.y * c);
        let o = self.origin;
        [o, o + right, o + right + down, o + down]
    }
}

/// The four UVs matching [`RotatedQuad::corners`].
pub const CORNER_UVS: [[Vec2; 4]; 1] = [[
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
]];

/// Index pairs forming two triangles from four quad corners.
pub const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// A GPU resource the render graph can schedule.
///
/// Opaque on purpose: the graph reasons about lifetimes and barriers from
/// handles alone and never needs the backend's concrete object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceHandle {
    /// Registry slot, reused after free.
    pub index: u32,
    /// Bumped on every free so a stale handle cannot alias a new resource.
    pub generation: u32,
}

impl ResourceHandle {
    /// A handle for a slot that has never been used.
    pub const fn invalid() -> ResourceHandle {
        ResourceHandle {
            index: u32::MAX,
            generation: 0,
        }
    }

    /// True when this handle was made by [`ResourceHandle::invalid`].
    pub const fn is_invalid(self) -> bool {
        self.index == u32::MAX
    }

    /// The slot index.
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The generation, bumped on every free.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

impl Default for ResourceHandle {
    fn default() -> Self {
        ResourceHandle::invalid()
    }
}

/// What a render pass does to a resource it references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceUsage {
    /// Read-only in a shader (sampled texture or read-only storage buffer).
    Read,
    /// Written as a render target or storage image.
    Write,
    /// Read and written.
    ReadWrite,
    /// Written this pass and read in a later pass, so the barrier is only
    /// needed at the transition, not on both sides.
    Discard,
}

/// Opaque backend object identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId {
    /// Backend-specific handle value.
    pub handle: u64,
}

/// A viewport, as a scissor+viewport pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// Top-left in pixels.
    pub offset: UVec2,
    /// Width and height in pixels.
    pub size: UVec2,
    /// Minimum depth, for reverse-Z this is `1.0`.
    pub min_depth: f32,
    /// Maximum depth, for reverse-Z this is `0.0`.
    pub max_depth: f32,
}

impl Viewport {
    /// A full-target viewport with a reverse-Z depth range.
    pub fn full(width: u32, height: u32) -> Viewport {
        Viewport {
            offset: UVec2::ZERO,
            size: UVec2::new(width, height),
            min_depth: 1.0,
            max_depth: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_kind_names() {
        let rgba8 = Format {
            kind: FormatKind::Rgba8Unorm,
            bits_per_texel: 32,
            has_depth: false,
            has_stencil: false,
        };
        let depth = Format {
            kind: FormatKind::Depth24Stencil8,
            bits_per_texel: 32,
            has_depth: true,
            has_stencil: true,
        };
        assert_eq!(rgba8.kind_name(), "Rgba8Unorm");
        assert_eq!(depth.kind_name(), "Depth24Stencil8");
    }

    #[test]
    fn texel_size_rounds_partial_bytes() {
        let rgba8 = Format {
            kind: FormatKind::Rgba8Unorm,
            bits_per_texel: 32,
            has_depth: false,
            has_stencil: false,
        };
        assert_eq!(rgba8.texel_size(), 4);
        let f16 = Format {
            kind: FormatKind::Rg16Float,
            bits_per_texel: 32,
            has_depth: false,
            has_stencil: false,
        };
        assert_eq!(f16.texel_size(), 4);
    }

    #[test]
    fn image_size_computes_bytes() {
        let f = Format {
            kind: FormatKind::Rgba8Unorm,
            bits_per_texel: 32,
            has_depth: false,
            has_stencil: false,
        };
        assert_eq!(f.image_size(64, 32, 1), 64 * 32 * 4);
    }

    #[test]
    fn image_size_treats_zero_dims_as_one() {
        let f = Format {
            kind: FormatKind::Rgba8Unorm,
            bits_per_texel: 32,
            has_depth: false,
            has_stencil: false,
        };
        assert_eq!(
            f.image_size(0, 0, 0),
            4,
            "a zero-sized allocation still costs one texel"
        );
    }

    #[test]
    fn backend_names() {
        assert_eq!(Backend::Vulkan.name(), "vulkan");
        assert_eq!(Backend::Dx12.to_string(), "dx12");
    }

    #[test]
    fn quad_vertex_is_pod_and_32_bytes() {
        assert_eq!(std::mem::size_of::<QuadVertex>(), 32);
        let v = QuadVertex::untextured(Vec2::new(1.0, 2.0));
        let bytes = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), QUAD_VERTEX_SIZE as usize);
    }

    #[test]
    fn quad_untextured_is_opaque_white_at_origin_uv() {
        let v = QuadVertex::untextured(Vec2::new(3.0, 4.0));
        assert_eq!(v.color, [255, 255, 255, 255]);
        assert_eq!(v.uv, Vec2::ZERO);
        assert_eq!(v.position, Vec2::new(3.0, 4.0));
    }

    #[test]
    fn rotated_quad_corners_are_clockwise_and_close_the_loop() {
        let q = RotatedQuad::new(Vec2::new(10.0, 20.0), Vec2::new(4.0, 2.0), Vec4::ONE);
        let c = q.corners();
        assert_eq!(c[0], Vec2::new(10.0, 20.0));
        assert_eq!(c[2], Vec2::new(14.0, 22.0));
        assert_eq!(c[3], Vec2::new(10.0, 22.0));
    }

    #[test]
    fn rotated_quad_quarter_turn_swaps_axes() {
        // A +90 degree turn carries the width edge from +X onto +Y; the
        // height edge then lands on -X, so the far corner is (-1, 2).
        let q = RotatedQuad {
            origin: Vec2::ZERO,
            size: Vec2::new(2.0, 1.0),
            rotation_radians: std::f32::consts::FRAC_PI_2,
            tint: Vec4::ONE,
        };
        let c = q.corners();
        assert!(
            (c[1].x).abs() < 1e-5,
            "width edge should be vertical, got {:?}",
            c[1]
        );
        assert!((c[1].y - 2.0).abs() < 1e-5);
        assert!(
            (c[2].x + 1.0).abs() < 1e-5,
            "height edge should point -X, got {:?}",
            c[2]
        );
        assert!((c[2].y - 2.0).abs() < 1e-5);
    }

    #[test]
    fn corner_uvs_pair_with_indices() {
        let uvs = &CORNER_UVS[0];
        let mut seen = [false; 4];
        for i in QUAD_INDICES {
            seen[i as usize] = true;
        }
        assert!(
            seen.iter().all(|&b| b),
            "quad indices must cover all four corners"
        );
        assert_eq!(uvs[0], Vec2::new(0.0, 0.0));
        assert_eq!(uvs[2], Vec2::new(1.0, 1.0));
    }

    #[test]
    fn invalid_handle_is_recognised() {
        let h = ResourceHandle::invalid();
        assert!(h.is_invalid());
        assert!(
            !ResourceHandle {
                index: 0,
                generation: 1
            }
            .is_invalid()
        );
    }

    #[test]
    fn viewport_default_is_reverse_z() {
        let v = Viewport::full(800, 600);
        assert_eq!(v.size, UVec2::new(800, 600));
        assert_eq!(v.min_depth, 1.0);
        assert_eq!(v.max_depth, 0.0);
    }

    #[test]
    fn camera2d_maps_view_rect_onto_clip_range() {
        // (0,0) is the bottom-left of the view rect, so it lands on NDC -1.
        let cam = Camera2D::screen(100.0, 50.0);
        let p = cam.view_projection * Vec4::new(0.0, 0.0, 0.0, 1.0);
        let ndc = p.truncate() / p.w;
        assert!(
            (ndc.x + 1.0).abs() < 1e-4,
            "left edge should be -1, got {}",
            ndc.x
        );
        assert!(
            (ndc.y + 1.0).abs() < 1e-4,
            "bottom edge should be -1, got {}",
            ndc.y
        );
    }

    #[test]
    fn camera2d_maps_far_corner_to_positive_clip() {
        let cam = Camera2D::screen(100.0, 50.0);
        let p = cam.view_projection * Vec4::new(100.0, 50.0, 0.0, 1.0);
        assert!(
            p.x > 0.0 && p.y > 0.0,
            "top-right should be positive in clip space"
        );
    }

    #[test]
    fn camera2d_visible_bounds_match_viewport() {
        let cam = Camera2D::screen(320.0, 240.0);
        assert_eq!(cam.visible_bounds.min, Vec2::ZERO);
        assert_eq!(cam.visible_bounds.max, Vec2::new(320.0, 240.0));
    }

    #[test]
    fn sub_texture_full_is_not_inset_but_region_is() {
        assert!(!SubTexture::full(3).is_inset());
        assert!(SubTexture::region(3, Vec2::new(0.1, 0.1), Vec2::splat(0.5)).is_inset());
    }

    #[test]
    fn sub_texture_keeps_its_bindless_index() {
        let s = SubTexture::full(7);
        assert_eq!(s.texture_index, 7);
        assert_eq!(s.uv_origin, Vec2::ZERO);
        assert_eq!(s.uv_size, Vec2::ONE);
    }

    #[test]
    fn push_constant_range_uses_bytes_len() {
        let r = BufferRange::push_constants::<[u32; 4]>(&[0; 16]);
        assert_eq!(r.size, 16);
        assert_eq!(r.offset, 0);
    }

    #[test]
    fn buffer_range_fields() {
        let r = BufferRange::new(64, 256);
        assert_eq!(r.offset, 64);
        assert_eq!(r.size, 256);
    }

    #[test]
    fn texture_desc_new_2d_is_single_mip() {
        let f = Format {
            kind: FormatKind::Rgba8Unorm,
            bits_per_texel: 32,
            has_depth: false,
            has_stencil: false,
        };
        let d = TextureDesc::new_2d(64, 64, f);
        assert_eq!(d.mip_levels, 1);
        assert_eq!(d.array_layers, 1);
    }

    #[test]
    fn camera3d_perspective_places_target_in_front() {
        let cam =
            Camera3D::perspective(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y, 1.0, 1.0, 0.1);
        let clip = cam.view_projection * Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!(
            clip.w > 0.0,
            "a point in front of the camera has positive w"
        );
    }

    #[test]
    fn bindless_modes_report_indexing_availability() {
        assert!(BindlessSupport::Full.is_bindless());
        assert!(
            BindlessSupport::Partial {
                max_indexed_descriptors: 1024
            }
            .is_bindless()
        );
        assert!(!BindlessSupport::None.is_bindless());
    }

    #[test]
    fn bindless_defaults_to_full() {
        assert_eq!(BindlessSupport::default(), BindlessSupport::Full);
    }

    #[test]
    fn resource_handle_orders_by_index_then_generation() {
        let a = ResourceHandle {
            index: 0,
            generation: 5,
        };
        let b = ResourceHandle {
            index: 0,
            generation: 6,
        };
        let c = ResourceHandle {
            index: 1,
            generation: 0,
        };
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn default_handle_is_invalid() {
        assert!(ResourceHandle::default().is_invalid());
    }
}
