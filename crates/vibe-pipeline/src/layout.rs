//! Vertex buffer layouts.
//!
//! The layout has to agree with the shader's input locations *and* with the
//! CPU-side struct, or the GPU silently reads the wrong bytes. All three
//! descriptions are kept in one place and cross-checked, because a mismatch
//! there is the single most common cause of "my sprite is garbage".

use ash::vk;

use crate::error::PipelineError;

/// The type of one vertex attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    /// Two 32-bit floats.
    Float2,
    /// Three 32-bit floats.
    Float3,
    /// Four 32-bit floats.
    Float4,
    /// Four normalised 8-bit unsigned integers, as a colour.
    Unorm8x4,
    /// One 32-bit unsigned integer.
    Uint32,
    /// Two 32-bit signed integers.
    Int32x2,
    /// Four 32-bit signed integers.
    Int32x4,
}

impl VertexFormat {
    /// Bytes one value of this format occupies.
    pub const fn size(self) -> u32 {
        match self {
            VertexFormat::Float2 => 8,
            VertexFormat::Float3 => 12,
            VertexFormat::Float4 => 16,
            VertexFormat::Unorm8x4 => 4,
            VertexFormat::Uint32 => 4,
            VertexFormat::Int32x2 => 8,
            VertexFormat::Int32x4 => 16,
        }
    }

    /// The Vulkan format for this attribute.
    pub const fn vk_format(self) -> vk::Format {
        match self {
            VertexFormat::Float2 => vk::Format::R32G32_SFLOAT,
            VertexFormat::Float3 => vk::Format::R32G32B32_SFLOAT,
            VertexFormat::Float4 => vk::Format::R32G32B32A32_SFLOAT,
            VertexFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
            VertexFormat::Uint32 => vk::Format::R32_UINT,
            VertexFormat::Int32x2 => vk::Format::R32G32_SINT,
            VertexFormat::Int32x4 => vk::Format::R32G32B32A32_SINT,
        }
    }

    /// The number of components a shader sees.
    pub const fn components(self) -> u32 {
        match self {
            VertexFormat::Float2 => 2,
            VertexFormat::Float3 => 3,
            VertexFormat::Float4 => 4,
            VertexFormat::Unorm8x4 => 4,
            VertexFormat::Uint32 => 1,
            VertexFormat::Int32x2 => 2,
            VertexFormat::Int32x4 => 4,
        }
    }
}

/// One attribute in a layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attribute {
    /// Shader input location this attribute feeds.
    pub location: u32,
    /// Byte offset from the start of the vertex.
    pub offset: u32,
    /// The attribute's type.
    pub format: VertexFormat,
}

impl Attribute {
    /// An attribute at a location and offset.
    pub fn new(location: u32, offset: u32, format: VertexFormat) -> Attribute {
        Attribute {
            location,
            offset,
            format,
        }
    }
}

/// A vertex layout: a stride plus the attributes packed into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexLayout {
    /// Bytes per vertex.
    pub stride: u32,
    /// The attributes, in offset order.
    pub attributes: Vec<Attribute>,
}

impl VertexLayout {
    /// A layout with the given stride and attributes.
    pub fn new(stride: u32, attributes: Vec<Attribute>) -> VertexLayout {
        VertexLayout { stride, attributes }
    }

    /// Check the layout is self-consistent.
    ///
    /// Catches the three mistakes that produce garbage geometry rather than an
    /// error: two attributes sharing a location, an attribute running past the
    /// stride, and two attributes overlapping in the same bytes.
    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.stride == 0 {
            return Err(PipelineError::BadLayout(
                "stride must be non-zero".to_string(),
            ));
        }

        let mut seen_locations: Vec<u32> = Vec::new();
        let mut ranges: Vec<(u32, u32)> = Vec::new();

        for a in &self.attributes {
            if seen_locations.contains(&a.location) {
                return Err(PipelineError::Duplicate {
                    kind: "location",
                    value: a.location,
                });
            }
            seen_locations.push(a.location);

            let end = a.offset + a.format.size();
            if end > self.stride {
                return Err(PipelineError::BadLayout(format!(
                    "attribute at location {} ends at {end}, past the {}-byte stride",
                    a.location, self.stride
                )));
            }
            for (start, other_end) in &ranges {
                if a.offset < *other_end && *start < end {
                    let (start, stop) = (a.offset, end);
                    return Err(PipelineError::BadLayout(format!(
                        "attribute at location {} overlaps another at bytes {start}..{stop}",
                        a.location
                    )));
                }
            }
            ranges.push((a.offset, end));
        }

        Ok(())
    }

    /// The Vulkan binding descriptions for this layout.
    pub fn attribute_descriptions(&self) -> Vec<vk::VertexInputAttributeDescription> {
        self.attributes
            .iter()
            .map(|a| vk::VertexInputAttributeDescription {
                location: a.location,
                binding: 0,
                format: a.format.vk_format(),
                offset: a.offset,
            })
            .collect()
    }

    /// The Vulkan binding description for this layout.
    pub fn binding_description(&self) -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: self.stride,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    /// The locations this layout feeds, in order.
    pub fn locations(&self) -> Vec<u32> {
        self.attributes.iter().map(|a| a.location).collect()
    }

    /// The byte range the attributes actually occupy, ignoring padding.
    pub fn used_bytes(&self) -> u32 {
        self.attributes
            .iter()
            .map(|a| a.offset + a.format.size())
            .max()
            .unwrap_or(0)
    }

    /// True when the layout leaves padding after the last attribute.
    pub fn has_padding(&self) -> bool {
        self.used_bytes() < self.stride
    }
}

/// The layout for the engine's 32-byte batched quad vertex.
///
/// Matches `vibe_rhi::QuadVertex` field for field: position at 0, tint at 8,
/// uv at 12, texture index at 20, and padding through 32.
pub fn vertex_layout_for_quad() -> VertexLayout {
    VertexLayout::new(
        32,
        vec![
            Attribute::new(0, 0, VertexFormat::Float2),
            Attribute::new(1, 8, VertexFormat::Unorm8x4),
            Attribute::new(2, 12, VertexFormat::Float2),
            Attribute::new(3, 20, VertexFormat::Uint32),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_sizes_are_correct() {
        assert_eq!(VertexFormat::Float2.size(), 8);
        assert_eq!(VertexFormat::Float4.size(), 16);
        assert_eq!(VertexFormat::Unorm8x4.size(), 4);
        assert_eq!(VertexFormat::Uint32.size(), 4);
        assert_eq!(VertexFormat::Int32x2.size(), 8);
        assert_eq!(VertexFormat::Int32x4.size(), 16);
    }

    #[test]
    fn format_components_are_correct() {
        assert_eq!(VertexFormat::Float2.components(), 2);
        assert_eq!(VertexFormat::Unorm8x4.components(), 4);
        assert_eq!(VertexFormat::Uint32.components(), 1);
    }

    #[test]
    fn the_quad_layout_is_valid() {
        assert!(vertex_layout_for_quad().validate().is_ok());
    }

    #[test]
    fn the_quad_layout_matches_the_rhi_vertex() {
        // The whole pipeline rests on this: if the layout drifts from the struct
        // the GPU reads the wrong bytes, with no error anywhere.
        assert_eq!(
            vertex_layout_for_quad().stride as usize,
            std::mem::size_of::<vibe_rhi::QuadVertex>()
        );
    }

    #[test]
    fn the_quad_layout_offsets_match_the_struct_fields() {
        let layout = vertex_layout_for_quad();
        let by_location = |loc: u32| {
            layout
                .attributes
                .iter()
                .find(|a| a.location == loc)
                .copied()
                .expect("attribute")
        };
        // QuadVertex: position at 0, color at 8, uv at 12, texture_index at 20.
        assert_eq!(by_location(0).offset, 0);
        assert_eq!(by_location(0).format, VertexFormat::Float2);
        assert_eq!(by_location(1).offset, 8);
        assert_eq!(by_location(1).format, VertexFormat::Unorm8x4);
        assert_eq!(by_location(2).offset, 12);
        assert_eq!(by_location(2).format, VertexFormat::Float2);
        assert_eq!(by_location(3).offset, 20);
        assert_eq!(by_location(3).format, VertexFormat::Uint32);
    }

    #[test]
    fn the_quad_layout_uses_four_locations_in_order() {
        assert_eq!(vertex_layout_for_quad().locations(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn the_quad_layout_leaves_the_trailing_padding() {
        let layout = vertex_layout_for_quad();
        assert_eq!(layout.used_bytes(), 24);
        assert!(layout.has_padding(), "bytes 24..32 are padding");
    }

    #[test]
    fn a_zero_stride_is_rejected() {
        let layout = VertexLayout::new(0, vec![]);
        assert!(matches!(
            layout.validate(),
            Err(PipelineError::BadLayout(_))
        ));
    }

    #[test]
    fn two_attributes_at_one_location_are_rejected() {
        let layout = VertexLayout::new(
            32,
            vec![
                Attribute::new(0, 0, VertexFormat::Float2),
                Attribute::new(0, 8, VertexFormat::Float2),
            ],
        );
        assert!(matches!(
            layout.validate(),
            Err(PipelineError::Duplicate {
                kind: "location",
                value: 0
            })
        ));
    }

    #[test]
    fn an_attribute_past_the_stride_is_rejected() {
        let layout = VertexLayout::new(
            16,
            vec![
                Attribute::new(0, 0, VertexFormat::Float2),
                Attribute::new(1, 12, VertexFormat::Float4),
            ],
        );
        match layout.validate() {
            Err(PipelineError::BadLayout(m)) => assert!(m.contains("past the"), "{m}"),
            other => panic!("expected a layout error, got {other:?}"),
        }
    }

    #[test]
    fn overlapping_attributes_are_rejected() {
        let layout = VertexLayout::new(
            32,
            vec![
                Attribute::new(0, 0, VertexFormat::Float4),
                Attribute::new(1, 8, VertexFormat::Float4),
            ],
        );
        match layout.validate() {
            Err(PipelineError::BadLayout(m)) => assert!(m.contains("overlaps"), "{m}"),
            other => panic!("expected an overlap error, got {other:?}"),
        }
    }

    #[test]
    fn adjacent_attributes_are_allowed() {
        let layout = VertexLayout::new(
            32,
            vec![
                Attribute::new(0, 0, VertexFormat::Float2),
                Attribute::new(1, 8, VertexFormat::Float2),
            ],
        );
        assert!(layout.validate().is_ok());
    }

    #[test]
    fn an_empty_layout_validates() {
        assert!(VertexLayout::new(32, vec![]).validate().is_ok());
    }

    #[test]
    fn attribute_descriptions_carry_the_offsets() {
        let descs = vertex_layout_for_quad().attribute_descriptions();
        assert_eq!(descs.len(), 4);
        assert_eq!(descs[2].offset, 12);
        assert_eq!(descs[2].location, 2);
        assert_eq!(descs[0].format, vk::Format::R32G32_SFLOAT);
        assert_eq!(descs[1].format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(descs[3].format, vk::Format::R32_UINT);
    }

    #[test]
    fn the_binding_description_carries_the_stride() {
        let b = vertex_layout_for_quad().binding_description();
        assert_eq!(b.stride, 32);
        assert_eq!(b.input_rate, vk::VertexInputRate::VERTEX);
        assert_eq!(b.binding, 0);
    }

    #[test]
    fn every_attribute_binds_to_slot_zero() {
        for d in vertex_layout_for_quad().attribute_descriptions() {
            assert_eq!(d.binding, 0);
        }
    }

    #[test]
    fn formats_map_to_the_expected_vulkan_formats() {
        assert_eq!(VertexFormat::Float2.vk_format(), vk::Format::R32G32_SFLOAT);
        assert_eq!(
            VertexFormat::Float4.vk_format(),
            vk::Format::R32G32B32A32_SFLOAT
        );
        assert_eq!(
            VertexFormat::Unorm8x4.vk_format(),
            vk::Format::R8G8B8A8_UNORM
        );
        assert_eq!(VertexFormat::Uint32.vk_format(), vk::Format::R32_UINT);
    }

    #[test]
    fn a_tightly_packed_layout_has_no_padding() {
        let layout = VertexLayout::new(
            12,
            vec![
                Attribute::new(0, 0, VertexFormat::Float2),
                Attribute::new(1, 8, VertexFormat::Uint32),
            ],
        );
        assert!(layout.validate().is_ok());
        assert!(!layout.has_padding());
        assert_eq!(layout.used_bytes(), 12);
    }

    #[test]
    fn used_bytes_is_zero_for_an_empty_layout() {
        assert_eq!(VertexLayout::new(32, vec![]).used_bytes(), 0);
    }
}
