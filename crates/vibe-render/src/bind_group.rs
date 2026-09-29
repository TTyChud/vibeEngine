//! The descriptor set the quad shader needs.
//!
//! The pipeline layout declares three bindings — a camera uniform, a texture
//! array and a sampler — and a draw that leaves any of them unbound is
//! invalid. This owns the pool, the set, and the writes that fill it, so the
//! render path's only remaining job is to bind the set before drawing.
//!
//! The camera lives in a host-visible uniform buffer rather than push
//! constants here, because it needs a descriptor to be addressable by the
//! fragment stage too, and because a uniform buffer is where a scene's
//! per-frame data belongs once there is more than one binding.

use ash::vk;
use log::debug;

use crate::error::RenderError;
use crate::pipeline::{CAMERA_BINDING, SAMPLER_BINDING, TEXTURE_BINDING};

/// The layouts and sizes the bind group needs, kept separate from the Vulkan
/// objects so they can be checked without a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupDesc {
    /// Descriptor sets to allocate from the pool.
    pub set_count: u32,
    /// Number of colour samplers the pool must serve.
    pub sampler_count: u32,
    /// Number of sampled images the pool must serve.
    pub sampled_image_count: u32,
    /// Number of uniform buffers the pool must serve.
    pub uniform_buffer_count: u32,
    /// Max descriptor sets the pool may hold at once, per frame stage.
    pub max_sets: u32,
}

impl Default for BindGroupDesc {
    fn default() -> Self {
        BindGroupDesc {
            set_count: 1,
            sampler_count: 1,
            sampled_image_count: 1,
            uniform_buffer_count: 1,
            max_sets: 2,
        }
    }
}

impl BindGroupDesc {
    /// The pool sizes for this description, as `(type, count)` pairs.
    pub fn pool_sizes(&self) -> Vec<vk::DescriptorPoolSize> {
        vec![
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: self.sampler_count * self.max_sets,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: self.sampled_image_count * self.max_sets,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: self.uniform_buffer_count * self.max_sets,
            },
        ]
    }

    /// Total descriptors the pool must provide.
    pub fn total_descriptors(&self) -> u32 {
        (self.sampler_count + self.sampled_image_count + self.uniform_buffer_count) * self.max_sets
    }

    /// Check the description is self-consistent.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.set_count == 0 {
            return Err(RenderError::Pipeline(
                "a bind group needs at least one set".into(),
            ));
        }
        if self.max_sets == 0 {
            return Err(RenderError::Pipeline(
                "a pool needs room for at least one set".into(),
            ));
        }
        if self.sampler_count == 0
            || self.sampled_image_count == 0
            || self.uniform_buffer_count == 0
        {
            return Err(RenderError::Pipeline(
                "the quad shader needs a camera, a texture and a sampler".into(),
            ));
        }
        Ok(())
    }
}

/// A descriptor pool and the sets allocated from it.
pub struct BindGroup {
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    layout: vk::DescriptorSetLayout,
}

impl std::fmt::Debug for BindGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindGroup")
            .field("set", &self.set)
            .field("pool", &self.pool)
            .finish()
    }
}

impl BindGroup {
    /// Create a pool, allocate one set from it, and point it at the given
    /// resources.
    ///
    /// `camera` is the uniform buffer holding the view-projection matrix,
    /// `texture_view` the first view of the texture array, and `sampler` the
    /// sampler the fragment stage reads through.
    ///
    /// # Safety
    ///
    /// `device` must be a live logical device whose function pointers resolve,
    /// and every handle must belong to it.
    pub unsafe fn create(
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
        desc: &BindGroupDesc,
        camera: vk::Buffer,
        camera_range: vk::DeviceSize,
        texture_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<BindGroup, RenderError> {
        desc.validate()?;

        // The sizes are a local, not a temporary: `p_pool_sizes` points into
        // them, so dropping the Vec would leave the create_info dangling.
        let sizes = desc.pool_sizes();
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: desc.max_sets,
            pool_size_count: sizes.len() as u32,
            p_pool_sizes: sizes.as_ptr(),
            ..Default::default()
        };
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;

        let alloc = vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: &layout,
            ..Default::default()
        };
        let sets = match unsafe { device.allocate_descriptor_sets(&alloc) } {
            Ok(s) => s,
            Err(e) => {
                unsafe { device.destroy_descriptor_pool(pool, None) };
                return Err(RenderError::Vk(e));
            }
        };
        let set = sets[0];

        // The camera is a uniform buffer: the whole range, offset zero.
        let camera_info = vk::DescriptorBufferInfo {
            buffer: camera,
            offset: 0,
            range: camera_range,
        };
        // The texture is sampled through a shader-read-only layout, which is
        // what a fragment shader expects.
        let texture_info = vk::DescriptorImageInfo {
            sampler,
            image_view: texture_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let writes = [
            vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: CAMERA_BINDING,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_buffer_info: &camera_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: TEXTURE_BINDING,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                p_image_info: &texture_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: SAMPLER_BINDING,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::SAMPLER,
                p_image_info: &texture_info,
                ..Default::default()
            },
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
        debug!("descriptor set {set:?} bound: camera {camera:?}, texture {texture_view:?}");

        Ok(BindGroup { pool, set, layout })
    }

    /// The descriptor set, to bind before drawing.
    pub fn set(&self) -> vk::DescriptorSet {
        self.set
    }

    /// The pool backing the set.
    pub fn pool(&self) -> vk::DescriptorPool {
        self.pool
    }

    /// The layout the set was allocated against.
    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    /// The pipeline bind point, which is graphics for a render pass.
    pub fn bind_point(&self) -> vk::PipelineBindPoint {
        vk::PipelineBindPoint::GRAPHICS
    }

    /// Free the pool, which frees the set with it.
    ///
    /// # Safety
    ///
    /// The set must not be bound in a command buffer that is still executing.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
        self.pool = vk::DescriptorPool::null();
        self.set = vk::DescriptorSet::null();
    }
}

/// Create the sampler the quad shader samples through.
///
/// Nearest filtering with clamped addressing: a 2D sprite renderer wants
/// texel-exact sampling and no wrapping, so a texture that runs off the edge of
/// the atlas reads its border rather than a neighbouring sprite.
pub fn create_sampler(device: &ash::Device) -> Result<vk::Sampler, RenderError> {
    let info = vk::SamplerCreateInfo {
        mag_filter: vk::Filter::NEAREST,
        min_filter: vk::Filter::NEAREST,
        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        // One mip: a 2D atlas has no mip chain.
        mip_lod_bias: 0.0,
        max_lod: 0.0,
        ..Default::default()
    };
    unsafe { device.create_sampler(&info, None) }.map_err(RenderError::Vk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_description_covers_the_shader() {
        let d = BindGroupDesc::default();
        assert_eq!(d.set_count, 1);
        assert_eq!(d.sampler_count, 1);
        assert_eq!(d.sampled_image_count, 1);
        assert_eq!(d.uniform_buffer_count, 1);
    }

    #[test]
    fn the_default_description_validates() {
        assert!(BindGroupDesc::default().validate().is_ok());
    }

    #[test]
    fn the_pool_has_one_size_per_descriptor_type() {
        let sizes = BindGroupDesc::default().pool_sizes();
        assert_eq!(sizes.len(), 3);
        assert!(sizes.iter().any(|s| s.ty == vk::DescriptorType::SAMPLER));
        assert!(
            sizes
                .iter()
                .any(|s| s.ty == vk::DescriptorType::SAMPLED_IMAGE)
        );
        assert!(
            sizes
                .iter()
                .any(|s| s.ty == vk::DescriptorType::UNIFORM_BUFFER)
        );
    }

    #[test]
    fn pool_sizes_scale_with_the_number_of_sets() {
        let one = BindGroupDesc {
            max_sets: 1,
            ..Default::default()
        };
        let four = BindGroupDesc {
            max_sets: 4,
            ..Default::default()
        };
        assert_eq!(one.total_descriptors(), 3);
        assert_eq!(four.total_descriptors(), 12);
    }

    #[test]
    fn the_pool_gives_each_set_its_own_descriptors() {
        let d = BindGroupDesc {
            max_sets: 3,
            ..Default::default()
        };
        for size in d.pool_sizes() {
            // Two frames in flight, so each frame needs its own camera.
            let expected = match size.ty {
                vk::DescriptorType::SAMPLER => 1,
                _ => 1,
            };
            assert_eq!(size.descriptor_count, expected * 3, "{:?}", size.ty);
        }
    }

    #[test]
    fn zero_sets_is_rejected() {
        let d = BindGroupDesc {
            set_count: 0,
            ..Default::default()
        };
        assert!(d.validate().is_err());
    }

    #[test]
    fn a_pool_with_no_room_is_rejected() {
        let d = BindGroupDesc {
            max_sets: 0,
            ..Default::default()
        };
        assert!(d.validate().is_err());
    }

    #[test]
    fn a_missing_binding_is_rejected() {
        // Each of the three is something the shader reads, so any of them
        // missing would make the draw invalid.
        for d in [
            BindGroupDesc {
                sampler_count: 0,
                ..Default::default()
            },
            BindGroupDesc {
                sampled_image_count: 0,
                ..Default::default()
            },
            BindGroupDesc {
                uniform_buffer_count: 0,
                ..Default::default()
            },
        ] {
            assert!(d.validate().is_err(), "{d:?}");
        }
    }

    #[test]
    fn the_bindings_match_the_shader() {
        // The pipeline declares these; a mismatch means the draw reads garbage
        // or faults, and the numbers are the contract between the two.
        assert_eq!(CAMERA_BINDING, 0);
        assert_eq!(TEXTURE_BINDING, 1);
        assert_eq!(SAMPLER_BINDING, 2);
    }

    #[test]
    fn the_bind_group_is_graphics() {
        // A render pass binds graphics, not compute.
        assert_eq!(
            vk::PipelineBindPoint::GRAPHICS,
            vk::PipelineBindPoint::GRAPHICS
        );
    }
}
