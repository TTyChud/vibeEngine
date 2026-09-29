//! Recording and drawing a frame of batched quads.

use ash::vk;
use vibe_pipeline::{ColorAttachment, RenderPassDesc};
use vibe_rhi::{BatchStats, Camera2D};
use vibe_vk::{Access, Barrier, BarrierEncoder, Stage, image_to_shader_read};

use crate::batch::Batcher;
use crate::error::RenderError;
use crate::pipeline::QuadPipeline;

/// What the renderer needs to draw.
#[derive(Debug, Clone, PartialEq)]
pub struct RendererDesc {
    /// The 2D camera, which supplies the view-projection matrix.
    pub camera: Camera2D,
    /// The target's width in pixels.
    pub width: u32,
    /// The target's height in pixels.
    pub height: u32,
    /// The target's format.
    pub format: vk::Format,
    /// The clear colour for the frame.
    pub clear_color: [f32; 4],
}

impl RendererDesc {
    /// A description for a target of the given size.
    pub fn new(width: u32, height: u32, format: vk::Format) -> RendererDesc {
        RendererDesc {
            camera: Camera2D::screen(width as f32, height as f32),
            width,
            height,
            format,
            clear_color: [0.02, 0.02, 0.03, 1.0],
        }
    }

    /// A description with a chosen clear colour.
    pub fn with_clear(mut self, color: [f32; 4]) -> RendererDesc {
        self.clear_color = color;
        self
    }
}

/// Draws batched quads into a swapchain image.
pub struct QuadRenderer {
    pipeline: QuadPipeline,
    batcher: Batcher,
    desc: RendererDesc,
    /// The last frame's statistics.
    pub stats: BatchStats,
}

impl std::fmt::Debug for QuadRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuadRenderer")
            .field("batcher", &self.batcher)
            .finish()
    }
}

impl QuadRenderer {
    /// A renderer drawing with a built pipeline.
    pub fn new(pipeline: QuadPipeline, desc: RendererDesc) -> QuadRenderer {
        QuadRenderer {
            pipeline,
            batcher: Batcher::default(),
            desc,
            stats: BatchStats::default(),
        }
    }

    /// The batcher, for adding quads before a frame.
    pub fn batcher(&self) -> &Batcher {
        &self.batcher
    }

    /// Mutably borrow the batcher.
    pub fn batcher_mut(&mut self) -> &mut Batcher {
        &mut self.batcher
    }

    /// The render description this renderer uses.
    pub fn desc(&self) -> &RendererDesc {
        &self.desc
    }

    /// Change the render description, e.g. after a resize.
    pub fn set_desc(&mut self, desc: RendererDesc) {
        self.desc = desc;
    }

    /// The pipeline handle.
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline.pipeline
    }

    /// Record a frame drawing the batch into `target`.
    ///
    /// `device` must be a device whose function pointers resolve: ash 0.38
    /// erases its own tables, so `vibe_vk::LogicalDevice` rebuilds them.
    ///
    /// # Safety
    ///
    /// `command_buffer` must be recording, `vertex_buffer` must hold
    /// [`Batcher::vertices`], and `target` must be a view of the image the frame
    /// will present.
    pub unsafe fn record(
        &mut self,
        device: &ash::Device,
        rendering: &ash::khr::dynamic_rendering::Device,
        command_buffer: vk::CommandBuffer,
        target: vk::ImageView,
        vertex_buffer: vk::Buffer,
        offset: vk::DeviceSize,
    ) -> Result<(), RenderError> {
        self.stats = self.batcher.stats();

        let attachment = ColorAttachment::new(target).with_clear(self.desc.clear_color);
        let pass = RenderPassDesc::color(attachment, self.desc.width, self.desc.height);
        pass.validate()
            .map_err(|e| RenderError::Pipeline(e.to_string()))?;

        // Reverse-Z: near maps to 1 and far to 0, matching the engine's
        // projection matrices and the pipeline's viewport state.
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.desc.width as f32,
            height: self.desc.height as f32,
            min_depth: 1.0,
            max_depth: 0.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.desc.width,
                height: self.desc.height,
            },
        };

        let mut colors = [vk::RenderingAttachmentInfo::default(); 1];
        let mut depth = [vk::RenderingAttachmentInfo::default(); 1];
        let (info, _, _) = vibe_pipeline::begin_rendering_info(&pass, &mut colors, &mut depth);

        unsafe {
            rendering.cmd_begin_rendering(command_buffer, &info);

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);
            device.cmd_set_primitive_topology(command_buffer, vk::PrimitiveTopology::TRIANGLE_LIST);
            device.cmd_set_cull_mode(command_buffer, vk::CullModeFlags::NONE);
            // 2D quads have no depth buffer, so the depth test is off.
            device.cmd_set_depth_test_enable(command_buffer, false);

            // The camera is a single mat4 in push constants: no descriptor, no
            // rebind, and it is the only per-frame data the shader needs.
            let matrix = self.desc.camera.view_projection;
            device.cmd_push_constants(
                command_buffer,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&matrix),
            );

            if !self.batcher.is_empty() && std::env::var("VIBE_SKIP_DRAW").is_err() {
                device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[offset]);
                device.cmd_draw(command_buffer, self.batcher.vertex_count() as u32, 1, 0, 0);
            }

            rendering.cmd_end_rendering(command_buffer);
        }
        Ok(())
    }

    /// The barriers needed before and after a frame draws into an image.
    ///
    /// Before: the image becomes a colour attachment. After: it leaves the
    /// attachment layout so the swapchain can present it. The two belong in
    /// different command buffers, because the second is only valid after the
    /// render pass has ended.
    pub fn frame_barriers(&self, target: vk::Image) -> [Barrier; 2] {
        [
            Barrier::image(target, vk::ImageAspectFlags::COLOR)
                .src(Stage::None, Access::None)
                .dst(Stage::ColorAttachmentOutput, Access::ColorAttachmentWrite),
            image_to_shader_read(target, vk::ImageAspectFlags::COLOR)
                .src(Stage::ColorAttachmentOutput, Access::ColorAttachmentWrite)
                .dst(Stage::None, Access::None),
        ]
    }

    /// Record the frame's barriers into an encoder.
    pub fn encode_barriers(&self, encoder: &mut BarrierEncoder, target: vk::Image) {
        encoder.extend(self.frame_barriers(target));
    }

    /// The bytes a batch of quads occupies, which the caller must have uploaded
    /// before recording.
    pub fn required_vertex_bytes(&self) -> usize {
        self.batcher.byte_len()
    }
}

/// How many bytes one quad occupies in the vertex buffer.
///
/// Exposed so a caller can size its staging buffer without depending on the
/// batcher's internals.
pub const BYTES_PER_QUAD: usize = 6 * std::mem::size_of::<vibe_rhi::QuadVertex>();

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle as _;

    fn desc() -> RendererDesc {
        RendererDesc::new(800, 600, vk::Format::B8G8R8A8_SRGB)
    }

    #[test]
    fn a_description_defaults_to_a_dark_opaque_clear() {
        let d = desc();
        assert_eq!(d.width, 800);
        assert_eq!(d.height, 600);
        assert!(d.clear_color[3] > 0.9, "the clear must be opaque");
    }

    #[test]
    fn the_clear_colour_can_be_chosen() {
        assert_eq!(
            desc().with_clear([1.0, 0.0, 0.0, 1.0]).clear_color,
            [1.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn the_camera_is_built_for_the_target_size() {
        assert_eq!(
            desc().camera.visible_bounds.max,
            glam::Vec2::new(800.0, 600.0)
        );
    }

    #[test]
    fn an_empty_batch_costs_nothing() {
        let b = Batcher::default();
        assert_eq!(b.stats().quads, 0);
        assert_eq!(b.stats().draw_calls, 0);
    }

    #[test]
    fn the_pre_pass_barrier_targets_colour_attachment_output() {
        let image = vk::Image::from_raw(0x1234);
        let pre = Barrier::image(image, vk::ImageAspectFlags::COLOR)
            .src(Stage::None, Access::None)
            .dst(Stage::ColorAttachmentOutput, Access::ColorAttachmentWrite);
        assert_eq!(pre.dst_stage, Stage::ColorAttachmentOutput);
        assert!(!pre.is_noop());
    }

    #[test]
    fn the_post_pass_barrier_leaves_the_attachment_layout() {
        let image = vk::Image::from_raw(0x1234);
        let post = image_to_shader_read(image, vk::ImageAspectFlags::COLOR)
            .src(Stage::ColorAttachmentOutput, Access::ColorAttachmentWrite)
            .dst(Stage::None, Access::None);
        assert_eq!(post.src_stage, Stage::ColorAttachmentOutput);
        assert!(!post.is_noop());
    }

    #[test]
    fn the_byte_count_per_quad_is_the_stride() {
        assert_eq!(BYTES_PER_QUAD, 6 * 32);
        assert_eq!(BYTES_PER_QUAD, 192);
    }

    #[test]
    fn a_batches_byte_count_matches_the_constant() {
        let mut b = Batcher::new(16);
        b.push_quads(
            glam::Vec2::ZERO,
            glam::Vec2::ONE,
            [255; 4],
            &vibe_rhi::SubTexture::full(0),
            3,
        )
        .unwrap();
        assert_eq!(b.byte_len(), 3 * BYTES_PER_QUAD);
    }
}
