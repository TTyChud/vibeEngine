//! Dynamic rendering: describing a pass's attachments without a render pass
//! object.
//!
//! Vulkan 1.3's `vkCmdBeginRendering` takes attachments inline, so a pass no
//! longer needs a `VkRenderPass` and `VkFramebuffer` pair. That is what lets the
//! render graph synthesize a pass per frame instead of owning fixed objects.

use ash::vk;

use crate::error::PipelineError;
use crate::state::BlendState;

/// One colour attachment in a pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorAttachment {
    /// The image view being rendered into.
    pub view: vk::ImageView,
    /// Layout the image is in when the pass begins.
    pub base_layout: vk::ImageLayout,
    /// Layout to transition to, or `UNDEFINED` to let the pass choose.
    pub final_layout: vk::ImageLayout,
    /// How fragments are combined with what is there.
    pub blend: BlendState,
    /// Clear the attachment when the pass begins.
    pub clear: Option<[f32; 4]>,
}

impl ColorAttachment {
    /// An opaque attachment with no clear.
    pub fn new(view: vk::ImageView) -> ColorAttachment {
        ColorAttachment {
            view,
            // Matches the layout the pre-pass barrier leaves the image in.
            // Claiming UNDEFINED here while the barrier produced
            // COLOR_ATTACHMENT_OPTIMAL makes the two disagree.
            base_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            blend: BlendState::Opaque,
            clear: None,
        }
    }

    /// Clear to a colour when the pass begins.
    pub fn with_clear(mut self, color: [f32; 4]) -> ColorAttachment {
        self.clear = Some(color);
        self
    }

    /// Blend fragments with what is already there.
    pub fn with_blend(mut self, blend: BlendState) -> ColorAttachment {
        self.blend = blend;
        self
    }

    /// Transition the image out of `UNDEFINED`, which discards its contents.
    ///
    /// This is what a render target wants: its previous contents are never read.
    pub fn discarding(mut self) -> ColorAttachment {
        self.base_layout = vk::ImageLayout::UNDEFINED;
        self
    }

    /// The load op implied by whether there is a clear.
    pub fn load_op(&self) -> vk::AttachmentLoadOp {
        if self.clear.is_some() {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::LOAD
        }
    }

    /// The store op: always write, since a pass's output is its purpose.
    pub fn store_op(&self) -> vk::AttachmentStoreOp {
        vk::AttachmentStoreOp::STORE
    }
}

/// A depth attachment, when the pass has one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthAttachment {
    /// The depth image view.
    pub view: vk::ImageView,
    /// Layout at the start of the pass.
    pub base_layout: vk::ImageLayout,
    /// Layout at the end of the pass.
    pub final_layout: vk::ImageLayout,
    /// Clear depth when the pass begins.
    pub clear: Option<f32>,
}

impl DepthAttachment {
    /// A depth attachment that loads what is there.
    pub fn new(view: vk::ImageView) -> DepthAttachment {
        DepthAttachment {
            view,
            base_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            clear: None,
        }
    }

    /// Clear depth when the pass begins.
    pub fn with_clear(mut self, depth: f32) -> DepthAttachment {
        self.clear = Some(depth);
        self
    }

    /// Discard the previous depth, which is what a fresh pass wants.
    pub fn discarding(mut self) -> DepthAttachment {
        self.base_layout = vk::ImageLayout::UNDEFINED;
        self
    }

    /// The load op implied by whether there is a clear.
    pub fn load_op(&self) -> vk::AttachmentLoadOp {
        if self.clear.is_some() {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::LOAD
        }
    }
}

/// The attachments and viewport of one rendering pass.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderPassDesc {
    /// Colour attachments, in slot order.
    pub colors: Vec<ColorAttachment>,
    /// The depth attachment, if any.
    pub depth: Option<DepthAttachment>,
    /// Viewport in pixels.
    pub viewport: (u32, u32),
    /// Scissor in pixels.
    pub scissor: (u32, u32),
    /// Area to shade, for tiled rendering; `None` shades everything.
    pub render_area: Option<vk::Rect2D>,
}

impl RenderPassDesc {
    /// A pass writing one colour attachment covering the whole target.
    pub fn color(attachment: ColorAttachment, width: u32, height: u32) -> RenderPassDesc {
        RenderPassDesc {
            colors: vec![attachment],
            depth: None,
            viewport: (width, height),
            scissor: (width, height),
            render_area: None,
        }
    }

    /// Add a depth attachment.
    pub fn with_depth(mut self, depth: DepthAttachment) -> RenderPassDesc {
        self.depth = Some(depth);
        self
    }

    /// Restrict the pass to a sub-rectangle, as a scissored post pass does.
    pub fn with_render_area(mut self, area: vk::Rect2D) -> RenderPassDesc {
        self.render_area = Some(area);
        self
    }

    /// The full-target rectangle.
    pub fn area(&self) -> vk::Rect2D {
        self.render_area.unwrap_or(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.viewport.0,
                height: self.viewport.1,
            },
        })
    }

    /// The viewport state, with a reverse-Z depth range.
    pub fn viewport_state(&self) -> vk::Viewport {
        vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.viewport.0 as f32,
            height: self.viewport.1 as f32,
            // Reverse-Z: near maps to 1 and far to 0, matching the engine's
            // projection matrices.
            min_depth: 1.0,
            max_depth: 0.0,
        }
    }

    /// The scissor state.
    pub fn scissor_state(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.scissor.0,
                height: self.scissor.1,
            },
        }
    }

    /// Check the pass is coherent.
    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.colors.len() > 8 {
            return Err(PipelineError::BadLayout(format!(
                "{} colour attachments exceeds the 8-slot limit",
                self.colors.len()
            )));
        }
        if self.viewport.0 == 0 || self.viewport.1 == 0 {
            return Err(PipelineError::BadLayout(
                "a render pass needs a non-zero viewport".to_string(),
            ));
        }
        Ok(())
    }
}

/// Fill the attachment slices for a pass, from a description.
///
/// The slices are supplied by the caller so they can live on the stack across
/// the `vkCmdBeginRendering` call, which is what the Vulkan API requires. The
/// returned [`vk::RenderingInfo`] borrows them, so it must not outlive them.
///
/// Returns the info plus a copy of what was written, so a caller can inspect
/// the attachments without keeping a borrow alive.
pub fn begin_rendering_info<'a>(
    desc: &RenderPassDesc,
    color_refs: &'a mut [vk::RenderingAttachmentInfo<'a>],
    depth_ref: &'a mut [vk::RenderingAttachmentInfo<'a>],
) -> (
    vk::RenderingInfo<'a>,
    Vec<vk::RenderingAttachmentInfo<'a>>,
    Option<vk::RenderingAttachmentInfo<'a>>,
) {
    // Dynamic rendering takes no per-attachment blend state: blending is
    // a property of the pipeline, not of the pass.
    for (info, attachment) in color_refs.iter_mut().zip(desc.colors.iter()) {
        *info = vk::RenderingAttachmentInfo {
            image_view: attachment.view,
            image_layout: attachment.base_layout,
            resolve_image_view: vk::ImageView::null(),
            resolve_mode: vk::ResolveModeFlags::empty(),
            load_op: attachment.load_op(),
            store_op: attachment.store_op(),
            clear_value: attachment
                .clear
                .map(|c| vk::ClearValue {
                    color: vk::ClearColorValue { float32: c },
                })
                .unwrap_or_default(),
            ..Default::default()
        };
    }

    if let (Some(slice), Some(depth)) = (depth_ref.first_mut(), desc.depth.as_ref()) {
        *slice = vk::RenderingAttachmentInfo {
            image_view: depth.view,
            image_layout: depth.base_layout,
            resolve_image_view: vk::ImageView::null(),
            resolve_mode: vk::ResolveModeFlags::empty(),
            load_op: depth.load_op(),
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: depth.clear.unwrap_or(0.0),
                    stencil: 0,
                },
            },
            ..Default::default()
        };
    }

    // Only pass a depth pointer when the pass actually has depth. A pointer to
    // a default-initialised attachment is a validation error, so the caller's
    // slice may be non-empty while the pass has no depth to bind.
    let depth_ptr = match desc.depth {
        Some(_) => depth_ref
            .first()
            .map(|d| d as *const _)
            .unwrap_or(std::ptr::null()),
        None => std::ptr::null(),
    };

    let info = vk::RenderingInfo {
        render_area: desc.area(),
        layer_count: 1,
        view_mask: 0,
        color_attachment_count: desc.colors.len() as u32,
        p_color_attachments: color_refs.as_ptr(),
        p_depth_attachment: depth_ptr,
        p_stencil_attachment: std::ptr::null(),
        ..Default::default()
    };
    let depth_copy = desc.depth.and(depth_ref.first().copied());
    (info, color_refs.to_vec(), depth_copy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle as _;

    fn view(n: u64) -> vk::ImageView {
        vk::ImageView::from_raw(n)
    }

    #[test]
    fn a_colour_attachment_with_no_clear_loads() {
        assert_eq!(
            ColorAttachment::new(view(1)).load_op(),
            vk::AttachmentLoadOp::LOAD
        );
    }

    #[test]
    fn a_colour_attachment_with_a_clear_clears() {
        let a = ColorAttachment::new(view(1)).with_clear([1.0, 0.0, 0.0, 1.0]);
        assert_eq!(a.load_op(), vk::AttachmentLoadOp::CLEAR);
    }

    #[test]
    fn a_colour_attachment_always_stores() {
        assert_eq!(
            ColorAttachment::new(view(1)).store_op(),
            vk::AttachmentStoreOp::STORE
        );
    }

    #[test]
    fn a_colour_attachment_starts_in_the_colour_attachment_layout() {
        // The base layout must match what the pre-pass barrier leaves behind.
        // Claiming UNDEFINED here while the barrier produced
        // COLOR_ATTACHMENT_OPTIMAL makes the two disagree, and the driver
        // rejects the pass.
        assert_eq!(
            ColorAttachment::new(view(1)).base_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
    }

    #[test]
    fn discarding_returns_the_attachment_to_undefined() {
        // A pass that really does discard its input starts from UNDEFINED and
        // needs no pre-pass transition.
        assert_eq!(
            ColorAttachment::new(view(1)).discarding().base_layout,
            vk::ImageLayout::UNDEFINED
        );
    }

    #[test]
    fn discarding_is_idempotent() {
        let a = ColorAttachment::new(view(1)).discarding();
        assert_eq!(a.base_layout, vk::ImageLayout::UNDEFINED);
    }

    #[test]
    fn a_colour_attachment_presents_at_the_end() {
        assert_eq!(
            ColorAttachment::new(view(1)).final_layout,
            vk::ImageLayout::PRESENT_SRC_KHR
        );
    }

    #[test]
    fn blending_is_carried_on_the_attachment() {
        let a = ColorAttachment::new(view(1)).with_blend(BlendState::AlphaBlend);
        assert_eq!(a.blend, BlendState::AlphaBlend);
    }

    #[test]
    fn a_depth_attachment_with_no_clear_loads() {
        assert_eq!(
            DepthAttachment::new(view(1)).load_op(),
            vk::AttachmentLoadOp::LOAD
        );
    }

    #[test]
    fn a_depth_attachment_with_a_clear_clears() {
        assert_eq!(
            DepthAttachment::new(view(1)).with_clear(1.0).load_op(),
            vk::AttachmentLoadOp::CLEAR
        );
    }

    #[test]
    fn discarding_depth_uses_undefined() {
        assert_eq!(
            DepthAttachment::new(view(1)).discarding().base_layout,
            vk::ImageLayout::UNDEFINED
        );
    }

    #[test]
    fn a_pass_covers_the_whole_target_by_default() {
        let d = RenderPassDesc::color(ColorAttachment::new(view(1)), 800, 600);
        let area = d.area();
        assert_eq!(area.extent.width, 800);
        assert_eq!(area.extent.height, 600);
        assert_eq!(area.offset, vk::Offset2D { x: 0, y: 0 });
    }

    #[test]
    fn a_restricted_area_is_honoured() {
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 10, y: 20 },
            extent: vk::Extent2D {
                width: 100,
                height: 200,
            },
        };
        let d =
            RenderPassDesc::color(ColorAttachment::new(view(1)), 800, 600).with_render_area(area);
        assert_eq!(d.area(), area);
    }

    #[test]
    fn the_viewport_uses_a_reverse_z_range() {
        let d = RenderPassDesc::color(ColorAttachment::new(view(1)), 800, 600);
        let v = d.viewport_state();
        assert_eq!(v.min_depth, 1.0);
        assert_eq!(v.max_depth, 0.0);
        assert_eq!(v.width, 800.0);
        assert_eq!(v.height, 600.0);
    }

    #[test]
    fn the_scissor_matches_the_viewport_by_default() {
        let d = RenderPassDesc::color(ColorAttachment::new(view(1)), 320, 240);
        assert_eq!(d.scissor_state().extent.width, 320);
        assert_eq!(d.scissor_state().extent.height, 240);
    }

    #[test]
    fn a_valid_pass_validates() {
        let d = RenderPassDesc::color(ColorAttachment::new(view(1)), 800, 600);
        assert!(d.validate().is_ok());
    }

    #[test]
    fn a_zero_viewport_is_rejected() {
        let d = RenderPassDesc::color(ColorAttachment::new(view(1)), 0, 600);
        assert!(d.validate().is_err());
    }

    #[test]
    fn too_many_attachments_are_rejected() {
        let d = RenderPassDesc {
            colors: (0..9).map(|i| ColorAttachment::new(view(i))).collect(),
            depth: None,
            viewport: (800, 600),
            scissor: (800, 600),
            render_area: None,
        };
        assert!(d.validate().is_err());
    }

    #[test]
    fn depth_makes_the_pass_valid() {
        let d = RenderPassDesc::color(ColorAttachment::new(view(1)), 800, 600)
            .with_depth(DepthAttachment::new(view(2)).with_clear(1.0));
        assert!(d.validate().is_ok());
        assert!(d.depth.is_some());
    }

    #[test]
    fn rendering_info_counts_the_colour_attachments() {
        let d = RenderPassDesc::color(ColorAttachment::new(view(1)), 800, 600);
        let mut colors = [vk::RenderingAttachmentInfo::default(); 1];
        let mut depth = [vk::RenderingAttachmentInfo::default(); 1];
        // Declared here and written in the block below, where the slices are
        // still borrowed.
        let view_seen;
        let count;
        let layers;
        {
            let (info, filled, _filled_depth) = begin_rendering_info(&d, &mut colors, &mut depth);
            count = info.color_attachment_count;
            layers = info.layer_count;
            view_seen = filled[0].image_view;
        }
        assert_eq!((count, layers), (1, 1));
        assert_eq!(view_seen, view(1));
    }

    #[test]
    fn rendering_info_carries_the_views_and_layouts() {
        let d = RenderPassDesc::color(ColorAttachment::new(view(7)).discarding(), 800, 600);
        let mut colors = [vk::RenderingAttachmentInfo::default(); 1];
        let mut depth = [vk::RenderingAttachmentInfo::default(); 1];
        let (seen_view, seen_layout) = {
            let (_info, filled, _filled_depth) = begin_rendering_info(&d, &mut colors, &mut depth);
            (filled[0].image_view, filled[0].image_layout)
        };
        assert_eq!(seen_view, view(7));
        assert_eq!(seen_layout, vk::ImageLayout::UNDEFINED);
    }

    #[test]
    fn rendering_info_turns_a_clear_into_a_clear_load_op() {
        let d = RenderPassDesc::color(
            ColorAttachment::new(view(1)).with_clear([0.0, 0.0, 1.0, 1.0]),
            800,
            600,
        );
        let mut colors = [vk::RenderingAttachmentInfo::default(); 1];
        let mut depth = [vk::RenderingAttachmentInfo::default(); 1];
        let (load_op, cleared) = {
            let (_info, filled, _filled_depth) = begin_rendering_info(&d, &mut colors, &mut depth);
            // ClearValue is a union, so its colour field is read directly.
            (filled[0].load_op, unsafe {
                filled[0].clear_value.color.float32
            })
        };
        assert_eq!(load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(cleared, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn a_pass_without_depth_passes_a_null_depth_pointer() {
        let d = RenderPassDesc::color(ColorAttachment::new(view(1)), 800, 600);
        let mut colors = [vk::RenderingAttachmentInfo::default(); 1];
        let mut depth = [vk::RenderingAttachmentInfo::default(); 1];
        let has_depth = {
            let (info, _filled, _filled_depth) = begin_rendering_info(&d, &mut colors, &mut depth);
            !info.p_depth_attachment.is_null()
        };
        assert!(!has_depth);
    }

    #[test]
    fn a_pass_with_depth_passes_the_depth_view() {
        let d = RenderPassDesc::color(ColorAttachment::new(view(1)), 800, 600)
            .with_depth(DepthAttachment::new(view(2)).with_clear(1.0));
        let mut colors = [vk::RenderingAttachmentInfo::default(); 1];
        let mut depth = [vk::RenderingAttachmentInfo::default(); 1];
        let (has_depth, depth_view, load_op) = {
            let (info, _filled, filled_depth) = begin_rendering_info(&d, &mut colors, &mut depth);
            (
                !info.p_depth_attachment.is_null(),
                filled_depth.map(|a| a.image_view),
                filled_depth.map(|a| a.load_op),
            )
        };
        assert!(has_depth);
        assert_eq!(depth_view, Some(view(2)));
        assert_eq!(load_op, Some(vk::AttachmentLoadOp::CLEAR));
    }

    #[test]
    fn multiple_colour_attachments_get_distinct_layers() {
        let d = RenderPassDesc {
            colors: vec![ColorAttachment::new(view(1)), ColorAttachment::new(view(2))],
            depth: None,
            viewport: (800, 600),
            scissor: (800, 600),
            render_area: None,
        };
        let mut colors = [vk::RenderingAttachmentInfo::default(); 2];
        let mut depth = [vk::RenderingAttachmentInfo::default(); 1];
        let (count, first, second) = {
            let (info, filled, _filled_depth) = begin_rendering_info(&d, &mut colors, &mut depth);
            (
                info.color_attachment_count,
                filled[0].image_view,
                filled[1].image_view,
            )
        };
        assert_eq!(count, 2);
        assert_eq!(first, view(1));
        assert_eq!(second, view(2));
    }

    #[test]
    fn the_render_area_reaches_the_rendering_info() {
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 4, y: 8 },
            extent: vk::Extent2D {
                width: 100,
                height: 100,
            },
        };
        let d =
            RenderPassDesc::color(ColorAttachment::new(view(1)), 800, 600).with_render_area(area);
        let mut colors = [vk::RenderingAttachmentInfo::default(); 1];
        let mut depth = [vk::RenderingAttachmentInfo::default(); 1];
        let got = {
            let (info, _filled, _filled_depth) = begin_rendering_info(&d, &mut colors, &mut depth);
            info.render_area
        };
        assert_eq!(got, area);
    }
}
