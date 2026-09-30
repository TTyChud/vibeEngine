//! The editor, running in a window.
//!
//! This is the interactive check on the editor: it builds the UI pipeline, runs
//! real egui frames against a real window, and keeps drawing until the window
//! closes. A frame that renders once proves the pipeline; a loop that survives a
//! redraw, a resize and a click proves the parts around it — the input path, the
//! buffer growth and the swapchain rebuild.
//!
//! Run with: cargo run -p vibe-editor --example editor
//!
//! Set VIBE_VALIDATION=1 to have the validation layer report anything wrong.

// Every Vulkan entry point is unsafe, and this example's entire purpose is to
// call them. The unsafe contract is documented on each function that has one
// beyond the call itself.
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::CStr;

use ash::vk;
use vibe_editor::{Editor, FontAtlas, Panel, UiDrawData};
use vibe_render::ui::{ScreenTransform, UiBindGroup, build_ui_pipeline, scissor_from_bounds};
use vibe_vk::memory::{BufferUsage, GpuBuffer, GpuImage, HeapLayout, ImageUsage, MemoryNeed};
use vibe_window::{SurfaceKind, surface};

/// How long to wait for the GPU, in nanoseconds.
const TIMEOUT: u64 = 10_000_000_000;

/// The background the panels sit over, in linear colour.
const BACKDROP: [f32; 4] = [40.0 / 255.0, 44.0 / 255.0, 52.0 / 255.0, 1.0];

/// The same colour as bytes, which is what a readback compares against. The
/// surface is BGRA, so the components are reversed from BACKDROP.
const BACKDROP_BGR: [u8; 3] = [52, 44, 40];

/// The largest vertex buffer the loop will grow to.
///
/// A frame wanting more is truncated rather than allowed to allocate without
/// bound: a pathological frame should cost a dropped panel, not the machine.
const MAX_VERTEX_BYTES: usize = 8 * 1024 * 1024;

fn main() {
    if std::env::var("VIBE_VALIDATION").is_ok() {
        vibe_vk::validation::enable_layer();
    }
    let event_loop = match winit::event_loop::EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("FAIL event loop: {e}");
            return;
        }
    };
    // The device needs a window's handles to create a surface, so setup waits
    // for `resumed`, where the window exists.
    let mut app = App::new();
    let _ = event_loop.run_app(&mut app);
    println!("drew {} frame(s)", app.frames);
    if let Some(gpu) = app.gpu.as_ref() {
        gpu.logical.wait_idle();
    }
    let _ = (CStr::from_bytes_with_nul(b"x\0"), SurfaceKind::Wayland);
}

/// What `setup` hands back to the caller.
struct Setup {
    gpu: Gpu,
    messenger: Option<vibe_vk::validation::ValidationMessenger>,
    surface: vk::SurfaceKHR,
}

/// Everything a redraw needs, created once and kept.
struct Gpu {
    logical: vibe_vk::LogicalDevice,
    /// Kept because a swapchain rebuild needs it and re-enumerating would be a
    /// chance to pick a different device.
    physical_device: vk::PhysicalDevice,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vibe_vk::Swapchain,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    pipeline: vibe_render::ui::UiPipeline,
    bind_group: UiBindGroup,
    /// Kept alive, never read: the descriptor set points at it, so dropping it
    /// would leave a bound set naming a destroyed buffer.
    #[allow(dead_code)]
    transform_buffer: GpuBuffer,
    vertex_buffer: GpuBuffer,
    pool: vibe_frame::CommandPool,
    present_pool: vibe_frame::CommandPool,
    /// A pool of its own for the font-atlas upload.
    ///
    /// Separate because the upload happens while the frame's pools have buffers
    /// in the recording state, and beginning a buffer in a pool that still has
    /// one open is what this driver crashes on.
    atlas_pool: vibe_frame::CommandPool,
    resources: vibe_frame::FrameResources,
    heap: HeapLayout,
    rendering: ash::khr::dynamic_rendering::Device,
    sync2: ash::khr::synchronization2::Device,
    /// The font atlas as a single-channel texture the UI shader samples.
    ///
    /// egui rasterises its font atlas once, into a single-channel coverage mask.
    /// It arrives as an RGBA image whose other three channels are unused, so it
    /// is reduced to one byte per texel and uploaded as `R8_UNORM`: sampling
    /// the red channel of an RGBA atlas would work and would move four times
    /// the bytes for a mask.
    atlas: Option<GpuImage>,
    /// A sampler for the atlas: linear, clamped.
    ///
    /// Linear because a glyph is magnified from a small atlas cell to the
    /// screen size and nearest filtering makes it a staircase of hard pixels;
    /// clamped because a UV outside the atlas should show the edge glyph rather
    /// than wrap to a different letter.
    atlas_sampler: vk::Sampler,
    /// Staging for an atlas upload, sized to the largest atlas seen.
    atlas_staging: GpuBuffer,
    /// The atlas currently on the GPU, so an unchanged one is not re-uploaded.
    uploaded_atlas: FontAtlas,
}

/// The running editor.
struct App {
    editor: Editor,
    /// The validation messenger, held for the run so its destructor does not
    /// run while the layer could still route to it.
    _messenger: Option<vibe_vk::validation::ValidationMessenger>,
    /// egui's winit integration, which turns window events into input.
    platform: Option<egui_winit::State>,
    window: Option<vibe_window::Window>,
    gpu: Option<Gpu>,
    surface: vk::SurfaceKHR,
    frames: u64,
    /// True once the first frame has been read back, which stalls the pipeline
    /// and is only worth doing once.
    readback_done: bool,
}

impl App {
    /// A new editor with a small scene, over the content of the temp directory.
    fn new() -> App {
        let mut editor = Editor::new();
        for name in ["Player", "Enemy", "Camera"] {
            editor.spawn(name);
        }
        editor.content.root = std::env::temp_dir();
        editor.content.rescan();
        // The hierarchy is the panel worth opening first: it is where the
        // scene's contents are, and clicking a row there moves the gizmo.
        editor.panel = Panel::Hierarchy;
        // A camera on the diagonal aimed at the origin, so the gizmo's handles
        // are all on screen rather than one pointing straight at it.
        editor.camera.position = glam::Vec3::new(6.0, 5.0, 6.0);
        editor.camera.yaw = std::f32::consts::FRAC_PI_4;
        editor.camera.pitch = -0.55;
        // Selecting the first entity puts the gizmo on it from the first frame.
        if let Some(first) = editor.world.entities().next() {
            editor.hierarchy.select(first);
            editor.session.selection = Some(first);
        }
        App {
            editor,
            _messenger: None,
            platform: None,
            window: None,
            gpu: None,
            surface: vk::SurfaceKHR::null(),
            frames: 0,
            readback_done: false,
        }
    }

    /// Draw one frame and present it.
    fn render(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let (w, h) = window.size();
        if w == 0 || h == 0 {
            // A minimised window has no images to draw into, and presenting into
            // one is a validation error rather than a no-op.
            return;
        }
        let Some(platform) = self.platform.as_mut() else {
            return;
        };
        let input = platform.take_egui_input(&window.inner);
        // A consumed shortcut must not also reach the camera, or F1 would both
        // switch panels and move the view.
        self.editor.handle_shortcut();
        let data = self.editor.draw(input, egui::Vec2::new(w as f32, h as f32));
        self.frames += 1;

        if self.frames == 1 || self.frames % 120 == 0 {
            println!(
                "[frame {}] egui: {} batch(es), {} vertex(es) -> {} byte(s); \
                 gizmo {} handle(s)",
                self.frames,
                data.batch_count(),
                data.len(),
                data.byte_len(),
                self.editor.viewport.handle_count,
            );
        }

        let Some(gpu) = self.gpu.as_mut() else {
            // The first frame runs before setup finishes, so there is nothing to
            // read back from yet. Returning here without touching the readback
            // flag is the point: consuming it would leave every later frame with
            // the one readback already spent, and the frame would never be read
            // back at all.
            return;
        };
        // Once only, and only now that there is a GPU to read from. A swapchain
        // image belongs to the presentation engine between the present and the
        // next acquire, so copying it every frame races the compositor rather
        // than merely costing a pipeline stall.
        let readback = !self.readback_done;
        self.readback_done = true;
        // SAFETY: every object is live, and the swapchain matches the size
        // checked above because a resize rebuilds it before the next redraw.
        unsafe { App::present(gpu, &data, &data.atlas, w, h, readback) };
    }

    /// Record, submit and present the UI.
    ///
    /// # Safety
    ///
    /// Every Vulkan object must be live and the swapchain must match the
    /// window's current size.
    unsafe fn present(
        gpu: &mut Gpu,
        data: &UiDrawData,
        atlas: &FontAtlas,
        w: u32,
        h: u32,
        readback: bool,
    ) {
        // The atlas upload takes &mut gpu, so it runs before the device
        // reference is taken for the rest of the frame.
        let Some(queue) = gpu.logical.graphics_queue() else {
            return;
        };
        unsafe { upload_atlas(gpu, atlas, queue) };
        let device = &gpu.logical.device;

        // Grow the vertex buffer when a frame needs more. Every batch is drawn
        // from one flat stream, so one buffer covers all of them.
        let needed = data.byte_len().max(32);
        if needed > gpu.vertex_buffer.size() as usize {
            let want = (needed.next_power_of_two() as u64).min(MAX_VERTEX_BYTES as u64);
            match unsafe {
                GpuBuffer::create(
                    device,
                    &gpu.heap,
                    want,
                    BufferUsage {
                        vertex: true,
                        ..Default::default()
                    },
                    MemoryNeed::upload(),
                )
            } {
                Ok(b) if b.size() as usize >= needed => gpu.vertex_buffer = b,
                _ => {
                    eprintln!("FAIL growing the vertex buffer to {needed} bytes");
                    return;
                }
            }
        }
        let bytes: Vec<u8> = data
            .batches
            .iter()
            .flat_map(|b| {
                b.vertices
                    .iter()
                    .flat_map(|v| bytemuck::bytes_of(v).to_vec())
                    .collect::<Vec<u8>>()
            })
            .collect();
        if !bytes.is_empty()
            && let Err(e) = unsafe { gpu.vertex_buffer.write_mapped(0, &bytes) }
        {
            eprintln!("FAIL vertex upload: {e}");
            return;
        }

        let Ok((slot, value)) = gpu.resources.next_slot() else {
            // The GPU is still frames_in_flight ahead; skipping is correct, and
            // a redraw is already requested every turn.
            return;
        };
        let Some(acquire_sem) = gpu.resources.acquire_semaphore(slot) else {
            eprintln!("FAIL no acquire semaphore for slot {slot}");
            return;
        };
        let Ok(vibe_vk::AcquireResult::Ready { index }) = (unsafe {
            gpu.swapchain.acquire(
                &gpu.swapchain_loader,
                TIMEOUT,
                acquire_sem,
                vk::Fence::null(),
            )
        }) else {
            return;
        };
        // Chosen after the acquire, from the image: a present semaphore is
        // per swapchain image, because the presentation engine holds the one it
        // was given until that image comes back around.
        let Some(present_sem) = gpu.resources.present_semaphore(index as usize) else {
            eprintln!("FAIL no present semaphore for image {index}");
            return;
        };
        let target_image = gpu.images[index as usize];
        let target_view = gpu.views[index as usize];

        // Both pools are reused every frame, and a command buffer still in the
        // executable state cannot be recorded into again.
        if gpu.pool.reset(device).is_err() || gpu.present_pool.reset(device).is_err() {
            eprintln!("FAIL resetting a command pool");
            return;
        }
        let Ok(cmd) = (unsafe { gpu.pool.begin(device, slot) }) else {
            return;
        };
        let mut enc = vibe_vk::BarrierEncoder::for_tier(gpu.logical.sync_tier());
        enc.push(
            vibe_vk::Barrier::image(target_image, vk::ImageAspectFlags::COLOR)
                .layouts(
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                )
                .src(vibe_vk::Stage::ColorAttachmentOutput, vibe_vk::Access::None)
                .dst(
                    vibe_vk::Stage::ColorAttachmentOutput,
                    vibe_vk::Access::ColorAttachmentWrite,
                ),
        );
        unsafe { enc.record(device, &gpu.sync2, cmd) };

        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: w,
                height: h,
            },
        };
        let attachment = [vk::RenderingAttachmentInfo {
            image_view: target_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            resolve_image_view: vk::ImageView::null(),
            resolve_mode: vk::ResolveModeFlags::empty(),
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue { float32: BACKDROP },
            },
            ..Default::default()
        }];
        let rendering_info = vk::RenderingInfo {
            render_area: area,
            layer_count: 1,
            view_mask: 0,
            color_attachment_count: 1,
            p_color_attachments: attachment.as_ptr(),
            ..Default::default()
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: w as f32,
            height: h as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        // The whole recording is one unsafe block rather than a dozen: the
        // commands are individually unsafe but jointly depend on the same
        // invariants, which the function's own contract already states.
        unsafe {
            gpu.rendering.cmd_begin_rendering(cmd, &rendering_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, gpu.pipeline.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                gpu.pipeline.layout,
                0,
                &[gpu.bind_group.set],
                &[],
            );
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[area]);
            device.cmd_set_cull_mode(cmd, vk::CullModeFlags::NONE);
            device.cmd_set_primitive_topology(cmd, vk::PrimitiveTopology::TRIANGLE_LIST);
            device.cmd_set_depth_test_enable(cmd, false);
            device.cmd_bind_vertex_buffers(cmd, 0, &[gpu.vertex_buffer.buffer()], &[0]);

            // One draw per batch, each with its own scissor: that is what keeps a
            // panel's contents inside the panel.
            let mut first = 0u32;
            for batch in &data.batches {
                if batch.vertices.is_empty() {
                    continue;
                }
                let (min_x, min_y, max_x, max_y) = batch.clip;
                let clip = scissor_from_bounds(min_x, min_y, max_x, max_y, w, h);
                device.cmd_set_scissor(cmd, 0, &[clip]);
                device.cmd_draw(cmd, batch.vertices.len() as u32, 1, first, 0);
                first += batch.vertices.len() as u32;
            }
            gpu.rendering.cmd_end_rendering(cmd);
        }

        // The present transition is a separate command buffer: it is only valid
        // once the render pass has ended, and beginning a second buffer in a pool
        // that still has one open is what the driver rejects.
        let Ok(present_cmd) = (unsafe { gpu.present_pool.begin(device, 0) }) else {
            return;
        };
        let mut present_enc = vibe_vk::BarrierEncoder::for_tier(gpu.logical.sync_tier());
        present_enc.push(
            vibe_vk::Barrier::image(target_image, vk::ImageAspectFlags::COLOR)
                .layouts(
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                )
                .src(
                    vibe_vk::Stage::ColorAttachmentOutput,
                    vibe_vk::Access::ColorAttachmentWrite,
                )
                .dst(vibe_vk::Stage::BottomOfPipe, vibe_vk::Access::None),
        );
        unsafe { present_enc.record(device, &gpu.sync2, present_cmd) };
        let _ = unsafe { gpu.present_pool.end(device, 0) };
        let _ = unsafe { gpu.pool.end(device, slot) };

        let Some(timeline) = gpu.resources.timeline_semaphore() else {
            eprintln!("FAIL the device is not on the timeline tier");
            return;
        };
        // The pacing wait is the previous frame's signalled value, so this
        // frame's wait consumes the last frame's signal. Without it the
        // present semaphore is signalled twice with nothing waiting in between,
        // which Vulkan rejects.
        let sync = vibe_frame::SubmitSync::Paced {
            timeline,
            wait_value: value.saturating_sub(1),
            acquire: acquire_sem,
            signal: present_sem,
            signal_value: value,
        };
        let mut batch = vibe_frame::SubmitBatch::single(cmd, sync);
        batch.push(present_cmd);
        let submission = vibe_frame::build_submit_info(&batch);
        let Some(fence) = gpu.resources.fence(slot) else {
            return;
        };
        // The fence is reused across frames, and a signalled one would make
        // this frame's wait return immediately while the GPU is still working.
        if let Err(e) = device.reset_fences(&[fence]) {
            eprintln!("FAIL fence reset: {e:?}");
            return;
        }
        if let Err(e) = gpu.sync2.queue_submit2(queue, &[submission.info], fence) {
            eprintln!("FAIL submit: {e:?}");
            return;
        }
        if let Err(e) = device.wait_for_fences(&[fence], true, TIMEOUT) {
            eprintln!("FAIL wait: {e:?}");
            return;
        }
        // The frame is finished, which frees the slot for the next one. Without
        // this the loop stalls after frames_in_flight submissions.
        gpu.resources.on_complete();

        if readback {
            // After the present transition, and before the submit: the image is
            // in PRESENT_SRC_KHR there, and restoring it to the same layout
            // leaves the present's own expectations intact. Once only, because
            // a swapchain image belongs to the presentation engine between the
            // present and the next acquire.
            let lit = read_back(
                device,
                &gpu.sync2,
                &gpu.logical,
                &gpu.present_pool,
                queue,
                target_image,
                w,
                h,
            );
            println!(
                "[readback] {lit} of {} pixel(s) differ from the clear",
                w as u64 * h as u64
            );
        }

        let _ = gpu
            .swapchain
            .present(&gpu.swapchain_loader, queue, index, present_sem);
    }

    /// Rebuild the swapchain after a resize.
    ///
    /// The old one is handed to the new so the driver can reuse its resources,
    /// and the old views are destroyed only once the new views exist —
    /// destroying first leaves a window in which nothing can be presented.
    fn recreate_swapchain(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let (w, h) = window.size();
        if w == 0 || h == 0 {
            return;
        }
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let device = &gpu.logical.device;
        let old = gpu.swapchain.handle();
        // SAFETY: the loaders and device are live, the surface is this
        // window's, and the old handle is not destroyed before the new one
        // exists.
        let rebuilt = unsafe {
            vibe_vk::Swapchain::new(
                &gpu.swapchain_loader,
                &gpu.surface_loader,
                gpu.physical_device,
                self.surface,
                gpu.logical.graphics_family().unwrap_or(0),
                gpu.logical.present_family().unwrap_or(0),
                (w, h),
                old,
                vibe_vk::FrameConfig::for_tier(gpu.logical.sync_tier()),
            )
        };
        let new = match rebuilt {
            Ok(s) => s,
            Err(e) => {
                eprintln!("swapchain rebuild failed: {e}");
                return;
            }
        };
        let images: Vec<vk::Image> = new.images().to_vec();
        let mut views = Vec::with_capacity(images.len());
        for image in &images {
            let info = vk::ImageViewCreateInfo {
                image: *image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: new.format().format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            // SAFETY: the device is live and the view outlives the swapchain.
            match unsafe { device.create_image_view(&info, None) } {
                Ok(v) => views.push(v),
                Err(e) => {
                    eprintln!("FAIL image view on resize: {e:?}");
                    return;
                }
            }
        }
        // The GPU must be finished with the old images before their views go.
        // Destroying a view on an image a frame is still presenting into is a
        // use-after-free, and the driver reports that as a lost device rather
        // than an error at the destroy, so the wait has to come first.
        gpu.logical.wait_idle();
        // The old views belong to the old images, which the new swapchain has
        // replaced.
        for view in gpu.views.drain(..) {
            unsafe { device.destroy_image_view(view, None) };
        }
        // A swapchain rebuilt at a new size can have a different number of
        // images, and the present semaphores are sized to the old count.
        unsafe {
            gpu.resources
                .resize_present_semaphores(device, images.len())
        };
        // The old swapchain has to be destroyed now the new one exists. It is
        // not dropped automatically, so without this a project that resizes a
        // hundred times leaks a hundred swapchains, each holding its images.
        let mut old_swapchain = std::mem::replace(&mut gpu.swapchain, new);
        unsafe { old_swapchain.destroy(&gpu.swapchain_loader) };
        gpu.images = images;
        gpu.views = views;
        println!("[resize] swapchain rebuilt at {w}x{h}");
    }
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = winit::window::WindowAttributes::default()
            .with_title("vibeEngine editor")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0))
            .with_visible(true);
        let window = match event_loop.create_window(attrs) {
            Ok(w) => vibe_window::Window::from_inner(w),
            Err(e) => {
                eprintln!("FAIL window: {e}");
                return;
            }
        };
        // The GPU is set up on the first about_to_wait rather than here.
        // Creating a Vulkan surface reads the window's Wayland handle, and
        // doing that from inside `resumed` -- while winit is mid-callback with
        // its own state held -- crashes the loop before it reaches a frame.
        self.platform = Some(egui_winit::State::new(
            self.editor.context.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        ));
        self.window = Some(window);
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        println!("[setup] window ready, entering poll mode");
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let (Some(platform), Some(window)) = (self.platform.as_mut(), self.window.as_ref()) {
            let _ = platform.on_window_event(&window.inner, &event);
        }
        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            winit::event::WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.recreate_swapchain();
                }
            }
            winit::event::WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.gpu.is_none() {
            // setup is an associated function, so the window borrow and the
            // self borrow do not overlap.
            let built = self.window.as_ref().map(|w| unsafe { Self::setup(w) });
            match built {
                Some(Ok(setup)) => {
                    println!(
                        "[setup] ui pipeline built, {} image view(s)",
                        setup.gpu.views.len()
                    );
                    self._messenger = setup.messenger;
                    self.surface = setup.surface;
                    self.gpu = Some(setup.gpu);
                }
                Some(Err(e)) => eprintln!("FAIL setup: {e}"),
                None => {}
            }
        }
        if let Some(window) = self.window.as_ref() {
            window.inner.request_redraw();
        }
    }
}

impl App {
    /// Build everything a redraw needs.
    ///
    /// # Safety
    ///
    /// Called once, with a live window whose handles are valid.
    unsafe fn setup(window: &vibe_window::Window) -> Result<Setup, String> {
        // Held for the run: a messenger whose destructor runs at the end of
        // setup leaves the layer routing into freed memory.
        let mut messenger = None;
        let entry = vibe_vk::load_entry().map_err(|e| e.to_string())?;
        let (w, h) = window.size();
        println!("[1] window {w}x{h}");

        let extensions =
            surface::required_instance_extensions(&window.inner).map_err(|e| e.to_string())?;
        let instance = vibe_vk::create_instance(&entry.ash, c"vibeEngine-editor", &extensions)
            .map_err(|e| e.to_string())?;
        if std::env::var("VIBE_VALIDATION").is_ok() {
            // Held for the run rather than dropped here: a messenger whose
            // destructor runs while the layer may still route to it is a
            // use-after-free on the way out of setup.
            messenger = vibe_vk::validation::ValidationMessenger::new(&entry.ash, &instance).ok();
        }
        let (vk_surface, _) = surface::create_surface(&entry.ash, &instance, &window.inner)
            .map_err(|e| e.to_string())?;
        let surface_loader = ash::khr::surface::Instance::new(&entry.ash, &instance);

        let devices = vibe_vk::enumerate_devices(&instance, entry.loader_version)
            .map_err(|e| e.to_string())?;
        let Some(pd) = vibe_vk::select_device(&devices, vibe_vk::QueueRequirements::all()) else {
            return Err("no suitable device".to_string());
        };
        let logical = vibe_vk::LogicalDevice::create(
            &instance,
            Some(&surface_loader),
            pd.handle,
            Some(vk_surface),
            vibe_vk::QueueRequirements::all(),
        )
        .map_err(|e| e.to_string())?;
        let device = &logical.device;
        let rendering = ash::khr::dynamic_rendering::Device::new(&instance, device);
        let sync2 = ash::khr::synchronization2::Device::new(&instance, device);
        println!("[2] device {} tier {}", pd.name, logical.sync_tier());

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, device);
        let swapchain = vibe_vk::Swapchain::new(
            &swapchain_loader,
            &surface_loader,
            pd.handle,
            vk_surface,
            logical.graphics_family().unwrap_or(0),
            logical.present_family().unwrap_or(0),
            (w, h),
            vk::SwapchainKHR::null(),
            vibe_vk::FrameConfig::for_tier(logical.sync_tier()),
        )
        .map_err(|e| e.to_string())?;
        let images: Vec<vk::Image> = swapchain.images().to_vec();
        let mut views = Vec::with_capacity(images.len());
        for image in &images {
            let info = vk::ImageViewCreateInfo {
                image: *image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: swapchain.format().format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            views.push(
                device
                    .create_image_view(&info, None)
                    .map_err(|e| format!("image view: {e:?}"))?,
            );
        }
        println!("[3] swapchain with {} image view(s)", views.len());

        let compiler = vibe_shader::WgslCompiler::new();
        let pipeline = build_ui_pipeline(device, &compiler, &[swapchain.format().format])
            .map_err(|e| e.to_string())?;
        println!("[4] ui pipeline built (key {:#x})", pipeline.cache_key);

        let props = instance.get_physical_device_memory_properties(pd.handle);
        let heap = HeapLayout {
            property_flags: props.memory_types[..props.memory_type_count as usize]
                .iter()
                .map(|m| m.property_flags)
                .collect(),
            heap_index: props.memory_types[..props.memory_type_count as usize]
                .iter()
                .map(|m| m.heap_index)
                .collect(),
            heap_sizes: props.memory_heaps[..props.memory_heap_count as usize]
                .iter()
                .map(|m| m.size)
                .collect(),
        };

        // The screen transform, as a uniform at binding 0. Vulkan requires the
        // set to be bound for every draw, so it is bound once and stays bound.
        let mut transform_buffer = GpuBuffer::create(
            device,
            &heap,
            std::mem::size_of::<ScreenTransform>() as u64,
            BufferUsage {
                uniform: true,
                ..Default::default()
            },
            MemoryNeed::upload(),
        )
        .map_err(|e| e.to_string())?;
        transform_buffer.map(device).map_err(|e| e.to_string())?;
        let transform = ScreenTransform::for_size(w, h);
        transform_buffer
            .write_mapped(0, bytemuck::bytes_of(&transform))
            .map_err(|e| e.to_string())?;
        // A 1x1 atlas to bind before egui has rasterised a real one, because
        // the descriptor set has to name a view and the shader reads it on
        // every draw whether or not there is text. The first frame replaces it.
        //
        // It has to be transitioned like any other texture the shader reads: a
        // descriptor naming a view is a promise about the image's layout, and
        // sampling an image left in UNDEFINED is what the validation layer
        // reports here.
        let atlas = GpuImage::create_2d(
            device,
            &heap,
            1,
            1,
            vk::Format::R8G8B8A8_UNORM,
            ImageUsage::texture(),
            vk::ImageTiling::OPTIMAL,
            vk::SampleCountFlags::TYPE_1,
        )
        .map_err(|e| e.to_string())?;
        // Linear and clamped: a glyph is magnified out of a small atlas cell, so
        // nearest filtering makes it a staircase, and a UV outside the atlas
        // should show the edge rather than wrap to another letter.
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            max_lod: 0.0,
            ..Default::default()
        };
        let atlas_sampler =
            unsafe { device.create_sampler(&sampler_info, None) }.map_err(|e| e.to_string())?;

        // Put the placeholder in the layout the descriptor names, so the frames
        // before the real atlas arrives sample a defined image. A descriptor
        // naming a view is a promise about that image's layout, and sampling one
        // left in UNDEFINED is what the validation layer reports.
        let Some(queue) = logical.graphics_queue() else {
            return Err("no graphics queue for the placeholder atlas".to_string());
        };
        let atlas_pool =
            vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 1)
                .map_err(|e| e.to_string())?;
        let atlas_cmd = atlas_pool.begin(device, 0).map_err(|e| e.to_string())?;
        let mut atlas_enc = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
        atlas_enc.push(
            vibe_vk::Barrier::image(atlas.image(), vk::ImageAspectFlags::COLOR)
                .layouts(
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
                .src(vibe_vk::Stage::None, vibe_vk::Access::None)
                .dst(vibe_vk::Stage::FragmentShader, vibe_vk::Access::ShaderRead),
        );
        atlas_enc.record(device, &sync2, atlas_cmd);
        atlas_pool.end(device, 0).map_err(|e| e.to_string())?;
        let atlas_batch = vibe_frame::SubmitBatch::single(atlas_cmd, vibe_frame::SubmitSync::None);
        let atlas_sub = vibe_frame::build_submit_info(&atlas_batch);
        let atlas_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| e.to_string())?;
        unsafe {
            sync2
                .queue_submit2(queue, &[atlas_sub.info], atlas_fence)
                .map_err(|e| e.to_string())?;
            device
                .wait_for_fences(&[atlas_fence], true, TIMEOUT)
                .map_err(|e| e.to_string())?;
            device.destroy_fence(atlas_fence, None);
        }

        let bind_group = UiBindGroup::create(
            device,
            pipeline.descriptor_layout,
            transform_buffer.buffer(),
            std::mem::size_of::<ScreenTransform>() as u64,
            atlas.view(),
            atlas_sampler,
        )
        .map_err(|e| e.to_string())?;

        // The vertex buffer starts small and grows: a first frame of a few
        // thousand bytes does not justify a megabyte, and a busy editor's later
        // frames do.
        let mut vertex_buffer = GpuBuffer::create(
            device,
            &heap,
            64 * 1024,
            BufferUsage {
                vertex: true,
                ..Default::default()
            },
            MemoryNeed::upload(),
        )
        .map_err(|e| e.to_string())?;
        vertex_buffer.map(device).map_err(|e| e.to_string())?;

        let pool = vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 2)
            .map_err(|e| e.to_string())?;
        let present_pool =
            vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 1)
                .map_err(|e| e.to_string())?;
        let mut resources = vibe_frame::FrameResources::new(
            device,
            vibe_frame::FrameConfig::for_tier(logical.sync_tier()),
        )
        .map_err(|e| e.to_string())?;
        // The present semaphores are one per swapchain image, and the swapchain
        // is already built, so they are sized to its image count here rather
        // than waiting for a resize that may never come.
        resources.resize_present_semaphores(device, images.len());

        // Built before the struct literal, because the heap is moved into it
        // and the staging needs a borrow of the heap to be allocated.
        // Built before the struct literal, because the heap and the device are
        // moved into it and the staging needs to borrow both.
        let atlas_pool =
            vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 1)
                .map_err(|e| e.to_string())?;
        let mut atlas_staging = GpuBuffer::create(
            device,
            &heap,
            1024 * 1024,
            BufferUsage {
                transfer_src: true,
                ..Default::default()
            },
            MemoryNeed::upload(),
        )
        .map_err(|e| e.to_string())?;
        atlas_staging.map(device).map_err(|e| e.to_string())?;

        Ok(Setup {
            gpu: Gpu {
                logical,
                physical_device: pd.handle,
                surface_loader,
                swapchain_loader,
                swapchain,
                images,
                views,
                pipeline,
                bind_group,
                transform_buffer,
                vertex_buffer,
                pool,
                present_pool,
                atlas_pool,
                resources,
                heap,
                rendering,
                sync2,
                atlas: Some(atlas),
                atlas_sampler,
                // Sized for a first atlas and grown if a later one is larger;
                // egui's is fixed once the fonts are loaded.
                atlas_staging,
                uploaded_atlas: FontAtlas::default(),
            },
            messenger,
            surface: vk_surface,
        })
    }
}

/// Copy the rendered image out and count the pixels that are not the clear.
/// Upload the font atlas when egui has rasterised a new one.
///
/// egui rasterises its atlas once and reuses it, so this runs on the first
/// frame and after a font change, not every frame. The image is recreated
/// because egui can grow it, and a bind group naming a view of the old image
/// would sample freed memory.
///
/// # Safety
///
/// The device must be live and the bind group is rebuilt before any draw.
/// Upload the font atlas when egui has rasterised a new one.
///
/// egui rasterises its atlas once and reuses it, so this runs on the first
/// frame that has a GPU and after a font change. The image is recreated
/// because egui can grow it, and a bind group naming a view of the old image
/// would sample freed memory.
///
/// # Safety
///
/// The device must be live and the bind group is rebuilt before any draw.
unsafe fn upload_atlas(gpu: &mut Gpu, atlas: &FontAtlas, queue: vk::Queue) {
    let device = &gpu.logical.device;
    if atlas.is_empty() || atlas == &gpu.uploaded_atlas {
        return;
    }

    let need = atlas.byte_len() as u64;
    if gpu.atlas_staging.size() < need {
        let bigger = unsafe {
            GpuBuffer::create(
                device,
                &gpu.heap,
                need.next_power_of_two(),
                BufferUsage {
                    transfer_src: true,
                    ..Default::default()
                },
                MemoryNeed::upload(),
            )
        };
        let Ok(mut bigger) = bigger else {
            eprintln!("FAIL growing the atlas staging buffer to {need} bytes");
            return;
        };
        if unsafe { bigger.map(device) }.is_err() {
            eprintln!("FAIL mapping the grown atlas staging buffer");
            return;
        }
        gpu.atlas_staging = bigger;
    }
    if gpu.atlas_staging.write_mapped(0, &atlas.pixels).is_err() {
        eprintln!("FAIL staging the font atlas");
        return;
    }

    // A new image each time, because egui can grow the atlas and a bind group
    // naming a view of the old one would sample freed memory.
    let image = match unsafe {
        GpuImage::create_2d(
            device,
            &gpu.heap,
            atlas.width,
            atlas.height,
            vk::Format::R8G8B8A8_UNORM,
            ImageUsage::texture(),
            vk::ImageTiling::OPTIMAL,
            vk::SampleCountFlags::TYPE_1,
        )
    } {
        Ok(i) => i,
        Err(e) => {
            eprintln!("FAIL creating the atlas image: {e}");
            return;
        }
    };

    // The atlas pool is its own: the frame's pools have buffers in the
    // recording state, and beginning a buffer in a pool that still has one open
    // is what this driver crashes on.
    if gpu.atlas_pool.reset(device).is_err() {
        eprintln!("FAIL resetting the atlas pool");
        return;
    }
    let Ok(cmd) = (unsafe { gpu.atlas_pool.begin(device, 0) }) else {
        eprintln!("FAIL beginning the atlas upload");
        return;
    };
    unsafe {
        let mut enc = vibe_vk::BarrierEncoder::for_tier(gpu.logical.sync_tier());
        enc.push(
            vibe_vk::Barrier::image(image.image(), vk::ImageAspectFlags::COLOR)
                .layouts(
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                )
                .src(vibe_vk::Stage::None, vibe_vk::Access::None)
                .dst(vibe_vk::Stage::Transfer, vibe_vk::Access::TransferWrite),
        );
        enc.record(device, &gpu.sync2, cmd);
        if image
            .record_upload(device, cmd, gpu.atlas_staging.buffer())
            .is_err()
        {
            eprintln!("FAIL recording the atlas upload");
            return;
        }
        let mut enc2 = vibe_vk::BarrierEncoder::for_tier(gpu.logical.sync_tier());
        enc2.push(
            vibe_vk::Barrier::image(image.image(), vk::ImageAspectFlags::COLOR)
                .layouts(
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
                .src(vibe_vk::Stage::Transfer, vibe_vk::Access::TransferWrite)
                .dst(vibe_vk::Stage::FragmentShader, vibe_vk::Access::ShaderRead),
        );
        enc2.record(device, &gpu.sync2, cmd);
    }
    if unsafe { gpu.atlas_pool.end(device, 0) }.is_err() {
        eprintln!("FAIL ending the atlas upload");
        return;
    }
    let batch = vibe_frame::SubmitBatch::single(cmd, vibe_frame::SubmitSync::None);
    let sub = vibe_frame::build_submit_info(&batch);
    let Ok(fence) = (unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }) else {
        return;
    };
    if unsafe { gpu.sync2.queue_submit2(queue, &[sub.info], fence) }.is_err() {
        unsafe { device.destroy_fence(fence, None) };
        return;
    }
    let _ = unsafe { device.wait_for_fences(&[fence], true, TIMEOUT) };
    unsafe { device.destroy_fence(fence, None) };

    // Swap in the new atlas and rebind, so the draws that follow sample it.
    if let Some(mut old) = gpu.atlas.replace(image) {
        unsafe { old.destroy(device) };
    }
    gpu.uploaded_atlas = atlas.clone();
    let view = gpu
        .atlas
        .as_ref()
        .map(|a| a.view())
        .unwrap_or(vk::ImageView::null());
    let bound = unsafe {
        UiBindGroup::create(
            device,
            gpu.pipeline.descriptor_layout,
            gpu.transform_buffer.buffer(),
            std::mem::size_of::<ScreenTransform>() as u64,
            view,
            gpu.atlas_sampler,
        )
    };
    let Ok(bound) = bound else {
        eprintln!("FAIL rebuilding the bind group for the new atlas");
        return;
    };
    unsafe { gpu.bind_group.destroy(device) };
    gpu.bind_group = bound;
    println!(
        "[atlas] uploaded {}x{}, {} bytes",
        atlas.width,
        atlas.height,
        atlas.pixels.len()
    );
}

/// Copy the rendered image out and count the pixels that are not the clear.
///
/// Takes a command pool from the caller rather than creating one. Creating a
/// second pool while the frame's own pool is still recording is the one Vulkan
/// call here the 2D example does not make at the same point, and on this driver
/// it takes the process down rather than returning an error.
///
/// # Safety
///
/// `pool` must have a free command buffer and the image must be in
/// `PRESENT_SRC_KHR`, which is where the frame's present transition leaves it.
unsafe fn read_back(
    device: &ash::Device,
    sync2: &ash::khr::synchronization2::Device,
    logical: &vibe_vk::LogicalDevice,
    pool: &vibe_frame::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    width: u32,
    height: u32,
) -> usize {
    let size = width as u64 * 4 * height as u64;
    let Ok(mut buffer) = GpuBuffer::create(
        device,
        logical.memory_layout(),
        size,
        BufferUsage {
            transfer_dst: true,
            ..Default::default()
        },
        MemoryNeed::upload(),
    ) else {
        return 0;
    };
    if buffer.map(device).is_err() {
        return 0;
    }
    let Ok(cmd) = pool.begin(device, 0) else {
        return 0;
    };
    {
        let mut enc = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
        // The frame's present transition has already been recorded, so the
        // image is in PRESENT_SRC_KHR. Reading from any other layout is a
        // mismatch the driver reports, and restoring to anything but
        // PRESENT_SRC_KHR breaks the present that follows.
        enc.push(
            vibe_vk::Barrier::image(image, vk::ImageAspectFlags::COLOR)
                .layouts(
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                )
                .src(
                    vibe_vk::Stage::ColorAttachmentOutput,
                    vibe_vk::Access::ColorAttachmentWrite,
                )
                .dst(vibe_vk::Stage::Transfer, vibe_vk::Access::TransferRead),
        );
        enc.record(device, sync2, cmd);
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer.buffer(),
            &[region],
        );
        let mut enc2 = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
        enc2.push(
            vibe_vk::Barrier::memory_barrier()
                .src(vibe_vk::Stage::Transfer, vibe_vk::Access::TransferWrite)
                .dst(vibe_vk::Stage::Host, vibe_vk::Access::HostRead),
        );
        // Back to PRESENT_SRC_KHR, which is the layout the present needs and
        // the one the image was in before the copy.
        enc2.push(
            vibe_vk::Barrier::image(image, vk::ImageAspectFlags::COLOR)
                .layouts(
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                )
                .src(vibe_vk::Stage::Transfer, vibe_vk::Access::TransferRead)
                .dst(vibe_vk::Stage::BottomOfPipe, vibe_vk::Access::None),
        );
        enc2.record(device, sync2, cmd);
    }
    if pool.end(device, 0).is_err() {
        return 0;
    }
    let batch = vibe_frame::SubmitBatch::single(cmd, vibe_frame::SubmitSync::None);
    let sub = vibe_frame::build_submit_info(&batch);
    let Ok(fence) = device.create_fence(&vk::FenceCreateInfo::default(), None) else {
        return 0;
    };
    if sync2.queue_submit2(queue, &[sub.info], fence).is_err() {
        return 0;
    }
    if device.wait_for_fences(&[fence], true, TIMEOUT).is_err() {
        return 0;
    }
    device.destroy_fence(fence, None);
    let Ok(data) = buffer.read_mapped(0, size as usize) else {
        return 0;
    };

    {
        let path = std::path::Path::new("/tmp/vibe-editor-frame.ppm");
        let mut ppm = format!("P6\n{} {}\n255\n", width, height).into_bytes();
        for p in data.chunks_exact(4) {
            ppm.extend_from_slice(&[p[2], p[1], p[0]]);
        }
        let _ = std::fs::write(path, ppm);
        println!("  wrote {}", path.display());
    }

    data.chunks_exact(4)
        .filter(|p| [p[2], p[1], p[0]] != BACKDROP_BGR)
        .count()
}
