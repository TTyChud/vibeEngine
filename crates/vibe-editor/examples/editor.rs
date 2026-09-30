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
use vibe_editor::{Editor, Panel, UiDrawData};
use vibe_render::ui::{ScreenTransform, UiBindGroup, build_ui_pipeline, scissor_from_bounds};
use vibe_vk::memory::{BufferUsage, GpuBuffer, HeapLayout, MemoryNeed};
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
    resources: vibe_frame::FrameResources,
    heap: HeapLayout,
    rendering: ash::khr::dynamic_rendering::Device,
    sync2: ash::khr::synchronization2::Device,
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
        // The log is the one panel with something to say before a scene is
        // opened, so the window is not a grid of empty boxes on first run.
        editor.panel = Panel::Log;
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
                "[frame {}] egui: {} batch(es), {} vertex(es) -> {} byte(s)",
                self.frames,
                data.batch_count(),
                data.len(),
                data.byte_len()
            );
        }

        let readback = !self.readback_done;
        self.readback_done = true;
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        // SAFETY: every object is live, and the swapchain matches the size
        // checked above because a resize rebuilds it before the next redraw.
        unsafe { App::present(gpu, &data, w, h, readback) };
    }

    /// Record, submit and present the UI.
    ///
    /// # Safety
    ///
    /// Every Vulkan object must be live and the swapchain must match the
    /// window's current size.
    unsafe fn present(gpu: &mut Gpu, data: &UiDrawData, w: u32, h: u32, readback: bool) {
        let device = &gpu.logical.device;
        let Some(queue) = gpu.logical.graphics_queue() else {
            return;
        };

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
            // One readback, not one per frame: it proves the first frame drew
            // and costs a full pipeline stall every time after.
            let lit = read_back(device, &gpu.sync2, &gpu.logical, queue, target_image, w, h);
            println!(
                "[readback] {lit} of {} pixel(s) differ from the clear",
                w as u64 * h as u64
            );
            if lit == 0 {
                eprintln!("WARNING: the frame is one flat colour, so nothing drew");
            }
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
        gpu.swapchain = new;
        gpu.images = images;
        gpu.views = views;
        gpu.logical.wait_idle();
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
        let bind_group = UiBindGroup::create(
            device,
            pipeline.descriptor_layout,
            transform_buffer.buffer(),
            std::mem::size_of::<ScreenTransform>() as u64,
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
                resources,
                heap,
                rendering,
                sync2,
            },
            messenger,
            surface: vk_surface,
        })
    }
}

/// Copy the rendered image out and count the pixels that are not the clear.
unsafe fn read_back(
    device: &ash::Device,
    sync2: &ash::khr::synchronization2::Device,
    logical: &vibe_vk::LogicalDevice,
    queue: vk::Queue,
    image: vk::Image,
    width: u32,
    height: u32,
) -> usize {
    let stride = width as u64 * 4;
    let size = stride * height as u64;
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
    let Ok(pool) = vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 1)
    else {
        return 0;
    };
    let Ok(cmd) = pool.begin(device, 0) else {
        return 0;
    };
    {
        let mut enc = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
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
        // The copy left the image in TRANSFER_SRC_OPTIMAL and present needs
        // PRESENT_SRC_KHR; reading a frame must not break the frame after it.
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
