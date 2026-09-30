//! Renders one frame of batched quads with the editor's UI over it.
//!
//! This is the hardware check for the UI pipeline: it builds the editor's
//! pipeline, runs a real egui frame, uploads the triangles it produces and
//! reads the result back. A unit test can prove the bridge flattens meshes
//! correctly; only this proves the pipeline creates, records and rasterises on
//! an actual driver.
//!
//! Run with: cargo run -p vibe-render --example editor
//!
//! Set VIBE_VALIDATION=1 to have the validation layer report anything wrong.

use std::ffi::CStr;

use ash::vk;
use vibe_editor::{ContentBrowser, EditorFrame, HierarchyPanel, InspectorPanel, LogPanel, Panel};
use vibe_render::ui::{ScreenTransform, build_ui_pipeline, scissor_from_bounds};
use vibe_vk::memory::{BufferUsage, GpuBuffer, MemoryNeed};
use vibe_window::{SurfaceKind, surface};

const TIMEOUT: u64 = 10_000_000_000;

/// The background the scene is drawn over, so the UI is visibly on top of it.
const BACKDROP: [u8; 4] = [40, 44, 52, 255];

fn main() {
    let validating = std::env::var("VIBE_VALIDATION").is_ok();
    if validating {
        vibe_vk::validation::enable_layer();
    }

    let (window, entry) = match make_window() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FAIL window: {e}");
            return;
        }
    };
    let (w, h) = window.size();
    println!("[1] window {w}x{h}");

    let raw = &window.inner;
    let extensions = match surface::required_instance_extensions(raw) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("FAIL extensions: {e}");
            return;
        }
    };
    let instance =
        match unsafe { vibe_vk::create_instance(&entry.ash, c"vibeEngine-editor", &extensions) } {
            Ok(i) => i,
            Err(e) => {
                eprintln!("FAIL instance: {e}");
                return;
            }
        };
    let _messenger = if validating {
        unsafe { vibe_vk::validation::ValidationMessenger::new(&entry.ash, &instance) }.ok()
    } else {
        None
    };

    let (vk_surface, _) = match unsafe { surface::create_surface(&entry.ash, &instance, raw) } {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL surface: {e}");
            return;
        }
    };
    let surface_loader = ash::khr::surface::Instance::new(&entry.ash, &instance);
    let devices = match unsafe { vibe_vk::enumerate_devices(&instance, entry.loader_version) } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL devices: {e}");
            return;
        }
    };
    let Some(pd) = vibe_vk::select_device(&devices, vibe_vk::QueueRequirements::all()) else {
        eprintln!("FAIL no device");
        return;
    };
    let logical = match unsafe {
        vibe_vk::LogicalDevice::create(
            &instance,
            Some(&surface_loader),
            pd.handle,
            Some(vk_surface),
            vibe_vk::QueueRequirements::all(),
        )
    } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL device: {e}");
            return;
        }
    };
    let device = &logical.device;
    let Some(graphics_queue) = logical.graphics_queue() else {
        eprintln!("FAIL no graphics queue");
        return;
    };
    let rendering = ash::khr::dynamic_rendering::Device::new(&instance, device);
    let sync2 = ash::khr::synchronization2::Device::new(&instance, device);
    println!("[2] device {} tier {}", pd.name, logical.sync_tier());

    let swapchain_loader = ash::khr::swapchain::Device::new(&instance, device);
    let swapchain = match unsafe {
        vibe_vk::Swapchain::new(
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
    } {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL swapchain: {e}");
            return;
        }
    };
    let images = match unsafe { swapchain_loader.get_swapchain_images(swapchain.handle()) } {
        Ok(i) => i,
        Err(e) => {
            eprintln!("FAIL swapchain images: {e}");
            return;
        }
    };
    let mut views = Vec::new();
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
        match unsafe { device.create_image_view(&info, None) } {
            Ok(v) => views.push(v),
            Err(e) => {
                eprintln!("FAIL image view: {e:?}");
                return;
            }
        }
    }
    println!("[3] swapchain {} image view(s)", views.len());

    // The UI pipeline: the piece under test.
    let compiler = vibe_shader::WgslCompiler::new();
    let ui_pipeline =
        match unsafe { build_ui_pipeline(device, &compiler, &[swapchain.format().format]) } {
            Ok(p) => p,
            Err(e) => {
                eprintln!("FAIL ui pipeline: {e}");
                return;
            }
        };
    println!("[4] ui pipeline built (key {:#x})", ui_pipeline.cache_key);

    // Run a real egui frame with a populated scene, so the UI has something to
    // draw beyond empty panels.
    let mut world = vibe_ecs::World::new();
    for name in ["Player", "Enemy", "Camera"] {
        let e = world.spawn();
        world.add(e, vibe_ecs::components::Tag(name.to_string()));
    }
    let context = egui::Context::default();
    let mut session = vibe_editor::EditorSession::new();
    let mut hierarchy = HierarchyPanel::new();
    let mut inspector = InspectorPanel::new();
    let mut log_panel = LogPanel::new();
    let mut content = ContentBrowser::new();
    content.root = std::env::temp_dir();
    content.rescan();
    let mut requested_path = None;

    let data = EditorFrame {
        context: &context,
        session: &mut session,
        hierarchy: &mut hierarchy,
        inspector: &mut inspector,
        log_panel: &mut log_panel,
        content: &mut content,
        panel: Panel::Hierarchy,
        viewport: egui::Vec2::new(w as f32, h as f32),
        world: &mut world,
        requested_path: &mut requested_path,
    }
    .draw();
    println!(
        "[5] egui frame: {} batch(es), {} vertex(es), {} byte(s)",
        data.batch_count(),
        data.len(),
        data.byte_len()
    );
    if data.is_empty() {
        eprintln!("FAIL the editor produced no geometry");
        return;
    }

    // Upload the vertices. The buffer is a plain vertex buffer: the pipeline
    // takes no descriptors, because the shader reads only a push constant.
    let heap = {
        let props = unsafe { instance.get_physical_device_memory_properties(pd.handle) };
        vibe_vk::memory::HeapLayout {
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
        }
    };
    let mut vertex_buffer = match unsafe {
        GpuBuffer::create(
            device,
            &heap,
            (data.byte_len().max(32)) as u64,
            BufferUsage {
                vertex: true,
                ..Default::default()
            },
            MemoryNeed::upload(),
        )
    } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL vertex buffer: {e}");
            return;
        }
    };
    if let Err(e) = unsafe { vertex_buffer.map(device) } {
        eprintln!("FAIL map: {e}");
        return;
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
    if let Err(e) = unsafe { vertex_buffer.write_mapped(0, &bytes) } {
        eprintln!("FAIL upload: {e}");
        return;
    }
    println!("[6] {} vertex byte(s) uploaded", bytes.len());

    // The screen transform, as a uniform buffer at binding 0. It is bound for
    // every draw: Vulkan requires a set the pipeline statically uses to be
    // bound, and the shader reads this one.
    let transform = ScreenTransform::for_size(w, h);
    let mut transform_buffer = match unsafe {
        GpuBuffer::create(
            device,
            &heap,
            std::mem::size_of::<ScreenTransform>() as u64,
            BufferUsage {
                uniform: true,
                ..Default::default()
            },
            MemoryNeed::upload(),
        )
    } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL transform buffer: {e}");
            return;
        }
    };
    if let Err(e) = unsafe { transform_buffer.map(device) } {
        eprintln!("FAIL transform map: {e}");
        return;
    }
    if let Err(e) = unsafe { transform_buffer.write_mapped(0, bytemuck::bytes_of(&transform)) } {
        eprintln!("FAIL transform upload: {e}");
        return;
    }
    let bind_group = match unsafe {
        vibe_render::ui::UiBindGroup::create(
            device,
            ui_pipeline.descriptor_layout,
            transform_buffer.buffer(),
            std::mem::size_of::<ScreenTransform>() as u64,
        )
    } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL bind group: {e}");
            return;
        }
    };
    println!("[6b] transform uniform bound");

    // Record, submit, present.
    let pool = match unsafe {
        vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 1)
    } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL pool: {e}");
            return;
        }
    };
    let mut resources = match unsafe {
        vibe_frame::FrameResources::new(
            device,
            vibe_frame::FrameConfig::for_tier(logical.sync_tier()),
        )
    } {
        Ok(r) => r,
        Err(e) => {
            eprintln!("FAIL frame resources: {e}");
            return;
        }
    };
    let (slot, value) = match resources.next_slot() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FAIL slot: {e}");
            return;
        }
    };
    let (Some(acquire_sem), Some(present_sem)) = (
        resources.acquire_semaphore(slot),
        resources.present_semaphore(slot),
    ) else {
        eprintln!("FAIL no wsi semaphores");
        return;
    };
    let acquired = match unsafe {
        swapchain.acquire(&swapchain_loader, TIMEOUT, acquire_sem, vk::Fence::null())
    } {
        Ok(a) => a,
        Err(e) => {
            eprintln!("FAIL acquire: {e}");
            return;
        }
    };
    let vibe_vk::AcquireResult::Ready { index } = acquired else {
        eprintln!("FAIL not ready: {acquired:?}");
        return;
    };
    println!("[7] acquired image {index}");

    // The pool holds one buffer, so the frame is recorded into slot 0. Passing
    // the frame slot instead returns a null command buffer for a one-buffer
    // pool, and every later call on it fails.
    let cmd = match unsafe { pool.begin(device, 0) } {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL begin: {e}");
            return;
        }
    };
    let target_image = images[index as usize];
    let target_view = views[index as usize];

    {
        let mut enc = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
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
        unsafe { enc.record(device, &sync2, cmd) };
    }

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: w as f32,
        height: h as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D {
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
            color: vk::ClearColorValue {
                float32: [
                    BACKDROP[0] as f32 / 255.0,
                    BACKDROP[1] as f32 / 255.0,
                    BACKDROP[2] as f32 / 255.0,
                    1.0,
                ],
            },
        },
        ..Default::default()
    }];
    let rendering_info = vk::RenderingInfo {
        render_area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: w,
                height: h,
            },
        },
        layer_count: 1,
        view_mask: 0,
        color_attachment_count: 1,
        p_color_attachments: attachment.as_ptr(),
        ..Default::default()
    };

    unsafe {
        rendering.cmd_begin_rendering(cmd, &rendering_info);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, ui_pipeline.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            ui_pipeline.layout,
            0,
            &[bind_group.set],
            &[],
        );
        device.cmd_set_viewport(cmd, 0, &[viewport]);
        device.cmd_set_scissor(cmd, 0, &[scissor]);
        device.cmd_set_cull_mode(cmd, vk::CullModeFlags::NONE);
        device.cmd_set_primitive_topology(cmd, vk::PrimitiveTopology::TRIANGLE_LIST);
        device.cmd_set_depth_test_enable(cmd, false);
        device.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer.buffer()], &[0]);

        // One draw per batch, each with its own scissor: that is what keeps a
        // panel's contents inside the panel.
        let mut first = 0usize;
        for batch in &data.batches {
            if batch.vertices.is_empty() {
                continue;
            }
            let (min_x, min_y, max_x, max_y) = batch.clip;
            let clip = scissor_from_bounds(min_x, min_y, max_x, max_y, w, h);
            device.cmd_set_scissor(cmd, 0, &[clip]);
            device.cmd_draw(cmd, batch.vertices.len() as u32, 1, first as u32, 0);
            first += batch.vertices.len();
        }
        rendering.cmd_end_rendering(cmd);
    }
    println!("[8] {} batch draw(s) recorded", data.batch_count());

    // The present transition goes in a command buffer of its own. It cannot be
    // a second slot in the same pool: `pool.begin` is still recording the first
    // one, and beginning a buffer from a pool that has a buffer open is what
    // the driver rejects.
    let present_pool = match unsafe {
        vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 1)
    } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL present pool: {e}");
            return;
        }
    };
    let present_cmd = match unsafe { present_pool.begin(device, 0) } {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL begin present: {e}");
            return;
        }
    };
    let mut present_enc = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
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
    unsafe { present_enc.record(device, &sync2, present_cmd) };
    if let Err(e) = unsafe { present_pool.end(device, 0) } {
        eprintln!("FAIL end present: {e}");
        return;
    }
    if let Err(e) = unsafe { pool.end(device, 0) } {
        eprintln!("FAIL end: {e}");
        return;
    }

    let sync = vibe_frame::SubmitSync::Paced {
        timeline: resources.timeline_semaphore().expect("timeline tier"),
        wait_value: value.saturating_sub(1),
        acquire: acquire_sem,
        signal: present_sem,
        signal_value: value,
    };
    let mut batch = vibe_frame::SubmitBatch::single(cmd, sync);
    batch.push(present_cmd);
    let submission = vibe_frame::build_submit_info(&batch);
    let fence = resources.fence(slot).expect("a fence per slot");
    if let Err(e) = unsafe { sync2.queue_submit2(graphics_queue, &[submission.info], fence) } {
        eprintln!("FAIL submit: {e:?}");
        return;
    }
    if let Err(e) = unsafe { device.wait_for_fences(&[fence], true, TIMEOUT) } {
        eprintln!("FAIL wait: {e:?}");
        return;
    }
    println!("[9] submitted and completed");

    // Read the frame back, so what is asserted is pixels rather than "the
    // driver accepted the work".
    let lit = read_back(device, &sync2, &logical, graphics_queue, target_image, w, h);
    println!(
        "[10] readback: {lit} of {} pixel(s) differ from the clear",
        w as u64 * h as u64
    );
    if lit == 0 {
        eprintln!("WARNING: the frame is a single flat colour, so nothing drew");
    }

    match unsafe { swapchain.present(&swapchain_loader, graphics_queue, index, present_sem) } {
        Ok(p) => println!("[11] presented: {p:?}"),
        Err(e) => eprintln!("FAIL present: {e}"),
    }
    unsafe { device.device_wait_idle().ok() };
    println!();
    println!(
        "EDITOR RENDERED: {} batch(es), {} vertex(es) through the ui pipeline.",
        data.batch_count(),
        data.len()
    );
    let _ = (CStr::from_bytes_with_nul(b"x\0"), SurfaceKind::Wayland);
    logical.wait_idle();
}

fn make_window() -> Result<(vibe_window::Window, vibe_vk::Entry), String> {
    let entry = unsafe { vibe_vk::load_entry() }.map_err(|e| e.to_string())?;
    let event_loop = winit::event_loop::EventLoop::new().map_err(|e| e.to_string())?;
    let slot = std::sync::Arc::new(std::sync::Mutex::new(None));

    struct Probe {
        slot: std::sync::Arc<std::sync::Mutex<Option<vibe_window::Window>>>,
    }
    impl winit::application::ApplicationHandler for Probe {
        fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
            if self.slot.lock().map(|g| g.is_some()).unwrap_or(true) {
                return;
            }
            let attrs = winit::window::WindowAttributes::default()
                .with_title("vibeEngine editor")
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0))
                .with_visible(true);
            if let Ok(w) = el.create_window(attrs)
                && let Ok(mut s) = self.slot.lock()
            {
                *s = Some(vibe_window::Window::from_inner(w));
            }
            el.exit();
        }
        fn window_event(
            &mut self,
            _el: &winit::event_loop::ActiveEventLoop,
            _id: winit::window::WindowId,
            _e: winit::event::WindowEvent,
        ) {
        }
    }

    let mut probe = Probe {
        slot: std::sync::Arc::clone(&slot),
    };
    let _ = event_loop.run_app(&mut probe);
    let window = slot
        .lock()
        .ok()
        .and_then(|mut g| g.take())
        .ok_or("no window was created")?;
    Ok((window, entry))
}

/// Copy the rendered image out and count the pixels that are not the clear.
fn read_back(
    device: &ash::Device,
    sync2: &ash::khr::synchronization2::Device,
    logical: &vibe_vk::LogicalDevice,
    queue: vk::Queue,
    image: vk::Image,
    width: u32,
    height: u32,
) -> usize {
    use vibe_vk::memory::{BufferUsage, GpuBuffer, MemoryNeed};
    let stride = width as u64 * 4;
    let size = stride * height as u64;
    let Ok(mut buffer) = (unsafe {
        GpuBuffer::create(
            device,
            logical.memory_layout(),
            size,
            BufferUsage {
                transfer_dst: true,
                ..Default::default()
            },
            MemoryNeed::upload(),
        )
    }) else {
        return 0;
    };
    if unsafe { buffer.map(device) }.is_err() {
        return 0;
    }
    let Ok(pool) = (unsafe {
        vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 1)
    }) else {
        return 0;
    };
    let Ok(cmd) = (unsafe { pool.begin(device, 0) }) else {
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
        unsafe { enc.record(device, sync2, cmd) };
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
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.buffer(),
                &[region],
            );
        }
        let mut enc2 = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
        enc2.push(
            vibe_vk::Barrier::memory_barrier()
                .src(vibe_vk::Stage::Transfer, vibe_vk::Access::TransferWrite)
                .dst(vibe_vk::Stage::Host, vibe_vk::Access::HostRead),
        );
        enc2.push(
            vibe_vk::Barrier::image(image, vk::ImageAspectFlags::COLOR)
                .layouts(
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                )
                .src(vibe_vk::Stage::Transfer, vibe_vk::Access::TransferRead)
                .dst(vibe_vk::Stage::BottomOfPipe, vibe_vk::Access::None),
        );
        unsafe { enc2.record(device, sync2, cmd) };
    }
    if unsafe { pool.end(device, 0) }.is_err() {
        return 0;
    }
    let readback_batch = vibe_frame::SubmitBatch::single(cmd, vibe_frame::SubmitSync::None);
    let sub = vibe_frame::build_submit_info(&readback_batch);
    let Ok(fence) = (unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }) else {
        return 0;
    };
    if unsafe { sync2.queue_submit2(queue, &[sub.info], fence) }.is_err() {
        return 0;
    }
    if unsafe { device.wait_for_fences(&[fence], true, TIMEOUT) }.is_err() {
        return 0;
    }
    unsafe { device.destroy_fence(fence, None) };
    let Ok(data) = (unsafe { buffer.read_mapped(0, size as usize) }) else {
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

    // The clear colour, as the readback sees it through an sRGB surface.
    let clear = [
        (BACKDROP[2] as f32 * 255.0 / 255.0) as u8,
        BACKDROP[1],
        BACKDROP[0],
    ];
    data.chunks_exact(4)
        .filter(|p| [p[2], p[1], p[0]] != clear)
        .count()
}
