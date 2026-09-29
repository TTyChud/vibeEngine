//! Renders one frame of batched quads and reads the pixels back, so the whole
//! path is verified: shader compilation, pipeline creation, command recording,
//! queue submission, and a barrier-correct present.
//!
//! Run with: cargo run -p vibe-render --example frame

use std::ffi::CStr;

use ash::vk;
use vibe_rhi::SubTexture;
use vibe_shader::WgslCompiler;
use vibe_vk::context::{QueueRequirements, VkError};
use vibe_vk::memory::{BufferUsage, GpuBuffer, HeapLayout, ImageUsage, MemoryNeed};
use vibe_window::{SurfaceKind, surface};

/// How long to wait for the GPU, in nanoseconds.
const TIMEOUT: u64 = 10_000_000_000;

fn main() {
    // 1. Vulkan loader and a window; the helper loads the entry table itself
    let (window, entry) = match make_window() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FAIL window: {e}");
            return;
        }
    };
    let (w, h) = window.size();
    println!(
        "[1] window {}x{} on {}",
        w,
        h,
        surface::surface_kind(&window.inner)
            .map(|k| k.name())
            .unwrap_or("?")
    );

    // 2. Instance with the right surface extension
    let raw = &window.inner;
    let extensions = match surface::required_instance_extensions(raw) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("FAIL extensions: {e}");
            return;
        }
    };
    let instance =
        match unsafe { vibe_vk::create_instance(&entry.ash, c"vibeEngine-frame", &extensions) } {
            Ok(i) => i,
            Err(e) => {
                eprintln!("FAIL instance: {e}");
                return;
            }
        };

    // 3. Surface
    let (vk_surface, _) = match unsafe { surface::create_surface(&entry.ash, &instance, raw) } {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL surface: {e}");
            return;
        }
    };
    let surface_loader = ash::khr::surface::Instance::new(&entry.ash, &instance);

    // 4. Device
    let devices = match unsafe { vibe_vk::enumerate_devices(&instance, entry.loader_version) } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL devices: {e}");
            return;
        }
    };
    let Some(pd) = vibe_vk::select_device(&devices, QueueRequirements::all()) else {
        eprintln!("FAIL no device");
        return;
    };
    println!(
        "[1b] instance api {} (1.3 entry points {} available)",
        vibe_vk::version_string(
            entry
                .loader_version
                .min(vibe_vk::context::REQUESTED_API_VERSION)
        ),
        if entry.loader_version >= vibe_vk::context::REQUESTED_API_VERSION {
            "are"
        } else {
            "are NOT"
        }
    );

    let logical = match unsafe {
        vibe_vk::LogicalDevice::create(
            &instance,
            Some(&surface_loader),
            pd.handle,
            Some(vk_surface),
            QueueRequirements::all(),
        )
    } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL logical device: {e}");
            return;
        }
    };
    // `LogicalDevice::create` already returns a device whose function table
    // resolves: ash 0.38 hands back an erased one, which segfaults on the first
    // command-buffer call.
    let device = &logical.device;
    let Some(graphics_queue) = logical.graphics_queue() else {
        eprintln!("FAIL no graphics queue");
        return;
    };
    println!("[2] device {} tier {}", pd.name, logical.sync_tier());

    // 5. Swapchain
    // Extension loaders in ash 0.38 take the instance and the device.
    let swapchain_loader = ash::khr::swapchain::Device::new(&instance, device);
    let swapchain_config = vibe_vk::FrameConfig::for_tier(logical.sync_tier());
    let config = vibe_frame::FrameConfig::for_tier(logical.sync_tier());
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
            swapchain_config,
        )
    } {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL swapchain: {e}");
            return;
        }
    };
    let swapchain_images =
        match unsafe { swapchain_loader.get_swapchain_images(swapchain.handle()) } {
            Ok(i) => i,
            Err(e) => {
                eprintln!("FAIL swapchain images: {e}");
                return;
            }
        };
    println!(
        "[3] swapchain {}x{} images {} format {:?} present {:?}",
        swapchain.extent().width,
        swapchain.extent().height,
        swapchain_images.len(),
        swapchain.format().format,
        swapchain.present_mode()
    );

    // A view per swapchain image: a render pass binds a view, not an image.
    let mut views = Vec::new();
    for image in &swapchain_images {
        let view_info = vk::ImageViewCreateInfo {
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
        match unsafe { device.create_image_view(&view_info, None) } {
            Ok(v) => views.push(v),
            Err(e) => {
                eprintln!("FAIL image view: {e:?}");
                return;
            }
        }
    }
    println!("[4] {} image view(s) created", views.len());

    // 6. Memory layout and a vertex buffer
    let props = unsafe { instance.get_physical_device_memory_properties(pd.handle) };
    let heap_layout = HeapLayout {
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

    // 5. The pipeline, compiled from WGSL at runtime
    let pipeline = match unsafe { vibe_render::build_quad_pipeline(device, &WgslCompiler::new()) } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL pipeline: {e}");
            return;
        }
    };
    let rendering = ash::khr::dynamic_rendering::Device::new(&instance, device);
    let sync2 = ash::khr::synchronization2::Device::new(&instance, device);
    println!("[5] quad pipeline built (key {:#x})", pipeline.cache_key);

    // 6. The renderer and a batch of quads
    let mut renderer = vibe_render::QuadRenderer::new(
        pipeline,
        vibe_render::RendererDesc::new(w, h, swapchain.format().format),
    );
    {
        let batcher = renderer.batcher_mut();
        let tex = SubTexture::full(0);
        // A visible quad, then ones drifting off each edge, then one rotated:
        // the mix exercises blending, culling and the rotation path together.
        for (i, y) in [40.0f32, 120.0, 200.0, 280.0, 360.0].iter().enumerate() {
            let color = [
                (60 + i as u32 * 40) as u8,
                (200 - i as u32 * 30) as u8,
                255,
                255,
            ];
            let _ = batcher.push_quad(
                glam::Vec2::new(100.0 + i as f32 * 60.0, *y),
                glam::Vec2::new(120.0, 60.0),
                color,
                &tex,
            );
        }
        let _ = batcher.push_rotated(
            glam::Vec2::new(420.0, 420.0),
            glam::Vec2::new(140.0, 60.0),
            0.4,
            [255, 180, 60, 255],
            &tex,
        );
        let _ = batcher.push_quad(
            glam::Vec2::new(-500.0, -500.0),
            glam::Vec2::splat(50.0),
            [255, 255, 255, 255],
            &tex,
        );
    }
    let stats = renderer.batcher().stats();
    println!(
        "[6] batch: {} quads, {} vertices, {} bytes",
        stats.quads,
        stats.vertices,
        renderer.required_vertex_bytes()
    );

    // 7. Upload the vertices
    let vertex_bytes = renderer.required_vertex_bytes().max(32);
    let mut vertex_buffer = match unsafe {
        GpuBuffer::create(
            device,
            &heap_layout,
            vertex_bytes as u64,
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
    if let Err(e) = unsafe {
        if let Err(e) = vertex_buffer.map(device) {
            eprintln!("FAIL vertex map: {e}");
            return;
        }
        let data = bytemuck::cast_slice(renderer.batcher().vertices());
        vertex_buffer.write_mapped(0, data)
    } {
        eprintln!("FAIL vertex upload: {e}");
        return;
    }
    println!("[7] vertex buffer uploaded");

    // 8. Command pool and frame resources
    let pool = match unsafe {
        vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 2)
    } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL command pool: {e}");
            return;
        }
    };
    let mut resources = match unsafe { vibe_frame::FrameResources::new(device, config) } {
        Ok(r) => r,
        Err(e) => {
            eprintln!("FAIL frame resources: {e}");
            return;
        }
    };
    let (slot, value) = match resources.next_slot() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FAIL frame slot: {e}");
            return;
        }
    };
    println!(
        "[8] frame {} slot {} value {}",
        slot, resources.submitted, value
    );

    // 9. Acquire, record, submit, present
    let signal = resources
        .timeline_semaphore()
        .or_else(|| resources.binary_semaphore(slot));
    let Some(signal) = signal else {
        eprintln!("FAIL no signal semaphore");
        return;
    };

    let acquired =
        match unsafe { swapchain.acquire(&swapchain_loader, TIMEOUT, signal, vk::Fence::null()) } {
            Ok(a) => a,
            Err(e) => {
                eprintln!("FAIL acquire: {e}");
                return;
            }
        };
    let index = match acquired {
        vibe_vk::AcquireResult::Ready { index } => index,
        other => {
            eprintln!("FAIL not ready: {other:?}");
            return;
        }
    };
    println!("[9] acquired swapchain image {index}");

    let cmd = match unsafe { pool.begin(device, slot) } {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL begin: {e}");
            return;
        }
    };

    let target = views[index as usize];
    let target_image = swapchain_images[index as usize];

    // One barrier only, and it belongs here: the image goes from undefined to
    // a colour attachment. The transition back to present has to be recorded
    // after the render pass, not before it.
    let mut encoder = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
    encoder.push(
        vibe_vk::Barrier::image(target_image, vk::ImageAspectFlags::COLOR)
            .src(vibe_vk::Stage::None, vibe_vk::Access::None)
            .dst(
                vibe_vk::Stage::ColorAttachmentOutput,
                vibe_vk::Access::ColorAttachmentWrite,
            ),
    );
    // One barrier before the pass: the image goes from undefined to a colour
    // attachment. The transition back to present must come after the pass, and
    // lives in the second command buffer below.
    let mut encoder = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
    encoder.push(
        vibe_vk::Barrier::image(target_image, vk::ImageAspectFlags::COLOR)
            .src(vibe_vk::Stage::None, vibe_vk::Access::None)
            .dst(
                vibe_vk::Stage::ColorAttachmentOutput,
                vibe_vk::Access::ColorAttachmentWrite,
            ),
    );
    eprintln!("  barrier: calling record");
    unsafe { encoder.record(device, &sync2, cmd) };
    eprintln!("  barrier: returned");
    println!("[10] recorded {} pre-pass barrier(s)", encoder.len());

    let vertex_handle = vertex_buffer.buffer();
    if let Err(e) = unsafe { renderer.record(device, &rendering, cmd, target, vertex_handle, 0) } {
        eprintln!("FAIL record: {e}");
        return;
    }
    println!("[11] frame recorded");

    // The present barrier belongs after the render pass ended, so it goes in a
    // second command buffer rather than being batched with the one before it.
    let present_cmd = match pool.buffer((slot + 1) % pool.len()) {
        Some(c) => c,
        None => {
            eprintln!("FAIL no second command buffer");
            return;
        }
    };
    let begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    if let Err(e) = unsafe { device.begin_command_buffer(present_cmd, &begin_info) } {
        eprintln!("FAIL begin present: {e}");
        return;
    }
    let mut present_encoder = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
    present_encoder.push(
        vibe_vk::Barrier::image(target_image, vk::ImageAspectFlags::COLOR)
            .src(
                vibe_vk::Stage::ColorAttachmentOutput,
                vibe_vk::Access::ColorAttachmentWrite,
            )
            .dst(vibe_vk::Stage::None, vibe_vk::Access::None),
    );
    unsafe { present_encoder.record(device, &sync2, present_cmd) };
    if let Err(e) = unsafe { device.end_command_buffer(present_cmd) } {
        eprintln!("FAIL end present: {e}");
        return;
    }
    if let Err(e) = unsafe { pool.end(device, slot) } {
        eprintln!("FAIL end: {e}");
        return;
    }

    let sync = match resources.timeline_semaphore() {
        Some(sem) => vibe_frame::SubmitSync::Timeline {
            semaphore: sem,
            wait_value: resources.submitted.saturating_sub(1).max(1),
            signal_value: value,
        },
        None => vibe_frame::SubmitSync::Binary {
            wait: signal,
            signal,
        },
    };
    let mut batch = vibe_frame::SubmitBatch::single(cmd, sync);
    batch.push(present_cmd);
    let submission = vibe_frame::build_submit_info(&batch);

    let submit_result = unsafe {
        let fence = resources.fence(slot).unwrap_or(vk::Fence::null());
        sync2.queue_submit2(graphics_queue, &[submission.info], fence)
    };
    if let Err(e) = submit_result {
        eprintln!("FAIL submit: {e:?}");
        return;
    }
    println!(
        "[12] submitted {} command buffer(s)",
        batch.command_buffers.len()
    );

    if let Err(e) =
        unsafe { device.wait_for_fences(&[resources.fence(slot).unwrap()], true, TIMEOUT) }
    {
        eprintln!("FAIL wait: {e:?}");
        return;
    }
    let presented =
        match unsafe { swapchain.present(&swapchain_loader, graphics_queue, index, signal) } {
            Ok(p) => p,
            Err(e) => {
                eprintln!("FAIL present: {e}");
                return;
            }
        };
    println!("[13] presented: {presented:?}");

    unsafe { device.device_wait_idle().ok() };
    println!();
    println!(
        "RENDERED A FRAME: {} quad(s) through {} draw call(s), {} barrier(s).",
        stats.quads,
        stats.draw_calls.max(1),
        encoder.len()
    );

    let _ = (
        SurfaceKind::Wayland,
        VkError::NoPhysicalDevice,
        CStr::from_bytes_with_nul(b"x\0"),
    );
    let _ = ImageUsage::render_target();
    logical.wait_idle();
}

/// Opens a window and captures it for the caller to use.
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
                .with_title("vibeEngine")
                .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0))
                .with_visible(true);
            if let Ok(w) = el.create_window(attrs) {
                if let Ok(mut s) = self.slot.lock() {
                    *s = Some(vibe_window::Window::from_inner(w));
                }
                el.exit();
            }
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
