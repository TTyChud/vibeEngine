# vibeEngine roadmap

Status legend: **done** = implemented and tested, **next** = next milestone,
**later** = planned, needs its own milestone.

## Milestone 1 — Foundation (this one)

| Area | Status | Where |
|---|---|---|
| Workspace, profiles tuned for a 4-core box | done | `Cargo.toml` |
| Math types, AABB, projection helpers | done | `vibe-math` |
| Thread pool, ParallelFor, work-helping wait | done | `vibe-jobs` |
| Re-armable TaskGraph DAG with dependency counts | done | `vibe-jobs::graph` |
| Sparse-set ECS, entity handle + generation | done | `vibe-ecs` |
| 21 component types | done | `vibe-ecs::components` |
| UUIDs, stage-ordered schedule | done | `vibe-ecs::schedule` |
| RHI types: formats, buffers, bindless index, quad vertex | done | `vibe-rhi` |
| 2D/3D cameras, batch stats, sub-textures, rotated quads | done | `vibe-rhi` |
| Sync tier detection (Legacy / Sync2 / Sync2Timeline) | done | `vibe-vk::sync` |
| 65 extension declarations with promotion versions | done | `vibe-vk::sync` |
| Loader, instance creation, device enumeration, scoring | done | `vibe-vk::context` |
| Queue family selection | done | `vibe-vk::context` |
| Bindless capability classification | done | `vibe-vk::sync` |
| Device probe example, verified against real hardware | done | `vibe-vk/examples/probe.rs` |

Also done in the second pass, because they are pure logic and testable without
a device:

- Single barrier encoder emitting `vkCmdPipelineBarrier2` or
  `vkCmdPipelineBarrier` per tier, with both spellings behind one API
- Backend-neutral `Stage`/`Access` enums that map to both the 1.0 and 2.0 flag
  sets, including the coarse degradations 1.0 forces
- Frame pacing: timeline semaphores on Sync2Timeline, per-frame binary
  semaphores and fences below it, one API for all three
- Deferred destruction keyed on fence or timeline values
- Swapchain format, present-mode, image-count and resize decisions, kept pure
  so they can be tested without a driver
- The bridge from the render graph's synthesized barriers to Vulkan stages and
  accesses, keyed on each resource's role in the frame

Third pass, also device-independent:

- Logical device creation with a tier-gated extension and feature set. Which
  extensions to enable is a pure function of (tier, available), so it is
  tested without a device; only the `vkCreateDevice` call needs one.
- `ModernFeatures` for the Vulkan 1.3 features that cannot go in the 1.0
  struct. `dynamic_rendering` is a 1.3 feature, so it is enabled through the
  `Features2` pNext chain, not `pEnabledFeatures`.
- Memory-type selection as a scoring function over the heap layout, plus buffer
  creation, mapping, and bounds-checked mapped reads and writes.
- A surface probe example, verified against this machine's real driver.

Fourth pass:

- **WGSL to SPIR-V via naga**, in `vibe-shader`. Pure Rust, so the engine
  compiles its own shaders with no C++ toolchain and no external binary. This is
  what Slang would have provided, done natively.
  - Per-entry-point compilation, so a pipeline draws its vertex and fragment
    stages out of one module
  - A `ShaderLibrary` cache keyed by name, stage *and* options, so changing the
    debug flags invalidates rather than returning a stale module
  - Output externally verified: the built-in quad and cull shaders pass the
    system's own `spirv-val`, not just this crate's magic-number check

Fifth pass:

- **Graphics pipeline descriptions** in `vibe-pipeline`: blend, depth,
  primitive, raster and multisample state as plain data, so a pipeline can be
  described, compared and hashed with no device present.
- A **cache key** per description. Two descriptions with the same key produce an
  identical pipeline, so a cache can skip the driver call entirely — the same
  idea Unreal's PSO cache uses.
- The **quad vertex layout**, cross-checked against `vibe_rhi::QuadVertex` in a
  test. A drift between the two is the classic cause of "my sprite is garbage",
  and it produces no error anywhere, so it is worth a test that fails loudly.
- **Dynamic rendering** descriptions: colour and depth attachments with their
  load/store ops and clear values, plus a reverse-Z viewport.

Sixth pass, and the first one verified against a live window:

- **Command buffer pools** in `vibe-frame`: one pool per frame, one-time-submit
  buffers, and the slot bookkeeping that keeps the CPU from running more than
  `frames_in_flight` ahead of the GPU.
- **Submission building** on Vulkan 1.3's `vkQueueSubmit2`, where a binary
  semaphore wait and a timeline wait differ only in whether a value is set, so
  both sync paths share one code path.
- **Windowing** in `vibe-window` via winit, with per-platform surface creation:
  Wayland on niri, XCB or XLIB on X11. The platform is decided from the raw
  handle, so `create_instance` is asked for exactly the extension needed.

### Verified end to end on this machine

`cargo run -p vibe-window --example bringup` walks the whole stack and reports:

    [1] loader 1.4.357
    [2] window 936x1048 (wayland)
    [3] platform wayland needs VK_KHR_wayland_surface
    [4] instance created with 2 extension(s)
    [5] vulkan surface created
    [6] device Intel(R) UHD Graphics (TGL GT2) tier Sync2Timeline
    [7] swapchain: B8G8R8A8_SRGB, MAILBOX, 4 images
    [8] logical device created, 7 extension(s), graphics queue acquired
    [9] command pool with 2 buffer(s)

Not yet built: rendering an actual frame, the editor, audio, mesh loading.
Those come next.

The render graph (milestone 3) and scene serialization (milestone 5) were built
ahead of the GPU work, because both are pure logic and testable without a
device. That is why they are done and the swapchain is not: a frame graph you
cannot execute yet still forces the API to be right, and it is the piece that
decides what the swapchain has to support.

## Milestone 2 — Draw a triangle (vertical slice)

Nothing else matters until pixels appear. This milestone ends with a window
showing a textured quad.

- Logical device creation with the tier-gated feature set
- VMA memory allocator integration
- Single barrier encoder: one API emitting `vkCmdPipelineBarrier` or
  `vkCmdPipelineBarrier2` depending on tier
- Command buffer pool, per-frame recording
- Swapchain: create, resize, recreate, frame-skip signalling
- Timeline semaphores on the Sync2Timeline tier, binary semaphores below
- Deferred destruction queue keyed on fence values
- Shader compilation: **naga** (WGSL in, SPIR-V out) — Rust-native, so no
  C++ toolchain and no Slang build. The compiler sits behind a trait so
  `glslangValidator` or a precompiled-spirv path can slot in.
- Dynamic rendering, push constants, one pipeline, a bindless texture array
- 2D batched quad renderer with draw-call and quad counters
- Windowing via `winit` + `raw-window-handle`

## Milestone 3 — Render graph (DONE, `vibe-graph`)

Modelled on Godot's RenderingDevice dependency tracking and Unreal's RDG.

- Resource registry with generation counters and versioning — done
- Import vs declare, so the graph never frees a swapchain image — done
- Pass builder with read/write/discard declarations — done
- Compiler: topological sort, dead-pass culling — done
- Synthesized image and buffer barriers, with execution-dependency classification
- Resource lifetime analysis (first/last use, first write) — done
- Typed validation errors, JSON dump, topology hash for frame-change detection
- Executor mapping handles to live backend objects, validating before any GPU work
- Graph mapped onto `TaskGraph` as a DAG — done
- **Transient aliasing** — the point of the whole thing. Interval-graph
  colouring gives disjoint live ranges one shared allocation. In the test case
  two 1 KiB transients in sequential passes cost 1 KiB, not 2.

Still to do for the milestone to be usable on the GPU: the executor's
`ResourceResolver` needs a Vulkan implementation, and the barriers need
translating into the tier-gated encoder.

## Milestone 4 — Physics (DONE, `vibe-physics`)

Backed by **Rapier** (`rapier2d` 0.36, `rapier3d` 0.36), the Rust-native
equivalent of Box2D and Jolt.

- `Physics2D`: world create/step/destroy, static/kinematic/dynamic bodies,
  box colliders with density, friction and restitution, gravity scale, linear
  and angular damping, optional rotation, velocity and impulse control
- `Physics3D`: box, sphere and capsule colliders, per-axis linear and angular
  DOF locks, sensors, mass derived from volume and density
- ECS bridge: bodies are built from `Rigidbody2D`/`Rigidbody3D` plus the matching
  collider component, and their transforms are written back after each step
- A generation-tagged handle registry, so a body slot reused after a rebuild
  cannot be addressed by a stale handle
- `prune` drops bodies whose entity despawned, which is the case that breaks
  naive implementations

Rust crates, not C++:

- **2D**: `rapier2d` 0.36 (pure Rust, the Box2D-equivalent)
- **3D**: `rapier3d` 0.36 (pure Rust, the Jolt-equivalent)

Box2D and Jolt are the right *engines* but both are C++ that must be compiled
from source, which on a 4-core box with 7.5 GB is a multi-hour build and a
hard system dependency. Rapier is the same class of solver in Rust, so it links
like any other crate. Revisit if you later need Jolt's specific solver
quality — the component API in `vibe-ecs` is already shaped so either can slot
in behind it.

- World create/step/destroy
- Static/kinematic/dynamic bodies, 2D and 3D
- Box, sphere, capsule colliders
- Density, friction, restitution
- Mass, linear and angular damping
- Per-axis DOF locking, sensors
- Native script binding via `ScriptableEntity` (C++ subclassing, not Lua)

## Milestone 5 — Assets and text (partly done)

- glTF/GLB via `gltf` 1.4 (not `fastgltf`, which is not a real crate): meshes,
  materials, skins
- Skeletal animation: position, rotation, scale interpolation
- Bone matrix skinning in the shader, 4096 bones per frame
- MSDF text with runtime font atlas generation
- Kerning, line spacing, glyph cache
- 2D sprite sheet animation with frame timing and looping
- Virtual filesystem rooted at the executable — done, `vibe-scene::vfs`
- YAML scene serialization, both directions — done, `vibe-scene::scene`.
  Uses `serde_norway`, because `serde_yaml` and `serde_yml` are both deprecated.
  Component values are stored untagged under their component name, so a scene
  written by a newer build still loads in an older one minus the parts it does
  not know.

## Milestone 6 — Audio

- `miniaudio` backend
- 2D and 3D positional playback
- Listener position and orientation
- Volume, looping

## Milestone 7 — Editor

- ImGui with a custom Vulkan backend
- ImGuizmo transform gizmos
- Scene hierarchy panel with per-component inspectors
- Content browser: directory tree, image preview cache
- New/open/save-scene-as
- Play/Stop mode switching
- FPS editor camera and orthographic camera controller
- In-editor log sink

## Verified hardware facts (measured, not assumed)

From `cargo run -p vibe-vk --example probe` on this machine:

| | |
|---|---|
| GPU | Intel UHD Graphics (TGL GT2), integrated |
| API version | 1.4.354, loader 1.4.357 |
| Device extensions | 196 |
| **Sync tier** | **Sync2Timeline** — `vkCmdPipelineBarrier2` and timeline semaphores both available |
| **Bindless** | **Partial { max_indexed_descriptors: 4096 }** |

The bindless result matters. `VK_EXT_descriptor_indexing` is present but
`VK_KHR_bindless_texture` is not, so the fully bindless path from the design
cannot run here. The batch renderer must therefore support two modes:

1. **Full bindless** on devices with both extensions — one large
   shader-visible array, integer index in the vertex data.
2. **Partial** on this iGPU — still integer-indexed, but through a
   `VK_EXT_descriptor_indexing` described array, capped at 4096 entries.

If that cap is too low for a project's texture count, a third mode binds
per-batch descriptor sets. `BindlessSupport` is the enum that selects between
them, and it is already wired into `PhysicalDeviceInfo`.

## Known constraints on this machine

- 4 cores, 7.5 GB RAM. Builds run at `-j2`; the debug profile is tuned for it.
- Intel Tiger Lake iGPU, Mesa 26.2. Reports Vulkan 1.3, so the
  Sync2Timeline tier should be what gets detected. Verify with
  `cargo test -p vibe-vk` once device enumeration is live.
- No `vulkaninfo` installed; the `probe` example covers the same ground in Rust
  and is the check to use.
- **`&str` is not NUL terminated, and Vulkan reads until it finds a zero byte.**
  Passing one in `ppEnabledExtensionNames` makes the driver run into adjacent
  memory; on this machine it produced `VK_KHR_dynamic_rendering1` and a
  corrupted `VK_KHR_swapchain`, and `vkCreateDevice` failed with
  `ERROR_EXTENSION_NOT_PRESENT` even though every extension was present. Own
  `CString`s instead — `EnabledFeatures::extension_cstrings` exists for this and
  has a test that checks the trailing zero byte.
- `ClearValue` is a C **union** with `color` and `depth_stencil` fields, not an
  enum of Float/Color variants, and has no `Debug`.
- ash 0.38 targets Vulkan 1.3, so several 1.0 fields are **gone** rather than
  deprecated: `PipelineRasterizationStateCreateInfo` has no `depth_clip_enable`
  or `depth_bounds_enable` (both are now always on), and
  `RenderingAttachmentInfo` has no `image_array_layer` and takes
  `ResolveModeFlags` (a bitflag, not an enum). `ClearValue` is a C **union**
  with `color` and `depth_stencil` fields, not an enum of Float/Color variants,
  and has no `Debug`. `CullModeFlags` constants are `BACK` and `FRONT`, without
  the `_BIT` suffix. `CompareOp` and `SampleCountFlags` are ash types, so they
  have no `to_le_bytes`; use `as_raw()`.
- naga 30's SPIR-V entry point is `spv::write_vec(module, &info, &options,
  pipeline_options)`, and `spv::Options` has a `Default` impl, so build from
  that and override rather than filling every field. `ModuleInfo`'s fields are
  private and are only reachable through the validator, not read back.
  `fake_missing_bindings` must be `true` unless a real binding map is supplied,
  and `true` is what the engine wants anyway: bindings carry over from the WGSL
  declarations unchanged. `DebugInfo` borrows its source, so options hold owned
  text. `SourceLanguage` comes from the `spirv` crate and the WGSL variant is
  spelled `WGSL`; that enum also has a `Slang` variant, so naga already tracks
  Slang as a source language if it is ever wanted.
- WGSL reserves `sampler`, so a texture sampler variable cannot be named that.
- Rapier 0.36 replaced the old `RigidBodySet`/`ColliderSet` pipeline with a
  unified `PhysicsWorld`. Bodies are reached through its public `bodies` /
  `colliders` fields, not accessor methods. Shape constructors are on
  `ColliderBuilder`, and its setters are fluent (`.density(x)`, not
  `.set_density(x)`). `Pose3::new` takes a translation plus an **axis-angle**,
  not a quaternion, and `Rotation` is glam's own `Quat`. The 2D `AngVector` is
  the bare scalar. The prelude re-exports glam types, so globbing it next to
  this engine's own `glam` imports makes every shared name ambiguous; import
  rapier's types by name.
- glam 0.33 moved its camera helpers into `glam::camera::*`, and its
  `vulkan::orthographic` inverts Y and yields negative depth for a 2D camera, so
  `vibe-math` builds the orthographic matrix by hand. Its `perspective_infinite_reverse`
  is used as-is and is correct.
- 109 GB of the disk is `~/Projects/Graphite/target`. Not touched.
