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

Not yet built: swapchain, command buffers, pipelines, editor, physics, audio,
mesh loading. Those are the milestones below.

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

## Milestone 4 — Physics

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
- glam 0.33 moved its camera helpers into `glam::camera::*`, and its
  `vulkan::orthographic` inverts Y and yields negative depth for a 2D camera, so
  `vibe-math` builds the orthographic matrix by hand. Its `perspective_infinite_reverse`
  is used as-is and is correct.
- 109 GB of the disk is `~/Projects/Graphite/target`. Not touched.
