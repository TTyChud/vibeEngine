//! Aliasing for transient resources.
//!
//! This is what the frame graph exists to enable. A pass that writes a colour
//! target and a later pass that writes a depth target never need both alive at
//! once, so they can share one allocation. Without this, every pass that
//! declares a transient resource costs a distinct heap block, and a typical
//! frame wastes most of them.

use vibe_rhi::ResourceHandle;

use crate::graph::{CompiledGraph, PassId};
use crate::registry::ResourceRegistry;

/// A set of resources that never need to be live simultaneously.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasGroup {
    /// The resources sharing one allocation, largest first.
    pub resources: Vec<ResourceHandle>,
    /// Bytes needed to satisfy the whole group.
    pub required_bytes: u64,
}

/// Assigns transient resources to shared allocations.
///
/// Aliasing is only sound when no two members of a group are live at the same
/// time, which the compiler's pass order decides: a resource's lifetime is the
/// span from its first write to its last use, and two resources may share memory
/// only when those spans do not overlap.
#[derive(Debug, Default)]
pub struct AliasAllocator {
    groups: Vec<AliasGroup>,
    /// Each planned resource's own size, kept so the saving can be reported.
    resource_sizes: Vec<u64>,
}

impl AliasAllocator {
    /// An allocator with no groups.
    pub fn new() -> AliasAllocator {
        AliasAllocator::default()
    }

    fn reset(&mut self) {
        self.groups.clear();
        self.resource_sizes.clear();
    }

    /// Plan allocations for every transient resource in a compiled graph.
    pub fn plan(&mut self, graph: &CompiledGraph, registry: &ResourceRegistry) {
        self.reset();

        let mut spans: Vec<(ResourceHandle, PassId, PassId, u64)> = Vec::new();
        for pass in &graph.passes {
            for life in &pass.lifetimes {
                // Only transients are candidates: an imported or long-lived
                // resource owns its memory outright.
                let Some(desc) = registry.desc(life.resource) else {
                    continue;
                };
                if !desc.is_transient() || registry.is_imported(life.resource) {
                    continue;
                }
                self.resource_sizes.push(desc.size());
                spans.push((life.resource, life.first_write, life.last_use, desc.size()));
            }
        }
        // Stable order so the plan is reproducible frame to frame.
        spans.sort_by_key(|(r, ..)| (r.index(), r.generation()));

        // Interval-graph colouring: walk resources in start order and give each
        // one to the first group whose members have all finished by now.
        // `ends[i]` is the last use of the i-th planned resource, indexed the
        // same way as `spans`.
        let mut ends: Vec<PassId> = spans.iter().map(|(_, _, e, _)| *e).collect();
        for i in 0..spans.len() {
            let (resource, start, end, bytes) = spans[i];
            let mut placed = false;
            for group in &mut self.groups {
                let overlaps = group
                    .resources
                    .iter()
                    .enumerate()
                    .any(|(j, existing)| *existing == resource || start <= ends[j]);
                if !overlaps {
                    group.resources.push(resource);
                    group.required_bytes = group.required_bytes.max(bytes);
                    placed = true;
                    break;
                }
            }
            if !placed {
                self.groups.push(AliasGroup {
                    resources: vec![resource],
                    required_bytes: bytes,
                });
            }
            ends[i] = end;
        }
    }

    /// The planned groups.
    pub fn groups(&self) -> &[AliasGroup] {
        &self.groups
    }

    /// Total bytes the plan needs, after aliasing.
    pub fn required_bytes(&self) -> u64 {
        self.groups.iter().map(|g| g.required_bytes).sum()
    }

    /// Total bytes the same resources would need with no aliasing at all.
    ///
    /// The saving is `unaliased_bytes - required_bytes`, which is the number
    /// the editor's memory panel shows next to the transient budget.
    pub fn unaliased_bytes(&self) -> u64 {
        self.resource_sizes.iter().sum()
    }

    /// How many resources share memory, i.e. allocations saved.
    pub fn saved_allocations(&self) -> usize {
        self.groups
            .iter()
            .map(|g| g.resources.len().saturating_sub(1))
            .sum()
    }

    /// The group a resource belongs to, if any.
    pub fn group_of(&self, resource: ResourceHandle) -> Option<&AliasGroup> {
        self.groups.iter().find(|g| g.resources.contains(&resource))
    }
}

/// True when two resources' live ranges overlap, and so cannot share memory.
pub fn lifetimes_overlap(
    a: &crate::graph::ResourceLifetime,
    b: &crate::graph::ResourceLifetime,
) -> bool {
    a.first_write <= b.last_use && b.first_write <= a.last_use
}

/// Describe a compiled graph's transient memory plan, for the editor's
/// frame-timing overlay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    /// Number of distinct allocations.
    pub allocations: usize,
    /// Bytes those allocations hold.
    pub bytes: u64,
    /// Allocations avoided by aliasing.
    pub saved: usize,
}

impl MemoryPlan {
    /// Summarise an allocator's plan.
    pub fn from_allocator(a: &AliasAllocator) -> MemoryPlan {
        MemoryPlan {
            allocations: a.groups().len(),
            bytes: a.required_bytes(),
            saved: a.saved_allocations(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PassGraph;
    use crate::registry::ResourceDesc;

    fn h(i: u32) -> ResourceHandle {
        ResourceHandle {
            index: i,
            generation: 0,
        }
    }

    fn transient(size: u64) -> ResourceDesc {
        ResourceDesc::Transient { size, format: 0 }
    }

    /// Two resources whose live ranges do not overlap.
    fn disjoint_graph() -> (CompiledGraph, ResourceRegistry) {
        let mut r = ResourceRegistry::new();
        let a = r.create(transient(1024), "color");
        let b = r.create(transient(1024), "depth");
        let mut g = PassGraph::new();
        g.add_pass("shadow").write(a).finish();
        g.add_pass("main").write(b).finish();
        (g.compile().unwrap(), r)
    }

    /// Two resources used by the same pass, so their ranges overlap.
    fn overlapping_graph() -> (CompiledGraph, ResourceRegistry) {
        let mut r = ResourceRegistry::new();
        let a = r.create(transient(1024), "color");
        let b = r.create(transient(1024), "depth");
        let mut g = PassGraph::new();
        g.add_pass("main").write(a).write(b).finish();
        (g.compile().unwrap(), r)
    }

    #[test]
    fn empty_plan_is_empty() {
        let a = AliasAllocator::new();
        assert!(a.groups().is_empty());
        assert_eq!(a.required_bytes(), 0);
        assert_eq!(a.saved_allocations(), 0);
    }

    #[test]
    fn two_transients_fit_one_group() {
        let (g, r) = disjoint_graph();
        let mut a = AliasAllocator::new();
        a.plan(&g, &r);
        assert_eq!(a.groups().len(), 1, "both transients share one allocation");
        assert_eq!(a.groups()[0].resources.len(), 2);
    }

    #[test]
    fn disjoint_lifetimes_alias() {
        let (g, r) = disjoint_graph();
        let mut a = AliasAllocator::new();
        a.plan(&g, &r);
        assert_eq!(
            a.saved_allocations(),
            1,
            "the second reuses the first allocation"
        );
        assert_eq!(a.required_bytes(), 1024, "one 1 KiB block serves both");
    }

    #[test]
    fn overlapping_lifetimes_do_not_alias() {
        let (g, r) = overlapping_graph();
        let mut a = AliasAllocator::new();
        a.plan(&g, &r);
        assert_eq!(
            a.saved_allocations(),
            0,
            "same-pass resources need separate memory"
        );
        assert_eq!(a.required_bytes(), 2048);
    }

    #[test]
    fn non_transient_resources_are_not_planned() {
        let mut r = ResourceRegistry::new();
        let a_res = r.create(ResourceDesc::Buffer { size: 4096 }, "ubo");
        let mut g = PassGraph::new();
        g.add_pass("main").write(a_res).finish();
        let mut a = AliasAllocator::new();
        a.plan(&g.compile().unwrap(), &r);
        assert!(a.groups().is_empty(), "a plain buffer is never aliased");
    }

    #[test]
    fn imported_resources_are_not_planned() {
        let mut r = ResourceRegistry::new();
        let imported = r.import(
            ResourceDesc::Transient {
                size: 512,
                format: 0,
            },
            "swapchain",
        );
        let mut g = PassGraph::new();
        g.add_pass("blit").write(imported).finish();
        let mut a = AliasAllocator::new();
        a.plan(&g.compile().unwrap(), &r);
        assert!(
            a.groups().is_empty(),
            "the graph does not own imported memory"
        );
    }

    #[test]
    fn a_group_takes_the_largest_size() {
        let mut r = ResourceRegistry::new();
        let small = r.create(transient(256), "small");
        let big = r.create(transient(4096), "big");
        let mut g = PassGraph::new();
        g.add_pass("one").write(small).finish();
        g.add_pass("two").write(big).finish();
        let mut a = AliasAllocator::new();
        a.plan(&g.compile().unwrap(), &r);
        assert_eq!(
            a.required_bytes(),
            4096,
            "the shared block must fit the larger"
        );
    }

    #[test]
    fn group_of_finds_the_containing_group() {
        let (g, r) = disjoint_graph();
        let mut a = AliasAllocator::new();
        a.plan(&g, &r);
        let first = a.groups()[0].resources[0];
        assert!(a.group_of(first).is_some());
    }

    #[test]
    fn planning_twice_is_stable() {
        let (g, r) = disjoint_graph();
        let mut a = AliasAllocator::new();
        a.plan(&g, &r);
        let first = a.groups().to_vec();
        a.plan(&g, &r);
        assert_eq!(
            a.groups(),
            first.as_slice(),
            "a per-frame plan must be reproducible"
        );
    }

    #[test]
    fn plan_resets_between_graphs() {
        let (g, r) = disjoint_graph();
        let mut a = AliasAllocator::new();
        a.plan(&g, &r);
        let before = a.groups().len();
        a.plan(&PassGraph::new().compile().unwrap(), &r);
        assert!(
            a.groups().len() < before,
            "a new graph replaces the old plan"
        );
    }

    #[test]
    fn saving_is_the_difference_between_the_two_totals() {
        let (g, r) = disjoint_graph();
        let mut a = AliasAllocator::new();
        a.plan(&g, &r);
        assert_eq!(a.unaliased_bytes(), 2048, "two 1 KiB transients, unaliased");
        assert_eq!(a.required_bytes(), 1024);
        assert_eq!(a.unaliased_bytes() - a.required_bytes(), 1024);
    }

    #[test]
    fn memory_plan_summarises() {
        let (g, r) = disjoint_graph();
        let mut a = AliasAllocator::new();
        a.plan(&g, &r);
        let plan = MemoryPlan::from_allocator(&a);
        assert_eq!(plan.allocations, 1);
        assert_eq!(plan.bytes, 1024);
        assert_eq!(plan.saved, 1);
    }

    #[test]
    fn overlap_detection() {
        let mut g = PassGraph::new();
        let a = g.add_pass("a").finish();
        let b = g.add_pass("b").finish();
        let life_a = crate::graph::ResourceLifetime {
            resource: h(0),
            first_write: a,
            last_use: a,
            first_use_is_a_write: true,
        };
        let life_b = crate::graph::ResourceLifetime {
            resource: h(1),
            first_write: b,
            last_use: b,
            first_use_is_a_write: true,
        };
        assert!(!lifetimes_overlap(&life_a, &life_b));
        assert!(
            lifetimes_overlap(&life_a, &life_a),
            "a resource overlaps itself"
        );
    }

    #[test]
    fn culled_passes_do_not_extend_lifetimes() {
        let mut r = ResourceRegistry::new();
        let a_res = r.create(transient(1024), "a");
        let mut g = PassGraph::new();
        g.add_pass("real").write(a_res).finish();
        g.add_pass("empty").mark_empty().finish();
        let mut a = AliasAllocator::new();
        a.plan(&g.compile().unwrap(), &r);
        assert_eq!(a.groups().len(), 1);
    }

    #[test]
    fn chain_of_three_shares_two_blocks() {
        let mut r = ResourceRegistry::new();
        let a = r.create(transient(1024), "a");
        let b = r.create(transient(1024), "b");
        let c = r.create(transient(1024), "c");
        let mut g = PassGraph::new();
        g.add_pass("p1").write(a).finish();
        g.add_pass("p2").write(b).finish();
        g.add_pass("p3").write(c).finish();
        let mut al = AliasAllocator::new();
        al.plan(&g.compile().unwrap(), &r);
        assert_eq!(al.groups().len(), 1, "only one block is ever needed");
        assert_eq!(al.saved_allocations(), 2);
    }
}
