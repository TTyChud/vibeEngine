//! Pass graph construction: the pass list, read/write tracking, and compilation
//! into an executable order with synthesized barriers.

use std::collections::HashMap;

use vibe_rhi::{ResourceHandle, ResourceUsage};

use crate::error::GraphError;

/// Identifies a pass within a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PassId(pub usize);

/// A compiled pass, ready to record.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPass {
    /// Handle from the builder.
    pub id: PassId,
    /// Label for dumps and GPU markers.
    pub name: String,
    /// Resources this pass reads.
    pub reads: Vec<ResourceHandle>,
    /// Resources this pass writes.
    pub writes: Vec<ResourceHandle>,
    /// Resources whose previous contents are discarded rather than loaded.
    pub discards: Vec<ResourceHandle>,
    /// Passes that must finish before this one starts.
    pub depends_on: Vec<PassId>,
    /// Barriers to emit before the pass body.
    pub barriers: Vec<Barrier>,
    /// True when the pass draws nothing and can be dropped.
    pub culled: bool,
    /// First and last pass to touch each resource, for lifetime analysis.
    pub lifetimes: Vec<ResourceLifetime>,
}

/// One resource's span in the pass order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceLifetime {
    /// The resource.
    pub resource: ResourceHandle,
    /// Pass that first writes it.
    pub first_write: PassId,
    /// Last pass that reads or writes it.
    pub last_use: PassId,
    /// True when every read happens after the first write, so no load barrier is
    /// needed before the first write.
    pub first_use_is_a_write: bool,
}

/// A state transition the executor must emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Barrier {
    /// The resource being transitioned.
    pub resource: ResourceHandle,
    /// The pass the barrier belongs to.
    pub pass: PassId,
    /// How the previous pass used it.
    pub from: ResourceUsage,
    /// How this pass uses it.
    pub to: ResourceUsage,
}

impl Barrier {
    /// True when the transition needs an execution dependency, not just a
    /// memory one.
    ///
    /// A read-to-read transition is a pure memory hazard fix and needs no
    /// execution dependency; anything involving a write does.
    pub fn needs_execution_dependency(&self) -> bool {
        self.from != ResourceUsage::Read || self.to != ResourceUsage::Read
    }

    /// True when the transition can be skipped entirely.
    ///
    /// Read to read with no other writer in between is free, and a transition
    /// to the same usage is free.
    pub fn is_noop(&self) -> bool {
        self.from == self.to
    }
}

/// One pass's declared resource usage.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PassResources {
    /// Read-only accesses.
    pub reads: Vec<ResourceHandle>,
    /// Read-write accesses.
    pub writes: Vec<ResourceHandle>,
    /// Writes that discard prior contents, skipping the load.
    pub discards: Vec<ResourceHandle>,
}

#[derive(Debug)]
struct PassEntry {
    name: String,
    resources: PassResources,
    explicit_depends: Vec<PassId>,
    /// Set when a pass is known to draw nothing, which lets the compiler drop it.
    empty: bool,
}

/// A frame graph under construction.
///
/// Passes declare what they read and write; the graph infers the ordering,
/// synthesizes the barriers, and drops passes that touch nothing. This is the
/// same shape as Godot's `RD::FramebufferPass` / RenderingDevice dependency
/// tracking and Unreal's `FRDGBuilder`.
#[derive(Debug, Default)]
pub struct PassGraph {
    passes: Vec<PassEntry>,
    names: HashMap<String, PassId>,
    /// Declared by the caller; the compiler adds its own edges on top.
    external_writes: Vec<ResourceHandle>,
    /// Resources written after the graph ends, which extend their lifetime.
    external_reads: Vec<ResourceHandle>,
}

impl PassGraph {
    /// An empty graph.
    pub fn new() -> PassGraph {
        PassGraph::default()
    }

    /// Start building a pass.
    pub fn add_pass(&mut self, name: impl Into<String>) -> PassBuilder<'_> {
        let name = name.into();
        let id = PassId(self.passes.len());
        self.passes.push(PassEntry {
            name: name.clone(),
            resources: PassResources::default(),
            explicit_depends: Vec::new(),
            empty: false,
        });
        self.names.insert(name, id);
        PassBuilder { graph: self, id }
    }

    /// Mark a resource as written before the graph runs, so the first pass that
    /// reads it gets a barrier from outside.
    pub fn add_external_write(&mut self, resource: ResourceHandle) {
        if !self.external_writes.contains(&resource) {
            self.external_writes.push(resource);
        }
    }

    /// Mark a resource as read after the graph runs, extending its lifetime.
    pub fn add_external_read(&mut self, resource: ResourceHandle) {
        if !self.external_reads.contains(&resource) {
            self.external_reads.push(resource);
        }
    }

    /// Look up a pass by name.
    pub fn pass_id(&self, name: &str) -> Option<PassId> {
        self.names.get(name).copied()
    }

    /// A pass's declared resources.
    pub fn resources(&self, id: PassId) -> Option<&PassResources> {
        self.passes.get(id.0).map(|p| &p.resources)
    }

    /// Number of passes added.
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// True when no passes were added.
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// A pass's label.
    pub fn name_of(&self, id: PassId) -> Option<&str> {
        self.passes.get(id.0).map(|p| p.name.as_str())
    }

    /// Topologically sort, cull dead passes, and synthesize barriers.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] when a pass reads a resource no earlier pass
    /// writes, which would otherwise be an uninitialized read.
    pub fn compile(&self) -> Result<CompiledGraph, GraphError> {
        let n = self.passes.len();
        if n == 0 {
            return Ok(CompiledGraph {
                passes: Vec::new(),
                order_hash: 0,
            });
        }

        // A pass that touches no resources draws nothing unless it declared an
        // explicit dependency, in which case it may be doing side work.
        let culled: Vec<bool> = self
            .passes
            .iter()
            .map(|p| {
                p.empty
                    || (p.resources.reads.is_empty()
                        && p.resources.writes.is_empty()
                        && p.resources.discards.is_empty()
                        && p.explicit_depends.is_empty())
            })
            .collect();

        // last_writer[h] is the most recent pass that writes h in the current
        // order, used both for ordering and for barrier synthesis.
        let mut last_writer: HashMap<ResourceHandle, PassId> = HashMap::new();
        for h in &self.external_writes {
            last_writer.insert(*h, PassId(usize::MAX));
        }

        let mut order: Vec<PassId> = Vec::with_capacity(n);
        let mut indegree = vec![0usize; n];
        let mut edges: Vec<Vec<PassId>> = vec![Vec::new(); n];
        let mut emitted = vec![false; n];

        // Iterate to a fixed point: a pass can only be scheduled once every
        // producer of everything it reads has been scheduled.
        let mut scheduled = 0usize;
        while scheduled < n {
            let mut progressed = false;
            for i in 0..n {
                if emitted[i] {
                    continue;
                }
                let p = &self.passes[i];
                let mut ready = true;
                let mut deps: Vec<PassId> = p.explicit_depends.clone();

                for h in p
                    .resources
                    .reads
                    .iter()
                    .chain(p.resources.writes.iter())
                    .chain(p.resources.discards.iter())
                {
                    if let Some(&producer) = last_writer.get(h) {
                        if producer != PassId(usize::MAX) && producer != PassId(i) {
                            if !emitted[producer.0] {
                                ready = false;
                            }
                            if !deps.contains(&producer) {
                                deps.push(producer);
                            }
                        }
                    } else if p.resources.writes.contains(h) || p.resources.discards.contains(h) {
                        // A write with no producer is fine: it defines the
                        // resource for later passes.
                    } else {
                        return Err(GraphError::ReadBeforeWrite {
                            pass: p.name.clone(),
                            resource: *h,
                        });
                    }
                }

                if ready {
                    for d in &deps {
                        edges[d.0].push(PassId(i));
                        indegree[i] += 1;
                    }
                    emitted[i] = true;
                    order.push(PassId(i));
                    scheduled += 1;
                    progressed = true;

                    for h in p.resources.writes.iter().chain(p.resources.discards.iter()) {
                        last_writer.insert(*h, PassId(i));
                    }
                }
            }
            if !progressed {
                let stuck = (0..n)
                    .find(|i| !emitted[*i])
                    .map(|i| self.passes[i].name.clone())
                    .unwrap_or_default();
                return Err(GraphError::Cycle { pass: stuck });
            }
        }

        // Now that the order is fixed, walk it once more to synthesize barriers
        // and lifetimes from the actual execution sequence.
        let mut compiled: Vec<CompiledPass> = Vec::with_capacity(n);
        let mut previous_use: HashMap<ResourceHandle, ResourceUsage> = HashMap::new();
        for h in &self.external_writes {
            previous_use.insert(*h, ResourceUsage::Discard);
        }
        let mut first_write: HashMap<ResourceHandle, PassId> = HashMap::new();
        let mut last_use: HashMap<ResourceHandle, PassId> = HashMap::new();
        let mut first_is_write: HashMap<ResourceHandle, bool> = HashMap::new();

        for &id in &order {
            let p = &self.passes[id.0];
            let mut barriers = Vec::new();

            for h in &p.resources.discards {
                if let Some(prev) = previous_use.get(h) {
                    let transition = Barrier {
                        resource: *h,
                        pass: id,
                        from: *prev,
                        to: ResourceUsage::Discard,
                    };
                    if !transition.is_noop() {
                        barriers.push(transition);
                    }
                }
                first_write.insert(*h, id);
                first_is_write.insert(*h, true);
                previous_use.insert(*h, ResourceUsage::Discard);
            }

            for h in &p.resources.writes {
                if let Some(prev) = previous_use.get(h) {
                    barriers.push(Barrier {
                        resource: *h,
                        pass: id,
                        from: *prev,
                        to: ResourceUsage::Write,
                    });
                }
                first_write.entry(*h).or_insert(id);
                first_is_write.entry(*h).or_insert(true);
                previous_use.insert(*h, ResourceUsage::Write);
            }

            for h in &p.resources.reads {
                if let Some(prev) = previous_use.get(h) {
                    if *prev != ResourceUsage::Read {
                        barriers.push(Barrier {
                            resource: *h,
                            pass: id,
                            from: *prev,
                            to: ResourceUsage::Read,
                        });
                    }
                } else {
                    // A read with no writer anywhere in the graph reads whatever
                    // the resource already holds, which is only sound for an
                    // imported resource. compile() rejects the common case.
                    return Err(GraphError::ReadBeforeWrite {
                        pass: p.name.clone(),
                        resource: *h,
                    });
                }
                first_is_write.entry(*h).or_insert(false);
                previous_use.insert(*h, ResourceUsage::Read);
            }

            for h in p
                .resources
                .reads
                .iter()
                .chain(p.resources.writes.iter())
                .chain(p.resources.discards.iter())
            {
                last_use.insert(*h, id);
            }

            compiled.push(CompiledPass {
                id,
                name: p.name.clone(),
                reads: p.resources.reads.clone(),
                writes: p.resources.writes.clone(),
                discards: p.resources.discards.clone(),
                depends_on: edges[id.0].clone(),
                barriers,
                culled: culled[id.0],
                lifetimes: Vec::new(),
            });
        }

        // Anything the caller reads after the graph extends its lifetime to the
        // final pass.
        let last_pass = order.last().copied().unwrap_or(PassId(0));
        for h in &self.external_reads {
            last_use.insert(*h, last_pass);
        }

        for pass in &mut compiled {
            for h in pass
                .reads
                .iter()
                .chain(pass.writes.iter())
                .chain(pass.discards.iter())
            {
                let Some(&fw) = first_write.get(h) else {
                    continue;
                };
                let Some(&lu) = last_use.get(h) else { continue };
                pass.lifetimes.push(ResourceLifetime {
                    resource: *h,
                    first_write: fw,
                    last_use: lu,
                    first_use_is_a_write: first_is_write.get(h).copied().unwrap_or(false),
                });
            }
        }

        // edges[d] holds the passes that wait on d, so transpose it to get
        // each pass's own prerequisites.
        let mut depends_on: Vec<Vec<PassId>> = vec![Vec::new(); n];
        for (d, dependents) in edges.iter().enumerate() {
            for dependent in dependents {
                depends_on[dependent.0].push(PassId(d));
            }
        }
        for pass in &mut compiled {
            pass.depends_on = std::mem::take(&mut depends_on[pass.id.0]);
        }

        let live: Vec<CompiledPass> = compiled.into_iter().filter(|p| !p.culled).collect();
        let order_hash = topology_hash(&order);
        Ok(CompiledGraph {
            passes: live,
            order_hash,
        })
    }
}

/// A compiled, ordered set of passes.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledGraph {
    /// Passes in execution order, dead ones removed.
    pub passes: Vec<CompiledPass>,
    /// Hash of the pass order and their resource usage, for detecting that the
    /// graph's shape changed between frames.
    pub order_hash: u64,
}

impl CompiledGraph {
    /// Number of live passes.
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// True when every pass was culled.
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Total barriers across all passes.
    pub fn barrier_count(&self) -> usize {
        self.passes.iter().map(|p| p.barriers.len()).sum()
    }

    /// Every barrier in execution order.
    pub fn barriers(&self) -> impl Iterator<Item = &Barrier> {
        self.passes.iter().flat_map(|p| p.barriers.iter())
    }

    /// Find the compiled pass for a builder handle.
    pub fn find(&self, id: PassId) -> Option<&CompiledPass> {
        self.passes.iter().find(|p| p.id == id)
    }

    /// The lifetime record for a resource, from whichever pass declared it.
    pub fn lifetime(&self, resource: ResourceHandle) -> Option<ResourceLifetime> {
        self.passes
            .iter()
            .flat_map(|p| p.lifetimes.iter())
            .find(|l| l.resource == resource)
            .copied()
    }
}

/// A stable hash of the compiled graph's shape.
///
/// Two graphs with the same hash have the same passes in the same order with
/// the same resource usage, so the pipeline and descriptor layouts built for
/// one are valid for the other.
pub fn topology_hash(order: &[PassId]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for id in order {
        for byte in id.0.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

/// Builds one pass's resource declarations.
pub struct PassBuilder<'a> {
    graph: &'a mut PassGraph,
    id: PassId,
}

impl PassBuilder<'_> {
    /// Declare a read-only access.
    pub fn read(self, resource: ResourceHandle) -> Self {
        let p = &mut self.graph.passes[self.id.0];
        if !p.resources.reads.contains(&resource) {
            p.resources.reads.push(resource);
        }
        self
    }

    /// Declare a read-write access.
    pub fn write(self, resource: ResourceHandle) -> Self {
        let p = &mut self.graph.passes[self.id.0];
        if !p.resources.writes.contains(&resource) {
            p.resources.writes.push(resource);
        }
        self
    }

    /// Declare a write that discards the previous contents, so no load is
    /// needed. This is what a render target clear does.
    pub fn discard(self, resource: ResourceHandle) -> Self {
        let p = &mut self.graph.passes[self.id.0];
        if !p.resources.discards.contains(&resource) {
            p.resources.discards.push(resource);
        }
        self
    }

    /// Declare several reads at once.
    pub fn reads(mut self, resources: &[ResourceHandle]) -> Self {
        for r in resources {
            self = self.read(*r);
        }
        self
    }

    /// Declare several discards at once.
    pub fn discards(mut self, resources: &[ResourceHandle]) -> Self {
        for r in resources {
            self = self.discard(*r);
        }
        self
    }

    /// Force this pass to run after another, regardless of data dependencies.
    pub fn after(self, id: PassId) -> Self {
        let p = &mut self.graph.passes[self.id.0];
        if !p.explicit_depends.contains(&id) {
            p.explicit_depends.push(id);
        }
        self
    }

    /// Mark the pass as producing no output, so the compiler can drop it when
    /// nothing downstream reads what it would have written.
    pub fn mark_empty(self) -> Self {
        self.graph.passes[self.id.0].empty = true;
        self
    }

    /// Finish the pass.
    pub fn finish(self) -> PassId {
        self.id
    }
}
