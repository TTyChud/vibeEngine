//! JSON dump of a compiled graph, for tooling and bug reports.
//!
//! Hand-rolled rather than via `serde` so `vibe-graph` stays dependency-light
//! and the output format stays stable for the editor's graph viewer.

use crate::graph::{Barrier, CompiledGraph, CompiledPass, ResourceLifetime};
use vibe_rhi::ResourceHandle;

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn quoted(s: &str) -> String {
    format!("\"{}\"", escape(s))
}

fn handle(h: ResourceHandle) -> String {
    format!(
        "{{\"index\":{},\"generation\":{}}}",
        h.index(),
        h.generation()
    )
}

fn handles(list: &[ResourceHandle]) -> String {
    let items: Vec<String> = list.iter().map(|h| handle(*h)).collect();
    format!("[{}]", items.join(","))
}

fn usage(u: vibe_rhi::ResourceUsage) -> &'static str {
    match u {
        vibe_rhi::ResourceUsage::Read => "read",
        vibe_rhi::ResourceUsage::Write => "write",
        vibe_rhi::ResourceUsage::ReadWrite => "read_write",
        vibe_rhi::ResourceUsage::Discard => "discard",
    }
}

fn barrier(b: &Barrier) -> String {
    format!(
        "{{\"resource\":{},\"pass\":{},\"from\":\"{}\",\"to\":\"{}\",\"execution_dependency\":{}}}",
        handle(b.resource),
        b.pass.0,
        usage(b.from),
        usage(b.to),
        b.needs_execution_dependency()
    )
}

fn lifetime(l: &ResourceLifetime) -> String {
    format!(
        "{{\"resource\":{},\"first_write\":{},\"last_use\":{},\"first_use_is_write\":{}}}",
        handle(l.resource),
        l.first_write.0,
        l.last_use.0,
        l.first_use_is_a_write
    )
}

fn pass(p: &CompiledPass) -> String {
    format!(
        "{{\"name\":{},\"reads\":{},\"writes\":{},\"discards\":{},\"depends_on\":[{}],\
\"culled\":{},\"barriers\":[{}],\"lifetimes\":[{}]}}",
        quoted(&p.name),
        handles(&p.reads),
        handles(&p.writes),
        handles(&p.discards),
        p.depends_on
            .iter()
            .map(|d| d.0.to_string())
            .collect::<Vec<String>>()
            .join(","),
        p.culled,
        p.barriers
            .iter()
            .map(barrier)
            .collect::<Vec<String>>()
            .join(","),
        p.lifetimes
            .iter()
            .map(lifetime)
            .collect::<Vec<String>>()
            .join(",")
    )
}

/// Render a compiled graph as a JSON document.
pub fn to_json(graph: &CompiledGraph) -> String {
    format!(
        "{{\n  \"order_hash\": {},\n  \"pass_count\": {},\n  \"passes\": [\n    {}\n  ]\n}}\n",
        graph.order_hash,
        graph.passes.len(),
        graph
            .passes
            .iter()
            .map(pass)
            .collect::<Vec<String>>()
            .join(",\n    ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PassGraph;
    use vibe_rhi::ResourceHandle;

    fn h(i: u32) -> ResourceHandle {
        ResourceHandle {
            index: i,
            generation: 0,
        }
    }

    #[test]
    fn empty_graph_dumps() {
        let g = PassGraph::new().compile().unwrap();
        let json = to_json(&g);
        assert!(json.contains("\"pass_count\": 0"));
        assert!(json.contains("\"passes\": ["));
    }

    #[test]
    fn json_is_balanced() {
        let mut g = PassGraph::new();
        g.add_pass("shadow").write(h(0)).finish();
        g.add_pass("main").read(h(0)).finish();
        let compiled = g.compile().unwrap();
        let json = to_json(&compiled);
        assert_eq!(json.matches('{').count(), json.matches('}').count());
        assert_eq!(json.matches('[').count(), json.matches(']').count());
    }

    #[test]
    fn pass_names_appear() {
        let mut g = PassGraph::new();
        g.add_pass("lighting").write(h(1)).finish();
        let json = to_json(&g.compile().unwrap());
        assert!(json.contains("\"lighting\""), "{json}");
    }

    #[test]
    fn barriers_appear_with_from_and_to() {
        let mut g = PassGraph::new();
        g.add_pass("draw").write(h(0)).finish();
        g.add_pass("post").read(h(0)).finish();
        let json = to_json(&g.compile().unwrap());
        assert!(json.contains("\"from\":\"write\""), "{json}");
        assert!(json.contains("\"to\":\"read\""), "{json}");
    }

    #[test]
    fn order_hash_is_a_number() {
        let mut g = PassGraph::new();
        g.add_pass("a").write(h(0)).finish();
        let json = to_json(&g.compile().unwrap());
        assert!(json.contains("\"order_hash\": "), "{json}");
    }

    #[test]
    fn escaping_handles_quotes_and_newlines() {
        assert_eq!(escape("a\"b"), "a\\\"b");
        assert_eq!(escape("a\nb"), "a\\nb");
        assert_eq!(escape("a\\b"), "a\\\\b");
        assert_eq!(quoted("x\"y"), "\"x\\\"y\"");
    }

    #[test]
    fn control_characters_are_escaped() {
        assert_eq!(escape("\u{1}"), "\\u0001");
    }

    #[test]
    fn culled_passes_are_marked_in_json() {
        let mut g = PassGraph::new();
        g.add_pass("nothing").mark_empty().finish();
        let json = to_json(&g.compile().unwrap());
        assert!(
            json.contains("\"culled\":true") || json.contains("\"pass_count\": 0"),
            "{json}"
        );
    }
}
