//! Benchmarks for forward chaining (Datalog-style fixpoint).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ein::{parse, KnowledgeBase};

/// Create a chain of parent facts: 0->1, 1->2, ..., (n-1)->n
fn setup_chain(n: usize) -> KnowledgeBase {
    let mut kb = KnowledgeBase::new();
    for i in 0..n {
        let parent = format!("P{}", i);
        let child = format!("P{}", i + 1);
        kb.add_fact("Parent", &[&parent, &child]);
    }

    // Add ancestor rules
    let rule1 = parse("Ancestor(x,y) <- Parent(x,y)").unwrap();
    let rule2 = parse("Ancestor(x,z) <- Ancestor(x,y) Parent(y,z)").unwrap();
    kb.add_rule(&rule1);
    kb.add_rule(&rule2);

    kb
}

/// Create a binary tree of parent facts
fn setup_tree(depth: usize) -> KnowledgeBase {
    let mut kb = KnowledgeBase::new();
    let mut id = 0;

    fn add_children(kb: &mut KnowledgeBase, parent_id: usize, depth: usize, id: &mut usize) {
        if depth == 0 {
            return;
        }

        let parent = format!("N{}", parent_id);

        // Left child
        *id += 1;
        let left = format!("N{}", *id);
        kb.add_fact("Parent", &[&parent, &left]);
        add_children(kb, *id, depth - 1, id);

        // Right child
        *id += 1;
        let right = format!("N{}", *id);
        kb.add_fact("Parent", &[&parent, &right]);
        add_children(kb, *id, depth - 1, id);
    }

    add_children(&mut kb, 0, depth, &mut id);

    // Add ancestor rules
    let rule1 = parse("Ancestor(x,y) <- Parent(x,y)").unwrap();
    let rule2 = parse("Ancestor(x,z) <- Ancestor(x,y) Parent(y,z)").unwrap();
    kb.add_rule(&rule1);
    kb.add_rule(&rule2);

    kb
}

fn bench_forward_chain_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_chain_linear");

    // Chain lengths - forward chain should complete in O(n) iterations
    for size in [10, 25, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bench, &size| {
                bench.iter(|| {
                    let mut kb = setup_chain(size);
                    kb.forward_chain()
                });
            },
        );
    }

    group.finish();
}

fn bench_forward_chain_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_chain_tree");

    // Tree depths - more nodes, but shallower derivation chains
    for depth in [3, 4, 5, 6].iter() {
        let num_nodes = (1 << (*depth + 1)) - 1; // 2^(d+1) - 1

        group.bench_with_input(
            BenchmarkId::new("depth", depth),
            depth,
            |bench, &depth| {
                bench.iter(|| {
                    let mut kb = setup_tree(depth);
                    kb.forward_chain()
                });
            },
        );

        // Log the size for context
        eprintln!("Tree depth {} has {} nodes", depth, num_nodes);
    }

    group.finish();
}

fn bench_iterations_count(c: &mut Criterion) {
    // This measures how many iterations it takes to reach fixpoint
    let group = c.benchmark_group("fixpoint_iterations");

    for size in [10, 25, 50].iter() {
        let mut kb = setup_chain(*size);
        let iterations = kb.forward_chain();
        eprintln!("Chain of {} needs {} iterations", size, iterations);
    }

    drop(group);
}

criterion_group!(
    benches,
    bench_forward_chain_linear,
    bench_forward_chain_tree,
    bench_iterations_count
);
criterion_main!(benches);