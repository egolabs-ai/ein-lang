//! Benchmarks for sparse tensor joins.
//!
//! Key comparison: Ein's sparse join vs DataFrog.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use datafrog::Iteration;
use ein::SparseBool;

/// Create a chain relation: 0->1, 1->2, ..., (n-1)->n
fn make_chain(n: usize) -> SparseBool {
    let mut rel = SparseBool::new(2);
    for i in 0..n {
        rel.insert(vec![i, i + 1]);
    }
    rel
}

/// Create a random relation with approximately n tuples
fn make_random(n: usize, domain: usize) -> SparseBool {
    let mut rel = SparseBool::new(2);
    // Simple pseudo-random using a fixed seed pattern
    let mut x = 12345usize;
    for _ in 0..n {
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let a = x % domain;
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let b = x % domain;
        rel.insert(vec![a, b]);
    }
    rel
}

fn bench_sparse_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_join");

    // Join on chain relations of different sizes
    for size in [100, 500, 1000, 5000].iter() {
        let rel = make_chain(*size);

        group.bench_with_input(
            BenchmarkId::new("chain_self_join", size),
            &rel,
            |bench, rel| {
                bench.iter(|| rel.join(rel, 1, 0));
            },
        );
    }

    // Join on random relations
    for size in [100, 500, 1000].iter() {
        let rel = make_random(*size, 1000);

        group.bench_with_input(
            BenchmarkId::new("random_self_join", size),
            &rel,
            |bench, rel| {
                bench.iter(|| rel.join(rel, 1, 0));
            },
        );
    }

    group.finish();
}

fn bench_transitive_closure(c: &mut Criterion) {
    let mut group = c.benchmark_group("transitive_closure");

    // Transitive closure on chains of different lengths
    for size in [10, 25, 50, 100].iter() {
        let parent = make_chain(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &parent,
            |bench, parent| {
                bench.iter(|| {
                    let mut ancestor = parent.clone();
                    for _ in 0..(*size + 5) {
                        let old_len = ancestor.len();
                        let joined = ancestor.join(parent, 1, 0);
                        let new_pairs = joined.project(&[0, 2]);
                        ancestor = ancestor.union(&new_pairs);
                        if ancestor.len() == old_len {
                            break;
                        }
                    }
                    ancestor
                });
            },
        );
    }

    group.finish();
}

fn bench_project(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_project");

    for size in [100, 500, 1000, 5000].iter() {
        // Create a ternary relation (from join result)
        let rel = make_chain(*size);
        let joined = rel.join(&rel, 1, 0); // Creates 3-ary relation

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &joined,
            |bench, joined| {
                bench.iter(|| joined.project(&[0, 2]));
            },
        );
    }

    group.finish();
}

/// DataFrog transitive closure for comparison
fn datafrog_tc(edges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut iteration = Iteration::new();
    let edges_var = iteration.variable::<(usize, usize)>("edges");
    let reachable = iteration.variable::<(usize, usize)>("reachable");

    // Load edges
    edges_var.extend(edges.iter().copied());
    reachable.extend(edges.iter().copied());

    while iteration.changed() {
        // reachable(x, z) <- reachable(x, y), edges(y, z)
        reachable.from_join(&reachable, &edges_var, |&_y, &x, &z| (x, z));
    }

    reachable.complete().iter().copied().collect()
}

fn bench_tc_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("tc_ein_vs_datafrog");

    for size in [10, 25, 50, 100].iter() {
        // Create chain edges for both
        let edges: Vec<(usize, usize)> = (0..*size).map(|i| (i, i + 1)).collect();
        let ein_rel = make_chain(*size);

        // Ein TC
        group.bench_with_input(
            BenchmarkId::new("ein", size),
            &ein_rel,
            |bench, parent| {
                bench.iter(|| {
                    let mut ancestor = parent.clone();
                    for _ in 0..(*size + 5) {
                        let old_len = ancestor.len();
                        let joined = ancestor.join(parent, 1, 0);
                        let new_pairs = joined.project(&[0, 2]);
                        ancestor = ancestor.union(&new_pairs);
                        if ancestor.len() == old_len {
                            break;
                        }
                    }
                    ancestor
                });
            },
        );

        // DataFrog TC
        group.bench_with_input(
            BenchmarkId::new("datafrog", size),
            &edges,
            |bench, edges| {
                bench.iter(|| datafrog_tc(edges));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sparse_join,
    bench_transitive_closure,
    bench_project,
    bench_tc_comparison
);
criterion_main!(benches);