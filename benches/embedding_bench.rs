//! Embedding benchmarks comparable to kocoro-lab/tensorlogic.
//!
//! Scenarios from their benchmark_suite.py:
//! - family: N=10, parent relation, target = parent ∘ parent (grandparent)
//! - smallkg: N=300, r0/r1/r2 relations, target = r0 ∘ r1
//! - synthetic: N=200, r0/r1 relations, target = r0 ∘ r0

use candle_core::Device;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ein::{EmbeddingSpace, SparseBool};

/// Create a chain relation: 0->1, 1->2, ..., (n-2)->(n-1)
fn make_chain(n: usize) -> SparseBool {
    let mut rel = SparseBool::new(2);
    for i in 0..(n - 1) {
        rel.insert(vec![i, i + 1]);
    }
    rel
}

/// Create a random sparse relation with approximately `density * n * n` tuples
fn make_random(n: usize, density: f64, seed: usize) -> SparseBool {
    let mut rel = SparseBool::new(2);
    let target_count = ((n * n) as f64 * density) as usize;

    let mut x = seed;
    for _ in 0..target_count {
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let a = x % n;
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let b = x % n;
        rel.insert(vec![a, b]);
    }
    rel
}

/// Benchmark: Embed a relation (sum of outer products)
fn bench_embed_relation(c: &mut Criterion) {
    let mut group = c.benchmark_group("embed_relation");

    // Match kocoro-lab scenarios
    for (name, n, density) in [
        ("family", 10, 0.1),     // ~10 facts
        ("smallkg", 300, 0.01),  // ~900 facts
        ("synthetic", 200, 0.02), // ~800 facts
    ] {
        let device = Device::Cpu;
        let space = EmbeddingSpace::new(n, 32, &device).unwrap(); // dim=32 like kocoro
        let rel = make_random(n, density, 12345);

        group.bench_with_input(BenchmarkId::new("cpu", name), &(&space, &rel), |b, (space, rel)| {
            b.iter(|| space.embed_relation(rel).unwrap());
        });
    }

    group.finish();
}

/// Benchmark: Query a single fact
fn bench_query_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_single");

    for (name, n, density) in [
        ("family", 10, 0.1),
        ("smallkg", 300, 0.01),
        ("synthetic", 200, 0.02),
    ] {
        let device = Device::Cpu;
        let space = EmbeddingSpace::new(n, 32, &device).unwrap();
        let rel = make_random(n, density, 12345);
        let emb_rel = space.embed_relation(&rel).unwrap();

        group.bench_with_input(BenchmarkId::new("t0", name), &(&space, &emb_rel), |b, (space, emb_rel)| {
            b.iter(|| space.query(emb_rel, 0, 1, 0.0).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("t0.2", name), &(&space, &emb_rel), |b, (space, emb_rel)| {
            b.iter(|| space.query(emb_rel, 0, 1, 0.2).unwrap());
        });
    }

    group.finish();
}

/// Benchmark: Query all pairs (returns N×N matrix)
fn bench_query_all_pairs(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_all_pairs");

    for (name, n, density) in [
        ("family", 10, 0.1),
        ("smallkg", 300, 0.01),
        ("synthetic", 200, 0.02),
    ] {
        let device = Device::Cpu;
        let space = EmbeddingSpace::new(n, 32, &device).unwrap();
        let rel = make_random(n, density, 12345);
        let emb_rel = space.embed_relation(&rel).unwrap();

        group.bench_with_input(BenchmarkId::new("t0", name), &(&space, &emb_rel), |b, (space, emb_rel)| {
            b.iter(|| space.query_all_pairs(emb_rel, 0.0).unwrap());
        });
    }

    group.finish();
}

/// Benchmark: Relation composition (grandparent = parent ∘ parent)
fn bench_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("composition");

    for (name, n, density) in [
        ("family", 10, 0.1),
        ("smallkg", 300, 0.01),
        ("synthetic", 200, 0.02),
    ] {
        let device = Device::Cpu;
        let space = EmbeddingSpace::new(n, 32, &device).unwrap();
        let rel = make_random(n, density, 12345);
        let emb_rel = space.embed_relation(&rel).unwrap();

        group.bench_with_input(BenchmarkId::new("cpu", name), &(&space, &emb_rel), |b, (space, emb_rel)| {
            b.iter(|| space.compose(emb_rel, emb_rel).unwrap());
        });
    }

    group.finish();
}

/// Benchmark: Full pipeline (embed + compose + query_all)
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    for (name, n, density) in [
        ("family", 10, 0.1),
        ("smallkg", 300, 0.01),
        ("synthetic", 200, 0.02),
    ] {
        let device = Device::Cpu;
        let space = EmbeddingSpace::new(n, 32, &device).unwrap();
        let rel = make_random(n, density, 12345);

        group.bench_with_input(BenchmarkId::new("cpu", name), &(&space, &rel), |b, (space, rel)| {
            b.iter(|| {
                let emb_rel = space.embed_relation(rel).unwrap();
                let composed = space.compose(&emb_rel, &emb_rel).unwrap();
                space.query_all_pairs(&composed, 0.0).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark: Embedding dimension scaling
fn bench_dim_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dim_scaling");
    let n = 100;
    let density = 0.02;
    let device = Device::Cpu;

    for dim in [32, 64, 128, 256] {
        let space = EmbeddingSpace::new(n, dim, &device).unwrap();
        let rel = make_random(n, density, 12345);

        group.bench_with_input(BenchmarkId::new("embed", dim), &(&space, &rel), |b, (space, rel)| {
            b.iter(|| space.embed_relation(rel).unwrap());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_embed_relation,
    bench_query_single,
    bench_query_all_pairs,
    bench_composition,
    bench_full_pipeline,
    bench_dim_scaling,
);
criterion_main!(benches);