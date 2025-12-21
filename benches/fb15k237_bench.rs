//! FB15k-237 Knowledge Graph Benchmark
//!
//! Tests embedding-based link prediction on the FB15k-237 dataset.
//! Compares CPU vs GPU (Metal) performance.
//!
//! Run with:
//!   cargo bench --bench fb15k237_bench --features metal

use candle_core::{Device, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ein::{EmbeddingSpace, SparseBool};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

/// Load FB15k-237 dataset from TSV file.
/// Returns: (entity_to_id, relation_to_id, triples as Vec<(head, rel, tail)>)
fn load_fb15k237(
    path: &str,
) -> (
    HashMap<String, usize>,
    HashMap<String, usize>,
    Vec<(usize, usize, usize)>,
) {
    let file = File::open(path).expect("Failed to open FB15k-237 file");
    let reader = BufReader::new(file);

    let mut entity_to_id: HashMap<String, usize> = HashMap::new();
    let mut relation_to_id: HashMap<String, usize> = HashMap::new();
    let mut triples = Vec::new();

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 3 {
            continue;
        }

        let head = parts[0].to_string();
        let rel = parts[1].to_string();
        let tail = parts[2].to_string();

        // Get or insert entity IDs
        let num_entities = entity_to_id.len();
        let head_id = *entity_to_id.entry(head).or_insert(num_entities);

        let num_entities = entity_to_id.len();
        let tail_id = *entity_to_id.entry(tail).or_insert(num_entities);

        let num_relations = relation_to_id.len();
        let rel_id = *relation_to_id.entry(rel).or_insert(num_relations);

        triples.push((head_id, rel_id, tail_id));
    }

    (entity_to_id, relation_to_id, triples)
}

/// Convert triples to per-relation SparseBool structures.
fn triples_to_relations(
    triples: &[(usize, usize, usize)],
    num_relations: usize,
) -> Vec<SparseBool> {
    let mut relations: Vec<SparseBool> = (0..num_relations).map(|_| SparseBool::new(2)).collect();

    for &(head, rel, tail) in triples {
        relations[rel].insert(vec![head, tail]);
    }

    relations
}

/// Compute AUC for link prediction.
/// For each positive triple (h, r, t), sample negative triples by corrupting tail.
/// Returns AUC score.
fn compute_auc(
    space: &EmbeddingSpace,
    emb_relations: &[Tensor],
    triples: &[(usize, usize, usize)],
    num_entities: usize,
    num_samples: usize,
) -> f64 {
    let mut correct = 0usize;
    let mut total = 0usize;

    // Use a deterministic "random" generator
    let mut rng_state = 42usize;

    for &(head, rel, tail) in triples.iter().take(num_samples) {
        // Score for positive triple
        let pos_score = space.query_raw_score(&emb_relations[rel], head, tail).unwrap_or(0.0);

        // Generate negative by corrupting tail
        let mut rng_state_local = rng_state;
        rng_state_local = rng_state_local.wrapping_mul(1103515245).wrapping_add(12345);
        let neg_tail = rng_state_local % num_entities;
        rng_state = rng_state_local;

        if neg_tail == tail {
            continue; // Skip if accidentally same
        }

        let neg_score = space.query_raw_score(&emb_relations[rel], head, neg_tail).unwrap_or(0.0);

        // AUC: count how often positive score > negative score
        if pos_score > neg_score {
            correct += 1;
        }
        total += 1;
    }

    if total == 0 {
        0.5
    } else {
        correct as f64 / total as f64
    }
}

/// Benchmark: Embed all relations
fn bench_embed_relations(c: &mut Criterion) {
    let data_path = "data/Release/train.txt";
    if !std::path::Path::new(data_path).exists() {
        eprintln!("FB15k-237 not found at {}. Skipping benchmark.", data_path);
        return;
    }

    let (entities, relations, triples) = load_fb15k237(data_path);
    let num_entities = entities.len();
    let num_relations = relations.len();
    let sparse_relations = triples_to_relations(&triples, num_relations);

    println!(
        "Loaded FB15k-237: {} entities, {} relations, {} triples",
        num_entities,
        num_relations,
        triples.len()
    );

    let mut group = c.benchmark_group("fb15k237_embed");

    // Test with different embedding dimensions
    for dim in [64, 128, 256] {
        // CPU benchmark
        let space_cpu = EmbeddingSpace::new(num_entities, dim, &Device::Cpu).unwrap();

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("dim{}", dim)),
            &(&space_cpu, &sparse_relations),
            |b, (space, rels)| {
                b.iter(|| {
                    let mut embedded = Vec::new();
                    for rel in rels.iter().take(10) {
                        // First 10 relations
                        embedded.push(space.embed_relation(rel).unwrap());
                    }
                    embedded
                });
            },
        );

        // GPU (Metal) benchmark
        #[cfg(feature = "metal")]
        {
            if let Ok(metal_device) = Device::new_metal(0) {
                let space_gpu = EmbeddingSpace::new(num_entities, dim, &metal_device).unwrap();

                group.bench_with_input(
                    BenchmarkId::new("metal", format!("dim{}", dim)),
                    &(&space_gpu, &sparse_relations),
                    |b, (space, rels)| {
                        b.iter(|| {
                            let mut embedded = Vec::new();
                            for rel in rels.iter().take(10) {
                                embedded.push(space.embed_relation(rel).unwrap());
                            }
                            embedded
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark: Query all pairs for a relation
fn bench_query_all(c: &mut Criterion) {
    let data_path = "data/Release/train.txt";
    if !std::path::Path::new(data_path).exists() {
        return;
    }

    let (entities, relations, triples) = load_fb15k237(data_path);
    let num_entities = entities.len();
    let num_relations = relations.len();
    let sparse_relations = triples_to_relations(&triples, num_relations);

    // Use a subset of entities for query_all (full 14k x 14k is too large)
    let subset_size = 1000;

    let mut group = c.benchmark_group("fb15k237_query_all");
    let dim = 128;

    // Create subset embedding space
    let space_cpu = EmbeddingSpace::new(subset_size, dim, &Device::Cpu).unwrap();

    // Create a subset relation
    let mut subset_rel = SparseBool::new(2);
    for tuple in sparse_relations[0].tuples.iter().take(100) {
        if tuple[0] < subset_size && tuple[1] < subset_size {
            subset_rel.insert(vec![tuple[0], tuple[1]]);
        }
    }

    let emb_rel_cpu = space_cpu.embed_relation(&subset_rel).unwrap();

    // Use temperature=0 (threshold mode) to avoid sigmoid which isn't implemented in Metal
    group.bench_with_input(
        BenchmarkId::new("cpu", format!("{}x{}", subset_size, subset_size)),
        &(&space_cpu, &emb_rel_cpu),
        |b, (space, emb_rel)| {
            b.iter(|| space.query_all_pairs(emb_rel, 0.0).unwrap());
        },
    );

    #[cfg(feature = "metal")]
    {
        if let Ok(metal_device) = Device::new_metal(0) {
            let space_gpu = EmbeddingSpace::new(subset_size, dim, &metal_device).unwrap();
            let mut subset_rel_gpu = SparseBool::new(2);
            for tuple in subset_rel.tuples.iter() {
                subset_rel_gpu.insert(tuple.clone());
            }
            let emb_rel_gpu = space_gpu.embed_relation(&subset_rel_gpu).unwrap();

            group.bench_with_input(
                BenchmarkId::new("metal", format!("{}x{}", subset_size, subset_size)),
                &(&space_gpu, &emb_rel_gpu),
                |b, (space, emb_rel)| {
                    b.iter(|| space.query_all_pairs(emb_rel, 0.0).unwrap());
                },
            );
        }
    }

    group.finish();
}

/// Benchmark: Relation composition (transitive closure)
fn bench_composition(c: &mut Criterion) {
    let data_path = "data/Release/train.txt";
    if !std::path::Path::new(data_path).exists() {
        return;
    }

    let (entities, relations, triples) = load_fb15k237(data_path);
    let num_entities = entities.len();
    let num_relations = relations.len();
    let sparse_relations = triples_to_relations(&triples, num_relations);

    let mut group = c.benchmark_group("fb15k237_composition");
    let dim = 128;

    // CPU
    let space_cpu = EmbeddingSpace::new(num_entities, dim, &Device::Cpu).unwrap();
    let emb_r0_cpu = space_cpu.embed_relation(&sparse_relations[0]).unwrap();
    let emb_r1_cpu = space_cpu.embed_relation(&sparse_relations[1]).unwrap();

    group.bench_with_input(
        BenchmarkId::new("cpu", "r0_compose_r1"),
        &(&space_cpu, &emb_r0_cpu, &emb_r1_cpu),
        |b, (space, r0, r1)| {
            b.iter(|| space.compose(r0, r1).unwrap());
        },
    );

    #[cfg(feature = "metal")]
    {
        if let Ok(metal_device) = Device::new_metal(0) {
            let space_gpu = EmbeddingSpace::new(num_entities, dim, &metal_device).unwrap();
            let emb_r0_gpu = space_gpu.embed_relation(&sparse_relations[0]).unwrap();
            let emb_r1_gpu = space_gpu.embed_relation(&sparse_relations[1]).unwrap();

            group.bench_with_input(
                BenchmarkId::new("metal", "r0_compose_r1"),
                &(&space_gpu, &emb_r0_gpu, &emb_r1_gpu),
                |b, (space, r0, r1)| {
                    b.iter(|| space.compose(r0, r1).unwrap());
                },
            );
        }
    }

    group.finish();
}

/// Run and print AUC evaluation
fn bench_auc_evaluation(c: &mut Criterion) {
    let data_path = "data/Release/train.txt";
    let test_path = "data/Release/test.txt";

    if !std::path::Path::new(data_path).exists() {
        return;
    }

    let (entities, relations, train_triples) = load_fb15k237(data_path);
    let num_entities = entities.len();
    let num_relations = relations.len();
    let sparse_relations = triples_to_relations(&train_triples, num_relations);

    // Load test triples (reusing same entity/relation mappings)
    let test_file = File::open(test_path).expect("Failed to open test file");
    let reader = BufReader::new(test_file);
    let mut test_triples = Vec::new();

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 3 {
            continue;
        }

        if let (Some(&head_id), Some(&rel_id), Some(&tail_id)) = (
            entities.get(parts[0]),
            relations.get(parts[1]),
            entities.get(parts[2]),
        ) {
            test_triples.push((head_id, rel_id, tail_id));
        }
    }

    println!("Test triples: {}", test_triples.len());

    let mut group = c.benchmark_group("fb15k237_auc");
    let dim = 128;
    let num_samples = 1000;

    // CPU
    let space_cpu = EmbeddingSpace::new(num_entities, dim, &Device::Cpu).unwrap();
    let emb_relations_cpu: Vec<Tensor> = sparse_relations
        .iter()
        .map(|r| space_cpu.embed_relation(r).unwrap())
        .collect();

    let start = Instant::now();
    let auc_cpu = compute_auc(&space_cpu, &emb_relations_cpu, &test_triples, num_entities, num_samples);
    let cpu_time = start.elapsed();
    println!("CPU AUC ({} samples): {:.4} in {:?}", num_samples, auc_cpu, cpu_time);

    group.bench_function(BenchmarkId::new("cpu", num_samples), |b| {
        b.iter(|| {
            compute_auc(&space_cpu, &emb_relations_cpu, &test_triples, num_entities, num_samples)
        });
    });

    #[cfg(feature = "metal")]
    {
        if let Ok(metal_device) = Device::new_metal(0) {
            let space_gpu = EmbeddingSpace::new(num_entities, dim, &metal_device).unwrap();
            let emb_relations_gpu: Vec<Tensor> = sparse_relations
                .iter()
                .map(|r| space_gpu.embed_relation(r).unwrap())
                .collect();

            let start = Instant::now();
            let auc_gpu = compute_auc(&space_gpu, &emb_relations_gpu, &test_triples, num_entities, num_samples);
            let gpu_time = start.elapsed();
            println!("Metal AUC ({} samples): {:.4} in {:?}", num_samples, auc_gpu, gpu_time);

            group.bench_function(BenchmarkId::new("metal", num_samples), |b| {
                b.iter(|| {
                    compute_auc(&space_gpu, &emb_relations_gpu, &test_triples, num_entities, num_samples)
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_embed_relations,
    bench_query_all,
    bench_composition,
    bench_auc_evaluation,
);
criterion_main!(benches);