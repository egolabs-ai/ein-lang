//! Ein benchmark for comparison with kocoro-lab.
//!
//! Run with: cargo run --release --bin ein_comparison
//!
//! Uses same parameters as kocoro-lab benchmark_suite.py:
//! - N=300, dim=32, temp=0.2
//! - Same random seeds for reproducibility
//! - Same training epochs and learning rate

use candle_core::Device;
use ein::{SparseBool, TrainableEmbeddingSpace};
use std::time::Instant;

/// Create random sparse relation with same LCG as kocoro comparison.
fn create_random_relation(n: usize, num_facts: usize, seed: usize) -> SparseBool {
    let mut rel = SparseBool::new(2);
    let mut x = seed;
    for _ in 0..num_facts {
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let a = x % n;
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let b = x % n;
        rel.insert(vec![a, b]);
    }
    rel
}

/// Compute AUC for embedded relation vs ground truth.
fn compute_auc(
    space: &TrainableEmbeddingSpace,
    emb_rel: &candle_core::Tensor,
    ground_truth: &SparseBool,
    n: usize,
) -> f64 {
    let scores = space.query_all_pairs(emb_rel).unwrap();

    let mut positive_scores = Vec::new();
    let mut negative_scores = Vec::new();

    for i in 0..n {
        for j in 0..n {
            let score: f32 = scores.get(i).unwrap().get(j).unwrap().to_scalar().unwrap();
            if ground_truth.contains(&[i, j]) {
                positive_scores.push(score);
            } else {
                negative_scores.push(score);
            }
        }
    }

    // Sample if too many comparisons
    let (pos_sample, neg_sample) = if positive_scores.len() * negative_scores.len() > 1_000_000 {
        let pos: Vec<_> = positive_scores.iter().take(1000).cloned().collect();
        let neg: Vec<_> = negative_scores.iter().take(1000).cloned().collect();
        (pos, neg)
    } else {
        (positive_scores, negative_scores)
    };

    let mut correct = 0usize;
    let total = pos_sample.len() * neg_sample.len();
    for &pos in &pos_sample {
        for &neg in &neg_sample {
            if pos > neg {
                correct += 1;
            }
        }
    }

    if total > 0 {
        correct as f64 / total as f64
    } else {
        0.5
    }
}

fn main() {
    println!();
    println!("{}", "=".repeat(60));
    println!("EIN TENSORLOGIC BENCHMARK");
    println!("{}", "=".repeat(60));

    // Parameters matching kocoro-lab exactly
    let n = 300;
    let dim = 32;
    let temp = 0.2;
    let lr = 0.01;
    let epochs = 100;
    let neg_ratio = 3;

    println!("\nParameters: N={}, dim={}, temp={}, lr={}, epochs={}", n, dim, temp, lr, epochs);

    // Create relations with same seeds as Python benchmark
    let r0 = create_random_relation(n, 900, 42);
    let r1 = create_random_relation(n, 900, 12345);

    // Compute ground truth composition
    let joined = r0.join(&r1, 1, 0);
    let target = joined.project(&[0, 2]);

    println!("\nDataset:");
    println!("  R0: {} facts", r0.len());
    println!("  R1: {} facts", r1.len());
    println!("  Target (r0 o r1): {} facts", target.len());

    let device = Device::Cpu;

    // Track memory (approximate - Rust doesn't have easy memory tracking)
    let memory_before = get_memory_usage();

    // ===== WITHOUT TRAINING =====
    println!("\n--- Without Training (Random Embeddings) ---");

    let space_random = TrainableEmbeddingSpace::new(n, dim, temp, &device).unwrap();

    let start = Instant::now();
    let emb_r0 = space_random.embed_relation(&r0).unwrap();
    let emb_r1 = space_random.embed_relation(&r1).unwrap();
    let emb_target = space_random.compose(&emb_r0, &emb_r1).unwrap();
    let _ = space_random.query_all_pairs(&emb_target).unwrap();
    let embed_time = start.elapsed();

    let auc_random = compute_auc(&space_random, &emb_target, &target, n);

    println!("  AUC: {:.4}", auc_random);
    println!("  Time: {:.2}ms", embed_time.as_secs_f64() * 1000.0);

    // ===== WITH TRAINING =====
    println!("\n--- With Training ---");

    let mut space_trained = TrainableEmbeddingSpace::new(n, dim, temp, &device).unwrap();

    let start = Instant::now();

    // Train on r0
    space_trained.train(&r0, epochs, lr, neg_ratio).unwrap();

    // Train on r1
    space_trained.train(&r1, epochs, lr, neg_ratio).unwrap();

    let train_time = start.elapsed();

    // Query after training
    let start = Instant::now();
    let emb_r0 = space_trained.embed_relation(&r0).unwrap();
    let emb_r1 = space_trained.embed_relation(&r1).unwrap();
    let emb_target = space_trained.compose(&emb_r0, &emb_r1).unwrap();
    let _ = space_trained.query_all_pairs(&emb_target).unwrap();
    let query_time = start.elapsed();

    let auc_trained = compute_auc(&space_trained, &emb_target, &target, n);

    let memory_after = get_memory_usage();
    let memory_used = memory_after.saturating_sub(memory_before);

    println!("  AUC: {:.4}", auc_trained);
    println!("  Training time: {:.2}ms", train_time.as_secs_f64() * 1000.0);
    println!("  Query time: {:.2}ms", query_time.as_secs_f64() * 1000.0);
    println!("  Memory: ~{:.2}MB", memory_used as f64 / 1024.0 / 1024.0);

    // Summary
    println!("\n{}", "-".repeat(60));
    println!("SUMMARY");
    println!("{}", "-".repeat(60));
    println!("  Random AUC:  {:.4}", auc_random);
    println!("  Trained AUC: {:.4}", auc_trained);
    let improvement = (auc_trained - 0.5) / (auc_random - 0.5).max(0.001);
    println!("  Improvement: {:.2}x over random baseline", improvement);
    println!("  Total time:  {:.2}ms", (train_time + query_time).as_secs_f64() * 1000.0);
    println!("  Memory:      ~{:.2}MB", memory_used as f64 / 1024.0 / 1024.0);

    // Output in machine-readable format for comparison script
    println!("\n{}", "-".repeat(60));
    println!("MACHINE-READABLE OUTPUT");
    println!("{}", "-".repeat(60));
    println!("EIN_AUC_RANDOM={:.4}", auc_random);
    println!("EIN_AUC_TRAINED={:.4}", auc_trained);
    println!("EIN_TRAIN_TIME_MS={:.2}", train_time.as_secs_f64() * 1000.0);
    println!("EIN_QUERY_TIME_MS={:.2}", query_time.as_secs_f64() * 1000.0);
    println!("EIN_MEMORY_MB={:.2}", memory_used as f64 / 1024.0 / 1024.0);
}

/// Get current memory usage (platform-specific, approximate).
fn get_memory_usage() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output();

        if let Ok(output) = output {
            if let Ok(s) = String::from_utf8(output.stdout) {
                if let Ok(kb) = s.trim().parse::<usize>() {
                    return kb * 1024; // Convert KB to bytes
                }
            }
        }
        0
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<_> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
        0
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}