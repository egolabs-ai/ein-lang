#!/usr/bin/env python3
"""
Kocoro-lab benchmark for comparison with Ein.

Run with:
    source .venv/bin/activate
    python3 benches/kocoro_comparison.py
"""

import time
import sys
import tracemalloc
import numpy as np

try:
    import torch
    from tensorlogic import EmbeddingSpace, TensorProgram
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install with: pip install git+https://github.com/Kocoro-lab/tensorlogic.git tqdm")
    sys.exit(1)


def create_random_relation(n: int, num_facts: int, seed: int) -> torch.Tensor:
    """Create random sparse relation tensor matching Ein's make_random_relation."""
    relation = torch.zeros(n, n)
    x = seed
    for _ in range(num_facts):
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        a = x % n
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        b = x % n
        relation[a, b] = 1.0
    return relation


def compose_relations(r0: torch.Tensor, r1: torch.Tensor) -> torch.Tensor:
    """Compute r0 o r1 (composition)."""
    composed = r0 @ r1
    return (composed > 0).float()


def tensor_to_pairs(tensor: torch.Tensor) -> set:
    """Convert adjacency tensor to set of pairs."""
    indices = (tensor > 0).nonzero()
    return set((int(i), int(j)) for i, j in indices)


def compute_all_scores(space, rel_matrix, n, temp):
    """Manually compute all pairwise scores."""
    emb = space.object_embeddings
    # Score(i,j) = sigmoid(emb[i] @ R @ emb[j] / temp)
    # Vectorized: scores = sigmoid(emb @ R @ emb.T / temp)
    intermediate = emb @ rel_matrix  # [n, dim]
    logits = intermediate @ emb.t()  # [n, n]
    scores = torch.sigmoid(logits / temp)
    return scores


def compute_auc(scores: np.ndarray, ground_truth: set, n: int) -> float:
    """Compute AUC: fraction of positive scores > negative scores."""
    positive_scores = []
    negative_scores = []

    for i in range(n):
        for j in range(n):
            score = float(scores[i, j])
            if (i, j) in ground_truth:
                positive_scores.append(score)
            else:
                negative_scores.append(score)

    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return 0.5

    # Sample to avoid O(n^4) comparison
    if len(positive_scores) * len(negative_scores) > 1000000:
        np.random.seed(42)
        pos_sample = np.random.choice(positive_scores, min(1000, len(positive_scores)), replace=False)
        neg_sample = np.random.choice(negative_scores, min(1000, len(negative_scores)), replace=False)
    else:
        pos_sample = np.array(positive_scores)
        neg_sample = np.array(negative_scores)

    correct = sum(1 for p in pos_sample for ng in neg_sample if p > ng)
    total = len(pos_sample) * len(neg_sample)

    return correct / total if total > 0 else 0.5


def benchmark_smallkg():
    """SmallKG benchmark matching kocoro-lab parameters."""
    print()
    print("=" * 60)
    print("KOCORO-LAB TENSORLOGIC BENCHMARK")
    print("=" * 60)

    # Parameters matching kocoro-lab
    n = 300
    dim = 32
    temp = 0.2
    lr = 0.01
    epochs = 100
    neg_ratio = 3

    print(f"\nParameters: N={n}, dim={dim}, temp={temp}, lr={lr}, epochs={epochs}")

    # Create relations with same seeds as Ein
    r0 = create_random_relation(n, 900, 42)
    r1 = create_random_relation(n, 900, 12345)
    target = compose_relations(r0, r1)
    target_pairs = tensor_to_pairs(target)
    r0_pairs = tensor_to_pairs(r0)
    r1_pairs = tensor_to_pairs(r1)

    print(f"\nDataset:")
    print(f"  R0: {int(r0.sum())} facts")
    print(f"  R1: {int(r1.sum())} facts")
    print(f"  Target (r0 o r1): {len(target_pairs)} facts")

    tracemalloc.start()
    device = torch.device("cpu")

    # ===== METHOD 1: BOOLEAN (TensorProgram) =====
    print("\n--- Method 1: Boolean (TensorProgram) ---")

    start = time.perf_counter()
    program = TensorProgram(n, device=device)
    program.add_tensor("r0", data=r0)
    program.add_tensor("r1", data=r1)
    program.add_equation("target", "r0 @ r1")
    result = program.query("target")
    scores_bool = (result > 0).float()
    bool_time = time.perf_counter() - start

    auc_bool = compute_auc(scores_bool.numpy(), target_pairs, n)
    print(f"  AUC: {auc_bool:.4f}")
    print(f"  Time: {bool_time*1000:.2f}ms")

    # ===== METHOD 2: EMBEDDING SPACE (Random) =====
    print("\n--- Method 2: Embedding Space (Random) ---")

    start = time.perf_counter()
    space = EmbeddingSpace(num_objects=n, embedding_dim=dim, device=device)
    space.add_relation("r0")
    space.add_relation("r1")

    r0_pairs_list = [(i, j) for i, j in r0_pairs]
    r1_pairs_list = [(i, j) for i, j in r1_pairs]

    space.embed_relation_from_facts("r0", r0_pairs_list)
    space.embed_relation_from_facts("r1", r1_pairs_list)

    # Compose in embedding space
    emb_r0 = space.relations["r0"]
    emb_r1 = space.relations["r1"]
    emb_target = emb_r0 @ emb_r1

    # Compute scores
    scores = compute_all_scores(space, emb_target, n, temp)
    embed_time = time.perf_counter() - start

    scores_np = scores.detach().numpy()
    auc_embed = compute_auc(scores_np, target_pairs, n)

    print(f"  AUC: {auc_embed:.4f}")
    print(f"  Time: {embed_time*1000:.2f}ms")

    # ===== METHOD 3: TRAINED EMBEDDINGS =====
    print("\n--- Method 3: Trained Embeddings ---")

    start = time.perf_counter()
    space_trained = EmbeddingSpace(num_objects=n, embedding_dim=dim, device=device)
    space_trained.add_relation("r0")
    space_trained.add_relation("r1")

    optimizer = torch.optim.AdamW(space_trained.parameters(), lr=lr, weight_decay=0.01)

    # Train on r0
    for epoch in range(epochs):
        space_trained.embed_relation_from_facts("r0", r0_pairs_list)
        emb = space_trained.object_embeddings
        emb_r0 = space_trained.relations["r0"]

        pos_batch = r0_pairs_list[:min(64, len(r0_pairs_list))]
        neg_batch = []
        while len(neg_batch) < len(pos_batch) * neg_ratio:
            i, j = np.random.randint(0, n, 2)
            if (i, j) not in r0_pairs:
                neg_batch.append((i, j))

        pos_scores = torch.stack([
            torch.sigmoid(torch.sum(emb[i] * (emb_r0 @ emb[j])) / temp)
            for i, j in pos_batch
        ])
        neg_scores = torch.stack([
            torch.sigmoid(torch.sum(emb[i] * (emb_r0 @ emb[j])) / temp)
            for i, j in neg_batch
        ])

        pos_loss = -torch.log(pos_scores.clamp(1e-7, 1-1e-7)).mean()
        neg_loss = -torch.log(1 - neg_scores.clamp(1e-7, 1-1e-7)).mean()
        loss = (pos_loss + neg_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Train on r1
    for epoch in range(epochs):
        space_trained.embed_relation_from_facts("r1", r1_pairs_list)
        emb = space_trained.object_embeddings
        emb_r1 = space_trained.relations["r1"]

        pos_batch = r1_pairs_list[:min(64, len(r1_pairs_list))]
        neg_batch = []
        while len(neg_batch) < len(pos_batch) * neg_ratio:
            i, j = np.random.randint(0, n, 2)
            if (i, j) not in r1_pairs:
                neg_batch.append((i, j))

        pos_scores = torch.stack([
            torch.sigmoid(torch.sum(emb[i] * (emb_r1 @ emb[j])) / temp)
            for i, j in pos_batch
        ])
        neg_scores = torch.stack([
            torch.sigmoid(torch.sum(emb[i] * (emb_r1 @ emb[j])) / temp)
            for i, j in neg_batch
        ])

        pos_loss = -torch.log(pos_scores.clamp(1e-7, 1-1e-7)).mean()
        neg_loss = -torch.log(1 - neg_scores.clamp(1e-7, 1-1e-7)).mean()
        loss = (pos_loss + neg_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_time = time.perf_counter() - start

    # Query after training
    start = time.perf_counter()
    space_trained.embed_relation_from_facts("r0", r0_pairs_list)
    space_trained.embed_relation_from_facts("r1", r1_pairs_list)
    emb_r0 = space_trained.relations["r0"]
    emb_r1 = space_trained.relations["r1"]
    emb_target = emb_r0 @ emb_r1
    scores_trained = compute_all_scores(space_trained, emb_target, n, temp)
    query_time = time.perf_counter() - start

    scores_trained_np = scores_trained.detach().numpy()
    auc_trained = compute_auc(scores_trained_np, target_pairs, n)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  AUC: {auc_trained:.4f}")
    print(f"  Training time: {train_time*1000:.2f}ms")
    print(f"  Query time: {query_time*1000:.2f}ms")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f}MB")

    # Summary
    print()
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"  Boolean AUC:     {auc_bool:.4f}")
    print(f"  Embedding AUC:   {auc_embed:.4f}")
    print(f"  Trained AUC:     {auc_trained:.4f}")
    improvement = (auc_trained - 0.5) / max(auc_embed - 0.5, 0.001)
    print(f"  Improvement:     {improvement:.2f}x over random baseline")
    print(f"  Total time:      {(train_time + query_time)*1000:.2f}ms")
    print(f"  Memory:          {peak / 1024 / 1024:.2f}MB")

    print()
    print("-" * 60)
    print("MACHINE-READABLE OUTPUT")
    print("-" * 60)
    print(f"KOCORO_AUC_BOOL={auc_bool:.4f}")
    print(f"KOCORO_AUC_RANDOM={auc_embed:.4f}")
    print(f"KOCORO_AUC_TRAINED={auc_trained:.4f}")
    print(f"KOCORO_TRAIN_TIME_MS={train_time*1000:.2f}")
    print(f"KOCORO_QUERY_TIME_MS={query_time*1000:.2f}")
    print(f"KOCORO_MEMORY_MB={peak / 1024 / 1024:.2f}")


if __name__ == "__main__":
    benchmark_smallkg()