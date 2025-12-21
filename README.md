# Ein

**A tensor logic language unifying neural and symbolic AI**

Ein combines Einstein notation tensor operations with Datalog-style logic programming in a single, elegant syntax. Write neural networks and symbolic reasoning in the same language.

## Quick Start

```bash
# Build
cargo build --release

# Or with GPU acceleration (Apple Silicon)
cargo build --release --features metal

# Run the REPL
./target/release/ein
```

```
Ein v0.1.0 - A tensor logic language
>>> X = [[1,2],[3,4]]
>>> Y[i,j] = X[j,i]
>>> :print Y
Y : [2, 2] =
  [  1.0000,   3.0000]
  [  2.0000,   4.0000]
```

## Features

### Einstein Notation for Everything

```ein
// Matrix multiplication - repeated indices are summed
C[i,k] = A[i,j] B[j,k]

// Batched multi-head attention in one line
Attn[b,h,i,j] = Q[b,h,i,d] K[b,h,j,d] / sqrt(64)

// Soft relation composition (differentiable Datalog!)
Derived[x,r,z] = State[x,r1,y] State[y,r2,z] Rules[r,r1,r2]
```

### Learnable Parameters & Training

```ein
@param W1: Float[128, 64]
@param W2: Float[10, 128]

H[i] = relu(W1[i,j] X[j])
Y[i] = softmax(W2[i,j] H[j])
Loss = cross_entropy(Y, Target)

:train Loss epochs=100 lr=0.001 optimizer=adamw
```

### Logic Programming (Datalog)

```ein
// Facts
Parent(alice, bob).
Parent(bob, charlie).

// Rules with recursion
Ancestor(x,y) <- Parent(x,y)
Ancestor(x,z) <- Ancestor(x,y) Parent(y,z)

// Queries
Ancestor(alice, x)?
```

### Language Modeling

```ein
// Load Shakespeare, train a GPT
:load_text data/tiny_shakespeare.txt seq_len=256
:batch batch_size=64
:load examples/gpt_nano.ein
:train Loss epochs=1000 lr=0.001 optimizer=adamw
:generate ROMEO: length=200 temperature=0.7
```

## Installation

Requires Rust 1.75+.

```bash
git clone https://github.com/yourusername/ein-lang.git
cd ein-lang
cargo build --release
```

### GPU Acceleration

```bash
# Apple Silicon (Metal)
cargo build --release --features metal

# NVIDIA (CUDA)
cargo build --release --features cuda
```

## Examples

See the `examples/` directory:

- `attention.ein` - Multi-head causal attention
- `gpt_nano.ein` - Full 6-layer GPT (10.7M params)
- `soft_relations.ein` - Differentiable relation composition
- `soft_rules.ein` - Learnable rule weights via 3-tensor contraction
- `kb_benchmark.ein` - Knowledge graph relation derivation
- `mlp.ein` - Simple MLP training

### Run an Example

```bash
./target/release/ein
:load examples/soft_relations.ein
:print Grandparent
```

## REPL Commands

| Command | Description |
|---------|-------------|
| `:print <tensor>` | Print tensor values |
| `:train <loss> epochs=N lr=R` | Train parameters |
| `:load <file.ein>` | Load an Ein program |
| `:save <file.safetensors>` | Save model checkpoint |
| `:load_checkpoint <file>` | Load model checkpoint |
| `:generate <seed> length=N` | Generate text |
| `:load_kb <train.txt>` | Load knowledge graph triples |
| `:train_kb epochs=N lr=R` | Train KG embeddings (DistMult) |
| `:eval_kb samples=N` | Evaluate link prediction (MRR, Hits@K) |
| `:tensors` | List all tensors |
| `:quit` | Exit |

## Theoretical Foundation

Ein is an implementation of **Tensor Logic** as described in:

> Pedro Domingos. *"Tensor Logic: The Language of AI"*
> University of Washington, 2025.
> [arXiv:2510.12269](https://arxiv.org/abs/2510.12269)

The central insight is that Datalog rules and Einstein summation are fundamentally the same operation. A logical rule:

```
Grandparent(x,z) <- Parent(x,y), Parent(y,z)
```

corresponds to the tensor contraction:

```ein
Grandparent[x,z] = Parent[x,y] Parent[y,z]
```

This equivalence enables:
- **Differentiable inference** — rule weights can be learned via gradient descent
- **Unified representation** — the same einsum notation expresses attention mechanisms, MLPs, and logical composition
- **N-tensor joins** — multi-relation reasoning in a single operation (e.g., `D[x,r,z] = S[x,r1,y] S[y,r2,z] R[r,r1,r2]`)

## Benchmarks

### Knowledge Graph Relation Composition

Deriving transitive relations from a Countries knowledge graph:

```bash
./target/release/ein
:load examples/kb_benchmark.ein
:print SameRegion
:print SharesLanguage
```

**SameRegion** shows block-diagonal structure (countries in same region connected):
- Europe cluster: france, germany, uk
- North America: usa, canada
- Asia: china, japan, india

**SharesLanguage** shows English-speaking countries connected (uk, usa, canada, india).

This demonstrates relation composition as matrix multiplication — the core insight from Tensor Logic.

### FB15k-237 Knowledge Graph Embedding

Training knowledge graph embeddings (DistMult model with margin loss) on FB15k-237:

```bash
./target/release/ein
:load_kb data/fb15k237/train.txt test=data/fb15k237/test.txt dim=128
:train_kb epochs=100 lr=0.001 batch_size=256 neg_ratio=5
```

**Training progress (10k triple subset, dim=64):**

| Epoch | Loss |
|-------|------|
| 1 | 0.984 |
| 5 | 0.003 |
| 10 | 0.001 |

The `:train_kb` command trains entity and relation embeddings using margin-based loss with negative sampling. After training, use `:eval_kb` to compute filtered MRR, Hits@1, Hits@3, Hits@10 on the test set.

### Language Modeling (GPT on Shakespeare)

Training a 6-layer, 10.7M parameter GPT on Tiny Shakespeare, matching nanoGPT architecture:

| Config | Value |
|--------|-------|
| Layers | 6 |
| Heads | 6 |
| d_model | 384 |
| Context | 256 tokens |
| Parameters | ~10.7M |

**Training progress:**

| Iterations | Loss |
|------------|------|
| 500 | 2.26 |
| 1000 | 2.01 |
| 1500 | 1.75 |
| 2000 | 1.57 |

nanoGPT reference achieves ~1.47 at 5000 iterations for readable Shakespeare. At 1.57, output shows structure (character names, formatting) but is not yet fluent.

**Train yourself:**
```bash
# Build with Metal support
cargo build --release --features metal

./target/release/ein
:load_text data/tiny_shakespeare.txt seq_len=256
:batch batch_size=64
:load examples/gpt_nano.ein
:train Loss epochs=500 lr=0.001 optimizer=adamw
:save checkpoint.safetensors
:generate ROMEO: length=200 temperature=0.7
```

### Running Benchmarks

```bash
# FB15k-237 knowledge graph benchmark (CPU vs GPU)
cargo bench --bench fb15k237_bench --features metal

# Tensor operations benchmark
cargo bench --bench tensor_ops

# Forward chaining benchmark
cargo bench --bench forward_chain

# Embedding training benchmark
cargo bench --bench embedding_bench
```

## Built With

- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and [tensorlogic](https://github.com/Kocoro-lab/tensorlogic)

## License

MIT

## Contributing

Issues and PRs welcome! Please open an issue to discuss larger changes before submitting.