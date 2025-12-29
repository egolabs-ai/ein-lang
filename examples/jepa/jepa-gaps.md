# Ein Primitives Gap Analysis for JEPA-SST

## Current Ein Primitives

### Functions (✅ Available)
| Function | Signature | Notes |
|----------|-----------|-------|
| `sigmoid(x)` | Tensor → Tensor | Element-wise |
| `relu(x)` | Tensor → Tensor | Element-wise |
| `gelu(x)` | Tensor → Tensor | Element-wise |
| `tanh(x)` | Tensor → Tensor | Element-wise |
| `softmax(x)` | Tensor → Tensor | **Last dim only** |
| `sqrt(x)` | Tensor → Tensor | Element-wise |
| `exp(x)` | Tensor → Tensor | Element-wise |
| `log(x)` | Tensor → Tensor | Element-wise |
| `lnorm(x)` | Tensor → Tensor | Layer normalization (last dim) |
| `cross_entropy(logits, targets)` | Tensor, Tensor → Scalar | Classification loss |
| `mse(pred, target)` | Tensor, Tensor → Scalar | Regression loss |
| `detach(x)` | Tensor → Tensor | Stop gradient flow (NEW) |
| `max(a, b)` | Tensor, Tensor → Tensor | Element-wise maximum (NEW) |
| `min(a, b)` | Tensor, Tensor → Tensor | Element-wise minimum (NEW) |
| `abs(x)` | Tensor → Tensor | Absolute value (NEW) |

### Operations (✅ Available)
| Operation | Example | Notes |
|-----------|---------|-------|
| Einsum contraction | `C[i,k] = A[i,j] B[j,k]` | Full einsum support |
| 3-tensor join | `D[x,r,z] = S[x,r1,y] S[y,r2,z] R[r,r1,r2]` | Core SST operation |
| Arithmetic | `+`, `-`, `*`, `/` | Element-wise with broadcasting |
| Scalar division | `X / 8.0` | For attention scaling |
| Array literals | `[[1,2],[3,4]]` | Tensor construction |
| Parameters | `@param W: Float[m, n]` | Learnable weights |

### Commands (✅ Available)
| Command | Notes |
|---------|-------|
| `:train Loss epochs=N lr=R optimizer=adamw` | Training loop |
| `:save`, `:load_checkpoint` | Model persistence |
| `:print Tensor` | Inspection |
| `:ema target source [tau=0.99]` | EMA update for target encoder (NEW) |

---

## Implemented Primitives for JEPA

### ✅ Critical (Now Available)

#### 1. `detach(tensor)` / `stop_gradient(tensor)` ✅ IMPLEMENTED
**Use:** Target encoder in JEPA must not receive gradients.

```ein
// Now works!
Z_t1 = detach(lnorm(Z_t1_raw))  // target embedding (no gradients)
```

#### 2. `max(A, B)` element-wise ✅ IMPLEMENTED
**Use:** State union in forward chaining.

```ein
// Now works!
State1 = max(State0, Derived1)
```

Also added: `min(A, B)`, `abs(A)` for element-wise operations.

---

### ✅ Important (Now Available)

#### 3. `:ema` command ✅ IMPLEMENTED
**Use:** Target encoder EMA update.

```bash
// After each training step:
:ema TgtEncW1 EncW1 tau=0.99
// Sets TgtEncW1 = tau * TgtEncW1 + (1-tau) * EncW1
```

---

### 🟡 Important (Still Needed)

#### 4. `softmax(tensor, dim)`
**Need:** Slot attention requires softmax over sequence dim, not last dim.

```ein
// Current: softmax always on last dim
Attn = softmax(Scores)  // applies to dim -1

// Need: specify dimension
Attn = softmax(Scores, dim=2)  // over sequence positions
```

**Implementation:** Add optional `dim` argument to softmax function.

#### 5. `var(tensor, dim)` and `cov(tensor)`
**Need:** VICReg-style collapse prevention.

```ein
// Variance loss: embeddings should have unit variance
VarLoss = relu(1.0 - sqrt(var(Z, dim=0)))

// Covariance loss: embedding dims should be decorrelated
CovLoss = cov(Z).off_diagonal().sqr().sum()
```

**Implementation:** Statistical functions with dimension specification.

#### 6. `concat(tensors, dim)`
**Need:** Building sequences, combining entity embeddings.

```ein
// Concatenate entity and relation info
Combined = concat([Entity, RelSummary], dim=2)
```

**Implementation:** Wrapper around Candle's `Tensor::cat`.

---

### 🟢 Nice to Have

| Primitive | Use Case |
|-----------|----------|
| `argmax(tensor, dim)` | Discrete entity selection |
| `gather(tensor, indices, dim)` | Advanced indexing |
| `scatter(tensor, indices, values, dim)` | Sparse updates |
| `clamp(tensor, min, max)` | Value bounding |
| `where(cond, a, b)` | Conditional selection |
| `sum(tensor, dim)` / `mean(tensor, dim)` | Explicit reduction with dim |

---

## Implementation Priority

### Phase 1: Basic JEPA (1-2 days)
1. `detach(tensor)` - trivial wrapper
2. `max(A, B)` - trivial wrapper

### Phase 2: Full JEPA (2-3 days)
3. `softmax(tensor, dim)` - add dim parameter
4. `:ema` command - new REPL command
5. `concat(tensors, dim)` - wrapper

### Phase 3: VICReg & Robustness (2-3 days)
6. `var(tensor, dim)` - statistical ops
7. `cov(tensor)` - covariance matrix
8. `clamp(tensor, min, max)` - bounding

---

## Workarounds for Now

### Target Encoder Without Detach
```ein
// Use completely separate param sets for target encoder
// Update externally via script after each epoch:
//   for each param pair (Enc*, Tgt*):
//     Tgt* = 0.99 * Tgt* + 0.01 * Enc*
```

### Max Without Native Support
```ein
// Soft OR for probabilities in [0,1]:
StateUnion = State0 + Derived - State0 * Derived

// For general tensors, use:
// max(a,b) = (a + b + |a - b|) / 2
// But |x| isn't available either...
```

### Softmax on Specific Dim
```ein
// Transpose, softmax (on last), transpose back
// Clunky but works for 3D tensors
ScoresT[b,s,e] = Scores[b,e,s]
AttnT = softmax(ScoresT)
Attn[b,e,s] = AttnT[b,s,e]
```

---

## Summary

**Can we implement JEPA in Ein today?** YES! Core JEPA loop is fully functional.

| Component | Status |
|-----------|--------|
| Slot attention encoder | ⚠️ Needs softmax(dim) workaround |
| Relation extraction | ✅ Works |
| Forward chaining | ✅ Works with `max(A, B)` |
| Predictor | ✅ Works |
| MSE loss | ✅ Works |
| Target encoder | ✅ Works with `detach()` and `:ema` |
| VICReg collapse prevention | ❌ No var/cov |

**Status Update:** Phase 1 is complete! `detach(tensor)`, `max(A, B)`, `min(A, B)`, `abs(A)`, and `:ema` command are all implemented. The basic JEPA example at `examples/jepa/jepa-basic.ein` now works.