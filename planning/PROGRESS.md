# Ein Implementation Progress

## Current Status

**Phase:** Phase 1 (Counting & Reductions) - Complete
**Last Updated:** 2026-01-12
**Target:** `experiments/logic/extreme_logic_tests.ein`

---

## Target File

```
experiments/logic/extreme_logic_tests.ein
```

Tests tensor logic on hard graph reasoning problems:
- Triangle counting (AML / fraud detection)
- Transitive closure / reachability
- Set intersection / difference
- Constrained reachability (GDPR / sanctions)
- Strongly connected components
- Clique detection

---

## In Progress

### Phase 2: Selection & Comparison

| Feature | Status | Notes |
|---------|--------|-------|
| `argmax(x, dim)` | Not started | Find index of maximum |
| `argmin(x, dim)` | Not started | Find index of minimum |
| `gt(a, b)` | Not started | Greater than comparison |
| `lt(a, b)` | Not started | Less than comparison |
| `eq(a, b)` | Not started | Equality comparison |
| `ge(a, b)` | Not started | Greater or equal |
| `le(a, b)` | Not started | Less or equal |
| `where(cond, a, b)` | Not started | Conditional selection |

---

## Completed Work

### 2026-01-12: Phase 1 - Counting & Reductions

**Commit:** (pending)

**Features Added:**
- `trace(x)` - Sum of diagonal elements (enables triangle counting)
- `diag(x)` - Extract diagonal from matrix
- `sum(x, dim)` - Sum with optional dimension parameter
- `mean(x, dim)` - Mean with optional dimension parameter

**Files Modified:**
- `src/syntax/token.rs` - Added Trace, Diag tokens
- `src/syntax/parser.rs` - Added to function recognition
- `src/runtime/context.rs` - Implemented functions + unit tests

**Test Results:**
```
cargo test --lib runtime::context::tests
running 9 tests
test runtime::context::tests::test_abs ... ok
test runtime::context::tests::test_diag ... ok
test runtime::context::tests::test_max_elementwise ... ok
test runtime::context::tests::test_min_elementwise ... ok
test runtime::context::tests::test_mean_dim ... ok
test runtime::context::tests::test_mean_full ... ok
test runtime::context::tests::test_sum_dim ... ok
test runtime::context::tests::test_sum_full ... ok
test runtime::context::tests::test_trace ... ok
test result: ok. 9 passed; 0 failed;
```

**REPL Verification:**
```
// Triangle counting test - 4-node graph with 1 triangle
A = [[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]]

trace(A) = 0           // correct (no self-loops)
diag(A) = [0,0,0,0]    // correct
sum(A, 1) = [2,2,3,1]  // out-degrees correct
sum(A, 0) = [2,2,3,1]  // in-degrees correct
sum(A) = 8             // total edges correct
trace(A³) = 6          // triangle count × 6 = correct!
```

### 2026-01-12: Foundation Features

**Commit:** (pending)

**Features Added:**
- `detach(tensor)` - Stop gradient flow
- `max(a, b)` - Element-wise maximum (state union)
- `min(a, b)` - Element-wise minimum
- `abs(x)` - Absolute value
- `:ema` command - EMA parameter updates

**Files Modified:**
- `src/syntax/token.rs` - Added Detach, Max, Min, Abs tokens
- `src/syntax/parser.rs` - Added to function recognition
- `src/runtime/context.rs` - Implemented functions
- `src/main.rs` - Added :ema command

---

## Test Results

### Extreme Logic Tests Status
```
Transitive closure:  ✅ Works (manual iteration)
Set operations:      ✅ Works
SCC membership:      ✅ Works
Clique existence:    ✅ Works
Triangle counting:   ✅ Works (trace(A³)/6)
Degree computation:  ✅ Works (sum(A, dim))
Finding witnesses:   ❌ Needs argmax(x, dim)
```

---

## Upcoming

### Phase 2: Selection & Comparison
- `argmax(x, dim)`, `argmin(x, dim)`
- Comparison ops: `gt`, `lt`, `eq`, `ge`, `le`
- `where(cond, a, b)`

### Phase 3: Control Flow
- `:iterate N` command
- `:until_stable` command
- `clamp(x, min, max)`

---

## Commit Log

| Date | Commit | Description |
|------|--------|-------------|
| 2026-01-12 | (pending) | Phase 1: trace, diag, sum(dim), mean(dim) |
| 2026-01-12 | (pending) | Foundation: detach, max, min, abs, :ema |

---

## Notes

- `lnorm(x)` is the layer normalization function (not `layernorm`)
- Ein tensor ordering: `Input[batch, feature] Weight[output, feature]`
- Triangle counting: `trace(A³)/6` where A is adjacency matrix
- Degree computation: `sum(A, 1)` for out-degree, `sum(A, 0)` for in-degree