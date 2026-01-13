# Ein Implementation Progress

## Current Status

**Phase:** Phase 2 (Selection & Comparison) - Complete
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

### Phase 3: Control Flow & Iteration

| Feature | Status | Notes |
|---------|--------|-------|
| `:iterate N` | Not started | Loop N times in REPL |
| `:until_stable` | Not started | Iterate until convergence |
| `clamp(x, min, max)` | Not started | Clamp values to range |

---

## Completed Work

### 2026-01-12: Phase 2 - Selection & Comparison

**Commit:** (pending)

**Features Added:**
- `argmax(x, dim)` - Index of maximum value along dimension
- `argmin(x, dim)` - Index of minimum value along dimension
- `gt(a, b)` - Element-wise greater than (returns 0/1)
- `lt(a, b)` - Element-wise less than
- `eq(a, b)` - Element-wise equality
- `ge(a, b)` - Element-wise greater or equal
- `le(a, b)` - Element-wise less or equal
- `where(cond, a, b)` - Conditional selection

**Files Modified:**
- `src/syntax/token.rs` - Added Argmax, Argmin, Gt, Lt, EqCmp, Ge, Le, Where tokens
- `src/syntax/parser.rs` - Added to function recognition
- `src/runtime/context.rs` - Implemented functions + 11 new unit tests

**Test Results:**
```
cargo test --lib runtime::context::tests
running 35 tests
test result: ok. 35 passed; 0 failed;
```

**Use Cases Enabled:**
- Finding witnesses: `ClosestNode = argmax(Proximity, 0)`
- Thresholding: `Active = gt(Score, 0.5)`
- Conditional logic: `Cost = where(EdgeExists, Weight, infinity)`

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

**REPL Verification:**
```
Triangle counting:   trace(A³)/6 = 1 ✅
Degree computation:  sum(A, dim) ✅
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
Finding witnesses:   ✅ Works (argmax(x, dim))
Conditional paths:   ✅ Works (where(cond, a, b))
Auto fixed-point:    ❌ Needs :iterate command
```

---

## Upcoming

### Phase 3: Control Flow
- `:iterate N` command - Run statements N times
- `:until_stable` command - Iterate until convergence
- `clamp(x, min, max)` - Clamp values to range

---

## Commit Log

| Date | Commit | Description |
|------|--------|-------------|
| 2026-01-12 | (pending) | Phase 2: argmax, argmin, comparison ops, where |
| 2026-01-12 | (pending) | Phase 1: trace, diag, sum(dim), mean(dim) |
| 2026-01-12 | (pending) | Foundation: detach, max, min, abs, :ema |

---

## Notes

- `lnorm(x)` is the layer normalization function (not `layernorm`)
- Ein tensor ordering: `Input[batch, feature] Weight[output, feature]`
- Triangle counting: `trace(A³)/6` where A is adjacency matrix
- Degree computation: `sum(A, 1)` for out-degree, `sum(A, 0)` for in-degree
- Finding witnesses: `argmax(Proximity, 0)` returns index of closest node
- Comparison ops return 0.0/1.0 tensors (F32 for consistency)