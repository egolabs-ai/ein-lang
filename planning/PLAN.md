# Ein Feature Implementation Plan

## Overview

This document tracks planned features for Ein, prioritized by use case requirements from the Extreme Logic Tests (`experiments/logic/extreme_logic_tests.ein`) - graph reasoning problems including:
- Triangle counting (fraud detection)
- Transitive closure / reachability
- Set intersection / difference
- Constrained reachability (GDPR/sanctions filtering)
- Strongly connected components
- Clique detection

---

## Current Feature Status

| Feature | Status | Impact | Notes |
|---------|--------|--------|-------|
| Matrix multiply / einsum | ✅ Available | Paths, reachability | Core tensor operations |
| `max(a, b)` | ✅ Available | State union | Implemented |
| `min(a, b)` | ✅ Available | Set operations | Implemented |
| `abs(x)` | ✅ Available | Distance metrics | Implemented |
| Element-wise ops | ✅ Available | Set intersection/diff | `+`, `-`, `*`, `/` |
| `sigmoid(x)` | ✅ Available | Soft logic | Probabilistic reasoning |
| `sum(tensor)` | ⚠️ Partial | Counting | Only full reduction, need `sum(x, dim)` |
| `mean(tensor)` | ⚠️ Partial | Averaging | Only full reduction, need `mean(x, dim)` |
| Iteration | ⚠️ Manual | Fixed-point | Requires manual unrolling |
| `trace(tensor)` | ❌ Missing | Triangle counting | `trace(A³)/6` |
| Comparison ops | ❌ Missing | Constraints | `>`, `<`, `==`, `>=`, `<=` |
| `argmax(x, dim)` | ❌ Missing | Finding witnesses | Which node? |
| `where(cond, a, b)` | ❌ Missing | Conditional logic | If-then-else |
| Negation as failure | ❌ Missing | Closed-world | "NOT reachable" |

---

## What Works Today

### 1. Triangle Detection (Existence) ✅
```ein
// Does any triangle exist?
T[i,j,k] = A[i,j] A[j,k] A[k,i]
// max over all T gives 1 if any triangle exists
```

### 2. Transitive Closure ✅ (with manual unrolling)
```ein
Reaches0[i,j] = A[i,j]
R1[i,j] = Reaches0[i,k] Reaches0[k,j]
Reaches1[i,j] = max(Reaches0[i,j], R1[i,j])
// ... continue for graph diameter iterations
```

### 3. Set Intersection / Difference ✅
```ein
// Reachable from BOTH 0 and 6:
Both[j] = ReachFrom0[j] * ReachFrom6[j]

// Reachable from 0 but NOT 6:
Only0[j] = ReachFrom0[j] * (1.0 - ReachFrom6[j])
```

### 4. Constrained Reachability ✅
```ein
// Red-only reachability
RedReaches0[i,j] = RedEdges[i,j]
RedR1[i,j] = RedReaches0[i,k] RedEdges[k,j]
RedReaches[i,j] = max(RedReaches0[i,j], RedR1[i,j])
```

### 5. SCC Membership Check ✅
```ein
// Same SCC iff mutually reachable
SameSCC[i,j] = Reaches[i,j] * Reaches[j,i]
```

### 6. Clique Existence ✅
```ein
// 4-clique indicator (all 6 edges must exist)
Clique4[i,j,k,l] = A[i,j] A[j,k] A[k,l] A[i,k] A[i,l] A[j,l]
```

---

## What Doesn't Work (Yet)

### 1. Triangle Counting - needs `sum(x, dim)` or `trace(x)`
```ein
// CURRENT: Can detect existence, can't count
T[i,j,k] = A[i,j] A[j,k] A[k,i]  // 1 for each triangle triple

// NEED:
TriangleCount = sum(T) / 6  // sum over all dimensions
// OR:
A2[i,j] = A[i,k] A[k,j]
A3[i,j] = A2[i,k] A[k,j]
TriangleCount = trace(A3) / 6
```

### 2. Degree Computation - needs `sum(x, dim)`
```ein
// NEED:
OutDegree[i] = sum(A, dim=1)[i]   // sum over columns
InDegree[j] = sum(A, dim=0)[j]    // sum over rows
```

### 3. Negative Reachability - needs closed-world negation
```ein
// "Nodes NOT reachable from 0"
// CURRENT WORKAROUND (works for {0,1} matrices):
NotReachable[j] = 1.0 - Reachable[j]

// TRUE NEGATION would need:
NotReachable[j] <- NOT Reachable[j]
```

### 4. Finding Witnesses - needs `argmax(x, dim)`
```ein
// "Which node is closest to node 0?"
// NEED:
ClosestNode = argmax(Proximity[0, :])
```

### 5. Conditional Selection - needs `where(cond, a, b)`
```ein
// "Use cost A if edge exists, infinity otherwise"
// NEED:
EffectiveCost[i,j] = where(A[i,j] > 0, Cost[i,j], infinity)
```

### 6. Fixed-Point Iteration - needs iteration construct
```ein
// CURRENT: Manual unrolling
Reaches0 = A
Reaches1 = max(Reaches0, Reaches0 @ Reaches0)
Reaches2 = max(Reaches1, Reaches1 @ Reaches1)
// ... hardcoded for expected diameter

// IDEAL: Automatic iteration until convergence
Reaches = fixpoint(A, λR. max(R, R @ R))
```

---

## Implementation Phases

### Phase 1: Counting (Critical for Logic Tests)

| Primitive | Implementation | Difficulty |
|-----------|----------------|------------|
| `sum(x, dim)` | Candle's `sum_keepdim(dim)` | Easy |
| `mean(x, dim)` | Candle's `mean_keepdim(dim)` | Easy |
| `trace(x)` | Diagonal extraction + sum | Easy |
| `diag(x)` | Extract/create diagonal | Easy |

**Files to modify:**
```
src/syntax/token.rs     # Add tokens: Trace, Diag
src/syntax/parser.rs    # Add to is_function_token(), function_name()
src/runtime/context.rs  # Implement in apply_function()
```

**Note:** `sum` and `mean` already exist but only do full reduction. Need to add dimension parameter support.

### Phase 2: Selection & Comparison

| Primitive | Implementation | Difficulty |
|-----------|----------------|------------|
| `argmax(x, dim)` | Candle's `argmax(dim)` | Easy |
| `argmin(x, dim)` | Candle's `argmin(dim)` | Easy |
| `gt(a, b)` | `a.gt(b)` | Easy |
| `lt(a, b)` | `a.lt(b)` | Easy |
| `eq(a, b)` | `a.eq(b)` | Easy |
| `ge(a, b)` | `a.ge(b)` | Easy |
| `le(a, b)` | `a.le(b)` | Easy |
| `where(cond, a, b)` | Candle's `where_cond` | Medium |

### Phase 3: Control Flow & Iteration

| Primitive | Type | Implementation | Difficulty |
|---------|------|----------------|------------|
| `:iterate N` | REPL command | Loop in main.rs | Medium |
| `:until_stable` | REPL command | Diff-based stopping | Medium |
| `clamp(x, min, max)` | Function | `Tensor::clamp` | Easy |

---

## Test Coverage Projection

| Test | Current | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|---------|
| Triangle existence | ✅ | ✅ | ✅ | ✅ |
| Triangle counting | ❌ | ✅ | ✅ | ✅ |
| Transitive closure | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual | ✅ Auto |
| Set operations | ✅ | ✅ | ✅ | ✅ |
| Constrained reach | ✅ | ✅ | ✅ | ✅ |
| SCC detection | ✅ | ✅ | ✅ | ✅ |
| Degree computation | ❌ | ✅ | ✅ | ✅ |
| Find closest node | ❌ | ❌ | ✅ | ✅ |
| Conditional paths | ❌ | ❌ | ✅ | ✅ |
| Auto fixed-point | ❌ | ❌ | ❌ | ✅ |

---

## Summary

**Can we run extreme_logic_tests.ein today?** Partially.

| Capability | Status |
|------------|--------|
| Graph reachability | ✅ Works (manual iteration) |
| Set operations | ✅ Works |
| SCC membership | ✅ Works |
| Clique existence | ✅ Works |
| Triangle counting | ❌ Needs `sum(x, dim)` or `trace(x)` |
| Degree computation | ❌ Needs `sum(x, dim)` |
| Finding witnesses | ❌ Needs `argmax(x, dim)` |
| Automatic iteration | ❌ Needs `:iterate` command |

**Next Step:** Implement Phase 1 - `sum(x, dim)` and `trace(x)` to unlock counting-based logic tests.