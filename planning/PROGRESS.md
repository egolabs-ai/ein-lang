# Ein Implementation Progress

## Current Status

**Phase:** Phase 1 (Counting & Reductions) - Not Started
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

### Phase 1: Counting & Reductions

| Feature | Status | Notes |
|---------|--------|-------|
| `sum(x, dim)` | Not started | Enables degree computation |
| `mean(x, dim)` | Not started | Enables averaging over axis |
| `trace(x)` | Not started | Enables triangle counting |
| `diag(x)` | Not started | Extract/create diagonal |

---

## Completed Work

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
Triangle counting:   ❌ Needs trace(x)
Degree computation:  ❌ Needs sum(x, dim)
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
| 2026-01-12 | (pending) | Foundation: detach, max, min, abs, :ema |

---

## Notes

- `lnorm(x)` is the layer normalization function (not `layernorm`)
- Ein tensor ordering: `Input[batch, feature] Weight[output, feature]`
- Current `sum()` and `mean()` only do full reduction - need dimension parameter