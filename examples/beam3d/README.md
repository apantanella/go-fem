# Cantilever Beam – Convergence Study

Convergence analysis of a cantilever beam solved with **Hexa8** solid elements at increasing mesh densities.

## Problem

A cantilever beam:
- **Length** L = 10 m
- **Cross-section** 1 × 1 m (square)
- **Material** Steel, E = 200 GPa, ν = 0.3
- **Load** P = 1000 N distributed on the free end (−Z direction)
- **Support** Fixed at x = 0

## Expected Solution

The theoretical maximum tip deflection (Euler-Bernoulli beam):

$$\delta = \frac{P L^3}{3 E I} \approx -0.1 \text{ m}$$

## Run

### Option 1: Convergence Study (live computation)

This is a **live convergence study** that runs locally:

```bash
go run ./examples/beam3d
```

Output shows tip deflection for mesh refinements (1×1×1, 2×2×2, 3×3×3, 4×4×4 Hexa8 elements) and convergence rate.

### Option 2: Static API Solve (2×2×2 mesh)

A pre-built `problem.json` with a 2×2×2 Hexa8 mesh:

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @examples/beam3d/problem.json | python -m json.tool
```

Expected maximum deflection at the free end ≈ −0.04–0.06 m (coarser than EB theory due to limited element count).

## Notes

- More elements = better agreement with continuum theory.
- Demonstrates auto-DOF detection (3 DOFs/node for solid elements).
- Run `main.go` for a full convergence study; use `problem.json` for quick validation with the API.
