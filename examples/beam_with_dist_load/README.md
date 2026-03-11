# Two-Span Beam – Uniformly Distributed Load

A **two-span steel beam** with both ends clamped, subjected to uniformly distributed load (UDL).

## Problem

- **Spans** 2 × 2000 mm (total 4 m)
- **Cross-section** 100 × 100 mm square solid
- **Material** Steel: E = 210 GPa, G = 80.769 GPa, ν = 0.3
- **Load** UDL = 5 N/mm = 5 kN/m in −Y direction
- **Support** Fixed (all DOFs) at nodes 0 and 2
- **Element type** `elastic_beam3d` (Euler-Bernoulli)

## Expected Solution

For a two-span beam with uniform load and fixed supports, the maximum mid-span deflection is approximately **1.9 mm** (downward).

## Run

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @examples/beam_with_dist_load/problem.json | python -m json.tool
```

## Output

- **Displacements** at all nodes (ux, uy, uz, rx, ry, rz)
- **Element forces** (end moments, shear, axial) for each beam segment
- **Summary** maximum deflection and location
