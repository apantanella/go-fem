# Steel Cube – Uniform Pressure Load

A **unit steel cube** compressed by uniform pressure on the top face.

## Problem

- **Geometry** Unit cube: 1 × 1 × 1 m
- **Material** Steel: E = 210 GPa, ν = 0.3
- **Load** Uniform surface pressure P = 1000 Pa (≈ 1 kPa) on top face (z = 1)
- **Support** Fixed at bottom face (z = 0)
- **Element type** `hexa8`

```
     [Pressure applied on z=1 face]
      ┌─────────────┐
      │             │
      │   P = 1kPa  │
      │             │
      └─────────────┘
      [Fixed at z=0]
```

## Expected Solution

For a cube under uniform pressure:
- Vertical stress σzz ≈ −1 kPa (compressive)
- Vertical settlement uz at top ≈ 4.8 × 10⁻⁶ m
- Lateral (Poisson) strains εxx, εyy ≈ 0.3 × 1/(210,000) ≈ 1.4 × 10⁻⁹

## Run

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @examples/pressure_on_cube/problem.json | python -m json.tool
```

## Output

- **Displacements** at 8 nodes (top face sinks by ~µm)
- **Centroidal stress components** (σzz dominant)
- **Von Mises equivalent stress**
- **Summary** maximum displacement (downward at top)

## Notes

- Demonstrates face-wise load distribution (2×2 Gauss integration on pressure face).
- Used in testing domain-level surface pressure assembly.
