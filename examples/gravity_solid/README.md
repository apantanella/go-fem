# Unit Cube – Self-Weight under Gravity

A **solid concrete cube** (1 m³) subjected to its own self-weight (body-force load).

## Problem

- **Geometry** Unit cube: 1 × 1 × 1 m
- **Material** Concrete: E = 30 GPa, ν = 0.2, ρ = 2400 kg/m³
- **Load** Gravity: g = [0, 0, −9.81] m/s²
- **Support** Fixed at the bottom face (z = 0)
- **Element type** `hexa8` (single element)

## Expected Solution

Total weight: ρ · V · g ≈ 2400 × 1 × 9.81 ≈ **23,544 N** distributed as body-force load.

Expected vertical stress at the base ≈ 23.5 kPa.

## Run

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @examples/gravity_solid/problem.json | python -m json.tool
```

## Output

- **Displacements** at 8 nodes (top face has non-zero uz)
- **Centroidal stresses** (Sxx, Syy, Szz, von Mises)
- **Summary** maximum displacement (vertical, at top center)
