# Orthotropic Cube – Softwood Compression

A **unit cube of softwood (spruce)** with orthotropic material properties, compressed along the grain.

## Problem

- **Geometry** Unit cube: 1 × 1 × 1 m
- **Material** Softwood (spruce):
  - Ex = 12,000 MPa (along grain)
  - Ey = 500 MPa (across grain)
  - Ez = 800 MPa (across grain)
  - νxy = 0.40, νyz = 0.30, νxz = 0.35
  - Gxy = 700 MPa, Gyz = 60 MPa, Gxz = 900 MPa
- **Load** Compressive force 1 kN in −X direction (along grain)
- **Support** Fixed at bottom face (z = 0)
- **Element type** `hexa8`

## Expected Solution

Since the load is along the strong axis (Ex is much larger than E_y/E_z), the cube deforms primarily in X.

Expected strain along X ≈ 1000/(1000×12000) ≈ 8.3 × 10⁻⁵.

## Run

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @examples/ortho_cube/problem.json | python -m json.tool
```

## Output

- **Displacements** showing compressive response in X
- **Stresses** (σxx dominant, σyy/σzz from Poisson effect)
- **Summary** maximum axial displacement

## Notes

- Demonstrates orthotropic linear elastic material with 9 independent constants.
- Maxwell reciprocity is enforced in material.NewOrthotropicLinear().
