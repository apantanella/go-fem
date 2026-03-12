# Triangular Plane Elements — Tri3 (CST) & Tri6 (LST)

Three example problems showcasing the 2D triangular plane elements.

## Files

| File | Element | Formulation | Description |
|------|---------|-------------|-------------|
| `problem_tri3.json` | Tri3 (CST) | Plane stress | 4-element cantilever, P = 1000 N at tip |
| `problem_tri3_strain.json` | Tri3 (CST) | Plane strain | Same geometry in plane strain |
| `problem_tri6.json` | Tri6 (LST) | Plane stress | 2-element cantilever with quadratic triangles |

## Geometry

Rectangular plate 2 × 1 m, thickness t = 0.1 m.
Left edge (x = 0) clamped, tip load P = 1000 N (split into two nodes for Tri3, concentrated for Tri6).

```
3----4----5         2
|   /|   /|         |\
|  / |  / |    or   5  4
| /  | /  |         |   \
0----1----2         0--3--1
  Tri3 mesh           Tri6 mesh
```

## Material

Steel: E = 210 000 MPa, ν = 0.3.

## Running

```bash
curl -X POST http://localhost:8080/solve -d @problem_tri3.json
curl -X POST http://localhost:8080/solve -d @problem_tri6.json
curl -X POST http://localhost:8080/solve -d @problem_tri3_strain.json
```

## Expected Behaviour

- **Tri3** (CST): constant strain per element → stress is uniform within each triangle. Stiffer than the exact solution (2 DOFs/node, linear shape functions).
- **Tri6** (LST): linear strain field → richer stress distribution. More accurate than Tri3 with fewer elements.
- **Plane strain** produces smaller displacements than plane stress (additional lateral constraint → stiffer response).
