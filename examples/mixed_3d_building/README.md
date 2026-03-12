# Mixed 3D Building — Multi-Element Integration Test

A 3D industrial building model that combines **5 different element types** in a
single analysis to stress-test global stiffness matrix assembly and solver
convergence with mixed DOF patterns.

## Geometry

| Component        | Type               | Count | Description                              |
|------------------|--------------------|------:|------------------------------------------|
| Foundation       | `hexa8_3d`         |     9 | 3×3×1 concrete slab (6 m × 6 m × 1 m)   |
| Columns          | `elastic_beam_3d`  |    16 | Steel columns (z = 1 → 5 m)              |
| Roof beams       | `elastic_beam_3d`  |    24 | Steel beams at roof level (z = 5 m)       |
| Wall bracing     | `truss_3d`         |    24 | X-bracing on 4 exterior walls             |
| Roof panels (Q)  | `shell_mitc4_3d`   |     4 | Corner roof panels                        |
| Roof panels (T)  | `dkt3_3d`          |    10 | Interior roof triangles                   |
| **Total**        |                    | **87**| **48 nodes · 288 DOFs (6 DOF/node)**     |

## Materials

- **Concrete** (foundation): E = 30 000 MPa, ν = 0.2
- **Steel** (frame / bracing / roof): E = 210 000 MPa, ν = 0.3

## Boundary Conditions

All 16 foundation base nodes (z = 0) are fully fixed (6 DOFs each).

## Loads

- 50 kN downward (−Z) on each roof node (16 loads)
- Body-force gravity (ρ = 2400 kg/m³, g = −9.81 m/s²) on each Hexa8 element

## Expected Results

The solver must converge successfully. Max vertical roof deflection ≈ −1.0 mm
(downward), confirming correct assembly of the mixed 6-DOF global stiffness
matrix with elements contributing [UX,UY,UZ] (solids, trusses) and
[UX,UY,UZ,RX,RY,RZ] (beams, shells).
