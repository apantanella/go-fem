# Portal Frame – Lateral Load

A simple **rectangular portal frame** with fixed bases and a horizontal sway load at the top.

## Problem

- **Columns** Height H = 3000 mm
- **Beam** Span L = 6000 mm (between column tops)
- **Cross-section** 100 × 100 mm square solid (all members)
- **Material** Steel: E = 210 GPa, G = 80.769 GPa
- **Load** Lateral force = 10 kN at node 1 (column top, −X direction)
- **Support** Fixed (all DOFs) at nodes 0 and 3 (column bases)
- **Element type** `elastic_beam3d`

```
    1 ──────────── 2
    |              |
  node1          node2
    |              |
    0 ──────────── 3
  (fixed)       (fixed)
```

## Expected Solution

For a portal frame under lateral load, typical horizontal sway at the top is **6–7 mm** (in +X direction relative to load).

## Run

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @examples/frame_portal/problem.json | python -m json.tool
```

## Output

- **Displacements** at all 4 nodes
- **Internal forces** (axial, shear, bending moments) in each member
- **Summary** maximum sway displacement
