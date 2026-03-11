# DKT3 Triangular Plate – Point Load

Single 3-node **Discrete Kirchhoff Triangle** plate element in the XY plane.
The edge between nodes 0 and 1 is clamped; a downward nodal load is applied at node 2.

```
node2 (0,200,0)   Fz = -1000 N
   *
   |\
   | \
   |  \
   |   \
   *----*
node0  node1
(clamped edge)
```

## Run

Start API server:

```bash
go run ./cmd/femgo
```

Solve this example:

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @examples/dkt3_tri_plate/problem.json | python -m json.tool
```

Solve the same case using alias element type (`discrete_kirchhoff_triangle`):

```bash
curl -s -X POST http://localhost:8080/solve \\
  -H "Content-Type: application/json" \\
  -d @examples/dkt3_tri_plate/problem_alias.json | python -m json.tool
```

You should see a non-zero `uz` displacement at node 2 and shell resultants with bending moments (`Mx`, `My`, `Mxy`).

## Equivalent distributed load (UDL)

`dkt3` does not currently expose a dedicated pressure/UDL load interface.
For a uniform transverse load `q` on a linear triangle, you can use equivalent nodal forces:

`Fz_i = q · A / 3` for each node `i = 0..2`

This repository includes a ready example:

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @examples/dkt3_tri_plate/problem_udl_equiv.json | python -m json.tool
```
