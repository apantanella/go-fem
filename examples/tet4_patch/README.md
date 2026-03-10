# Tet4 Patch – 3-Element Aluminum Block

A small patch of 3 tetrahedra in aluminum (E=70 GPa, ν=0.33).
Node 0 is fully fixed, node 1 is constrained in Y/Z, node 2 in Z.
Upward forces applied on the top nodes (3 and 5).

## Run

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @problem.json | python -m json.tool
```
