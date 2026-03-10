# Single Hexa8 – Uniform Compression

A single unit cube (1×1×1) in steel, fixed at the bottom face (z=0),
with a total downward force of 1000 N distributed on the top face.

```
    7 ──────── 6        z=1  (loaded: -250N each)
   /|         /|
  4 ──────── 5 |
  |  3 ──────|── 2     z=0  (fixed)
  | /        | /
  0 ──────── 1
```

## Run

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @problem.json | python -m json.tool
```
