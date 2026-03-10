# Cantilever Beam – 4 Hexa8 Elements

A steel cantilever beam (L=10, section 1×1) modelled with 4 hex8 elements.
Fixed at x=0, total tip load P=1000 N downward (-Z) distributed on 4 tip nodes.

```
  15───16───17───18───19
  │    │    │    │    │    z=1
  10───11───12───13───14
  │    │    │    │    │
   5────6────7────8────9
  │    │    │    │    │    z=0
   0────1────2────3────4
  x=0                x=10
  ▓▓▓                 ↓ P
  (fixed)
```

Analytical reference (Euler-Bernoulli): δ = PL³/3EI = 20.0

## Run

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @problem.json | python -m json.tool
```
