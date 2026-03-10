# Simply-Supported Slab – Mid-span Load

A concrete slab (E=30 GPa, ν=0.2) modelled as 2 hex8 elements (4×1×0.5 m).
Left edge fully fixed, right edge pinned (free in X), mid-span loaded downward.
Uses the LU solver as an alternative to Cholesky.

```
   9───10───11
   │    │    │   z=0.5
   6────7────8       ↓ load at 7,10
   │    │    │
   3────4────5
   │    │    │   z=0
   0────1────2
  x=0  x=2  x=4
  ▓▓▓       ○○○
  (fixed)  (pinned)
```

## Run

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @problem.json | python -m json.tool
```
