# Plane Frame 2D — Portal Frame

Simple portal frame composed of two columns and one beam, using **elastic_beam2d** elements.

## Geometry

```
    ←— 6 m —→
  1 ═══════════ 2   ← beam (UDL 20 kN/m ↓)
  │             │
  │  4 m        │
  │             │
  0 (fixed)     3 (fixed)
```

## Properties

| Property | Value | Unit |
|----------|------:|------|
| E | 210 000 | MPa |
| A | 53.8 | cm² |
| Iz | 8 356 | cm⁴ |
| UDL on beam | 20 | kN/m |

## Element Type

**`elastic_beam2d`** — 2-node Euler-Bernoulli beam in the XY plane.
- 3 DOFs per node: UX, UY, RZ
- 6 DOFs per element

## Expected Behaviour

- Columns develop bending moments due to rigid joints
- Maximum vertical displacement at mid-span of the beam
- Fixed supports provide full restraint (UX, UY, RZ = 0)
