# Deep Cantilever Beam – Timoshenko vs. Euler-Bernoulli

A **short cantilever beam** where **shear deformation is significant**, demonstrating the difference between Timoshenko and Euler-Bernoulli theories.

## Problem

- **Length** L = 400 mm
- **Cross-section** 100 × 200 mm rectangular (solid)
  - Area A = 20,000 mm²
  - Iz = 100·200³/12 ≈ 66.67 × 10⁶ mm⁴
  - Shear correction factor κ = 5/6 → As = κ·A = 16,667 mm²
- **Material** Steel: E = 210 GPa, G = 80.769 GPa
- **Load** Tip load P = 10 kN in −Y direction
- **Support** Fixed at x = 0
- **Slenderness ratio** L/h = 400/200 = 2 (short/deep beam)

## Expected Solution

### Euler-Bernoulli (no shear)

$$\delta_{\text{EB}} = \frac{P L^3}{3 E I} \approx -0.01524 \text{ mm}$$

### Timoshenko (with shear)

Bending deflection ≈ −0.015238 mm
Shear deflection ≈ −0.002971 mm
**Total** ≈ **−0.018209 mm**

**Shear contribution ≈ 16.3%** — significant for L/h = 2 beam!

## Run

```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d @examples/timoshenko_deep_beam/problem.json | python -m json.tool
```

## Output

- **Displacement** at node 1 (free end): uy ≈ −0.018209 mm
- **Internal moments/shears** at element ends
- **Summary** maximum vertical displacement

## Comparison

To compare with `elastic_beam3d`, replace `"type": "timoshenko_beam3d"` with `"type": "elastic_beam3d"` in the same problem file:

```bash
# Edit problem.json: change type to elastic_beam3d
curl ... -d @problem_elastic.json
```

Expected difference: Timoshenko ≈ 16–20% more flexible for L/h = 2.

## Notes

- **When to use Timoshenko**: L/h < 10 (thick beams, stress concentration, shear-dominated)
- **When to use Euler-Bernoulli**: L/h > 20 (slender beams, bending-dominated)
- Shear areas can be customized via `"Asy"` and `"Asz"` (default: κ = 5/6)
