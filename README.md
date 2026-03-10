# go-fem

A 3D structural Finite Element Method library in pure Go, inspired by the modular architecture of [OpenSees](https://opensees.berkeley.edu/).

## Architecture

Four-layer design mirroring OpenSees:

```
Layer 1 – material/     Constitutive models (Material3D interface)
Layer 2 – element/      Finite elements (Element interface)
Layer 3 – domain/       Mesh, assembly, boundary conditions
Layer 4 – analysis/     Solution strategies + solver/
```

### Layer 1 – Materials
| Type | Description |
|------|-------------|
| `IsotropicLinear` | 3D linear elastic isotropic (E, ν) |

### Layer 2 – Elements
| Type | Nodes | DOFs | Integration |
|------|-------|------|-------------|
| `Tet4` | 4 | 12 | 1-point (exact for linear) |
| `Hexa8` | 8 | 24 | 2×2×2 full Gauss |

### Layer 3 – Domain
- Node management with 3 translational DOFs
- Global stiffness assembly (scatter)
- Dirichlet BCs (row/column zeroing)
- Nodal loads

### Layer 4 – Analysis
- `StaticLinearAnalysis` – single-step linear static solve
- Cholesky and LU solvers

## Quick Start

```bash
# Clone and build
git clone https://github.com/your-org/go-fem.git
cd go-fem
go mod tidy

# Run the cantilever beam example
go run ./cmd/femgo

# Run convergence study
go run ./examples/beam3d

# Run tests
go test ./...
```

## Example Output

```
=== go-fem: 3D Cantilever Beam (Hexa8) ===
Mesh: 10×2×2 = 40 elements, 99 nodes, 297 DOFs

Results:
  Max tip Uz  = -19.xxxxxx  (node XX)
  Beam theory = -20.000000  (Euler-Bernoulli PL³/3EI)
  Ratio FEM/EB = 0.9xxx
```

## Dependencies

- Go 1.22+
- [gonum](https://gonum.org/) – `gonum.org/v1/gonum/mat` for dense linear algebra

## Extending

All layers are defined by Go interfaces, making it straightforward to add:

- **Materials**: implement `material.Material3D` (e.g., `ElastoPlastic`, `Hyperelastic`)
- **Elements**: implement `element.Element` (e.g., `Tet10`, `Hexa20`, shells, beams)
- **Solvers**: implement `solver.LinearSolver` (e.g., sparse iterative solvers)
- **Analysis**: nonlinear Newton-Raphson, dynamic (Newmark), modal analysis

## License

MIT – see [LICENSE](LICENSE).
