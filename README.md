# go-fem

A 3D structural **Finite Element Method** library and HTTP API server written in pure Go, inspired by the modular architecture of [OpenSees](https://opensees.berkeley.edu/).

## Overview

`go-fem` solves linear static structural problems via a layered architecture that mirrors OpenSees: materials → elements → domain → analysis. Problems are defined programmatically through the Go API or submitted as JSON to the built-in HTTP server.

**Key capabilities:**
- 3D solid, truss, frame, shell, and spring elements
- Linear isotropic elastic material
- Automatic DOF detection (3 or 6 per node) based on element types
- DOF-type-aware global stiffness assembly
- Cholesky and LU linear solvers
- JSON HTTP API for language-agnostic integration

---

## Architecture

Four-layer design mirroring OpenSees:

```
Layer 1 – material/     Constitutive models          (Material3D interface)
Layer 2 – element/      Finite elements               (Element interface)
Layer 3 – domain/       Mesh, assembly, BCs           (Domain struct)
Layer 4 – analysis/     Solution strategy + solver/   (StaticLinearAnalysis)
```

Support packages:

| Package | Description |
|---------|-------------|
| `dof/` | DOF type definitions: `UX, UY, UZ, RX, RY, RZ` |
| `integration/` | Gauss quadrature rules (line, quad, hex, tet, tri) |
| `section/` | Cross-section properties for beams and shells (`BeamSection3D`) |

---

## Layer 1 – Materials

All materials implement `material.Material3D`, which returns a 6×6 constitutive matrix (`C`).

| Type | Package | Parameters | Description |
|------|---------|------------|-------------|
| `IsotropicLinear` | `material/` | `E`, `ν` | 3D linear elastic isotropic |

```go
mat := material.NewIsotropicLinear(200000.0, 0.3) // E=200 GPa, ν=0.3
```

---

## Layer 2 – Elements

All elements implement the `element.Element` interface:

```go
type Element interface {
    GetTangentStiffness() *mat.Dense    // element stiffness matrix Ke
    GetResistingForce()   *mat.VecDense // internal force vector
    NodeIDs()             []int
    NumDOF()              int
    DOFPerNode()          int
    DOFTypes()            []dof.Type    // for global DOF mapping
    Update(disp []float64) error
    CommitState()  error
    RevertToStart() error
}
```

### Solid Elements

| Package | Type | Nodes | DOFs | Integration | Notes |
|---------|------|-------|------|-------------|-------|
| `element/` | `Tet4` | 4 | 12 | 1-pt (exact) | Linear tetrahedron, constant strain |
| `element/` | `Hexa8` | 8 | 24 | 2×2×2 Gauss | Trilinear hexahedron |
| `element/solid/` | `Tet10` | 10 | 30 | 4-pt tet | Quadratic tetrahedron |
| `element/solid/` | `Brick20` | 20 | 60 | 3×3×3 Gauss | Serendipity hexahedron |

### Structural Elements

| Package | Type | Nodes | DOFs | Integration | Notes |
|---------|------|-------|------|-------------|-------|
| `element/truss/` | `Truss3D` | 2 | 6 | Analytical | 3D bar element, axial only |
| `element/truss/` | `CorotTruss` | 2 | 6 | Corotational | Geometrically nonlinear truss |
| `element/frame/` | `ElasticBeam3D` | 2 | 12 | Analytical | Euler-Bernoulli 3D beam (6 DOF/node) |
| `element/quad/` | `Quad4` | 4 | 8 | 2×2 Gauss | Plane stress or plane strain |
| `element/shell/` | `ShellMITC4` | 4 | 24 | 2×2 + SRI | Flat shell: membrane + bending, 6 DOF/node |

### Connector Elements

| Package | Type | Nodes | DOFs | Notes |
|---------|------|-------|------|-------|
| `element/zerolength/` | `ZeroLength` | 2 | 12 | 6-DOF spring connector (6 DOF/node) |
| `element/zerolength/` | `ZeroLength3DOF` | 2 | 6 | 3-DOF spring connector (3 DOF/node) |

---

## Layer 3 – Domain

`domain.Domain` holds the complete FEM model and handles global assembly.

```go
dom := domain.NewDomain()

// Add nodes
n0 := dom.AddNode(0, 0, 0) // returns node ID
n1 := dom.AddNode(1, 0, 0)

// Add elements
dom.AddElement(truss.NewTruss3D(0, [2]int{n0, n1}, coords, E, A))

// Boundary conditions
dom.FixDOF(n0, 0)     // fix UX on node 0
dom.FixNode(n0)       // fix UX, UY, UZ (translations only)
dom.FixNodeAll(n0)    // fix all 6 DOFs

// Loads
dom.ApplyLoad(n1, 1, -10000) // force on node 1, DOF UY = -10 kN

// Assemble global K and F
dom.Assemble()
dom.ApplyDirichletBC()
```

**DOF auto-detection**: `Assemble()` scans all elements and sets `DOFPerNode = max(DOFPerNode across elements)`. Pure solid/truss models use 3 DOF/node; models with beams or shells use 6 DOF/node.

**Dirichlet BCs**: applied by the penalty-free row/column zeroing method (K[gdof,:] = 0, K[:,gdof] = 0, K[gdof,gdof] = 1, F[gdof] = value).

---

## Layer 4 – Analysis & Solvers

### StaticLinearAnalysis

Single-step linear static solve:

1. Assemble global `K` and `F`
2. Apply Dirichlet BCs
3. Solve `K·U = F`
4. Return displacement vector `U`

```go
slv := solver.Cholesky{} // or solver.LU{}
ana := analysis.StaticLinearAnalysis{Dom: dom, Solver: slv}
U, err := ana.Run()

disp := dom.SetDisplacements(U) // [][6]float64 indexed by node ID
```

### Solvers

| Name | Type | Use when |
|------|------|----------|
| `Cholesky` | `solver.Cholesky{}` | Default; symmetric positive definite K (most structural problems) |
| `LU` | `solver.LU{}` | General; use when Cholesky fails (e.g. models with ZeroLength springs) |

Both implement `solver.LinearSolver`:
```go
type LinearSolver interface {
    Solve(K *mat.Dense, F *mat.VecDense) (*mat.VecDense, error)
}
```

---

## HTTP API Server

The `cmd/femgo` binary exposes a JSON REST API.

### Running the server

```bash
# Default port 8080
go run ./cmd/femgo

# Custom port
go run ./cmd/femgo -addr :9090

# Build binary
go build -o femgo ./cmd/femgo
./femgo -addr :8080
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/solve` | Submit a FEM problem (JSON body), receive displacements |
| `GET` | `/health` | Health check: `{"status":"ok","service":"go-fem"}` |
| `GET` | `/` | API info: version, supported elements, materials, solvers |

### POST /solve – Request Format

```json
{
  "materials": [
    {"id": "steel", "type": "isotropic_linear", "E": 200000, "nu": 0.3}
  ],
  "nodes": [
    [0, 0, 0],
    [1, 0, 0]
  ],
  "elements": [
    {"type": "truss3d", "nodes": [0, 1], "E": 200000, "A": 0.01}
  ],
  "boundary_conditions": [
    {"node": 0, "dofs": [0, 1, 2]}
  ],
  "loads": [
    {"node": 1, "dof": 0, "value": 5000}
  ],
  "solver": "cholesky"
}
```

**DOF indices**: `0=UX`, `1=UY`, `2=UZ`, `3=RX`, `4=RY`, `5=RZ`

**Solver options**: `"cholesky"` (default) or `"lu"`

### POST /solve – Response Format

```json
{
  "success": true,
  "info": {
    "num_nodes": 2,
    "num_elements": 1,
    "num_dofs": 6,
    "dof_per_node": 3,
    "solver": "cholesky"
  },
  "displacements": [
    {"node": 0, "ux": 0.0, "uy": 0.0, "uz": 0.0},
    {"node": 1, "ux": 2.5e-4, "uy": 0.0, "uz": 0.0}
  ],
  "summary": {
    "max_abs_displacement": {"node": 1, "component": "ux", "value": 2.5e-4}
  },
  "elapsed_ms": 1.3
}
```

On error (`"success": false`), the response has HTTP 422 and an `"error"` field.

### Element types in JSON

| `"type"` | Nodes | Required fields | Optional |
|----------|-------|-----------------|----------|
| `"tet4"` | 4 | `material` | — |
| `"hexa8"` | 8 | `material` | — |
| `"tet10"` | 10 | `material` | — |
| `"brick20"` | 20 | `material` | — |
| `"truss3d"` | 2 | `E`, `A` | — |
| `"corot_truss"` | 2 | `E`, `A` | — |
| `"elastic_beam3d"` | 2 | `E`, `G`, `A`, `Iy`, `Iz`, `J`, `vec_xz` | — |
| `"shell_mitc4"` | 4 | `E`, `nu`, `thickness` | — |
| `"zerolength"` | 2 | `springs` (array of 6 stiffnesses: kUX..kRZ) | — |

Solid elements (`tet4`, `hexa8`, `tet10`, `brick20`) require a `"material"` string referencing an entry in `"materials"`.

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/your-org/go-fem.git
cd go-fem
go mod tidy

# Run the API server
go run ./cmd/femgo

# Run all tests
go test ./...

# Solve an example problem
curl -X POST http://localhost:8080/solve \
     -H "Content-Type: application/json" \
     -d @examples/bridge_load/problem.json
```

---

## Examples

| Directory | Element Type | Description |
|-----------|-------------|-------------|
| `examples/cantilever_beam/` | `Hexa8` | 3D cantilever beam with tip point load |
| `examples/single_hex8/` | `Hexa8` | Single cube under uniaxial compression |
| `examples/tet4_patch/` | `Tet4` | Tetrahedral patch test (constant stress) |
| `examples/bridge_load/` | `Truss3D` | 2D Warren truss bridge under vertical load |
| `examples/frame_portal/` | `ElasticBeam3D` | Portal frame under lateral (sway) load |
| `examples/beam3d/` | `ElasticBeam3D` | Single 3D beam with end moment |

Each example directory contains a `problem.json` ready to POST to `/solve` and, where applicable, a `main.go` for programmatic use.

### Example: Portal Frame (`examples/frame_portal/problem.json`)

A 6 m × 3 m portal frame with fixed bases, 3 beam elements, lateral load of 1 kN at left column top:

```json
{
  "nodes": [[0,0,0],[0,3,0],[6,3,0],[6,0,0]],
  "elements": [
    {"type":"elastic_beam3d","nodes":[0,1],"E":200000,"G":80000,
     "A":0.01,"Iy":8.33e-6,"Iz":8.33e-6,"J":1.41e-5,"vec_xz":[0,0,1]},
    {"type":"elastic_beam3d","nodes":[1,2],"E":200000,"G":80000,
     "A":0.01,"Iy":8.33e-6,"Iz":8.33e-6,"J":1.41e-5,"vec_xz":[0,0,1]},
    {"type":"elastic_beam3d","nodes":[3,2],"E":200000,"G":80000,
     "A":0.01,"Iy":8.33e-6,"Iz":8.33e-6,"J":1.41e-5,"vec_xz":[0,0,1]}
  ],
  "boundary_conditions": [
    {"node":0,"dofs":[0,1,2,3,4,5]},
    {"node":3,"dofs":[0,1,2,3,4,5]}
  ],
  "loads": [{"node":1,"dof":0,"value":1000}],
  "solver": "lu"
}
```

### Example: Truss Bridge (`examples/bridge_load/problem.json`)

A 5-node Warren truss, 7 members, 10 kN downward load at midspan:

```json
{
  "nodes": [[0,0,0],[5,0,0],[10,0,0],[2.5,3,0],[7.5,3,0]],
  "elements": [
    {"type":"truss3d","nodes":[0,1],"E":200000,"A":0.01},
    {"type":"truss3d","nodes":[1,2],"E":200000,"A":0.01},
    {"type":"truss3d","nodes":[0,3],"E":200000,"A":0.01},
    {"type":"truss3d","nodes":[1,3],"E":200000,"A":0.01},
    {"type":"truss3d","nodes":[1,4],"E":200000,"A":0.01},
    {"type":"truss3d","nodes":[2,4],"E":200000,"A":0.01},
    {"type":"truss3d","nodes":[3,4],"E":200000,"A":0.01}
  ],
  "boundary_conditions": [
    {"node":0,"dofs":[0,1,2]},
    {"node":1,"dofs":[2]},
    {"node":2,"dofs":[1,2]},
    {"node":3,"dofs":[2]},
    {"node":4,"dofs":[2]}
  ],
  "loads": [{"node":1,"dof":1,"value":-10000}],
  "solver": "lu"
}
```

---

## Validation

`validation/main.go` runs a suite of problems with known closed-form solutions and verifies that the numerical result matches analytical theory to tight tolerances.

```bash
go run ./validation
```

### Results (March 10, 2026)

| Case | Element | Numerical | Theoretical | Rel. Err (%) | Status |
|------|---------|-----------|-------------|--------------|--------|
| Truss – axial deformation | `Truss3D` | 5.000e-04 | 5.000e-04 | 0.00e+00 | PASS ✓ |
| Beam – cantilever tip deflection | `ElasticBeam3D` | 3.333e-01 | 3.333e-01 | 0.00e+00 | PASS ✓ |
| Beam – simply-supported midspan | `ElasticBeam3D` | 1.667e-01 | 1.667e-01 | 0.00e+00 | PASS ✓ |
| Hexa8 – uniaxial patch test | `Hexa8` | 1.000e+00 | 1.000e+00 | 0.00e+00 | PASS ✓ |

### Case descriptions

| Case | Theoretical formula | Parameters |
|------|--------------------|-----------------------|
| Truss axial deformation | $\delta_x = \frac{FL}{EA}$ | $F=10000\ \text{N},\ L=1\ \text{m},\ E=200000\ \text{N/mm}^2,\ A=100\ \text{mm}^2$ |
| Cantilever tip deflection | $\delta_y = \frac{FL^3}{3EI_z}$ | $F=1,\ L=1,\ E=1,\ I_z=1$ |
| Simply-supported midspan | $\delta_{\text{mid}} = \frac{FL^3}{48EI_z}$ | $F=1,\ L=2,\ E=1,\ I_z=1$ |
| Hexa8 uniaxial patch test | $u_x = \frac{\sigma_x}{E}\cdot L = 1$ | Unit cube, $E=1,\ \nu=0,\ F=1$ distributed over $x=1$ face |

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Go | 1.22+ | Language runtime |
| [gonum](https://gonum.org/) `gonum.org/v1/gonum` | v0.15.0 | Dense matrix operations, Cholesky/LU factorization |

No other external dependencies.

---

## Extending go-fem

All layers are defined by Go interfaces, making it straightforward to plug in new implementations:

### Adding a Material

Implement `material.Material3D`:
```go
type Material3D interface {
    C() *mat.Dense // 6×6 constitutive matrix (Voigt notation)
}
```
Examples to add: `ElastoPlastic`, `Hyperelastic`, `Orthotropic`.

### Adding an Element

Implement `element.Element`. The key methods for nonlinear analysis are `Update()`, `CommitState()`, and `RevertToStart()`. For linear elements these can be no-ops.

### Adding a Solver

Implement `solver.LinearSolver`:
```go
type LinearSolver interface {
    Solve(K *mat.Dense, F *mat.VecDense) (*mat.VecDense, error)
}
```
Examples to add: sparse iterative solvers (CG, GMRES), sparse direct solvers.

### Adding an Analysis Strategy

Planned extensions:
- **Nonlinear static** (Newton-Raphson with load stepping)
- **Dynamic** (Newmark-β time integration)
- **Modal** (eigenvalue analysis for natural frequencies)

---

## License

MIT – see [LICENSE](LICENSE).
