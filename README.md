# go-fem

A 3D structural **Finite Element Method** library and HTTP API server written in pure Go, inspired by the modular architecture of [OpenSees](https://opensees.berkeley.edu/).

## Overview

`go-fem` solves linear static structural problems via a layered architecture that mirrors OpenSees: materials → elements → domain → analysis. Problems are defined programmatically through the Go API or submitted as JSON to the built-in HTTP server.

**Key capabilities:**
- 3D solid, truss, frame (Euler-Bernoulli and Timoshenko), shell, and spring elements
- Isotropic and orthotropic linear elastic materials
- Automatic DOF detection (3 or 6 per node) based on element types
- DOF-type-aware global stiffness assembly
- Cholesky and LU linear solvers
- Multiple load types: nodal forces, surface pressure, beam UDL, body force/gravity
- Non-zero prescribed displacements (imposed settlement)
- JSON HTTP API for language-agnostic integration
- Consistent N-mm-MPa unit system throughout all examples

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

All materials implement `material.Material3D`, which exposes a 6×6 constitutive matrix D (stiffness) and stress computation via `SetTrialStrain` / `GetStress`.

| Type | JSON `"type"` | Parameters | Description |
|------|--------------|------------|-------------|
| `IsotropicLinear` | `"isotropic_linear"` | `E`, `ν` | 3D linear elastic isotropic |
| `OrthotropicLinear` | `"orthotropic_linear"` | `Ex/Ey/Ez`, `νxy/νyz/νxz`, `Gxy/Gyz/Gxz` | 3D linear elastic orthotropic (9 constants) |

```go
// Isotropic
iso := material.NewIsotropicLinear(200000.0, 0.3)

// Orthotropic — D = S⁻¹ where S is the compliance matrix
ortho, err := material.NewOrthotropicLinear(
    12000, 500, 800,    // Ex, Ey, Ez
    0.40, 0.30, 0.35,  // Nxy, Nyz, Nxz
    700, 60, 900,       // Gxy, Gyz, Gxz
)
```

**Maxwell reciprocity** is enforced automatically: `ν_yx = ν_xy · Ey/Ex`, etc.
The stiffness matrix D is obtained by inverting the 6×6 compliance matrix S; an error is returned if S is singular (invalid parameter combination).

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
| `element/solid/` | `Tet4` | 4 | 12 | 1-pt (exact) | Linear tetrahedron, constant strain |
| `element/solid/` | `Hexa8` | 8 | 24 | 2×2×2 Gauss | Trilinear hexahedron |
| `element/solid/` | `Tet10` | 10 | 30 | 4-pt tet | Quadratic tetrahedron |
| `element/solid/` | `Brick20` | 20 | 60 | 3×3×3 Gauss | Serendipity hexahedron |

### Structural Elements

| Package | Type | Nodes | DOFs | Integration | Notes |
|---------|------|-------|------|-------------|-------|
| `element/truss/` | `Truss3D` | 2 | 6 | Analytical | 3D bar element, axial only (3 DOF/node) |
| `element/truss/` | `Truss2D` | 2 | 4 | Analytical | 2D bar element, axial only (2 DOF/node: UX, UY) |
| `element/truss/` | `CorotTruss` | 2 | 6 | Corotational | Geometrically nonlinear truss |
| `element/frame/` | `ElasticBeam3D` | 2 | 12 | Analytical | Euler-Bernoulli 3D beam (6 DOF/node) |
| `element/frame/` | `ElasticBeam2D` | 2 | 6 | Analytical | Euler-Bernoulli 2D beam for plane frames (3 DOF/node: UX, UY, RZ) |
| `element/frame/` | `TimoshenkoBeam3D` | 2 | 12 | Analytical | Timoshenko 3D beam — shear-deformable (6 DOF/node) |
| `element/frame/` | `TimoshenkoBeam2D` | 2 | 6 | Analytical | Timoshenko 2D beam — shear-deformable for plane frames (3 DOF/node) |
| `element/quad/` | `Quad4` | 4 | 8 | 2×2 Gauss | Plane stress or plane strain |
| `element/quad/` | `Tri3` | 3 | 6 | Exact (constant B) | CST — constant strain triangle, plane stress/strain |
| `element/quad/` | `Tri6` | 6 | 12 | 3-pt triangle | LST — linear strain triangle, plane stress/strain |
| `element/shell/` | `ShellMITC4` | 4 | 24 | 2×2 + SRI | Flat shell: membrane + bending, 6 DOF/node |
| `element/shell/` | `DKT3` | 3 | 18 | Triangular (area-constant) | Thin plate bending (Discrete Kirchhoff style), active DOFs: UZ/RX/RY |

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

// Nodal load
dom.ApplyLoad(n1, 1, -10000) // force on node 1, DOF UY = -10 kN

// Beam uniformly distributed load (requires ElasticBeam3D element)
dom.AddBeamDistLoad(elemIdx, [3]float64{0, -1, 0}, 5000) // 5 kN/m downward

// Surface pressure on a 4-node face (CCW from outside = outward normal)
dom.AddSurfacePressure([4]int{4, 5, 6, 7}, 1000) // 1000 Pa on top face

// Body force / gravity on a solid element
dom.AddBodyForce(elemIdx, 7800, [3]float64{0, -9.81, 0}) // steel self-weight

// Assemble global K and F
dom.Assemble()
dom.ApplyDirichletBC()
```

**DOF auto-detection**: `Assemble()` scans all elements and sets `DOFPerNode = max(DOFPerNode across elements)`. Pure plane-stress/strain models (Tri3, Tri6, Quad4) and Truss2D use 2 DOF/node; solid/truss-3D models use 3 DOF/node; plane frames (ElasticBeam2D, TimoshenkoBeam2D) use 3 DOF/node; models with 3D beams or shells use 6 DOF/node.

**Dirichlet BCs**: applied by the penalty-free row/column zeroing method (K[gdof,:] = 0, K[:,gdof] = 0, K[gdof,gdof] = 1, F[gdof] = value). Non-zero prescribed displacements set `bc.Value ≠ 0`.

### Load types

| Method | Description | Supported elements |
|--------|-------------|-------------------|
| `ApplyLoad(node, dof, value)` | Concentrated force or moment | All |
| `AddBeamDistLoad(elemIdx, dir, intensity)` | UDL (N/m), work-equivalent nodal forces | `ElasticBeam3D`, `ElasticBeam2D`, `TimoshenkoBeam3D`, `TimoshenkoBeam2D` |
| `AddSurfacePressure(faceNodes[4], P)` | Uniform pressure on quad face (2×2 Gauss) | Any 4 nodes |
| `AddBodyForce(elemIdx, rho, g)` | Gravity/body force | `Tet4` (exact), `Hexa8` (Gauss) |

Elements implement optional load interfaces:
- `element.EquivalentNodalLoader` — converts distributed load to nodal forces
- `element.BodyForceLoader` — computes body-force nodal contributions

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
    {"node": 0, "dofs": [0, 1, 2]},
    {"node": 1, "dofs": [2], "values": [0.005]}
  ],
  "loads": [
    {"node": 1, "dof": 0, "value": 5000}
  ],
  "solver": "cholesky"
}
```

**DOF indices**: `0=UX`, `1=UY`, `2=UZ`, `3=RX`, `4=RY`, `5=RZ`

**Solver options**: `"cholesky"` (default) or `"lu"`

**Prescribed displacements**: add a `"values"` array to `boundary_conditions` parallel to `"dofs"`. Omitted values default to 0.

#### Load types

All load types use a `"type"` discriminator field. Omitting `"type"` is equivalent to `"nodal"`.

| `"type"` | Required fields | Description |
|----------|----------------|-------------|
| `"nodal"` *(default)* | `node`, `dof`, `value` | Concentrated force or moment |
| `"surface_pressure"` | `face_nodes` (4 ints), `pressure` | Uniform pressure on a quad face |
| `"beam_dist"` | `element`, `dir` ([3]float64), `intensity` | UDL on a beam element (N/m) |
| `"body_force"` | `element`, `rho`, `g` ([3]float64) | Gravity/body force on a solid element |

```json
"loads": [
  {"node": 3, "dof": 1, "value": -10000},
  {"type": "beam_dist", "element": 0, "dir": [0, -1, 0], "intensity": 5000},
  {"type": "surface_pressure", "face_nodes": [4, 5, 6, 7], "pressure": 1000},
  {"type": "body_force", "element": 0, "rho": 7800, "g": [0, -9.81, 0]}
]
```

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

### Material types in JSON

| `"type"` | Required fields | Description |
|----------|----------------|-------------|
| `"isotropic_linear"` | `E`, `nu` | Isotropic linear elastic |
| `"orthotropic_linear"` | `Ex`, `Ey`, `Ez`, `nxy`, `nyz`, `nxz`, `Gxy`, `Gyz`, `Gxz` | Orthotropic linear elastic (9 constants) |

```json
{"id": "timber", "type": "orthotropic_linear",
 "Ex": 12000, "Ey": 500,  "Ez": 800,
 "nxy": 0.40, "nyz": 0.30, "nxz": 0.35,
 "Gxy": 700,  "Gyz": 60,  "Gxz": 900}
```

### Element types in JSON

| `"type"` | Nodes | Required fields | Optional |
|----------|-------|-----------------|----------|
| `"tet4"` | 4 | `material` | — |
| `"hexa8"` | 8 | `material` | — |
| `"tet10"` | 10 | `material` | — |
| `"brick20"` | 20 | `material` | — |
| `"truss3d"` | 2 | `E`, `A` | — |
| `"truss2d"` | 2 | `E`, `A` | 2 DOF/node (UX, UY) |
| `"corot_truss"` | 2 | `E`, `A` | — |
| `"elastic_beam3d"` | 2 | `E`, `G`, `A`, `Iy`, `Iz`, `J`, `vec_xz` | — |
| `"elastic_beam2d"` | 2 | `E`, `A`, `Iz` | 3 DOF/node (UX, UY, RZ) |
| `"timoshenko_beam3d"` | 2 | `E`, `G`, `A`, `Iy`, `Iz`, `J`, `vec_xz` | `Asy`, `Asz` |
| `"timoshenko_beam2d"` | 2 | `E`, `G`, `A`, `Iz` | `Asy`; 3 DOF/node (UX, UY, RZ) |
| `"shell_mitc4"` | 4 | `E`, `nu`, `thickness` | — |
| `"dkt3"` | 3 | `E`, `nu`, `thickness` | alias: `"discrete_kirchhoff_triangle"` |
| `"quad4"` | 4 | `E`, `nu`, `thickness` | `plane_type`: `"stress"` (default) or `"strain"` |
| `"tri3"` | 3 | `E`, `nu`, `thickness` | CST; `plane_type`: `"stress"` or `"strain"` |
| `"tri6"` | 6 | `E`, `nu`, `thickness` | LST; `plane_type`: `"stress"` or `"strain"` |
| `"zerolength"` | 2 | `springs` (array of 6 stiffnesses: kUX..kRZ) | — |

Solid elements (`tet4`, `hexa8`, `tet10`, `brick20`) require a `"material"` string referencing an entry in `"materials"`.

#### Timoshenko beam — shear correction

`timoshenko_beam3d` and `timoshenko_beam2d` share the same DOF layout and JSON fields as their Euler-Bernoulli counterparts but account for transverse shear deformation via the shear-flexibility parameter:

```
Φ = 12·E·I / (G·As·L²)
```

where `As = κ·A` is the effective shear area (`κ` = shear correction factor). The stiffness matrix reduces to the Euler-Bernoulli matrix when `Φ → 0` (rigid shear, slender beam).

Shear areas in JSON (`Asy` for bending in the x-y plane, `Asz` for the x-z plane — 3D only):

```json
{
  "type": "timoshenko_beam3d",
  "nodes": [0, 1],
  "E": 210000, "G": 80769,
  "A": 20000, "Iy": 66666667, "Iz": 66666667, "J": 50000000,
  "Asy": 16667, "Asz": 16667,
  "vec_xz": [0, 0, 1]
}
```

If `Asy`/`Asz` are omitted, the default `κ = 5/6` (rectangular solid section) is applied automatically.

Typical shear correction factors:

| Cross-section | κ |
|---------------|---|
| Rectangular solid | 5/6 ≈ 0.833 *(default)* |
| Circular solid | 0.9 |
| Thin-walled I (strong axis) | ~0.85 |
| Thin-walled I (weak axis) | ~0.40 |

**When to use Timoshenko instead of Euler-Bernoulli**: the shear correction becomes significant when the slenderness ratio `L/h` is small. As a rule of thumb:

| L/h | Shear contribution | Recommendation |
|-----|-------------------|----------------|
| > 20 | < 1 % | `elastic_beam3d` is sufficient |
| 10–20 | 1–5 % | either element |
| < 10 | > 5 % | use `timoshenko_beam3d` |

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

| Directory | Element Type | Load type | Description |
|-----------|-------------|-----------|-------------|
| `examples/cantilever_beam/` | `Hexa8` | Nodal | 3D cantilever beam with tip point load |
| `examples/single_hex8/` | `Hexa8` | Nodal | Single cube under uniaxial compression |
| `examples/tet4_patch/` | `Tet4` | Nodal | Tetrahedral patch test (constant stress) |
| `examples/bridge_load/` | `Truss3D` | Nodal | 2D Warren truss bridge under vertical load |
| `examples/frame_portal/` | `ElasticBeam3D` | Nodal | Portal frame under lateral (sway) load |
| `examples/beam3d/` | `ElasticBeam3D` | Nodal | Single 3D beam with end moment |
| `examples/beam_with_dist_load/` | `ElasticBeam3D` | `beam_dist` | Two-span beam with UDL 1 kN/m, fixed ends |
| `examples/pressure_on_cube/` | `Hexa8` | `surface_pressure` | Steel cube with uniform pressure on top face |
| `examples/gravity_solid/` | `Hexa8` | `body_force` | Concrete cube under self-weight (gravity) |
| `examples/ortho_cube/` | `Hexa8` | Nodal | Timber cube (orthotropic) under axial compression along grain |
| `examples/timoshenko_deep_beam/` | `TimoshenkoBeam3D` | Nodal | Deep cantilever (L/h=2): shear adds 16 % to EB deflection |

Each example directory contains a `problem.json` ready to POST to `/solve` and, where applicable, a `main.go` for programmatic use.

### Example: Portal Frame (`examples/frame_portal/problem.json`)

A 6000 mm × 3000 mm portal frame (N-mm-MPa), fixed bases, lateral load 10 kN at left column top. Expected sway ≈ 11 mm.

```json
{
  "nodes": [[0,0,0],[0,3000,0],[6000,3000,0],[6000,0,0]],
  "elements": [
    {"type":"elastic_beam3d","nodes":[0,1],
     "E":210000,"G":80769,"A":10000,"Iy":8333333,"Iz":8333333,"J":14062500,"vec_xz":[0,0,1]},
    {"type":"elastic_beam3d","nodes":[1,2],
     "E":210000,"G":80769,"A":10000,"Iy":8333333,"Iz":8333333,"J":14062500,"vec_xz":[0,0,1]},
    {"type":"elastic_beam3d","nodes":[3,2],
     "E":210000,"G":80769,"A":10000,"Iy":8333333,"Iz":8333333,"J":14062500,"vec_xz":[0,0,1]}
  ],
  "boundary_conditions": [
    {"node":0,"dofs":[0,1,2,3,4,5]},
    {"node":3,"dofs":[0,1,2,3,4,5]}
  ],
  "loads": [{"node":1,"dof":0,"value":10000}],
  "solver": "lu"
}
```

### Example: Truss Bridge (`examples/bridge_load/problem.json`)

A 5-node Warren truss (N-mm-MPa), 7 members, 100 kN midspan downward load:

```json
{
  "nodes": [[0,0,0],[5000,0,0],[10000,0,0],[2500,3000,0],[7500,3000,0]],
  "elements": [
    {"type":"truss3d","nodes":[0,1],"E":210000,"A":3000},
    {"type":"truss3d","nodes":[1,2],"E":210000,"A":3000},
    {"type":"truss3d","nodes":[0,3],"E":210000,"A":3000},
    {"type":"truss3d","nodes":[1,3],"E":210000,"A":3000},
    {"type":"truss3d","nodes":[1,4],"E":210000,"A":3000},
    {"type":"truss3d","nodes":[2,4],"E":210000,"A":3000},
    {"type":"truss3d","nodes":[3,4],"E":210000,"A":3000}
  ],
  "boundary_conditions": [
    {"node":0,"dofs":[0,1,2]},
    {"node":1,"dofs":[2]},
    {"node":2,"dofs":[1,2]},
    {"node":3,"dofs":[2]},
    {"node":4,"dofs":[2]}
  ],
  "loads": [{"node":1,"dof":1,"value":-100000}],
  "solver": "lu"
}
```

### Example: Timoshenko Deep Beam (`examples/timoshenko_deep_beam/problem.json`)

A 400 mm × 200 mm deep cantilever (L/h = 2), tip load 10 kN. Shear deformation contributes ~16 % to the total deflection:

```json
{
  "nodes": [[0,0,0],[400,0,0]],
  "elements": [
    {"type":"timoshenko_beam3d","nodes":[0,1],
     "E":210000,"G":80769,"A":20000,
     "Iy":66666667,"Iz":66666667,"J":50000000,
     "Asy":16667,"Asz":16667,"vec_xz":[0,0,1]}
  ],
  "boundary_conditions": [{"node":0,"dofs":[0,1,2,3,4,5]}],
  "loads": [{"node":1,"dof":1,"value":-10000}],
  "solver": "lu"
}
```

Expected tip deflection: **−0.01821 mm** (EB bending 0.01524 + shear 0.00297).

---

## Validation

`validation/main.go` runs a suite of problems with known closed-form solutions and verifies that the numerical result matches analytical theory to tight tolerances.

```bash
go run ./validation
```

### Results (March 11, 2026)

| Case | Element | Numerical | Theoretical | Rel. Err (%) | Status |
|------|---------|-----------|-------------|--------------|--------|
| Truss – axial deformation | `Truss3D` | 5.000e-04 | 5.000e-04 | 0.00e+00 | PASS ✓ |
| Beam – cantilever tip deflection | `ElasticBeam3D` | 3.333e-01 | 3.333e-01 | 0.00e+00 | PASS ✓ |
| Beam – simply-supported midspan | `ElasticBeam3D` | 1.667e-01 | 1.667e-01 | 0.00e+00 | PASS ✓ |
| Hexa8 – uniaxial patch test | `Hexa8` | 1.000e+00 | 1.000e+00 | 0.00e+00 | PASS ✓ |
| Truss – `AxialForce()` post-processing | `Truss3D` | 1.000e+04 | 1.000e+04 | 0.00e+00 | PASS ✓ |
| Beam – `EndForces()` Mz at support | `ElasticBeam3D` | 1.000e+00 | 1.000e+00 | 0.00e+00 | PASS ✓ |
| Hexa8 – `StressCentroid()` σxx | `Hexa8` | 1.000e+00 | 1.000e+00 | 1.11e-14 | PASS ✓ |

### Case descriptions

| Case | Theoretical formula | Parameters |
|------|--------------------|-----------------------|
| Truss axial deformation | $\delta_x = \frac{FL}{EA}$ | $F=10000\ \text{N},\ L=1\ \text{m},\ E=200000\ \text{N/mm}^2,\ A=100\ \text{mm}^2$ |
| Cantilever tip deflection | $\delta_y = \frac{FL^3}{3EI_z}$ | $F=1,\ L=1,\ E=1,\ I_z=1$ |
| Simply-supported midspan | $\delta_{\text{mid}} = \frac{FL^3}{48EI_z}$ | $F=1,\ L=2,\ E=1,\ I_z=1$ |
| Hexa8 uniaxial patch test | $u_x = \frac{\sigma_x}{E}\cdot L = 1$ | Unit cube, $E=1,\ \nu=0,\ F=1$ distributed over $x=1$ face |
| Truss `AxialForce()` | $N = \frac{EA}{L}\Delta u_{axial} = F$ | Same as truss deformation case |
| Beam `EndForces()` Mz | $M_z^{(i)} = F \cdot L$ | Cantilever, $F=1,\ L=1$ |
| Hexa8 `StressCentroid()` | $\sigma_{xx} = F/A = 1$ | Same as uniaxial patch case |

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
