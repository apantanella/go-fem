# go-fem

A 3D structural **Finite Element Method** library and HTTP API server written in pure Go, inspired by the modular architecture of [OpenSees](https://opensees.berkeley.edu/).

## Overview

`go-fem` solves linear static and dynamic structural problems via a layered architecture that mirrors OpenSees: materials → elements → domain → analysis. Problems are defined programmatically through the Go API or submitted as JSON to the built-in HTTP server.

**Key capabilities:**
- 2D and 3D solid, truss, frame (Euler-Bernoulli and Timoshenko), shell, and spring/connector elements
- Isotropic and orthotropic linear elastic materials
- Automatic DOF detection (2, 3, or 6 per node) based on element types
- DOF-type-aware global stiffness assembly
- Five linear solvers: `Cholesky` (dense SPD), `LU` (dense general), `SkylineLDL` (sparse banded), `CG` (iterative SPD), `GMRES` (iterative general/non-symmetric)
- **Modal (free-vibration) analysis**: natural frequencies, mode shapes, participation factors, effective mass ratios — via generalized eigenvalue problem K·φ = ω²·M·φ
- Consistent mass matrices for all element types (truss, frame, solid, shell)
- Multiple load types: nodal forces, surface pressure, beam UDL, body force/gravity
- Non-zero prescribed displacements (imposed settlement)
- Optional `"dimensions"` field (`"2D"` / `"3D"`) validates element compatibility at solve time
- JSON HTTP API for language-agnostic integration, including `GET /elements` grouped by dimension
- Consistent N-mm-MPa unit system throughout all examples
- All element type strings use explicit `_2d`/`_3d` suffix; legacy names (without suffix) remain as backward-compatible aliases

---

## Architecture

Four-layer design mirroring OpenSees:

```
Layer 1 – material/     Constitutive models          (Material3D interface)
Layer 2 – element/      Finite elements               (Element interface)
Layer 3 – domain/       Mesh, assembly, BCs           (Domain struct)
Layer 4 – analysis/     Solution strategy + solver/   (StaticLinearAnalysis, ModalAnalysis)
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

Elements that support dynamic analysis additionally implement the optional `MassMatrixAssembler` interface:

```go
type MassMatrixAssembler interface {
    GetMassMatrix(rho float64) *mat.Dense // consistent element mass matrix Me
}
```

| Element | Mass formulation |
|---------|----------------|
| `Truss2D`, `Truss3D` | ρAL/6·[2I, I; I, 2I] — consistent bar (frame-invariant) |
| `ElasticBeam2D` | ρAL/420 Hermitian (axial + bending), rotated to global |
| `ElasticBeam3D` | ρAL/420 Hermitian (axial, torsion, bending xy+xz), rotated to global |
| `Tet4` | Analytical: ρV/20·(1+δₙₘ) per DOF direction — exact for linear tet |
| `Hexa8` | 2×2×2 Gauss integration of ρ·N·Nᵀ |

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
| `element/quad/` | `Quad8` | 8 | 16 | 3×3 Gauss | Serendipity 8-node quad, plane stress or strain — accurate near stress concentrations |
| `element/quad/` | `Tri3` | 3 | 6 | Exact (constant B) | CST — constant strain triangle, plane stress/strain |
| `element/quad/` | `Tri6` | 6 | 12 | 3-pt triangle | LST — linear strain triangle, plane stress/strain |
| `element/shell/` | `ShellMITC4` | 4 | 24 | 2×2 + SRI | Flat shell: membrane + bending, 6 DOF/node |
| `element/shell/` | `DiscreteKirchhoffTriangle` (`DKT3`) | 3 | 18 | Area-constant | Thin plate bending (Discrete Kirchhoff), active DOFs: UZ/RX/RY per node |

### Connector Elements

| Package | Type | Nodes | DOFs | DOF/node | JSON `"type"` | Notes |
|---------|------|-------|------|----------|--------------|-------|
| `element/zerolength/` | `ZeroLength` | 2 | 12 | 6 (UX..RZ) | `zerolength_3d` | Full 6-DOF spring connector in 3D |
| `element/zerolength/` | `ZeroLength3DOF` | 2 | 6 | 3 (UX,UY,UZ) | `zerolength_trans_3d` | 3D translational spring (no rotation DOFs) |
| `element/zerolength/` | `ZeroLength2D` | 2 | 4 | 2 (UX,UY) | `zerolength_2d` | 2D translational spring connector |
| `element/zerolength/` | `ZeroLength2DFrame` | 2 | 6 | 3 (UX,UY,RZ) | `zerolength_frame_2d` | 2D plane-frame spring connector (UX+UY+RZ) |

All spring/connector elements output a `spring_forces` object in the response (see response format below). Spring stiffnesses are passed in the `"springs"` JSON array — only the first N entries are used for an N-DOF-per-node element.

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

// Assemble global mass matrix (for modal analysis)
masses := []domain.ElementMass{
    {ElemIdx: 0, Rho: 7850}, // element 0, steel density kg/m³
}
dom.AssembleMassMatrix(masses)

// Query free DOFs (not constrained by Dirichlet BCs)
freeDOFs := dom.FreeDOFs() // []int — global DOF indices

// Query DOF type at a global DOF index (UX, UY, UZ, RX, RY, RZ)
dt := dom.DOFTypeAt(globalDOFIdx) // dof.Type
```

**DOF auto-detection**: `Assemble()` scans all elements and sets `DOFPerNode = max(DOFPerNode across elements)`. Pure plane-stress/strain models (Tri3, Tri6, Quad4) and Truss2D use 2 DOF/node; solid/truss-3D models use 3 DOF/node; plane frames (ElasticBeam2D, TimoshenkoBeam2D) use 3 DOF/node; models with 3D beams or shells use 6 DOF/node.

**Dirichlet BCs**: applied by the penalty-free row/column zeroing method (K[gdof,:] = 0, K[:,gdof] = 0, K[gdof,gdof] = 1, F[gdof] = value). Non-zero prescribed displacements set `bc.Value ≠ 0`.

### Load types

| Method | Description | Supported elements |
|--------|-------------|-------------------|
| `ApplyLoad(node, dof, value)` | Concentrated force or moment | All |
| `AddBeamDistLoad(elemIdx, dir, intensity)` | UDL (N/m), work-equivalent nodal forces | `ElasticBeam3D`, `ElasticBeam2D`, `TimoshenkoBeam3D`, `TimoshenkoBeam2D` |
| `AddSurfacePressure(faceNodes[4], P)` | Uniform pressure on quad face (2×2 Gauss) | Any 4 nodes |
| `AddBodyForce(elemIdx, rho, g)` | Gravity/body force | **All elements** (see table below) |

Elements implement optional load interfaces:
- `element.EquivalentNodalLoader` — converts distributed load to nodal forces
- `element.BodyForceLoader` — computes body-force nodal contributions (now implemented by **all** element types)

#### Body-force implementation per element family

| Family | Elements | Method |
|--------|----------|--------|
| **3D solids** | `Tet4`, `Tet10`, `Hexa8`, `Brick20` | Gauss quadrature — same rule as `formKe` |
| **Truss/bar** | `Truss2D`, `Truss3D`, `CorotTruss` | `ρ·A·L/2·g` lumped equally at both nodes |
| **Beam/frame** | `ElasticBeam2D`, `ElasticBeam3D`, `TimoshenkoBeam2D`, `TimoshenkoBeam3D` | Delegates to `EquivalentNodalLoad(g, ρ·A)` — UDL fixed-end forces |
| **Shell** | `ShellMITC4`, `DKT3` | 2×2 Gauss (MITC4) / area/3 lumped per node (DKT3) — translational DOFs only |
| **2D plane** | `Quad4`, `Quad8`, `Tri3`, `Tri6` | Same Gauss rule as `formKe` — in-plane (X,Y) components only |
| **ZeroLength** | all four variants | Returns zero vector (no physical extent) |

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

### ModalAnalysis

Free-vibration (eigenvalue) analysis — natural frequencies, mode shapes, participation factors:

1. Assemble global `K` and `M`
2. Identify free DOFs (not constrained by Dirichlet BCs)
3. Extract reduced `K_red` and `M_red` submatrices
4. Solve K·φ = ω²·M·φ via Cholesky transformation → standard symmetric eigenvalue problem
5. Expand mode shapes to full DOF space and compute participation factors

```go
a := &analysis.ModalAnalysis{
    Dom:      dom,
    Masses:   []domain.ElementMass{{ElemIdx: 0, Rho: 7850}},
    NumModes: 10, // number of modes to extract (default 10)
}
res, err := a.Run()

// Results
res.NumModes              // int — number of modes extracted
res.Omega2[k]             // float64 — ω² [rad²/s²] for mode k (ascending)
res.Frequencies[k]        // float64 — f [Hz]
res.Periods[k]            // float64 — T = 1/f [s]
res.ModeShapes[k][dof]    // float64 — M-normalised mode shape (φᵀ·M·φ = 1)
res.ParticipationFactors[k][d] // float64 — Γ_{k,d} = φₖᵀ·M·r_d (d=0:X,1:Y,2:Z)
res.EffectiveMass[k][d]        // float64 — fraction 0–1: Γ²_{k,d}/m_total_d
res.CumulativeEffectiveMass[k][d] // float64 — cumulative sum over modes 0..k
res.AngularFrequency(k)   // float64 — ω [rad/s]
```

**DOF type conventions for `FixDOF`**: always pass the `dof.Type` enum value cast to `int`, not the DOF offset within the element. For example, `RZ` has enum value 5 regardless of whether the element uses 3 or 6 DOF/node:

```go
dom.FixDOF(nodeID, int(dof.UX))  // = 0
dom.FixDOF(nodeID, int(dof.UY))  // = 1
dom.FixDOF(nodeID, int(dof.UZ))  // = 2
dom.FixDOF(nodeID, int(dof.RX))  // = 3
dom.FixDOF(nodeID, int(dof.RY))  // = 4
dom.FixDOF(nodeID, int(dof.RZ))  // = 5
```

**Completeness**: effective mass fractions sum to 100% per direction only when all free-DOF modes are extracted (`NumModes = len(freeDOFs)`). The computation is performed on the reduced system to satisfy this condition exactly.

#### Example: Cantilever Beam Modal Analysis

```go
dom := domain.NewDomain()
n0 := dom.AddNode(0, 0, 0)
n1 := dom.AddNode(2000, 0, 0) // L = 2000 mm

sec := section.BeamSection2D{A: 100, Iz: 833.33}
elem := frame.NewElasticBeam2D(0, [2]int{n0, n1}, coords, 210000, sec)
dom.AddElement(elem)

// Clamp node 0 using enum values
dom.FixDOF(n0, int(dof.UX))
dom.FixDOF(n0, int(dof.UY))
dom.FixDOF(n0, int(dof.RZ))

res, err := (&analysis.ModalAnalysis{
    Dom:      dom,
    Masses:   []domain.ElementMass{{ElemIdx: 0, Rho: 7.85e-6}}, // kg/mm³
    NumModes: 3,
}).Run()

fmt.Printf("f₁ = %.4f Hz\n", res.Frequencies[0])
```

### Linear Solvers

All linear solvers implement `solver.LinearSolver`:
```go
type LinearSolver interface {
    Solve(K *mat.Dense, F *mat.VecDense) (*mat.VecDense, error)
}
```

| JSON `"solver"` | Go type | Kind | Use when |
|----------------|---------|------|----------|
| `"cholesky"` *(default)* | `solver.Cholesky{}` | Dense direct | SPD K — most structural problems |
| `"lu"` | `solver.LU{}` | Dense direct | General (non-SPD); models with ZeroLength springs |
| `"skyline"` | `solver.SkylineLDL{}` | Sparse direct | Banded SPD K — faster than dense Cholesky when bandwidth ≪ n |
| `"cg"` | `solver.CG{}` | Iterative | Large SPD systems; O(n·bandwidth) per iteration |
| `"gmres"` | `solver.GMRES{}` | Iterative | Non-symmetric or indefinite systems; configurable restart |

#### Solver tuning via `solver_options`

Iterative and sparse solvers accept optional parameters through the `"solver_options"` JSON object:

| Field | Type | Applies to | Description |
|-------|------|------------|-------------|
| `"tol"` | float64 | `cg`, `gmres` | Relative residual tolerance `‖r‖/‖F‖` (default `1e-10`) |
| `"max_iter"` | int | `cg`, `gmres` | Maximum iterations / outer restarts (default: `3n` for CG, `⌈3n/restart⌉` for GMRES) |
| `"restart"` | int | `gmres` | Krylov subspace dimension per restart cycle (default `50`) |
| `"zero_tol"` | float64 | `skyline` | Off-diagonal sparsity threshold relative to max diagonal entry (default `1e-14`) |

```go
// Dense direct (default)
slv := solver.Cholesky{}

// Sparse direct — skyline LDL^T
slv = solver.SkylineLDL{} // or: SkylineLDL{ZeroTol: 1e-12}

// Iterative — Conjugate Gradient (SPD)
slv = solver.CG{Tol: 1e-12, MaxIter: 500}

// Iterative — GMRES (general / non-symmetric)
slv = solver.GMRES{Tol: 1e-10, Restart: 30}
```

### Eigenvalue Solver

The `solver` package also provides the generalized eigenvalue solver used by `ModalAnalysis`:

```go
// Solve K·φ = ω²·M·φ for the numModes smallest eigenvalues.
// K and M must be the reduced (free-DOF) matrices — symmetric, positive definite.
res, err := solver.SolveGeneralizedEigen(Kred, Mred, numModes)
// res.Omega2  []float64    — eigenvalues ω² [rad²/s²] in ascending order
// res.Modes   *mat.Dense   — (nFree × numModes), M-normalised columns

// Utility functions
f := solver.FrequencyHz(omega2)    // ω² → f [Hz]
T := solver.PeriodSeconds(omega2)  // ω² → T [s]
```

**Algorithm** (Bathe §10.2):
1. Cholesky-factorize M = L·Lᵀ
2. Form A = L⁻¹·K·L⁻ᵀ (symmetric)
3. Solve the standard symmetric problem A·y = ω²·y via `gonum mat.EigenSym`
4. Back-transform φ = L⁻ᵀ·y (M-normalised: φᵀ·M·φ = 1)

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
| `POST` | `/solve` | Submit a FEM problem (JSON body), receive displacements and element forces |
| `GET` | `/health` | Health check: `{"status":"ok","service":"go-fem"}` |
| `GET` | `/elements` | List supported elements grouped by dimension (`"2D"` / `"3D"`) |
| `GET` | `/` | API info: version, supported elements, materials, solvers |

### POST /solve – Request Format

```json
{
  "dimensions": "2D",
  "materials": [
    {"id": "steel", "type": "isotropic_linear", "E": 200000, "nu": 0.3}
  ],
  "nodes": [
    [0, 0, 0],
    [1, 0, 0]
  ],
  "elements": [
    {"type": "truss_3d", "nodes": [0, 1], "E": 200000, "A": 0.01}
  ],
  "boundary_conditions": [
    {"node": 0, "dofs": [0, 1, 2]},
    {"node": 1, "dofs": [2], "values": [0.005]}
  ],
  "loads": [
    {"node": 1, "dof": 0, "value": 5000}
  ],
  "solver": "cg",
  "solver_options": {"tol": 1e-12, "max_iter": 1000}
}
```

**`"dimensions"`**: optional `"2D"` or `"3D"`. When set, every element in the problem is validated to belong to that dimensionality and an error is returned on mismatch. Echoed back in the `info` response field.

**Element naming**: all element types use an explicit `_2d` or `_3d` suffix (e.g. `"truss_3d"`, `"elastic_beam_2d"`). Legacy names without a suffix (e.g. `"truss3d"`, `"elastic_beam3d"`) are accepted as backward-compatible aliases.

**`"solver"`**: `"cholesky"` (default) \| `"lu"` \| `"skyline"` \| `"cg"` \| `"gmres"`

**`"solver_options"`** (optional): tuning for iterative / sparse solvers. See [Solver tuning](#solver-tuning-via-solver_options) above.

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
    "solver": "cholesky",
    "dimensions": "3D"
  },
  "displacements": [
    {"node": 0, "ux": 0.0, "uy": 0.0, "uz": 0.0},
    {"node": 1, "ux": 2.5e-4, "uy": 0.0, "uz": 0.0}
  ],
  "element_forces": [
    {"id": 0, "type": "truss_3d", "N": 5000.0, "sigma": 50.0}
  ],
  "summary": {
    "max_abs_displacement": {"node": 1, "component": "ux", "value": 2.5e-4}
  },
  "elapsed_ms": 1.3
}
```

`element_forces` is always populated. The fields present depend on element type:

| Field | Element types |
|-------|---------------|
| `N`, `sigma` | `truss_3d`, `corot_truss_3d`, `truss_2d` — axial force and stress |
| `end_i`, `end_j` | beam elements — 6-component cross-section forces in local frame (N, Vy, Vz, Mx, My, Mz) |
| `stress` | solid elements (`tet4_3d`, `hexa8_3d`, …) and 2D quads — centroidal Cauchy stress + von Mises |
| `shell_forces` | `shell_mitc4_3d` — in-plane resultants Nx, Ny, Nxy and bending moments Mx, My, Mxy per unit length |
| `spring_forces` | all `zerolength_*` types — spring forces/moments (tension positive). Fields: `Fx`, `Fy`, `Fz`, `Mx`, `My`, `Mz`; only active components are included |

Example `spring_forces` for a `zerolength_frame_2d` (UX + UY + RZ springs):

```json
{"id": 5, "type": "zerolength_frame_2d", "spring_forces": {"Fx": 120.5, "Fy": -80.0, "Mz": 0.0}}
```

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

All types use an explicit `_2d` or `_3d` suffix. The bare names (e.g. `"tet4"`, `"truss3d"`) are accepted as backward-compatible aliases.

| `"type"` | Alias | Nodes | Required fields | Notes |
|----------|-------|-------|-----------------|-------|
| `"tet4_3d"` | `"tet4"` | 4 | `material` | |
| `"hexa8_3d"` | `"hexa8"` | 8 | `material` | |
| `"tet10_3d"` | `"tet10"` | 10 | `material` | |
| `"brick20_3d"` | `"brick20"` | 20 | `material` | |
| `"truss_3d"` | `"truss3d"` | 2 | `E`, `A` | 3 DOF/node (UX,UY,UZ) |
| `"truss_2d"` | `"truss2d"` | 2 | `E`, `A` | 2 DOF/node (UX,UY) |
| `"corot_truss_3d"` | `"corot_truss"` | 2 | `E`, `A` | Geometrically nonlinear, 3 DOF/node |
| `"elastic_beam_3d"` | `"elastic_beam3d"` | 2 | `E`, `G`, `A`, `Iy`, `Iz`, `J`, `vec_xz` | 6 DOF/node |
| `"elastic_beam_2d"` | `"elastic_beam2d"` | 2 | `E`, `A`, `Iz` | 3 DOF/node (UX,UY,RZ) |
| `"timoshenko_beam_3d"` | `"timoshenko_beam3d"` | 2 | `E`, `G`, `A`, `Iy`, `Iz`, `J`, `vec_xz` | optional: `Asy`, `Asz`; 6 DOF/node |
| `"timoshenko_beam_2d"` | `"timoshenko_beam2d"` | 2 | `E`, `G`, `A`, `Iz` | optional: `Asy`; 3 DOF/node |
| `"shell_mitc4_3d"` | `"shell_mitc4"` | 4 | `E`, `nu`, `thickness` | 6 DOF/node |
| `"dkt3_3d"` | `"dkt3"`, `"discrete_kirchhoff_triangle"` | 3 | `E`, `nu`, `thickness` | active DOFs: UZ/RX/RY; 6 DOF/node global layout |
| `"quad4_2d"` | `"quad4"` | 4 | `E`, `nu`, `thickness` | `plane_type`: `"stress"` (default) or `"strain"` |
| `"quad8_2d"` | `"quad8"` | 8 | `E`, `nu`, `thickness` | 3×3 Gauss; `plane_type`: `"stress"` or `"strain"` |
| `"tri3_2d"` | `"tri3"` | 3 | `E`, `nu`, `thickness` | CST; `plane_type`: `"stress"` or `"strain"` |
| `"tri6_2d"` | `"tri6"` | 6 | `E`, `nu`, `thickness` | LST; `plane_type`: `"stress"` or `"strain"` |
| `"zerolength_3d"` | `"zerolength"` | 2 | `springs[0..5]`: [kUX,kUY,kUZ,kRX,kRY,kRZ] | 6 DOF/node, all 6 spring stiffnesses |
| `"zerolength_trans_3d"` | — | 2 | `springs[0..2]`: [kUX,kUY,kUZ] | 3 DOF/node, translational only |
| `"zerolength_2d"` | — | 2 | `springs[0..1]`: [kUX,kUY] | 2 DOF/node, 2D translational |
| `"zerolength_frame_2d"` | — | 2 | `springs[0..2]`: [kUX,kUY,kRZ] | 3 DOF/node, 2D plane-frame |

Solid elements require a `"material"` string referencing an entry in `"materials"`. All other elements take direct `E`, `G`, etc. parameters.

Spring stiffness arrays are always JSON `"springs": [k0, k1, …, k5]` — unused entries beyond the element's DOF count are ignored.

#### Timoshenko beam — shear correction

`timoshenko_beam3d` and `timoshenko_beam2d` share the same DOF layout and JSON fields as their Euler-Bernoulli counterparts but account for transverse shear deformation via the shear-flexibility parameter:

```
Φ = 12·E·I / (G·As·L²)
```

where `As = κ·A` is the effective shear area (`κ` = shear correction factor). The stiffness matrix reduces to the Euler-Bernoulli matrix when `Φ → 0` (rigid shear, slender beam).

Shear areas in JSON (`Asy` for bending in the x-y plane, `Asz` for the x-z plane — 3D only):

```json
{
  "type": "timoshenko_beam_3d",
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
| > 20 | < 1 % | `elastic_beam_3d` is sufficient |
| 10–20 | 1–5 % | either element |
| < 10 | > 5 % | use `timoshenko_beam_3d` |

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
| `examples/cantilever_beam/` | `hexa8_3d` | Nodal | 3D cantilever beam with tip point load |
| `examples/single_hex8/` | `hexa8_3d` | Nodal | Single cube under uniaxial compression |
| `examples/tet4_patch/` | `tet4_3d` | Nodal | Tetrahedral patch test (constant stress) |
| `examples/bridge_load/` | `truss_3d` | Nodal | 2D Warren truss bridge under vertical load |
| `examples/frame_portal/` | `elastic_beam_3d` | Nodal | 3D portal frame under lateral (sway) load |
| `examples/plane_frame/` | `elastic_beam_2d` | Nodal | 2D portal frame with gravity UDL on crossbeam |
| `examples/truss2d/` | `truss_2d` | Nodal | Simple 2-bar 2D truss under vertical load |
| `examples/beam3d/` | `elastic_beam_3d` | Nodal | Single 3D beam with end moment |
| `examples/beam_with_dist_load/` | `elastic_beam_3d` | `beam_dist` | Two-span beam with UDL 1 kN/m, fixed ends |
| `examples/pressure_on_cube/` | `hexa8_3d` | `surface_pressure` | Steel cube with uniform pressure on top face |
| `examples/gravity_solid/` | `hexa8_3d` | `body_force` | Concrete cube under self-weight (gravity) |
| `examples/ortho_cube/` | `hexa8_3d` | Nodal | Timber cube (orthotropic) under axial compression along grain |
| `examples/timoshenko_deep_beam/` | `timoshenko_beam_3d` | Nodal | Deep cantilever (L/h=2): shear adds 16 % to EB deflection |
| `examples/dkt3_tri_plate/` | `dkt3_3d` | Nodal | Single DKT3 triangle: clamped edge, point load + UDL equiv. variants |
| `examples/tri_plane/` | `tri3_2d`, `tri6_2d` | Nodal | Plane-stress cantilever meshed with CST/LST triangles |
| `examples/mixed_3d_building/` | `hexa8_3d`, `elastic_beam_3d`, `truss_3d`, `shell_mitc4_3d`, `dkt3_3d` | Nodal + `body_force` | 3D building with 5 element types: concrete foundation, steel frame, truss bracing, roof shell |

Each example directory contains a `problem.json` ready to POST to `/solve` and, where applicable, a `main.go` for programmatic use.

### Example: Portal Frame (`examples/frame_portal/problem.json`)

A 6000 mm × 3000 mm portal frame (N-mm-MPa), fixed bases, lateral load 10 kN at left column top. Expected sway ≈ 11 mm.

```json
{
  "dimensions": "3D",
  "nodes": [[0,0,0],[0,3000,0],[6000,3000,0],[6000,0,0]],
  "elements": [
    {"type":"elastic_beam_3d","nodes":[0,1],
     "E":210000,"G":80769,"A":10000,"Iy":8333333,"Iz":8333333,"J":14062500,"vec_xz":[0,0,1]},
    {"type":"elastic_beam_3d","nodes":[1,2],
     "E":210000,"G":80769,"A":10000,"Iy":8333333,"Iz":8333333,"J":14062500,"vec_xz":[0,0,1]},
    {"type":"elastic_beam_3d","nodes":[3,2],
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
  "dimensions": "3D",
  "nodes": [[0,0,0],[5000,0,0],[10000,0,0],[2500,3000,0],[7500,3000,0]],
  "elements": [
    {"type":"truss_3d","nodes":[0,1],"E":210000,"A":3000},
    {"type":"truss_3d","nodes":[1,2],"E":210000,"A":3000},
    {"type":"truss_3d","nodes":[0,3],"E":210000,"A":3000},
    {"type":"truss_3d","nodes":[1,3],"E":210000,"A":3000},
    {"type":"truss_3d","nodes":[1,4],"E":210000,"A":3000},
    {"type":"truss_3d","nodes":[2,4],"E":210000,"A":3000},
    {"type":"truss_3d","nodes":[3,4],"E":210000,"A":3000}
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
  "dimensions": "3D",
  "nodes": [[0,0,0],[400,0,0]],
  "elements": [
    {"type":"timoshenko_beam_3d","nodes":[0,1],
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

### Unit Tests

Package-level Go tests cover the solver layer independently of the full FEM pipeline:

```bash
go test ./solver/...
```

#### `solver/linear_solver_test.go`

| Test | What is verified |
|------|-----------------|
| `TestSolversAgainstReference` | All five linear solvers (`LU`, `SkylineLDL`, `CG`, `GMRES`) produce a solution that agrees with the dense Cholesky reference and satisfies `‖K·u − F‖/‖F‖ < 1e-8` on a 20-DOF tridiagonal SPD system |
| `TestCGNotSPD` | CG returns an error for an indefinite matrix (negative diagonal entry) rather than silently diverging |
| `TestGMRESNonSymmetric` | GMRES solves a non-symmetric 4×4 system to residual `< 1e-8` |

#### `solver/eigen_test.go`

| Test | What is verified |
|------|-----------------|
| `TestSolveGeneralizedEigen_EigenvaluesIdentityMass` | Returned ω² match the analytically known values `{1, 3}` for K=tridiagonal-2×2, M=I (tolerance 1e-10); ascending-sort invariant is also checked |
| `TestSolveGeneralizedEigen_EigenResidual` | Dynamic equilibrium residual `‖K·φₖ − ω²ₖ·M·φₖ‖₂ < 1e-10` for every mode |
| `TestSolveGeneralizedEigen_MOrthonormality` | M-normalization (φᵢᵀ·M·φᵢ = 1) and M-orthogonality (φᵢᵀ·M·φⱼ = 0, i≠j) within 1e-10 |
| `TestSolveGeneralizedEigen_NonIdentityMass` | Non-identity diagonal M: analytical ω² = {1, 9/4} recovered from 4ω⁴−13ω²+9=0; residual and M-orthonormality also verified |
| `TestSolveGeneralizedEigen_NumModesSubset` | Requesting numModes < n returns exactly the right number of eigenvalues and mode columns, with the single mode satisfying the dynamic residual |
| `TestSolveGeneralizedEigen_MassNotPD` | A zero (singular) mass matrix triggers a descriptive error rather than a panic or silent NaN |
| `TestExtractSubmatrix_Basic` | Correct entry selection from a 4×4 matrix for free DOF indices {1, 3} (mirrors the reduction step before the eigenvalue solve) |
| `TestExtractSubmatrix_AllDOFs` | Selecting all DOF indices is a no-op — extracted matrix equals the original entry-by-entry |
| `TestExpandModes_Basic` | Reduced mode columns are placed at the correct global DOF rows; constrained DOF rows remain zero (mirrors the expansion step after the eigenvalue solve) |
| `TestFrequencyHz` | Conversion f = √ω²/(2π): exact values at ω²=4π² (1 Hz) and ω²=π² (0.5 Hz); zero-frequency guard (ω²=0→0 Hz); negative-ω² guard (→0 Hz, prevents NaN) |
| `TestPeriodSeconds` | Conversion T = 1/f: T=1 s at ω²=4π², T=2 s at ω²=π², T=+∞ for rigid-body zero eigenvalue |

### Results (March 12, 2026)

| Case | Element | Numerical | Theoretical | Rel. Err (%) | Status |
|------|---------|-----------|-------------|--------------|--------|
| Truss – axial deformation | `Truss3D` | 5.000e-04 | 5.000e-04 | 0.00e+00 | PASS ✓ |
| Beam – cantilever tip deflection | `ElasticBeam3D` | 3.333e-01 | 3.333e-01 | 0.00e+00 | PASS ✓ |
| Beam – simply-supported midspan | `ElasticBeam3D` | 1.667e-01 | 1.667e-01 | 0.00e+00 | PASS ✓ |
| Hexa8 – uniaxial patch test | `Hexa8` | 1.000e+00 | 1.000e+00 | 0.00e+00 | PASS ✓ |
| Truss – `AxialForce()` post-processing | `Truss3D` | 1.000e+04 | 1.000e+04 | 0.00e+00 | PASS ✓ |
| Beam – `EndForces()` Mz at support | `ElasticBeam3D` | 1.000e+00 | 1.000e+00 | 0.00e+00 | PASS ✓ |
| Hexa8 – `StressCentroid()` σxx | `Hexa8` | 1.000e+00 | 1.000e+00 | 1.11e-14 | PASS ✓ |
| Mixed – large 3D model convergence | 5 element types (87 elements, 48 nodes, 288 DOFs) | 1.000e+00 | 1.000e+00 | 0.00e+00 | PASS ✓ |

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
| Mixed 3D model | Convergence check: numerical result = 1 when solver succeeds | 3D building — 9 Hexa8 (foundation) + 40 ElasticBeam3D (columns/beams) + 24 Truss3D (bracing) + 4 ShellMITC4 + 10 DKT3 (roof); 50 kN/node gravity on 16 roof nodes + Hexa8 body forces |

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
The package already ships five implementations: `Cholesky`, `LU`, `SkylineLDL`, `CG`, and `GMRES`. Potential extensions: AMG preconditioned CG, sparse LU (CSR format), GPU-accelerated solvers.

### Adding an Analysis Strategy

`ModalAnalysis` (eigenvalue analysis) is fully implemented — see [Layer 4 – Analysis & Solvers](#layer-4--analysis--solvers).

Planned extensions:
- **Nonlinear static** (Newton-Raphson with load stepping)
- **Dynamic** (Newmark-β time integration)

---

## License

MIT – see [LICENSE](LICENSE).
