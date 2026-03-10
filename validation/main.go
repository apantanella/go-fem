// Package main runs validation problems with known theoretical solutions.
//
// Each case computes a numerical result with go-fem and compares it against
// the closed-form analytical value. A relative-error tolerance is enforced.
//
// Usage:
//
//	go run ./validation
package main

import (
	"fmt"
	"math"
	"os"
	"strings"

	"go-fem/analysis"
	"go-fem/domain"
	"go-fem/element/frame"
	"go-fem/element/solid"
	"go-fem/element/truss"
	"go-fem/material"
	"go-fem/section"
	"go-fem/solver"

	"gonum.org/v1/gonum/mat"
)

// ─────────────────────────────────────────────────────────────────────────────
// Case 1 – Axial truss deformation
//
//   Single Truss3D element:
//     L = 1 m,  E = 200 000 N/mm²,  A = 100 mm²
//     Fixed at node 0 (UX, UY, UZ).
//     Axial force F = 10 000 N at node 1 in +X.
//
//   Theoretical (Hooke's law):
//     δ_x = F·L / (E·A) = 10000·1 / (200000·100) = 5×10⁻⁴
// ─────────────────────────────────────────────────────────────────────────────

func caseAxialTruss() (numerical, theoretical float64) {
	theoretical = 10000.0 * 1.0 / (200000.0 * 100.0) // 5e-4

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)
	n1 := dom.AddNode(1, 0, 0)

	coords := [2][3]float64{{0, 0, 0}, {1, 0, 0}}
	dom.AddElement(truss.NewTruss3D(0, [2]int{n0, n1}, coords, 200000, 100))

	dom.FixNode(n0)   // fix UX, UY, UZ at root
	dom.FixDOF(n1, 1) // UY at free end (no transverse stiffness in pure truss)
	dom.FixDOF(n1, 2) // UZ at free end
	dom.ApplyLoad(n1, 0, 10000)

	U := mustSolve(dom)
	numerical = dom.SetDisplacements(U)[n1][0] // UX at free node
	return
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 2 – Cantilever beam tip deflection (Euler-Bernoulli)
//
//   Single ElasticBeam3D element:
//     L = 1,  E = 1,  G = 0.5,  A = 1,  Iz = 1  (Iz governs XY bending)
//     Fully fixed at node 0 (all 6 DOFs).
//     Transverse force F = 1 in +UY at node 1.
//
//   Theoretical (Euler-Bernoulli cantilever):
//     δ_y = F·L³ / (3·E·Iz) = 1·1 / (3·1·1) = 1/3 ≈ 0.333…
// ─────────────────────────────────────────────────────────────────────────────

func caseCantileverBeam() (numerical, theoretical float64) {
	const (
		E  = 1.0
		Iz = 1.0
		L  = 1.0
		F  = 1.0
	)
	theoretical = F * L * L * L / (3 * E * Iz) // 1/3

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)
	n1 := dom.AddNode(L, 0, 0)

	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	sec := section.BeamSection3D{A: 1, Iy: 1, Iz: Iz, J: 1}
	dom.AddElement(frame.NewElasticBeam3D(0, [2]int{n0, n1}, coords, E, 0.5, sec, [3]float64{0, 0, 1}))

	dom.FixNodeAll(n0) // fix all 6 DOFs
	dom.ApplyLoad(n1, 1, F)

	U := mustSolve(dom)
	numerical = dom.SetDisplacements(U)[n1][1] // UY at free tip
	return
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 3 – Hexa8 uniaxial patch test
//
//   Single Hexa8 element, unit cube [0,1]³:
//     E = 1,  ν = 0  (no Poisson coupling)
//     BCs: UX=0 on x=0 face; UY=0 on y=0 face; UZ=0 on z=0 face.
//     Load: total force F = 1 in +X distributed uniformly to 4 nodes on x=1 face.
//
//   Theoretical (uniaxial bar):
//     σ_x = F/A = 1/1 = 1
//     ε_x = σ_x/E = 1
//     u_x(x=1) = ε_x·L = 1
//
//   A trilinear element reproduces linear displacement fields exactly.
// ─────────────────────────────────────────────────────────────────────────────

func caseHexa8Uniaxial() (numerical, theoretical float64) {
	theoretical = 1.0 // u_x at x=1 face

	dom := domain.NewDomain()

	// Nodes in hex8Ref order: (-1,-1,-1) → (0,0,0) etc.
	nids := [8]int{
		dom.AddNode(0, 0, 0), // 0 – x=0, y=0, z=0
		dom.AddNode(1, 0, 0), // 1 – x=1, y=0, z=0
		dom.AddNode(1, 1, 0), // 2 – x=1, y=1, z=0
		dom.AddNode(0, 1, 0), // 3 – x=0, y=1, z=0
		dom.AddNode(0, 0, 1), // 4 – x=0, y=0, z=1
		dom.AddNode(1, 0, 1), // 5 – x=1, y=0, z=1
		dom.AddNode(1, 1, 1), // 6 – x=1, y=1, z=1
		dom.AddNode(0, 1, 1), // 7 – x=0, y=1, z=1
	}
	coords := [8][3]float64{
		{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
		{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
	}
	mat3d := material.NewIsotropicLinear(1, 0)
	dom.AddElement(solid.NewHexa8(0, nids, coords, mat3d))

	// Prevent rigid-body motions
	for _, id := range []int{nids[0], nids[3], nids[4], nids[7]} { // x=0 face
		dom.FixDOF(id, 0) // UX = 0
	}
	for _, id := range []int{nids[0], nids[1], nids[4], nids[5]} { // y=0 face
		dom.FixDOF(id, 1) // UY = 0
	}
	for _, id := range []int{nids[0], nids[1], nids[2], nids[3]} { // z=0 face
		dom.FixDOF(id, 2) // UZ = 0
	}

	// Distribute total force F=1 equally to the 4 nodes of the x=1 face
	for _, id := range []int{nids[1], nids[2], nids[5], nids[6]} {
		dom.ApplyLoad(id, 0, 0.25)
	}

	U := mustSolve(dom)
	numerical = dom.SetDisplacements(U)[nids[1]][0] // UX at any x=1 node
	return
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 4 – Simply-supported beam midspan deflection
//
//   Two ElasticBeam3D elements, nodes at x = 0, L/2, L:
//     L = 2,  E = 1,  Iz = 1
//     Pinned at node 0 (UX, UY, UZ, RX, RZ); roller at node 2 (UY, UZ, RX).
//     Concentrated load F = 1 in +UY at midspan node 1.
//
//   Theoretical (Euler-Bernoulli simply-supported):
//     δ_mid = F·L³ / (48·E·Iz) = 1·8 / (48·1·1) = 1/6 ≈ 0.1667
// ─────────────────────────────────────────────────────────────────────────────

func caseSimplySupportedBeam() (numerical, theoretical float64) {
	const (
		L  = 2.0
		E  = 1.0
		Iz = 1.0
		F  = 1.0
	)
	theoretical = F * L * L * L / (48 * E * Iz) // 8/48 = 1/6

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)
	n1 := dom.AddNode(L/2, 0, 0)
	n2 := dom.AddNode(L, 0, 0)

	sec := section.BeamSection3D{A: 1, Iy: 1, Iz: Iz, J: 1}
	vxz := [3]float64{0, 0, 1}
	c01 := [2][3]float64{{0, 0, 0}, {L / 2, 0, 0}}
	c12 := [2][3]float64{{L / 2, 0, 0}, {L, 0, 0}}
	dom.AddElement(frame.NewElasticBeam3D(0, [2]int{n0, n1}, c01, E, 0.5, sec, vxz))
	dom.AddElement(frame.NewElasticBeam3D(1, [2]int{n1, n2}, c12, E, 0.5, sec, vxz))

	// Pin at node 0: fix UX, UY, UZ, RX, RY
	// RZ (bending rotation about Z) is left free — it is the pin rotation.
	// RY is fixed here to prevent lateral rigid-body rotation (no out-of-plane load).
	for _, dof := range []int{0, 1, 2, 3, 4} {
		dom.FixDOF(n0, dof)
	}
	// Roller at node 2: fix UY, UZ, RX (allow UX slide and bending rotation RZ)
	for _, dof := range []int{1, 2, 3} {
		dom.FixDOF(n2, dof)
	}

	dom.ApplyLoad(n1, 1, F)

	U := mustSolve(dom)
	numerical = dom.SetDisplacements(U)[n1][1] // UY at midspan
	return
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 5 – Truss AxialForce post-processing
//
//   Same setup as Case 1.  After solving, AxialForce() must return F = 10 000 N.
// ─────────────────────────────────────────────────────────────────────────────

func caseTrussAxialForce() (numerical, theoretical float64) {
	theoretical = 10000.0 // N

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)
	n1 := dom.AddNode(1, 0, 0)

	coords := [2][3]float64{{0, 0, 0}, {1, 0, 0}}
	el := truss.NewTruss3D(0, [2]int{n0, n1}, coords, 200000, 100)
	dom.AddElement(el)
	dom.FixNode(n0)
	dom.FixDOF(n1, 1)
	dom.FixDOF(n1, 2)
	dom.ApplyLoad(n1, 0, 10000)

	mustSolve(dom) // Update() is called inside Run()
	numerical = el.AxialForce()
	return
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 6 – Cantilever beam EndForces (moment at fixed support)
//
//   Same cantilever as Case 2 (L=1, E=1, Iz=1, F=1).
//   At node i (fixed end): |Mz| = F·L = 1  (theoretical bending moment).
//   At node j (free end):  |Mz| = 0 (free end — zero moment).
// ─────────────────────────────────────────────────────────────────────────────

func caseBeamEndMoment() (numerical, theoretical float64) {
	const (
		E  = 1.0
		Iz = 1.0
		L  = 1.0
		F  = 1.0
	)
	theoretical = F * L // moment at fixed support = 1

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)
	n1 := dom.AddNode(L, 0, 0)

	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	sec := section.BeamSection3D{A: 1, Iy: 1, Iz: Iz, J: 1}
	el := frame.NewElasticBeam3D(0, [2]int{n0, n1}, coords, E, 0.5, sec, [3]float64{0, 0, 1})
	dom.AddElement(el)
	dom.FixNodeAll(n0)
	dom.ApplyLoad(n1, 1, F)

	mustSolve(dom)
	ef := el.EndForces()
	numerical = math.Abs(ef.I[5]) // |Mz| at fixed end
	return
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 7 – Hexa8 centroidal stress
//
//   Same uniaxial patch as Case 3 (unit cube, E=1, ν=0, F=1).
//   Theoretical: σxx = F/A = 1,  all other components = 0.
// ─────────────────────────────────────────────────────────────────────────────

func caseHexa8Stress() (numerical, theoretical float64) {
	theoretical = 1.0 // σxx

	dom := domain.NewDomain()
	nids := [8]int{
		dom.AddNode(0, 0, 0),
		dom.AddNode(1, 0, 0),
		dom.AddNode(1, 1, 0),
		dom.AddNode(0, 1, 0),
		dom.AddNode(0, 0, 1),
		dom.AddNode(1, 0, 1),
		dom.AddNode(1, 1, 1),
		dom.AddNode(0, 1, 1),
	}
	coords := [8][3]float64{
		{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
		{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
	}
	mat3d := material.NewIsotropicLinear(1, 0)
	el := solid.NewHexa8(0, nids, coords, mat3d)
	dom.AddElement(el)
	for _, id := range []int{nids[0], nids[3], nids[4], nids[7]} {
		dom.FixDOF(id, 0)
	}
	for _, id := range []int{nids[0], nids[1], nids[4], nids[5]} {
		dom.FixDOF(id, 1)
	}
	for _, id := range []int{nids[0], nids[1], nids[2], nids[3]} {
		dom.FixDOF(id, 2)
	}
	for _, id := range []int{nids[1], nids[2], nids[5], nids[6]} {
		dom.ApplyLoad(id, 0, 0.25)
	}
	mustSolve(dom)
	numerical = el.StressCentroid()[0] // σxx
	return
}

func mustSolve(dom *domain.Domain) *mat.VecDense {
	ana := analysis.StaticLinearAnalysis{Dom: dom, Solver: solver.LU{}}
	U, err := ana.Run()
	if err != nil {
		panic(fmt.Sprintf("solver error: %v", err))
	}
	return U
}

// ─────────────────────────────────────────────────────────────────────────────
// Runner
// ─────────────────────────────────────────────────────────────────────────────

type testCase struct {
	name   string
	run    func() (numerical, theoretical float64)
	tolPct float64 // acceptable relative error in percent
}

func main() {
	cases := []testCase{
		{"Truss     – axial deformation", caseAxialTruss, 1e-8},
		{"Beam      – cantilever tip deflection", caseCantileverBeam, 1e-8},
		{"Beam      – simply-supported midspan", caseSimplySupportedBeam, 1e-8},
		{"Hexa8     – uniaxial patch test", caseHexa8Uniaxial, 1e-8},
		{"Truss     – AxialForce() post-processing", caseTrussAxialForce, 1e-8},
		{"Beam      – EndForces() Mz at support", caseBeamEndMoment, 1e-8},
		{"Hexa8     – StressCentroid() σxx", caseHexa8Stress, 1e-8},
	}

	sep := strings.Repeat("─", 88)
	fmt.Println(sep)
	fmt.Printf("  %-44s  %12s  %12s  %10s  %s\n",
		"Case", "Numerical", "Theoretical", "Rel.Err(%)", "")
	fmt.Println(sep)

	allPass := true
	for _, c := range cases {
		num, th := c.run()
		relErrPct := math.Abs(num-th) / math.Abs(th) * 100
		pass := relErrPct <= c.tolPct
		mark := "PASS ✓"
		if !pass {
			mark = "FAIL ✗"
			allPass = false
		}
		fmt.Printf("  %-44s  %12.6g  %12.6g  %10.2e  %s\n",
			c.name, num, th, relErrPct, mark)
	}

	fmt.Println(sep)
	if allPass {
		fmt.Println("  All validation cases PASSED.")
	} else {
		fmt.Println("  Some validation cases FAILED.")
		os.Exit(1)
	}
	fmt.Println(sep)
}
