package main_test

// Validation tests for a simply supported steel beam under four load models.
//
// Reference: engineering benchmark document (manual in daN/cm units).
//
// Beam properties:
//
//	E  = 2 100 000 daN/cm²
//	Iz = 67 500 cm⁴
//	A  = 100 cm²  (arbitrary; does not affect bending response)
//	L  = 500 cm
//
// Support conditions: pin at node 0 (UX + UY fixed), roller at last node (UY fixed).
//
// All four models use the elastic_beam_2d element.  FEM reactions are obtained
// via the global force balance:
//
//	R_constrained = K_orig · U − F_ext
//
// (K_orig and F_ext are saved immediately after Assemble() and before
// ApplyDirichletBC() which zeros out the constrained rows.)
//
// ┌──────┬────────────────────────────────────────────┬──────────┬──────────┬────────────────┐
// │ Model│ Load                                        │  R_A     │  R_B     │  M_max         │
// ├──────┼────────────────────────────────────────────┼──────────┼──────────┼────────────────┤
// │  A   │ Triangular 0→q over full span              │ +1 250   │ +2 500   │  240 562.6     │
// │  B   │ Triangular 0→q over second half [L/2, L]   │  +312.5  │ +1 562.5 │   99 387.9     │
// │  C   │ P↓ at b=150  +  P↑ at b+c=350 (antisym.)  │ +1 200   │ −1 200   │  180 000       │
// │  D   │ P↓ at b=150  +  P↓ at b+c=350 (sym.)      │ +3 000   │ +3 000   │  450 000       │
// └──────┴────────────────────────────────────────────┴──────────┴──────────┴────────────────┘
// Reactions in daN, moments in daN·cm.  M_max for A and B is at an interior cross-section;
// for C the absolute maximum is at x=b (antisymmetric, −M_max at x=b+c);
// for D the moment is constant = M_max between the two load points.
//
// Tolerance: reactions and moments to within 1 × 10⁻⁶ relative analytical value.

import (
	"math"
	"testing"

	"go-fem/domain"
	"go-fem/element/frame"
	"go-fem/section"
	"go-fem/solver"

	"gonum.org/v1/gonum/mat"
)

// ─── shared constants ─────────────────────────────────────────────────────────

const (
	ssbE  = 2_100_000.0 // daN/cm²
	ssbIz = 67_500.0    // cm⁴
	ssbA  = 100.0       // cm²  (dummy, not relevant for bending)
	ssbL  = 500.0       // cm
	ssbq  = 15.0        // daN/cm  (distributed load intensity)
	ssbP  = 3_000.0     // daN    (concentrated load)
	ssbA2 = 250.0       // cm  (a: half-span length)
	ssbB  = 150.0       // cm  (distance to first concentrated load)
	ssbC  = 200.0       // cm  (distance between concentrated loads)
)

// ssbSec returns the beam cross-section for all models.
func ssbSec() section.BeamSection2D {
	return section.BeamSection2D{A: ssbA, Iz: ssbIz}
}

// ssbNodeCoord returns the 2D coordinate of a node on the beam axis.
func ssbNodeCoord(x float64) [2]float64 { return [2]float64{x, 0} }

// computeReactions solves the domain and returns the vertical (UY) reaction
// at nodeA and nodeB, together with the full displacement vector.
//
// The domain must be fully configured (elements, BCs, loads) before calling.
// Reactions are computed as:
//
//	R = K_orig · U  −  F_ext
//
// at the constrained DOFs (UY of nodeA, UY of nodeB).
// nodeA is assumed to also have UX constrained (pin); nodeB only UY (roller).
func solveSupportedBeam(t *testing.T, dom *domain.Domain, nodeA, nodeB int) (R_A, R_B float64, U *mat.VecDense) {
	t.Helper()

	// 1. Assemble K and F_ext.
	dom.Assemble()

	dpn := dom.DOFPerNode // 3 for 2D beam: [UX, UY, RZ]
	ndof := len(dom.Nodes) * dpn
	uyOff := 1 // UY is local offset 1 in the 2D beam DOF layout (UX=0, UY=1, RZ=2)

	// 2. Save K_orig and F_ext before ApplyDirichletBC modifies them.
	K_orig := mat.NewDense(ndof, ndof, nil)
	K_orig.Copy(dom.K)
	F_ext := mat.NewVecDense(ndof, nil)
	F_ext.CopyVec(dom.F)

	// 3. Apply BCs and solve.
	dom.ApplyDirichletBC()
	var err error
	U, err = solver.LU{}.Solve(dom.K, dom.F)
	if err != nil {
		t.Fatalf("solve: %v", err)
	}

	// 4. Compute reactions: R = K_orig·U − F_ext at the constrained DOFs.
	KU := mat.NewVecDense(ndof, nil)
	KU.MulVec(K_orig, U)

	gA := nodeA*dpn + uyOff
	gB := nodeB*dpn + uyOff
	R_A = KU.AtVec(gA) - F_ext.AtVec(gA)
	R_B = KU.AtVec(gB) - F_ext.AtVec(gB)

	// 5. Update elements for post-processing (enables EndForces()).
	for _, elem := range dom.Elements {
		ue := dom.ElementDisp(elem, U)
		elem.Update(ue)
	}
	return
}

// ─── Model A ─────────────────────────────────────────────────────────────────

// TestSSBeam_A_TriangularFullSpan validates a simply supported beam under a
// triangular distributed load that varies from 0 at node 0 (left) to q at
// node 1 (right) over the full span L.
//
// Analytical solution (Euler-Bernoulli):
//
//	R_A = q·L/6 = 15·500/6 = 1 250 daN   (pin end, under lighter-loaded side)
//	R_B = q·L/3 = 15·500/3 = 2 500 daN   (roller end, under heavier-loaded side)
//
// Deflection (exact, from double integration of M(x)):
//
//	|δ_max| = 0.04314 cm  at x ≈ 259.7 cm  (= 0.5193·L)
//
// Why valuable: confirms that the beam_tri_dist load correctly assembles the
// asymmetric fixed-end reactions (Hermite basis) so that a single element gives
// the exact reactions for a triangular polynomial load.
func TestSSBeam_A_TriangularFullSpan(t *testing.T) {
	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)    // pin (left)
	n1 := dom.AddNode(ssbL, 0, 0) // roller (right)

	sec := ssbSec()
	c01 := [2][2]float64{ssbNodeCoord(0), ssbNodeCoord(ssbL)}
	elem := frame.NewElasticBeam2D(0, [2]int{n0, n1}, c01, ssbE, sec)
	dom.AddElement(elem)

	// Pin at n0: fix UX (DOF 0) and UY (DOF 1).
	dom.FixDOF(n0, 0)
	dom.FixDOF(n0, 1)
	// Roller at n1: fix UY (DOF 1) only.
	dom.FixDOF(n1, 1)

	// Triangular load: intensity_i = 0 (at n0), intensity_j = q (at n1), downward.
	dom.AddBeamLinearLoad(0, [3]float64{0, -1, 0}, 0, ssbq)

	R_A, R_B, U := solveSupportedBeam(t, dom, n0, n1)

	wantRA := ssbq * ssbL / 6 // 1250 daN
	wantRB := ssbq * ssbL / 3 // 2500 daN
	tol := 1e-6 * wantRB      // relative tolerance against largest reaction

	if math.Abs(R_A-wantRA) > tol {
		t.Errorf("Model A: R_A = %.6f daN, want %.6f daN", R_A, wantRA)
	}
	if math.Abs(R_B-wantRB) > tol {
		t.Errorf("Model A: R_B = %.6f daN, want %.6f daN", R_B, wantRB)
	}

	// End rotations at supports:
	//   θ_A = −7·q·L³/(360·EI)   (negative = clockwise)
	//   θ_B = +8·q·L³/(360·EI)
	dpn := dom.DOFPerNode
	rz0 := U.AtVec(n0*dpn + 2)
	rz1 := U.AtVec(n1*dpn + 2)
	EI := ssbE * ssbIz
	wantRZ0 := -7 * ssbq * ssbL * ssbL * ssbL / (360 * EI)
	wantRZ1 := +8 * ssbq * ssbL * ssbL * ssbL / (360 * EI)
	rotTol := 1e-4 * math.Abs(wantRZ0)
	if math.Abs(rz0-wantRZ0) > rotTol {
		t.Errorf("Model A: θ_A = %.9f rad, want %.9f rad", rz0, wantRZ0)
	}
	if math.Abs(rz1-wantRZ1) > rotTol {
		t.Errorf("Model A: θ_B = %.9f rad, want %.9f rad", rz1, wantRZ1)
	}

	// Maximum bending moment (at x_m = L/√3 ≈ 288.68 cm, interior of element):
	//   M_max = q·L²/(9·√3) ≈ 240 562.6 daN·cm
	// Derived from the FEM reaction R_A via statics; R_A is independently verified above.
	xM := math.Sqrt(2 * R_A * ssbL / ssbq) // L/√3
	Mmax := R_A*xM - ssbq*xM*xM*xM/(6*ssbL)
	wantMmax := ssbq * ssbL * ssbL / (9 * math.Sqrt(3))
	if math.Abs(Mmax-wantMmax)/wantMmax > 1e-6 {
		t.Errorf("Model A: M_max = %.4f daN·cm, want %.4f daN·cm", Mmax, wantMmax)
	}
}

// ─── Model B ─────────────────────────────────────────────────────────────────

// TestSSBeam_B_TriangularHalfSpan validates the same beam under a triangular
// load that covers only the second half [a, L] with a = L/2 = 250 cm:
// 0 at x=a, q at x=L, zero load for x < a.
//
// Mesh: 2 elements — element 0 covers [0, a], element 1 covers [a, L].
// Intermediate node at x = a = 250 cm (free: no support).
//
// Analytical solution:
//
//	Total resultant: Q = q·a/2 = 15·250/2 = 1 875 daN.
//	Centroid from left support: x_c = a + 2·a/3 = 250 + 166.67 = 416.67 cm.
//	R_A = Q·(L − x_c) / L = 1875·(500−416.67)/500 = 1875·83.33/500 = 312.5 daN
//	R_B = Q − R_A = 1875 − 312.5 = 1562.5 daN
//
// Why valuable: confirms the multi-element assembly and that a partial-span
// triangular load on the second half produces the correct asymmetric reactions
// (heavier reaction on the loaded side).
func TestSSBeam_B_TriangularHalfSpan(t *testing.T) {
	a := ssbA2 // 250 cm — start of triangular load region

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)    // pin (left)
	nM := dom.AddNode(a, 0, 0)    // free intermediate node
	n1 := dom.AddNode(ssbL, 0, 0) // roller (right)

	sec := ssbSec()
	c0M := [2][2]float64{ssbNodeCoord(0), ssbNodeCoord(a)}
	cML := [2][2]float64{ssbNodeCoord(a), ssbNodeCoord(ssbL)}
	dom.AddElement(frame.NewElasticBeam2D(0, [2]int{n0, nM}, c0M, ssbE, sec)) // elem 0: [0, a]   — no load
	dom.AddElement(frame.NewElasticBeam2D(1, [2]int{nM, n1}, cML, ssbE, sec)) // elem 1: [a, L]   — triangular load

	// Pin at n0: fix UX and UY.
	dom.FixDOF(n0, 0)
	dom.FixDOF(n0, 1)
	// Roller at n1: fix UY.
	dom.FixDOF(n1, 1)

	// Triangular load on element 1 only: 0 at nM (intensity_i), q at n1 (intensity_j).
	dom.AddBeamLinearLoad(1, [3]float64{0, -1, 0}, 0, ssbq)

	R_A, R_B, _ := solveSupportedBeam(t, dom, n0, n1)

	wantRA := 312.5   // daN  (Q·(L−x_c)/L = 1875·83.33/500)
	wantRB := 1_562.5 // daN  (Q − R_A)
	tol := 1e-4       // absolute tolerance (daN)

	if math.Abs(R_A-wantRA) > tol {
		t.Errorf("Model B: R_A = %.6f daN, want %.6f daN", R_A, wantRA)
	}
	if math.Abs(R_B-wantRB) > tol {
		t.Errorf("Model B: R_B = %.6f daN, want %.6f daN", R_B, wantRB)
	}

	// Maximum bending moment (inside element 1, at ξ_m = √(2·R_A·a/q) ≈ 102.06 cm from a):
	//   M_max = R_A·(a + 2·ξ_m/3) ≈ 99 387.9 daN·cm
	// Derived from the FEM reaction R_A via statics.
	xiM := math.Sqrt(2 * R_A * ssbA2 / ssbq)
	Mmax := R_A*(ssbA2+xiM) - ssbq*xiM*xiM*xiM/(6*ssbA2)
	wantMmax := 312.5 * (ssbA2 + 2.0/3*math.Sqrt(2*312.5*ssbA2/ssbq))
	if math.Abs(Mmax-wantMmax)/wantMmax > 1e-6 {
		t.Errorf("Model B: M_max = %.4f daN·cm, want %.4f daN·cm", Mmax, wantMmax)
	}
}

// ─── Model C ─────────────────────────────────────────────────────────────────

// TestSSBeam_C_AntisymmetricPair validates the beam under an antisymmetric
// pair of concentrated forces: P downward at x=b=150 cm, P upward at
// x=b+c=350 cm.
//
// Mesh: 3 elements with nodes at x = 0, 150, 350, 500 cm.
//
// Analytical solution (statics):
//
//	Sum Fy = 0:  R_A + R_B = 0
//	Sum M_B = 0: R_A·L − P·(L−b) + P·(L−b−c) = 0
//	           = R_A·500 − 3000·350 + 3000·150 = 0
//	           → R_A = 3000·(350−150)/500 = 3000·200/500 = +1 200 daN
//	           → R_B = −1 200 daN  (downward; pulls beam into support)
//
// Why valuable: confirms that the sign convention for nodal loads and reactions
// is consistent, and that a negative (upward-pulling) support reaction is
// correctly captured.
func TestSSBeam_C_AntisymmetricPair(t *testing.T) {
	b := ssbB          // 150 cm
	bPc := ssbB + ssbC // 350 cm

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)    // pin (left)
	n1 := dom.AddNode(b, 0, 0)    // load point P↓
	n2 := dom.AddNode(bPc, 0, 0)  // load point P↑
	n3 := dom.AddNode(ssbL, 0, 0) // roller (right)

	sec := ssbSec()
	dom.AddElement(frame.NewElasticBeam2D(0, [2]int{n0, n1}, [2][2]float64{ssbNodeCoord(0), ssbNodeCoord(b)}, ssbE, sec))
	dom.AddElement(frame.NewElasticBeam2D(1, [2]int{n1, n2}, [2][2]float64{ssbNodeCoord(b), ssbNodeCoord(bPc)}, ssbE, sec))
	dom.AddElement(frame.NewElasticBeam2D(2, [2]int{n2, n3}, [2][2]float64{ssbNodeCoord(bPc), ssbNodeCoord(ssbL)}, ssbE, sec))

	// Pin at n0: fix UX (DOF 0) and UY (DOF 1).
	dom.FixDOF(n0, 0)
	dom.FixDOF(n0, 1)
	// Roller at n3: fix UY (DOF 1).
	dom.FixDOF(n3, 1)

	// Antisymmetric load pair:
	//   P downward at n1  → DOF 1, value = −P
	//   P upward   at n2  → DOF 1, value = +P
	dom.ApplyLoad(n1, 1, -ssbP)
	dom.ApplyLoad(n2, 1, +ssbP)

	R_A, R_B, _ := solveSupportedBeam(t, dom, n0, n3)

	wantRA := +1_200.0 // daN  (upward)
	wantRB := -1_200.0 // daN  (downward — beam pulls support up → support pushes down)
	tol := 1e-4        // absolute tolerance (daN)

	if math.Abs(R_A-wantRA) > tol {
		t.Errorf("Model C: R_A = %.6f daN, want %.6f daN", R_A, wantRA)
	}
	if math.Abs(R_B-wantRB) > tol {
		t.Errorf("Model C: R_B = %.6f daN, want %.6f daN", R_B, wantRB)
	}

	// Maximum bending moment at x=b (load node n1), read directly from EndForces:
	//   M_max = R_A · b = 1200 · 150 = 180 000 daN·cm (sagging).
	//   At x=b+c the moment is −M_max by antisymmetry.
	ef0C := dom.Elements[0].(*frame.ElasticBeam2D).EndForces()
	wantMmaxC := R_A * ssbB
	if math.Abs(math.Abs(ef0C.J[2])-wantMmaxC)/wantMmaxC > 1e-6 {
		t.Errorf("Model C: |M at x=b| = %.4f daN·cm, want %.4f daN·cm", math.Abs(ef0C.J[2]), wantMmaxC)
	}
}

// ─── Model D ─────────────────────────────────────────────────────────────────

// TestSSBeam_D_SymmetricPair validates the beam under a symmetric pair of
// downward concentrated forces: P↓ at x=b=150 cm, P↓ at x=b+c=350 cm.
//
// Mesh: same 3-element mesh as Model C (nodes at x = 0, 150, 350, 500 cm).
//
// Analytical solution (statics):
//
//	Sum M_B = 0: R_A·L = P·(L−b) + P·(L−b−c)
//	           = 3000·350 + 3000·150 = 1 500 000 daN·cm
//	           → R_A = 1 500 000 / 500 = 3 000 daN
//	Sum M_A = 0: R_B·L = P·b + P·(b+c)
//	           = 3000·150 + 3000·350 = 1 500 000 daN·cm
//	           → R_B = 1 500 000 / 500 = 3 000 daN
//
// Symmetry check: R_A = R_B = P = 3 000 daN ✓ (symmetric load about midspan).
//
// Why valuable: confirms that two simultaneous nodal loads are correctly
// superimposed in the force vector and that the symmetric-loading symmetry
// of reactions holds numerically.
func TestSSBeam_D_SymmetricPair(t *testing.T) {
	b := ssbB          // 150 cm
	bPc := ssbB + ssbC // 350 cm

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)    // pin (left)
	n1 := dom.AddNode(b, 0, 0)    // load point P↓
	n2 := dom.AddNode(bPc, 0, 0)  // load point P↓
	n3 := dom.AddNode(ssbL, 0, 0) // roller (right)

	sec := ssbSec()
	dom.AddElement(frame.NewElasticBeam2D(0, [2]int{n0, n1}, [2][2]float64{ssbNodeCoord(0), ssbNodeCoord(b)}, ssbE, sec))
	dom.AddElement(frame.NewElasticBeam2D(1, [2]int{n1, n2}, [2][2]float64{ssbNodeCoord(b), ssbNodeCoord(bPc)}, ssbE, sec))
	dom.AddElement(frame.NewElasticBeam2D(2, [2]int{n2, n3}, [2][2]float64{ssbNodeCoord(bPc), ssbNodeCoord(ssbL)}, ssbE, sec))

	// Pin at n0.
	dom.FixDOF(n0, 0)
	dom.FixDOF(n0, 1)
	// Roller at n3.
	dom.FixDOF(n3, 1)

	// Symmetric downward loads.
	dom.ApplyLoad(n1, 1, -ssbP)
	dom.ApplyLoad(n2, 1, -ssbP)

	R_A, R_B, _ := solveSupportedBeam(t, dom, n0, n3)

	wantRA := 3_000.0 // daN
	wantRB := 3_000.0 // daN
	tol := 1e-4       // absolute tolerance (daN)

	if math.Abs(R_A-wantRA) > tol {
		t.Errorf("Model D: R_A = %.6f daN, want %.6f daN", R_A, wantRA)
	}
	if math.Abs(R_B-wantRB) > tol {
		t.Errorf("Model D: R_B = %.6f daN, want %.6f daN", R_B, wantRB)
	}

	// Maximum bending moment at x=b (load node n1), read directly from EndForces:
	//   M_max = R_A · b = 3000 · 150 = 450 000 daN·cm.
	//   The moment is constant = M_max between n1 and n2.
	ef0D := dom.Elements[0].(*frame.ElasticBeam2D).EndForces()
	wantMmaxD := R_A * ssbB
	if math.Abs(math.Abs(ef0D.J[2])-wantMmaxD)/wantMmaxD > 1e-6 {
		t.Errorf("Model D: |M at x=b| = %.4f daN·cm, want %.4f daN·cm", math.Abs(ef0D.J[2]), wantMmaxD)
	}
}
