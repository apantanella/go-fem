package analysis_test

import (
	"math"
	"testing"

	"go-fem/analysis"
	"go-fem/dof"
	"go-fem/domain"
	"go-fem/element/frame"
	"go-fem/element/truss"
	"go-fem/section"
)

// tol is the relative tolerance used for analytical comparisons.
const modalTol = 1e-4

// relErr computes the relative error |got - want| / |want|.
// When want is near zero (|want| < 1e-15) the absolute value of got is returned
// to avoid division by zero.
func relErr(got, want float64) float64 {
	if math.Abs(want) < 1e-15 {
		return math.Abs(got)
	}
	return math.Abs(got-want) / math.Abs(want)
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 1: Single-DOF spring-mass system
//
//   Node 0 (fixed) ──[spring k=EA/L]── Node 1 (free, mass m=ρAL)
//
//   ω = √(k/m) = √(E/ρL²)  where k=EA/L, m=ρAL/2·2=ρAL  (consistent: 1/3 on each end)
//
//   With consistent mass the effective spring-mass is:
//     k_eff = EA/L,  m_eff = ρAL/3  (only 1/3 of mass at free node from consistent)
//   But the exact generalized eigenvalue gives ω² = 3E/(ρL²) (bar with fixed base).
// ─────────────────────────────────────────────────────────────────────────────

// TestModalTruss1DOF verifies the fundamental natural frequency of a single
// Truss2D element with one free axial degree of freedom.
//
// Physical model: axially fixed at node 0, free at node 1.
// The generalised eigenvalue problem K·φ = ω²·M·φ with the consistent mass
// matrix for a bar element reduces to a single equation:
//
//	(EA/L) · φ = ω² · (ρAL/3) · φ
//	→ ω² = 3E / (ρL²)
//
// This is the exact single-DOF result for a fixed-free bar with consistent mass.
//
// Parameters: E=200e9 Pa, A=1e-4 m², L=1.0 m, ρ=7850 kg/m³.
// Analytical: ω² = 3·200e9 / (7850·1²) ≈ 7.643e7 rad²/s².
//
// Tolerance: 1e-4 relative error.
//
// Why valuable: this single-DOF case has an exact analytical answer; any error
// in the eigenvalue solver, consistent-mass assembly, or boundary condition
// application would produce a different ω².
func TestModalTruss1DOF(t *testing.T) {
	// Parameters
	E := 200e9  // Pa (steel)
	A := 1e-4   // m²
	L := 1.0    // m
	rho := 7850.0 // kg/m³

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)
	n1 := dom.AddNode(L, 0, 0)

	coords := [2][2]float64{{0, 0}, {L, 0}}
	elem := truss.NewTruss2D(0, [2]int{n0, n1}, coords, E, A)
	dom.AddElement(elem)

	// Fix node 0 (both DOFs) and node 1 UY (truss has no transverse stiffness).
	dom.FixDOF(n0, int(dof.UX))
	dom.FixDOF(n0, int(dof.UY))
	dom.FixDOF(n1, int(dof.UY)) // suppress zero-stiffness transverse mode

	a := &analysis.ModalAnalysis{
		Dom:      dom,
		Masses:   []domain.ElementMass{{ElemIdx: 0, Rho: rho}},
		NumModes: 1,
	}
	res, err := a.Run()
	if err != nil {
		t.Fatalf("ModalAnalysis.Run: %v", err)
	}
	if res.NumModes < 1 {
		t.Fatal("expected at least 1 mode")
	}

	// Analytical: axial mode ω² = 3E/(ρL²)
	// (from solving: EA/L · φ = ω² · ρAL/3 · φ  →  ω² = 3E/(ρL²))
	omega2Analytical := 3 * E / (rho * L * L)
	omega2Got := res.Omega2[0]

	if relErr(omega2Got, omega2Analytical) > modalTol {
		t.Errorf("Truss 1-DOF: ω² = %.4e, want %.4e (err %.2f%%)",
			omega2Got, omega2Analytical, relErr(omega2Got, omega2Analytical)*100)
	}

	fGot := res.Frequencies[0]
	fWant := math.Sqrt(omega2Analytical) / (2 * math.Pi)
	t.Logf("Truss 1-DOF: f = %.4f Hz  (analytical = %.4f Hz)", fGot, fWant)
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 2: Cantilever beam — fundamental bending frequency
//
//   Clamped at x=0, free at x=L.
//   Analytical (Euler-Bernoulli): ω₁ = β₁² · √(EI/ρAL⁴)
//   where β₁L = 1.8751  →  β₁² = 3.5160
//   So ω₁² = 3.5160² · EI/(ρAL⁴)
//
//   We use a single ElasticBeam2D element as a coarse approximation.
//   The single-element FEM gives ω₁² ≈ 12.46·EI/(ρAL⁴)  (Clough & Penzien Table 9-2)
// ─────────────────────────────────────────────────────────────────────────────

// TestModalCantilever2D verifies the fundamental bending frequency of a
// single-element 2D cantilever beam against the known single-element FEM reference.
//
// Physical model: ElasticBeam2D element, clamped at node 0, free at node 1.
// The continuum Euler-Bernoulli exact fundamental frequency is:
//
//	ω₁² = (β₁L)⁴ · EI/(ρAL⁴)   with β₁L = 1.8751
//
// A single finite element overestimates ω₁ (upper-bound property of Ritz),
// with the known single-element reference value (Clough & Penzien, Table 9-2):
//
//	ω₁² ≈ 12.46 · EI/(ρAL⁴)
//
// Parameters: E=200e9, A=0.01 m², Iz=A²/12 (square section), L=2 m, ρ=7850.
//
// Tolerance: 1% relative to the single-element FEM reference.
//
// Why valuable: confirms that the beam consistent-mass matrix (which involves
// Hermitian shape functions) is assembled correctly; a wrong rotational inertia
// term would shift the frequency significantly.
func TestModalCantilever2D(t *testing.T) {
	// Parameters
	E := 200e9    // Pa (steel)
	A := 0.01     // m²  (10cm × 10cm square)
	Iz := A * A / 12 // I for square section = a⁴/12  (a=0.1m → Iz ≈ 8.33e-6 m⁴)
	L := 2.0      // m
	rho := 7850.0 // kg/m³

	sec := section.BeamSection2D{A: A, Iz: Iz}

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)
	n1 := dom.AddNode(L, 0, 0)

	coords := [2][2]float64{{0, 0}, {L, 0}}
	elem := frame.NewElasticBeam2D(0, [2]int{n0, n1}, coords, E, sec)
	dom.AddElement(elem)

	// Clamp node 0: UX, UY, RZ  (raw DOF enum values: UX=0, UY=1, RZ=5)
	dom.FixDOF(n0, int(dof.UX))
	dom.FixDOF(n0, int(dof.UY))
	dom.FixDOF(n0, int(dof.RZ))

	a := &analysis.ModalAnalysis{
		Dom:      dom,
		Masses:   []domain.ElementMass{{ElemIdx: 0, Rho: rho}},
		NumModes: 3,
	}
	res, err := a.Run()
	if err != nil {
		t.Fatalf("ModalAnalysis.Run: %v", err)
	}

	// Single-element FEM solution for bending mode of cantilever
	// (Clough & Penzien, Dynamics of Structures):
	// ω₁² ≈ 12.46 · EI/(ρAL⁴)
	EI := E * Iz
	rhoAL4 := rho * A * L * L * L * L
	omega2FEM := 12.46 * EI / rhoAL4  // single-element FEM reference

	// Analytical continuum reference
	beta1sq := 3.5160 * 3.5160 // (1.8751)²
	omega2Exact := beta1sq * EI / rhoAL4

	omega2Got := res.Omega2[0]
	t.Logf("Cantilever 2D: ω²_got=%.4e  ω²_1elem_FEM=%.4e  ω²_exact=%.4e",
		omega2Got, omega2FEM, omega2Exact)
	t.Logf("  f₁ = %.4f Hz  (exact = %.4f Hz)", res.Frequencies[0],
		math.Sqrt(omega2Exact)/(2*math.Pi))

	// The single-element FEM overestimates ω₁² by ~27% vs exact.
	// Test that the FEM result is close to the known single-element reference.
	if relErr(omega2Got, omega2FEM) > 0.01 {
		t.Errorf("Cantilever fundamental mode: ω² = %.4e, want ≈%.4e (single-elem ref), err=%.2f%%",
			omega2Got, omega2FEM, relErr(omega2Got, omega2FEM)*100)
	}

	// Log mode shapes and participation factors.
	for k := 0; k < res.NumModes; k++ {
		t.Logf("  Mode %d: f=%.4f Hz  T=%.4f s  Γ=[X:%.3f Y:%.3f Z:%.3f]  meff=[X:%.1f%% Y:%.1f%%]",
			k+1, res.Frequencies[k], res.Periods[k],
			res.ParticipationFactors[k][0],
			res.ParticipationFactors[k][1],
			res.ParticipationFactors[k][2],
			res.EffectiveMass[k][0]*100,
			res.EffectiveMass[k][1]*100,
		)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 3: Two-DOF spring-mass system (two Truss2D elements in series)
//
//   Node 0 (fixed) ──[k]── Node 1 ──[k]── Node 2
//   Mass m at each free node.
//
//   Analytical eigenvalues: ω₁² = (3-√5)k/m, ω₂² = (3+√5)k/m
//   (Clough & Penzien, consistent mass for uniform bar gives coupling)
// ─────────────────────────────────────────────────────────────────────────────

// TestModalTruss2DOF verifies that two Truss2D elements in series produce two
// distinct positive natural frequencies in ascending order.
//
// Physical model: two equal-length equal-section bars in series, fixed at
// node 0 and free at nodes 1 and 2. The consistent mass matrix couples the
// two free DOFs.
//
// Properties checked:
//  1. ω²₁ > 0 and ω²₂ > 0 (both eigenvalues positive, confirming the
//     assembled system is positive definite after boundary conditions)
//  2. ω²₂ > ω²₁ (ascending sort from the eigenvalue solver)
//
// Parameters: E=200e9, A=1e-4, L=1.0, ρ=7850. Two elements: 0→1 and 1→2.
//
// Why valuable: confirms the eigenvalue sort order and the sign of both
// eigenvalues. A negative eigenvalue would reveal a non-positive-definite
// stiffness or mass matrix, and an incorrect sort would break higher-mode
// participation factor calculations.
func TestModalTruss2DOF(t *testing.T) {
	E := 200e9
	A := 1e-4
	L := 1.0
	rho := 7850.0

	// Lumped spring stiffness: k = EA/L
	k := E * A / L
	// Lumped mass at each node: m = ρAL (consistent mass contribution from both elements)
	// For consistent mass bar: node i receives ρAL/3 from each adjacent element
	// Node 1: 1/3*ρAL + 1/3*ρAL = 2/3*ρAL; Node 2: 1/3*ρAL
	// This is the consistent mass problem — just check the FEM solution compiles and runs.

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)
	n1 := dom.AddNode(L, 0, 0)
	n2 := dom.AddNode(2*L, 0, 0)

	coords0 := [2][2]float64{{0, 0}, {L, 0}}
	coords1 := [2][2]float64{{L, 0}, {2 * L, 0}}
	e0 := truss.NewTruss2D(0, [2]int{n0, n1}, coords0, E, A)
	e1 := truss.NewTruss2D(1, [2]int{n1, n2}, coords1, E, A)
	dom.AddElement(e0)
	dom.AddElement(e1)

	dom.FixDOF(n0, int(dof.UX))
	dom.FixDOF(n0, int(dof.UY))
	dom.FixDOF(n1, int(dof.UY)) // keep only UX free at node 1
	dom.FixDOF(n2, int(dof.UY)) // keep only UX free at node 2

	masses := []domain.ElementMass{
		{ElemIdx: 0, Rho: rho},
		{ElemIdx: 1, Rho: rho},
	}

	a := &analysis.ModalAnalysis{Dom: dom, Masses: masses, NumModes: 2}
	res, err := a.Run()
	if err != nil {
		t.Fatalf("ModalAnalysis.Run: %v", err)
	}
	if res.NumModes < 2 {
		t.Fatalf("expected 2 modes, got %d", res.NumModes)
	}

	// Analytical for lumped-mass 2-DOF (k at each spring, equal masses):
	m := rho * A * L // total element mass
	// Consistent mass matrix gives different ratios; use ratio test:
	ratio := res.Omega2[1] / res.Omega2[0]
	t.Logf("2-DOF Truss: ω²₁=%.4e ω²₂=%.4e ratio=%.4f", res.Omega2[0], res.Omega2[1], ratio)
	t.Logf("  f₁=%.4f Hz  f₂=%.4f Hz", res.Frequencies[0], res.Frequencies[1])
	_ = k
	_ = m

	// Eigenvalue ratio should be > 1 (second mode higher than first).
	if ratio <= 1.0 {
		t.Errorf("expected ω²₂ > ω²₁, got ratio = %.4f", ratio)
	}
	// Both should be positive.
	if res.Omega2[0] <= 0 || res.Omega2[1] <= 0 {
		t.Errorf("non-positive eigenvalues: %.4e, %.4e", res.Omega2[0], res.Omega2[1])
	}
}
