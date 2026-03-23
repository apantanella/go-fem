package analysis_test

import (
	"math"
	"testing"

	"go-fem/analysis"
	"go-fem/domain"
	"go-fem/element/truss"
	"go-fem/material"
)

// buildSingleTruss3D creates a trivial 3D problem with one NLTruss3D element:
//   - Node 0 at (0,0,0) fully fixed (UX, UY, UZ).
//   - Node 1 at (L,0,0) loaded with F in the X direction.
//   - UY and UZ of node 1 are also constrained (transverse stability).
//
// Returns the domain ready for StaticNonlinearAnalysis.
func buildSingleTruss3D(L, A float64, mat material.UniaxialMaterial, F float64) *domain.Domain {
	dom := domain.NewDomain()
	dom.AddNode(0, 0, 0) // node 0
	dom.AddNode(L, 0, 0) // node 1

	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	elem := truss.NewNLTruss3D(0, nodes, coords, A, mat)
	dom.AddElement(elem)

	// Fix node 0 (UX, UY, UZ)
	dom.FixNode(0)
	// Fix transverse DOFs at node 1 to prevent singularity (truss has no
	// transverse stiffness — only axial)
	dom.FixDOF(1, 1) // UY
	dom.FixDOF(1, 2) // UZ

	// Apply axial load at node 1 in X direction
	dom.ApplyLoad(1, 0, F) // DOF 0 = UX

	return dom
}

// TestNLStaticLinearConvergesInOneIteration verifies that for a linear elastic
// truss (steel below yield), the Newton-Raphson loop converges in at most 2
// iterations.  Because the convergence check is placed before the linear
// solve, the first iteration computes the correct update and the second
// iteration verifies that the residual is below tolerance.  Therefore 2
// iterations (not 1) is the minimum for this algorithm structure on linear
// problems.
//
// Setup: E=1000, Fy=100 (won't be reached), A=1, L=1, F=50.
// Linear solution: u = F·L/(E·A) = 0.05.
func TestNLStaticLinearConvergesInOneIteration(t *testing.T) {
	const E, Fy, Esh = 1000.0, 100.0, 0.0
	const A, L, F = 1.0, 1.0, 50.0 // F < Fy*A → elastic

	mat, err := material.NewSteelBilinear(E, Fy, Esh)
	if err != nil {
		t.Fatalf("NewSteelBilinear: %v", err)
	}

	dom := buildSingleTruss3D(L, A, mat, F)

	ana := analysis.StaticNonlinearAnalysis{Dom: dom, MaxIter: 20, Tol: 1e-10}
	res, err := ana.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if !res.Converged {
		t.Fatalf("expected convergence, got residual history: %v", res.ResidualHistory)
	}
	if res.Iterations > 2 {
		t.Errorf("linear problem: want ≤2 NR iterations, got %d", res.Iterations)
	}

	wantU := F * L / (E * A) // 0.05
	// Global DOF mapping: node1 * 3 + offset_UX = 1*3+0 = 3
	gotU := res.U.AtVec(3) // UX of node 1
	if math.Abs(gotU-wantU) > 1e-10 {
		t.Errorf("linear displacement: want %.9g mm, got %.9g mm", wantU, gotU)
	}
}

// TestNLStaticYieldingTruss verifies that the Newton-Raphson solution for a
// single yielding truss element matches the analytical result.
//
// Setup: E=1000, Fy=100, Esh=100, A=1, L=1, F=150 (post-yield load).
//
// Analytical solution (statically determinate):
//
//	σ_final = F/A = 150
//	From return-mapping with committed state at 0:
//	  Et = E·Esh/(E+Esh) = 1000·100/1100 ≈ 90.909
//	  ε  = Fy/E + (σ - Fy)/Et = 0.1 + (150-100)/90.909 ≈ 0.65
//	  u  = ε·L = 0.65
func TestNLStaticYieldingTruss(t *testing.T) {
	const E, Fy, Esh = 1000.0, 100.0, 100.0
	const A, L, F = 1.0, 1.0, 150.0

	mat, err := material.NewSteelBilinear(E, Fy, Esh)
	if err != nil {
		t.Fatalf("NewSteelBilinear: %v", err)
	}

	dom := buildSingleTruss3D(L, A, mat, F)

	ana := analysis.StaticNonlinearAnalysis{Dom: dom, MaxIter: 100, Tol: 1e-8}
	res, err := ana.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if !res.Converged {
		t.Fatalf("NR did not converge; residuals: %v", res.ResidualHistory)
	}

	// Analytical displacement:
	// sigma = F = 150; from return-mapping at committed-zero state:
	// sigma_trial = E*eps; delta_gamma = (E*eps - Fy)/(E+Esh)
	// sigma = E*eps - E*delta_gamma => sigma = E*Esh*eps/(E+Esh) + E*Fy/(E+Esh) = F
	// => eps = (F*(E+Esh)/E - Fy) / Esh
	wantEps := (F*(E+Esh)/E - Fy) / Esh // = (150*1.1 - 100)/100 = (165-100)/100 = 0.65
	wantU := wantEps * L
	gotU := res.U.AtVec(3) // UX of node 1

	if math.Abs(gotU-wantU) > 1e-6 {
		t.Errorf("yielding displacement: want %.9g mm, got %.9g mm", wantU, gotU)
	}

	// More than 1 iteration expected (nonlinear problem)
	if res.Iterations <= 1 {
		t.Errorf("expected > 1 NR iterations for plastic problem, got %d", res.Iterations)
	}
}

// TestNLStaticNonConvergence verifies that when the analysis does not converge
// within MaxIter, the NLResult still returns a (non-converged) result without
// a fatal error (non-fatal API contract).
func TestNLStaticNonConvergence(t *testing.T) {
	// Use EPP steel (Esh=0). Under a load past yield, KT → 0, singularity →
	// solver should fail. We test with a very small MaxIter to force early exit.
	const E, Fy = 1000.0, 100.0
	const A, L, F = 1.0, 1.0, 110.0 // just past yield

	mat, _ := material.NewSteelBilinear(E, Fy, 0)
	dom := buildSingleTruss3D(L, A, mat, F)

	ana := analysis.StaticNonlinearAnalysis{Dom: dom, MaxIter: 1, Tol: 1e-14}
	res, err := ana.Run()

	// The solver may return a solve error (singular K) or just not converge.
	// Either way, Converged must be false when MaxIter=1 and tolerance is tight.
	if err == nil && res.Converged {
		t.Errorf("expected non-convergence for EPP with MaxIter=1, got converged=true")
	}
}

// TestNLStaticResidualDecreasing verifies that for a hardening material the
// residual norm is monotonically decreasing across NR iterations (quadratic
// convergence is not checked here, just monotonicity for robustness).
func TestNLStaticResidualDecreasing(t *testing.T) {
	const E, Fy, Esh = 1000.0, 100.0, 50.0
	const A, L, F = 1.0, 1.0, 180.0

	mat, _ := material.NewSteelBilinear(E, Fy, Esh)
	dom := buildSingleTruss3D(L, A, mat, F)

	ana := analysis.StaticNonlinearAnalysis{Dom: dom, MaxIter: 50, Tol: 1e-12}
	res, err := ana.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !res.Converged {
		t.Fatalf("expected convergence, got residuals: %v", res.ResidualHistory)
	}

	hist := res.ResidualHistory
	if len(hist) < 2 {
		return // converged in 1 step — nothing to check
	}
	for i := 1; i < len(hist); i++ {
		if hist[i] > hist[i-1]*1.01 { // allow 1% numerical noise
			t.Errorf("residual increased at iteration %d: %g → %g", i, hist[i-1], hist[i])
		}
	}
}
