package solver_test

import (
	"math"
	"testing"

	"go-fem/solver"

	"gonum.org/v1/gonum/mat"
)

// buildSPD returns a simple n×n symmetric positive-definite stiffness matrix
// and a load vector, both used as a shared fixture across solver tests.
//
// K is the standard tridiagonal finite-difference Laplacian:
//
//	K_ii  =  2    (diagonal)
//	K_i,i±1 = −1  (off-diagonal)
//
// It is SPD for any n ≥ 1 with eigenvalues in (0, 4), making it a
// representative stand-in for a real structural stiffness matrix after
// Dirichlet BC elimination.  F is the all-ones vector so that the solution
// u = K⁻¹·F has a smooth, well-conditioned profile.
func buildSPD(n int) (*mat.Dense, *mat.VecDense) {
	K := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		K.Set(i, i, 2)
		if i > 0 {
			K.Set(i, i-1, -1)
			K.Set(i-1, i, -1)
		}
	}
	F := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		F.SetVec(i, 1)
	}
	return K, F
}

// checkResidual is a test helper that measures the relative residual of a
// linear-system solution and fails the test if it exceeds tol.
//
// Metric: ‖K·u − F‖₂ / ‖F‖₂
//
// Dividing by ‖F‖₂ gives a scale-independent measure so that the same
// tolerance applies regardless of the magnitude of the applied load.
func checkResidual(t *testing.T, name string, K *mat.Dense, F, u *mat.VecDense, tol float64) {
	t.Helper()
	n, _ := K.Dims()
	r := mat.NewVecDense(n, nil)
	r.MulVec(K, u)
	r.SubVec(r, F)
	res := mat.Norm(r, 2) / mat.Norm(F, 2)
	if res > tol {
		t.Errorf("%s: relative residual %g exceeds tolerance %g", name, res, tol)
	}
}

// TestSolversAgainstReference cross-validates every linear solver against the
// dense Cholesky reference on a 20-DOF tridiagonal SPD system.
//
// For each solver the test checks two independent conditions:
//  1. Physical residual: ‖K·u − F‖₂/‖F‖₂ < 1e-8 — the returned vector u
//     actually satisfies the linear system to engineering precision.
//  2. Numerical agreement: |u[i] − u_chol[i]| < 1e-8 for every DOF — the
//     solution is bit-for-bit consistent with the Cholesky baseline, ruling
//     out solver-specific systematic errors.
//
// Solvers under test and their configurations:
//   - LU{}          — dense LU (no special tuning needed for SPD)
//   - SkylineLDL{}  — sparse-banded LDL^T (exploits tridiagonal sparsity)
//   - CG{Tol:1e-12} — Conjugate Gradient, tight tolerance to ensure accuracy
//   - GMRES{Tol:1e-12, Restart:10} — GMRES with small restart to also verify
//     restart logic on a system that fits in a single Krylov space
func TestSolversAgainstReference(t *testing.T) {
	const n = 20
	K, F := buildSPD(n)

	// Reference solution via Cholesky.
	ref, err := (solver.Cholesky{}).Solve(K, F)
	if err != nil {
		t.Fatalf("Cholesky reference failed: %v", err)
	}

	cases := []struct {
		name string
		slv  solver.LinearSolver
	}{
		{"LU", solver.LU{}},
		{"SkylineLDL", solver.SkylineLDL{}},
		{"CG", solver.CG{Tol: 1e-12}},
		{"GMRES", solver.GMRES{Tol: 1e-12, Restart: 10}},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			u, err := tc.slv.Solve(K, F)
			if err != nil {
				t.Fatalf("Solve error: %v", err)
			}
			checkResidual(t, tc.name, K, F, u, 1e-8)

			// Compare with Cholesky reference.
			for i := 0; i < n; i++ {
				if math.Abs(u.AtVec(i)-ref.AtVec(i)) > 1e-8 {
					t.Errorf("%s: u[%d] = %g, want %g", tc.name, i, u.AtVec(i), ref.AtVec(i))
				}
			}
		})
	}
}

// TestCGNotSPD verifies that the CG solver detects an indefinite matrix and
// returns a non-nil error instead of silently producing a wrong result.
//
// CG is mathematically guaranteed to converge only for SPD matrices; on
// indefinite systems it can produce arbitrarily bad results or diverge.
// The implementation is expected to detect the non-SPD condition (e.g. via
// a negative diagonal check or a negative inner product during the iteration)
// and surface it as an error so that callers can fall back to LU or GMRES.
//
// The test uses a 3×3 diagonal matrix with a negative first entry (λ₁ = −1),
// which makes it indefinite while keeping the system simple to reason about.
func TestCGNotSPD(t *testing.T) {
	K := mat.NewDense(3, 3, []float64{
		-1, 0, 0,
		0, 2, 0,
		0, 0, 3,
	})
	F := mat.NewVecDense(3, []float64{1, 1, 1})
	_, err := (solver.CG{}).Solve(K, F)
	if err == nil {
		t.Error("CG: expected error for indefinite matrix, got nil")
	}
}

// TestGMRESNonSymmetric confirms that GMRES correctly solves a non-symmetric
// system that Cholesky and CG cannot handle.
//
// The 4×4 matrix has an asymmetric sparsity pattern (upper and lower
// off-diagonals differ) so it is neither symmetric nor positive definite.
// GMRES is the only solver in the package that is theoretically sound for
// such systems (e.g. models with ZeroLength spring connectors that break
// global symmetry).
//
// The test checks the relative residual ‖K·u − F‖₂/‖F‖₂ < 1e-8, which is
// the same criterion used for SPD solvers, confirming that GMRES meets the
// same engineering accuracy bar regardless of matrix symmetry.
func TestGMRESNonSymmetric(t *testing.T) {
	K := mat.NewDense(4, 4, []float64{
		4, 1, 0, 0,
		3, 4, 1, 0,
		0, 2, 4, 1,
		0, 0, 1, 3,
	})
	F := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	u, err := (solver.GMRES{Tol: 1e-12}).Solve(K, F)
	if err != nil {
		t.Fatalf("GMRES non-symmetric: %v", err)
	}
	checkResidual(t, "GMRES non-symmetric", K, F, u, 1e-8)
}
