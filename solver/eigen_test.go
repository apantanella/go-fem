package solver_test

import (
	"math"
	"testing"

	"go-fem/solver"

	"gonum.org/v1/gonum/mat"
)

// build2DOF returns a 2×2 stiffness and mass matrix for which the
// generalised eigenvalues are known analytically.
//
// K = [[2,-1],[-1,2]]  M = I  →  ω² = {1, 3}
//
// These values follow from det(K − λI) = (2−λ)²−1 = 0  →  λ = 1, 3.
// Corresponding normalised mode shapes: φ₁ = [1,1]/√2, φ₂ = [1,−1]/√2.
func build2DOF() (K, M *mat.Dense) {
	K = mat.NewDense(2, 2, []float64{2, -1, -1, 2})
	M = mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	return
}

// checkEigenResidual verifies the dynamic equilibrium residual for every mode.
//
// For each mode k it computes ‖K·φₖ − ω²ₖ·M·φₖ‖₂ and fails if it exceeds
// tol, confirming that the returned (ω², φ) pairs are genuine solutions of
// the generalised problem K·φ = ω²·M·φ.
func checkEigenResidual(t *testing.T, K, M *mat.Dense, res *solver.EigenResult, tol float64) {
	t.Helper()
	n, _ := K.Dims()
	numModes := len(res.Omega2)
	for k := 0; k < numModes; k++ {
		phi := mat.NewVecDense(n, nil)
		for i := 0; i < n; i++ {
			phi.SetVec(i, res.Modes.At(i, k))
		}
		// lhs = K·φ
		lhs := mat.NewVecDense(n, nil)
		lhs.MulVec(K, phi)
		// rhs = ω²·M·φ
		rhs := mat.NewVecDense(n, nil)
		rhs.MulVec(M, phi)
		rhs.ScaleVec(res.Omega2[k], rhs)

		lhs.SubVec(lhs, rhs)
		err := mat.Norm(lhs, 2)
		if err > tol {
			t.Errorf("mode %d: residual ‖K·φ - ω²·M·φ‖ = %g, want < %g", k, err, tol)
		}
	}
}

// checkMOrthonormality verifies mass-orthonormality of the returned mode shapes.
//
// For every pair (i, j) it evaluates the inner product φᵢᵀ·M·φⱼ and checks
// that it equals the Kronecker delta δᵢⱼ within tol.  This guarantees:
//   - M-normalization  (i = j): φᵢᵀ·M·φᵢ = 1
//   - M-orthogonality (i ≠ j): φᵢᵀ·M·φⱼ = 0
func checkMOrthonormality(t *testing.T, M *mat.Dense, res *solver.EigenResult, tol float64) {
	t.Helper()
	n, _ := M.Dims()
	numModes := len(res.Omega2)
	for i := 0; i < numModes; i++ {
		for j := 0; j < numModes; j++ {
			phiI := mat.NewVecDense(n, nil)
			phiJ := mat.NewVecDense(n, nil)
			for r := 0; r < n; r++ {
				phiI.SetVec(r, res.Modes.At(r, i))
				phiJ.SetVec(r, res.Modes.At(r, j))
			}
			tmp := mat.NewVecDense(n, nil)
			tmp.MulVec(M, phiJ)
			dot := mat.Dot(phiI, tmp)

			want := 0.0
			if i == j {
				want = 1.0
			}
			if math.Abs(dot-want) > tol {
				t.Errorf("M-orthonormality [%d,%d]: got %g, want %g", i, j, dot, want)
			}
		}
	}
}

// ── SolveGeneralizedEigen ────────────────────────────────────────────────────

// TestSolveGeneralizedEigen_EigenvaluesIdentityMass checks that the returned
// eigenvalues ω² match the analytically known values for the 2-DOF system
// K = [[2,−1],[−1,2]], M = I.
//
// Closed-form solution: det(K − λI) = 0  →  λ = 1 (first mode) and λ = 3
// (second mode).  The test asserts each ω² is within 1e-10 of the exact
// value, verifying the ascending-sort invariant as well.
func TestSolveGeneralizedEigen_EigenvaluesIdentityMass(t *testing.T) {
	K, M := build2DOF()
	res, err := solver.SolveGeneralizedEigen(K, M, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	wantOmega2 := []float64{1.0, 3.0}
	for k, w := range wantOmega2 {
		if math.Abs(res.Omega2[k]-w) > 1e-10 {
			t.Errorf("ω²[%d] = %g, want %g", k, res.Omega2[k], w)
		}
	}
}

// TestSolveGeneralizedEigen_EigenResidual verifies that the returned mode
// shapes exactly satisfy the governing equation K·φₖ = ω²ₖ·M·φₖ.
//
// For each of the 2 modes the test forms the residual vector
// r = K·φ − ω²·M·φ and checks ‖r‖₂ < 1e-10.  A non-zero residual would
// indicate a numerical defect in the Cholesky-transformation algorithm.
func TestSolveGeneralizedEigen_EigenResidual(t *testing.T) {
	K, M := build2DOF()
	res, err := solver.SolveGeneralizedEigen(K, M, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	checkEigenResidual(t, K, M, res, 1e-10)
}

// TestSolveGeneralizedEigen_MOrthonormality checks that the returned mode
// shapes are orthonormal with respect to the mass matrix M.
//
// Verified conditions (all within 1e-10):
//   - M-normalization  (i = j): φᵢᵀ·M·φᵢ = 1
//   - M-orthogonality (i ≠ j): φᵢᵀ·M·φⱼ = 0
//
// The property is a direct consequence of the back-transformation
// φ = L⁻ᵀ·y and the orthonormality of the standard eigenvectors y.
func TestSolveGeneralizedEigen_MOrthonormality(t *testing.T) {
	K, M := build2DOF()
	res, err := solver.SolveGeneralizedEigen(K, M, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	checkMOrthonormality(t, M, res, 1e-10)
}

// TestSolveGeneralizedEigen_NonIdentityMass exercises the solver with a
// non-identity diagonal mass matrix, for which the analytical solution is
// still available.
//
// System: K = [[5,−1],[−1,2]], M = [[4,0],[0,1]].
// Characteristic equation: det(K − ω²M) = 4ω⁴ − 13ω² + 9 = 0
// →  ω² = 1  and  ω² = 9/4.
//
// In addition to eigenvalue accuracy the test also verifies the dynamic
// residual and M-orthonormality for this non-identity mass case.
func TestSolveGeneralizedEigen_NonIdentityMass(t *testing.T) {
	K := mat.NewDense(2, 2, []float64{5, -1, -1, 2})
	M := mat.NewDense(2, 2, []float64{4, 0, 0, 1})

	res, err := solver.SolveGeneralizedEigen(K, M, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	wantOmega2 := []float64{1.0, 9.0 / 4.0}
	for k, w := range wantOmega2 {
		if math.Abs(res.Omega2[k]-w) > 1e-10 {
			t.Errorf("ω²[%d] = %g, want %g", k, res.Omega2[k], w)
		}
	}
	checkEigenResidual(t, K, M, res, 1e-10)
	checkMOrthonormality(t, M, res, 1e-10)
}

// TestSolveGeneralizedEigen_NumModesSubset confirms that when numModes < n
// only the requested number of (eigenvalue, mode-shape) pairs are returned.
//
// A 4-DOF tridiagonal system (K_ii=2, K_i,i±1=−1, M=I) is solved requesting
// only the first mode.  The test checks:
//   - len(Omega2) == 1
//   - Modes has exactly 1 column
//   - The single mode satisfies the dynamic residual ‖K·φ − ω²·M·φ‖ < 1e-10
func TestSolveGeneralizedEigen_NumModesSubset(t *testing.T) {
	n := 4
	K := mat.NewDense(n, n, nil)
	M := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		K.Set(i, i, 2)
		M.Set(i, i, 1)
		if i > 0 {
			K.Set(i, i-1, -1)
			K.Set(i-1, i, -1)
		}
	}

	res, err := solver.SolveGeneralizedEigen(K, M, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.Omega2) != 1 {
		t.Fatalf("expected 1 eigenvalue, got %d", len(res.Omega2))
	}
	_, cols := res.Modes.Dims()
	if cols != 1 {
		t.Fatalf("expected 1 mode column, got %d", cols)
	}
	checkEigenResidual(t, K, M, res, 1e-10)
}

// TestSolveGeneralizedEigen_MassNotPD verifies that the solver returns a
// descriptive error when the mass matrix is not positive definite.
//
// A zero mass matrix is used as the degenerate case: the Cholesky
// factorization of M must fail, and SolveGeneralizedEigen must propagate
// the error rather than producing silently incorrect results.
func TestSolveGeneralizedEigen_MassNotPD(t *testing.T) {
	K := mat.NewDense(2, 2, []float64{2, -1, -1, 2})
	M := mat.NewDense(2, 2, []float64{0, 0, 0, 0})
	_, err := solver.SolveGeneralizedEigen(K, M, 2)
	if err == nil {
		t.Error("expected error for non-positive-definite mass matrix, got nil")
	}
}

// ── ExtractSubmatrix ─────────────────────────────────────────────────────────

// TestExtractSubmatrix_Basic verifies that the correct entries are copied
// from the full matrix when only a subset of DOF indices is selected.
//
// A 4×4 matrix with easily recognisable values is used; free DOFs are {1, 3}.
// Expected 2×2 result:
//
//	sub[0,0] = A[1,1] = 15
//	sub[0,1] = A[1,3] = 17
//	sub[1,0] = A[3,1] = 23
//	sub[1,1] = A[3,3] = 25
//
// This mirrors the reduction step that ModalAnalysis performs before the
// generalized eigenvalue solve.
func TestExtractSubmatrix_Basic(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		10, 11, 12, 13,
		14, 15, 16, 17,
		18, 19, 20, 21,
		22, 23, 24, 25,
	})
	free := []int{1, 3}
	sub := solver.ExtractSubmatrix(A, free)

	rows, cols := sub.Dims()
	if rows != 2 || cols != 2 {
		t.Fatalf("expected 2×2, got %d×%d", rows, cols)
	}
	expected := [][]float64{{15, 17}, {23, 25}}
	for i, row := range expected {
		for j, v := range row {
			if sub.At(i, j) != v {
				t.Errorf("sub[%d,%d] = %g, want %g", i, j, sub.At(i, j), v)
			}
		}
	}
}

// TestExtractSubmatrix_AllDOFs is a boundary-condition test: when all DOF
// indices are selected the extracted submatrix must be identical to the
// original matrix entry-by-entry, i.e. ExtractSubmatrix is a no-op.
func TestExtractSubmatrix_AllDOFs(t *testing.T) {
	n := 3
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	A := mat.NewDense(n, n, data)
	free := []int{0, 1, 2}
	sub := solver.ExtractSubmatrix(A, free)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if sub.At(i, j) != A.At(i, j) {
				t.Errorf("[%d,%d] = %g, want %g", i, j, sub.At(i, j), A.At(i, j))
			}
		}
	}
}

// ── ExpandModes ──────────────────────────────────────────────────────────────

// TestExpandModes_Basic checks that ExpandModes correctly inserts the reduced
// mode columns into their proper positions in the full DOF layout and pads
// constrained DOFs with zeros.
//
// Setup: 2-column reduced mode matrix for free DOFs {1, 3} in a 4-DOF system.
//
// Expected full matrix (4 rows × 2 cols):
//
//	row 0 (constrained): 0     0
//	row 1 (free DOF 0):  0.5   0.8
//	row 2 (constrained): 0     0
//	row 3 (free DOF 1):  0.3   0.6
//
// This mirrors the expansion step that ModalAnalysis performs after the
// eigenvalue solve to produce mode shapes over the complete DOF space.
func TestExpandModes_Basic(t *testing.T) {
	modesRed := mat.NewDense(2, 2, []float64{
		0.5, 0.8,
		0.3, 0.6,
	})
	free := []int{1, 3}
	ndof := 4
	full := solver.ExpandModes(modesRed, free, ndof)

	rows, cols := full.Dims()
	if rows != ndof || cols != 2 {
		t.Fatalf("expected %d×2, got %d×%d", ndof, rows, cols)
	}
	// DOFs 0, 2 must be zero; DOFs 1, 3 carry the reduced values.
	if full.At(0, 0) != 0 || full.At(2, 0) != 0 {
		t.Error("constrained DOFs should be zero")
	}
	if full.At(1, 0) != 0.5 || full.At(3, 0) != 0.3 {
		t.Errorf("free DOF values mismatch: got (%g,%g), want (0.5,0.3)",
			full.At(1, 0), full.At(3, 0))
	}
}

// ── FrequencyHz ──────────────────────────────────────────────────────────────

// TestFrequencyHz verifies the conversion f = √ω² / (2π) across four cases:
//
//   - ω² = 4π²  →  ω = 2π rad/s  →  f = 1 Hz   (unit check)
//   - ω² = π²   →  ω = π   rad/s  →  f = 0.5 Hz (half-frequency)
//   - ω² = 0    →  rigid-body mode →  f = 0 Hz   (zero guard)
//   - ω² < 0    →  non-physical    →  f = 0 Hz   (negative guard)
//
// The negative-ω² guard prevents NaN propagation for numerically slightly
// negative eigenvalues that can arise from floating-point errors in
// near-singular systems.
func TestFrequencyHz(t *testing.T) {
	cases := []struct {
		omega2 float64
		wantHz float64
	}{
		{4 * math.Pi * math.Pi, 1.0},
		{0, 0},
		{-1, 0},
		{math.Pi * math.Pi, 0.5},
	}
	for _, tc := range cases {
		got := solver.FrequencyHz(tc.omega2)
		if math.Abs(got-tc.wantHz) > 1e-12 {
			t.Errorf("FrequencyHz(%g) = %g, want %g", tc.omega2, got, tc.wantHz)
		}
	}
}

// ── PeriodSeconds ────────────────────────────────────────────────────────────

// TestPeriodSeconds verifies the conversion T = 1/f across three cases:
//
//   - ω² = 4π²  →  f = 1 Hz    →  T = 1 s        (unit check)
//   - ω² = π²   →  f = 0.5 Hz  →  T = 2 s        (half-frequency)
//   - ω² = 0    →  rigid-body   →  T = +∞         (zero-frequency guard)
//
// The +∞ return for a zero eigenvalue is important for callers that display
// modal periods: rather than a divide-by-zero panic they receive a sentinel
// value they can handle gracefully.
func TestPeriodSeconds(t *testing.T) {
	T := solver.PeriodSeconds(4 * math.Pi * math.Pi)
	if math.Abs(T-1.0) > 1e-12 {
		t.Errorf("PeriodSeconds(4π²) = %g, want 1.0", T)
	}

	if !math.IsInf(solver.PeriodSeconds(0), 1) {
		t.Error("PeriodSeconds(0) should be +Inf")
	}

	T2 := solver.PeriodSeconds(math.Pi * math.Pi)
	if math.Abs(T2-2.0) > 1e-12 {
		t.Errorf("PeriodSeconds(π²) = %g, want 2.0", T2)
	}
}
