package solver

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// EigenResult holds the output of a generalized eigenvalue solve.
type EigenResult struct {
	// Omega2 contains the n smallest eigenvalues ω² (rad²/s²), sorted ascending.
	Omega2 []float64
	// Modes is an (ndof × n) matrix whose columns are the M-normalised mode shapes.
	// φᵢᵀ · M · φᵢ = 1  for each mode i.
	Modes *mat.Dense
}

// SolveGeneralizedEigen solves the generalized eigenvalue problem
//
//	K · φ = ω² · M · φ
//
// for the numModes smallest eigenvalues, using a dense Cholesky transformation.
// K and M must be square, symmetric, and (after BC elimination) positive definite.
//
// Algorithm (Bathe §10.2):
//  1. Cholesky-factorize M = L · Lᵀ
//  2. Form A = L⁻¹ · K · L⁻ᵀ  (A is symmetric)
//  3. Solve the standard problem A · y = ω² · y
//  4. Back-transform φ = L⁻ᵀ · y  (already M-normalized since ‖y‖=1)
func SolveGeneralizedEigen(K, M *mat.Dense, numModes int) (*EigenResult, error) {
	n, _ := K.Dims()
	if numModes <= 0 || numModes > n {
		numModes = n
	}

	// Symmetrize K and M to remove floating-point asymmetry.
	symK := mat.NewSymDense(n, nil)
	symM := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			symK.SetSym(i, j, (K.At(i, j)+K.At(j, i))/2)
			symM.SetSym(i, j, (M.At(i, j)+M.At(j, i))/2)
		}
	}

	// ── Step 1: Cholesky of M ──────────────────────────────────────────────
	var cholM mat.Cholesky
	if ok := cholM.Factorize(symM); !ok {
		return nil, fmt.Errorf("eigen: mass matrix is not positive definite (check BCs and densities)")
	}

	// Extract lower-triangular factor L (M = L · Lᵀ).
	var L mat.TriDense
	cholM.LTo(&L)

	// ── Step 2: Compute L⁻¹ by solving L · X = I ─────────────────────────
	eye := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		eye.Set(i, i, 1.0)
	}
	var Linv mat.Dense
	if err := Linv.Solve(&L, eye); err != nil {
		return nil, fmt.Errorf("eigen: cannot invert Cholesky factor: %w", err)
	}

	// ── A = L⁻¹ · K · L⁻ᵀ ───────────────────────────────────────────────
	KDense := mat.DenseCopyOf(symK)
	tmp := mat.NewDense(n, n, nil)
	tmp.Mul(&Linv, KDense)        // tmp = L⁻¹ · K
	Adense := mat.NewDense(n, n, nil)
	Adense.Mul(tmp, Linv.T())     // A = L⁻¹ · K · L⁻ᵀ

	// Symmetrize A to compensate for any residual numerical asymmetry.
	Asym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			Asym.SetSym(i, j, (Adense.At(i, j)+Adense.At(j, i))/2)
		}
	}

	// ── Step 3: Standard symmetric eigenvalue problem A · y = λ · y ───────
	var eig mat.EigenSym
	if ok := eig.Factorize(Asym, true); !ok {
		return nil, fmt.Errorf("eigen: symmetric eigenvalue factorization failed")
	}

	omega2all := eig.Values(nil) // ascending order
	var V mat.Dense
	eig.VectorsTo(&V) // columns = eigenvectors

	// ── Step 4: Back-transform modes φ = L⁻ᵀ · y ─────────────────────────
	// L⁻ᵀ = (Linv)ᵀ
	LinvT := Linv.T()
	modes := mat.NewDense(n, numModes, nil)
	omega2 := make([]float64, numModes)

	for k := 0; k < numModes; k++ {
		omega2[k] = omega2all[k]

		// Extract k-th eigenvector y_k from V.
		y := mat.NewVecDense(n, nil)
		for i := 0; i < n; i++ {
			y.SetVec(i, V.At(i, k))
		}

		// φ_k = L⁻ᵀ · y_k
		phi := mat.NewVecDense(n, nil)
		phi.MulVec(LinvT, y)

		// Store column k of modes matrix.
		for i := 0; i < n; i++ {
			modes.Set(i, k, phi.AtVec(i))
		}
	}

	return &EigenResult{Omega2: omega2, Modes: modes}, nil
}

// ExtractSubmatrix builds the reduced (nFree × nFree) submatrix of A
// corresponding to the free DOF indices.
func ExtractSubmatrix(A *mat.Dense, freeDOFs []int) *mat.Dense {
	n := len(freeDOFs)
	sub := mat.NewDense(n, n, nil)
	for i, gi := range freeDOFs {
		for j, gj := range freeDOFs {
			sub.Set(i, j, A.At(gi, gj))
		}
	}
	return sub
}

// ExpandModes inserts a reduced mode matrix (nFree × numModes) into a full
// (ndof × numModes) matrix, padding constrained DOFs with zeros.
func ExpandModes(modesRed *mat.Dense, freeDOFs []int, ndof int) *mat.Dense {
	_, numModes := modesRed.Dims()
	full := mat.NewDense(ndof, numModes, nil)
	for i, g := range freeDOFs {
		for k := 0; k < numModes; k++ {
			full.Set(g, k, modesRed.At(i, k))
		}
	}
	return full
}

// FrequencyHz converts ω² (rad²/s²) to frequency in Hz.
func FrequencyHz(omega2 float64) float64 {
	if omega2 <= 0 {
		return 0
	}
	return math.Sqrt(omega2) / (2 * math.Pi)
}

// PeriodSeconds returns the natural period T = 1/f.
func PeriodSeconds(omega2 float64) float64 {
	f := FrequencyHz(omega2)
	if f <= 0 {
		return math.Inf(1)
	}
	return 1.0 / f
}
