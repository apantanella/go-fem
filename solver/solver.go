// Package solver provides linear system solvers (Layer 4 support).
package solver

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// LinearSolver solves K·U = F for U.
type LinearSolver interface {
	Solve(K *mat.Dense, F *mat.VecDense) (*mat.VecDense, error)
}

// Cholesky solves a symmetric positive-definite system via Cholesky factorization.
type Cholesky struct{}

func (Cholesky) Solve(K *mat.Dense, F *mat.VecDense) (*mat.VecDense, error) {
	n, _ := K.Dims()

	// Convert Dense → SymDense (average to enforce perfect symmetry).
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			sym.SetSym(i, j, (K.At(i, j)+K.At(j, i))/2)
		}
	}

	var chol mat.Cholesky
	if ok := chol.Factorize(sym); !ok {
		return nil, fmt.Errorf("cholesky: matrix is not positive definite")
	}

	u := mat.NewVecDense(n, nil)
	if err := chol.SolveVecTo(u, F); err != nil {
		return nil, fmt.Errorf("cholesky solve: %w", err)
	}
	return u, nil
}

// LU solves a general system via LU factorization (fallback solver).
type LU struct{}

func (LU) Solve(K *mat.Dense, F *mat.VecDense) (*mat.VecDense, error) {
	n, _ := K.Dims()

	// Wrap F as n×1 Dense for mat.Dense.Solve.
	fMat := mat.NewDense(n, 1, nil)
	for i := 0; i < n; i++ {
		fMat.Set(i, 0, F.AtVec(i))
	}

	var uMat mat.Dense
	if err := uMat.Solve(K, fMat); err != nil {
		return nil, fmt.Errorf("LU solve: %w", err)
	}

	u := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		u.SetVec(i, uMat.At(i, 0))
	}
	return u, nil
}
