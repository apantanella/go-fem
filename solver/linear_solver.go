// Package solver provides linear system solvers (Layer 4 support).
package solver

import (
	"fmt"
	"math"

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

// ── Sparse direct solver ────────────────────────────────────────────────────

// SkylineLDL implements a symmetric skyline (variable-bandwidth profile) LDL^T
// sparse direct solver, well-suited for banded FEM stiffness matrices.
// The sparsity pattern is inferred from entries with |K[i,j]| > ZeroTol·max|K[i,i]|.
// ZeroTol defaults to 1e-14; increase it to ignore small off-diagonal fill.
type SkylineLDL struct {
	ZeroTol float64
}

func (s SkylineLDL) Solve(K *mat.Dense, F *mat.VecDense) (*mat.VecDense, error) {
	n, _ := K.Dims()
	zeroTol := s.ZeroTol
	if zeroTol <= 0 {
		zeroTol = 1e-14
	}

	// Maximum diagonal entry for pattern thresholding.
	maxDiag := 0.0
	for i := 0; i < n; i++ {
		if v := math.Abs(K.At(i, i)); v > maxDiag {
			maxDiag = v
		}
	}
	if maxDiag == 0 {
		return nil, fmt.Errorf("skyline ldl: all diagonal entries are zero")
	}
	thresh := zeroTol * maxDiag

	// sky[i] = index of the first nonzero column in row i of the lower triangle.
	sky := make([]int, n)
	for i := 0; i < n; i++ {
		sky[i] = i
		for j := 0; j < i; j++ {
			if math.Abs(K.At(i, j)) > thresh || math.Abs(K.At(j, i)) > thresh {
				sky[i] = j
				break
			}
		}
	}

	// Row-oriented skyline storage: row i holds columns sky[i]..i-1 (diagonal excluded).
	rowPtr := make([]int, n+1)
	for i := 0; i < n; i++ {
		rowPtr[i+1] = rowPtr[i] + (i - sky[i])
	}
	L := make([]float64, rowPtr[n]) // strictly lower-triangular factors
	D := make([]float64, n)

	// Initialise from K (symmetrised).
	for i := 0; i < n; i++ {
		D[i] = K.At(i, i)
		for j := sky[i]; j < i; j++ {
			L[rowPtr[i]+(j-sky[i])] = (K.At(i, j) + K.At(j, i)) / 2
		}
	}

	// colList[j] = rows k > j where sky[k] ≤ j  (L[k][j] is nonzero).
	colList := make([][]int, n)
	for k := 0; k < n; k++ {
		for j := sky[k]; j < k; j++ {
			colList[j] = append(colList[j], k)
		}
	}

	// LDL^T factorisation.
	for i := 0; i < n; i++ {
		for j := sky[i]; j < i; j++ {
			lij := L[rowPtr[i]+(j-sky[i])]
			D[i] -= lij * lij * D[j]
		}
		if math.Abs(D[i]) < 1e-14*maxDiag {
			return nil, fmt.Errorf("skyline ldl: zero pivot at row %d (D = %g)", i, D[i])
		}
		for _, k := range colList[i] {
			lki := L[rowPtr[k]+(i-sky[k])]
			jStart := sky[k]
			if sky[i] > jStart {
				jStart = sky[i]
			}
			for j := jStart; j < i; j++ {
				lki -= L[rowPtr[k]+(j-sky[k])] * L[rowPtr[i]+(j-sky[i])] * D[j]
			}
			L[rowPtr[k]+(i-sky[k])] = lki / D[i]
		}
	}

	// Forward substitution: L z = F  (L is unit lower-triangular).
	z := make([]float64, n)
	for i := 0; i < n; i++ {
		z[i] = F.AtVec(i)
		for j := sky[i]; j < i; j++ {
			z[i] -= L[rowPtr[i]+(j-sky[i])] * z[j]
		}
	}

	// Diagonal solve: D w = z.
	for i := 0; i < n; i++ {
		z[i] /= D[i]
	}

	// Backward substitution: L^T u = w.
	for i := n - 1; i >= 0; i-- {
		for _, k := range colList[i] {
			z[i] -= L[rowPtr[k]+(i-sky[k])] * z[k]
		}
	}

	return mat.NewVecDense(n, z), nil
}

// ── Iterative solvers ───────────────────────────────────────────────────────

// CG implements the Conjugate Gradient iterative method for symmetric
// positive-definite systems K·U = F (the typical FEM case).
// Convergence criterion: ||r||/||F|| < Tol  (default Tol = 1e-10).
// MaxIter defaults to 3·n when ≤ 0.
type CG struct {
	Tol     float64
	MaxIter int
}

func (c CG) Solve(K *mat.Dense, F *mat.VecDense) (*mat.VecDense, error) {
	n, _ := K.Dims()
	tol := c.Tol
	if tol <= 0 {
		tol = 1e-10
	}
	maxIter := c.MaxIter
	if maxIter <= 0 {
		maxIter = 3 * n
	}
	if maxIter < 1 {
		maxIter = 1
	}

	x := mat.NewVecDense(n, nil)
	r := mat.NewVecDense(n, nil)
	r.CopyVec(F)                   // r₀ = F  (x₀ = 0)
	p := mat.NewVecDense(n, nil)
	p.CopyVec(r)
	Ap := mat.NewVecDense(n, nil)
	pNext := mat.NewVecDense(n, nil)

	bnorm2 := mat.Dot(F, F)
	if bnorm2 == 0 {
		return x, nil
	}
	rsold := bnorm2 // r₀ = F, so r₀·r₀ = F·F

	for i := 0; i < maxIter; i++ {
		Ap.MulVec(K, p)
		pAp := mat.Dot(p, Ap)
		if pAp <= 0 {
			return nil, fmt.Errorf("cg: matrix not positive-definite at iteration %d (p·K·p = %g)", i, pAp)
		}
		alpha := rsold / pAp
		x.AddScaledVec(x, alpha, p)    // xₖ₊₁ = xₖ + α pₖ
		r.AddScaledVec(r, -alpha, Ap)  // rₖ₊₁ = rₖ − α K pₖ
		rsnew := mat.Dot(r, r)
		if rsnew <= tol*tol*bnorm2 {
			return x, nil
		}
		beta := rsnew / rsold
		pNext.AddScaledVec(r, beta, p) // pₖ₊₁ = rₖ₊₁ + β pₖ  (no aliasing)
		p, pNext = pNext, p
		rsold = rsnew
	}

	return x, fmt.Errorf("cg: did not converge in %d iterations (relative residual = %g)",
		maxIter, math.Sqrt(rsold/bnorm2))
}

// GMRES implements the restarted GMRES(m) iterative method for general linear
// systems K·U = F. Suitable for non-symmetric or indefinite FEM systems.
// Convergence criterion: ||r||/||F|| < Tol  (default 1e-10).
// Restart is the Krylov subspace dimension per restart cycle (default 50).
// MaxIter is the outer restart limit (default ⌈3n/Restart⌉).
type GMRES struct {
	Tol     float64
	MaxIter int
	Restart int
}

func (g GMRES) Solve(K *mat.Dense, F *mat.VecDense) (*mat.VecDense, error) {
	n, _ := K.Dims()
	tol := g.Tol
	if tol <= 0 {
		tol = 1e-10
	}
	m := g.Restart
	if m <= 0 {
		m = 50
	}
	if m > n {
		m = n
	}
	maxOuter := g.MaxIter
	if maxOuter <= 0 {
		maxOuter = (3*n + m - 1) / m
		if maxOuter < 1 {
			maxOuter = 1
		}
	}

	bnorm := mat.Norm(F, 2)
	if bnorm == 0 {
		return mat.NewVecDense(n, nil), nil
	}

	x := mat.NewVecDense(n, nil)
	r := mat.NewVecDense(n, nil)
	w := mat.NewVecDense(n, nil)
	tmp := mat.NewVecDense(n, nil)

	// Arnoldi basis Q[0..m] and upper-Hessenberg matrix H (m+1)×m.
	Q := make([]*mat.VecDense, m+1)
	for i := range Q {
		Q[i] = mat.NewVecDense(n, nil)
	}
	H := make([][]float64, m+1)
	for i := range H {
		H[i] = make([]float64, m)
	}
	cs := make([]float64, m) // Givens cosines
	sn := make([]float64, m) // Givens sines
	gv := make([]float64, m+1) // rotated RHS of the least-squares problem

	for outer := 0; outer < maxOuter; outer++ {
		// r = F − K x
		tmp.MulVec(K, x)
		r.SubVec(F, tmp)
		rnorm := mat.Norm(r, 2)
		if rnorm <= tol*bnorm {
			return x, nil
		}

		Q[0].ScaleVec(1/rnorm, r)
		for i := range gv {
			gv[i] = 0
		}
		gv[0] = rnorm

		kSteps := m
		for j := 0; j < m; j++ {
			// Arnoldi step: w = K Q[j];  modified Gram-Schmidt orthogonalisation.
			w.MulVec(K, Q[j])
			for i := 0; i <= j; i++ {
				H[i][j] = mat.Dot(w, Q[i])
				w.AddScaledVec(w, -H[i][j], Q[i])
			}
			H[j+1][j] = mat.Norm(w, 2)
			if H[j+1][j] > 1e-14 {
				Q[j+1].ScaleVec(1/H[j+1][j], w)
			}

			// Apply all previous Givens rotations to column j of H.
			for i := 0; i < j; i++ {
				hij, hi1j := H[i][j], H[i+1][j]
				H[i][j] = cs[i]*hij + sn[i]*hi1j
				H[i+1][j] = -sn[i]*hij + cs[i]*hi1j
			}

			// New Givens rotation to annihilate H[j+1][j].
			hjj, hj1j := H[j][j], H[j+1][j]
			denom := math.Hypot(hjj, hj1j)
			if denom < 1e-14 {
				cs[j], sn[j] = 1, 0
			} else {
				cs[j] = hjj / denom
				sn[j] = hj1j / denom
			}
			H[j][j] = denom
			H[j+1][j] = 0

			gvj := gv[j]
			gv[j] = cs[j] * gvj
			gv[j+1] = -sn[j] * gvj

			if math.Abs(gv[j+1]) <= tol*bnorm {
				kSteps = j + 1
				break
			}
		}

		// Back-substitution: solve H[0:kSteps,0:kSteps] · y = gv[0:kSteps].
		y := make([]float64, kSteps)
		for i := kSteps - 1; i >= 0; i-- {
			y[i] = gv[i]
			for k := i + 1; k < kSteps; k++ {
				y[i] -= H[i][k] * y[k]
			}
			if math.Abs(H[i][i]) < 1e-14 {
				return nil, fmt.Errorf("gmres: singular upper Hessenberg at step %d", i)
			}
			y[i] /= H[i][i]
		}

		// x += Q[0:kSteps] · y
		for i := 0; i < kSteps; i++ {
			x.AddScaledVec(x, y[i], Q[i])
		}
	}

	// Final true-residual check.
	tmp.MulVec(K, x)
	r.SubVec(F, tmp)
	relRes := mat.Norm(r, 2) / bnorm
	if relRes <= tol {
		return x, nil
	}
	return x, fmt.Errorf("gmres: did not converge after %d restarts (relative residual = %g)",
		maxOuter, relRes)
}

