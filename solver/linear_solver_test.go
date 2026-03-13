package solver_test

import (
	"math"
	"testing"

	"go-fem/solver"

	"gonum.org/v1/gonum/mat"
)

// buildSPD returns a simple n×n symmetric positive-definite stiffness matrix
// (tridiagonal: 2 on diagonal, -1 on off-diagonal) and an RHS F = ones.
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

func TestCGNotSPD(t *testing.T) {
	// Indefinite matrix: diagonal entry forced negative.
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

func TestGMRESNonSymmetric(t *testing.T) {
	// Non-symmetric 4×4 system.
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
