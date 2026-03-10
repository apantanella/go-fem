package element

import (
	"math"
	"testing"

	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

// unitTet returns a right-angle tetrahedron at the origin with unit edges.
// Nodes: (0,0,0), (1,0,0), (0,1,0), (0,0,1).
func unitTet(e, nu float64) *Tet4 {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}
	m := material.NewIsotropicLinear(e, nu)
	return NewTet4(0, nodes, coords, m)
}

func TestTet4Volume(t *testing.T) {
	tet := unitTet(1, 0)
	want := 1.0 / 6.0
	if math.Abs(tet.Volume()-want) > 1e-14 {
		t.Errorf("Volume = %v, want %v", tet.Volume(), want)
	}
}

func TestTet4Symmetry(t *testing.T) {
	tet := unitTet(200000, 0.3)
	ke := tet.GetTangentStiffness()
	n := 12
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			if diff > 1e-10 {
				t.Errorf("Ke not symmetric: K[%d][%d]=%.6e, K[%d][%d]=%.6e, diff=%.2e",
					i, j, ke.At(i, j), j, i, ke.At(j, i), diff)
			}
		}
	}
}

func TestTet4RigidBody(t *testing.T) {
	tet := unitTet(200000, 0.3)
	ke := tet.GetTangentStiffness()

	// Three rigid-body translations: (1,0,0), (0,1,0), (0,0,1).
	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(12, nil)
		for n := 0; n < 4; n++ {
			u.SetVec(3*n+dir, 1.0)
		}
		f := mat.NewVecDense(12, nil)
		f.MulVec(ke, u)

		for i := 0; i < 12; i++ {
			if math.Abs(f.AtVec(i)) > 1e-10 {
				t.Errorf("Rigid body dir=%d: F[%d] = %.6e (want 0)", dir, i, f.AtVec(i))
			}
		}
	}
}

func TestTet4KnownDiagonal(t *testing.T) {
	// Unit tet, E=1, nu=0 → Ke[0][0] = V · (Bcol0ᵀ D Bcol0) = (1/6)·2 = 1/3.
	tet := unitTet(1, 0)
	ke := tet.GetTangentStiffness()
	want := 1.0 / 3.0
	got := ke.At(0, 0)
	if math.Abs(got-want) > 1e-14 {
		t.Errorf("Ke[0][0] = %v, want %v", got, want)
	}
}

func TestTet4PositiveDefiniteReduced(t *testing.T) {
	// After fixing node 0 (removing 3 DOFs for translation) and constraining
	// 3 more DOFs to remove rotations, the reduced Ke should be positive definite.
	// We test by checking all diagonal values of Ke are positive (necessary condition).
	tet := unitTet(200000, 0.3)
	ke := tet.GetTangentStiffness()
	for i := 0; i < 12; i++ {
		if ke.At(i, i) < 0 {
			t.Errorf("Ke[%d][%d] = %v < 0 (diagonal should be non-negative)", i, i, ke.At(i, i))
		}
	}
}
