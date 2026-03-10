package solid

import (
	"math"
	"testing"

	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

func TestTet10_Symmetry(t *testing.T) {
	tet := makeTet10()
	ke := tet.GetTangentStiffness()
	r, c := ke.Dims()
	for i := 0; i < r; i++ {
		for j := i + 1; j < c; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			avg := (math.Abs(ke.At(i, j)) + math.Abs(ke.At(j, i))) / 2
			if avg > 1e-10 && diff/avg > 1e-6 {
				t.Errorf("Ke not symmetric: K[%d,%d]=%v != K[%d,%d]=%v",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

func TestTet10_RigidBody(t *testing.T) {
	tet := makeTet10()
	ke := tet.GetTangentStiffness()

	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(30, nil)
		for n := 0; n < 10; n++ {
			u.SetVec(3*n+dir, 1)
		}

		f := mat.NewVecDense(30, nil)
		f.MulVec(ke, u)

		for i := 0; i < 30; i++ {
			if math.Abs(f.AtVec(i)) > 1e-4 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

func TestTet10_PositiveDiag(t *testing.T) {
	tet := makeTet10()
	ke := tet.GetTangentStiffness()
	for i := 0; i < 30; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d] = %v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}

// makeTet10 creates a standard 10-node tet with corner nodes at unit tet
// and midside nodes at edge midpoints.
func makeTet10() *Tet10 {
	nodes := [10]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	coords := [10][3]float64{
		{0, 0, 0},    // 0 - corner
		{1, 0, 0},    // 1 - corner
		{0, 1, 0},    // 2 - corner
		{0, 0, 1},    // 3 - corner
		{0.5, 0, 0},  // 4 - midside 0-1
		{0.5, 0.5, 0}, // 5 - midside 1-2
		{0, 0.5, 0},  // 6 - midside 0-2
		{0, 0, 0.5},  // 7 - midside 0-3
		{0.5, 0, 0.5}, // 8 - midside 1-3
		{0, 0.5, 0.5}, // 9 - midside 2-3
	}
	m := material.NewIsotropicLinear(200000, 0.3)
	return NewTet10(0, nodes, coords, m)
}
