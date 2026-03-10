package shell

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestShellMITC4_Symmetry(t *testing.T) {
	// Flat shell in XY plane
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{2, 0, 0},
		{2, 1, 0},
		{0, 1, 0},
	}
	s := NewShellMITC4(0, nodes, coords, 200000, 0.3, 0.1)

	ke := s.GetTangentStiffness()
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

func TestShellMITC4_RigidBodyTranslation(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 0},
		{0, 1, 0},
	}
	s := NewShellMITC4(0, nodes, coords, 200000, 0.3, 0.05)

	ke := s.GetTangentStiffness()

	// Uniform translation should produce zero forces
	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(24, nil)
		for n := 0; n < 4; n++ {
			u.SetVec(n*6+dir, 1.0)
		}

		f := mat.NewVecDense(24, nil)
		f.MulVec(ke, u)

		for i := 0; i < 24; i++ {
			if math.Abs(f.AtVec(i)) > 1e-3 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

func TestShellMITC4_PositiveDiag(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 0},
		{0, 1, 0},
	}
	s := NewShellMITC4(0, nodes, coords, 200000, 0.3, 0.1)

	ke := s.GetTangentStiffness()
	for i := 0; i < 24; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d] = %v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}
