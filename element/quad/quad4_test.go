package quad

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestQuad4_Symmetry(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][2]float64{{0, 0}, {2, 0}, {2, 1}, {0, 1}}
	q := NewQuad4(0, nodes, coords, 200000, 0.3, 1.0, PlaneStress)

	ke := q.GetTangentStiffness()
	for i := 0; i < 8; i++ {
		for j := i + 1; j < 8; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			avg := (math.Abs(ke.At(i, j)) + math.Abs(ke.At(j, i))) / 2
			if avg > 1e-10 && diff/avg > 1e-6 {
				t.Errorf("not symmetric: K[%d,%d]=%v != K[%d,%d]=%v",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

func TestQuad4_RigidBody(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	q := NewQuad4(0, nodes, coords, 200000, 0.3, 0.1, PlaneStress)

	ke := q.GetTangentStiffness()

	// Uniform translation
	for dir := 0; dir < 2; dir++ {
		u := mat.NewVecDense(8, nil)
		for n := 0; n < 4; n++ {
			u.SetVec(2*n+dir, 1)
		}

		f := mat.NewVecDense(8, nil)
		f.MulVec(ke, u)

		for i := 0; i < 8; i++ {
			if math.Abs(f.AtVec(i)) > 1e-6 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

func TestQuad4_PositiveDiag(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	q := NewQuad4(0, nodes, coords, 200000, 0.3, 0.1, PlaneStress)

	ke := q.GetTangentStiffness()
	for i := 0; i < 8; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d] = %v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}

func TestQuad4_PlaneStrainVsStress(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}

	qs := NewQuad4(0, nodes, coords, 200000, 0.3, 0.1, PlaneStress)
	qe := NewQuad4(0, nodes, coords, 200000, 0.3, 0.1, PlaneStrain)

	// Plane strain should be stiffer than plane stress
	if qs.GetTangentStiffness().At(0, 0) >= qe.GetTangentStiffness().At(0, 0) {
		t.Error("plane strain should be stiffer than plane stress")
	}
}
