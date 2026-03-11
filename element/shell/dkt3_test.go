package shell

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestDKT3_Symmetry(t *testing.T) {
	nodes := [3]int{0, 1, 2}
	coords := [3][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	e := NewDiscreteKirchhoffTriangle(0, nodes, coords, 200000, 0.3, 0.1)

	ke := e.GetTangentStiffness()
	r, c := ke.Dims()
	for i := 0; i < r; i++ {
		for j := i + 1; j < c; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			avg := (math.Abs(ke.At(i, j)) + math.Abs(ke.At(j, i))) / 2
			if avg > 1e-10 && diff/avg > 1e-6 {
				t.Errorf("Ke not symmetric: K[%d,%d]=%v != K[%d,%d]=%v", i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

func TestDKT3_RigidBodyModes(t *testing.T) {
	nodes := [3]int{0, 1, 2}
	coords := [3][3]float64{{0, 0, 0}, {2, 0, 0}, {0, 1, 0}}
	e := NewDiscreteKirchhoffTriangle(0, nodes, coords, 200000, 0.3, 0.05)
	ke := e.GetTangentStiffness()

	// Rigid out-of-plane translation: w = const
	uW := mat.NewVecDense(18, nil)
	for n := 0; n < 3; n++ {
		uW.SetVec(6*n+2, 1.0)
	}
	fW := mat.NewVecDense(18, nil)
	fW.MulVec(ke, uW)
	for i := 0; i < 18; i++ {
		if math.Abs(fW.AtVec(i)) > 1e-3 {
			t.Errorf("rigid w translation: f[%d]=%v, want ~0", i, fW.AtVec(i))
		}
	}

	// Rigid rotation-compatible field: w = a*y, rx = a, ry = 0
	a := 0.2
	uR := mat.NewVecDense(18, nil)
	for n := 0; n < 3; n++ {
		y := coords[n][1]
		uR.SetVec(6*n+2, a*y)
		uR.SetVec(6*n+3, a)
		uR.SetVec(6*n+4, 0)
	}
	fR := mat.NewVecDense(18, nil)
	fR.MulVec(ke, uR)
	for i := 0; i < 18; i++ {
		if math.Abs(fR.AtVec(i)) > 1e-3 {
			t.Errorf("rigid rotation field: f[%d]=%v, want ~0", i, fR.AtVec(i))
		}
	}
}

func TestDKT3_PositiveDiagonal(t *testing.T) {
	nodes := [3]int{0, 1, 2}
	coords := [3][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	e := NewDiscreteKirchhoffTriangle(0, nodes, coords, 200000, 0.3, 0.1)

	ke := e.GetTangentStiffness()
	for i := 0; i < 18; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d]=%v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}
