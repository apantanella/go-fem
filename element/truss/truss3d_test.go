package truss

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestTruss3D_AxialStiffness(t *testing.T) {
	// Truss along X-axis, length=2, E=100, A=0.5
	// Expected: k = AE/L = 0.5*100/2 = 25
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {2, 0, 0}}
	tr := NewTruss3D(0, nodes, coords, 100, 0.5)

	ke := tr.GetTangentStiffness()

	// For an axial truss along X: K[0,0] = AE/L, K[3,3] = AE/L
	if got := ke.At(0, 0); math.Abs(got-25) > 1e-10 {
		t.Errorf("K[0,0] = %v, want 25", got)
	}
	if got := ke.At(3, 3); math.Abs(got-25) > 1e-10 {
		t.Errorf("K[3,3] = %v, want 25", got)
	}
	// Off-diagonal coupling
	if got := ke.At(0, 3); math.Abs(got-(-25)) > 1e-10 {
		t.Errorf("K[0,3] = %v, want -25", got)
	}
	// Y and Z DOFs should have zero stiffness for axial truss
	if got := ke.At(1, 1); math.Abs(got) > 1e-10 {
		t.Errorf("K[1,1] = %v, want 0", got)
	}
}

func TestTruss3D_Symmetry(t *testing.T) {
	// Diagonal truss
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {3, 4, 5}}
	tr := NewTruss3D(0, nodes, coords, 200000, 0.01)

	ke := tr.GetTangentStiffness()
	r, c := ke.Dims()
	for i := 0; i < r; i++ {
		for j := i + 1; j < c; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-6 {
				t.Errorf("Ke not symmetric: K[%d,%d]=%v != K[%d,%d]=%v",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

func TestTruss3D_RigidBody(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{1, 2, 3}, {4, 5, 6}}
	tr := NewTruss3D(0, nodes, coords, 200000, 0.01)

	ke := tr.GetTangentStiffness()

	// Uniform translation should produce zero forces
	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(6, nil)
		u.SetVec(dir, 1)
		u.SetVec(dir+3, 1)

		f := mat.NewVecDense(6, nil)
		f.MulVec(ke, u)

		for i := 0; i < 6; i++ {
			if math.Abs(f.AtVec(i)) > 1e-8 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want 0", dir, i, f.AtVec(i))
			}
		}
	}
}

func TestCorotTruss_LinearEquivalence(t *testing.T) {
	// For zero displacement, CorotTruss should equal Truss3D
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {3, 4, 0}}

	tr := NewTruss3D(0, nodes, coords, 200000, 0.01)
	ct := NewCorotTruss(0, nodes, coords, 200000, 0.01)

	ke1 := tr.GetTangentStiffness()
	ke2 := ct.GetTangentStiffness()

	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			if math.Abs(ke1.At(i, j)-ke2.At(i, j)) > 1e-6 {
				t.Errorf("K[%d,%d]: truss=%v, corot=%v", i, j, ke1.At(i, j), ke2.At(i, j))
			}
		}
	}
}
