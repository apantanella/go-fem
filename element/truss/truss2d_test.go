package truss

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestTruss2D_AxialStiffness(t *testing.T) {
	// Truss along X-axis, length=2, E=100, A=0.5
	// Expected: k = AE/L = 0.5*100/2 = 25
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {2, 0}}
	tr := NewTruss2D(0, nodes, coords, 100, 0.5)

	ke := tr.GetTangentStiffness()

	if got := ke.At(0, 0); math.Abs(got-25) > 1e-10 {
		t.Errorf("K[0,0] = %v, want 25", got)
	}
	if got := ke.At(2, 2); math.Abs(got-25) > 1e-10 {
		t.Errorf("K[2,2] = %v, want 25", got)
	}
	if got := ke.At(0, 2); math.Abs(got-(-25)) > 1e-10 {
		t.Errorf("K[0,2] = %v, want -25", got)
	}
	// Y DOFs should have zero stiffness for axial truss along X
	if got := ke.At(1, 1); math.Abs(got) > 1e-10 {
		t.Errorf("K[1,1] = %v, want 0", got)
	}
}

func TestTruss2D_Symmetry(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {3, 4}}
	tr := NewTruss2D(0, nodes, coords, 200000, 0.01)

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

func TestTruss2D_RigidBody(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{1, 2}, {4, 6}}
	tr := NewTruss2D(0, nodes, coords, 200000, 0.01)

	ke := tr.GetTangentStiffness()

	// Uniform translation should produce zero forces
	for dir := 0; dir < 2; dir++ {
		u := mat.NewVecDense(4, nil)
		u.SetVec(dir, 1)
		u.SetVec(dir+2, 1)

		f := mat.NewVecDense(4, nil)
		f.MulVec(ke, u)

		for i := 0; i < 4; i++ {
			if math.Abs(f.AtVec(i)) > 1e-8 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want 0", dir, i, f.AtVec(i))
			}
		}
	}
}

func TestTruss2D_DiagonalElement(t *testing.T) {
	// 45° truss: cos=sin=1/√2
	E, A := 200000.0, 0.01
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {1, 1}}
	tr := NewTruss2D(0, nodes, coords, E, A)

	L := math.Sqrt(2.0)
	k := E * A / L
	expected00 := k * 0.5 // k·cos²(45°) = k/2

	ke := tr.GetTangentStiffness()
	if got := ke.At(0, 0); math.Abs(got-expected00)/expected00 > 1e-10 {
		t.Errorf("K[0,0] = %v, want %v", got, expected00)
	}
	// Cross-coupling K[0,1] = k·cos·sin = k/2
	if got := ke.At(0, 1); math.Abs(got-expected00)/expected00 > 1e-10 {
		t.Errorf("K[0,1] = %v, want %v", got, expected00)
	}
}

func TestTruss2D_AxialForce(t *testing.T) {
	E, A := 100.0, 0.5
	L := 2.0
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	tr := NewTruss2D(0, nodes, coords, E, A)

	// Apply unit extension: node 1 moves 0.01 in X
	disp := []float64{0, 0, 0.01, 0}
	tr.Update(disp)

	// N = EA/L · Δu = 100*0.5/2 * 0.01 = 0.25
	N := tr.AxialForce()
	if math.Abs(N-0.25) > 1e-10 {
		t.Errorf("AxialForce = %v, want 0.25", N)
	}

	sigma := tr.AxialStress()
	if math.Abs(sigma-0.5) > 1e-10 {
		t.Errorf("AxialStress = %v, want 0.5", sigma)
	}
}

func TestTruss2D_MatchesTruss3D(t *testing.T) {
	// Verify that Truss2D produces the same in-plane stiffness as Truss3D
	E, A := 200000.0, 0.01
	n2 := [2]int{0, 1}
	c2 := [2][2]float64{{0, 0}, {3, 4}}
	c3 := [2][3]float64{{0, 0, 0}, {3, 4, 0}}

	tr2d := NewTruss2D(0, n2, c2, E, A)
	tr3d := NewTruss3D(0, n2, c3, E, A)

	ke2 := tr2d.GetTangentStiffness()
	ke3 := tr3d.GetTangentStiffness()

	// Compare UX/UY rows/cols: 2D DOFs (0,1,2,3) map to 3D DOFs (0,1,3,4)
	map2to3 := [4]int{0, 1, 3, 4}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			v2 := ke2.At(i, j)
			v3 := ke3.At(map2to3[i], map2to3[j])
			if math.Abs(v2-v3) > 1e-6 {
				t.Errorf("K2D[%d,%d]=%v != K3D[%d,%d]=%v",
					i, j, v2, map2to3[i], map2to3[j], v3)
			}
		}
	}
}
