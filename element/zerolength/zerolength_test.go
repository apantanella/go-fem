package zerolength

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestZeroLength_Symmetry(t *testing.T) {
	z := NewZeroLength(0, [2]int{0, 1}, [6]float64{100, 200, 300, 10, 20, 30})
	ke := z.GetTangentStiffness()
	for i := 0; i < 12; i++ {
		for j := i + 1; j < 12; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-10 {
				t.Errorf("not symmetric: K[%d,%d]=%v != K[%d,%d]=%v",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

func TestZeroLength_SpringStiffness(t *testing.T) {
	springs := [6]float64{100, 200, 300, 10, 20, 30}
	z := NewZeroLength(0, [2]int{0, 1}, springs)
	ke := z.GetTangentStiffness()

	// Check diagonal blocks
	for i := 0; i < 6; i++ {
		if got := ke.At(i, i); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d] = %v, want %v", i, i, got, springs[i])
		}
		if got := ke.At(i+6, i+6); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d] = %v, want %v", i+6, i+6, got, springs[i])
		}
		if got := ke.At(i, i+6); math.Abs(got-(-springs[i])) > 1e-10 {
			t.Errorf("K[%d,%d] = %v, want %v", i, i+6, got, -springs[i])
		}
	}
}

func TestZeroLength_RigidBody(t *testing.T) {
	z := NewZeroLength(0, [2]int{0, 1}, [6]float64{100, 200, 300, 10, 20, 30})
	ke := z.GetTangentStiffness()

	// Equal displacements at both nodes → zero forces
	for dir := 0; dir < 6; dir++ {
		u := mat.NewVecDense(12, nil)
		u.SetVec(dir, 1)
		u.SetVec(dir+6, 1)

		f := mat.NewVecDense(12, nil)
		f.MulVec(ke, u)

		for i := 0; i < 12; i++ {
			if math.Abs(f.AtVec(i)) > 1e-10 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want 0", dir, i, f.AtVec(i))
			}
		}
	}
}
