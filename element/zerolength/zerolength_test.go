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

// ── ZeroLength2D ─────────────────────────────────────────────────────────────

func TestZeroLength2D_Symmetry(t *testing.T) {
	z := NewZeroLength2D(0, [2]int{0, 1}, [2]float64{100, 200})
	ke := z.GetTangentStiffness()
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-10 {
				t.Errorf("not symmetric K[%d,%d]=%v != K[%d,%d]=%v", i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

func TestZeroLength2D_SpringStiffness(t *testing.T) {
	springs := [2]float64{100, 200}
	z := NewZeroLength2D(0, [2]int{0, 1}, springs)
	ke := z.GetTangentStiffness()
	for i := 0; i < 2; i++ {
		if got := ke.At(i, i); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i, i, got, springs[i])
		}
		if got := ke.At(i+2, i+2); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i+2, i+2, got, springs[i])
		}
		if got := ke.At(i, i+2); math.Abs(got-(-springs[i])) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i, i+2, got, -springs[i])
		}
	}
}

func TestZeroLength2D_SpringForce(t *testing.T) {
	z := NewZeroLength2D(0, [2]int{0, 1}, [2]float64{100, 200})
	// node 1 moves +2 in X, +3 in Y relative to node 0
	_ = z.Update([]float64{0, 0, 2, 3})
	f := z.SpringForce()
	if math.Abs(f[0]-200) > 1e-10 {
		t.Errorf("Fx=%v want 200", f[0])
	}
	if math.Abs(f[1]-600) > 1e-10 {
		t.Errorf("Fy=%v want 600", f[1])
	}
}

func TestZeroLength2D_RigidBody(t *testing.T) {
	z := NewZeroLength2D(0, [2]int{0, 1}, [2]float64{100, 200})
	ke := z.GetTangentStiffness()
	for dir := 0; dir < 2; dir++ {
		u := mat.NewVecDense(4, nil)
		u.SetVec(dir, 1)
		u.SetVec(dir+2, 1)
		f := mat.NewVecDense(4, nil)
		f.MulVec(ke, u)
		for i := 0; i < 4; i++ {
			if math.Abs(f.AtVec(i)) > 1e-10 {
				t.Errorf("rigid body dir %d: f[%d]=%v want 0", dir, i, f.AtVec(i))
			}
		}
	}
}

// ── ZeroLength2DFrame ─────────────────────────────────────────────────────────

func TestZeroLength2DFrame_Symmetry(t *testing.T) {
	z := NewZeroLength2DFrame(0, [2]int{0, 1}, [3]float64{100, 200, 50})
	ke := z.GetTangentStiffness()
	for i := 0; i < 6; i++ {
		for j := i + 1; j < 6; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-10 {
				t.Errorf("not symmetric K[%d,%d]=%v != K[%d,%d]=%v", i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

func TestZeroLength2DFrame_SpringStiffness(t *testing.T) {
	springs := [3]float64{100, 200, 50}
	z := NewZeroLength2DFrame(0, [2]int{0, 1}, springs)
	ke := z.GetTangentStiffness()
	for i := 0; i < 3; i++ {
		if got := ke.At(i, i); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i, i, got, springs[i])
		}
		if got := ke.At(i+3, i+3); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i+3, i+3, got, springs[i])
		}
		if got := ke.At(i, i+3); math.Abs(got-(-springs[i])) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i, i+3, got, -springs[i])
		}
	}
}

func TestZeroLength2DFrame_SpringForce(t *testing.T) {
	z := NewZeroLength2DFrame(0, [2]int{0, 1}, [3]float64{100, 200, 50})
	// node 1 moves: ux=1, uy=2, rz=0.1 relative to node 0
	_ = z.Update([]float64{0, 0, 0, 1, 2, 0.1})
	f := z.SpringForce()
	if math.Abs(f[0]-100) > 1e-10 {
		t.Errorf("Fx=%v want 100", f[0])
	}
	if math.Abs(f[1]-400) > 1e-10 {
		t.Errorf("Fy=%v want 400", f[1])
	}
	if math.Abs(f[2]-5) > 1e-10 {
		t.Errorf("Mz=%v want 5", f[2])
	}
}

func TestZeroLength2DFrame_RigidBody(t *testing.T) {
	z := NewZeroLength2DFrame(0, [2]int{0, 1}, [3]float64{100, 200, 50})
	ke := z.GetTangentStiffness()
	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(6, nil)
		u.SetVec(dir, 1)
		u.SetVec(dir+3, 1)
		f := mat.NewVecDense(6, nil)
		f.MulVec(ke, u)
		for i := 0; i < 6; i++ {
			if math.Abs(f.AtVec(i)) > 1e-10 {
				t.Errorf("rigid body dir %d: f[%d]=%v want 0", dir, i, f.AtVec(i))
			}
		}
	}
}
