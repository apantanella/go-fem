package solid

import (
	"math"
	"testing"

	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

func TestBrick20_Symmetry(t *testing.T) {
	b := makeBrick20()
	ke := b.GetTangentStiffness()
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

func TestBrick20_RigidBody(t *testing.T) {
	b := makeBrick20()
	ke := b.GetTangentStiffness()

	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(60, nil)
		for n := 0; n < 20; n++ {
			u.SetVec(3*n+dir, 1)
		}

		f := mat.NewVecDense(60, nil)
		f.MulVec(ke, u)

		for i := 0; i < 60; i++ {
			if math.Abs(f.AtVec(i)) > 1e-4 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

func TestBrick20_PositiveDiag(t *testing.T) {
	b := makeBrick20()
	ke := b.GetTangentStiffness()
	for i := 0; i < 60; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d] = %v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}

// makeBrick20 creates a unit cube [0,1]³ 20-node element.
func makeBrick20() *Brick20 {
	var nodes [20]int
	for i := 0; i < 20; i++ {
		nodes[i] = i
	}
	// Map reference coords [-1,1]³ to physical [0,1]³: x = (1+ξ)/2
	var coords [20][3]float64
	for i := 0; i < 20; i++ {
		coords[i][0] = (1 + brick20Ref[i][0]) / 2
		coords[i][1] = (1 + brick20Ref[i][1]) / 2
		coords[i][2] = (1 + brick20Ref[i][2]) / 2
	}
	m := material.NewIsotropicLinear(200000, 0.3)
	return NewBrick20(0, nodes, coords, m)
}
