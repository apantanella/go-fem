package quad_test

import (
	"math"
	"testing"

	"go-fem/element/quad"
)

// ---------- Tri6 tests ----------

// unitTri6 returns a right triangle with midside nodes at midpoints.
func unitTri6() [6][2]float64 {
	return [6][2]float64{
		{0, 0},     // 0 corner
		{1, 0},     // 1 corner
		{0, 1},     // 2 corner
		{0.5, 0},   // 3 midside 0-1
		{0.5, 0.5}, // 4 midside 1-2
		{0, 0.5},   // 5 midside 2-0
	}
}

func TestTri6_Symmetry(t *testing.T) {
	nodes := [6]int{0, 1, 2, 3, 4, 5}
	coords := unitTri6()
	tri := quad.NewTri6(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStress)

	ke := tri.GetTangentStiffness()
	for i := 0; i < 12; i++ {
		for j := i + 1; j < 12; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			avg := (math.Abs(ke.At(i, j)) + math.Abs(ke.At(j, i))) / 2
			if avg > 1e-10 && diff/avg > 1e-6 {
				t.Errorf("not symmetric: K[%d,%d]=%v != K[%d,%d]=%v",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

func TestTri6_RigidBody(t *testing.T) {
	nodes := [6]int{0, 1, 2, 3, 4, 5}
	coords := unitTri6()
	tri := quad.NewTri6(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStress)
	ke := tri.GetTangentStiffness()

	for dir := 0; dir < 2; dir++ {
		u := make([]float64, 12)
		for n := 0; n < 6; n++ {
			u[2*n+dir] = 1.0
		}
		fvec := make([]float64, 12)
		for i := 0; i < 12; i++ {
			for j := 0; j < 12; j++ {
				fvec[i] += ke.At(i, j) * u[j]
			}
		}
		for i := 0; i < 12; i++ {
			if math.Abs(fvec[i]) > 1e-6 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, fvec[i])
			}
		}
	}
}

func TestTri6_PositiveDiag(t *testing.T) {
	nodes := [6]int{0, 1, 2, 3, 4, 5}
	coords := unitTri6()
	tri := quad.NewTri6(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStress)
	ke := tri.GetTangentStiffness()
	for i := 0; i < 12; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d] = %v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}

func TestTri6_PlaneStrainStiffer(t *testing.T) {
	nodes := [6]int{0, 1, 2, 3, 4, 5}
	coords := unitTri6()
	ts := quad.NewTri6(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStress)
	te := quad.NewTri6(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStrain)

	if ts.GetTangentStiffness().At(0, 0) >= te.GetTangentStiffness().At(0, 0) {
		t.Error("plane strain should be stiffer than plane stress")
	}
}

func TestTri6_PatchTest(t *testing.T) {
	E, nu := 210000.0, 0.3
	eps := 1e-3
	nodes := [6]int{0, 1, 2, 3, 4, 5}
	coords := unitTri6()
	tri := quad.NewTri6(0, nodes, coords, E, nu, 1.0, quad.PlaneStress)

	var u [12]float64
	for n := 0; n < 6; n++ {
		u[2*n] = eps * coords[n][0]
		u[2*n+1] = -nu * eps * coords[n][1]
	}
	tri.Update(u[:])

	s := tri.StressCentroid()
	expected := E * eps
	if math.Abs(s[0]-expected)/expected > 1e-10 {
		t.Errorf("σxx = %.6g, want %.6g", s[0], expected)
	}
	if math.Abs(s[1]) > 1e-6 {
		t.Errorf("σyy = %.6g, want 0", s[1])
	}
	if math.Abs(s[2]) > 1e-6 {
		t.Errorf("τxy = %.6g, want 0", s[2])
	}
}

func TestTri6_PureShear(t *testing.T) {
	E, nu := 210000.0, 0.3
	G := E / (2 * (1 + nu))
	gamma := 1e-3
	nodes := [6]int{0, 1, 2, 3, 4, 5}
	coords := unitTri6()
	tri := quad.NewTri6(0, nodes, coords, E, nu, 1.0, quad.PlaneStress)

	var u [12]float64
	for n := 0; n < 6; n++ {
		u[2*n] = gamma * coords[n][1]
	}
	tri.Update(u[:])

	s := tri.StressCentroid()
	if math.Abs(s[0]) > 1e-6 {
		t.Errorf("σxx under pure shear = %.6g, want 0", s[0])
	}
	if math.Abs(s[1]) > 1e-6 {
		t.Errorf("σyy under pure shear = %.6g, want 0", s[1])
	}
	expected := G * gamma
	if math.Abs(s[2]-expected)/expected > 1e-10 {
		t.Errorf("τxy = %.6g, want G·γ = %.6g", s[2], expected)
	}
}
