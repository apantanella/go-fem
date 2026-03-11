package quad_test

import (
	"math"
	"testing"

	"go-fem/element/quad"
)

// unit square Quad8 with corners at (0,0),(1,0),(1,1),(0,1) and midside nodes.
func unitSquareQuad8() [8][2]float64 {
	return [8][2]float64{
		{0, 0}, {1, 0}, {1, 1}, {0, 1}, // corners
		{0.5, 0}, {1, 0.5}, {0.5, 1}, {0, 0.5}, // midsides
	}
}

func TestQuad8Symmetry(t *testing.T) {
	nodes := [8]int{0, 1, 2, 3, 4, 5, 6, 7}
	coords := unitSquareQuad8()
	q := quad.NewQuad8(0, nodes, coords, 210000, 0.3, 1.0, quad.PlaneStress)
	ke := q.GetTangentStiffness()
	for i := 0; i < 16; i++ {
		for j := i + 1; j < 16; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-6 {
				t.Errorf("Ke not symmetric at [%d,%d]: %.6g vs %.6g", i, j, ke.At(i, j), ke.At(j, i))
			}
		}
	}
}

// TestQuad8PatchTest verifies that a uniform stress state is reproduced exactly
// (patch test):  unit square in plane stress, σxx = E·εxx, εxx = 1e-3,
// applied via prescribed nodal displacements ux = εxx·x.
func TestQuad8PatchTest(t *testing.T) {
	E, nu := 210000.0, 0.3
	nodes := [8]int{0, 1, 2, 3, 4, 5, 6, 7}
	coords := unitSquareQuad8()
	q := quad.NewQuad8(0, nodes, coords, E, nu, 1.0, quad.PlaneStress)

	// Apply uniaxial strain εxx = 1e-3, εyy = -nu*εxx (free lateral)
	eps := 1e-3
	var u [16]float64
	for n := 0; n < 8; n++ {
		u[2*n] = eps * coords[n][0]     // ux = εxx·x
		u[2*n+1] = -nu * eps * coords[n][1] // uy = -ν·εxx·y (Poisson contraction)
	}
	q.Update(u[:])
	s := q.StressCentroid()
	// For plane stress uniaxial with free Poisson: σxx = E·εxx, σyy = 0
	// We prescribed εyy = -ν·εxx, which is the true uniaxial stress state.
	expected := E * eps
	if math.Abs(s[0]-expected)/expected > 1e-10 {
		t.Errorf("σxx = %.6g, want %.6g", s[0], expected)
	}
	if math.Abs(s[1]) > 1e-6 {
		t.Errorf("σyy = %.6g, want 0 (free lateral)", s[1])
	}
	if math.Abs(s[2]) > 1e-6 {
		t.Errorf("τxy = %.6g, want 0", s[2])
	}
}

// TestQuad8SofterThanQuad4 for a mesh with a single element, the Quad8 is
// equivalent to Quad4 on regular meshes (constant strain field captured
// exactly by both). We verify the centroidal stress for a pure shear case.
func TestQuad8PureShear(t *testing.T) {
	E, nu := 210000.0, 0.3
	G := E / (2 * (1 + nu))
	nodes := [8]int{0, 1, 2, 3, 4, 5, 6, 7}
	coords := unitSquareQuad8()
	q := quad.NewQuad8(0, nodes, coords, E, nu, 1.0, quad.PlaneStress)

	// Pure shear: γxy = 1e-3, ux = γ·y, uy = 0
	gamma := 1e-3
	var u [16]float64
	for n := 0; n < 8; n++ {
		u[2*n] = gamma * coords[n][1] // ux = γ·y
		u[2*n+1] = 0
	}
	q.Update(u[:])
	s := q.StressCentroid()
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

// TestQuad8RigidBody verifies that rigid-body displacement produces zero stress.
func TestQuad8RigidBody(t *testing.T) {
	nodes := [8]int{0, 1, 2, 3, 4, 5, 6, 7}
	coords := unitSquareQuad8()
	q := quad.NewQuad8(0, nodes, coords, 210000, 0.3, 1.0, quad.PlaneStress)

	// Rigid body translation: ux = 1, uy = 2
	var u [16]float64
	for n := 0; n < 8; n++ {
		u[2*n] = 1.0
		u[2*n+1] = 2.0
	}
	q.Update(u[:])
	s := q.StressCentroid()
	for i, v := range s {
		if math.Abs(v) > 1e-8 {
			t.Errorf("stress[%d] under rigid body = %.6g, want 0", i, v)
		}
	}
}
