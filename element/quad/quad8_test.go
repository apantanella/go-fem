package quad_test

import (
	"math"
	"testing"

	"go-fem/element/quad"
)

// unitSquareQuad8 returns the nodal coordinates of a unit square Quad8 element.
//
// The 8-node serendipity element has 4 corner nodes and 4 midside nodes:
//   - Corners: (0,0), (1,0), (1,1), (0,1)  (nodes 0–3)
//   - Midsides: (0.5,0), (1,0.5), (0.5,1), (0,0.5)  (nodes 4–7)
//
// This unit square geometry is used across all Quad8 tests because the
// Jacobian is constant (identity) and closed-form stress results are available.
func unitSquareQuad8() [8][2]float64 {
	return [8][2]float64{
		{0, 0}, {1, 0}, {1, 1}, {0, 1}, // corners
		{0.5, 0}, {1, 0.5}, {0.5, 1}, {0, 0.5}, // midsides
	}
}

// TestQuad8Symmetry verifies that the 16×16 element stiffness matrix of the
// 8-node serendipity quadrilateral (Quad8) is symmetric.
//
// Property: Ke = Keᵀ. For Ke = t·∫ Bᵀ·D·B dA with D symmetric this holds
// analytically; the 3×3 Gauss quadrature used for the serendipity element
// must not break it.
//
// Parameters: unit square, E=210000, ν=0.3, thickness=1.0, plane stress.
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-6 for all i < j.
//
// Why valuable: catches errors in the serendipity shape-function derivative
// assembly; a non-unit Jacobian or mismatched indices would break symmetry.
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
// by the Quad8 element under prescribed nodal displacements (patch test).
//
// Physical state: uniaxial stress σxx = E·εxx with free Poisson contraction.
// The displacement field is:
//
//	ux = εxx · x
//	uy = -ν · εxx · y   (Poisson contraction)
//
// where εxx = 1e-3. Under this field the stresses must be:
//
//	σxx = E · εxx = 210000 · 1e-3 = 210
//	σyy = 0  (free lateral: prescribed contraction annuls Poisson coupling)
//	τxy = 0  (no shear strain)
//
// Parameters: E=210000, ν=0.3, unit square, plane stress.
//
// Why valuable: the patch test is the fundamental completeness criterion for
// finite elements. Failure would mean the Quad8 cannot represent a constant
// stress state exactly, which would prevent convergence under mesh refinement.
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

// TestQuad8PureShear verifies that the Quad8 element correctly reproduces a
// pure shear stress state under a prescribed shear displacement field.
//
// Physical state: pure shear with engineering shear strain γxy = 1e-3.
// The displacement field is:
//
//	ux = γxy · y
//	uy = 0
//
// which gives ε = [0, 0, γ/2] (Voigt notation with engineering shear).
// The resulting stresses must be:
//
//	σxx = 0, σyy = 0, τxy = G · γxy
//	where G = E / (2(1+ν))
//
// Parameters: E=210000, ν=0.3, unit square, plane stress.
// G = 210000 / (2·1.3) ≈ 80769.
//
// Why valuable: catches sign errors in the shear strain component of the B-matrix
// and verifies that normal stresses do not couple into pure shear for an
// isotropic material.
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

// TestQuad8RigidBody verifies that a rigid-body translation produces zero stress
// in the Quad8 element.
//
// Property: rigid translation (ux=const, uy=const) implies zero strain
// everywhere (all shape-function gradients sum to zero by partition of unity),
// so all stress components must be zero.
//
// Parameters: unit square, E=210000, ν=0.3, plane stress.
// Translation: ux=1, uy=2 applied to all 8 nodes.
//
// Expected: |σ[i]| < 1e-8 for all stress components (σxx, σyy, τxy).
//
// Why valuable: confirms the partition-of-unity property of the serendipity
// shape functions and verifies that the strain computation does not produce
// spurious gradients from a constant displacement field.
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
