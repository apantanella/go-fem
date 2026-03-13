package quad_test

import (
	"math"
	"testing"

	"go-fem/element/quad"
)

// ---------- Tri6 tests ----------

// unitTri6 returns the nodal coordinates of a unit right triangle with midside
// nodes at edge midpoints, used for Tri6 (LST) element tests.
//
// Node layout:
//   - 0: corner (0, 0)
//   - 1: corner (1, 0)
//   - 2: corner (0, 1)
//   - 3: midside 0–1 at (0.5, 0)
//   - 4: midside 1–2 at (0.5, 0.5)
//   - 5: midside 2–0 at (0, 0.5)
//
// The unit right triangle has area 1/2, which allows simple closed-form
// verification of stiffness entries.
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

// TestTri6_Symmetry verifies that the 12×12 element stiffness matrix of the
// 6-node linear-strain triangle (LST / Tri6) is symmetric.
//
// Property: Ke = Keᵀ. For Ke = t · ∫ Bᵀ·D·B dA with D symmetric, symmetry
// follows analytically. The 3-point Gauss quadrature rule used for LST must
// preserve this.
//
// Parameters: unit right triangle via unitTri6(), E=200000, ν=0.3,
// thickness=1.0, plane stress.
//
// Expected: relative asymmetry < 1e-6 for entries with avg > 1e-10.
//
// Why valuable: catches errors in the quadratic shape-function derivative
// assembly or in the Gauss-point weight application.
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

// TestTri6_RigidBody verifies that a uniform rigid-body translation produces
// zero nodal forces for the Tri6 element.
//
// Property: Ke · u_rigid = 0 for u_rigid applying unit displacement in the X or Y
// direction to all 6 nodes (including midside nodes). The quadratic shape
// functions of the LST element must reproduce a constant displacement field
// exactly (completeness requirement).
//
// Parameters: unit right triangle via unitTri6(), E=200000, ν=0.3, plane stress.
// A manual matrix-vector product is used.
//
// Expected: |f[i]| < 1e-6 for all 12 force components.
//
// Why valuable: a violation would indicate that the quadratic shape-function
// gradients do not sum to zero (partition-of-unity failure), causing spurious
// forces under rigid-body translation.
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

// TestTri6_PositiveDiag verifies that all 12 diagonal entries of the Tri6
// stiffness matrix are strictly positive.
//
// Property: each nodal DOF (UX and UY for each of the 6 nodes, including 3
// midside nodes) must have a positive self-stiffness contribution.
//
// Parameters: unit right triangle via unitTri6(), E=200000, ν=0.3, plane stress.
//
// Why valuable: a zero diagonal for a midside node DOF would indicate that
// the corresponding quadratic shape function produces no strain energy,
// which could arise from a Gauss-rule with insufficient order.
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

// TestTri6_PlaneStrainStiffer verifies that the plane strain Tri6 element is
// stiffer than the plane stress element for the same geometry and material.
//
// Physical property: constraining εzz=0 in plane strain generates an additional
// out-of-plane stress σzz that increases the effective in-plane normal stiffness
// via the (1-ν)/ν coupling term. The same inequality holds for both CST and LST.
//
// Parameters: unit right triangle via unitTri6(), E=200000, ν=0.3.
//
// Expected: K_strain[0,0] > K_stress[0,0].
//
// Why valuable: if the plane strain constitutive matrix were computed incorrectly
// (e.g., using a plane stress formula), this comparison would fail.
func TestTri6_PlaneStrainStiffer(t *testing.T) {
	nodes := [6]int{0, 1, 2, 3, 4, 5}
	coords := unitTri6()
	ts := quad.NewTri6(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStress)
	te := quad.NewTri6(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStrain)

	if ts.GetTangentStiffness().At(0, 0) >= te.GetTangentStiffness().At(0, 0) {
		t.Error("plane strain should be stiffer than plane stress")
	}
}

// TestTri6_PatchTest verifies that the Tri6 (LST) element reproduces a uniform
// uniaxial stress state exactly — the patch test for the linear-strain triangle.
//
// Physical state: uniaxial strain εxx = 1e-3 with free Poisson contraction.
// The displacement field is:
//
//	ux = εxx · x,  uy = -ν · εxx · y
//
// This is a linear (not just constant) field, which the LST reproduces exactly.
// The stresses must be:
//
//	σxx = E · εxx = 210,  σyy = 0,  τxy = 0
//
// Parameters: unit right triangle via unitTri6(), E=210000, ν=0.3, plane stress.
//
// Why valuable: the LST patch test verifies completeness to linear order. A
// failure would indicate that the midside nodes are not correctly incorporated
// into the strain interpolation, which would prevent O(h²) convergence.
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

// TestTri6_PureShear verifies that the Tri6 element correctly reproduces a
// pure shear stress state under a prescribed shear displacement field.
//
// Physical state: engineering shear strain γxy = 1e-3.
// The displacement field is:
//
//	ux = γxy · y,  uy = 0
//
// This is a linearly varying field that the LST element represents exactly.
// The stresses must be:
//
//	σxx = 0,  σyy = 0,  τxy = G · γxy
//	where G = E / (2(1+ν))
//
// Parameters: unit right triangle via unitTri6(), E=210000, ν=0.3, plane stress.
//
// Why valuable: confirms the shear component of the LST B-matrix is correctly
// assembled for a linearly varying shear field (which involves contributions
// from the midside nodes).
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
