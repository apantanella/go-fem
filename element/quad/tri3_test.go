package quad_test

import (
	"math"
	"testing"

	"go-fem/element/quad"
)

// ---------- Tri3 tests ----------

// TestTri3_Symmetry verifies that the 6×6 element stiffness matrix of the
// constant-strain triangle (CST / Tri3) is symmetric.
//
// Property: Ke = Keᵀ. For the CST element Ke = t·A·Bᵀ·D·B (constant over
// the element), symmetry follows directly from D being symmetric.
//
// Parameters: scalene triangle with nodes at (0,0), (2,0), (0,1); E=200000,
// ν=0.3, thickness=1.0, plane stress.
//
// Expected: relative asymmetry |Ke[i,j]-Ke[j,i]| / avg < 1e-6 for entries
// with avg > 1e-10.
//
// Why valuable: catches any non-symmetric assembly in the outer-product BᵀDB
// or an incorrect sign in the area computation.
func TestTri3_Symmetry(t *testing.T) {
	nodes := [3]int{0, 1, 2}
	coords := [3][2]float64{{0, 0}, {2, 0}, {0, 1}}
	tri := quad.NewTri3(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStress)

	ke := tri.GetTangentStiffness()
	for i := 0; i < 6; i++ {
		for j := i + 1; j < 6; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			avg := (math.Abs(ke.At(i, j)) + math.Abs(ke.At(j, i))) / 2
			if avg > 1e-10 && diff/avg > 1e-6 {
				t.Errorf("not symmetric: K[%d,%d]=%v != K[%d,%d]=%v",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

// TestTri3_RigidBody verifies that a uniform rigid-body translation produces
// zero nodal forces for the Tri3 (CST) element.
//
// Property: Ke · u_rigid = 0 for u_rigid applying unit displacement in the X
// or Y direction to all 3 nodes. For the constant-strain triangle this is
// checked by a direct manual matrix-vector product (not using gonum's MulVec),
// because the CST has constant B and the result must be exactly zero.
//
// Parameters: triangle with nodes at (0,0), (1,0), (0.5,0.8) (general scalene
// shape to exercise all terms).
//
// Expected: |f[i]| < 1e-6 for all 6 force components.
//
// Why valuable: confirms that the shape-function gradients in B sum to zero
// (partition of unity), so that no spurious forces are produced by rigid motion.
func TestTri3_RigidBody(t *testing.T) {
	nodes := [3]int{0, 1, 2}
	coords := [3][2]float64{{0, 0}, {1, 0}, {0.5, 0.8}}
	tri := quad.NewTri3(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStress)
	ke := tri.GetTangentStiffness()

	// Uniform translation in X and Y
	for dir := 0; dir < 2; dir++ {
		u := make([]float64, 6)
		for n := 0; n < 3; n++ {
			u[2*n+dir] = 1.0
		}
		f := ke.T() // just use ke directly
		fvec := make([]float64, 6)
		for i := 0; i < 6; i++ {
			for j := 0; j < 6; j++ {
				fvec[i] += ke.At(i, j) * u[j]
			}
		}
		for i := 0; i < 6; i++ {
			if math.Abs(fvec[i]) > 1e-6 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, fvec[i])
			}
		}
		_ = f
	}
}

// TestTri3_PositiveDiag verifies that all 6 diagonal entries of the Tri3
// stiffness matrix are strictly positive.
//
// Property: each nodal DOF (UX and UY for each of the 3 nodes) must have a
// positive self-stiffness contribution (necessary condition for positive
// semi-definiteness).
//
// Parameters: triangle at (0,0)-(1,0)-(0.5,0.8), E=200000, ν=0.3, plane stress.
//
// Why valuable: a zero diagonal for a DOF would indicate an unrestrained
// degree of freedom in the element, which would cause singularity in the
// assembled stiffness unless boundary conditions are applied — but the
// element itself should resist deformation in all directions.
func TestTri3_PositiveDiag(t *testing.T) {
	nodes := [3]int{0, 1, 2}
	coords := [3][2]float64{{0, 0}, {1, 0}, {0.5, 0.8}}
	tri := quad.NewTri3(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStress)
	ke := tri.GetTangentStiffness()
	for i := 0; i < 6; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d] = %v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}

// TestTri3_PlaneStrainStiffer verifies that the plane strain Tri3 element is
// stiffer than the plane stress element for the same geometry and material.
//
// Physical property: in plane strain the out-of-plane strain εzz is constrained
// to zero, generating an out-of-plane stress σzz that increases the effective
// in-plane stiffness. Specifically K_strain[0,0] > K_stress[0,0].
//
// Parameters: right triangle at (0,0)-(1,0)-(0,1), E=200000, ν=0.3.
//
// Why valuable: using the wrong constitutive matrix (e.g., swapping the two
// cases) would cause this comparison to fail.
func TestTri3_PlaneStrainStiffer(t *testing.T) {
	nodes := [3]int{0, 1, 2}
	coords := [3][2]float64{{0, 0}, {1, 0}, {0, 1}}
	ts := quad.NewTri3(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStress)
	te := quad.NewTri3(0, nodes, coords, 200000, 0.3, 1.0, quad.PlaneStrain)

	if ts.GetTangentStiffness().At(0, 0) >= te.GetTangentStiffness().At(0, 0) {
		t.Error("plane strain should be stiffer than plane stress")
	}
}

// TestTri3_PatchTest verifies that the CST element reproduces a uniform
// uniaxial stress state exactly (the classical patch test for CST elements).
//
// Physical state: uniaxial strain εxx = 1e-3 with free Poisson contraction.
// The prescribed displacement field is:
//
//	ux = εxx · x
//	uy = -ν · εxx · y
//
// Under this constant strain field the stresses must be:
//
//	σxx = E · εxx = 210000 · 1e-3 = 210
//	σyy = 0  (free Poisson contraction annuls lateral coupling)
//	τxy = 0  (no shear strain)
//
// Parameters: right triangle at (0,0)-(1,0)-(0,1), E=210000, ν=0.3, plane stress.
//
// Why valuable: the patch test is the completeness criterion for CST.
// Failure would imply the element cannot capture a constant stress state,
// violating the convergence requirement of the finite element method.
func TestTri3_PatchTest(t *testing.T) {
	// Uniform uniaxial strain εxx = 1e-3 with free Poisson contraction.
	E, nu := 210000.0, 0.3
	eps := 1e-3
	nodes := [3]int{0, 1, 2}
	coords := [3][2]float64{{0, 0}, {1, 0}, {0, 1}}
	tri := quad.NewTri3(0, nodes, coords, E, nu, 1.0, quad.PlaneStress)

	var u [6]float64
	for n := 0; n < 3; n++ {
		u[2*n] = eps * coords[n][0]         // ux = εxx·x
		u[2*n+1] = -nu * eps * coords[n][1] // uy = -ν·εxx·y
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

// TestTri3_PureShear verifies that the Tri3 (CST) element correctly reproduces
// a pure shear stress state under a prescribed shear displacement field.
//
// Physical state: pure shear with engineering shear strain γxy = 1e-3.
// The displacement field is:
//
//	ux = γxy · y,  uy = 0
//
// which gives a constant shear strain in the CST element. The stresses must be:
//
//	σxx = 0, σyy = 0, τxy = G · γxy
//	where G = E / (2(1+ν))
//
// Parameters: right triangle at (0,0)-(1,0)-(0,1), E=210000, ν=0.3, plane stress.
//
// Why valuable: confirms the shear term in the B-matrix is correctly assembled
// and that the isotropic constitutive law does not generate spurious normal
// stresses from pure shear.
func TestTri3_PureShear(t *testing.T) {
	E, nu := 210000.0, 0.3
	G := E / (2 * (1 + nu))
	gamma := 1e-3
	nodes := [3]int{0, 1, 2}
	coords := [3][2]float64{{0, 0}, {1, 0}, {0, 1}}
	tri := quad.NewTri3(0, nodes, coords, E, nu, 1.0, quad.PlaneStress)

	var u [6]float64
	for n := 0; n < 3; n++ {
		u[2*n] = gamma * coords[n][1] // ux = γ·y
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
