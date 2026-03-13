package quad

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestQuad4_Symmetry verifies that the 8×8 element stiffness matrix of the
// bilinear quadrilateral element (Quad4) is symmetric.
//
// Property: Ke = Keᵀ. For Ke = t · ∫ Bᵀ·D·B dA with D symmetric this holds
// analytically; the 2×2 Gauss quadrature rule must not break it.
//
// Parameters: 2×1 rectangle (nodes at (0,0),(2,0),(2,1),(0,1)), E=200000,
// ν=0.3, thickness=1.0, plane stress.
//
// Expected: relative asymmetry |Ke[i,j]-Ke[j,i]| / avg < 1e-6 for entries
// with avg > 1e-10.
//
// Why valuable: catches errors in the isoparametric B-matrix assembly where
// a non-square Jacobian (aspect ratio 2:1) could expose a sign mistake.
func TestQuad4_Symmetry(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][2]float64{{0, 0}, {2, 0}, {2, 1}, {0, 1}}
	q := NewQuad4(0, nodes, coords, 200000, 0.3, 1.0, PlaneStress)

	ke := q.GetTangentStiffness()
	for i := 0; i < 8; i++ {
		for j := i + 1; j < 8; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			avg := (math.Abs(ke.At(i, j)) + math.Abs(ke.At(j, i))) / 2
			if avg > 1e-10 && diff/avg > 1e-6 {
				t.Errorf("not symmetric: K[%d,%d]=%v != K[%d,%d]=%v",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

// TestQuad4_RigidBody verifies that a uniform rigid-body translation produces
// zero nodal forces for the Quad4 element.
//
// Property: Ke · u_rigid = 0 for u_rigid applying unit displacement in the X or Y
// direction to all 4 nodes. A rigid translation implies zero strain everywhere
// (the bilinear shape functions reproduce constant displacements exactly), so
// no internal forces should arise.
//
// Parameters: unit square, E=200000, ν=0.3, thickness=0.1, plane stress.
//
// Expected: |Ke · u_rigid|_∞ < 1e-6 for X and Y translation modes.
//
// Why valuable: a non-zero result would indicate a partition-of-unity error
// in the Quad4 shape functions or an incorrectly zeroed B-matrix column.
func TestQuad4_RigidBody(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	q := NewQuad4(0, nodes, coords, 200000, 0.3, 0.1, PlaneStress)

	ke := q.GetTangentStiffness()

	// Uniform translation
	for dir := 0; dir < 2; dir++ {
		u := mat.NewVecDense(8, nil)
		for n := 0; n < 4; n++ {
			u.SetVec(2*n+dir, 1)
		}

		f := mat.NewVecDense(8, nil)
		f.MulVec(ke, u)

		for i := 0; i < 8; i++ {
			if math.Abs(f.AtVec(i)) > 1e-6 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

// TestQuad4_PositiveDiag verifies that all 8 diagonal entries of the Quad4
// stiffness matrix are strictly positive.
//
// Property: a necessary condition for positive semi-definiteness is that all
// diagonal entries are non-negative. Each nodal DOF (UX and UY at each of
// the 4 nodes) must have a positive self-stiffness contribution.
//
// Parameters: unit square, E=200000, ν=0.3, thickness=0.1, plane stress.
//
// Why valuable: a zero diagonal entry would indicate that a DOF is not
// connected to the stiffness (e.g., an unrestrained corner DOF in a
// singular element configuration), which would cause a singular stiffness.
func TestQuad4_PositiveDiag(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	q := NewQuad4(0, nodes, coords, 200000, 0.3, 0.1, PlaneStress)

	ke := q.GetTangentStiffness()
	for i := 0; i < 8; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d] = %v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}

// TestQuad4_PlaneStrainVsStress verifies that the plane strain Quad4 element
// is stiffer than the plane stress element for the same geometry and material.
//
// Physical property: in plane strain the out-of-plane strain εzz is constrained
// to zero. The corresponding constraint generates an out-of-plane stress σzz
// via the Poisson coupling, which increases the effective in-plane stiffness.
// Specifically, the normal stiffness D[0,0] in plane strain is:
//
//	D_ps_strain[0,0] = E(1-ν)/((1+ν)(1-2ν))  > D_ps_stress[0,0] = E/(1-ν²)
//
// This difference is captured in K[0,0].
//
// Parameters: unit square, E=200000, ν=0.3, thickness=0.1.
//
// Expected: K_plane_strain[0,0] > K_plane_stress[0,0].
//
// Why valuable: using the wrong constitutive matrix (e.g., swapping plane
// stress for plane strain) would cause this comparison to fail, revealing
// a bug in the conditional assembly of D.
func TestQuad4_PlaneStrainVsStress(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}

	qs := NewQuad4(0, nodes, coords, 200000, 0.3, 0.1, PlaneStress)
	qe := NewQuad4(0, nodes, coords, 200000, 0.3, 0.1, PlaneStrain)

	// Plane strain should be stiffer than plane stress
	if qs.GetTangentStiffness().At(0, 0) >= qe.GetTangentStiffness().At(0, 0) {
		t.Error("plane strain should be stiffer than plane stress")
	}
}
