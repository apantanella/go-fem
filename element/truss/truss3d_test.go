package truss

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestTruss3D_AxialStiffness verifies the axial stiffness of a 3D truss element
// aligned with the global X axis.
//
// Property: The element stiffness matrix for an axially aligned bar is
//
//	K = (AE/L) · [1 -1; -1 1]  (in the axial direction)
//
// with all transverse terms equal to zero.
//
// Parameters: E=100, A=0.5, L=2, so k = AE/L = 0.5·100/2 = 25.
// The nodes are placed at (0,0,0) and (2,0,0).
//
// Expected results (DOF layout [UX₀,UY₀,UZ₀,UX₁,UY₁,UZ₁]):
//   - K[0,0] = 25  (axial stiffness at node 0)
//   - K[3,3] = 25  (axial stiffness at node 1)
//   - K[0,3] = -25 (coupling between axial DOFs)
//   - K[1,1] = 0   (no transverse stiffness for a 1D bar)
//
// Why valuable: catches sign errors in the direction-cosine assembly and
// incorrect AE/L scaling, and confirms that transverse rows/columns are zeroed.
func TestTruss3D_AxialStiffness(t *testing.T) {
	// Truss along X-axis, length=2, E=100, A=0.5
	// Expected: k = AE/L = 0.5*100/2 = 25
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {2, 0, 0}}
	tr := NewTruss3D(0, nodes, coords, 100, 0.5)

	ke := tr.GetTangentStiffness()

	// For an axial truss along X: K[0,0] = AE/L, K[3,3] = AE/L
	if got := ke.At(0, 0); math.Abs(got-25) > 1e-10 {
		t.Errorf("K[0,0] = %v, want 25", got)
	}
	if got := ke.At(3, 3); math.Abs(got-25) > 1e-10 {
		t.Errorf("K[3,3] = %v, want 25", got)
	}
	// Off-diagonal coupling
	if got := ke.At(0, 3); math.Abs(got-(-25)) > 1e-10 {
		t.Errorf("K[0,3] = %v, want -25", got)
	}
	// Y and Z DOFs should have zero stiffness for axial truss
	if got := ke.At(1, 1); math.Abs(got) > 1e-10 {
		t.Errorf("K[1,1] = %v, want 0", got)
	}
}

// TestTruss3D_Symmetry verifies that the 6×6 element stiffness matrix is
// symmetric for an arbitrarily oriented 3D truss element.
//
// Property: Ke = Keᵀ (Maxwell reciprocity / self-adjointness of elastic
// stiffness). Symmetry must hold for any orientation, not just axis-aligned.
//
// Parameters: diagonal truss from (0,0,0) to (3,4,5), L=√50; E=200000, A=0.01.
// The direction-cosine vector (3,4,5)/√50 exercises all three components
// simultaneously.
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-6 for all i < j.
//
// Why valuable: catches unsymmetric tensor products in the outer-product
// assembly  ke = k·(l⊗l)  where l is the unit direction vector.
func TestTruss3D_Symmetry(t *testing.T) {
	// Diagonal truss
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {3, 4, 5}}
	tr := NewTruss3D(0, nodes, coords, 200000, 0.01)

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

// TestTruss3D_RigidBody verifies that a uniform rigid-body translation produces
// zero nodal forces for a 3D truss element.
//
// Property: Ke · u_rigid = 0, i.e. the null space of Ke contains rigid-body
// translation modes. For a two-node bar element there are three translation
// rigid-body modes: u = [1,0,0,1,0,0], [0,1,0,0,1,0], [0,0,1,0,0,1].
//
// Parameters: truss from (1,2,3) to (4,5,6) (arbitrary, non-axis-aligned),
// E=200000, A=0.01.
//
// Expected: |Ke · u_rigid|_∞ < 1e-8 for each of the three directions.
//
// Why valuable: a non-zero result would reveal incorrect treatment of the
// reference configuration or an error in the element-level equilibrium
// (Ke should have three zero eigenvalues for a free bar).
func TestTruss3D_RigidBody(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{1, 2, 3}, {4, 5, 6}}
	tr := NewTruss3D(0, nodes, coords, 200000, 0.01)

	ke := tr.GetTangentStiffness()

	// Uniform translation should produce zero forces
	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(6, nil)
		u.SetVec(dir, 1)
		u.SetVec(dir+3, 1)

		f := mat.NewVecDense(6, nil)
		f.MulVec(ke, u)

		for i := 0; i < 6; i++ {
			if math.Abs(f.AtVec(i)) > 1e-8 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want 0", dir, i, f.AtVec(i))
			}
		}
	}
}

// TestCorotTruss_LinearEquivalence verifies that the corotational truss element
// produces the same tangent stiffness as the linear Truss3D element when the
// current displacement is zero (undeformed configuration).
//
// Property: at zero deformation the corotational frame has not rotated, so the
// geometric stiffness vanishes and the material stiffness coincides with the
// small-strain formulation. Both formulations must agree entry-by-entry.
//
// Parameters: truss from (0,0,0) to (3,4,0) — a 2D diagonal element in 3D
// space — with E=200000, A=0.01.
//
// Expected: |Ke_linear[i,j] - Ke_corot[i,j]| < 1e-6 for all i, j.
//
// Why valuable: any non-zero residual at zero deformation indicates an error
// in the corotational frame initialisation (e.g., a pre-rotated local frame
// or a wrong sign in the geometric stiffness contribution).
func TestCorotTruss_LinearEquivalence(t *testing.T) {
	// For zero displacement, CorotTruss should equal Truss3D
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {3, 4, 0}}

	tr := NewTruss3D(0, nodes, coords, 200000, 0.01)
	ct := NewCorotTruss(0, nodes, coords, 200000, 0.01)

	ke1 := tr.GetTangentStiffness()
	ke2 := ct.GetTangentStiffness()

	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			if math.Abs(ke1.At(i, j)-ke2.At(i, j)) > 1e-6 {
				t.Errorf("K[%d,%d]: truss=%v, corot=%v", i, j, ke1.At(i, j), ke2.At(i, j))
			}
		}
	}
}
