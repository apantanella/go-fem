package truss

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestTruss2D_AxialStiffness verifies the axial stiffness of a 2D truss element
// aligned with the global X axis.
//
// Property: for a horizontal bar the element stiffness matrix degenerates to
//
//	K = (AE/L) · [1 0 -1 0; 0 0 0 0; -1 0 1 0; 0 0 0 0]
//
// Parameters: E=100, A=0.5, L=2, so k = AE/L = 25.
// Nodes are at (0,0) and (2,0).
//
// Expected results (DOF layout [UX₀,UY₀,UX₁,UY₁]):
//   - K[0,0] = 25  (axial stiffness at node 0)
//   - K[2,2] = 25  (axial stiffness at node 1)
//   - K[0,2] = -25 (coupling)
//   - K[1,1] = 0   (transverse has no stiffness)
//
// Why valuable: catches AE/L scaling errors and confirms that transverse DOF
// rows/columns remain zero for an axis-aligned element.
func TestTruss2D_AxialStiffness(t *testing.T) {
	// Truss along X-axis, length=2, E=100, A=0.5
	// Expected: k = AE/L = 0.5*100/2 = 25
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {2, 0}}
	tr := NewTruss2D(0, nodes, coords, 100, 0.5)

	ke := tr.GetTangentStiffness()

	if got := ke.At(0, 0); math.Abs(got-25) > 1e-10 {
		t.Errorf("K[0,0] = %v, want 25", got)
	}
	if got := ke.At(2, 2); math.Abs(got-25) > 1e-10 {
		t.Errorf("K[2,2] = %v, want 25", got)
	}
	if got := ke.At(0, 2); math.Abs(got-(-25)) > 1e-10 {
		t.Errorf("K[0,2] = %v, want -25", got)
	}
	// Y DOFs should have zero stiffness for axial truss along X
	if got := ke.At(1, 1); math.Abs(got) > 1e-10 {
		t.Errorf("K[1,1] = %v, want 0", got)
	}
}

// TestTruss2D_Symmetry verifies that the 4×4 element stiffness matrix is
// symmetric for an arbitrarily oriented 2D truss element.
//
// Property: Ke = Keᵀ (self-adjointness). Must hold for any orientation.
//
// Parameters: truss from (0,0) to (3,4), L=5; E=200000, A=0.01.
// Direction cosines (3/5, 4/5) exercise both in-plane components.
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-6 for all i < j.
//
// Why valuable: catches errors in the outer-product assembly of direction
// cosines that could produce an unsymmetric stiffness matrix.
func TestTruss2D_Symmetry(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {3, 4}}
	tr := NewTruss2D(0, nodes, coords, 200000, 0.01)

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

// TestTruss2D_RigidBody verifies that a uniform rigid-body translation produces
// zero nodal forces for a 2D truss element.
//
// Property: Ke · u_rigid = 0 for u_rigid = [δ,0,δ,0] or [0,δ,0,δ].
// A two-node bar has two translational rigid-body modes in 2D.
//
// Parameters: truss from (1,2) to (4,6) (arbitrary orientation),
// E=200000, A=0.01.
//
// Expected: |Ke · u_rigid|_∞ < 1e-8 for each direction.
//
// Why valuable: a non-zero result would indicate incorrect equilibrium at the
// element level (Ke should have two zero eigenvalues for a free bar in 2D).
func TestTruss2D_RigidBody(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{1, 2}, {4, 6}}
	tr := NewTruss2D(0, nodes, coords, 200000, 0.01)

	ke := tr.GetTangentStiffness()

	// Uniform translation should produce zero forces
	for dir := 0; dir < 2; dir++ {
		u := mat.NewVecDense(4, nil)
		u.SetVec(dir, 1)
		u.SetVec(dir+2, 1)

		f := mat.NewVecDense(4, nil)
		f.MulVec(ke, u)

		for i := 0; i < 4; i++ {
			if math.Abs(f.AtVec(i)) > 1e-8 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want 0", dir, i, f.AtVec(i))
			}
		}
	}
}

// TestTruss2D_DiagonalElement verifies the direction-cosine projection for a
// 45° truss element.
//
// Property: for a truss with direction cosines (cos θ, sin θ), the stiffness
// entries are:
//
//	K[0,0] = k·cos²θ,  K[0,1] = k·cos θ·sin θ
//
// At 45° both direction cosines equal 1/√2, so:
//
//	K[0,0] = k·(1/2),  K[0,1] = k·(1/2)
//	where k = EA/L = 200000·0.01/√2 ≈ 1414.2
//
// Parameters: E=200000, A=0.01, nodes at (0,0) and (1,1), L=√2.
//
// Expected: K[0,0] = k/2, K[0,1] = k/2 (both equal).
//
// Why valuable: exercises the trigonometric projection of the local stiffness
// into global coordinates; a sign or cosine/sine swap would produce a
// wrong off-diagonal value.
func TestTruss2D_DiagonalElement(t *testing.T) {
	// 45° truss: cos=sin=1/√2
	E, A := 200000.0, 0.01
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {1, 1}}
	tr := NewTruss2D(0, nodes, coords, E, A)

	L := math.Sqrt(2.0)
	k := E * A / L
	expected00 := k * 0.5 // k·cos²(45°) = k/2

	ke := tr.GetTangentStiffness()
	if got := ke.At(0, 0); math.Abs(got-expected00)/expected00 > 1e-10 {
		t.Errorf("K[0,0] = %v, want %v", got, expected00)
	}
	// Cross-coupling K[0,1] = k·cos·sin = k/2
	if got := ke.At(0, 1); math.Abs(got-expected00)/expected00 > 1e-10 {
		t.Errorf("K[0,1] = %v, want %v", got, expected00)
	}
}

// TestTruss2D_AxialForce verifies the axial force and stress after applying a
// small axial extension to a 2D truss element.
//
// Property: linear elasticity gives
//
//	N = (EA/L) · Δu,  σ = N/A = (E/L) · Δu
//
// Parameters: E=100, A=0.5, L=2. Node 1 is displaced 0.01 m in X.
//
//	N = (100·0.5/2) · 0.01 = 25 · 0.01 = 0.25
//	σ = N/A = 0.25/0.5 = 0.5
//
// Why valuable: ensures that the Update pathway correctly computes the
// deformation measure and that AxialForce/AxialStress return physically
// consistent values; would catch sign errors or missing area scaling.
func TestTruss2D_AxialForce(t *testing.T) {
	E, A := 100.0, 0.5
	L := 2.0
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	tr := NewTruss2D(0, nodes, coords, E, A)

	// Apply unit extension: node 1 moves 0.01 in X
	disp := []float64{0, 0, 0.01, 0}
	tr.Update(disp)

	// N = EA/L · Δu = 100*0.5/2 * 0.01 = 0.25
	N := tr.AxialForce()
	if math.Abs(N-0.25) > 1e-10 {
		t.Errorf("AxialForce = %v, want 0.25", N)
	}

	sigma := tr.AxialStress()
	if math.Abs(sigma-0.5) > 1e-10 {
		t.Errorf("AxialStress = %v, want 0.5", sigma)
	}
}

// TestTruss2D_MatchesTruss3D verifies that the in-plane stiffness of a 2D truss
// element equals the corresponding sub-block of the 3D truss element.
//
// Property: when restricted to the XY plane (Z=0), a Truss3D element must
// produce identical stiffness entries to the Truss2D element. This establishes
// that the 2D implementation is a consistent specialisation of the 3D one.
//
// DOF mapping: 2D indices [0,1,2,3] correspond to 3D indices [0,1,3,4]
// (skipping UZ at each node: 3D DOF 2 and 5 are omitted).
//
// Parameters: E=200000, A=0.01, truss from (0,0) to (3,4), L=5.
//
// Expected: |Ke2D[i,j] - Ke3D[map[i],map[j]]| < 1e-6 for all i, j.
//
// Why valuable: a discrepancy would indicate either a coordinate-transform bug
// in the 3D element or an incorrect DOF ordering in the 2D element.
func TestTruss2D_MatchesTruss3D(t *testing.T) {
	// Verify that Truss2D produces the same in-plane stiffness as Truss3D
	E, A := 200000.0, 0.01
	n2 := [2]int{0, 1}
	c2 := [2][2]float64{{0, 0}, {3, 4}}
	c3 := [2][3]float64{{0, 0, 0}, {3, 4, 0}}

	tr2d := NewTruss2D(0, n2, c2, E, A)
	tr3d := NewTruss3D(0, n2, c3, E, A)

	ke2 := tr2d.GetTangentStiffness()
	ke3 := tr3d.GetTangentStiffness()

	// Compare UX/UY rows/cols: 2D DOFs (0,1,2,3) map to 3D DOFs (0,1,3,4)
	map2to3 := [4]int{0, 1, 3, 4}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			v2 := ke2.At(i, j)
			v3 := ke3.At(map2to3[i], map2to3[j])
			if math.Abs(v2-v3) > 1e-6 {
				t.Errorf("K2D[%d,%d]=%v != K3D[%d,%d]=%v",
					i, j, v2, map2to3[i], map2to3[j], v3)
			}
		}
	}
}
