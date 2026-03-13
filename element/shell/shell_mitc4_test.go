package shell

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestShellMITC4_Symmetry verifies that the 24×24 element stiffness matrix
// of a flat MITC4 shell element is symmetric.
//
// Property: Ke = Keᵀ (self-adjointness). The MITC4 formulation combines a
// bilinear membrane with a mixed-interpolation plate bending component; both
// sub-matrices must combine symmetrically.
//
// Parameters: flat shell in the XY plane, 2×1 rectangle (nodes at
// (0,0,0), (2,0,0), (2,1,0), (0,1,0)), E=200000, ν=0.3, thickness=0.1.
//
// Expected: relative asymmetry |Ke[i,j]-Ke[j,i]| / avg < 1e-6 for all i < j.
// An absolute threshold of 1e-10 is used for near-zero entries.
//
// Why valuable: catches unsymmetric contributions from the drilling DOF
// regularisation or from an incorrectly transposed transformation matrix.
func TestShellMITC4_Symmetry(t *testing.T) {
	// Flat shell in XY plane
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{2, 0, 0},
		{2, 1, 0},
		{0, 1, 0},
	}
	s := NewShellMITC4(0, nodes, coords, 200000, 0.3, 0.1)

	ke := s.GetTangentStiffness()
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

// TestShellMITC4_RigidBodyTranslation verifies that a uniform rigid-body
// translation in each global direction produces zero nodal forces.
//
// Property: Ke · u_rigid = 0 for u_rigid = [δ_d, 0, 0, 0, 0, 0, ...]
// with δ_d = 1 in the d-th translational DOF for each of the 4 nodes.
// A flat shell has three translational rigid-body modes: X, Y (in-plane)
// and Z (out-of-plane).
//
// Parameters: unit square shell (1×1) in XY plane, E=200000, ν=0.3,
// thickness=0.05.
//
// Tolerance: 1e-3. This is larger than for bar elements because the MITC4
// element includes a drilling DOF stabilisation term that introduces a
// small but finite numerical coupling.
//
// Why valuable: a violation indicates that the drilling regularisation or
// the local-to-global frame transformation contains a residual that breaks
// the null-space property of rigid-body motions.
func TestShellMITC4_RigidBodyTranslation(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 0},
		{0, 1, 0},
	}
	s := NewShellMITC4(0, nodes, coords, 200000, 0.3, 0.05)

	ke := s.GetTangentStiffness()

	// Uniform translation should produce zero forces
	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(24, nil)
		for n := 0; n < 4; n++ {
			u.SetVec(n*6+dir, 1.0)
		}

		f := mat.NewVecDense(24, nil)
		f.MulVec(ke, u)

		for i := 0; i < 24; i++ {
			if math.Abs(f.AtVec(i)) > 1e-3 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

// TestShellMITC4_PositiveDiag verifies that all 24 diagonal entries of the
// MITC4 shell stiffness matrix are strictly positive.
//
// Property: for a positive semi-definite stiffness matrix, all diagonal entries
// must be non-negative. For a constrained (non-singular) element in a mesh
// context, all diagonal entries should be strictly positive, which is a
// necessary condition for the Cholesky factorisation used in direct solvers.
//
// Parameters: unit square, E=200000, ν=0.3, thickness=0.1.
// Each node has 6 DOFs: UX, UY, UZ, RX, RY, RZ.
// The drilling DOF (RZ about the shell normal) is stabilised explicitly.
//
// Why valuable: a zero or negative diagonal entry would indicate a missing
// stiffness contribution for that DOF (e.g., an unrestrained drilling mode)
// and would cause a singular or indefinite global stiffness matrix.
func TestShellMITC4_PositiveDiag(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 0},
		{0, 1, 0},
	}
	s := NewShellMITC4(0, nodes, coords, 200000, 0.3, 0.1)

	ke := s.GetTangentStiffness()
	for i := 0; i < 24; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d] = %v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}
