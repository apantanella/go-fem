package solid

import (
	"math"
	"testing"

	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

// unitTet returns a right-angle tetrahedron at the origin with unit edges.
// Nodes: (0,0,0), (1,0,0), (0,1,0), (0,0,1).
//
// The unit tet has an analytically known volume V = 1/6, which allows
// closed-form verification of the stiffness matrix entries via
// Ke = V · Bᵀ·D·B (constant strain tetrahedron).
func unitTet(e, nu float64) *Tet4 {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}
	m := material.NewIsotropicLinear(e, nu)
	return NewTet4(0, nodes, coords, m)
}

// TestTet4Volume verifies that the volume of the unit right-angle tetrahedron
// is exactly 1/6.
//
// Property: V = |det[x₁-x₀, x₂-x₀, x₃-x₀]| / 6 for a linear tetrahedron.
// For unit edge vectors the determinant equals 1, giving V = 1/6.
//
// Expected: |V - 1/6| < 1e-14 (machine-precision comparison).
//
// Why valuable: an incorrect volume would corrupt all stiffness entries
// (since Ke = V·BᵀDB for a constant-strain tet) and body-force loads
// (which scale with V).
func TestTet4Volume(t *testing.T) {
	tet := unitTet(1, 0)
	want := 1.0 / 6.0
	if math.Abs(tet.Volume()-want) > 1e-14 {
		t.Errorf("Volume = %v, want %v", tet.Volume(), want)
	}
}

// TestTet4Symmetry verifies that the 12×12 element stiffness matrix of the
// Tet4 constant-strain tetrahedron is symmetric.
//
// Property: Ke = Keᵀ. For Ke = V·BᵀDB with D symmetric, this follows
// algebraically; any floating-point asymmetry would indicate an
// implementation error in B or D.
//
// Parameters: unit tet, E=200000, ν=0.3.
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-10 for all i < j (absolute tolerance).
//
// Why valuable: a non-symmetric stiffness matrix would cause incorrect results
// in direct solvers that exploit symmetry (e.g., Cholesky) and would indicate
// an error in the B-matrix assembly.
func TestTet4Symmetry(t *testing.T) {
	tet := unitTet(200000, 0.3)
	ke := tet.GetTangentStiffness()
	n := 12
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			if diff > 1e-10 {
				t.Errorf("Ke not symmetric: K[%d][%d]=%.6e, K[%d][%d]=%.6e, diff=%.2e",
					i, j, ke.At(i, j), j, i, ke.At(j, i), diff)
			}
		}
	}
}

// TestTet4RigidBody verifies that a uniform rigid-body translation produces
// zero nodal forces for the Tet4 element.
//
// Property: Ke · u_rigid = 0 for the three translational rigid-body modes
// u_rigid = [δ,0,0, δ,0,0, δ,0,0, δ,0,0] etc. For the constant-strain
// tetrahedron this follows from the fact that rigid translation implies zero
// strain, so σ = D·ε = 0 and thus f = Bᵀσ·V = 0.
//
// Parameters: unit tet, E=200000, ν=0.3. Directions X, Y, Z are tested.
//
// Expected: |Ke · u_rigid|_∞ < 1e-10 for each direction.
//
// Why valuable: a non-zero result would reveal an error in the B-matrix
// (non-constant strains under rigid translation) or a wrong volume factor.
func TestTet4RigidBody(t *testing.T) {
	tet := unitTet(200000, 0.3)
	ke := tet.GetTangentStiffness()

	// Three rigid-body translations: (1,0,0), (0,1,0), (0,0,1).
	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(12, nil)
		for n := 0; n < 4; n++ {
			u.SetVec(3*n+dir, 1.0)
		}
		f := mat.NewVecDense(12, nil)
		f.MulVec(ke, u)

		for i := 0; i < 12; i++ {
			if math.Abs(f.AtVec(i)) > 1e-10 {
				t.Errorf("Rigid body dir=%d: F[%d] = %.6e (want 0)", dir, i, f.AtVec(i))
			}
		}
	}
}

// TestTet4KnownDiagonal verifies that Ke[0][0] equals the analytically derived
// value for a unit tet with E=1 and ν=0.
//
// Derivation: for the constant-strain tetrahedron Ke = V · Bᵀ·D·B.
// With E=1, ν=0 the constitutive matrix D is diagonal:
//
//	D = diag(1, 1, 1, 0.5, 0.5, 0.5)  (Lamé: λ=0, μ=0.5)
//
// The B-matrix for node 0 in the unit tet has B[0,0] = -1 (from ∂N₀/∂x = -1).
// Therefore Ke[0][0] = V · (B[:,0]ᵀ · D · B[:,0]) = (1/6) · 2 = 1/3.
//
// Parameters: E=1, ν=0. Expected: Ke[0][0] = 1/3 within 1e-14.
//
// Why valuable: this closed-form check would catch any incorrect volume factor
// or B-matrix entry, including a sign error in the shape-function gradient.
func TestTet4KnownDiagonal(t *testing.T) {
	// Unit tet, E=1, nu=0 → Ke[0][0] = V · (Bcol0ᵀ D Bcol0) = (1/6)·2 = 1/3.
	tet := unitTet(1, 0)
	ke := tet.GetTangentStiffness()
	want := 1.0 / 3.0
	got := ke.At(0, 0)
	if math.Abs(got-want) > 1e-14 {
		t.Errorf("Ke[0][0] = %v, want %v", got, want)
	}
}

// TestTet4PositiveDefiniteReduced verifies that all diagonal entries of the
// Tet4 stiffness matrix are non-negative.
//
// Property: a necessary condition for positive semi-definiteness is that all
// diagonal entries are non-negative. The full 12×12 Ke is singular (three
// null eigenvalues corresponding to rigid translations), but each diagonal
// entry must be non-negative.
//
// Parameters: unit tet, E=200000, ν=0.3.
//
// Why valuable: a negative diagonal entry would indicate an ill-conditioned
// or incorrectly assembled stiffness matrix, and would cause a direct solver
// to report a non-positive-definite system even after proper boundary conditions
// are applied.
func TestTet4PositiveDefiniteReduced(t *testing.T) {
	// After fixing node 0 (removing 3 DOFs for translation) and constraining
	// 3 more DOFs to remove rotations, the reduced Ke should be positive definite.
	// We test by checking all diagonal values of Ke are positive (necessary condition).
	tet := unitTet(200000, 0.3)
	ke := tet.GetTangentStiffness()
	for i := 0; i < 12; i++ {
		if ke.At(i, i) < 0 {
			t.Errorf("Ke[%d][%d] = %v < 0 (diagonal should be non-negative)", i, i, ke.At(i, i))
		}
	}
}
