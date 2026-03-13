package solid

import (
	"math"
	"testing"

	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

// TestTet10_Symmetry verifies that the 30×30 element stiffness matrix of the
// 10-node quadratic tetrahedron (Tet10) is symmetric.
//
// Property: Ke = Keᵀ. For Ke = ∫ Bᵀ·D·B dV with D symmetric this holds
// algebraically; numerical quadrature must not break it.
//
// Parameters: unit tet with 4 corners at the standard positions and 6 midside
// nodes at edge midpoints (via makeTet10), E=200000, ν=0.3.
//
// Expected: relative asymmetry |Ke[i,j]-Ke[j,i]| / avg < 1e-6 for entries
// with avg > 1e-10.
//
// Why valuable: an unsymmetric Ke from the Tet10 would indicate an error in
// the quadratic shape-function derivatives or the Gauss-quadrature loop.
func TestTet10_Symmetry(t *testing.T) {
	tet := makeTet10()
	ke := tet.GetTangentStiffness()
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

// TestTet10_RigidBody verifies that a uniform rigid-body translation produces
// zero nodal forces for the Tet10 element.
//
// Property: Ke · u_rigid = 0 for u_rigid = [δ,0,0, δ,0,0, ...] applied to
// all 10 nodes. Pure translation implies zero strain everywhere (the quadratic
// shape functions reproduce constant fields exactly), so no internal forces
// should arise.
//
// Parameters: unit Tet10 (makeTet10), E=200000, ν=0.3.
// Directions X, Y, Z are tested individually.
//
// Expected: |Ke · u_rigid|_∞ < 1e-4 for each direction.
//
// Why valuable: a violation would indicate that the quadratic shape-function
// gradients do not sum to zero (partition-of-unity error), producing
// spurious strain under rigid translation.
func TestTet10_RigidBody(t *testing.T) {
	tet := makeTet10()
	ke := tet.GetTangentStiffness()

	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(30, nil)
		for n := 0; n < 10; n++ {
			u.SetVec(3*n+dir, 1)
		}

		f := mat.NewVecDense(30, nil)
		f.MulVec(ke, u)

		for i := 0; i < 30; i++ {
			if math.Abs(f.AtVec(i)) > 1e-4 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

// TestTet10_PositiveDiag verifies that all 30 diagonal entries of the Tet10
// stiffness matrix are strictly positive.
//
// Property: a necessary condition for positive semi-definiteness is that all
// diagonal entries are non-negative. For a well-formed quadratic tet element
// each DOF must have positive self-stiffness.
//
// Parameters: unit Tet10 (makeTet10), E=200000, ν=0.3.
//
// Why valuable: a zero or negative diagonal entry for a midside node DOF
// would indicate that the midside node shape function produces no strain energy
// (possible if the Gauss-point positions are incorrectly placed on the element).
func TestTet10_PositiveDiag(t *testing.T) {
	tet := makeTet10()
	ke := tet.GetTangentStiffness()
	for i := 0; i < 30; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d] = %v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}

// makeTet10 creates a standard 10-node quadratic tetrahedron with corner nodes
// at the unit-tet positions and midside nodes at the midpoints of each edge.
//
// Node numbering:
//   - 0–3: corner nodes at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
//   - 4:   midside of edge 0–1 at (0.5, 0, 0)
//   - 5:   midside of edge 1–2 at (0.5, 0.5, 0)
//   - 6:   midside of edge 0–2 at (0, 0.5, 0)
//   - 7:   midside of edge 0–3 at (0, 0, 0.5)
//   - 8:   midside of edge 1–3 at (0.5, 0, 0.5)
//   - 9:   midside of edge 2–3 at (0, 0.5, 0.5)
//
// The material is steel-like: E=200000, ν=0.3.
func makeTet10() *Tet10 {
	nodes := [10]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	coords := [10][3]float64{
		{0, 0, 0},    // 0 - corner
		{1, 0, 0},    // 1 - corner
		{0, 1, 0},    // 2 - corner
		{0, 0, 1},    // 3 - corner
		{0.5, 0, 0},  // 4 - midside 0-1
		{0.5, 0.5, 0}, // 5 - midside 1-2
		{0, 0.5, 0},  // 6 - midside 0-2
		{0, 0, 0.5},  // 7 - midside 0-3
		{0.5, 0, 0.5}, // 8 - midside 1-3
		{0, 0.5, 0.5}, // 9 - midside 2-3
	}
	m := material.NewIsotropicLinear(200000, 0.3)
	return NewTet10(0, nodes, coords, m)
}
