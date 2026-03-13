package solid

import (
	"math"
	"testing"

	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

// TestBrick20_Symmetry verifies that the 60×60 element stiffness matrix of the
// 20-node serendipity brick element (Brick20) is symmetric.
//
// Property: Ke = Keᵀ. For Ke = ∫ Bᵀ·D·B dV this holds algebraically with a
// symmetric D; numerical quadrature using a 3×3×3 Gauss rule must not break it.
//
// Parameters: unit cube [0,1]³ via makeBrick20, E=200000, ν=0.3.
//
// Expected: relative asymmetry |Ke[i,j]-Ke[j,i]| / avg < 1e-6 for entries
// with avg > 1e-10.
//
// Why valuable: unsymmetry in the 60×60 matrix would indicate an error in the
// serendipity shape-function derivatives or an incorrectly assembled coordinate
// Jacobian for the 20-node element.
func TestBrick20_Symmetry(t *testing.T) {
	b := makeBrick20()
	ke := b.GetTangentStiffness()
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

// TestBrick20_RigidBody verifies that a uniform rigid-body translation produces
// zero nodal forces for the Brick20 element.
//
// Property: Ke · u_rigid = 0 for u_rigid applying equal unit displacement in
// one global direction to all 20 nodes. The serendipity shape functions must
// reproduce a constant displacement field exactly (completeness), giving zero
// strain and therefore zero internal forces.
//
// Parameters: unit cube via makeBrick20, E=200000, ν=0.3.
// Directions X, Y, Z are tested individually.
//
// Expected: |Ke · u_rigid|_∞ < 1e-4.
//
// Why valuable: a non-zero result would reveal that the serendipity shape
// functions do not satisfy the partition-of-unity property, which would make
// the element fail the patch test and produce locking or spurious forces.
func TestBrick20_RigidBody(t *testing.T) {
	b := makeBrick20()
	ke := b.GetTangentStiffness()

	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(60, nil)
		for n := 0; n < 20; n++ {
			u.SetVec(3*n+dir, 1)
		}

		f := mat.NewVecDense(60, nil)
		f.MulVec(ke, u)

		for i := 0; i < 60; i++ {
			if math.Abs(f.AtVec(i)) > 1e-4 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

// TestBrick20_PositiveDiag verifies that all 60 diagonal entries of the Brick20
// stiffness matrix are strictly positive.
//
// Property: a necessary condition for positive semi-definiteness is that all
// diagonal entries are non-negative. For the serendipity brick, both corner
// nodes (with 8 shape functions similar to Hex8) and midside nodes (12 additional
// bubble-like shape functions) must have positive self-stiffness.
//
// Parameters: unit cube via makeBrick20, E=200000, ν=0.3.
//
// Why valuable: a zero diagonal for a midside node DOF would indicate that the
// corresponding serendipity shape function produces no strain energy, which
// could arise if the midside node is placed at a Gauss-point where the shape
// function evaluates to zero at all integration points.
func TestBrick20_PositiveDiag(t *testing.T) {
	b := makeBrick20()
	ke := b.GetTangentStiffness()
	for i := 0; i < 60; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d] = %v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}

// makeBrick20 creates a unit cube [0,1]³ 20-node serendipity element by
// mapping the reference coordinates from [-1,1]³ to [0,1]³ using x=(1+ξ)/2.
//
// The node ordering follows the standard serendipity connectivity:
//   - Nodes 0–7: corner nodes (same as Hex8)
//   - Nodes 8–19: midside nodes on each of the 12 edges
//
// Reference coordinates are taken from brick20Ref (defined in the element
// implementation) and linearly scaled to the physical domain.
// The material is steel-like: E=200000, ν=0.3.
func makeBrick20() *Brick20 {
	var nodes [20]int
	for i := 0; i < 20; i++ {
		nodes[i] = i
	}
	// Map reference coords [-1,1]³ to physical [0,1]³: x = (1+ξ)/2
	var coords [20][3]float64
	for i := 0; i < 20; i++ {
		coords[i][0] = (1 + brick20Ref[i][0]) / 2
		coords[i][1] = (1 + brick20Ref[i][1]) / 2
		coords[i][2] = (1 + brick20Ref[i][2]) / 2
	}
	m := material.NewIsotropicLinear(200000, 0.3)
	return NewBrick20(0, nodes, coords, m)
}
