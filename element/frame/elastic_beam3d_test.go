package frame

import (
	"math"
	"testing"

	"go-fem/section"

	"gonum.org/v1/gonum/mat"
)

// TestElasticBeam3D_Symmetry verifies that the 12×12 element stiffness matrix
// of a 3D elastic beam is symmetric.
//
// Property: Ke = Keᵀ (self-adjointness of the linear elastic stiffness operator).
// This must hold for any beam orientation and section properties.
//
// Parameters: beam from (0,0,0) to (5,0,0), E=200000, G=80000,
// A=0.01, Iy=Iz=1e-4, J=2e-4.
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-4 for all i < j.
// (Tolerance is 1e-4 to accommodate the diagonal-beam coordinate transform
// used in other tests; this test uses an axis-aligned beam.)
//
// Why valuable: catches sign errors or transposed sub-block assembly in the
// local-to-global transformation.
func TestElasticBeam3D_Symmetry(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {5, 0, 0}}
	sec := section.BeamSection3D{A: 0.01, Iy: 1e-4, Iz: 1e-4, J: 2e-4}
	b := NewElasticBeam3D(0, nodes, coords, 200000, 80000, sec, [3]float64{})

	ke := b.GetTangentStiffness()
	r, c := ke.Dims()
	for i := 0; i < r; i++ {
		for j := i + 1; j < c; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-4 {
				t.Errorf("Ke not symmetric: K[%d,%d]=%v != K[%d,%d]=%v",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

// TestElasticBeam3D_AxialStiffness verifies the axial stiffness entry K[0,0]
// of a 3D elastic beam element aligned with the X axis.
//
// Property: the axial mode of a bar is decoupled from bending and torsion;
// the axial stiffness coefficient at the first translational DOF equals:
//
//	K[0,0] = EA/L
//
// Parameters: E=200000, A=0.02, L=4, so EA/L = 200000·0.02/4 = 1000.
//
// Expected: K[0,0] = 1000 within relative tolerance 1e-8.
//
// Why valuable: ensures that axial and bending/torsional sub-problems are
// assembled independently and that the section area A is not confused with
// any moment-of-inertia value.
func TestElasticBeam3D_AxialStiffness(t *testing.T) {
	// Beam along X-axis: axial stiffness should be EA/L
	L := 4.0
	E := 200000.0
	A := 0.02
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	sec := section.BeamSection3D{A: A, Iy: 1e-4, Iz: 1e-4, J: 2e-4}
	b := NewElasticBeam3D(0, nodes, coords, E, 80000, sec, [3]float64{})

	ke := b.GetTangentStiffness()

	// For beam along X: DOF 0 (UX at node 0) should have K[0,0] = EA/L
	expected := E * A / L
	if got := ke.At(0, 0); math.Abs(got-expected)/expected > 1e-8 {
		t.Errorf("K[0,0] = %v, want %v (EA/L)", got, expected)
	}
}

// TestElasticBeam3D_RigidBodyTranslation verifies that a uniform rigid-body
// translation produces zero nodal forces for a 3D elastic beam.
//
// Property: Ke · u_rigid = 0 for u_rigid consisting of equal translation at
// both nodes in each global direction. A beam element (with no rotational
// rigid-body mode tested here) must not generate forces under pure translation.
//
// Parameters: beam from (0,0,0) to (3,4,0) — a diagonal orientation in the
// XY plane — to exercise the full local-to-global transformation.
// Tolerance is 1e-4 because the coordinate-transform residual accumulates for
// a non-axis-aligned beam.
//
// Expected: |Ke · u_rigid|_∞ < 1e-4 for X, Y, Z translation modes.
//
// Why valuable: a non-zero result would indicate a rigid-body mode error in
// the local-to-global rotation matrix, which would cause spurious self-equilibrated
// forces under zero-strain motion.
func TestElasticBeam3D_RigidBodyTranslation(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {3, 4, 0}}
	sec := section.BeamSection3D{A: 0.01, Iy: 1e-4, Iz: 1e-4, J: 2e-4}
	b := NewElasticBeam3D(0, nodes, coords, 200000, 80000, sec, [3]float64{})

	ke := b.GetTangentStiffness()

	// Uniform translation in each direction should produce zero forces
	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(12, nil)
		u.SetVec(dir, 1)   // node 0 translation
		u.SetVec(dir+6, 1) // node 1 translation

		f := mat.NewVecDense(12, nil)
		f.MulVec(ke, u)

		for i := 0; i < 12; i++ {
			if math.Abs(f.AtVec(i)) > 1e-4 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

// TestElasticBeam3D_BendingStiffness verifies the transverse bending stiffness
// coefficient K[1,1] for a 3D beam aligned with the X axis.
//
// Property: for an Euler-Bernoulli beam the consistent stiffness sub-block
// in the transverse (Y) direction gives:
//
//	K[1,1] = 12·E·Iz/L³
//
// which is the stiffness resisting equal-and-opposite unit transverse
// displacements at both ends (double-fixed-end reaction).
//
// Parameters: E=210000, L=5, Iz=8.33e-6 (approximate value for a rectangular
// section). Expected: 12·210000·8.33e-6/125 ≈ 0.1679.
//
// Why valuable: ensures the bending sub-matrix is built with the correct
// cubic dependence on L and is not confused with the rotational stiffness
// 4EI/L, which has a different L-dependence.
func TestElasticBeam3D_BendingStiffness(t *testing.T) {
	// Beam along X: 12EI/L³ for shear-like bending stiffness
	L := 5.0
	E := 210000.0
	Iz := 8.33e-6 // for bending in xy plane
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	sec := section.BeamSection3D{A: 0.01, Iy: 8.33e-6, Iz: Iz, J: 1e-5}
	b := NewElasticBeam3D(0, nodes, coords, E, 80000, sec, [3]float64{})

	ke := b.GetTangentStiffness()

	// K[1,1] = 12·E·Iz/L³ (transverse Y stiffness at node 0)
	expected := 12 * E * Iz / (L * L * L)
	if got := ke.At(1, 1); math.Abs(got-expected)/expected > 1e-6 {
		t.Errorf("K[1,1] = %v, want %v (12EIz/L³)", got, expected)
	}
}
