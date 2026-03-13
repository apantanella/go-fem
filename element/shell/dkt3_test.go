package shell

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestDKT3_Symmetry verifies that the 18×18 element stiffness matrix of a
// Discrete Kirchhoff Triangle (DKT3) is symmetric.
//
// Property: Ke = Keᵀ. The DKT3 formulation is based on a hybrid
// displacement/rotation interpolation; the resulting stiffness integral must
// still yield a symmetric matrix.
//
// Parameters: right-triangle with vertices at (0,0,0), (1,0,0), (0,1,0) in
// the XY plane; E=200000, ν=0.3, thickness=0.1.
// Each node has 6 DOFs [UX,UY,UZ,RX,RY,RZ], giving 18 DOFs total,
// though only the out-of-plane bending DOFs carry stiffness.
//
// Expected: relative asymmetry < 1e-6 for entries with magnitude > 1e-10.
//
// Why valuable: catches sign errors in the Kirchhoff constraint equations
// or incorrectly transposed sub-block assembly.
func TestDKT3_Symmetry(t *testing.T) {
	nodes := [3]int{0, 1, 2}
	coords := [3][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	e := NewDiscreteKirchhoffTriangle(0, nodes, coords, 200000, 0.3, 0.1)

	ke := e.GetTangentStiffness()
	r, c := ke.Dims()
	for i := 0; i < r; i++ {
		for j := i + 1; j < c; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			avg := (math.Abs(ke.At(i, j)) + math.Abs(ke.At(j, i))) / 2
			if avg > 1e-10 && diff/avg > 1e-6 {
				t.Errorf("Ke not symmetric: K[%d,%d]=%v != K[%d,%d]=%v", i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

// TestDKT3_RigidBodyModes verifies that the DKT3 element exactly represents
// two plate rigid-body modes: (a) uniform out-of-plane translation w=const,
// and (b) a rigid-plate rotation field.
//
// Property: for both rigid-body modes the product Ke · u_rigid must be zero
// (within numerical tolerance), confirming that the DKT3 element is complete
// with respect to rigid-plate motions (completeness condition for plate elements).
//
// Mode (a) — uniform out-of-plane translation:
//   - DOF layout per node: [UX,UY,UZ,RX,RY,RZ] at global index 6n+k
//   - u_w[6n+2] = 1 for all nodes n (unit vertical translation)
//   - Expected: Ke · u_w = 0
//
// Mode (b) — rigid plate rotation about the X axis at amplitude a=0.2:
//   - w(y) = a·y  (out-of-plane displacement proportional to Y coordinate)
//   - RX = a      (rotation about X equals the slope a)
//   - RY = 0      (no slope in X direction)
//   - u_R[6n+2] = a·y_n, u_R[6n+3] = a, u_R[6n+4] = 0
//   - Expected: Ke · u_R = 0
//
// Parameters: right-angle triangle from (0,0,0)-(2,0,0)-(0,1,0),
// E=200000, ν=0.3, thickness=0.05.
//
// Tolerance: 1e-3 (plate elements have higher round-off than bar elements).
//
// Why valuable: failure would indicate that the DKT3 constraint equations
// are incomplete and that rigid-plate motions would generate spurious internal
// moments — a fundamental flaw that would corrupt all plate bending results.
func TestDKT3_RigidBodyModes(t *testing.T) {
	nodes := [3]int{0, 1, 2}
	coords := [3][3]float64{{0, 0, 0}, {2, 0, 0}, {0, 1, 0}}
	e := NewDiscreteKirchhoffTriangle(0, nodes, coords, 200000, 0.3, 0.05)
	ke := e.GetTangentStiffness()

	// Rigid out-of-plane translation: w = const
	uW := mat.NewVecDense(18, nil)
	for n := 0; n < 3; n++ {
		uW.SetVec(6*n+2, 1.0)
	}
	fW := mat.NewVecDense(18, nil)
	fW.MulVec(ke, uW)
	for i := 0; i < 18; i++ {
		if math.Abs(fW.AtVec(i)) > 1e-3 {
			t.Errorf("rigid w translation: f[%d]=%v, want ~0", i, fW.AtVec(i))
		}
	}

	// Rigid rotation-compatible field: w = a*y, rx = a, ry = 0
	a := 0.2
	uR := mat.NewVecDense(18, nil)
	for n := 0; n < 3; n++ {
		y := coords[n][1]
		uR.SetVec(6*n+2, a*y)
		uR.SetVec(6*n+3, a)
		uR.SetVec(6*n+4, 0)
	}
	fR := mat.NewVecDense(18, nil)
	fR.MulVec(ke, uR)
	for i := 0; i < 18; i++ {
		if math.Abs(fR.AtVec(i)) > 1e-3 {
			t.Errorf("rigid rotation field: f[%d]=%v, want ~0", i, fR.AtVec(i))
		}
	}
}

// TestDKT3_PositiveDiagonal verifies that all 18 diagonal entries of the DKT3
// stiffness matrix are strictly positive.
//
// Property: a necessary condition for the stiffness matrix to be positive
// semi-definite is that all diagonal entries are non-negative. For a bending
// element all active DOFs (out-of-plane displacement w and rotations RX, RY)
// must have a positive self-stiffness.
//
// Parameters: right-triangle at (0,0,0)-(1,0,0)-(0,1,0), E=200000, ν=0.3,
// thickness=0.1.
//
// Expected: Ke[i,i] > 0 for i = 0, ..., 17.
//
// Why valuable: a zero diagonal entry would mean that DOF i has no resistance
// to self-deformation, indicating an unrestrained degree of freedom that would
// cause singularity in the assembled global stiffness matrix.
func TestDKT3_PositiveDiagonal(t *testing.T) {
	nodes := [3]int{0, 1, 2}
	coords := [3][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	e := NewDiscreteKirchhoffTriangle(0, nodes, coords, 200000, 0.3, 0.1)

	ke := e.GetTangentStiffness()
	for i := 0; i < 18; i++ {
		if ke.At(i, i) <= 0 {
			t.Errorf("K[%d,%d]=%v, expected positive diagonal", i, i, ke.At(i, i))
		}
	}
}
