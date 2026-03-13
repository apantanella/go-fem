package zerolength

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestZeroLength_Symmetry verifies that the 12×12 element stiffness matrix of
// a 3D ZeroLength element is symmetric.
//
// Property: Ke = Keᵀ. A zero-length element couples node 0 and node 1 through
// six independent springs (3 translational, 3 rotational). The spring law
// F = k·Δu is symmetric by construction, but an implementation error in the
// off-diagonal coupling block could break symmetry.
//
// Parameters: spring stiffnesses [100, 200, 300, 10, 20, 30] for the six DOFs
// [UX, UY, UZ, RX, RY, RZ].
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-10 for all i < j.
//
// Why valuable: unsymmetry in a spring element would violate Newton's third law
// at the element level and corrupt any symmetric solver.
func TestZeroLength_Symmetry(t *testing.T) {
	z := NewZeroLength(0, [2]int{0, 1}, [6]float64{100, 200, 300, 10, 20, 30})
	ke := z.GetTangentStiffness()
	for i := 0; i < 12; i++ {
		for j := i + 1; j < 12; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-10 {
				t.Errorf("not symmetric: K[%d,%d]=%v != K[%d,%d]=%v",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

// TestZeroLength_SpringStiffness verifies the individual stiffness entries of
// the 3D ZeroLength element stiffness matrix.
//
// Property: for a zero-length spring with stiffness k_i in the i-th DOF,
// the spring law F = k_i · (u_i1 - u_i0) gives a 2×2 sub-block:
//
//	[  k_i  -k_i ]
//	[ -k_i   k_i ]
//
// where row/column i corresponds to node 0 DOF i and row/column i+6 to node 1.
//
// Parameters: springs = [100, 200, 300, 10, 20, 30].
//
// Expected for each i = 0..5:
//   - Ke[i,i]     = springs[i]    (node 0 self-stiffness)
//   - Ke[i+6,i+6] = springs[i]    (node 1 self-stiffness)
//   - Ke[i,i+6]   = -springs[i]   (off-diagonal coupling)
//
// Why valuable: any sign error or incorrect DOF offset in the coupling block
// would be caught, which would otherwise generate wrong restoring forces.
func TestZeroLength_SpringStiffness(t *testing.T) {
	springs := [6]float64{100, 200, 300, 10, 20, 30}
	z := NewZeroLength(0, [2]int{0, 1}, springs)
	ke := z.GetTangentStiffness()

	// Check diagonal blocks
	for i := 0; i < 6; i++ {
		if got := ke.At(i, i); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d] = %v, want %v", i, i, got, springs[i])
		}
		if got := ke.At(i+6, i+6); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d] = %v, want %v", i+6, i+6, got, springs[i])
		}
		if got := ke.At(i, i+6); math.Abs(got-(-springs[i])) > 1e-10 {
			t.Errorf("K[%d,%d] = %v, want %v", i, i+6, got, -springs[i])
		}
	}
}

// TestZeroLength_RigidBody verifies that equal displacement at both nodes of
// a 3D ZeroLength element produces zero spring forces.
//
// Property: a zero-length spring measures the relative displacement between
// node 0 and node 1. If both nodes move by the same amount in any DOF direction,
// the relative displacement is zero and no force is generated:
//
//	F = k · (u_node1 - u_node0) = k · 0 = 0
//
// Parameters: springs = [100, 200, 300, 10, 20, 30].
// Six directions are tested (three translational + three rotational).
//
// Expected: |Ke · u_rigid|_∞ < 1e-10 for each direction.
//
// Why valuable: confirms the element correctly computes relative (not absolute)
// deformation; an error would cause spurious forces under rigid-body motion.
func TestZeroLength_RigidBody(t *testing.T) {
	z := NewZeroLength(0, [2]int{0, 1}, [6]float64{100, 200, 300, 10, 20, 30})
	ke := z.GetTangentStiffness()

	// Equal displacements at both nodes → zero forces
	for dir := 0; dir < 6; dir++ {
		u := mat.NewVecDense(12, nil)
		u.SetVec(dir, 1)
		u.SetVec(dir+6, 1)

		f := mat.NewVecDense(12, nil)
		f.MulVec(ke, u)

		for i := 0; i < 12; i++ {
			if math.Abs(f.AtVec(i)) > 1e-10 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want 0", dir, i, f.AtVec(i))
			}
		}
	}
}

// ── ZeroLength2D ─────────────────────────────────────────────────────────────

// TestZeroLength2D_Symmetry verifies that the 4×4 element stiffness matrix of
// a 2D ZeroLength element is symmetric.
//
// Property: Ke = Keᵀ. Two independent springs (kx and ky) connect node 0 to
// node 1 in the X and Y directions; the 4×4 stiffness must be symmetric.
//
// Parameters: kx=100, ky=200.
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-10 for all i < j.
//
// Why valuable: a sign error in the off-diagonal 2×2 blocks would break
// symmetry and violate Newton's third law.
func TestZeroLength2D_Symmetry(t *testing.T) {
	z := NewZeroLength2D(0, [2]int{0, 1}, [2]float64{100, 200})
	ke := z.GetTangentStiffness()
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-10 {
				t.Errorf("not symmetric K[%d,%d]=%v != K[%d,%d]=%v", i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

// TestZeroLength2D_SpringStiffness verifies the individual stiffness entries of
// the 2D ZeroLength element stiffness matrix.
//
// Property: for springs [kx, ky] the 4×4 stiffness is block-diagonal:
//
//	[ kx   0  -kx   0  ]
//	[  0  ky    0  -ky ]
//	[-kx   0   kx   0  ]
//	[  0  -ky   0   ky ]
//
// DOF layout: [UX₀, UY₀, UX₁, UY₁].
//
// Parameters: kx=100, ky=200.
//
// Why valuable: catches any DOF-index error in the assembly of the 2D spring
// stiffness, such as swapping kx and ky or using the wrong offset.
func TestZeroLength2D_SpringStiffness(t *testing.T) {
	springs := [2]float64{100, 200}
	z := NewZeroLength2D(0, [2]int{0, 1}, springs)
	ke := z.GetTangentStiffness()
	for i := 0; i < 2; i++ {
		if got := ke.At(i, i); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i, i, got, springs[i])
		}
		if got := ke.At(i+2, i+2); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i+2, i+2, got, springs[i])
		}
		if got := ke.At(i, i+2); math.Abs(got-(-springs[i])) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i, i+2, got, -springs[i])
		}
	}
}

// TestZeroLength2D_SpringForce verifies that the spring force computed after
// a relative nodal displacement equals k·Δu for each independent direction.
//
// Physical derivation: node 1 moves (+2, +3) relative to node 0. The spring
// forces are:
//
//	Fx = kx · Δux = 100 · 2 = 200
//	Fy = ky · Δuy = 200 · 3 = 600
//
// Parameters: kx=100, ky=200; u = [0, 0, 2, 3] (node 0 at rest, node 1 displaced).
//
// Why valuable: confirms that the Update() and SpringForce() methods use the
// correct relative-displacement convention (u_node1 - u_node0) and apply the
// right stiffness to the right DOF.
func TestZeroLength2D_SpringForce(t *testing.T) {
	z := NewZeroLength2D(0, [2]int{0, 1}, [2]float64{100, 200})
	// node 1 moves +2 in X, +3 in Y relative to node 0
	_ = z.Update([]float64{0, 0, 2, 3})
	f := z.SpringForce()
	if math.Abs(f[0]-200) > 1e-10 {
		t.Errorf("Fx=%v want 200", f[0])
	}
	if math.Abs(f[1]-600) > 1e-10 {
		t.Errorf("Fy=%v want 600", f[1])
	}
}

// TestZeroLength2D_RigidBody verifies that equal displacement at both nodes
// of a 2D ZeroLength element produces zero spring forces.
//
// Property: F = k·(u_node1 - u_node0) = 0 when u_node0 = u_node1.
//
// Parameters: kx=100, ky=200. X and Y directions are tested individually.
//
// Expected: |Ke · u_rigid|_∞ < 1e-10.
//
// Why valuable: confirms the element measures relative (not absolute)
// displacement; a bug would cause spurious forces in any model where nodes
// sharing the same position are translated together.
func TestZeroLength2D_RigidBody(t *testing.T) {
	z := NewZeroLength2D(0, [2]int{0, 1}, [2]float64{100, 200})
	ke := z.GetTangentStiffness()
	for dir := 0; dir < 2; dir++ {
		u := mat.NewVecDense(4, nil)
		u.SetVec(dir, 1)
		u.SetVec(dir+2, 1)
		f := mat.NewVecDense(4, nil)
		f.MulVec(ke, u)
		for i := 0; i < 4; i++ {
			if math.Abs(f.AtVec(i)) > 1e-10 {
				t.Errorf("rigid body dir %d: f[%d]=%v want 0", dir, i, f.AtVec(i))
			}
		}
	}
}

// ── ZeroLength2DFrame ─────────────────────────────────────────────────────────

// TestZeroLength2DFrame_Symmetry verifies that the 6×6 element stiffness matrix
// of a ZeroLength2DFrame element is symmetric.
//
// Property: Ke = Keᵀ. The frame element has three independent springs:
// kUX, kUY, and kRZ. The 6×6 stiffness must be symmetric across all
// translational and rotational DOF pairs.
//
// Parameters: springs = [100, 200, 50] for [kUX, kUY, kRZ].
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-10 for all i < j.
//
// Why valuable: a rotational spring implemented with incorrect coupling to
// translational DOFs would break symmetry and generate inconsistent moments.
func TestZeroLength2DFrame_Symmetry(t *testing.T) {
	z := NewZeroLength2DFrame(0, [2]int{0, 1}, [3]float64{100, 200, 50})
	ke := z.GetTangentStiffness()
	for i := 0; i < 6; i++ {
		for j := i + 1; j < 6; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-10 {
				t.Errorf("not symmetric K[%d,%d]=%v != K[%d,%d]=%v", i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

// TestZeroLength2DFrame_SpringStiffness verifies the diagonal and off-diagonal
// coupling entries of the ZeroLength2DFrame stiffness matrix.
//
// Property: for springs [kUX, kUY, kRZ] the 6×6 stiffness has the same 2×2
// sub-block structure for each DOF pair as the scalar spring law F = k·Δu.
// DOF layout: [UX₀, UY₀, RZ₀, UX₁, UY₁, RZ₁].
//
// Parameters: springs = [100, 200, 50].
//
// Expected for each i = 0, 1, 2:
//   - Ke[i,i]     = springs[i]
//   - Ke[i+3,i+3] = springs[i]
//   - Ke[i,i+3]   = -springs[i]
//
// Why valuable: catches DOF-index errors or missing contributions for the
// rotational spring (kRZ) which often has a different code path.
func TestZeroLength2DFrame_SpringStiffness(t *testing.T) {
	springs := [3]float64{100, 200, 50}
	z := NewZeroLength2DFrame(0, [2]int{0, 1}, springs)
	ke := z.GetTangentStiffness()
	for i := 0; i < 3; i++ {
		if got := ke.At(i, i); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i, i, got, springs[i])
		}
		if got := ke.At(i+3, i+3); math.Abs(got-springs[i]) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i+3, i+3, got, springs[i])
		}
		if got := ke.At(i, i+3); math.Abs(got-(-springs[i])) > 1e-10 {
			t.Errorf("K[%d,%d]=%v want %v", i, i+3, got, -springs[i])
		}
	}
}

// TestZeroLength2DFrame_SpringForce verifies the spring forces computed after
// a prescribed relative nodal displacement for a ZeroLength2DFrame element.
//
// Physical derivation: node 1 moves (ux=1, uy=2, rz=0.1) relative to node 0.
// The spring forces are:
//
//	Fx  = kUX · Δux  = 100 · 1 = 100
//	Fy  = kUY · Δuy  = 200 · 2 = 400
//	Mz  = kRZ · Δrz  =  50 · 0.1 = 5
//
// Parameters: springs = [100, 200, 50]; u = [0, 0, 0, 1, 2, 0.1].
//
// Why valuable: confirms that Update() and SpringForce() correctly extract
// the relative displacement and apply separate stiffness values to each DOF,
// including the rotational DOF (rz).
func TestZeroLength2DFrame_SpringForce(t *testing.T) {
	z := NewZeroLength2DFrame(0, [2]int{0, 1}, [3]float64{100, 200, 50})
	// node 1 moves: ux=1, uy=2, rz=0.1 relative to node 0
	_ = z.Update([]float64{0, 0, 0, 1, 2, 0.1})
	f := z.SpringForce()
	if math.Abs(f[0]-100) > 1e-10 {
		t.Errorf("Fx=%v want 100", f[0])
	}
	if math.Abs(f[1]-400) > 1e-10 {
		t.Errorf("Fy=%v want 400", f[1])
	}
	if math.Abs(f[2]-5) > 1e-10 {
		t.Errorf("Mz=%v want 5", f[2])
	}
}

// TestZeroLength2DFrame_RigidBody verifies that equal displacement at both
// nodes of a ZeroLength2DFrame element produces zero spring forces.
//
// Property: F = k·(u_node1 - u_node0) = 0 when u_node0 = u_node1 for each
// of the three independent springs (UX, UY, RZ).
//
// Parameters: springs = [100, 200, 50]. Three directions (UX, UY, RZ) tested.
//
// Expected: |Ke · u_rigid|_∞ < 1e-10 for each direction.
//
// Why valuable: confirms the relative-displacement convention is correctly
// implemented for the rotational DOF (rz), which is often treated differently
// from translational DOFs.
func TestZeroLength2DFrame_RigidBody(t *testing.T) {
	z := NewZeroLength2DFrame(0, [2]int{0, 1}, [3]float64{100, 200, 50})
	ke := z.GetTangentStiffness()
	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(6, nil)
		u.SetVec(dir, 1)
		u.SetVec(dir+3, 1)
		f := mat.NewVecDense(6, nil)
		f.MulVec(ke, u)
		for i := 0; i < 6; i++ {
			if math.Abs(f.AtVec(i)) > 1e-10 {
				t.Errorf("rigid body dir %d: f[%d]=%v want 0", dir, i, f.AtVec(i))
			}
		}
	}
}
