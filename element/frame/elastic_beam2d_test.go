package frame

import (
	"math"
	"testing"

	"go-fem/section"

	"gonum.org/v1/gonum/mat"
)

// TestElasticBeam2D_Symmetry verifies that the 6×6 element stiffness matrix
// of a 2D elastic beam is symmetric.
//
// Property: Ke = Keᵀ (self-adjointness). Applies to the full 6×6 matrix
// including the coupling between bending and axial DOFs.
//
// Parameters: beam from (0,0) to (5,0), E=200000, A=0.01, Iz=1e-4.
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-4 for all i < j.
//
// Why valuable: catches sign errors in the off-diagonal bending-shear coupling
// entries (e.g., the 6EI/L² term that couples rotational and translational DOFs).
func TestElasticBeam2D_Symmetry(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {5, 0}}
	sec := section.BeamSection2D{A: 0.01, Iz: 1e-4}
	b := NewElasticBeam2D(0, nodes, coords, 200000, sec)

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

// TestElasticBeam2D_AxialStiffness verifies the axial stiffness entry K[0,0]
// of a 2D elastic beam element aligned with the X axis.
//
// Property: the axial mode is decoupled from bending for a straight beam.
// The stiffness coefficient at the first translational DOF equals:
//
//	K[0,0] = EA/L
//
// Parameters: E=200000, A=0.02, L=4; expected EA/L = 1000.
//
// Why valuable: confirms the axial sub-block is assembled correctly and that
// A is not accidentally replaced by a moment-of-inertia value.
func TestElasticBeam2D_AxialStiffness(t *testing.T) {
	L := 4.0
	E := 200000.0
	A := 0.02
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	sec := section.BeamSection2D{A: A, Iz: 1e-4}
	b := NewElasticBeam2D(0, nodes, coords, E, sec)

	ke := b.GetTangentStiffness()

	expected := E * A / L
	if got := ke.At(0, 0); math.Abs(got-expected)/expected > 1e-8 {
		t.Errorf("K[0,0] = %v, want %v (EA/L)", got, expected)
	}
}

// TestElasticBeam2D_BendingStiffness verifies the two key bending stiffness
// coefficients for a 2D Euler-Bernoulli beam along the X axis.
//
// Properties (DOF layout [UX₀,UY₀,RZ₀,UX₁,UY₁,RZ₁]):
//
//	K[1,1] = 12·E·Iz/L³  (transverse stiffness, equal-and-opposite unit displacements)
//	K[2,2] =  4·E·Iz/L   (rotational stiffness, near-end moment for unit rotation)
//
// Parameters: E=210000, L=5, Iz=8.33e-6.
//
// Why valuable: checks both the cubic L-dependence of the transverse entry and
// the linear L-dependence of the rotational entry; a factor-of-3 error (mixing
// 4EI/L with 12EI/L³) would be caught.
func TestElasticBeam2D_BendingStiffness(t *testing.T) {
	L := 5.0
	E := 210000.0
	Iz := 8.33e-6
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	sec := section.BeamSection2D{A: 0.01, Iz: Iz}
	b := NewElasticBeam2D(0, nodes, coords, E, sec)

	ke := b.GetTangentStiffness()

	// K[1,1] = 12·E·Iz/L³ (transverse stiffness at node 0)
	expected := 12 * E * Iz / (L * L * L)
	if got := ke.At(1, 1); math.Abs(got-expected)/expected > 1e-6 {
		t.Errorf("K[1,1] = %v, want %v (12EIz/L³)", got, expected)
	}

	// K[2,2] = 4·E·Iz/L (rotational stiffness at node 0)
	expected4 := 4 * E * Iz / L
	if got := ke.At(2, 2); math.Abs(got-expected4)/expected4 > 1e-6 {
		t.Errorf("K[2,2] = %v, want %v (4EIz/L)", got, expected4)
	}
}

// TestElasticBeam2D_RigidBodyTranslation verifies that a uniform rigid-body
// translation produces zero nodal forces for a 2D elastic beam.
//
// Property: Ke · u_rigid = 0 for u_rigid representing equal translation at both
// nodes. Tested for both X and Y directions on a non-axis-aligned beam so that
// the full local-to-global rotation matrix is exercised.
//
// Parameters: beam from (0,0) to (3,4) (diagonal orientation), E=200000,
// A=0.01, Iz=1e-4. Tolerance 1e-4 due to accumulated round-off in the
// rotation matrix.
//
// Why valuable: catches any rotation-matrix error that would cause a
// displacement-free rigid-body motion to generate spurious internal forces.
func TestElasticBeam2D_RigidBodyTranslation(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {3, 4}}
	sec := section.BeamSection2D{A: 0.01, Iz: 1e-4}
	b := NewElasticBeam2D(0, nodes, coords, 200000, sec)

	ke := b.GetTangentStiffness()

	for dir := 0; dir < 2; dir++ {
		u := mat.NewVecDense(6, nil)
		u.SetVec(dir, 1)   // node 0 translation
		u.SetVec(dir+3, 1) // node 1 translation

		f := mat.NewVecDense(6, nil)
		f.MulVec(ke, u)

		for i := 0; i < 6; i++ {
			if math.Abs(f.AtVec(i)) > 1e-4 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

// TestElasticBeam2D_EndForces verifies the local end forces recovered by a
// single 2D beam element under a cantilever (tip-load) displacement pattern.
//
// Physical scenario: fixed at node 0, tip displacement δ at node 1.
// The equivalent tip load producing δ under pure bending is (cantilever formula):
//
//	P = 3·E·Iz·δ / L³
//
// The corresponding tip rotation is:
//
//	θ_tip = P·L² / (2·E·Iz)
//
// Expected end forces in the local frame (N = axial, V = shear, M = moment):
//
//	Node i (fixed end): N=0, V=-P, M=-P·L
//	Node j (free end):  N=0, V=+P, M=0
//
// Parameters: E=210000, L=3, A=0.01, Iz=1e-4, δ=0.001.
//
// Why valuable: confirms that the EndForces() method correctly transforms from
// the nodal displacement vector back to local section forces, and verifies the
// sign convention (V and M at the two ends must satisfy equilibrium: V_i + V_j = 0,
// M_i + M_j + V_i·L = 0).
func TestElasticBeam2D_EndForces(t *testing.T) {
	// Horizontal cantilever beam: fixed at node 0, tip load Fy at node 1
	L := 3.0
	E := 210000.0
	A := 0.01
	Iz := 1e-4
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	sec := section.BeamSection2D{A: A, Iz: Iz}
	b := NewElasticBeam2D(0, nodes, coords, E, sec)

	// Small transverse displacement at node 1 only (cantilever)
	delta := 0.001
	// For a cantilever with unit tip load: δ = PL³/(3EI) → P = 3EIδ/L³
	P := 3 * E * Iz * delta / (L * L * L)

	// The full 6x6 system with BCs at node 0 gives:
	// Displacement at dofs 0,1,2 = 0; solve for dofs 3,4,5
	// For horizontal beam (cos=1, sin=0), global = local
	// Tip displacements: u₂=0, v₂=δ, θ₂ = PL²/(2EI)
	theta := P * L * L / (2 * E * Iz)
	disp := []float64{0, 0, 0, 0, delta, theta}
	b.Update(disp)

	ef := b.EndForces()

	// At node i (fixed end): N=0, V=-P, M=-PL
	if math.Abs(ef.I[0]) > 1e-4 {
		t.Errorf("EndForces.I[N] = %v, want ~0", ef.I[0])
	}
	if math.Abs(ef.I[1]+P) > 1e-4*math.Abs(P) {
		t.Errorf("EndForces.I[V] = %v, want %v", ef.I[1], -P)
	}
	if math.Abs(ef.I[2]+P*L) > 1e-4*math.Abs(P*L) {
		t.Errorf("EndForces.I[M] = %v, want %v", ef.I[2], -P*L)
	}

	// At node j (free end): N=0, V=P, M=0
	if math.Abs(ef.J[0]) > 1e-4 {
		t.Errorf("EndForces.J[N] = %v, want ~0", ef.J[0])
	}
	if math.Abs(ef.J[1]-P) > 1e-4*math.Abs(P) {
		t.Errorf("EndForces.J[V] = %v, want %v", ef.J[1], P)
	}
	if math.Abs(ef.J[2]) > 1e-4*math.Abs(P*L) {
		t.Errorf("EndForces.J[M] = %v, want ~0", ef.J[2])
	}
}

// TestElasticBeam2D_MatchesBeam3D verifies that the in-plane stiffness sub-block
// of a 3D beam matches the corresponding 2D beam stiffness matrix entry-by-entry.
//
// Property: for a beam in the XY plane (Z=0) the 2D formulation must be a
// consistent specialisation of the 3D formulation. The DOF mapping is:
//
//	2D index: 0   1   2   3   4   5
//	3D index: 0   1   5   6   7  11
//	DOF:     UX₀ UY₀ RZ₀ UX₁ UY₁ RZ₁
//
// Parameters: E=210000, G=80000, A=0.01, Iz=1e-4, L=5.
//
// Why valuable: a discrepancy would indicate an inconsistency between the 2D
// and 3D local stiffness formulations (e.g., a wrong sign in the 2D bending
// rows relative to the 3D Bernoulli-Euler sub-block).
func TestElasticBeam2D_MatchesBeam3D(t *testing.T) {
	// For a beam in the XY plane, the 2D beam stiffness should match the
	// corresponding rows/cols of the 3D beam stiffness.
	E := 210000.0
	G := 80000.0
	A := 0.01
	Iz := 1e-4
	nodes := [2]int{0, 1}
	c2 := [2][2]float64{{0, 0}, {5, 0}}
	c3 := [2][3]float64{{0, 0, 0}, {5, 0, 0}}
	sec2 := section.BeamSection2D{A: A, Iz: Iz}
	sec3 := section.BeamSection3D{A: A, Iy: 1e-4, Iz: Iz, J: 2e-4}

	b2 := NewElasticBeam2D(0, nodes, c2, E, sec2)
	b3 := NewElasticBeam3D(0, nodes, c3, E, G, sec3, [3]float64{})

	ke2 := b2.GetTangentStiffness()
	ke3 := b3.GetTangentStiffness()

	// 2D DOFs: [UX₁,UY₁,RZ₁, UX₂,UY₂,RZ₂] = indices 0,1,2,3,4,5
	// 3D DOFs: [UX₁,UY₁,UZ₁,RX₁,RY₁,RZ₁, UX₂,...] = indices 0,1,2,3,4,5,6,7,8,9,10,11
	// Mapping: 2D[0]→3D[0], 2D[1]→3D[1], 2D[2]→3D[5], 2D[3]→3D[6], 2D[4]→3D[7], 2D[5]→3D[11]
	map2to3 := [6]int{0, 1, 5, 6, 7, 11}

	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			v2 := ke2.At(i, j)
			v3 := ke3.At(map2to3[i], map2to3[j])
			if math.Abs(v2) > 1e-10 {
				rel := math.Abs(v2-v3) / math.Abs(v2)
				if rel > 1e-6 {
					t.Errorf("K2D[%d,%d]=%.6g != K3D[%d,%d]=%.6g (rel=%.3e)",
						i, j, v2, map2to3[i], map2to3[j], v3, rel)
				}
			} else if math.Abs(v3) > 1e-6 {
				t.Errorf("K2D[%d,%d]=%.6g but K3D[%d,%d]=%.6g",
					i, j, v2, map2to3[i], map2to3[j], v3)
			}
		}
	}
}

// TestElasticBeam2D_EquivalentNodalLoad verifies the work-equivalent nodal load
// vector for a uniformly distributed transverse load (UDL) on a 2D beam.
//
// Physical derivation (consistent nodal loads via virtual work):
// For a UDL of intensity q (force per unit length) over a beam of length L,
// the equivalent nodal forces and moments in the local frame are:
//
//	Fy_i = Fy_j = -q·L/2      (equal transverse reactions)
//	Mz_i = -q·L²/12           (hogging moment at near end)
//	Mz_j = +q·L²/12           (sagging moment at far end)
//
// Parameters: L=4, E=200000 (not used for load calculation), q=10.
//
//	Fy = -10·4/2 = -20  (downward, negative Y direction)
//	Mz_i = -10·16/12 ≈ -13.33
//	Mz_j = +10·16/12 ≈ +13.33
//
// The load direction vector is (0,-1,0) with magnitude q=10, giving
// a downward load on a horizontal beam (local = global).
//
// Why valuable: confirms the EquivalentNodalLoad() method applies the
// correct Hermitian shape-function integrals; an error in the sign convention
// or the L² factor in the moment terms would be detected.
func TestElasticBeam2D_EquivalentNodalLoad(t *testing.T) {
	L := 4.0
	E := 200000.0
	A := 0.01
	Iz := 1e-4
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	sec := section.BeamSection2D{A: A, Iz: Iz}
	b := NewElasticBeam2D(0, nodes, coords, E, sec)

	q := 10.0 // uniform load in -Y direction
	f := b.EquivalentNodalLoad([3]float64{0, -1, 0}, q)

	// For horizontal beam: local = global
	// Fy_i = Fy_j = q*L/2 = -20 (downward)
	// Mz_i = q*L²/12, Mz_j = -q*L²/12
	tol := 1e-8
	if math.Abs(f.AtVec(0)) > tol {
		t.Errorf("f[0] = %v, want 0", f.AtVec(0))
	}
	if math.Abs(f.AtVec(1)-(-q*L/2)) > tol {
		t.Errorf("f[1] = %v, want %v", f.AtVec(1), -q*L/2)
	}
	if math.Abs(f.AtVec(2)-(-q*L*L/12)) > tol {
		t.Errorf("f[2] = %v, want %v", f.AtVec(2), -q*L*L/12)
	}
	if math.Abs(f.AtVec(3)) > tol {
		t.Errorf("f[3] = %v, want 0", f.AtVec(3))
	}
	if math.Abs(f.AtVec(4)-(-q*L/2)) > tol {
		t.Errorf("f[4] = %v, want %v", f.AtVec(4), -q*L/2)
	}
	if math.Abs(f.AtVec(5)-(q*L*L/12)) > tol {
		t.Errorf("f[5] = %v, want %v", f.AtVec(5), q*L*L/12)
	}
}
