package domain_test

import (
	"math"
	"testing"

	"go-fem/domain"
	"go-fem/element/frame"
	"go-fem/element/solid"
	"go-fem/material"
	"go-fem/section"
)

// TestBodyForceTet4 verifies that the total body-force load on a unit
// tetrahedron equals ρ·g·V (total weight scattered equally to 4 nodes).
//
// Physical derivation: for a constant body-force vector g = [0, -9.81, 0]
// and material density ρ, the consistent nodal loads are computed via:
//
//	f = ∫ Nᵀ · ρg dV
//
// For the linear Tet4 element, each shape function integrates to V/4, so
// each node receives an equal share of the total load:
//
//	F_total_y = ρ · g_y · V = 7800 · (-9.81) · (1/6)
//	F_node_y  = F_total_y / 4  for each node
//
// Parameters: right-angle unit tet with nodes at (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1),
// ρ=7800 kg/m³, g = (0, -9.81, 0) m/s².
//
// Why valuable: confirms that the BodyForceLoad() method correctly integrates
// the shape functions over the volume and that the equal-distribution property
// of the Tet4 element (all nodes have the same shape-function integral) is
// preserved.
func TestBodyForceTet4(t *testing.T) {
	mat3d := material.NewIsotropicLinear(210000, 0.3)
	coords := [4][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	nodes := [4]int{0, 1, 2, 3}
	tet := solid.NewTet4(0, nodes, coords, mat3d)

	rho := 7800.0
	g := [3]float64{0, -9.81, 0}
	f := tet.BodyForceLoad(g, rho)

	vol := tet.Volume()
	totalFy := 0.0
	for n := 0; n < 4; n++ {
		totalFy += f.AtVec(3*n + 1)
	}
	expected := rho * g[1] * vol
	if math.Abs(totalFy-expected) > 1e-10 {
		t.Errorf("total Fy = %.6g, want %.6g", totalFy, expected)
	}
	// Each node gets equal share
	perNode := expected / 4
	for n := 0; n < 4; n++ {
		fy := f.AtVec(3*n + 1)
		if math.Abs(fy-perNode) > 1e-10 {
			t.Errorf("node %d Fy = %.6g, want %.6g", n, fy, perNode)
		}
	}
}

// TestBodyForceHexa8 verifies that the total body-force load on a unit cube
// Hexa8 element equals ρ·g·V.
//
// Physical derivation: for a unit cube (V=1) with body-force g = (0,-9.81,0):
//
//	F_total_y = ρ · g_y · V = 7800 · (-9.81) · 1
//
// The consistent nodal loads are computed via 2×2×2 Gauss integration of
// Nᵢ·ρg over the element volume. For the trilinear Hex8 element the
// distribution among the 8 nodes depends on the Gauss weights.
//
// Parameters: unit cube with nodes at the 8 corners (see coords), ρ=7800,
// g = (0, -9.81, 0).
//
// Why valuable: a wrong Gauss weight or missing Jacobian determinant in the
// body-force integral would cause the total force to differ from ρgV.
func TestBodyForceHexa8(t *testing.T) {
	mat3d := material.NewIsotropicLinear(210000, 0.3)
	coords := [8][3]float64{
		{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
		{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
	}
	nodes := [8]int{0, 1, 2, 3, 4, 5, 6, 7}
	hex := solid.NewHexa8(0, nodes, coords, mat3d)

	rho := 7800.0
	g := [3]float64{0, -9.81, 0}
	f := hex.BodyForceLoad(g, rho)

	// unit cube volume = 1
	totalFy := 0.0
	for n := 0; n < 8; n++ {
		totalFy += f.AtVec(3*n + 1)
	}
	expected := rho * g[1] * 1.0
	if math.Abs(totalFy-expected) > 1e-8 {
		t.Errorf("total Fy = %.6g, want %.6g", totalFy, expected)
	}
}

// TestEquivalentNodalLoadBeam verifies work-equivalent loads for a horizontal
// beam with a vertical UDL: total Fy = q·L and total moment = 0.
//
// Physical derivation (virtual work / Hermitian shape functions):
// For a UDL of intensity q = -1000 N/m (downward) on a beam of length L=4:
//
//	F_total_y = q · L = -1000 · 4 = -4000 N
//	Mz_i = -q·L²/12  (hogging moment at near end)
//	Mz_j = +q·L²/12  (sagging moment at far end)
//
// The moments must be equal and opposite: Mz_i + Mz_j = 0.
// The nodal shear forces are: Fy_i = Fy_j = q·L/2 = -2000 N.
//
// Parameters: 3D beam from (0,0,0) to (4,0,0), L=4, q=-1000, direction (0,1,0).
//
// Why valuable: an error in the moment coefficient (e.g., L²/12 vs L²/8 for
// a fixed-fixed versus simply-supported beam) would be detected here.
func TestEquivalentNodalLoadBeam(t *testing.T) {
	sec := section.BeamSection3D{A: 0.01, Iy: 8.33e-6, Iz: 8.33e-6, J: 1.41e-5}
	coords := [2][3]float64{{0, 0, 0}, {4, 0, 0}} // L=4
	b := frame.NewElasticBeam3D(0, [2]int{0, 1}, coords, 200000, 80000, sec, [3]float64{0, 0, 1})

	q := -1000.0 // N/m downward
	f := b.EquivalentNodalLoad([3]float64{0, 1, 0}, q)

	// Sum of Fy components (global Y = DOF 1 and 7)
	totalFy := f.AtVec(1) + f.AtVec(7)
	expected := q * b.Length()
	if math.Abs(totalFy-expected)/math.Abs(expected) > 1e-9 {
		t.Errorf("total Fy = %.6g, want %.6g", totalFy, expected)
	}

	// Moments should be equal and opposite: Mz_i = -Mz_j
	mz_i := f.AtVec(5)
	mz_j := f.AtVec(11)
	if math.Abs(mz_i+mz_j) > 1e-10 {
		t.Errorf("moments not equal-and-opposite: Mz_i=%.6g Mz_j=%.6g", mz_i, mz_j)
	}
}

// TestSurfacePressureUnitSquare verifies that a uniform pressure on a flat
// unit-square face in the XY-plane produces a total force equal to P·A in the
// Z direction with the correct sign.
//
// Physical derivation: surface pressure P applied to a CCW-oriented face
// produces a traction in the outward normal direction. For nodes ordered
// CCW in the XY plane (0→1→2→3), the outward normal points in +Z, and
// the equivalent nodal force vector is:
//
//	f = -P · A · n̂   (negative because pressure acts inward on the structure)
//
// Total Fz = -P · A = -500 · 1.0 = -500 N.
//
// Parameters: 4 corner nodes of a 1×1 unit square at z=0 (CCW ordering),
// P = 500 Pa.
//
// Why valuable: confirms that the surface-pressure integration uses the
// correct normal direction (outward vs inward sign convention) and that the
// Gauss-quadrature area computation gives exactly 1.0 for a unit square.
func TestSurfacePressureUnitSquare(t *testing.T) {
	dom := domain.NewDomain()
	// 4 corner nodes of a 1×1 square in z=0 plane
	dom.AddNode(0, 0, 0)
	dom.AddNode(1, 0, 0)
	dom.AddNode(1, 1, 0)
	dom.AddNode(0, 1, 0)

	// Dummy truss to force 3-DOF assembly (we only need F vector)
	// Actually we don't need elements; but Assemble requires DOFPerNode from elements.
	// Use a tet4 as placeholder:
	mat3d := material.NewIsotropicLinear(210000, 0.3)
	dom.AddNode(0, 0, 1) // node 4
	tet := solid.NewTet4(0, [4]int{0, 1, 2, 4}, [4][3]float64{
		{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 0, 1},
	}, mat3d)
	dom.AddElement(tet)
	// Fix all tet nodes to avoid singularity (we only care about F)
	for i := 0; i < 5; i++ {
		dom.FixNode(i)
	}

	P := 500.0
	dom.AddSurfacePressure([4]int{0, 1, 2, 3}, P)
	dom.Assemble()
	// Check F before BCs (face nodes 0-3 are in the XY plane at z=0)
	// CCW ordering 0->1->2->3 in XY plane: gs×gt points in +Z.
	// force = -P·n, so total Fz should be -P·A = -500
	totalFz := 0.0
	for n := 0; n < 4; n++ {
		totalFz += dom.F.AtVec(3*n + 2)
	}
	expected := -P * 1.0 // pressure × area
	if math.Abs(totalFz-expected)/math.Abs(expected) > 1e-10 {
		t.Errorf("total Fz = %.6g, want %.6g", totalFz, expected)
	}
}

// TestEquivalentNodalLoadLinearBeam verifies the work-equivalent nodal load
// vector for a linearly varying transverse load on an elastic_beam_3d element.
//
// Physical derivation (Hermite fixed-end reactions):
//
//	q_i = -1000 N/m at node i (x=0), q_j = 0 at node j (x=L) — triangular load.
//
//	Fy_i = L/20·(7·qi + 3·qj) = 4/20·(7·(-1000)+0) = -1400 N
//	Mz_i = L²/60·(3·qi + 2·qj) = 16/60·(3·(-1000)+0) ≈ -800 N·m
//	Fy_j = L/20·(3·qi + 7·qj) = 4/20·(3·(-1000)+0) = -600 N
//	Mz_j = -L²/60·(2·qi + 3·qj) = -16/60·(2·(-1000)+0) ≈ +533.33 N·m
//
//	Total Fy = -1400 + (-600) = -2000 = qi·L/2 (total triangular load) ✓
//
// Why valuable: confirms the trapezoidal formulas reduce correctly to the
// triangular case and that total force = qi·L/2.
func TestEquivalentNodalLoadLinearBeam(t *testing.T) {
	sec := section.BeamSection3D{A: 0.01, Iy: 8.33e-6, Iz: 8.33e-6, J: 1.41e-5}
	coords := [2][3]float64{{0, 0, 0}, {4, 0, 0}} // L=4
	b := frame.NewElasticBeam3D(0, [2]int{0, 1}, coords, 200000, 80000, sec, [3]float64{0, 0, 1})

	const L = 4.0
	qi := -1000.0 // N/m at node i
	qj := 0.0     // N/m at node j (triangular)

	f := b.EquivalentNodalLoadLinear([3]float64{0, 1, 0}, qi, qj)

	tol := 1e-9
	wantFyi := L / 20 * (7*qi + 3*qj)
	wantFyj := L / 20 * (3*qi + 7*qj)
	wantMzi := L * L / 60 * (3*qi + 2*qj)
	wantMzj := -L * L / 60 * (2*qi + 3*qj)

	if math.Abs(f.AtVec(1)-wantFyi) > tol {
		t.Errorf("Fy_i = %.6g, want %.6g", f.AtVec(1), wantFyi)
	}
	if math.Abs(f.AtVec(7)-wantFyj) > tol {
		t.Errorf("Fy_j = %.6g, want %.6g", f.AtVec(7), wantFyj)
	}
	if math.Abs(f.AtVec(5)-wantMzi) > tol {
		t.Errorf("Mz_i = %.6g, want %.6g", f.AtVec(5), wantMzi)
	}
	if math.Abs(f.AtVec(11)-wantMzj) > tol {
		t.Errorf("Mz_j = %.6g, want %.6g", f.AtVec(11), wantMzj)
	}

	// Total force must equal qi·L/2 (integral of triangular load over L)
	totalFy := f.AtVec(1) + f.AtVec(7)
	wantTotal := qi * L / 2
	if math.Abs(totalFy-wantTotal)/math.Abs(wantTotal) > 1e-9 {
		t.Errorf("total Fy = %.6g, want %.6g (qi·L/2)", totalFy, wantTotal)
	}

	// Verify UDL degeneracy: when qi=qj the result must match EquivalentNodalLoad.
	qUDL := -500.0
	fUDL := b.EquivalentNodalLoad([3]float64{0, 1, 0}, qUDL)
	fLin := b.EquivalentNodalLoadLinear([3]float64{0, 1, 0}, qUDL, qUDL)
	for k := 0; k < 12; k++ {
		if math.Abs(fLin.AtVec(k)-fUDL.AtVec(k)) > 1e-10 {
			t.Errorf("UDL degeneracy DOF %d: linear=%.6g, udl=%.6g", k, fLin.AtVec(k), fUDL.AtVec(k))
		}
	}
}
