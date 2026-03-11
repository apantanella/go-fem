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

// TestBodyForceHexa8 verifies total body-force sum equals ρ·g·V for a unit cube.
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

// TestSurfacePressureUnitSquare verifies that a unit pressure on a flat
// unit-square face in the XY-plane produces total force = P·A in the Z direction.
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
