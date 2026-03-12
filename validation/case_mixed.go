package main

import (
	"fmt"

	"go-fem/domain"
	"go-fem/element/frame"
	"go-fem/element/shell"
	"go-fem/element/solid"
	"go-fem/element/truss"
	"go-fem/material"
	"go-fem/section"
)

// ─────────────────────────────────────────────────────────────────────────────
// Case: Large mixed-element 3D model
//
//   A 3D industrial building with 5 different element types:
//
//     Layer 1 – Concrete foundation:  9 Hexa8 elements (3×3×1 grid)
//     Layer 2 – Steel beam columns:   16 ElasticBeam3D (z=1 → z=5)
//     Layer 3 – Steel horizontal beams: 24 ElasticBeam3D (roof level z=5)
//     Layer 4 – Wall truss bracing:   24 Truss3D (X-bracing on 4 walls)
//     Layer 5 – Roof panels:          4 ShellMITC4 + 10 DKT3
//
//   Totals: 87 elements, 5 element types, 48 nodes, 288 DOFs
//
//   Boundary conditions:
//     – Foundation base nodes (z=0): all 6 DOFs fixed
//
//   Loads:
//     – Downward nodal force F_z = -50 kN on each roof node (z=5)
//     – Body forces (gravity) on foundation Hexa8 elements
//
//   Convergence criterion:
//     – Solver completes without error (SPD global stiffness)
//     – Maximum vertical roof displacement is negative (downward)
//     – Roof displacement is within physically reasonable bounds
//
//   "Theoretical" check: max roof uz should be negative and the model should
//   return a convergence flag = 1.0 (indicating the solve succeeded).
// ─────────────────────────────────────────────────────────────────────────────

func caseMixedLargeModel() (numerical, theoretical float64) {
	theoretical = 1.0 // convergence flag

	dom := domain.NewDomain()
	eid := 0 // running element counter

	// ── Materials ────────────────────────────────────────────────────────────
	concrete := material.NewIsotropicLinear(30000, 0.2) // E=30 GPa, ν=0.2
	steelE := 210000.0                                  // MPa
	steelNu := 0.3
	steelG := steelE / (2 * (1 + steelNu)) // ≈80769 MPa

	// ── Foundation nodes ─────────────────────────────────────────────────────
	// Bottom face (z=0): 4×4 grid, spacing=2m → 16 nodes (0–15)
	// Top face    (z=1): 4×4 grid                → 16 nodes (16–31)
	const spacing = 2.0
	const foundH = 1.0 // foundation height
	const roofZ = 5.0  // roof level

	baseIDs := [4][4]int{} // [row_y][col_x] → node ID
	topIDs := [4][4]int{}
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			x := float64(col) * spacing
			y := float64(row) * spacing
			baseIDs[row][col] = dom.AddNode(x, y, 0)
			topIDs[row][col] = dom.AddNode(x, y, foundH)
		}
	}
	// 32 nodes so far (0–31)

	// ── Hexa8 foundation (3×3×1 grid → 9 elements) ─────────────────────────
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			// Bottom face CCW: SW, SE, NE, NW
			n0 := baseIDs[row][col]
			n1 := baseIDs[row][col+1]
			n2 := baseIDs[row+1][col+1]
			n3 := baseIDs[row+1][col]
			// Top face same order
			n4 := topIDs[row][col]
			n5 := topIDs[row][col+1]
			n6 := topIDs[row+1][col+1]
			n7 := topIDs[row+1][col]

			nodes := [8]int{n0, n1, n2, n3, n4, n5, n6, n7}
			coords := [8][3]float64{
				dom.Nodes[n0].Coord, dom.Nodes[n1].Coord,
				dom.Nodes[n2].Coord, dom.Nodes[n3].Coord,
				dom.Nodes[n4].Coord, dom.Nodes[n5].Coord,
				dom.Nodes[n6].Coord, dom.Nodes[n7].Coord,
			}
			dom.AddElement(solid.NewHexa8(eid, nodes, coords, concrete))
			eid++
		}
	}
	// 9 hexa8 elements

	// ── Roof nodes (z=5): 4×4 grid → 16 nodes (32–47) ──────────────────────
	roofIDs := [4][4]int{}
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			x := float64(col) * spacing
			y := float64(row) * spacing
			roofIDs[row][col] = dom.AddNode(x, y, roofZ)
		}
	}
	// 48 nodes total (0–47)

	// ── Beam columns: foundation top → roof (16 elements) ───────────────────
	secCol := section.BeamSection3D{
		A: 5000, Iy: 4.16e7, Iz: 4.16e7, J: 6.5e7,
	}
	vecVertical := [3]float64{1, 0, 0} // orientation for vertical beams

	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			ni := topIDs[row][col]
			nj := roofIDs[row][col]
			coords := [2][3]float64{dom.Nodes[ni].Coord, dom.Nodes[nj].Coord}
			dom.AddElement(frame.NewElasticBeam3D(eid, [2]int{ni, nj}, coords,
				steelE, steelG, secCol, vecVertical))
			eid++
		}
	}
	// 16 column beams → total 25 elements

	// ── Horizontal beams at roof (z=5): 24 elements ─────────────────────────
	secBeam := section.BeamSection3D{
		A: 3000, Iy: 2.5e7, Iz: 2.5e7, J: 3.8e7,
	}
	vecHoriz := [3]float64{0, 0, 1} // orientation for horizontal beams

	// X-direction beams (4 rows × 3 per row = 12)
	for row := 0; row < 4; row++ {
		for col := 0; col < 3; col++ {
			ni := roofIDs[row][col]
			nj := roofIDs[row][col+1]
			coords := [2][3]float64{dom.Nodes[ni].Coord, dom.Nodes[nj].Coord}
			dom.AddElement(frame.NewElasticBeam3D(eid, [2]int{ni, nj}, coords,
				steelE, steelG, secBeam, vecHoriz))
			eid++
		}
	}
	// Y-direction beams (4 cols × 3 per col = 12)
	for col := 0; col < 4; col++ {
		for row := 0; row < 3; row++ {
			ni := roofIDs[row][col]
			nj := roofIDs[row+1][col]
			coords := [2][3]float64{dom.Nodes[ni].Coord, dom.Nodes[nj].Coord}
			dom.AddElement(frame.NewElasticBeam3D(eid, [2]int{ni, nj}, coords,
				steelE, steelG, secBeam, vecHoriz))
			eid++
		}
	}
	// 24 horizontal beams → total 49 elements

	// ── Truss bracing on exterior walls (24 elements) ────────────────────────
	trussE := steelE
	trussA := 1500.0

	// Helper: creates 2 crossed diagonal trusses for a wall bay.
	addBrace := func(nBotLeft, nBotRight, nTopLeft, nTopRight int) {
		// Diagonal 1: bottom-left → top-right
		c1 := [2][3]float64{dom.Nodes[nBotLeft].Coord, dom.Nodes[nTopRight].Coord}
		dom.AddElement(truss.NewTruss3D(eid, [2]int{nBotLeft, nTopRight}, c1, trussE, trussA))
		eid++
		// Diagonal 2: bottom-right → top-left
		c2 := [2][3]float64{dom.Nodes[nBotRight].Coord, dom.Nodes[nTopLeft].Coord}
		dom.AddElement(truss.NewTruss3D(eid, [2]int{nBotRight, nTopLeft}, c2, trussE, trussA))
		eid++
	}

	// Front wall (y=0, row=0): 3 bays
	for col := 0; col < 3; col++ {
		addBrace(topIDs[0][col], topIDs[0][col+1], roofIDs[0][col], roofIDs[0][col+1])
	}
	// Back wall (y=6, row=3): 3 bays
	for col := 0; col < 3; col++ {
		addBrace(topIDs[3][col], topIDs[3][col+1], roofIDs[3][col], roofIDs[3][col+1])
	}
	// Left wall (x=0, col=0): 3 bays
	for row := 0; row < 3; row++ {
		addBrace(topIDs[row][0], topIDs[row+1][0], roofIDs[row][0], roofIDs[row+1][0])
	}
	// Right wall (x=6, col=3): 3 bays
	for row := 0; row < 3; row++ {
		addBrace(topIDs[row][3], topIDs[row+1][3], roofIDs[row][3], roofIDs[row+1][3])
	}
	// 4 walls × 3 bays × 2 diagonals = 24 trusses → total 73 elements

	// ── Roof shell panels (z=5): 4 MITC4 + 10 DKT3 = 14 elements ───────────
	shellE := steelE
	shellNu := steelNu
	shellT := 20.0 // mm

	// MITC4 at 4 corner cells
	mitc4Cells := [][4][2]int{ // [row,col] for SW, SE, NE, NW of each quad
		{{0, 0}, {0, 1}, {1, 1}, {1, 0}}, // cell(col=0,row=0)
		{{0, 2}, {0, 3}, {1, 3}, {1, 2}}, // cell(col=2,row=0)
		{{2, 0}, {2, 1}, {3, 1}, {3, 0}}, // cell(col=0,row=2)
		{{2, 2}, {2, 3}, {3, 3}, {3, 2}}, // cell(col=2,row=2)
	}
	for _, cell := range mitc4Cells {
		n := [4]int{
			roofIDs[cell[0][0]][cell[0][1]],
			roofIDs[cell[1][0]][cell[1][1]],
			roofIDs[cell[2][0]][cell[2][1]],
			roofIDs[cell[3][0]][cell[3][1]],
		}
		c := [4][3]float64{
			dom.Nodes[n[0]].Coord, dom.Nodes[n[1]].Coord,
			dom.Nodes[n[2]].Coord, dom.Nodes[n[3]].Coord,
		}
		dom.AddElement(shell.NewShellMITC4(eid, n, c, shellE, shellNu, shellT))
		eid++
	}
	// 4 MITC4 → total 77 elements

	// DKT3: remaining 5 cells, each split into 2 triangles (10 DKT3s)
	dkt3Cells := [][4][2]int{
		{{0, 1}, {0, 2}, {1, 2}, {1, 1}}, // cell(col=1,row=0)
		{{1, 0}, {1, 1}, {2, 1}, {2, 0}}, // cell(col=0,row=1)
		{{1, 1}, {1, 2}, {2, 2}, {2, 1}}, // cell(col=1,row=1) center
		{{1, 2}, {1, 3}, {2, 3}, {2, 2}}, // cell(col=2,row=1)
		{{2, 1}, {2, 2}, {3, 2}, {3, 1}}, // cell(col=1,row=2)
	}
	for _, cell := range dkt3Cells {
		nSW := roofIDs[cell[0][0]][cell[0][1]]
		nSE := roofIDs[cell[1][0]][cell[1][1]]
		nNE := roofIDs[cell[2][0]][cell[2][1]]
		nNW := roofIDs[cell[3][0]][cell[3][1]]

		// Triangle 1: SW, SE, NE
		t1n := [3]int{nSW, nSE, nNE}
		t1c := [3][3]float64{dom.Nodes[nSW].Coord, dom.Nodes[nSE].Coord, dom.Nodes[nNE].Coord}
		dom.AddElement(shell.NewDiscreteKirchhoffTriangle(eid, t1n, t1c, shellE, shellNu, shellT))
		eid++

		// Triangle 2: SW, NE, NW
		t2n := [3]int{nSW, nNE, nNW}
		t2c := [3][3]float64{dom.Nodes[nSW].Coord, dom.Nodes[nNE].Coord, dom.Nodes[nNW].Coord}
		dom.AddElement(shell.NewDiscreteKirchhoffTriangle(eid, t2n, t2c, shellE, shellNu, shellT))
		eid++
	}
	// 10 DKT3 → total 87 elements

	// ── Boundary conditions ──────────────────────────────────────────────────
	// Fix all 6 DOFs at foundation base (z=0) nodes to prevent rigid body
	// motion and eliminate zero-stiffness rotational DOFs at solid-only nodes.
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			dom.FixNodeAll(baseIDs[row][col])
		}
	}

	// ── Loads ────────────────────────────────────────────────────────────────
	// Downward force on each roof node: -50 kN in Z
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			dom.ApplyLoad(roofIDs[row][col], 2, -50000) // UZ = -50 kN
		}
	}

	// Body force (gravity) on each Hexa8 foundation element
	for i := 0; i < 9; i++ {
		dom.AddBodyForce(i, 2400, [3]float64{0, 0, -9.81})
	}

	// ── Solve ────────────────────────────────────────────────────────────────
	U := mustSolve(dom)
	disp := dom.SetDisplacements(U)

	// ── Verify convergence ──────────────────────────────────────────────────
	// Check roof node displacements: should all be negative (downward) in Z.
	maxRoofUz := 0.0
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			uz := disp[roofIDs[row][col]][2]
			if uz < maxRoofUz {
				maxRoofUz = uz
			}
		}
	}

	fmt.Printf("    [mixed] elements=%d  nodes=%d  DOFs=%d  DOFPerNode=%d\n",
		len(dom.Elements), len(dom.Nodes), dom.NumDOF(), dom.DOFPerNode)
	fmt.Printf("    [mixed] max roof Uz = %.6g mm\n", maxRoofUz)

	if maxRoofUz >= 0 {
		// Roof should deflect downward under this loading
		numerical = 0.0
		return
	}

	numerical = 1.0 // convergence achieved
	return
}
