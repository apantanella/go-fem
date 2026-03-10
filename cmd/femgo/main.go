// Cantilever beam example – 3D Hexa8 mesh.
//
// Geometry: beam along X, length L, cross-section b×h (Y×Z).
// Fixed at x = 0, tip load P in -Z direction.
//
// Usage:  go run ./cmd/femgo
package main

import (
	"fmt"
	"math"

	"go-fem/analysis"
	"go-fem/domain"
	"go-fem/element"
	"go-fem/material"
	"go-fem/solver"
)

func main() {
	// ---- Parameters ----
	const (
		L  = 10.0   // beam length (X)
		b  = 1.0    // width (Y)
		h  = 1.0    // height (Z)
		E  = 2e5    // Young's modulus [MPa]
		nu = 0.3    // Poisson's ratio
		P  = -1000.0 // total tip load [N] in Z direction

		nx = 10 // divisions along X
		ny = 2  // divisions along Y
		nz = 2  // divisions along Z
	)

	mat3d := material.NewIsotropicLinear(E, nu)
	dom := domain.NewDomain()

	// ---- Generate nodes ----
	nxp1, nyp1, nzp1 := nx+1, ny+1, nz+1
	nodeIdx := func(ix, iy, iz int) int {
		return iz*nyp1*nxp1 + iy*nxp1 + ix
	}

	for iz := 0; iz < nzp1; iz++ {
		for iy := 0; iy < nyp1; iy++ {
			for ix := 0; ix < nxp1; ix++ {
				x := float64(ix) * L / float64(nx)
				y := float64(iy) * b / float64(ny)
				z := float64(iz) * h / float64(nz)
				dom.AddNode(x, y, z)
			}
		}
	}

	// ---- Generate Hexa8 elements ----
	elemID := 0
	for iz := 0; iz < nz; iz++ {
		for iy := 0; iy < ny; iy++ {
			for ix := 0; ix < nx; ix++ {
				n := [8]int{
					nodeIdx(ix, iy, iz),
					nodeIdx(ix+1, iy, iz),
					nodeIdx(ix+1, iy+1, iz),
					nodeIdx(ix, iy+1, iz),
					nodeIdx(ix, iy, iz+1),
					nodeIdx(ix+1, iy, iz+1),
					nodeIdx(ix+1, iy+1, iz+1),
					nodeIdx(ix, iy+1, iz+1),
				}
				var coords [8][3]float64
				for i, nid := range n {
					coords[i] = dom.Nodes[nid].Coord
				}
				dom.AddElement(element.NewHexa8(elemID, n, coords, mat3d))
				elemID++
			}
		}
	}

	// ---- Boundary conditions: fix all DOFs at x = 0 ----
	for iz := 0; iz < nzp1; iz++ {
		for iy := 0; iy < nyp1; iy++ {
			dom.FixNode(nodeIdx(0, iy, iz))
		}
	}

	// ---- Tip load: distribute P among nodes at x = L ----
	tipNodes := 0
	for iz := 0; iz < nzp1; iz++ {
		for iy := 0; iy < nyp1; iy++ {
			tipNodes++
		}
	}
	pPerNode := P / float64(tipNodes)
	for iz := 0; iz < nzp1; iz++ {
		for iy := 0; iy < nyp1; iy++ {
			dom.ApplyLoad(nodeIdx(nx, iy, iz), 2, pPerNode) // Z direction
		}
	}

	// ---- Solve ----
	ana := analysis.StaticLinearAnalysis{
		Dom:    dom,
		Solver: solver.Cholesky{},
	}

	fmt.Println("=== go-fem: 3D Cantilever Beam (Hexa8) ===")
	fmt.Printf("Mesh: %d×%d×%d = %d elements, %d nodes, %d DOFs\n",
		nx, ny, nz, len(dom.Elements), len(dom.Nodes), dom.NumDOF())

	U, err := ana.Run()
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
		return
	}

	disp := dom.SetDisplacements(U)

	// ---- Find max tip deflection in Z ----
	var maxUz float64
	var maxNode int
	for iz := 0; iz < nzp1; iz++ {
		for iy := 0; iy < nyp1; iy++ {
			nid := nodeIdx(nx, iy, iz)
			uz := disp[nid][2]
			if math.Abs(uz) > math.Abs(maxUz) {
				maxUz = uz
				maxNode = nid
			}
		}
	}

	// ---- Analytical reference (Euler-Bernoulli) ----
	I := b * h * h * h / 12.0
	deltaEB := math.Abs(P) * L * L * L / (3 * E * I)

	fmt.Printf("\nResults:\n")
	fmt.Printf("  Max tip Uz  = %.6f  (node %d)\n", maxUz, maxNode)
	fmt.Printf("  Beam theory = %.6f  (Euler-Bernoulli PL³/3EI)\n", -deltaEB)
	fmt.Printf("  Ratio FEM/EB = %.4f\n", math.Abs(maxUz)/deltaEB)

	// ---- Print displacement field at tip ----
	fmt.Printf("\nTip node displacements (x = %.1f):\n", L)
	fmt.Printf("  %-6s %12s %12s %12s\n", "Node", "Ux", "Uy", "Uz")
	for iz := 0; iz < nzp1; iz++ {
		for iy := 0; iy < nyp1; iy++ {
			nid := nodeIdx(nx, iy, iz)
			d := disp[nid]
			fmt.Printf("  %-6d %12.6f %12.6f %12.6f\n", nid, d[0], d[1], d[2])
		}
	}
}
