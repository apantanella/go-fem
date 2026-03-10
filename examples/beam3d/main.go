// Cantilever beam convergence study – Hexa8 mesh refinement.
//
// Runs the same cantilever beam with increasing mesh density
// and reports convergence towards the Euler-Bernoulli solution.
//
// Usage:  go run ./examples/beam3d
package main

import (
	"fmt"
	"math"

	"go-fem/analysis"
	"go-fem/domain"
	"go-fem/element/solid"
	"go-fem/material"
	"go-fem/solver"
)

func cantilever(nx, ny, nz int) float64 {
	const (
		L  = 10.0
		b  = 1.0
		h  = 1.0
		E  = 2e5
		nu = 0.3
		P  = -1000.0
	)

	mat3d := material.NewIsotropicLinear(E, nu)
	dom := domain.NewDomain()

	nxp1, nyp1, nzp1 := nx+1, ny+1, nz+1
	idx := func(ix, iy, iz int) int { return iz*nyp1*nxp1 + iy*nxp1 + ix }

	for iz := 0; iz < nzp1; iz++ {
		for iy := 0; iy < nyp1; iy++ {
			for ix := 0; ix < nxp1; ix++ {
				dom.AddNode(
					float64(ix)*L/float64(nx),
					float64(iy)*b/float64(ny),
					float64(iz)*h/float64(nz),
				)
			}
		}
	}

	eid := 0
	for iz := 0; iz < nz; iz++ {
		for iy := 0; iy < ny; iy++ {
			for ix := 0; ix < nx; ix++ {
				n := [8]int{
					idx(ix, iy, iz), idx(ix+1, iy, iz),
					idx(ix+1, iy+1, iz), idx(ix, iy+1, iz),
					idx(ix, iy, iz+1), idx(ix+1, iy, iz+1),
					idx(ix+1, iy+1, iz+1), idx(ix, iy+1, iz+1),
				}
				var c [8][3]float64
				for i, nid := range n {
					c[i] = dom.Nodes[nid].Coord
				}
				dom.AddElement(solid.NewHexa8(eid, n, c, mat3d))
				eid++
			}
		}
	}

	for iz := 0; iz < nzp1; iz++ {
		for iy := 0; iy < nyp1; iy++ {
			dom.FixNode(idx(0, iy, iz))
		}
	}

	tipCount := nyp1 * nzp1
	p := P / float64(tipCount)
	for iz := 0; iz < nzp1; iz++ {
		for iy := 0; iy < nyp1; iy++ {
			dom.ApplyLoad(idx(nx, iy, iz), 2, p)
		}
	}

	ana := analysis.StaticLinearAnalysis{Dom: dom, Solver: solver.Cholesky{}}
	U, err := ana.Run()
	if err != nil {
		panic(err)
	}

	var maxUz float64
	for iz := 0; iz < nzp1; iz++ {
		for iy := 0; iy < nyp1; iy++ {
			nid := idx(nx, iy, iz)
			uz := U.AtVec(3*nid + 2)
			if math.Abs(uz) > math.Abs(maxUz) {
				maxUz = uz
			}
		}
	}
	return maxUz
}

func main() {
	const (
		L = 10.0
		b = 1.0
		h = 1.0
		E = 2e5
		P = 1000.0
	)
	I := b * h * h * h / 12.0
	deltaEB := P * L * L * L / (3 * E * I)

	fmt.Println("=== Cantilever Beam Convergence Study (Hexa8) ===")
	fmt.Printf("Euler-Bernoulli reference: δ = %.6f\n\n", deltaEB)
	fmt.Printf("%-8s %-8s %-8s %10s %10s %10s %10s\n",
		"nx", "ny", "nz", "Elements", "DOFs", "δ_FEM", "δ/δ_EB")
	fmt.Println("----------------------------------------------------------------------")

	configs := [][3]int{
		{4, 1, 1},
		{8, 2, 2},
		{16, 4, 4},
	}

	for _, cfg := range configs {
		nx, ny, nz := cfg[0], cfg[1], cfg[2]
		uz := cantilever(nx, ny, nz)
		nElem := nx * ny * nz
		nDOF := (nx + 1) * (ny + 1) * (nz + 1) * 3
		fmt.Printf("%-8d %-8d %-8d %10d %10d %10.6f %10.4f\n",
			nx, ny, nz, nElem, nDOF, uz, math.Abs(uz)/deltaEB)
	}
}
