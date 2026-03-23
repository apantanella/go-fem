package shell

import (
	"math"
	"testing"
)

// TestWinklerShellMITC4_ZeroKs verifies that with Ks=0, WinklerShellMITC4 gives
// a stiffness matrix identical to ShellMITC4.
func TestWinklerShellMITC4_ZeroKs(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{1000, 0, 0},
		{1000, 800, 0},
		{0, 800, 0},
	}
	E, nu, thick := 30000.0, 0.2, 250.0

	s := NewShellMITC4(1, nodes, coords, E, nu, thick)
	w := NewWinklerShellMITC4(1, nodes, coords, E, nu, thick, 0)

	ks := s.GetTangentStiffness()
	kw := w.GetTangentStiffness()

	for i := 0; i < 24; i++ {
		for j := 0; j < 24; j++ {
			diff := math.Abs(ks.At(i, j) - kw.At(i, j))
			tol := 1e-9 * math.Max(1, math.Abs(ks.At(i, j)))
			if diff > tol {
				t.Errorf("Ks=0: Ke[%d,%d] shell=%g winkler=%g", i, j, ks.At(i, j), kw.At(i, j))
			}
		}
	}
}

// TestWinklerShellMITC4_Symmetry verifies the global stiffness matrix is symmetric.
func TestWinklerShellMITC4_Symmetry(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{1200, 0, 0},
		{1200, 1000, 0},
		{0, 1000, 0},
	}
	w := NewWinklerShellMITC4(1, nodes, coords, 30000, 0.2, 200, 0.05)
	ke := w.GetTangentStiffness()
	for i := 0; i < 24; i++ {
		for j := 0; j < 24; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			if diff > 1e-9*math.Max(1, math.Abs(ke.At(i, j))) {
				t.Errorf("Ke not symmetric [%d,%d]=%g [%d,%d]=%g",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

// TestWinklerShellMITC4_WinklerContribution validates the 4×4 UZ Winkler
// sub-matrix against the analytical formula for a rectangular element:
//
//	K_w = ks · area/36 · [[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]]
//
// UZ global positions in the 24-DOF local frame are: 2, 8, 14, 20.
func TestWinklerShellMITC4_WinklerContribution(t *testing.T) {
	a, b := 1000.0, 800.0 // element dimensions
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0},
		{a, 0, 0},
		{a, b, 0},
		{0, b, 0},
	}
	E, nu, thick := 30000.0, 0.2, 200.0
	ks := 0.05

	ref := NewShellMITC4(1, nodes, coords, E, nu, thick)
	w := NewWinklerShellMITC4(1, nodes, coords, E, nu, thick, ks)

	kref := ref.GetTangentStiffness()
	kw := w.GetTangentStiffness()

	area := a * b
	factor := ks * area / 36.0
	// Analytical K_w for bilinear quad (normalised by factor):
	// [[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]]
	analytical := [4][4]float64{
		{4, 2, 1, 2},
		{2, 4, 2, 1},
		{1, 2, 4, 2},
		{2, 1, 2, 4},
	}
	uzpos := [4]int{2, 8, 14, 20}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			ri, rj := uzpos[i], uzpos[j]
			delta := kw.At(ri, rj) - kref.At(ri, rj)
			expected := factor * analytical[i][j]
			err := math.Abs(delta - expected)
			tol := 1e-6 * math.Max(1, math.Abs(expected))
			if err > tol {
				t.Errorf("Winkler K[%d,%d] UZ[%d,%d]: expected %g got delta %g",
					ri, rj, i, j, expected, delta)
			}
		}
	}
}

// TestWinklerShellMITC4_WinklerOnlyAffectsUZ verifies that Winkler spring does
// not affect membrane or drilling DOFs.
func TestWinklerShellMITC4_WinklerOnlyAffectsUZ(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{
		{0, 0, 0}, {1000, 0, 0}, {1000, 800, 0}, {0, 800, 0},
	}
	ref := NewShellMITC4(1, nodes, coords, 30000, 0.2, 200)
	w := NewWinklerShellMITC4(1, nodes, coords, 30000, 0.2, 200, 0.05)

	kr := ref.GetTangentStiffness()
	kw := w.GetTangentStiffness()

	uzpos := map[int]bool{2: true, 8: true, 14: true, 20: true}
	for i := 0; i < 24; i++ {
		for j := 0; j < 24; j++ {
			diff := kw.At(i, j) - kr.At(i, j)
			if diff == 0 {
				continue
			}
			// Any nonzero delta must involve at least one UZ position
			if !uzpos[i] && !uzpos[j] {
				t.Errorf("Winkler changed non-UZ entry [%d,%d] by %g", i, j, diff)
			}
		}
	}
}

// TestWinklerShellMITC4_LocalForces_ZeroDisp verifies that LocalForces returns
// zeros when displacements are zero.
func TestWinklerShellMITC4_LocalForces_ZeroDisp(t *testing.T) {
	nodes := [4]int{0, 1, 2, 3}
	coords := [4][3]float64{{0, 0, 0}, {1000, 0, 0}, {1000, 800, 0}, {0, 800, 0}}
	w := NewWinklerShellMITC4(1, nodes, coords, 30000, 0.2, 200, 0.05)
	sf := w.LocalForces()
	if sf.Nx != 0 || sf.Ny != 0 || sf.Mx != 0 || sf.My != 0 {
		t.Errorf("expected zero forces for zero displacement: %+v", sf)
	}
}
