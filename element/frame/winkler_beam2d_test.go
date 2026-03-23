package frame

import (
	"math"
	"testing"

	"go-fem/section"
)

// analyticalWinklerMidspan computes the midspan deflection of a simply-supported
// beam on Winkler foundation under a central point load P using the infinite-series
// approximation:
//   v_mid ≈ P·L³/(48·EI) · 1/(1 + ks·b·L⁴/(48·π⁴·EI)) — approximate, for very stiff beams
//
// For the finite-element check we use a known closed-form result for a uniform load:
//   For a single-span simply-supported beam on elastic foundation with UDL q,
//   midspan moment M₀ = (q/8 - ...) — this is complex. Instead we validate
//   by checking Ks=0 reproduces the elastic beam.

// TestWinklerBeam2D_ZeroKs verifies that with Ks=0, WinklerBeam2D gives a
// stiffness matrix identical to ElasticBeam2D (horizontal beam, 45° incline).
func TestWinklerBeam2D_ZeroKs(t *testing.T) {
	for _, angle := range []float64{0, math.Pi / 6, math.Pi / 4} {
		L := 2000.0
		coords := [2][2]float64{
			{0, 0},
			{L * math.Cos(angle), L * math.Sin(angle)},
		}
		sec := section.BeamSection2D{A: 120e2, Iz: 40000e4}
		E := 210000.0

		elastic := NewElasticBeam2D(1, [2]int{0, 1}, coords, E, sec)
		winkler, err := NewWinklerBeam2D(1, [2]int{0, 1}, coords, E, sec, 0, 1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		ke := elastic.GetTangentStiffness()
		kw := winkler.GetTangentStiffness()
		for i := 0; i < 6; i++ {
			for j := 0; j < 6; j++ {
				diff := math.Abs(ke.At(i, j) - kw.At(i, j))
				if diff > 1e-8*math.Max(1, math.Abs(ke.At(i, j))) {
					t.Errorf("Ks=0: Ke[%d,%d] elastic=%g winkler=%g (angle=%.2f)",
						i, j, ke.At(i, j), kw.At(i, j), angle)
				}
			}
		}
	}
}

// TestWinklerBeam2D_Symmetry verifies the stiffness matrix is symmetric.
func TestWinklerBeam2D_Symmetry(t *testing.T) {
	coords := [2][2]float64{{0, 0}, {1500, 0}}
	sec := section.BeamSection2D{A: 80e2, Iz: 25000e4}
	wb, _ := NewWinklerBeam2D(1, [2]int{0, 1}, coords, 210000, sec, 50, 300)
	ke := wb.GetTangentStiffness()
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			if diff > 1e-10*math.Max(1, math.Abs(ke.At(i, j))) {
				t.Errorf("Ke not symmetric: [%d,%d]=%g [%d,%d]=%g", i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

// TestWinklerBeam2D_PositiveDiagonal verifies diagonal entries increase with Ks.
func TestWinklerBeam2D_PositiveDiagonal(t *testing.T) {
	coords := [2][2]float64{{0, 0}, {2000, 0}}
	sec := section.BeamSection2D{A: 100e2, Iz: 40000e4}
	E := 210000.0

	wb0, _ := NewWinklerBeam2D(1, [2]int{0, 1}, coords, E, sec, 0, 1)
	wb1, _ := NewWinklerBeam2D(1, [2]int{0, 1}, coords, E, sec, 30, 200)

	// Transverse DOFs (1, 4) must have larger diagonal with Ks > 0
	k0 := wb0.GetTangentStiffness()
	k1 := wb1.GetTangentStiffness()
	for _, idx := range []int{1, 4} {
		if k1.At(idx, idx) <= k0.At(idx, idx) {
			t.Errorf("Winkler should increase K[%d,%d]: got %g <= %g", idx, idx, k1.At(idx, idx), k0.At(idx, idx))
		}
	}
	// Axial DOF (0, 3) must be unchanged
	for _, idx := range []int{0, 3} {
		d := math.Abs(k1.At(idx, idx) - k0.At(idx, idx))
		if d > 1e-10 {
			t.Errorf("Winkler should not change axial K[%d,%d]", idx, idx)
		}
	}
}

// TestWinklerBeam2D_WinklerMatrix validates the 2×2 sub-matrix at [v_i, v_j] positions.
// For a horizontal beam: kw[1,1] += 156*alpha and kw[1,4] += 54*alpha.
func TestWinklerBeam2D_WinklerMatrix(t *testing.T) {
	L := 1000.0
	coords := [2][2]float64{{0, 0}, {L, 0}}
	sec := section.BeamSection2D{A: 100, Iz: 1000}
	ks, b := 0.01, 100.0
	alpha := ks * b * L / 420.0

	elastic := NewElasticBeam2D(1, [2]int{0, 1}, coords, 210000, sec)
	winkler, _ := NewWinklerBeam2D(1, [2]int{0, 1}, coords, 210000, sec, ks, b)

	ke := elastic.GetTangentStiffness()
	kw := winkler.GetTangentStiffness()

	// For horizontal beam, v_i=DOF1, v_j=DOF4
	tests := []struct {
		i, j  int
		delta float64
	}{
		{1, 1, 156 * alpha},
		{1, 4, 54 * alpha},
		{4, 1, 54 * alpha},
		{4, 4, 156 * alpha},
	}
	for _, tc := range tests {
		got := kw.At(tc.i, tc.j) - ke.At(tc.i, tc.j)
		if math.Abs(got-tc.delta) > 1e-8*math.Max(1, math.Abs(tc.delta)) {
			t.Errorf("[%d,%d]: expected delta=%g got %g", tc.i, tc.j, tc.delta, got)
		}
	}
}

// TestWinklerBeam2D_InvalidInput verifies constructor errors.
func TestWinklerBeam2D_InvalidInput(t *testing.T) {
	coords := [2][2]float64{{0, 0}, {1000, 0}}
	sec := section.BeamSection2D{A: 100, Iz: 1000}

	if _, err := NewWinklerBeam2D(1, [2]int{0, 1}, coords, 0, sec, 0, 1); err == nil {
		t.Error("expected error for E=0")
	}
	if _, err := NewWinklerBeam2D(1, [2]int{0, 1}, coords, 210000, sec, -1, 1); err == nil {
		t.Error("expected error for negative Ks")
	}
}

// TestWinklerBeam2D_DefaultWidth verifies that b≤0 is treated as b=1.
func TestWinklerBeam2D_DefaultWidth(t *testing.T) {
	coords := [2][2]float64{{0, 0}, {1000, 0}}
	sec := section.BeamSection2D{A: 100, Iz: 1000}
	ks := 0.05
	wb0, _ := NewWinklerBeam2D(1, [2]int{0, 1}, coords, 210000, sec, ks, 0) // b=0 → b=1
	wb1, _ := NewWinklerBeam2D(1, [2]int{0, 1}, coords, 210000, sec, ks, 1) // explicit b=1
	ke0 := wb0.GetTangentStiffness()
	ke1 := wb1.GetTangentStiffness()
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			if math.Abs(ke0.At(i, j)-ke1.At(i, j)) > 1e-12 {
				t.Errorf("b=0 should equal b=1 at [%d,%d]", i, j)
			}
		}
	}
}
