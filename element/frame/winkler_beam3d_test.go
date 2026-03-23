package frame

import (
	"math"
	"testing"

	"go-fem/section"
)

// TestWinklerBeam3D_ZeroKs verifies that with Ksy=Ksz=0, WinklerBeam3D produces
// a stiffness matrix identical to ElasticBeam3D.
func TestWinklerBeam3D_ZeroKs(t *testing.T) {
	coords := [2][3]float64{{0, 0, 0}, {3000, 0, 0}}
	sec := section.BeamSection3D{A: 150e2, Iy: 50000e4, Iz: 80000e4, J: 10000e4}
	E, G := 210000.0, 80769.0

	elastic := NewElasticBeam3D(1, [2]int{0, 1}, coords, E, G, sec, [3]float64{})
	winkler, err := NewWinklerBeam3D(1, [2]int{0, 1}, coords, E, G, sec, [3]float64{}, 0, 0, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ke := elastic.GetTangentStiffness()
	kw := winkler.GetTangentStiffness()
	for i := 0; i < 12; i++ {
		for j := 0; j < 12; j++ {
			diff := math.Abs(ke.At(i, j) - kw.At(i, j))
			tol := 1e-8 * math.Max(1, math.Abs(ke.At(i, j)))
			if diff > tol {
				t.Errorf("Ks=0: Ke[%d,%d] elastic=%g winkler=%g", i, j, ke.At(i, j), kw.At(i, j))
			}
		}
	}
}

// TestWinklerBeam3D_Symmetry verifies symmetry of the global stiffness matrix.
func TestWinklerBeam3D_Symmetry(t *testing.T) {
	// Inclined beam with orthogonal springs
	coords := [2][3]float64{{0, 0, 0}, {2000, 500, 300}}
	sec := section.BeamSection3D{A: 100e2, Iy: 30000e4, Iz: 60000e4, J: 5000e4}
	wb, _ := NewWinklerBeam3D(1, [2]int{0, 1}, coords, 210000, 80769, sec,
		[3]float64{0, 0, 1}, 40, 20, 250)
	ke := wb.GetTangentStiffness()
	for i := 0; i < 12; i++ {
		for j := 0; j < 12; j++ {
			diff := math.Abs(ke.At(i, j) - ke.At(j, i))
			if diff > 1e-9*math.Max(1, math.Abs(ke.At(i, j))) {
				t.Errorf("Ke not symmetric [%d,%d]=%g [%d,%d]=%g",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

// TestWinklerBeam3D_IndependentYZ verifies Y and Z spring contributions are independent.
// Applying only Ksy should change Y-DOFs but not Z-DOFs (beyond beam coupling),
// and vice versa for Ksz.
func TestWinklerBeam3D_IndependentYZ(t *testing.T) {
	coords := [2][3]float64{{0, 0, 0}, {2000, 0, 0}}
	sec := section.BeamSection3D{A: 100e2, Iy: 30000e4, Iz: 60000e4, J: 5000e4}
	E, G := 210000.0, 80769.0

	base, _ := NewWinklerBeam3D(1, [2]int{0, 1}, coords, E, G, sec, [3]float64{}, 0, 0, 1)
	wy, _ := NewWinklerBeam3D(1, [2]int{0, 1}, coords, E, G, sec, [3]float64{}, 30, 0, 200)
	wz, _ := NewWinklerBeam3D(1, [2]int{0, 1}, coords, E, G, sec, [3]float64{}, 0, 30, 200)

	kb := base.GetTangentStiffness()
	ky := wy.GetTangentStiffness()
	kz := wz.GetTangentStiffness()

	// For axis-aligned beam (along global X):
	// Y-spring affects DOFs 1,5,7,11 (v_i=1, θz_i=5, v_j=7, θz_j=11)
	// Z-spring affects DOFs 2,4,8,10 (w_i=2, θy_i=4, w_j=8, θy_j=10)
	yDOFs := [4]int{1, 5, 7, 11}
	zDOFs := [4]int{2, 4, 8, 10}
	alpha := 30.0 * 200.0 * 2000.0 / 420.0

	// Check Y-spring diagonal contribution at v_i (DOF 1)
	delta := ky.At(1, 1) - kb.At(1, 1)
	expected := 156 * alpha
	if math.Abs(delta-expected)/expected > 1e-8 {
		t.Errorf("Y-spring K[1,1]: expected %g, got %g", expected, delta)
	}

	// Ksy should not affect Z-diagonal DOFs
	for _, d := range zDOFs {
		diff := math.Abs(ky.At(d, d) - kb.At(d, d))
		if diff > 1e-8 {
			t.Errorf("Y-spring should not affect Z-DOF [%d,%d], diff=%g", d, d, diff)
		}
	}

	// Z-spring diagonal contribution at w_i (DOF 2)
	delta = kz.At(2, 2) - kb.At(2, 2)
	expected = 156 * alpha
	if math.Abs(delta-expected)/expected > 1e-8 {
		t.Errorf("Z-spring K[2,2]: expected %g, got %g", expected, delta)
	}

	// Ksz should not affect Y-diagonal DOFs
	for _, d := range yDOFs {
		diff := math.Abs(kz.At(d, d) - kb.At(d, d))
		if diff > 1e-8 {
			t.Errorf("Z-spring should not affect Y-DOF [%d,%d], diff=%g", d, d, diff)
		}
	}
}

// TestWinklerBeam3D_WinklerZSignConvention verifies that the Z-direction Winkler
// off-diagonal terms [w_i, θy_i] are negative (sign convention dw/dx = -θy).
func TestWinklerBeam3D_WinklerZSignConvention(t *testing.T) {
	L := 1500.0
	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	sec := section.BeamSection3D{A: 100, Iy: 1000, Iz: 2000, J: 500}
	ks, b := 0.01, 100.0
	alpha := ks * b * L / 420.0

	base, _ := NewWinklerBeam3D(1, [2]int{0, 1}, coords, 210000, 80769, sec, [3]float64{}, 0, 0, 1)
	wz, _ := NewWinklerBeam3D(1, [2]int{0, 1}, coords, 210000, 80769, sec, [3]float64{}, 0, ks, b)

	kb := base.GetTangentStiffness()
	kz := wz.GetTangentStiffness()

	// w_i=2, θy_i=4: off-diagonal should be negative
	delta_24 := kz.At(2, 4) - kb.At(2, 4)
	expected_24 := -22 * L * alpha
	if math.Abs(delta_24-expected_24)/math.Abs(expected_24) > 1e-8 {
		t.Errorf("Z-spring K[2,4] delta: expected %g, got %g", expected_24, delta_24)
	}

	// w_i=2, θy_j=10: off-diagonal should be positive
	delta_210 := kz.At(2, 10) - kb.At(2, 10)
	expected_210 := 13 * L * alpha
	if math.Abs(delta_210-expected_210)/math.Abs(expected_210) > 1e-8 {
		t.Errorf("Z-spring K[2,10] delta: expected %g, got %g", expected_210, delta_210)
	}
}

// TestWinklerBeam3D_EndForcesOnlyBeam verifies EndForces uses beam-only stiffness.
// A pure axial displacement should produce the same axial force regardless of Ks.
func TestWinklerBeam3D_EndForcesOnlyBeam(t *testing.T) {
	L := 2000.0
	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	sec := section.BeamSection3D{A: 100e2, Iy: 20000e4, Iz: 40000e4, J: 5000e4}
	E, G := 210000.0, 80769.0

	elastic := NewElasticBeam3D(1, [2]int{0, 1}, coords, E, G, sec, [3]float64{})
	winkler, _ := NewWinklerBeam3D(1, [2]int{0, 1}, coords, E, G, sec, [3]float64{}, 50, 50, 200)

	// Apply unit axial displacement at node j
	uAxial := make([]float64, 12)
	uAxial[6] = 1.0

	elastic.Update(uAxial)
	winkler.Update(uAxial)

	efE := elastic.EndForces()
	efW := winkler.EndForces()

	// Axial force must match (EndForces uses beam-only kl)
	if math.Abs(efE.I[0]-efW.I[0]) > 1e-8 {
		t.Errorf("EndForces axial mismatch: elastic=%g winkler=%g", efE.I[0], efW.I[0])
	}
}

// TestWinklerBeam3D_InvalidInput verifies constructor errors.
func TestWinklerBeam3D_InvalidInput(t *testing.T) {
	coords := [2][3]float64{{0, 0, 0}, {1000, 0, 0}}
	sec := section.BeamSection3D{A: 100, Iy: 1000, Iz: 2000, J: 500}

	if _, err := NewWinklerBeam3D(1, [2]int{0, 1}, coords, 0, 80769, sec, [3]float64{}, 0, 0, 1); err == nil {
		t.Error("expected error for E=0")
	}
	if _, err := NewWinklerBeam3D(1, [2]int{0, 1}, coords, 210000, 80769, sec, [3]float64{}, -1, 0, 1); err == nil {
		t.Error("expected error for negative Ksy")
	}
	if _, err := NewWinklerBeam3D(1, [2]int{0, 1}, coords, 210000, 80769, sec, [3]float64{}, 0, -1, 1); err == nil {
		t.Error("expected error for negative Ksz")
	}
}
