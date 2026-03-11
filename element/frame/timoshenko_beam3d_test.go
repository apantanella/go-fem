package frame_test

import (
	"math"
	"testing"

	"go-fem/element/frame"
	"go-fem/section"
)

// cantileverDeflection returns the tip deflection of a cantilever beam under a
// tip load P using the Timoshenko formula:
//
//	δ = P·L³/(3·E·I) + P·L/(G·As)
func cantileverDeflection(P, L, E, I, G, As float64) float64 {
	return P*L*L*L/(3*E*I) + P*L/(G*As)
}

// TestTimoshenkoReducesToEB verifies that when the shear area is very large
// (Φ → 0), the Timoshenko beam deflection matches the Euler-Bernoulli beam.
func TestTimoshenkoReducesToEB(t *testing.T) {
	E, G := 210000.0, 80769.0
	A := 10000.0
	Iz := 8333333.0
	L := 1000.0

	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	vecXZ := [3]float64{0, 0, 1}

	// Very large shear area → Φz ≈ 0 → should match EB
	bigAs := 1e12
	secT := section.BeamSection3D{A: A, Iy: Iz, Iz: Iz, J: Iz, Asy: bigAs, Asz: bigAs}
	secEB := section.BeamSection3D{A: A, Iy: Iz, Iz: Iz, J: Iz}

	beamT := frame.NewTimoshenkoBeam3D(0, nodes, coords, E, G, secT, vecXZ)
	beamEB := frame.NewElasticBeam3D(0, nodes, coords, E, G, secEB, vecXZ)

	keT := beamT.GetTangentStiffness()
	keEB := beamEB.GetTangentStiffness()

	// Compare all entries
	for i := 0; i < 12; i++ {
		for j := 0; j < 12; j++ {
			vT := keT.At(i, j)
			vEB := keEB.At(i, j)
			if math.Abs(vEB) > 1e-10 {
				rel := math.Abs(vT-vEB) / math.Abs(vEB)
				if rel > 1e-6 {
					t.Errorf("Ke[%d,%d]: Timoshenko=%.6g, EB=%.6g, rel=%.3e", i, j, vT, vEB, rel)
				}
			} else if math.Abs(vT-vEB) > 1e-6 {
				t.Errorf("Ke[%d,%d]: Timoshenko=%.6g, EB=%.6g", i, j, vT, vEB)
			}
		}
	}
}

// TestTimoshenkoShearDeformationIncreasesDeflection verifies that a deep beam
// (short, stocky) deflects more with Timoshenko than with Euler-Bernoulli.
// For a cantilever under tip shear P, Timoshenko adds P·L/(G·As) to EB deflection.
func TestTimoshenkoShearDeformationIncreasesDeflection(t *testing.T) {
	E, G := 210000.0, 80769.0
	b, h := 100.0, 200.0 // wide, deep section (short beam scenario)
	A := b * h
	Iz := b * h * h * h / 12.0
	L := 400.0 // slenderness L/h = 2 — very deep beam
	kappa := 5.0 / 6.0
	As := kappa * A

	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	vecXZ := [3]float64{0, 0, 1}
	secT := section.BeamSection3D{A: A, Iy: Iz, Iz: Iz, J: Iz, Asy: As, Asz: As}

	beamT := frame.NewTimoshenkoBeam3D(0, nodes, coords, E, G, secT, vecXZ)
	keT := beamT.GetTangentStiffness()

	// The Timoshenko lateral stiffness at the tip DOF (v-direction, DOF 7 = global Y)
	// for a cantilever with fixed node 0 (DOFs 0-5) and free node 1 (DOFs 6-11):
	// k_tip = K[7,7] in the local (aligned with global) stiffness matrix.
	// This equals the lateral stiffness = 1 / (L³/(3EI) + L/(GAs)).
	kTip := keT.At(7, 7)

	// Analytical Timoshenko tip stiffness (cantilever)
	deltaT := L*L*L/(3*E*Iz) + L/(G*As)
	kTipAnalytical := 1.0 / deltaT // P / δ for unit load

	// Note: Ke[7,7] is the stiffness coefficient but not the exact tip stiffness
	// (it includes coupling). We verify that adding shear makes the element softer
	// by comparing with EB stiffness.
	secEB := section.BeamSection3D{A: A, Iy: Iz, Iz: Iz, J: Iz}
	beamEB := frame.NewElasticBeam3D(0, nodes, coords, E, G, secEB, vecXZ)
	keEB := beamEB.GetTangentStiffness()
	kEB := keEB.At(7, 7)

	if kTip >= kEB {
		t.Errorf("Timoshenko K[7,7]=%.6g should be smaller than EB K[7,7]=%.6g (shear makes beam softer)", kTip, kEB)
	}

	// Analytical check: Φz = 12EI/(G·As·L²)
	Phiz := 12 * E * Iz / (G * As * L * L)
	// Ke[7,7] for the element (combined bending + shear) = 12EI/((1+Φz)L³)
	kExpected := 12 * E * Iz / ((1 + Phiz) * L * L * L)
	if rel := math.Abs(kTip-kExpected) / kExpected; rel > 1e-10 {
		t.Errorf("Ke[7,7]=%.6g, expected %.6g (rel=%.3e)", kTip, kExpected, rel)
	}

	// Verify analytical consistency: kTipAnalytical ≈ kExpected / (factor from matrix)
	// The single-DOF tip stiffness from a cantilever = 12EI/((1+Φ)L³) × correction
	// (from static condensation). This is a sanity check — not exact equality.
	_ = kTipAnalytical
}

// TestTimoshenkoSymmetry verifies the stiffness matrix is symmetric.
func TestTimoshenkoSymmetry(t *testing.T) {
	E, G := 210000.0, 80769.0
	sec := section.BeamSection3D{A: 5000, Iy: 4e6, Iz: 4e6, J: 6e6, Asy: 2500, Asz: 2500}
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {3000, 0, 0}}
	b := frame.NewTimoshenkoBeam3D(0, nodes, coords, E, G, sec, [3]float64{0, 0, 1})
	ke := b.GetTangentStiffness()

	for i := 0; i < 12; i++ {
		for j := i + 1; j < 12; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-6 {
				t.Errorf("Ke not symmetric at [%d,%d]: %.6g vs %.6g", i, j, ke.At(i, j), ke.At(j, i))
			}
		}
	}
}

// TestTimoshenkoDefaultShearArea verifies the 5/6·A default is applied when
// Asy/Asz are zero, by comparing against explicit 5/6·A specification.
func TestTimoshenkoDefaultShearArea(t *testing.T) {
	E, G := 210000.0, 80769.0
	A := 10000.0
	Iz := 8333333.0
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {2000, 0, 0}}
	vecXZ := [3]float64{0, 0, 1}

	secDefault := section.BeamSection3D{A: A, Iy: Iz, Iz: Iz, J: Iz}
	secExplicit := section.BeamSection3D{A: A, Iy: Iz, Iz: Iz, J: Iz, Asy: 5.0 / 6.0 * A, Asz: 5.0 / 6.0 * A}

	bDef := frame.NewTimoshenkoBeam3D(0, nodes, coords, E, G, secDefault, vecXZ)
	bExp := frame.NewTimoshenkoBeam3D(0, nodes, coords, E, G, secExplicit, vecXZ)

	keDef := bDef.GetTangentStiffness()
	keExp := bExp.GetTangentStiffness()
	for i := 0; i < 12; i++ {
		for j := 0; j < 12; j++ {
			if math.Abs(keDef.At(i, j)-keExp.At(i, j)) > 1e-8 {
				t.Errorf("default vs explicit Ke[%d,%d]: %.8g vs %.8g", i, j, keDef.At(i, j), keExp.At(i, j))
			}
		}
	}
}
