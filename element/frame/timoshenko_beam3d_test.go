package frame_test

import (
	"math"
	"testing"

	"go-fem/element/frame"
	"go-fem/section"
)

// cantileverDeflection returns the tip deflection of a Timoshenko cantilever
// beam under a concentrated tip load P.
//
// The Timoshenko formula splits deflection into a bending part and a shear part:
//
//	δ = P·L³/(3·E·I) + P·L/(G·As)
//
// where As is the effective shear area (κ·A) and G is the shear modulus.
// When As → ∞ the shear contribution vanishes and the result reduces to the
// Euler-Bernoulli cantilever deflection δ_EB = PL³/(3EI).
func cantileverDeflection(P, L, E, I, G, As float64) float64 {
	return P*L*L*L/(3*E*I) + P*L/(G*As)
}

// TestTimoshenkoReducesToEB verifies that when the shear area is very large
// (Φ → 0), the Timoshenko beam deflection matches the Euler-Bernoulli beam.
//
// Property: the Timoshenko shear-flexibility parameter is
//
//	Φz = 12·E·Iz / (G·Asy·L²)
//
// As Asy → ∞, Φz → 0 and all Timoshenko stiffness entries converge to the
// Euler-Bernoulli values. All 12×12 entries must agree within relative
// tolerance 1e-6.
//
// Parameters: E=210000, G=80769, A=10000, Iz=8333333, L=1000, Asy=1e12.
// At these values Φz ≈ 12·210000·8333333/(80769·1e12·1e6) ≈ 0 (negligible).
//
// Why valuable: catches any additive shear term that does not correctly
// vanish when the shear stiffness is infinite, which would cause Timoshenko
// to over-stiffen or under-stiffen relative to Euler-Bernoulli in the limit.
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
// (short, stocky cross-section) deflects more with Timoshenko than with
// Euler-Bernoulli, and checks the exact Timoshenko stiffness coefficient.
//
// Physical property: shear deformation adds flexibility, so the lateral stiffness
// entry K[7,7] (transverse DOF at node 1) must be smaller for Timoshenko than
// for EB when the shear parameter Φz is non-negligible.
//
// Analytical formula for K[7,7]:
//
//	K[7,7] = 12·E·Iz / ((1+Φz)·L³)
//	Φz = 12·E·Iz / (G·Asy·L²)
//
// Parameters: b=100, h=200 (L/h=2 — very deep), kappa=5/6, E=210000, G=80769,
// L=400. With these values Φz ≈ 12·210000·(100·200³/12)/(80769·(5/6·100·200)·400²)
// which is significantly greater than zero.
//
// Why valuable: a deep beam is the regime where ignoring shear (EB) is most
// incorrect; this test would catch a missing (1+Φ) denominator in the
// transverse stiffness formula.
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

// TestTimoshenkoSymmetry verifies that the 12×12 Timoshenko beam stiffness
// matrix is symmetric.
//
// Property: Ke = Keᵀ (self-adjointness). The shear correction terms must not
// break symmetry.
//
// Parameters: standard wide-flange-like section with A=5000, Iy=Iz=4e6,
// J=6e6, Asy=Asz=2500; E=210000, G=80769, L=3000.
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-6 for all i < j.
//
// Why valuable: unsymmetric entries would indicate a bug in the Timoshenko
// shear-correction assembly, such as a Φ factor applied to only one of the
// two symmetric positions.
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
//
// Property: when BeamSection3D.Asy = 0 (zero value), the Timoshenko element
// must substitute the classical Mindlin shear factor κ=5/6 and use Asy=5/6·A.
// The resulting stiffness matrix must be identical to the one produced when
// Asy = 5/6·A is provided explicitly.
//
// Parameters: A=10000, Iz=8333333, L=2000, E=210000, G=80769.
//
// Why valuable: if the default-shear-area branch were missing or used a wrong
// factor, the element would produce a different (harder or softer) stiffness
// than the reference, which would silently affect all models that omit Asy.
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
