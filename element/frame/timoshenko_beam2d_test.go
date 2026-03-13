package frame

import (
	"math"
	"testing"

	"go-fem/section"
)

// TestTimoshenkoBeam2D_Symmetry verifies that the 6×6 element stiffness matrix
// of a 2D Timoshenko beam is symmetric.
//
// Property: Ke = Keᵀ. The shear-flexibility correction modifies both the
// transverse and rotational terms; symmetry must be preserved across all
// coupling entries.
//
// Parameters: A=5000, Iz=4e6, Asy=2500 (explicit shear area), L=3000,
// E=210000, G=80769.
//
// Expected: |Ke[i,j] - Ke[j,i]| < 1e-6 for all i < j.
//
// Why valuable: a shear factor applied asymmetrically (e.g., to row but not
// column) would break symmetry and cause non-conservative forces in dynamics.
func TestTimoshenkoBeam2D_Symmetry(t *testing.T) {
	E, G := 210000.0, 80769.0
	sec := section.BeamSection2D{A: 5000, Iz: 4e6, Asy: 2500}
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {3000, 0}}
	b := NewTimoshenkoBeam2D(0, nodes, coords, E, G, sec)
	ke := b.GetTangentStiffness()

	for i := 0; i < 6; i++ {
		for j := i + 1; j < 6; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-6 {
				t.Errorf("Ke not symmetric at [%d,%d]: %.6g vs %.6g", i, j, ke.At(i, j), ke.At(j, i))
			}
		}
	}
}

// TestTimoshenkoBeam2D_ReducesToEB verifies that the 2D Timoshenko beam
// converges to the Euler-Bernoulli formulation when shear area is very large.
//
// Property: with Asy → ∞ the shear parameter Φ = 12·E·Iz/(G·Asy·L²) → 0,
// so all stiffness entries of the Timoshenko element must match those of the
// elastic (EB) beam within relative tolerance 1e-6.
//
// Parameters: E=210000, G=80769, A=10000, Iz=8333333, L=1000, Asy=1e12.
//
// Why valuable: if the Φ → 0 limit is not smooth (e.g., due to a wrong sign
// or a missing 1/(1+Φ) factor), the test would fail and reveal the regression.
func TestTimoshenkoBeam2D_ReducesToEB(t *testing.T) {
	E := 210000.0
	A := 10000.0
	Iz := 8333333.0
	L := 1000.0
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}

	// Very large shear area → Φ ≈ 0 → matches EB
	bigAs := 1e12
	secT := section.BeamSection2D{A: A, Iz: Iz, Asy: bigAs}
	secEB := section.BeamSection2D{A: A, Iz: Iz}

	beamT := NewTimoshenkoBeam2D(0, nodes, coords, E, 80769.0, secT)
	beamEB := NewElasticBeam2D(0, nodes, coords, E, secEB)

	keT := beamT.GetTangentStiffness()
	keEB := beamEB.GetTangentStiffness()

	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
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

// TestTimoshenkoBeam2D_ShearMakesBeamSofter verifies that including shear
// deformation reduces the lateral stiffness coefficient K[4,4] relative to
// the Euler-Bernoulli value, and checks the exact Timoshenko formula.
//
// Physical property: shear deformation adds flexibility; for a very deep beam
// (L/h = 2, kappa=5/6) the shear contribution is significant.
//
// Analytical formula:
//
//	K[4,4] = 12·E·Iz / ((1+Φ)·L³)
//	Φ = 12·E·Iz / (G·Asy·L²)
//
// Parameters: b=100, h=200, L=400, E=210000, G=80769, kappa=5/6.
//
// Expected: K_Timoshenko[4,4] < K_EB[4,4], and the exact value matches
// 12EI/((1+Φ)L³) within 1e-10 relative error.
//
// Why valuable: a missing denominator (1+Φ) in the transverse stiffness term
// would produce the same value as EB even for deep beams, silently ignoring
// shear flexibility.
func TestTimoshenkoBeam2D_ShearMakesBeamSofter(t *testing.T) {
	E, G := 210000.0, 80769.0
	b, h := 100.0, 200.0
	A := b * h
	Iz := b * h * h * h / 12.0
	L := 400.0 // very deep beam (L/h = 2)
	kappa := 5.0 / 6.0
	As := kappa * A

	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	secT := section.BeamSection2D{A: A, Iz: Iz, Asy: As}
	secEB := section.BeamSection2D{A: A, Iz: Iz}

	beamT := NewTimoshenkoBeam2D(0, nodes, coords, E, G, secT)
	beamEB := NewElasticBeam2D(0, nodes, coords, E, secEB)

	kTip := beamT.GetTangentStiffness().At(4, 4)
	kEB := beamEB.GetTangentStiffness().At(4, 4)

	if kTip >= kEB {
		t.Errorf("Timoshenko K[4,4]=%.6g should be < EB K[4,4]=%.6g", kTip, kEB)
	}

	// Verify exact value: 12EI/((1+Φ)L³)
	Phi := 12 * E * Iz / (G * As * L * L)
	kExpected := 12 * E * Iz / ((1 + Phi) * L * L * L)
	if rel := math.Abs(kTip-kExpected) / kExpected; rel > 1e-10 {
		t.Errorf("Ke[4,4]=%.6g, expected %.6g (rel=%.3e)", kTip, kExpected, rel)
	}
}

// TestTimoshenkoBeam2D_DefaultShearArea verifies that when BeamSection2D.Asy
// is zero, the implementation substitutes the default shear area 5/6·A, and
// that the resulting stiffness matrix is identical to the one produced by
// explicitly providing Asy = 5/6·A.
//
// Parameters: A=10000, Iz=8333333, L=2000, E=210000, G=80769.
//
// Expected: |Ke_default[i,j] - Ke_explicit[i,j]| < 1e-8 for all entries.
//
// Why valuable: if the default branch were absent or used a different factor
// (e.g., κ=1), all user models omitting Asy would silently receive wrong results.
func TestTimoshenkoBeam2D_DefaultShearArea(t *testing.T) {
	E, G := 210000.0, 80769.0
	A := 10000.0
	Iz := 8333333.0
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {2000, 0}}

	secDefault := section.BeamSection2D{A: A, Iz: Iz}
	secExplicit := section.BeamSection2D{A: A, Iz: Iz, Asy: 5.0 / 6.0 * A}

	bDef := NewTimoshenkoBeam2D(0, nodes, coords, E, G, secDefault)
	bExp := NewTimoshenkoBeam2D(0, nodes, coords, E, G, secExplicit)

	keDef := bDef.GetTangentStiffness()
	keExp := bExp.GetTangentStiffness()
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			if math.Abs(keDef.At(i, j)-keExp.At(i, j)) > 1e-8 {
				t.Errorf("default vs explicit Ke[%d,%d]: %.8g vs %.8g", i, j, keDef.At(i, j), keExp.At(i, j))
			}
		}
	}
}

// TestTimoshenkoBeam2D_MatchesBeam3D verifies that the in-plane stiffness sub-block
// of a 3D Timoshenko beam matches the 2D Timoshenko beam stiffness entry-by-entry.
//
// Property: the 2D formulation is a specialisation of the 3D one for in-plane
// motion. The DOF mapping is:
//
//	2D index: 0   1   2   3   4   5
//	3D index: 0   1   5   6   7  11
//	DOF:     UX₀ UY₀ RZ₀ UX₁ UY₁ RZ₁
//
// Parameters: A=5000, Iz=4e6, Asy=2500 in 2D; matching Asz=2500 and
// Iy=4e6 in 3D to decouple the two bending planes. L=3000.
//
// Why valuable: a discrepancy would indicate that the 2D and 3D shear
// correction factors are not derived from the same formula, which would
// cause inconsistent results between 2D and 3D models of the same structure.
func TestTimoshenkoBeam2D_MatchesBeam3D(t *testing.T) {
	E, G := 210000.0, 80769.0
	A := 5000.0
	Iz := 4e6
	Asy := 2500.0
	nodes := [2]int{0, 1}
	c2 := [2][2]float64{{0, 0}, {3000, 0}}
	c3 := [2][3]float64{{0, 0, 0}, {3000, 0, 0}}
	sec2 := section.BeamSection2D{A: A, Iz: Iz, Asy: Asy}
	sec3 := section.BeamSection3D{A: A, Iy: 4e6, Iz: Iz, J: 6e6, Asy: Asy, Asz: 2500}

	b2 := NewTimoshenkoBeam2D(0, nodes, c2, E, G, sec2)
	b3 := NewTimoshenkoBeam3D(0, nodes, c3, E, G, sec3, [3]float64{0, 0, 1})

	ke2 := b2.GetTangentStiffness()
	ke3 := b3.GetTangentStiffness()

	// 2D→3D mapping: 0→0, 1→1, 2→5, 3→6, 4→7, 5→11
	map2to3 := [6]int{0, 1, 5, 6, 7, 11}
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			v2 := ke2.At(i, j)
			v3 := ke3.At(map2to3[i], map2to3[j])
			if math.Abs(v2) > 1e-10 {
				rel := math.Abs(v2-v3) / math.Abs(v2)
				if rel > 1e-6 {
					t.Errorf("K2D[%d,%d]=%.6g != K3D[%d,%d]=%.6g (rel=%.3e)",
						i, j, v2, map2to3[i], map2to3[j], v3, rel)
				}
			} else if math.Abs(v3) > 1e-6 {
				t.Errorf("K2D[%d,%d]=%.6g but K3D[%d,%d]=%.6g",
					i, j, v2, map2to3[i], map2to3[j], v3)
			}
		}
	}
}
