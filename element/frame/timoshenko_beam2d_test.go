package frame

import (
	"math"
	"testing"

	"go-fem/section"
)

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
