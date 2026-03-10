package frame

import (
	"math"
	"testing"

	"go-fem/section"

	"gonum.org/v1/gonum/mat"
)

func TestElasticBeam3D_Symmetry(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {5, 0, 0}}
	sec := section.BeamSection3D{A: 0.01, Iy: 1e-4, Iz: 1e-4, J: 2e-4}
	b := NewElasticBeam3D(0, nodes, coords, 200000, 80000, sec, [3]float64{})

	ke := b.GetTangentStiffness()
	r, c := ke.Dims()
	for i := 0; i < r; i++ {
		for j := i + 1; j < c; j++ {
			if math.Abs(ke.At(i, j)-ke.At(j, i)) > 1e-4 {
				t.Errorf("Ke not symmetric: K[%d,%d]=%v != K[%d,%d]=%v",
					i, j, ke.At(i, j), j, i, ke.At(j, i))
			}
		}
	}
}

func TestElasticBeam3D_AxialStiffness(t *testing.T) {
	// Beam along X-axis: axial stiffness should be EA/L
	L := 4.0
	E := 200000.0
	A := 0.02
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	sec := section.BeamSection3D{A: A, Iy: 1e-4, Iz: 1e-4, J: 2e-4}
	b := NewElasticBeam3D(0, nodes, coords, E, 80000, sec, [3]float64{})

	ke := b.GetTangentStiffness()

	// For beam along X: DOF 0 (UX at node 0) should have K[0,0] = EA/L
	expected := E * A / L
	if got := ke.At(0, 0); math.Abs(got-expected)/expected > 1e-8 {
		t.Errorf("K[0,0] = %v, want %v (EA/L)", got, expected)
	}
}

func TestElasticBeam3D_RigidBodyTranslation(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {3, 4, 0}}
	sec := section.BeamSection3D{A: 0.01, Iy: 1e-4, Iz: 1e-4, J: 2e-4}
	b := NewElasticBeam3D(0, nodes, coords, 200000, 80000, sec, [3]float64{})

	ke := b.GetTangentStiffness()

	// Uniform translation in each direction should produce zero forces
	for dir := 0; dir < 3; dir++ {
		u := mat.NewVecDense(12, nil)
		u.SetVec(dir, 1)   // node 0 translation
		u.SetVec(dir+6, 1) // node 1 translation

		f := mat.NewVecDense(12, nil)
		f.MulVec(ke, u)

		for i := 0; i < 12; i++ {
			if math.Abs(f.AtVec(i)) > 1e-4 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

func TestElasticBeam3D_BendingStiffness(t *testing.T) {
	// Beam along X: 12EI/L³ for shear-like bending stiffness
	L := 5.0
	E := 210000.0
	Iz := 8.33e-6 // for bending in xy plane
	nodes := [2]int{0, 1}
	coords := [2][3]float64{{0, 0, 0}, {L, 0, 0}}
	sec := section.BeamSection3D{A: 0.01, Iy: 8.33e-6, Iz: Iz, J: 1e-5}
	b := NewElasticBeam3D(0, nodes, coords, E, 80000, sec, [3]float64{})

	ke := b.GetTangentStiffness()

	// K[1,1] = 12·E·Iz/L³ (transverse Y stiffness at node 0)
	expected := 12 * E * Iz / (L * L * L)
	if got := ke.At(1, 1); math.Abs(got-expected)/expected > 1e-6 {
		t.Errorf("K[1,1] = %v, want %v (12EIz/L³)", got, expected)
	}
}
