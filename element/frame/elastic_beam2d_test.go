package frame

import (
	"math"
	"testing"

	"go-fem/section"

	"gonum.org/v1/gonum/mat"
)

func TestElasticBeam2D_Symmetry(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {5, 0}}
	sec := section.BeamSection2D{A: 0.01, Iz: 1e-4}
	b := NewElasticBeam2D(0, nodes, coords, 200000, sec)

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

func TestElasticBeam2D_AxialStiffness(t *testing.T) {
	L := 4.0
	E := 200000.0
	A := 0.02
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	sec := section.BeamSection2D{A: A, Iz: 1e-4}
	b := NewElasticBeam2D(0, nodes, coords, E, sec)

	ke := b.GetTangentStiffness()

	expected := E * A / L
	if got := ke.At(0, 0); math.Abs(got-expected)/expected > 1e-8 {
		t.Errorf("K[0,0] = %v, want %v (EA/L)", got, expected)
	}
}

func TestElasticBeam2D_BendingStiffness(t *testing.T) {
	L := 5.0
	E := 210000.0
	Iz := 8.33e-6
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	sec := section.BeamSection2D{A: 0.01, Iz: Iz}
	b := NewElasticBeam2D(0, nodes, coords, E, sec)

	ke := b.GetTangentStiffness()

	// K[1,1] = 12·E·Iz/L³ (transverse stiffness at node 0)
	expected := 12 * E * Iz / (L * L * L)
	if got := ke.At(1, 1); math.Abs(got-expected)/expected > 1e-6 {
		t.Errorf("K[1,1] = %v, want %v (12EIz/L³)", got, expected)
	}

	// K[2,2] = 4·E·Iz/L (rotational stiffness at node 0)
	expected4 := 4 * E * Iz / L
	if got := ke.At(2, 2); math.Abs(got-expected4)/expected4 > 1e-6 {
		t.Errorf("K[2,2] = %v, want %v (4EIz/L)", got, expected4)
	}
}

func TestElasticBeam2D_RigidBodyTranslation(t *testing.T) {
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {3, 4}}
	sec := section.BeamSection2D{A: 0.01, Iz: 1e-4}
	b := NewElasticBeam2D(0, nodes, coords, 200000, sec)

	ke := b.GetTangentStiffness()

	for dir := 0; dir < 2; dir++ {
		u := mat.NewVecDense(6, nil)
		u.SetVec(dir, 1)   // node 0 translation
		u.SetVec(dir+3, 1) // node 1 translation

		f := mat.NewVecDense(6, nil)
		f.MulVec(ke, u)

		for i := 0; i < 6; i++ {
			if math.Abs(f.AtVec(i)) > 1e-4 {
				t.Errorf("rigid body dir %d: f[%d] = %v, want ~0", dir, i, f.AtVec(i))
			}
		}
	}
}

func TestElasticBeam2D_EndForces(t *testing.T) {
	// Horizontal cantilever beam: fixed at node 0, tip load Fy at node 1
	L := 3.0
	E := 210000.0
	A := 0.01
	Iz := 1e-4
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	sec := section.BeamSection2D{A: A, Iz: Iz}
	b := NewElasticBeam2D(0, nodes, coords, E, sec)

	// Small transverse displacement at node 1 only (cantilever)
	delta := 0.001
	// For a cantilever with unit tip load: δ = PL³/(3EI) → P = 3EIδ/L³
	P := 3 * E * Iz * delta / (L * L * L)

	// The full 6x6 system with BCs at node 0 gives:
	// Displacement at dofs 0,1,2 = 0; solve for dofs 3,4,5
	// For horizontal beam (cos=1, sin=0), global = local
	// Tip displacements: u₂=0, v₂=δ, θ₂ = PL²/(2EI)
	theta := P * L * L / (2 * E * Iz)
	disp := []float64{0, 0, 0, 0, delta, theta}
	b.Update(disp)

	ef := b.EndForces()

	// At node i (fixed end): N=0, V=-P, M=-PL
	if math.Abs(ef.I[0]) > 1e-4 {
		t.Errorf("EndForces.I[N] = %v, want ~0", ef.I[0])
	}
	if math.Abs(ef.I[1]+P) > 1e-4*math.Abs(P) {
		t.Errorf("EndForces.I[V] = %v, want %v", ef.I[1], -P)
	}
	if math.Abs(ef.I[2]+P*L) > 1e-4*math.Abs(P*L) {
		t.Errorf("EndForces.I[M] = %v, want %v", ef.I[2], -P*L)
	}

	// At node j (free end): N=0, V=P, M=0
	if math.Abs(ef.J[0]) > 1e-4 {
		t.Errorf("EndForces.J[N] = %v, want ~0", ef.J[0])
	}
	if math.Abs(ef.J[1]-P) > 1e-4*math.Abs(P) {
		t.Errorf("EndForces.J[V] = %v, want %v", ef.J[1], P)
	}
	if math.Abs(ef.J[2]) > 1e-4*math.Abs(P*L) {
		t.Errorf("EndForces.J[M] = %v, want ~0", ef.J[2])
	}
}

func TestElasticBeam2D_MatchesBeam3D(t *testing.T) {
	// For a beam in the XY plane, the 2D beam stiffness should match the
	// corresponding rows/cols of the 3D beam stiffness.
	E := 210000.0
	G := 80000.0
	A := 0.01
	Iz := 1e-4
	nodes := [2]int{0, 1}
	c2 := [2][2]float64{{0, 0}, {5, 0}}
	c3 := [2][3]float64{{0, 0, 0}, {5, 0, 0}}
	sec2 := section.BeamSection2D{A: A, Iz: Iz}
	sec3 := section.BeamSection3D{A: A, Iy: 1e-4, Iz: Iz, J: 2e-4}

	b2 := NewElasticBeam2D(0, nodes, c2, E, sec2)
	b3 := NewElasticBeam3D(0, nodes, c3, E, G, sec3, [3]float64{})

	ke2 := b2.GetTangentStiffness()
	ke3 := b3.GetTangentStiffness()

	// 2D DOFs: [UX₁,UY₁,RZ₁, UX₂,UY₂,RZ₂] = indices 0,1,2,3,4,5
	// 3D DOFs: [UX₁,UY₁,UZ₁,RX₁,RY₁,RZ₁, UX₂,...] = indices 0,1,2,3,4,5,6,7,8,9,10,11
	// Mapping: 2D[0]→3D[0], 2D[1]→3D[1], 2D[2]→3D[5], 2D[3]→3D[6], 2D[4]→3D[7], 2D[5]→3D[11]
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

func TestElasticBeam2D_EquivalentNodalLoad(t *testing.T) {
	L := 4.0
	E := 200000.0
	A := 0.01
	Iz := 1e-4
	nodes := [2]int{0, 1}
	coords := [2][2]float64{{0, 0}, {L, 0}}
	sec := section.BeamSection2D{A: A, Iz: Iz}
	b := NewElasticBeam2D(0, nodes, coords, E, sec)

	q := 10.0 // uniform load in -Y direction
	f := b.EquivalentNodalLoad([3]float64{0, -1, 0}, q)

	// For horizontal beam: local = global
	// Fy_i = Fy_j = q*L/2 = -20 (downward)
	// Mz_i = q*L²/12, Mz_j = -q*L²/12
	tol := 1e-8
	if math.Abs(f.AtVec(0)) > tol {
		t.Errorf("f[0] = %v, want 0", f.AtVec(0))
	}
	if math.Abs(f.AtVec(1)-(-q*L/2)) > tol {
		t.Errorf("f[1] = %v, want %v", f.AtVec(1), -q*L/2)
	}
	if math.Abs(f.AtVec(2)-(-q*L*L/12)) > tol {
		t.Errorf("f[2] = %v, want %v", f.AtVec(2), -q*L*L/12)
	}
	if math.Abs(f.AtVec(3)) > tol {
		t.Errorf("f[3] = %v, want 0", f.AtVec(3))
	}
	if math.Abs(f.AtVec(4)-(-q*L/2)) > tol {
		t.Errorf("f[4] = %v, want %v", f.AtVec(4), -q*L/2)
	}
	if math.Abs(f.AtVec(5)-(q*L*L/12)) > tol {
		t.Errorf("f[5] = %v, want %v", f.AtVec(5), q*L*L/12)
	}
}
