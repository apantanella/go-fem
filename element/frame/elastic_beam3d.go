// Package frame implements beam/column frame elements.
package frame

import (
	"math"

	"go-fem/dof"
	"go-fem/section"

	"gonum.org/v1/gonum/mat"
)

// ElasticBeam3D is a 2-node 3D Euler-Bernoulli beam element.
// 6 DOFs per node (UX, UY, UZ, RX, RY, RZ), 12 DOFs total.
type ElasticBeam3D struct {
	ID     int
	Nds    [2]int
	Coords [2][3]float64
	E      float64 // Young's modulus
	G      float64 // Shear modulus
	Sec    section.BeamSection3D

	// VecXZ is a vector in the local x-z plane (defines element orientation).
	// If zero, it is auto-computed.
	VecXZ [3]float64

	ke     *mat.Dense
	length float64
	R      [3][3]float64 // rotation matrix: rows are local axes in global coords
}

// NewElasticBeam3D creates a 3D elastic beam element.
func NewElasticBeam3D(id int, nodes [2]int, coords [2][3]float64,
	e, g float64, sec section.BeamSection3D, vecXZ [3]float64) *ElasticBeam3D {

	b := &ElasticBeam3D{
		ID:     id,
		Nds:    nodes,
		Coords: coords,
		E:      e,
		G:      g,
		Sec:    sec,
		VecXZ:  vecXZ,
	}
	b.computeGeometry()
	b.formKe()
	return b
}

func (b *ElasticBeam3D) computeGeometry() {
	dx := b.Coords[1][0] - b.Coords[0][0]
	dy := b.Coords[1][1] - b.Coords[0][1]
	dz := b.Coords[1][2] - b.Coords[0][2]
	b.length = math.Sqrt(dx*dx + dy*dy + dz*dz)

	// Local x-axis: beam direction
	xAxis := [3]float64{dx / b.length, dy / b.length, dz / b.length}

	// Auto-compute VecXZ if zero
	vxz := b.VecXZ
	if vxz[0] == 0 && vxz[1] == 0 && vxz[2] == 0 {
		// Use global Z if beam is not near-vertical, otherwise global X
		if math.Abs(xAxis[2]) > 0.9 {
			vxz = [3]float64{1, 0, 0}
		} else {
			vxz = [3]float64{0, 0, 1}
		}
	}

	// local y = cross(xAxis, vecXZ), normalised
	yAxis := cross(xAxis, vxz)
	ny := norm(yAxis)
	yAxis[0] /= ny
	yAxis[1] /= ny
	yAxis[2] /= ny

	// local z = cross(xAxis, yAxis)
	zAxis := cross(xAxis, yAxis)

	b.R = [3][3]float64{xAxis, yAxis, zAxis}
}

// formKe builds the 12×12 global stiffness: Ke = Tᵀ·Klocal·T
func (b *ElasticBeam3D) formKe() {
	L := b.length
	E := b.E
	G := b.G
	A := b.Sec.A
	Iy := b.Sec.Iy
	Iz := b.Sec.Iz
	J := b.Sec.J

	L2 := L * L
	L3 := L2 * L

	// Local stiffness matrix (12×12) — Euler-Bernoulli beam
	// DOF order per node: [u, v, w, θx, θy, θz] (local coords)
	kl := mat.NewDense(12, 12, nil)

	// Axial
	ea := E * A / L
	kl.Set(0, 0, ea)
	kl.Set(0, 6, -ea)
	kl.Set(6, 0, -ea)
	kl.Set(6, 6, ea)

	// Torsion
	gj := G * J / L
	kl.Set(3, 3, gj)
	kl.Set(3, 9, -gj)
	kl.Set(9, 3, -gj)
	kl.Set(9, 9, gj)

	// Bending in x-y plane (stiffness uses Iz)
	v1 := 12 * E * Iz / L3
	v2 := 6 * E * Iz / L2
	v3 := 4 * E * Iz / L
	v4 := 2 * E * Iz / L
	kl.Set(1, 1, v1)
	kl.Set(1, 5, v2)
	kl.Set(1, 7, -v1)
	kl.Set(1, 11, v2)
	kl.Set(5, 1, v2)
	kl.Set(5, 5, v3)
	kl.Set(5, 7, -v2)
	kl.Set(5, 11, v4)
	kl.Set(7, 1, -v1)
	kl.Set(7, 5, -v2)
	kl.Set(7, 7, v1)
	kl.Set(7, 11, -v2)
	kl.Set(11, 1, v2)
	kl.Set(11, 5, v4)
	kl.Set(11, 7, -v2)
	kl.Set(11, 11, v3)

	// Bending in x-z plane (stiffness uses Iy)
	w1 := 12 * E * Iy / L3
	w2 := 6 * E * Iy / L2
	w3 := 4 * E * Iy / L
	w4 := 2 * E * Iy / L
	kl.Set(2, 2, w1)
	kl.Set(2, 4, -w2)
	kl.Set(2, 8, -w1)
	kl.Set(2, 10, -w2)
	kl.Set(4, 2, -w2)
	kl.Set(4, 4, w3)
	kl.Set(4, 8, w2)
	kl.Set(4, 10, w4)
	kl.Set(8, 2, -w1)
	kl.Set(8, 4, w2)
	kl.Set(8, 8, w1)
	kl.Set(8, 10, w2)
	kl.Set(10, 2, -w2)
	kl.Set(10, 4, w4)
	kl.Set(10, 8, w2)
	kl.Set(10, 10, w3)

	// Transformation T (12×12 block diagonal: [R, R, R, R])
	T := mat.NewDense(12, 12, nil)
	for block := 0; block < 4; block++ {
		off := block * 3
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				T.Set(off+i, off+j, b.R[i][j])
			}
		}
	}

	// Ke = Tᵀ · Klocal · T
	tmp := mat.NewDense(12, 12, nil)
	tmp.Mul(kl, T)
	b.ke = mat.NewDense(12, 12, nil)
	b.ke.Mul(T.T(), tmp)
}

// ---------- Element interface ----------

func (b *ElasticBeam3D) GetTangentStiffness() *mat.Dense  { return b.ke }
func (b *ElasticBeam3D) GetResistingForce() *mat.VecDense { return mat.NewVecDense(12, nil) }
func (b *ElasticBeam3D) NodeIDs() []int                   { return b.Nds[:] }
func (b *ElasticBeam3D) NumDOF() int                      { return 12 }
func (b *ElasticBeam3D) DOFPerNode() int                  { return 6 }
func (b *ElasticBeam3D) DOFTypes() []dof.Type             { return dof.Full6D(2) }
func (b *ElasticBeam3D) Update(_ []float64) error         { return nil }
func (b *ElasticBeam3D) CommitState() error               { return nil }
func (b *ElasticBeam3D) RevertToStart() error             { return nil }

// Length returns the beam length.
func (b *ElasticBeam3D) Length() float64 { return b.length }

// ---------- Helpers ----------

func cross(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

func norm(v [3]float64) float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}
