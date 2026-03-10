// Package truss implements 1D bar/truss elements.
package truss

import (
	"math"

	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// Truss3D is a 2-node 3D truss (bar) element with axial stiffness only.
// 3 translational DOFs per node, 6 DOFs total.
type Truss3D struct {
	ID     int
	Nds    [2]int
	Coords [2][3]float64
	E      float64 // Young's modulus
	A      float64 // Cross-sectional area

	ke     *mat.Dense
	length float64
	cos    [3]float64 // direction cosines
}

// NewTruss3D creates a 3D truss element.
func NewTruss3D(id int, nodes [2]int, coords [2][3]float64, e, a float64) *Truss3D {
	t := &Truss3D{ID: id, Nds: nodes, Coords: coords, E: e, A: a}
	t.computeGeometry()
	t.formKe()
	return t
}

func (t *Truss3D) computeGeometry() {
	dx := t.Coords[1][0] - t.Coords[0][0]
	dy := t.Coords[1][1] - t.Coords[0][1]
	dz := t.Coords[1][2] - t.Coords[0][2]
	t.length = math.Sqrt(dx*dx + dy*dy + dz*dz)
	t.cos[0] = dx / t.length
	t.cos[1] = dy / t.length
	t.cos[2] = dz / t.length
}

// formKe builds the 6×6 global stiffness matrix:
//
//	Ke = (AE/L) · Tᵀ·[1,-1;-1,1]·T
//
// where T maps global DOFs to the axial direction using direction cosines.
func (t *Truss3D) formKe() {
	k := t.A * t.E / t.length
	c := t.cos

	t.ke = mat.NewDense(6, 6, nil)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			v := k * c[i] * c[j]
			t.ke.Set(i, j, v)
			t.ke.Set(i+3, j+3, v)
			t.ke.Set(i, j+3, -v)
			t.ke.Set(i+3, j, -v)
		}
	}
}

// ---------- Element interface ----------

func (t *Truss3D) GetTangentStiffness() *mat.Dense  { return t.ke }
func (t *Truss3D) GetResistingForce() *mat.VecDense  { return mat.NewVecDense(6, nil) }
func (t *Truss3D) NodeIDs() []int                    { return t.Nds[:] }
func (t *Truss3D) NumDOF() int                       { return 6 }
func (t *Truss3D) DOFPerNode() int                   { return 3 }
func (t *Truss3D) DOFTypes() []dof.Type              { return dof.Translational3D(2) }
func (t *Truss3D) Update(_ []float64) error          { return nil }
func (t *Truss3D) CommitState() error                { return nil }
func (t *Truss3D) RevertToStart() error              { return nil }

// Length returns the element length.
func (t *Truss3D) Length() float64 { return t.length }
