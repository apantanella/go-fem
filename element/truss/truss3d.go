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
	ue     [6]float64 // element displacements (set by Update)
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

func (t *Truss3D) GetTangentStiffness() *mat.Dense { return t.ke }

// GetResistingForce returns Ke·ue (internal nodal force vector in global coords).
func (t *Truss3D) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(6, nil)
	f.MulVec(t.ke, mat.NewVecDense(6, t.ue[:]))
	return f
}

func (t *Truss3D) NodeIDs() []int       { return t.Nds[:] }
func (t *Truss3D) NumDOF() int          { return 6 }
func (t *Truss3D) DOFPerNode() int      { return 3 }
func (t *Truss3D) DOFTypes() []dof.Type { return dof.Translational3D(2) }

// Update stores the element displacements for subsequent post-processing calls.
func (t *Truss3D) Update(disp []float64) error {
	copy(t.ue[:], disp)
	return nil
}

func (t *Truss3D) CommitState() error   { return nil }
func (t *Truss3D) RevertToStart() error { t.ue = [6]float64{}; return nil }

// BodyForceLoad computes work-equivalent nodal forces due to a body force.
// For a linear 3D bar each node receives half the total weight (ρ·A·L/2·g).
func (t *Truss3D) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(6, nil)
	q := rho * t.A * t.length / 2.0
	f.SetVec(0, q*g[0])
	f.SetVec(1, q*g[1])
	f.SetVec(2, q*g[2])
	f.SetVec(3, q*g[0])
	f.SetVec(4, q*g[1])
	f.SetVec(5, q*g[2])
	return f
}

// GetMassMatrix returns the 6×6 consistent mass matrix in global coordinates.
// For a linear bar: Me = ρAL/6 · [2I₃, I₃; I₃, 2I₃]
// The translational mass matrix is frame-invariant (no rotation required).
func (t *Truss3D) GetMassMatrix(rho float64) *mat.Dense {
	me := mat.NewDense(6, 6, nil)
	c := rho * t.A * t.length / 6.0
	for i := 0; i < 3; i++ {
		me.Set(i, i, 2*c)
		me.Set(i+3, i+3, 2*c)
		me.Set(i, i+3, c)
		me.Set(i+3, i, c)
	}
	return me
}

// Length returns the element length.
func (t *Truss3D) Length() float64 { return t.length }

// AxialForce returns the axial force N = EA/L · Δu_axial (positive = tension).
func (t *Truss3D) AxialForce() float64 {
	du := [3]float64{t.ue[3] - t.ue[0], t.ue[4] - t.ue[1], t.ue[5] - t.ue[2]}
	return t.E * t.A / t.length * (t.cos[0]*du[0] + t.cos[1]*du[1] + t.cos[2]*du[2])
}

// AxialStress returns the axial normal stress σ = N/A.
func (t *Truss3D) AxialStress() float64 { return t.AxialForce() / t.A }
