package truss

import (
	"math"

	"go-fem/dof"
	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

// NLTruss2D is a 2-node 2D nonlinear truss element.
// It uses a UniaxialMaterial for the axial σ–ε constitutive law, enabling
// material nonlinearity while keeping linear (small-strain) kinematics.
//
// DOF layout: 2 translations per node (UX, UY), 4 DOFs total.
// Axial strain: ε = ((u3−u1)·cos + (u4−u2)·sin) / L₀
type NLTruss2D struct {
	ID     int
	Nds    [2]int
	Coords [2][2]float64
	A      float64 // cross-sectional area
	Mat    material.UniaxialMaterial

	length float64 // initial length
	cos    float64 // direction cosine cx = dx/L
	sin    float64 // direction sine   cy = dy/L
	ue     [4]float64
}

// NewNLTruss2D creates a 2D nonlinear truss element.
func NewNLTruss2D(id int, nodes [2]int, coords [2][2]float64, A float64, mat material.UniaxialMaterial) *NLTruss2D {
	t := &NLTruss2D{ID: id, Nds: nodes, Coords: coords, A: A, Mat: mat}
	t.computeGeometry()
	return t
}

func (t *NLTruss2D) computeGeometry() {
	dx := t.Coords[1][0] - t.Coords[0][0]
	dy := t.Coords[1][1] - t.Coords[0][1]
	t.length = math.Sqrt(dx*dx + dy*dy)
	t.cos = dx / t.length
	t.sin = dy / t.length
}

// ---------- Element interface ----------

// GetTangentStiffness builds Ke = (Et·A/L) · [cc, cs, -cc, -cs; ...]
func (t *NLTruss2D) GetTangentStiffness() *mat.Dense {
	et := t.Mat.GetTangent()
	k := et * t.A / t.length
	c, s := t.cos, t.sin
	cc := k * c * c
	cs := k * c * s
	ss := k * s * s

	ke := mat.NewDense(4, 4, nil)
	ke.Set(0, 0, cc)
	ke.Set(0, 1, cs)
	ke.Set(0, 2, -cc)
	ke.Set(0, 3, -cs)
	ke.Set(1, 0, cs)
	ke.Set(1, 1, ss)
	ke.Set(1, 2, -cs)
	ke.Set(1, 3, -ss)
	ke.Set(2, 0, -cc)
	ke.Set(2, 1, -cs)
	ke.Set(2, 2, cc)
	ke.Set(2, 3, cs)
	ke.Set(3, 0, -cs)
	ke.Set(3, 1, -ss)
	ke.Set(3, 2, cs)
	ke.Set(3, 3, ss)
	return ke
}

// GetResistingForce returns f_int = σ·A · [-c, -s, c, s].
func (t *NLTruss2D) GetResistingForce() *mat.VecDense {
	N := t.Mat.GetStress() * t.A
	c, s := t.cos, t.sin
	f := mat.NewVecDense(4, nil)
	f.SetVec(0, -N*c)
	f.SetVec(1, -N*s)
	f.SetVec(2, N*c)
	f.SetVec(3, N*s)
	return f
}

func (t *NLTruss2D) NodeIDs() []int  { return t.Nds[:] }
func (t *NLTruss2D) NumDOF() int     { return 4 }
func (t *NLTruss2D) DOFPerNode() int { return 2 }
func (t *NLTruss2D) DOFTypes() []dof.Type {
	return []dof.Type{dof.UX, dof.UY, dof.UX, dof.UY}
}

// Update computes axial strain from given displacements and calls SetTrialStrain.
func (t *NLTruss2D) Update(disp []float64) error {
	copy(t.ue[:], disp)
	du0 := disp[2] - disp[0]
	du1 := disp[3] - disp[1]
	eps := (du0*t.cos + du1*t.sin) / t.length
	return t.Mat.SetTrialStrain(eps)
}

func (t *NLTruss2D) CommitState() error {
	return t.Mat.CommitState()
}

func (t *NLTruss2D) RevertToStart() error {
	t.ue = [4]float64{}
	return t.Mat.RevertToStart()
}

// AxialForce returns the current axial force N = σ·A (tension positive).
func (t *NLTruss2D) AxialForce() float64 { return t.Mat.GetStress() * t.A }

// AxialStress returns the current axial stress σ (tension positive).
func (t *NLTruss2D) AxialStress() float64 { return t.Mat.GetStress() }

// BodyForceLoad approximates body force as ρ·A·L/2 lumped at each node (XY only).
func (t *NLTruss2D) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(4, nil)
	q := rho * t.A * t.length / 2.0
	f.SetVec(0, q*g[0])
	f.SetVec(1, q*g[1])
	f.SetVec(2, q*g[0])
	f.SetVec(3, q*g[1])
	return f
}

// GetMassMatrix returns the 4×4 consistent mass matrix (same as linear Truss2D).
func (t *NLTruss2D) GetMassMatrix(rho float64) *mat.Dense {
	me := mat.NewDense(4, 4, nil)
	c := rho * t.A * t.length / 6.0
	for i := 0; i < 2; i++ {
		me.Set(i, i, 2*c)
		me.Set(i+2, i+2, 2*c)
		me.Set(i, i+2, c)
		me.Set(i+2, i, c)
	}
	return me
}
