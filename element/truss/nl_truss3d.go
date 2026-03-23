package truss

import (
	"math"

	"go-fem/dof"
	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

// NLTruss3D is a 2-node 3D nonlinear truss element.
// It uses a UniaxialMaterial for the axial σ–ε constitutive law, enabling
// material nonlinearity (yielding, softening, etc.) while keeping linear
// (small-strain) kinematics.
//
// DOF layout: 3 translations per node (UX, UY, UZ), 6 DOFs total.
// Axial strain: ε = (Δu · cos) / L₀  where Δu = u_j − u_i (global).
type NLTruss3D struct {
	ID     int
	Nds    [2]int
	Coords [2][3]float64
	A      float64 // cross-sectional area
	Mat    material.UniaxialMaterial

	length float64    // initial (reference) length
	cos    [3]float64 // unit direction cosines (i→j)
	ue     [6]float64 // committed element displacements
}

// NewNLTruss3D creates a 3D nonlinear truss element.
func NewNLTruss3D(id int, nodes [2]int, coords [2][3]float64, A float64, mat material.UniaxialMaterial) *NLTruss3D {
	t := &NLTruss3D{ID: id, Nds: nodes, Coords: coords, A: A, Mat: mat}
	t.computeGeometry()
	return t
}

func (t *NLTruss3D) computeGeometry() {
	dx := t.Coords[1][0] - t.Coords[0][0]
	dy := t.Coords[1][1] - t.Coords[0][1]
	dz := t.Coords[1][2] - t.Coords[0][2]
	t.length = math.Sqrt(dx*dx + dy*dy + dz*dz)
	t.cos[0] = dx / t.length
	t.cos[1] = dy / t.length
	t.cos[2] = dz / t.length
}

// ---------- Element interface ----------

// GetTangentStiffness builds Ke = (Et·A/L) · cᵀ c  extended to 6×6.
// Et is the current algorithmic tangent from the material.
func (t *NLTruss3D) GetTangentStiffness() *mat.Dense {
	et := t.Mat.GetTangent()
	k := et * t.A / t.length
	c := t.cos
	ke := mat.NewDense(6, 6, nil)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			v := k * c[i] * c[j]
			ke.Set(i, j, v)
			ke.Set(i+3, j+3, v)
			ke.Set(i, j+3, -v)
			ke.Set(i+3, j, -v)
		}
	}
	return ke
}

// GetResistingForce returns f_int = σ·A · [-c, c] in global coordinates.
func (t *NLTruss3D) GetResistingForce() *mat.VecDense {
	N := t.Mat.GetStress() * t.A // axial force (tension positive)
	c := t.cos
	f := mat.NewVecDense(6, nil)
	f.SetVec(0, -N*c[0])
	f.SetVec(1, -N*c[1])
	f.SetVec(2, -N*c[2])
	f.SetVec(3, N*c[0])
	f.SetVec(4, N*c[1])
	f.SetVec(5, N*c[2])
	return f
}

func (t *NLTruss3D) NodeIDs() []int       { return t.Nds[:] }
func (t *NLTruss3D) NumDOF() int          { return 6 }
func (t *NLTruss3D) DOFPerNode() int      { return 3 }
func (t *NLTruss3D) DOFTypes() []dof.Type { return dof.Translational3D(2) }

// Update computes the axial strain from the given displacements and calls
// SetTrialStrain on the material.
func (t *NLTruss3D) Update(disp []float64) error {
	copy(t.ue[:], disp)
	// Axial elongation = (u_j − u_i) · cos
	du := [3]float64{disp[3] - disp[0], disp[4] - disp[1], disp[5] - disp[2]}
	elongation := du[0]*t.cos[0] + du[1]*t.cos[1] + du[2]*t.cos[2]
	eps := elongation / t.length
	return t.Mat.SetTrialStrain(eps)
}

func (t *NLTruss3D) CommitState() error {
	return t.Mat.CommitState()
}

func (t *NLTruss3D) RevertToStart() error {
	t.ue = [6]float64{}
	return t.Mat.RevertToStart()
}

// AxialForce returns the current axial force N = σ·A (tension positive).
func (t *NLTruss3D) AxialForce() float64 { return t.Mat.GetStress() * t.A }

// AxialStress returns the current axial stress σ (tension positive).
func (t *NLTruss3D) AxialStress() float64 { return t.Mat.GetStress() }

// BodyForceLoad approximates body force as ρ·A·L/2 lumped at each node.
func (t *NLTruss3D) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(6, nil)
	q := rho * t.A * t.length / 2.0
	for i := 0; i < 3; i++ {
		f.SetVec(i, q*g[i])
		f.SetVec(i+3, q*g[i])
	}
	return f
}

// GetMassMatrix returns the 6×6 consistent mass matrix (same as linear Truss3D).
func (t *NLTruss3D) GetMassMatrix(rho float64) *mat.Dense {
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
