package quad

import (
	"math"

	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// Tri3 is a 3-node constant-strain triangle (CST) for 2D plane stress or
// plane strain.  2 DOFs per node (UX, UY), 6 DOFs total.
// Uses 1-point centroid Gauss rule (exact for linear displacement field).
//
// Node numbering (CCW):
//
//	2
//	|\
//	| \
//	0--1
//
// Natural (area) coordinates: L1 = ξ, L2 = η, L3 = 1 − ξ − η.
type Tri3 struct {
	ID     int
	Nds    [3]int
	Coords [3][2]float64
	E      float64
	Nu     float64
	Thick  float64
	PType  PlaneType

	ke *mat.Dense
	ue [6]float64
}

// NewTri3 creates a 3-node constant-strain triangle element.
func NewTri3(id int, nodes [3]int, coords [3][2]float64, e, nu, thick float64, ptype PlaneType) *Tri3 {
	t := &Tri3{
		ID: id, Nds: nodes, Coords: coords,
		E: e, Nu: nu, Thick: thick, PType: ptype,
	}
	t.formKe()
	return t
}

// triArea2 returns twice the signed area of a triangle (positive if CCW).
func triArea2(c [3][2]float64) float64 {
	return (c[1][0]-c[0][0])*(c[2][1]-c[0][1]) - (c[2][0]-c[0][0])*(c[1][1]-c[0][1])
}

// formKe computes the 6×6 stiffness matrix.
// For CST the B-matrix is constant, so the integral reduces to Ke = t·A·Bᵀ·D·B.
func (t *Tri3) formKe() {
	const ndof = 6
	D := tri3PlaneD(t.E, t.Nu, t.PType)
	t.ke = mat.NewDense(ndof, ndof, nil)

	A2 := triArea2(t.Coords)
	area := math.Abs(A2) / 2

	// Shape function derivatives in physical coordinates (constant):
	// Ni = (ai + bi*x + ci*y) / (2A)  →  dNi/dx = bi/(2A),  dNi/dy = ci/(2A)
	x := [3]float64{t.Coords[0][0], t.Coords[1][0], t.Coords[2][0]}
	y := [3]float64{t.Coords[0][1], t.Coords[1][1], t.Coords[2][1]}

	b := [3]float64{y[1] - y[2], y[2] - y[0], y[0] - y[1]}
	c := [3]float64{x[2] - x[1], x[0] - x[2], x[1] - x[0]}

	inv2A := 1.0 / A2

	B := mat.NewDense(3, ndof, nil)
	for n := 0; n < 3; n++ {
		col := 2 * n
		B.Set(0, col, b[n]*inv2A)   // εxx = dN/dx
		B.Set(1, col+1, c[n]*inv2A) // εyy = dN/dy
		B.Set(2, col, c[n]*inv2A)   // γxy = dN/dy
		B.Set(2, col+1, b[n]*inv2A) //     + dN/dx
	}

	// Ke = t · A · Bᵀ · D · B
	DB := mat.NewDense(3, ndof, nil)
	DB.Mul(D, B)
	t.ke.Mul(B.T(), DB)
	t.ke.Scale(t.Thick*area, t.ke)
}

func tri3PlaneD(E, nu float64, ptype PlaneType) *mat.Dense {
	D := mat.NewDense(3, 3, nil)
	if ptype == PlaneStress {
		c := E / (1 - nu*nu)
		D.Set(0, 0, c)
		D.Set(0, 1, c*nu)
		D.Set(1, 0, c*nu)
		D.Set(1, 1, c)
		D.Set(2, 2, c*(1-nu)/2)
	} else {
		c := E / ((1 + nu) * (1 - 2*nu))
		D.Set(0, 0, c*(1-nu))
		D.Set(0, 1, c*nu)
		D.Set(1, 0, c*nu)
		D.Set(1, 1, c*(1-nu))
		D.Set(2, 2, c*(1-2*nu)/2)
	}
	return D
}

// ---------- Element interface ----------

func (t *Tri3) GetTangentStiffness() *mat.Dense { return t.ke }

func (t *Tri3) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(6, nil)
	f.MulVec(t.ke, mat.NewVecDense(6, t.ue[:]))
	return f
}

func (t *Tri3) NodeIDs() []int  { return t.Nds[:] }
func (t *Tri3) NumDOF() int     { return 6 }
func (t *Tri3) DOFPerNode() int { return 2 }
func (t *Tri3) DOFTypes() []dof.Type {
	return []dof.Type{
		dof.UX, dof.UY,
		dof.UX, dof.UY,
		dof.UX, dof.UY,
	}
}

func (t *Tri3) Update(disp []float64) error { copy(t.ue[:], disp); return nil }
func (t *Tri3) CommitState() error          { return nil }
func (t *Tri3) RevertToStart() error        { t.ue = [6]float64{}; return nil }

// BodyForceLoad computes work-equivalent nodal forces due to a body force.
// For a CST triangle each node receives area/3 of the total surface weight
// (ρ·thick·area/3·g). Only X and Y components of g are used.
func (t *Tri3) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(6, nil)
	q := rho * t.Thick * math.Abs(triArea2(t.Coords)) / 6.0
	f.SetVec(0, q*g[0])
	f.SetVec(1, q*g[1])
	f.SetVec(2, q*g[0])
	f.SetVec(3, q*g[1])
	f.SetVec(4, q*g[0])
	f.SetVec(5, q*g[1])
	return f
}

// StressCentroid returns the in-plane stress (constant over the element)
// as [σxx, σyy, τxy] = D·B·ue.
func (t *Tri3) StressCentroid() [3]float64 {
	A2 := triArea2(t.Coords)
	x := [3]float64{t.Coords[0][0], t.Coords[1][0], t.Coords[2][0]}
	y := [3]float64{t.Coords[0][1], t.Coords[1][1], t.Coords[2][1]}
	b := [3]float64{y[1] - y[2], y[2] - y[0], y[0] - y[1]}
	c := [3]float64{x[2] - x[1], x[0] - x[2], x[1] - x[0]}
	inv2A := 1.0 / A2

	B := mat.NewDense(3, 6, nil)
	for n := 0; n < 3; n++ {
		col := 2 * n
		B.Set(0, col, b[n]*inv2A)
		B.Set(1, col+1, c[n]*inv2A)
		B.Set(2, col, c[n]*inv2A)
		B.Set(2, col+1, b[n]*inv2A)
	}

	D := tri3PlaneD(t.E, t.Nu, t.PType)
	Bu := mat.NewVecDense(3, nil)
	Bu.MulVec(B, mat.NewVecDense(6, t.ue[:]))
	sigma := mat.NewVecDense(3, nil)
	sigma.MulVec(D, Bu)
	return [3]float64{sigma.AtVec(0), sigma.AtVec(1), sigma.AtVec(2)}
}
