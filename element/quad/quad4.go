// Package quad implements 2D quadrilateral elements for plane stress/strain.
package quad

import (
	"math"

	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// PlaneType selects the constitutive model variant.
type PlaneType int

const (
	PlaneStress PlaneType = iota
	PlaneStrain
)

// Quad4 is a 4-node bilinear quadrilateral for 2D plane stress or plane strain.
// 2 DOFs per node (UX, UY), 8 DOFs total. 2×2 full Gauss integration.
type Quad4 struct {
	ID     int
	Nds    [4]int
	Coords [4][2]float64
	E      float64
	Nu     float64
	Thick  float64
	PType  PlaneType

	ke *mat.Dense
	ue [8]float64 // element displacements (set by Update)
}

// quad4Ref holds reference coords for the 4-node quad.
var quad4Ref = [4][2]float64{
	{-1, -1}, // 0
	{+1, -1}, // 1
	{+1, +1}, // 2
	{-1, +1}, // 3
}

// NewQuad4 creates a 4-node 2D plane element.
func NewQuad4(id int, nodes [4]int, coords [4][2]float64, e, nu, thick float64, ptype PlaneType) *Quad4 {
	q := &Quad4{
		ID: id, Nds: nodes, Coords: coords,
		E: e, Nu: nu, Thick: thick, PType: ptype,
	}
	q.formKe()
	return q
}

// formKe computes the 8×8 stiffness with 2×2 Gauss quadrature.
func (q *Quad4) formKe() {
	const ndof = 8
	D := q.planeD()
	q.ke = mat.NewDense(ndof, ndof, nil)

	gp := 1.0 / math.Sqrt(3.0)
	pts := [2]float64{-gp, gp}

	X := mat.NewDense(4, 2, nil)
	for i := 0; i < 4; i++ {
		X.Set(i, 0, q.Coords[i][0])
		X.Set(i, 1, q.Coords[i][1])
	}

	dNnat := mat.NewDense(2, 4, nil)
	J := mat.NewDense(2, 2, nil)
	dN := mat.NewDense(2, 4, nil)
	B := mat.NewDense(3, ndof, nil)
	DB := mat.NewDense(3, ndof, nil)
	BtDB := mat.NewDense(ndof, ndof, nil)
	var Jinv mat.Dense

	for _, xi := range pts {
		for _, eta := range pts {
			// Shape function derivatives in natural coords
			for i := 0; i < 4; i++ {
				si, ei := quad4Ref[i][0], quad4Ref[i][1]
				dNnat.Set(0, i, si*(1+ei*eta)/4) // ∂Ni/∂ξ
				dNnat.Set(1, i, (1+si*xi)*ei/4)  // ∂Ni/∂η
			}

			// Jacobian J = dNnat · X (2×2)
			J.Mul(dNnat, X)
			detJ := J.At(0, 0)*J.At(1, 1) - J.At(0, 1)*J.At(1, 0)

			if err := Jinv.Inverse(J); err != nil {
				panic("quad4: singular Jacobian")
			}

			// Physical derivatives
			dN.Mul(&Jinv, dNnat)

			// B matrix (3×8)
			B.Zero()
			for n := 0; n < 4; n++ {
				dx := dN.At(0, n)
				dy := dN.At(1, n)
				c := 2 * n
				B.Set(0, c, dx)   // εxx
				B.Set(1, c+1, dy) // εyy
				B.Set(2, c, dy)   // γxy
				B.Set(2, c+1, dx) //
			}

			// Ke += t · |detJ| · Bᵀ · D · B
			DB.Mul(D, B)
			BtDB.Mul(B.T(), DB)
			BtDB.Scale(q.Thick*math.Abs(detJ), BtDB)
			q.ke.Add(q.ke, BtDB)
		}
	}
}

// planeD returns the 3×3 constitutive matrix.
func (q *Quad4) planeD() *mat.Dense {
	E, nu := q.E, q.Nu
	D := mat.NewDense(3, 3, nil)

	if q.PType == PlaneStress {
		c := E / (1 - nu*nu)
		D.Set(0, 0, c)
		D.Set(0, 1, c*nu)
		D.Set(1, 0, c*nu)
		D.Set(1, 1, c)
		D.Set(2, 2, c*(1-nu)/2)
	} else { // PlaneStrain
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

func (q *Quad4) GetTangentStiffness() *mat.Dense { return q.ke }

// GetResistingForce returns Ke·ue (internal nodal force vector).
func (q *Quad4) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(8, nil)
	f.MulVec(q.ke, mat.NewVecDense(8, q.ue[:]))
	return f
}

func (q *Quad4) NodeIDs() []int  { return q.Nds[:] }
func (q *Quad4) NumDOF() int     { return 8 }
func (q *Quad4) DOFPerNode() int { return 2 }
func (q *Quad4) DOFTypes() []dof.Type {
	return []dof.Type{
		dof.UX, dof.UY,
		dof.UX, dof.UY,
		dof.UX, dof.UY,
		dof.UX, dof.UY,
	}
}

// Update stores the element displacements for post-processing.
func (q *Quad4) Update(disp []float64) error { copy(q.ue[:], disp); return nil }

func (q *Quad4) CommitState() error   { return nil }
func (q *Quad4) RevertToStart() error { q.ue = [8]float64{}; return nil }

// BodyForceLoad computes work-equivalent nodal forces due to a body force
// using the same 2×2 Gauss quadrature as formKe.
// Only X and Y components of g are used.
func (q *Quad4) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(8, nil)
	gp := 1.0 / math.Sqrt(3.0)
	pts := [2]float64{-gp, gp}
	for _, xi := range pts {
		for _, eta := range pts {
			var N [4]float64
			var j00, j01, j10, j11 float64
			for i := 0; i < 4; i++ {
				si, ei := quad4Ref[i][0], quad4Ref[i][1]
				N[i] = (1 + si*xi) * (1 + ei*eta) / 4
				dxi := si * (1 + ei*eta) / 4
				deta := (1 + si*xi) * ei / 4
				j00 += dxi * q.Coords[i][0]
				j01 += dxi * q.Coords[i][1]
				j10 += deta * q.Coords[i][0]
				j11 += deta * q.Coords[i][1]
			}
			detJ := j00*j11 - j01*j10
			scale := rho * q.Thick * math.Abs(detJ)
			for n := 0; n < 4; n++ {
				f.SetVec(2*n, f.AtVec(2*n)+scale*N[n]*g[0])
				f.SetVec(2*n+1, f.AtVec(2*n+1)+scale*N[n]*g[1])
			}
		}
	}
	return f
}

// StressCentroid returns the in-plane stress at the element centroid (ξ=η=0)
// as [sxx, syy, txy] = D·B·ue.
func (q *Quad4) StressCentroid() [3]float64 {
	ref := quad4Ref
	X := mat.NewDense(4, 2, nil)
	for i := 0; i < 4; i++ {
		X.Set(i, 0, q.Coords[i][0])
		X.Set(i, 1, q.Coords[i][1])
	}
	// dNnat at centroid (ξ=η=0): dNnat[0,i]=si/4, dNnat[1,i]=ei/4
	dNnat := mat.NewDense(2, 4, nil)
	for i := 0; i < 4; i++ {
		dNnat.Set(0, i, ref[i][0]/4)
		dNnat.Set(1, i, ref[i][1]/4)
	}
	J := mat.NewDense(2, 2, nil)
	J.Mul(dNnat, X)
	var Jinv mat.Dense
	Jinv.Inverse(J)
	dN := mat.NewDense(2, 4, nil)
	dN.Mul(&Jinv, dNnat)

	B := mat.NewDense(3, 8, nil)
	for n := 0; n < 4; n++ {
		dx, dy := dN.At(0, n), dN.At(1, n)
		c := 2 * n
		B.Set(0, c, dx)
		B.Set(1, c+1, dy)
		B.Set(2, c, dy)
		B.Set(2, c+1, dx)
	}

	D := q.planeD()
	Bu := mat.NewVecDense(3, nil)
	Bu.MulVec(B, mat.NewVecDense(8, q.ue[:]))
	sigma := mat.NewVecDense(3, nil)
	sigma.MulVec(D, Bu)
	return [3]float64{sigma.AtVec(0), sigma.AtVec(1), sigma.AtVec(2)}
}
