package quad

import (
	"math"

	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// Quad8 is an 8-node serendipity quadrilateral for 2D plane stress or plane
// strain.  2 DOFs per node (UX, UY), 16 DOFs total.  Full 3×3 Gauss integration
// provides exact integration of the quadratic displacement field and produces
// accurate stress gradients — this makes Quad8 the element of choice near
// stress concentrations, openings, and re-entrant corners.
//
// Node numbering (CCW):
//
//	3 — 6 — 2
//	|       |
//	7       5
//	|       |
//	0 — 4 — 1
//
// Nodes 0-3 are corners at (±1,±1); nodes 4-7 are midside nodes.
type Quad8 struct {
	ID     int
	Nds    [8]int
	Coords [8][2]float64
	E      float64
	Nu     float64
	Thick  float64
	PType  PlaneType

	ke *mat.Dense
	ue [16]float64
}

// quad8Ref holds the natural coordinates for the 8 nodes.
var quad8Ref = [8][2]float64{
	{-1, -1}, // 0 corner
	{+1, -1}, // 1 corner
	{+1, +1}, // 2 corner
	{-1, +1}, // 3 corner
	{0, -1},  // 4 midside (0-1)
	{+1, 0},  // 5 midside (1-2)
	{0, +1},  // 6 midside (2-3)
	{-1, 0},  // 7 midside (3-0)
}

// NewQuad8 creates an 8-node serendipity plane element.
func NewQuad8(id int, nodes [8]int, coords [8][2]float64, e, nu, thick float64, ptype PlaneType) *Quad8 {
	q := &Quad8{
		ID: id, Nds: nodes, Coords: coords,
		E: e, Nu: nu, Thick: thick, PType: ptype,
	}
	q.formKe()
	return q
}

// quad8Shape computes the 8 shape functions and their natural-coordinate
// derivatives at (xi, eta).
//
// Returns N[8], dNdxi[8], dNdeta[8].
func quad8Shape(xi, eta float64) (N, dNdxi, dNdeta [8]float64) {
	// Corner nodes (i = 0..3)
	for i, ref := range quad8Ref[:4] {
		xi0, eta0 := ref[0], ref[1]
		N[i] = (1 + xi*xi0) * (1 + eta*eta0) * (xi*xi0 + eta*eta0 - 1) / 4
		dNdxi[i] = xi0 * (1 + eta*eta0) * (2*xi*xi0 + eta*eta0) / 4
		dNdeta[i] = eta0 * (1 + xi*xi0) * (xi*xi0 + 2*eta*eta0) / 4
	}
	// Midside nodes
	// Node 4: (0,-1)  N4 = (1-xi²)(1-eta)/2
	N[4] = (1 - xi*xi) * (1 - eta) / 2
	dNdxi[4] = -xi * (1 - eta)
	dNdeta[4] = -(1 - xi*xi) / 2
	// Node 5: (1,0)  N5 = (1+xi)(1-eta²)/2
	N[5] = (1 + xi) * (1 - eta*eta) / 2
	dNdxi[5] = (1 - eta*eta) / 2
	dNdeta[5] = -(1 + xi) * eta
	// Node 6: (0,1)  N6 = (1-xi²)(1+eta)/2
	N[6] = (1 - xi*xi) * (1 + eta) / 2
	dNdxi[6] = -xi * (1 + eta)
	dNdeta[6] = (1 - xi*xi) / 2
	// Node 7: (-1,0)  N7 = (1-xi)(1-eta²)/2
	N[7] = (1 - xi) * (1 - eta*eta) / 2
	dNdxi[7] = -(1 - eta*eta) / 2
	dNdeta[7] = -(1 - xi) * eta
	return
}

// formKe assembles the 16×16 stiffness using 3×3 Gauss quadrature.
func (q *Quad8) formKe() {
	const ndof = 16
	D := quad8PlaneD(q.E, q.Nu, q.PType)
	q.ke = mat.NewDense(ndof, ndof, nil)

	// 3-point Gauss rule
	gp := math.Sqrt(3.0 / 5.0)
	pts := [3]float64{-gp, 0, gp}
	wts := [3]float64{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0}

	X := mat.NewDense(8, 2, nil)
	for i := 0; i < 8; i++ {
		X.Set(i, 0, q.Coords[i][0])
		X.Set(i, 1, q.Coords[i][1])
	}

	J := mat.NewDense(2, 2, nil)
	B := mat.NewDense(3, ndof, nil)
	DB := mat.NewDense(3, ndof, nil)
	BtDB := mat.NewDense(ndof, ndof, nil)

	for ai, xi := range pts {
		for bi, eta := range pts {
			w := wts[ai] * wts[bi]
			_, dNdxi, dNdeta := quad8Shape(xi, eta)

			// Jacobian: J = [dNdxi; dNdeta] × X  (2×2)
			J.Zero()
			for n := 0; n < 8; n++ {
				J.Set(0, 0, J.At(0, 0)+dNdxi[n]*X.At(n, 0))
				J.Set(0, 1, J.At(0, 1)+dNdxi[n]*X.At(n, 1))
				J.Set(1, 0, J.At(1, 0)+dNdeta[n]*X.At(n, 0))
				J.Set(1, 1, J.At(1, 1)+dNdeta[n]*X.At(n, 1))
			}
			detJ := J.At(0, 0)*J.At(1, 1) - J.At(0, 1)*J.At(1, 0)

			// J⁻¹
			j00 := J.At(1, 1) / detJ
			j01 := -J.At(0, 1) / detJ
			j10 := -J.At(1, 0) / detJ
			j11 := J.At(0, 0) / detJ

			// Physical derivatives dN/dx, dN/dy
			var dNdx, dNdy [8]float64
			for n := 0; n < 8; n++ {
				dNdx[n] = j00*dNdxi[n] + j01*dNdeta[n]
				dNdy[n] = j10*dNdxi[n] + j11*dNdeta[n]
			}

			// B matrix (3×16)
			B.Zero()
			for n := 0; n < 8; n++ {
				c := 2 * n
				B.Set(0, c, dNdx[n])   // εxx
				B.Set(1, c+1, dNdy[n]) // εyy
				B.Set(2, c, dNdy[n])   // γxy
				B.Set(2, c+1, dNdx[n]) //
			}

			// Ke += t · w · |detJ| · Bᵀ D B
			DB.Mul(D, B)
			BtDB.Mul(B.T(), DB)
			scale := q.Thick * w * math.Abs(detJ)
			for i := 0; i < ndof; i++ {
				for j := 0; j < ndof; j++ {
					q.ke.Set(i, j, q.ke.At(i, j)+scale*BtDB.At(i, j))
				}
			}
		}
	}
}

func quad8PlaneD(E, nu float64, ptype PlaneType) *mat.Dense {
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

func (q *Quad8) GetTangentStiffness() *mat.Dense { return q.ke }

func (q *Quad8) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(16, nil)
	f.MulVec(q.ke, mat.NewVecDense(16, q.ue[:]))
	return f
}

func (q *Quad8) NodeIDs() []int  { return q.Nds[:] }
func (q *Quad8) NumDOF() int     { return 16 }
func (q *Quad8) DOFPerNode() int { return 2 }
func (q *Quad8) DOFTypes() []dof.Type {
	types := make([]dof.Type, 16)
	for i := 0; i < 8; i++ {
		types[2*i] = dof.UX
		types[2*i+1] = dof.UY
	}
	return types
}

func (q *Quad8) Update(disp []float64) error { copy(q.ue[:], disp); return nil }
func (q *Quad8) CommitState() error          { return nil }
func (q *Quad8) RevertToStart() error        { q.ue = [16]float64{}; return nil }

// BodyForceLoad computes work-equivalent nodal forces due to a body force
// using the same 3×3 Gauss quadrature as formKe.
// Only X and Y components of g are used.
func (q *Quad8) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(16, nil)
	gp := math.Sqrt(3.0 / 5.0)
	pts := [3]float64{-gp, 0, gp}
	wts := [3]float64{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0}
	for ai, xi := range pts {
		for bi, eta := range pts {
			w := wts[ai] * wts[bi]
			N, dNdxi, dNdeta := quad8Shape(xi, eta)
			var j00, j01, j10, j11 float64
			for n := 0; n < 8; n++ {
				j00 += dNdxi[n] * q.Coords[n][0]
				j01 += dNdxi[n] * q.Coords[n][1]
				j10 += dNdeta[n] * q.Coords[n][0]
				j11 += dNdeta[n] * q.Coords[n][1]
			}
			detJ := j00*j11 - j01*j10
			scale := rho * q.Thick * w * math.Abs(detJ)
			for n := 0; n < 8; n++ {
				f.SetVec(2*n, f.AtVec(2*n)+scale*N[n]*g[0])
				f.SetVec(2*n+1, f.AtVec(2*n+1)+scale*N[n]*g[1])
			}
		}
	}
	return f
}

// StressCentroid returns the in-plane stress at the element centroid (ξ=η=0).
func (q *Quad8) StressCentroid() [3]float64 {
	_, dNdxi, dNdeta := quad8Shape(0, 0)

	J := [2][2]float64{}
	for n := 0; n < 8; n++ {
		J[0][0] += dNdxi[n] * q.Coords[n][0]
		J[0][1] += dNdxi[n] * q.Coords[n][1]
		J[1][0] += dNdeta[n] * q.Coords[n][0]
		J[1][1] += dNdeta[n] * q.Coords[n][1]
	}
	detJ := J[0][0]*J[1][1] - J[0][1]*J[1][0]
	j00 := J[1][1] / detJ
	j01 := -J[0][1] / detJ
	j10 := -J[1][0] / detJ
	j11 := J[0][0] / detJ

	var dNdx, dNdy [8]float64
	for n := 0; n < 8; n++ {
		dNdx[n] = j00*dNdxi[n] + j01*dNdeta[n]
		dNdy[n] = j10*dNdxi[n] + j11*dNdeta[n]
	}

	B := mat.NewDense(3, 16, nil)
	for n := 0; n < 8; n++ {
		c := 2 * n
		B.Set(0, c, dNdx[n])
		B.Set(1, c+1, dNdy[n])
		B.Set(2, c, dNdy[n])
		B.Set(2, c+1, dNdx[n])
	}

	D := quad8PlaneD(q.E, q.Nu, q.PType)
	Bu := mat.NewVecDense(3, nil)
	Bu.MulVec(B, mat.NewVecDense(16, q.ue[:]))
	sigma := mat.NewVecDense(3, nil)
	sigma.MulVec(D, Bu)
	return [3]float64{sigma.AtVec(0), sigma.AtVec(1), sigma.AtVec(2)}
}
