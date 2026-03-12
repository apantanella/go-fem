package quad

import (
	"math"

	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// Tri6 is a 6-node quadratic triangle (LST — Linear Strain Triangle) for 2D
// plane stress or plane strain.  2 DOFs per node (UX, UY), 12 DOFs total.
// Uses 3-point Gauss quadrature on the triangle (exact for quadratic fields).
//
// Node numbering (CCW — corners then midsides):
//
//	2
//	|\
//	5  4
//	|   \
//	0--3--1
//
// Natural (area) coordinates: L1 = ξ, L2 = η, L3 = 1 − ξ − η.
//
// Shape functions (quadratic Lagrange on a triangle):
//
//	N0 = L1(2L1 − 1)    (corner 0)
//	N1 = L2(2L2 − 1)    (corner 1)
//	N2 = L3(2L3 − 1)    (corner 2)
//	N3 = 4 L1 L2         (midside 0-1)
//	N4 = 4 L2 L3         (midside 1-2)
//	N5 = 4 L3 L1         (midside 2-0)
type Tri6 struct {
	ID     int
	Nds    [6]int
	Coords [6][2]float64
	E      float64
	Nu     float64
	Thick  float64
	PType  PlaneType

	ke *mat.Dense
	ue [12]float64
}

// NewTri6 creates a 6-node quadratic triangle element.
func NewTri6(id int, nodes [6]int, coords [6][2]float64, e, nu, thick float64, ptype PlaneType) *Tri6 {
	t := &Tri6{
		ID: id, Nds: nodes, Coords: coords,
		E: e, Nu: nu, Thick: thick, PType: ptype,
	}
	t.formKe()
	return t
}

// tri6Shape evaluates the 6 shape functions and their derivatives w.r.t.
// area coordinates L1 = ξ, L2 = η at the point (xi, eta).
// Returns dNdxi[6] = ∂N/∂ξ, dNdeta[6] = ∂N/∂η.
func tri6Shape(xi, eta float64) (N [6]float64, dNdxi, dNdeta [6]float64) {
	L1 := xi
	L2 := eta
	L3 := 1 - xi - eta

	// Corner nodes
	N[0] = L1 * (2*L1 - 1)
	N[1] = L2 * (2*L2 - 1)
	N[2] = L3 * (2*L3 - 1)
	// Midside nodes
	N[3] = 4 * L1 * L2
	N[4] = 4 * L2 * L3
	N[5] = 4 * L3 * L1

	// ∂N/∂ξ  (∂L1/∂ξ = 1, ∂L2/∂ξ = 0, ∂L3/∂ξ = -1)
	dNdxi[0] = 4*L1 - 1      // ∂(L1(2L1-1))/∂ξ
	dNdxi[1] = 0             // ∂(L2(2L2-1))/∂ξ
	dNdxi[2] = -4*L3 + 1     // ∂(L3(2L3-1))/∂ξ = (2L3-1)(-1) + L3·2·(-1) = -(4L3-1)
	dNdxi[3] = 4 * L2        // ∂(4L1·L2)/∂ξ
	dNdxi[4] = -4 * L2       // ∂(4L2·L3)/∂ξ = 4L2·(-1)
	dNdxi[5] = 4 * (L3 - L1) // ∂(4L3·L1)/∂ξ = 4(L3·1 + L1·(-1))

	// ∂N/∂η  (∂L1/∂η = 0, ∂L2/∂η = 1, ∂L3/∂η = -1)
	dNdeta[0] = 0             // ∂(L1(2L1-1))/∂η
	dNdeta[1] = 4*L2 - 1      // ∂(L2(2L2-1))/∂η
	dNdeta[2] = -4*L3 + 1     // same form as dNdxi[2] by symmetry
	dNdeta[3] = 4 * L1        // ∂(4L1·L2)/∂η
	dNdeta[4] = 4 * (L3 - L2) // ∂(4L2·L3)/∂η = 4(L3·1 + L2·(-1))
	dNdeta[5] = -4 * L1       // ∂(4L3·L1)/∂η = 4L1·(-1)

	return
}

// 3-point Gauss rule for triangle in area coordinates:
// Points at (1/6, 1/6), (2/3, 1/6), (1/6, 2/3), weight = 1/6 each.
// The natural-coordinates triangle has area 1/2, so the total weight = 1/2.
var tri6GP = [3][3]float64{
	{1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0},
	{2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0},
	{1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0},
}

// formKe assembles the 12×12 stiffness with 3-point Gauss on the triangle.
func (t *Tri6) formKe() {
	const ndof = 12
	D := tri6PlaneD(t.E, t.Nu, t.PType)
	t.ke = mat.NewDense(ndof, ndof, nil)

	X := mat.NewDense(6, 2, nil)
	for i := 0; i < 6; i++ {
		X.Set(i, 0, t.Coords[i][0])
		X.Set(i, 1, t.Coords[i][1])
	}

	J := mat.NewDense(2, 2, nil)
	B := mat.NewDense(3, ndof, nil)
	DB := mat.NewDense(3, ndof, nil)
	BtDB := mat.NewDense(ndof, ndof, nil)

	for _, gp := range tri6GP {
		xi, eta, w := gp[0], gp[1], gp[2]
		_, dNdxi, dNdeta := tri6Shape(xi, eta)

		// Jacobian: J = [[Σ dNdxi·x, Σ dNdxi·y],[Σ dNdeta·x, Σ dNdeta·y]]
		J.Zero()
		for n := 0; n < 6; n++ {
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

		// Physical derivatives
		var dNdx, dNdy [6]float64
		for n := 0; n < 6; n++ {
			dNdx[n] = j00*dNdxi[n] + j01*dNdeta[n]
			dNdy[n] = j10*dNdxi[n] + j11*dNdeta[n]
		}

		// B matrix (3×12)
		B.Zero()
		for n := 0; n < 6; n++ {
			c := 2 * n
			B.Set(0, c, dNdx[n])   // εxx
			B.Set(1, c+1, dNdy[n]) // εyy
			B.Set(2, c, dNdy[n])   // γxy
			B.Set(2, c+1, dNdx[n])
		}

		// Ke += t · w · |detJ| · Bᵀ D B
		DB.Mul(D, B)
		BtDB.Mul(B.T(), DB)
		scale := t.Thick * w * math.Abs(detJ)
		for i := 0; i < ndof; i++ {
			for j := 0; j < ndof; j++ {
				t.ke.Set(i, j, t.ke.At(i, j)+scale*BtDB.At(i, j))
			}
		}
	}
}

func tri6PlaneD(E, nu float64, ptype PlaneType) *mat.Dense {
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

func (t *Tri6) GetTangentStiffness() *mat.Dense { return t.ke }

func (t *Tri6) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(12, nil)
	f.MulVec(t.ke, mat.NewVecDense(12, t.ue[:]))
	return f
}

func (t *Tri6) NodeIDs() []int  { return t.Nds[:] }
func (t *Tri6) NumDOF() int     { return 12 }
func (t *Tri6) DOFPerNode() int { return 2 }
func (t *Tri6) DOFTypes() []dof.Type {
	types := make([]dof.Type, 12)
	for i := 0; i < 6; i++ {
		types[2*i] = dof.UX
		types[2*i+1] = dof.UY
	}
	return types
}

func (t *Tri6) Update(disp []float64) error { copy(t.ue[:], disp); return nil }
func (t *Tri6) CommitState() error          { return nil }
func (t *Tri6) RevertToStart() error        { t.ue = [12]float64{}; return nil }

// BodyForceLoad computes work-equivalent nodal forces due to a body force
// using the same 3-point Gauss quadrature as formKe.
// Only X and Y components of g are used.
func (t *Tri6) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(12, nil)
	for _, gp := range tri6GP {
		xi, eta, w := gp[0], gp[1], gp[2]
		N, dNdxi, dNdeta := tri6Shape(xi, eta)
		var j00, j01, j10, j11 float64
		for n := 0; n < 6; n++ {
			j00 += dNdxi[n] * t.Coords[n][0]
			j01 += dNdxi[n] * t.Coords[n][1]
			j10 += dNdeta[n] * t.Coords[n][0]
			j11 += dNdeta[n] * t.Coords[n][1]
		}
		detJ := j00*j11 - j01*j10
		scale := rho * t.Thick * w * math.Abs(detJ)
		for n := 0; n < 6; n++ {
			f.SetVec(2*n, f.AtVec(2*n)+scale*N[n]*g[0])
			f.SetVec(2*n+1, f.AtVec(2*n+1)+scale*N[n]*g[1])
		}
	}
	return f
}

// StressCentroid returns the in-plane stress at the element centroid
// (ξ = η = 1/3) as [σxx, σyy, τxy] = D·B·ue.
func (t *Tri6) StressCentroid() [3]float64 {
	xi, eta := 1.0/3.0, 1.0/3.0
	_, dNdxi, dNdeta := tri6Shape(xi, eta)

	X := mat.NewDense(6, 2, nil)
	for i := 0; i < 6; i++ {
		X.Set(i, 0, t.Coords[i][0])
		X.Set(i, 1, t.Coords[i][1])
	}

	J := [2][2]float64{}
	for n := 0; n < 6; n++ {
		J[0][0] += dNdxi[n] * t.Coords[n][0]
		J[0][1] += dNdxi[n] * t.Coords[n][1]
		J[1][0] += dNdeta[n] * t.Coords[n][0]
		J[1][1] += dNdeta[n] * t.Coords[n][1]
	}
	detJ := J[0][0]*J[1][1] - J[0][1]*J[1][0]
	j00 := J[1][1] / detJ
	j01 := -J[0][1] / detJ
	j10 := -J[1][0] / detJ
	j11 := J[0][0] / detJ

	var dNdx, dNdy [6]float64
	for n := 0; n < 6; n++ {
		dNdx[n] = j00*dNdxi[n] + j01*dNdeta[n]
		dNdy[n] = j10*dNdxi[n] + j11*dNdeta[n]
	}

	B := mat.NewDense(3, 12, nil)
	for n := 0; n < 6; n++ {
		c := 2 * n
		B.Set(0, c, dNdx[n])
		B.Set(1, c+1, dNdy[n])
		B.Set(2, c, dNdy[n])
		B.Set(2, c+1, dNdx[n])
	}

	D := tri6PlaneD(t.E, t.Nu, t.PType)
	Bu := mat.NewVecDense(3, nil)
	Bu.MulVec(B, mat.NewVecDense(12, t.ue[:]))
	sigma := mat.NewVecDense(3, nil)
	sigma.MulVec(D, Bu)
	return [3]float64{sigma.AtVec(0), sigma.AtVec(1), sigma.AtVec(2)}
}
