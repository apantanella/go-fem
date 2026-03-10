// Package solid implements 3D solid (continuum) elements.
package solid

import (
	"math"

	"go-fem/dof"
	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

// Tet10 is a 10-node quadratic tetrahedron with 4-point Gauss quadrature.
// 3 DOFs per node, 30 DOFs total.
type Tet10 struct {
	ID     int
	Nds    [10]int
	Coords [10][3]float64
	Mat    material.Material3D

	ke *mat.Dense
	ue [30]float64 // element displacements (set by Update)
}

// NewTet10 creates a 10-node quadratic tetrahedron.
func NewTet10(id int, nodes [10]int, coords [10][3]float64, m material.Material3D) *Tet10 {
	t := &Tet10{ID: id, Nds: nodes, Coords: coords, Mat: m}
	t.formKe()
	return t
}

// formKe computes the 30×30 stiffness using 4-point Gauss quadrature.
func (t *Tet10) formKe() {
	const ndof = 30
	D := t.Mat.GetTangent()
	t.ke = mat.NewDense(ndof, ndof, nil)

	// 4-point Gauss quadrature for tetrahedron
	a := 0.5854101966249685
	b := 0.1381966011250105
	gpts := [4][3]float64{
		{a, b, b},
		{b, a, b},
		{b, b, a},
		{b, b, b},
	}
	w := 1.0 / 24.0 // weight for each point

	X := mat.NewDense(10, 3, nil)
	for i := 0; i < 10; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, t.Coords[i][j])
		}
	}

	dNnat := mat.NewDense(3, 10, nil)
	J := mat.NewDense(3, 3, nil)
	dN := mat.NewDense(3, 10, nil)
	B := mat.NewDense(6, ndof, nil)
	DB := mat.NewDense(6, ndof, nil)
	BtDB := mat.NewDense(ndof, ndof, nil)
	var Jinv mat.Dense

	for _, gp := range gpts {
		xi, eta, zeta := gp[0], gp[1], gp[2]
		L1 := 1 - xi - eta - zeta
		L2 := xi
		L3 := eta
		L4 := zeta

		// Shape function derivatives ∂Ni/∂(ξ, η, ζ) for 10-node tet
		// Corner nodes: Ni = Li(2Li-1)
		// Midside nodes: Nij = 4·Li·Lj
		dNnat.Zero()

		// ∂L1/∂ξ = -1, ∂L1/∂η = -1, ∂L1/∂ζ = -1
		// ∂L2/∂ξ =  1, ∂L2/∂η =  0, ∂L2/∂ζ =  0
		// ∂L3/∂ξ =  0, ∂L3/∂η =  1, ∂L3/∂ζ =  0
		// ∂L4/∂ξ =  0, ∂L4/∂η =  0, ∂L4/∂ζ =  1

		// Node 0: N0 = L1(2L1-1),  dN0/dξ = (4L1-1)·(-1) = 1-4L1
		v0 := 4*L1 - 1
		dNnat.Set(0, 0, -v0) // ∂N0/∂ξ
		dNnat.Set(1, 0, -v0) // ∂N0/∂η
		dNnat.Set(2, 0, -v0) // ∂N0/∂ζ

		// Node 1: N1 = L2(2L2-1),  dN1/dξ = 4L2-1
		v1 := 4*L2 - 1
		dNnat.Set(0, 1, v1)

		// Node 2: N2 = L3(2L3-1),  dN2/dη = 4L3-1
		v2 := 4*L3 - 1
		dNnat.Set(1, 2, v2)

		// Node 3: N3 = L4(2L4-1),  dN3/dζ = 4L4-1
		v3 := 4*L4 - 1
		dNnat.Set(2, 3, v3)

		// Node 4: N4 = 4·L1·L2, midside 0-1
		dNnat.Set(0, 4, 4*(L1-L2)) // 4(-L2 + L1·1) = 4(L1-L2)
		dNnat.Set(1, 4, -4*L2)
		dNnat.Set(2, 4, -4*L2)

		// Node 5: N5 = 4·L2·L3, midside 1-2
		dNnat.Set(0, 5, 4*L3)
		dNnat.Set(1, 5, 4*L2)

		// Node 6: N6 = 4·L1·L3, midside 0-2
		dNnat.Set(0, 6, -4*L3)
		dNnat.Set(1, 6, 4*(L1-L3))
		dNnat.Set(2, 6, -4*L3)

		// Node 7: N7 = 4·L1·L4, midside 0-3
		dNnat.Set(0, 7, -4*L4)
		dNnat.Set(1, 7, -4*L4)
		dNnat.Set(2, 7, 4*(L1-L4))

		// Node 8: N8 = 4·L2·L4, midside 1-3
		dNnat.Set(0, 8, 4*L4)
		dNnat.Set(2, 8, 4*L2)

		// Node 9: N9 = 4·L3·L4, midside 2-3
		dNnat.Set(1, 9, 4*L4)
		dNnat.Set(2, 9, 4*L3)

		// Jacobian J = dNnat · X (3×3)
		J.Mul(dNnat, X)
		detJ := mat.Det(J)

		if err := Jinv.Inverse(J); err != nil {
			panic("tet10: singular Jacobian – degenerate element")
		}

		// Physical derivatives
		dN.Mul(&Jinv, dNnat)

		// B matrix (6×30)
		B.Zero()
		for n := 0; n < 10; n++ {
			dx := dN.At(0, n)
			dy := dN.At(1, n)
			dz := dN.At(2, n)
			c := 3 * n

			B.Set(0, c, dx)
			B.Set(1, c+1, dy)
			B.Set(2, c+2, dz)
			B.Set(3, c, dy)
			B.Set(3, c+1, dx)
			B.Set(4, c+1, dz)
			B.Set(4, c+2, dy)
			B.Set(5, c, dz)
			B.Set(5, c+2, dx)
		}

		// Ke += w · |detJ| · Bᵀ · D · B
		// Note: w already accounts for tet reference volume (1/6)
		DB.Mul(D, B)
		BtDB.Mul(B.T(), DB)
		BtDB.Scale(w*math.Abs(detJ)*6, BtDB) // multiply by 6 because w=1/24 and tet vol factor is 1/6
		t.ke.Add(t.ke, BtDB)
	}
}

// ---------- Element interface ----------

func (t *Tet10) GetTangentStiffness() *mat.Dense { return t.ke }

// GetResistingForce returns Ke·ue (internal nodal force vector).
func (t *Tet10) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(30, nil)
	f.MulVec(t.ke, mat.NewVecDense(30, t.ue[:]))
	return f
}

func (t *Tet10) NodeIDs() []int       { return t.Nds[:] }
func (t *Tet10) NumDOF() int          { return 30 }
func (t *Tet10) DOFPerNode() int      { return 3 }
func (t *Tet10) DOFTypes() []dof.Type { return dof.Translational3D(10) }

// Update stores the element displacements for post-processing.
func (t *Tet10) Update(disp []float64) error { copy(t.ue[:], disp); return nil }

func (t *Tet10) CommitState() error   { return nil }
func (t *Tet10) RevertToStart() error { t.ue = [30]float64{}; return nil }

// StressCentroid returns the Cauchy stress at the element centroid (L1=L2=L3=L4=1/4)
// in Voigt notation [sxx, syy, szz, txy, tyz, txz].
func (t *Tet10) StressCentroid() [6]float64 {
	const ndof = 30
	D := t.Mat.GetTangent()

	xi, eta, zeta := 0.25, 0.25, 0.25
	L1 := 1 - xi - eta - zeta // = 0.25
	L2, L3, L4 := xi, eta, zeta

	dNnat := mat.NewDense(3, 10, nil)
	// Corner nodes (all zero at centroid since 4Li-1=0)
	_ = L1
	_ = L2
	_ = L3
	_ = L4
	// Midside node 4 (L1-L2):  dN/dξ=4(L1-L2), dN/dη=-4L2, dN/dζ=-4L2
	dNnat.Set(0, 4, 4*(L1-L2))
	dNnat.Set(1, 4, -4*L2)
	dNnat.Set(2, 4, -4*L2)
	// Midside node 5 (L2-L3):  dN/dξ=4L3, dN/dη=4L2
	dNnat.Set(0, 5, 4*L3)
	dNnat.Set(1, 5, 4*L2)
	// Midside node 6 (L1-L3):  dN/dξ=-4L3, dN/dη=4(L1-L3), dN/dζ=-4L3
	dNnat.Set(0, 6, -4*L3)
	dNnat.Set(1, 6, 4*(L1-L3))
	dNnat.Set(2, 6, -4*L3)
	// Midside node 7 (L1-L4):  dN/dξ=-4L4, dN/dη=-4L4, dN/dζ=4(L1-L4)
	dNnat.Set(0, 7, -4*L4)
	dNnat.Set(1, 7, -4*L4)
	dNnat.Set(2, 7, 4*(L1-L4))
	// Midside node 8 (L2-L4):  dN/dξ=4L4, dN/dζ=4L2
	dNnat.Set(0, 8, 4*L4)
	dNnat.Set(2, 8, 4*L2)
	// Midside node 9 (L3-L4):  dN/dη=4L4, dN/dζ=4L3
	dNnat.Set(1, 9, 4*L4)
	dNnat.Set(2, 9, 4*L3)

	X := mat.NewDense(10, 3, nil)
	for i := 0; i < 10; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, t.Coords[i][j])
		}
	}
	J := mat.NewDense(3, 3, nil)
	J.Mul(dNnat, X)
	var Jinv mat.Dense
	Jinv.Inverse(J)
	dN := mat.NewDense(3, 10, nil)
	dN.Mul(&Jinv, dNnat)

	B := mat.NewDense(6, ndof, nil)
	for n := 0; n < 10; n++ {
		dx, dy, dz := dN.At(0, n), dN.At(1, n), dN.At(2, n)
		c := 3 * n
		B.Set(0, c, dx)
		B.Set(1, c+1, dy)
		B.Set(2, c+2, dz)
		B.Set(3, c, dy)
		B.Set(3, c+1, dx)
		B.Set(4, c+1, dz)
		B.Set(4, c+2, dy)
		B.Set(5, c, dz)
		B.Set(5, c+2, dx)
	}

	Bu := mat.NewVecDense(6, nil)
	Bu.MulVec(B, mat.NewVecDense(ndof, t.ue[:]))
	sigma := mat.NewVecDense(6, nil)
	sigma.MulVec(D, Bu)
	var s [6]float64
	for i := range s {
		s[i] = sigma.AtVec(i)
	}
	return s
}
