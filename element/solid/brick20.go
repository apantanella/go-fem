package solid

import (
	"math"

	"go-fem/dof"
	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

// Brick20 is a 20-node serendipity hexahedron with 3×3×3 full Gauss integration.
// 3 DOFs per node, 60 DOFs total.
type Brick20 struct {
	ID     int
	Nds    [20]int
	Coords [20][3]float64
	Mat    material.Material3D

	ke *mat.Dense
	ue [60]float64 // element displacements (set by Update)
}

// brick20Ref holds the reference coordinates for the 20-node serendipity element.
// Nodes 0-7: corners (same as Hex8)
// Nodes 8-19: midside nodes on edges.
var brick20Ref = [20][3]float64{
	// Corners
	{-1, -1, -1}, // 0
	{+1, -1, -1}, // 1
	{+1, +1, -1}, // 2
	{-1, +1, -1}, // 3
	{-1, -1, +1}, // 4
	{+1, -1, +1}, // 5
	{+1, +1, +1}, // 6
	{-1, +1, +1}, // 7
	// Midside (ξ-edges, ζ=-1)
	{0, -1, -1}, // 8  (0-1)
	{+1, 0, -1}, // 9  (1-2)
	{0, +1, -1}, // 10 (2-3)
	{-1, 0, -1}, // 11 (3-0)
	// Midside (ξ-edges, ζ=+1)
	{0, -1, +1}, // 12 (4-5)
	{+1, 0, +1}, // 13 (5-6)
	{0, +1, +1}, // 14 (6-7)
	{-1, 0, +1}, // 15 (7-4)
	// Midside (ζ-edges)
	{-1, -1, 0}, // 16 (0-4)
	{+1, -1, 0}, // 17 (1-5)
	{+1, +1, 0}, // 18 (2-6)
	{-1, +1, 0}, // 19 (3-7)
}

// NewBrick20 creates a 20-node serendipity hexahedron.
func NewBrick20(id int, nodes [20]int, coords [20][3]float64, m material.Material3D) *Brick20 {
	b := &Brick20{ID: id, Nds: nodes, Coords: coords, Mat: m}
	b.formKe()
	return b
}

// formKe computes the 60×60 stiffness using 3×3×3 = 27 Gauss points.
func (b *Brick20) formKe() {
	const ndof = 60
	D := b.Mat.GetTangent()
	b.ke = mat.NewDense(ndof, ndof, nil)

	// 3-point 1D Gauss quadrature
	g := math.Sqrt(3.0 / 5.0)
	gpts := [3]float64{-g, 0, g}
	gwts := [3]float64{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0}

	X := mat.NewDense(20, 3, nil)
	for i := 0; i < 20; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, b.Coords[i][j])
		}
	}

	dNnat := mat.NewDense(3, 20, nil)
	J := mat.NewDense(3, 3, nil)
	dN := mat.NewDense(3, 20, nil)
	Bmat := mat.NewDense(6, ndof, nil)
	DB := mat.NewDense(6, ndof, nil)
	BtDB := mat.NewDense(ndof, ndof, nil)
	var Jinv mat.Dense

	for ix := 0; ix < 3; ix++ {
		for iy := 0; iy < 3; iy++ {
			for iz := 0; iz < 3; iz++ {
				xi := gpts[ix]
				eta := gpts[iy]
				zeta := gpts[iz]
				w := gwts[ix] * gwts[iy] * gwts[iz]

				// Compute shape function derivatives
				b.shapeDeriv(xi, eta, zeta, dNnat)

				// Jacobian J = dNnat · X (3×3)
				J.Mul(dNnat, X)
				detJ := mat.Det(J)

				if err := Jinv.Inverse(J); err != nil {
					panic("brick20: singular Jacobian – degenerate element")
				}

				// Physical derivatives
				dN.Mul(&Jinv, dNnat)

				// Build B matrix (6×60)
				Bmat.Zero()
				for n := 0; n < 20; n++ {
					dx := dN.At(0, n)
					dy := dN.At(1, n)
					dz := dN.At(2, n)
					c := 3 * n

					Bmat.Set(0, c, dx)
					Bmat.Set(1, c+1, dy)
					Bmat.Set(2, c+2, dz)
					Bmat.Set(3, c, dy)
					Bmat.Set(3, c+1, dx)
					Bmat.Set(4, c+1, dz)
					Bmat.Set(4, c+2, dy)
					Bmat.Set(5, c, dz)
					Bmat.Set(5, c+2, dx)
				}

				// Ke += w · |detJ| · Bᵀ · D · B
				DB.Mul(D, Bmat)
				BtDB.Mul(Bmat.T(), DB)
				BtDB.Scale(w*math.Abs(detJ), BtDB)
				b.ke.Add(b.ke, BtDB)
			}
		}
	}
}

// shapeDeriv computes ∂Ni/∂(ξ,η,ζ) for all 20 nodes at a given natural coordinate.
func (b *Brick20) shapeDeriv(xi, eta, zeta float64, dN *mat.Dense) {
	for i := 0; i < 20; i++ {
		si := brick20Ref[i][0]
		ei := brick20Ref[i][1]
		zi := brick20Ref[i][2]

		isCorner := math.Abs(si) > 0.5 && math.Abs(ei) > 0.5 && math.Abs(zi) > 0.5
		isXiMid := math.Abs(si) < 0.5
		isEtaMid := math.Abs(ei) < 0.5
		isZetaMid := math.Abs(zi) < 0.5

		if isCorner {
			// Corner node: Ni = (1/8)(1+si·ξ)(1+ei·η)(1+zi·ζ)(si·ξ+ei·η+zi·ζ-2)
			f := 1 + si*xi
			g := 1 + ei*eta
			h := 1 + zi*zeta
			p := si*xi + ei*eta + zi*zeta - 2

			dN.Set(0, i, 0.125*si*g*h*(2*si*xi+ei*eta+zi*zeta-1))
			dN.Set(1, i, 0.125*f*ei*h*(si*xi+2*ei*eta+zi*zeta-1))
			dN.Set(2, i, 0.125*f*g*zi*(si*xi+ei*eta+2*zi*zeta-1))
			_ = p // derivation uses product rule giving the simplified form above
		} else if isXiMid {
			// ξ-midside: Ni = (1/4)(1-ξ²)(1+ei·η)(1+zi·ζ)
			g := 1 + ei*eta
			h := 1 + zi*zeta
			dN.Set(0, i, -0.5*xi*g*h)
			dN.Set(1, i, 0.25*(1-xi*xi)*ei*h)
			dN.Set(2, i, 0.25*(1-xi*xi)*g*zi)
		} else if isEtaMid {
			// η-midside: Ni = (1/4)(1+si·ξ)(1-η²)(1+zi·ζ)
			f := 1 + si*xi
			h := 1 + zi*zeta
			dN.Set(0, i, 0.25*si*(1-eta*eta)*h)
			dN.Set(1, i, -0.5*eta*f*h)
			dN.Set(2, i, 0.25*f*(1-eta*eta)*zi)
		} else if isZetaMid {
			// ζ-midside: Ni = (1/4)(1+si·ξ)(1+ei·η)(1-ζ²)
			f := 1 + si*xi
			g := 1 + ei*eta
			dN.Set(0, i, 0.25*si*g*(1-zeta*zeta))
			dN.Set(1, i, 0.25*f*ei*(1-zeta*zeta))
			dN.Set(2, i, -0.5*zeta*f*g)
		}
	}
}

// ---------- Element interface ----------

func (b *Brick20) GetTangentStiffness() *mat.Dense { return b.ke }

// GetResistingForce returns Ke·ue (internal nodal force vector).
func (b *Brick20) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(60, nil)
	f.MulVec(b.ke, mat.NewVecDense(60, b.ue[:]))
	return f
}

func (b *Brick20) NodeIDs() []int       { return b.Nds[:] }
func (b *Brick20) NumDOF() int          { return 60 }
func (b *Brick20) DOFPerNode() int      { return 3 }
func (b *Brick20) DOFTypes() []dof.Type { return dof.Translational3D(20) }

// Update stores the element displacements for post-processing.
func (b *Brick20) Update(disp []float64) error { copy(b.ue[:], disp); return nil }

func (b *Brick20) CommitState() error   { return nil }
func (b *Brick20) RevertToStart() error { b.ue = [60]float64{}; return nil }

// BodyForceLoad computes work-equivalent nodal forces due to a body force
// using the same 3×3×3 Gauss quadrature as formKe: f_i = ρ·∫N_i·g·dV.
func (b *Brick20) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(60, nil)

	gr := math.Sqrt(3.0 / 5.0)
	gpts := [3]float64{-gr, 0, gr}
	gwts := [3]float64{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0}

	X := mat.NewDense(20, 3, nil)
	for i := 0; i < 20; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, b.Coords[i][j])
		}
	}

	dNnat := mat.NewDense(3, 20, nil)
	J := mat.NewDense(3, 3, nil)

	for ix := 0; ix < 3; ix++ {
		for iy := 0; iy < 3; iy++ {
			for iz := 0; iz < 3; iz++ {
				xi := gpts[ix]
				eta := gpts[iy]
				zeta := gpts[iz]
				w := gwts[ix] * gwts[iy] * gwts[iz]

				var N [20]float64
				for i := 0; i < 20; i++ {
					si := brick20Ref[i][0]
					ei := brick20Ref[i][1]
					zi := brick20Ref[i][2]
					if math.Abs(si) > 0.5 && math.Abs(ei) > 0.5 && math.Abs(zi) > 0.5 {
						N[i] = 0.125 * (1 + si*xi) * (1 + ei*eta) * (1 + zi*zeta) * (si*xi + ei*eta + zi*zeta - 2)
					} else if math.Abs(si) < 0.5 {
						N[i] = 0.25 * (1 - xi*xi) * (1 + ei*eta) * (1 + zi*zeta)
					} else if math.Abs(ei) < 0.5 {
						N[i] = 0.25 * (1 + si*xi) * (1 - eta*eta) * (1 + zi*zeta)
					} else {
						N[i] = 0.25 * (1 + si*xi) * (1 + ei*eta) * (1 - zeta*zeta)
					}
				}

				b.shapeDeriv(xi, eta, zeta, dNnat)
				J.Mul(dNnat, X)
				detJ := math.Abs(mat.Det(J))
				scale := rho * w * detJ
				for n := 0; n < 20; n++ {
					f.SetVec(3*n, f.AtVec(3*n)+scale*N[n]*g[0])
					f.SetVec(3*n+1, f.AtVec(3*n+1)+scale*N[n]*g[1])
					f.SetVec(3*n+2, f.AtVec(3*n+2)+scale*N[n]*g[2])
				}
			}
		}
	}
	return f
}

// StressCentroid returns the Cauchy stress at the element centroid (ξ=η=ζ=0)
// in Voigt notation [sxx, syy, szz, txy, tyz, txz].
func (b *Brick20) StressCentroid() [6]float64 {
	const ndof = 60
	D := b.Mat.GetTangent()

	dNnat := mat.NewDense(3, 20, nil)
	b.shapeDeriv(0, 0, 0, dNnat)

	X := mat.NewDense(20, 3, nil)
	for i := 0; i < 20; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, b.Coords[i][j])
		}
	}
	J := mat.NewDense(3, 3, nil)
	J.Mul(dNnat, X)
	var Jinv mat.Dense
	Jinv.Inverse(J)
	dN := mat.NewDense(3, 20, nil)
	dN.Mul(&Jinv, dNnat)

	Bmat := mat.NewDense(6, ndof, nil)
	for n := 0; n < 20; n++ {
		dx, dy, dz := dN.At(0, n), dN.At(1, n), dN.At(2, n)
		c := 3 * n
		Bmat.Set(0, c, dx)
		Bmat.Set(1, c+1, dy)
		Bmat.Set(2, c+2, dz)
		Bmat.Set(3, c, dy)
		Bmat.Set(3, c+1, dx)
		Bmat.Set(4, c+1, dz)
		Bmat.Set(4, c+2, dy)
		Bmat.Set(5, c, dz)
		Bmat.Set(5, c+2, dx)
	}

	Bu := mat.NewVecDense(6, nil)
	Bu.MulVec(Bmat, mat.NewVecDense(ndof, b.ue[:]))
	sigma := mat.NewVecDense(6, nil)
	sigma.MulVec(D, Bu)
	var s [6]float64
	for i := range s {
		s[i] = sigma.AtVec(i)
	}
	return s
}
