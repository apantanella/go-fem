package solid

import (
	"math"

	"go-fem/dof"
	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

// hex8Ref holds the reference coordinates (ξ, η, ζ) ∈ {-1,+1}³
// for the standard 8-node hexahedron.
var hex8Ref = [8][3]float64{
	{-1, -1, -1}, // 0
	{+1, -1, -1}, // 1
	{+1, +1, -1}, // 2
	{-1, +1, -1}, // 3
	{-1, -1, +1}, // 4
	{+1, -1, +1}, // 5
	{+1, +1, +1}, // 6
	{-1, +1, +1}, // 7
}

// Hexa8 is an 8-node trilinear hexahedron with 2×2×2 full Gauss integration.
type Hexa8 struct {
	ID     int
	Nds    [8]int        // global node IDs
	Coords [8][3]float64 // nodal coordinates
	Mat    material.Material3D

	ke *mat.Dense  // 24×24 stiffness
	ue [24]float64 // element displacements (set by Update)
}

// NewHexa8 creates and initialises an 8-node hexahedron.
func NewHexa8(id int, nodes [8]int, coords [8][3]float64, m material.Material3D) *Hexa8 {
	h := &Hexa8{ID: id, Nds: nodes, Coords: coords, Mat: m}
	h.formKe()
	return h
}

// formKe computes Ke using 2×2×2 Gauss quadrature (8 integration points).
func (h *Hexa8) formKe() {
	const ndof = 24
	D := h.Mat.GetTangent()
	h.ke = mat.NewDense(ndof, ndof, nil)

	// Gauss points: ±1/√3,  weight = 1.0 each
	gp := 1.0 / math.Sqrt(3.0)
	pts := [2]float64{-gp, gp}

	// Pre-allocate working matrices.
	dNnat := mat.NewDense(3, 8, nil)
	X := mat.NewDense(8, 3, nil)
	for i := 0; i < 8; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, h.Coords[i][j])
		}
	}

	J := mat.NewDense(3, 3, nil)
	dN := mat.NewDense(3, 8, nil)
	B := mat.NewDense(6, ndof, nil)
	DB := mat.NewDense(6, ndof, nil)
	BtDB := mat.NewDense(ndof, ndof, nil)
	var Jinv mat.Dense

	for _, xi := range pts {
		for _, eta := range pts {
			for _, zeta := range pts {
				// --- Shape function derivatives in natural coords (3×8) ---
				for i := 0; i < 8; i++ {
					si := hex8Ref[i][0]
					ei := hex8Ref[i][1]
					zi := hex8Ref[i][2]
					dNnat.Set(0, i, si*(1+ei*eta)*(1+zi*zeta)/8) // ∂Ni/∂ξ
					dNnat.Set(1, i, (1+si*xi)*ei*(1+zi*zeta)/8)  // ∂Ni/∂η
					dNnat.Set(2, i, (1+si*xi)*(1+ei*eta)*zi/8)   // ∂Ni/∂ζ
				}

				// --- Jacobian  J = dNnat · X  (3×3) ---
				J.Mul(dNnat, X)
				detJ := mat.Det(J)

				if err := Jinv.Inverse(J); err != nil {
					panic("hexa8: singular Jacobian – degenerate element")
				}

				// --- Physical derivatives  dN = J⁻¹ · dNnat  (3×8) ---
				dN.Mul(&Jinv, dNnat)

				// --- B matrix (6×24) ---
				B.Zero()
				for n := 0; n < 8; n++ {
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

				// --- Ke += |detJ| · Bᵀ · D · B ---
				DB.Mul(D, B)
				BtDB.Mul(B.T(), DB)
				BtDB.Scale(math.Abs(detJ), BtDB)
				h.ke.Add(h.ke, BtDB)
			}
		}
	}
}

// ---------- Element interface ----------

func (h *Hexa8) GetTangentStiffness() *mat.Dense { return h.ke }

// GetResistingForce returns Ke·ue (internal nodal force vector).
func (h *Hexa8) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(24, nil)
	f.MulVec(h.ke, mat.NewVecDense(24, h.ue[:]))
	return f
}

func (h *Hexa8) NodeIDs() []int { return h.Nds[:] }
func (h *Hexa8) NumDOF() int    { return 24 }

// Update stores the element displacements for post-processing.
func (h *Hexa8) Update(disp []float64) error { copy(h.ue[:], disp); return nil }

func (h *Hexa8) DOFPerNode() int      { return 3 }
func (h *Hexa8) DOFTypes() []dof.Type { return dof.Translational3D(8) }
func (h *Hexa8) CommitState() error   { return nil }
func (h *Hexa8) RevertToStart() error { h.ue = [24]float64{}; return nil }

// BodyForceLoad computes work-equivalent nodal forces due to a body force
// using 2×2×2 Gauss integration: f_i = ρ · ∫ N_i · g · dV.
func (h *Hexa8) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(24, nil)

	gp := 1.0 / math.Sqrt(3.0)
	pts := [2]float64{-gp, gp}

	X := mat.NewDense(8, 3, nil)
	for i := 0; i < 8; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, h.Coords[i][j])
		}
	}
	dNnat := mat.NewDense(3, 8, nil)
	J := mat.NewDense(3, 3, nil)

	for _, xi := range pts {
		for _, eta := range pts {
			for _, zeta := range pts {
				var N [8]float64
				for i := 0; i < 8; i++ {
					si := hex8Ref[i][0]
					ei := hex8Ref[i][1]
					zi := hex8Ref[i][2]
					N[i] = (1 + si*xi) * (1 + ei*eta) * (1 + zi*zeta) / 8
					dNnat.Set(0, i, si*(1+ei*eta)*(1+zi*zeta)/8)
					dNnat.Set(1, i, (1+si*xi)*ei*(1+zi*zeta)/8)
					dNnat.Set(2, i, (1+si*xi)*(1+ei*eta)*zi/8)
				}
				J.Mul(dNnat, X)
				detJ := math.Abs(mat.Det(J)) // weight = 1 for 2-pt Gauss

				scale := rho * detJ
				for n := 0; n < 8; n++ {
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
func (h *Hexa8) StressCentroid() [6]float64 {
	const ndof = 24
	D := h.Mat.GetTangent()

	// B matrix at centroid (ξ=η=ζ=0):
	// dNnat[0,i]=si/8, dNnat[1,i]=ei/8, dNnat[2,i]=zi/8
	dNnat := mat.NewDense(3, 8, nil)
	for i := 0; i < 8; i++ {
		dNnat.Set(0, i, hex8Ref[i][0]/8)
		dNnat.Set(1, i, hex8Ref[i][1]/8)
		dNnat.Set(2, i, hex8Ref[i][2]/8)
	}
	X := mat.NewDense(8, 3, nil)
	for i := 0; i < 8; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, h.Coords[i][j])
		}
	}
	J := mat.NewDense(3, 3, nil)
	J.Mul(dNnat, X)
	var Jinv mat.Dense
	Jinv.Inverse(J)
	dN := mat.NewDense(3, 8, nil)
	dN.Mul(&Jinv, dNnat)

	B := mat.NewDense(6, ndof, nil)
	for n := 0; n < 8; n++ {
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
	Bu.MulVec(B, mat.NewVecDense(ndof, h.ue[:]))
	sigma := mat.NewVecDense(6, nil)
	sigma.MulVec(D, Bu)
	var s [6]float64
	for i := range s {
		s[i] = sigma.AtVec(i)
	}
	return s
}
