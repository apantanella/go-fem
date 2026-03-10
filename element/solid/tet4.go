package solid

import (
	"math"

	"go-fem/dof"
	"go-fem/material"

	"gonum.org/v1/gonum/mat"
)

// Tet4 is a 4-node linear tetrahedron (constant strain).
// Single Gauss point (exact integration for linear shape functions).
type Tet4 struct {
	ID     int
	Nds    [4]int        // global node IDs
	Coords [4][3]float64 // nodal coordinates (x, y, z)
	Mat    material.Material3D

	ke  *mat.Dense  // 12×12 stiffness
	b   *mat.Dense  // 6×12 strain-displacement
	vol float64     // element volume
	ue  [12]float64 // element displacements (set by Update)
}

// NewTet4 creates and initialises a 4-node tetrahedron.
func NewTet4(id int, nodes [4]int, coords [4][3]float64, m material.Material3D) *Tet4 {
	t := &Tet4{ID: id, Nds: nodes, Coords: coords, Mat: m}
	t.formB()
	t.formKe()
	return t
}

// formB computes the constant B matrix (6×12) and element volume.
//
// Jacobian convention:
//
//	J[a][b] = ∂x_b / ∂ξ_a   (a = natural dir, b = physical dir)
//
// Shape function derivatives in physical space:
//
//	∂N/∂x = J⁻¹ · ∂N/∂ξ
func (t *Tet4) formB() {
	// --- Jacobian (3×3) ---
	J := mat.NewDense(3, 3, nil)
	for a := 0; a < 3; a++ { // natural direction
		for b := 0; b < 3; b++ { // physical direction
			J.Set(a, b, t.Coords[a+1][b]-t.Coords[0][b])
		}
	}

	detJ := mat.Det(J)
	t.vol = math.Abs(detJ) / 6.0

	// --- Inverse Jacobian ---
	var Jinv mat.Dense
	if err := Jinv.Inverse(J); err != nil {
		panic("tet4: singular Jacobian – degenerate element")
	}

	// --- Shape function natural derivatives (3×4) ---
	// N0 = 1-ξ-η-ζ,  N1 = ξ,  N2 = η,  N3 = ζ
	dNnat := mat.NewDense(3, 4, []float64{
		-1, 1, 0, 0,
		-1, 0, 1, 0,
		-1, 0, 0, 1,
	})

	// --- Physical derivatives: dN = J⁻¹ · dNnat  (3×4) ---
	dN := mat.NewDense(3, 4, nil)
	dN.Mul(&Jinv, dNnat)

	// --- B matrix (6×12) ---
	// Voigt: [εxx, εyy, εzz, γxy, γyz, γxz]
	t.b = mat.NewDense(6, 12, nil)
	for n := 0; n < 4; n++ {
		dx := dN.At(0, n) // ∂Nn/∂x
		dy := dN.At(1, n) // ∂Nn/∂y
		dz := dN.At(2, n) // ∂Nn/∂z
		c := 3 * n

		t.b.Set(0, c, dx)   // εxx
		t.b.Set(1, c+1, dy) // εyy
		t.b.Set(2, c+2, dz) // εzz

		t.b.Set(3, c, dy)   // γxy = ∂u/∂y + ∂v/∂x
		t.b.Set(3, c+1, dx) //

		t.b.Set(4, c+1, dz) // γyz = ∂v/∂z + ∂w/∂y
		t.b.Set(4, c+2, dy) //

		t.b.Set(5, c, dz)   // γxz = ∂u/∂z + ∂w/∂x
		t.b.Set(5, c+2, dx) //
	}
}

// formKe computes Ke = V · Bᵀ · D · B.
func (t *Tet4) formKe() {
	D := t.Mat.GetTangent()

	// DB = D · B  (6×12)
	DB := mat.NewDense(6, 12, nil)
	DB.Mul(D, t.b)

	// Ke = Bᵀ · DB  (12×12), then scale by volume
	t.ke = mat.NewDense(12, 12, nil)
	t.ke.Mul(t.b.T(), DB)
	t.ke.Scale(t.vol, t.ke)
}

// ---------- Element interface ----------

func (t *Tet4) GetTangentStiffness() *mat.Dense { return t.ke }

// GetResistingForce returns Ke·ue (internal nodal force vector).
func (t *Tet4) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(12, nil)
	f.MulVec(t.ke, mat.NewVecDense(12, t.ue[:]))
	return f
}

func (t *Tet4) NodeIDs() []int { return t.Nds[:] }
func (t *Tet4) NumDOF() int    { return 12 }

// Update stores the element displacements for post-processing.
func (t *Tet4) Update(disp []float64) error { copy(t.ue[:], disp); return nil }

func (t *Tet4) DOFPerNode() int      { return 3 }
func (t *Tet4) DOFTypes() []dof.Type { return dof.Translational3D(4) }
func (t *Tet4) CommitState() error   { return nil }
func (t *Tet4) RevertToStart() error { t.ue = [12]float64{}; return nil }

// Volume returns the element volume.
func (t *Tet4) Volume() float64 { return t.vol }

// B returns the strain-displacement matrix (6×12).
func (t *Tet4) B() *mat.Dense { return t.b }

// StressCentroid returns the Cauchy stress vector at the element centroid
// in Voigt notation [sxx, syy, szz, txy, tyz, txz] = D·B·ue.
// For Tet4 the strain (and stress) is constant throughout the element.
func (t *Tet4) StressCentroid() [6]float64 {
	D := t.Mat.GetTangent()
	Bu := mat.NewVecDense(6, nil)
	Bu.MulVec(t.b, mat.NewVecDense(12, t.ue[:]))
	sigma := mat.NewVecDense(6, nil)
	sigma.MulVec(D, Bu)
	var s [6]float64
	for i := range s {
		s[i] = sigma.AtVec(i)
	}
	return s
}

// VonMises computes the von Mises equivalent stress from a Voigt stress vector
// [sxx, syy, szz, txy, tyz, txz].
func VonMises(s [6]float64) float64 {
	a, b, c := s[0]-s[1], s[1]-s[2], s[2]-s[0]
	return math.Sqrt(0.5 * (a*a + b*b + c*c + 6*(s[3]*s[3]+s[4]*s[4]+s[5]*s[5])))
}
