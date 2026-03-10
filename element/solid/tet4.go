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

	ke  *mat.Dense // 12×12 stiffness
	b   *mat.Dense // 6×12 strain-displacement
	vol float64    // element volume
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

func (t *Tet4) GetResistingForce() *mat.VecDense {
	// For linear analysis, not used (forces come from Ke·U).
	return mat.NewVecDense(12, nil)
}

func (t *Tet4) NodeIDs() []int { return t.Nds[:] }
func (t *Tet4) NumDOF() int    { return 12 }

func (t *Tet4) Update(_ []float64) error { return nil }
func (t *Tet4) DOFPerNode() int            { return 3 }
func (t *Tet4) DOFTypes() []dof.Type       { return dof.Translational3D(4) }
func (t *Tet4) CommitState() error         { return nil }
func (t *Tet4) RevertToStart() error       { return nil }

// Volume returns the element volume.
func (t *Tet4) Volume() float64 { return t.vol }

// B returns the strain-displacement matrix (6×12).
func (t *Tet4) B() *mat.Dense { return t.b }
