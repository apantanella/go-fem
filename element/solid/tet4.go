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

// BodyForceLoad computes work-equivalent nodal forces due to a body force.
// For Tet4 the result is exact: f_i = ρ · g · V/4 at each of the 4 nodes.
func (t *Tet4) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(12, nil)
	q := rho * t.vol / 4.0
	for n := 0; n < 4; n++ {
		f.SetVec(3*n, q*g[0])
		f.SetVec(3*n+1, q*g[1])
		f.SetVec(3*n+2, q*g[2])
	}
	return f
}

// GetMassMatrix returns the 12×12 consistent mass matrix (analytical, exact).
// For a linear tetrahedron: Mₑ[3n+k, 3m+k] = ρV/20·(1 + δₙₘ)  (n,m = nodes 0..3).
func (t *Tet4) GetMassMatrix(rho float64) *mat.Dense {
	me := mat.NewDense(12, 12, nil)
	diag := rho * t.vol / 10.0 // 2/20
	offD := rho * t.vol / 20.0 // 1/20
	for n := 0; n < 4; n++ {
		for m := 0; m < 4; m++ {
			v := offD
			if n == m {
				v = diag
			}
			for k := 0; k < 3; k++ {
				me.Set(3*n+k, 3*m+k, v)
			}
		}
	}
	return me
}

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

// Tresca computes the Tresca equivalent stress (σ₁ − σ₃, max minus min principal
// stress) from a Voigt stress vector [sxx, syy, szz, txy, tyz, txz].
//
// Principal stresses are obtained analytically via the trigonometric (Lode-angle)
// method for the 3×3 symmetric stress tensor. Works for both 3D and plane-stress
// cases (where szz = tyz = txz = 0).
func Tresca(s [6]float64) float64 {
	sxx, syy, szz := s[0], s[1], s[2]
	txy, tyz, txz := s[3], s[4], s[5]

	p := (sxx + syy + szz) / 3.0

	// Deviatoric components
	dx, dy, dz := sxx-p, syy-p, szz-p

	// Second deviatoric invariant J₂ = ½ tr(s_dev²)
	j2 := 0.5*(dx*dx+dy*dy+dz*dz) + txy*txy + tyz*tyz + txz*txz

	if j2 < 1e-30 {
		// Hydrostatic state — all principal stresses equal p, Tresca = 0
		return 0
	}

	// Third deviatoric invariant J₃ = det(s_dev)
	j3 := dx*(dy*dz-tyz*tyz) - txy*(txy*dz-tyz*txz) + txz*(txy*tyz-dy*txz)

	// Lode angle θ ∈ [0, π/3]
	arg := 3.0 * math.Sqrt(3.0) * j3 / (2.0 * math.Pow(j2, 1.5))
	if arg > 1 {
		arg = 1
	} else if arg < -1 {
		arg = -1
	}
	theta := math.Acos(arg) / 3.0

	r := 2.0 * math.Sqrt(j2/3.0)
	s1 := p + r*math.Cos(theta)
	s2 := p + r*math.Cos(theta-2.0*math.Pi/3.0)
	s3 := p + r*math.Cos(theta+2.0*math.Pi/3.0)

	maxS := math.Max(s1, math.Max(s2, s3))
	minS := math.Min(s1, math.Min(s2, s3))
	return maxS - minS
}
