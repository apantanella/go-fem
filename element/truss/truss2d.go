package truss

import (
	"math"

	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// Truss2D is a 2-node 2D truss (bar) element with axial stiffness only.
// 2 translational DOFs per node (UX, UY), 4 DOFs total.
type Truss2D struct {
	ID     int
	Nds    [2]int
	Coords [2][2]float64
	E      float64 // Young's modulus
	A      float64 // Cross-sectional area

	ke     *mat.Dense
	length float64
	cos    float64 // direction cosine (dx/L)
	sin    float64 // direction sine   (dy/L)
	ue     [4]float64
}

// NewTruss2D creates a 2D truss element.
func NewTruss2D(id int, nodes [2]int, coords [2][2]float64, e, a float64) *Truss2D {
	t := &Truss2D{ID: id, Nds: nodes, Coords: coords, E: e, A: a}
	t.computeGeometry()
	t.formKe()
	return t
}

func (t *Truss2D) computeGeometry() {
	dx := t.Coords[1][0] - t.Coords[0][0]
	dy := t.Coords[1][1] - t.Coords[0][1]
	t.length = math.Sqrt(dx*dx + dy*dy)
	t.cos = dx / t.length
	t.sin = dy / t.length
}

// formKe builds the 4×4 global stiffness matrix:
//
//	Ke = (AE/L) · Tᵀ·[1,-1;-1,1]·T
//
// where T = [c, s] maps global DOFs to the axial direction.
func (t *Truss2D) formKe() {
	k := t.A * t.E / t.length
	c, s := t.cos, t.sin

	t.ke = mat.NewDense(4, 4, nil)
	cc := k * c * c
	cs := k * c * s
	ss := k * s * s

	t.ke.Set(0, 0, cc)
	t.ke.Set(0, 1, cs)
	t.ke.Set(0, 2, -cc)
	t.ke.Set(0, 3, -cs)

	t.ke.Set(1, 0, cs)
	t.ke.Set(1, 1, ss)
	t.ke.Set(1, 2, -cs)
	t.ke.Set(1, 3, -ss)

	t.ke.Set(2, 0, -cc)
	t.ke.Set(2, 1, -cs)
	t.ke.Set(2, 2, cc)
	t.ke.Set(2, 3, cs)

	t.ke.Set(3, 0, -cs)
	t.ke.Set(3, 1, -ss)
	t.ke.Set(3, 2, cs)
	t.ke.Set(3, 3, ss)
}

// ---------- Element interface ----------

func (t *Truss2D) GetTangentStiffness() *mat.Dense { return t.ke }

func (t *Truss2D) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(4, nil)
	f.MulVec(t.ke, mat.NewVecDense(4, t.ue[:]))
	return f
}

func (t *Truss2D) NodeIDs() []int       { return t.Nds[:] }
func (t *Truss2D) NumDOF() int          { return 4 }
func (t *Truss2D) DOFPerNode() int      { return 2 }
func (t *Truss2D) DOFTypes() []dof.Type { return dof.Translational2D(2) }

func (t *Truss2D) Update(disp []float64) error {
	copy(t.ue[:], disp)
	return nil
}

func (t *Truss2D) CommitState() error   { return nil }
func (t *Truss2D) RevertToStart() error { t.ue = [4]float64{}; return nil }

// Length returns the element length.
func (t *Truss2D) Length() float64 { return t.length }

// AxialForce returns the axial force N = EA/L · Δu_axial (positive = tension).
func (t *Truss2D) AxialForce() float64 {
	du := [2]float64{t.ue[2] - t.ue[0], t.ue[3] - t.ue[1]}
	return t.E * t.A / t.length * (t.cos*du[0] + t.sin*du[1])
}

// AxialStress returns the axial normal stress σ = N/A.
func (t *Truss2D) AxialStress() float64 { return t.AxialForce() / t.A }
