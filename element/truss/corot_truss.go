package truss

import (
	"math"

	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// CorotTruss is a corotational 3D truss element for geometric nonlinearity.
// Uses updated Lagrangian formulation: geometry is tracked in the deformed configuration.
// For linear analysis it reduces to Truss3D.
type CorotTruss struct {
	ID     int
	Nds    [2]int
	Coords [2][3]float64
	E      float64
	A      float64

	ke   *mat.Dense
	re   *mat.VecDense
	L0   float64    // initial length
	Ln   float64    // current length
	cos0 [3]float64 // initial direction cosines
	cosn [3]float64 // current direction cosines

	strain float64 // engineering strain
	stress float64 // axial stress
}

// NewCorotTruss creates a corotational truss element.
func NewCorotTruss(id int, nodes [2]int, coords [2][3]float64, e, a float64) *CorotTruss {
	t := &CorotTruss{ID: id, Nds: nodes, Coords: coords, E: e, A: a}
	t.computeInitialGeometry()
	t.Ln = t.L0
	t.cosn = t.cos0
	t.formKe()
	t.re = mat.NewVecDense(6, nil)
	return t
}

func (t *CorotTruss) computeInitialGeometry() {
	dx := t.Coords[1][0] - t.Coords[0][0]
	dy := t.Coords[1][1] - t.Coords[0][1]
	dz := t.Coords[1][2] - t.Coords[0][2]
	t.L0 = math.Sqrt(dx*dx + dy*dy + dz*dz)
	t.cos0[0] = dx / t.L0
	t.cos0[1] = dy / t.L0
	t.cos0[2] = dz / t.L0
}

// Update recomputes the element state in the deformed configuration.
func (t *CorotTruss) Update(disp []float64) error {
	// Deformed configuration
	dx := (t.Coords[1][0] + disp[3]) - (t.Coords[0][0] + disp[0])
	dy := (t.Coords[1][1] + disp[4]) - (t.Coords[0][1] + disp[1])
	dz := (t.Coords[1][2] + disp[5]) - (t.Coords[0][2] + disp[2])
	t.Ln = math.Sqrt(dx*dx + dy*dy + dz*dz)
	t.cosn[0] = dx / t.Ln
	t.cosn[1] = dy / t.Ln
	t.cosn[2] = dz / t.Ln

	// Engineering strain and stress (linear elastic)
	t.strain = (t.Ln - t.L0) / t.L0
	t.stress = t.E * t.strain

	t.formKe()
	t.formRe()
	return nil
}

// formKe builds the corotational tangent stiffness:
//
//	K = (EA/Ln)·c⊗c + (N/Ln)·(I - c⊗c)
//
// where c is the current direction cosine vector and N = A·σ is the axial force.
func (t *CorotTruss) formKe() {
	N := t.A * t.stress
	kmat := t.A * t.E / t.Ln
	kgeo := N / t.Ln

	t.ke = mat.NewDense(6, 6, nil)
	c := t.cosn
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			delta := 0.0
			if i == j {
				delta = 1.0
			}
			v := kmat*c[i]*c[j] + kgeo*(delta-c[i]*c[j])
			t.ke.Set(i, j, v)
			t.ke.Set(i+3, j+3, v)
			t.ke.Set(i, j+3, -v)
			t.ke.Set(i+3, j, -v)
		}
	}
}

// formRe builds the internal resisting force vector.
func (t *CorotTruss) formRe() {
	N := t.A * t.stress
	t.re = mat.NewVecDense(6, nil)
	c := t.cosn
	for i := 0; i < 3; i++ {
		t.re.SetVec(i, -N*c[i])
		t.re.SetVec(i+3, N*c[i])
	}
}

// ---------- Element interface ----------

func (t *CorotTruss) GetTangentStiffness() *mat.Dense { return t.ke }
func (t *CorotTruss) GetResistingForce() *mat.VecDense {
	if t.re == nil {
		return mat.NewVecDense(6, nil)
	}
	return t.re
}
func (t *CorotTruss) NodeIDs() []int       { return t.Nds[:] }
func (t *CorotTruss) NumDOF() int          { return 6 }
func (t *CorotTruss) DOFPerNode() int      { return 3 }
func (t *CorotTruss) DOFTypes() []dof.Type { return dof.Translational3D(2) }
func (t *CorotTruss) CommitState() error   { return nil }
func (t *CorotTruss) RevertToStart() error {
	t.Ln = t.L0
	t.cosn = t.cos0
	t.strain = 0
	t.stress = 0
	t.formKe()
	t.re = mat.NewVecDense(6, nil)
	return nil
}
