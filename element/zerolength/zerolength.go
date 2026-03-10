// Package zerolength implements spring/connector elements with zero physical length.
package zerolength

import (
	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// ZeroLength is a 2-node spring element connecting coincident nodes.
// Each node has 6 DOFs (UX, UY, UZ, RX, RY, RZ), 12 DOFs total.
// Stiffness is defined by 6 spring constants (one per DOF direction).
type ZeroLength struct {
	ID      int
	Nds     [2]int
	Springs [6]float64 // [kUX, kUY, kUZ, kRX, kRY, kRZ]

	ke *mat.Dense
	ue [12]float64 // element displacements (set by Update)
}

// NewZeroLength creates a zero-length spring element.
// springs[i] is the stiffness for DOF i (0=UX, 1=UY, 2=UZ, 3=RX, 4=RY, 5=RZ).
func NewZeroLength(id int, nodes [2]int, springs [6]float64) *ZeroLength {
	z := &ZeroLength{ID: id, Nds: nodes, Springs: springs}
	z.formKe()
	return z
}

// formKe builds the 12×12 stiffness:
//
//	K = [ Ks  -Ks ]
//	    [-Ks   Ks ]
//
// where Ks = diag(springs).
func (z *ZeroLength) formKe() {
	z.ke = mat.NewDense(12, 12, nil)
	for i := 0; i < 6; i++ {
		k := z.Springs[i]
		z.ke.Set(i, i, k)     //  Ks
		z.ke.Set(i+6, i+6, k) //  Ks
		z.ke.Set(i, i+6, -k)  // -Ks
		z.ke.Set(i+6, i, -k)  // -Ks
	}
}

// ---------- Element interface ----------

func (z *ZeroLength) GetTangentStiffness() *mat.Dense { return z.ke }

// GetResistingForce returns Ke·ue (spring reaction force vector).
func (z *ZeroLength) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(12, nil)
	f.MulVec(z.ke, mat.NewVecDense(12, z.ue[:]))
	return f
}

func (z *ZeroLength) NodeIDs() []int       { return z.Nds[:] }
func (z *ZeroLength) NumDOF() int          { return 12 }
func (z *ZeroLength) DOFPerNode() int      { return 6 }
func (z *ZeroLength) DOFTypes() []dof.Type { return dof.Full6D(2) }

// Update stores the element displacements for post-processing.
func (z *ZeroLength) Update(disp []float64) error { copy(z.ue[:], disp); return nil }

func (z *ZeroLength) CommitState() error   { return nil }
func (z *ZeroLength) RevertToStart() error { z.ue = [12]float64{}; return nil }

// ZeroLength3DOF is a simpler 2-node spring with only 3 translational DOFs per node.
type ZeroLength3DOF struct {
	ID      int
	Nds     [2]int
	Springs [3]float64 // [kUX, kUY, kUZ]

	ke *mat.Dense
	ue [6]float64 // element displacements (set by Update)
}

// NewZeroLength3DOF creates a zero-length spring with translational DOFs only.
func NewZeroLength3DOF(id int, nodes [2]int, springs [3]float64) *ZeroLength3DOF {
	z := &ZeroLength3DOF{ID: id, Nds: nodes, Springs: springs}
	z.formKe()
	return z
}

func (z *ZeroLength3DOF) formKe() {
	z.ke = mat.NewDense(6, 6, nil)
	for i := 0; i < 3; i++ {
		k := z.Springs[i]
		z.ke.Set(i, i, k)
		z.ke.Set(i+3, i+3, k)
		z.ke.Set(i, i+3, -k)
		z.ke.Set(i+3, i, -k)
	}
}

func (z *ZeroLength3DOF) GetTangentStiffness() *mat.Dense { return z.ke }

// GetResistingForce returns Ke·ue (spring reaction force vector).
func (z *ZeroLength3DOF) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(6, nil)
	f.MulVec(z.ke, mat.NewVecDense(6, z.ue[:]))
	return f
}

func (z *ZeroLength3DOF) NodeIDs() []int       { return z.Nds[:] }
func (z *ZeroLength3DOF) NumDOF() int          { return 6 }
func (z *ZeroLength3DOF) DOFPerNode() int      { return 3 }
func (z *ZeroLength3DOF) DOFTypes() []dof.Type { return dof.Translational3D(2) }

// Update stores the element displacements for post-processing.
func (z *ZeroLength3DOF) Update(disp []float64) error { copy(z.ue[:], disp); return nil }

func (z *ZeroLength3DOF) CommitState() error   { return nil }
func (z *ZeroLength3DOF) RevertToStart() error { z.ue = [6]float64{}; return nil }
