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

// SpringForce returns [Fx, Fy, Fz, Mx, My, Mz] — spring force/moment (tension/CCW positive).
func (z *ZeroLength) SpringForce() [6]float64 {
	var f [6]float64
	for i := 0; i < 6; i++ {
		f[i] = z.Springs[i] * (z.ue[6+i] - z.ue[i])
	}
	return f
}

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

// SpringForce returns [Fx, Fy, Fz] — spring force (tension positive).
func (z *ZeroLength3DOF) SpringForce() [3]float64 {
	var f [3]float64
	for i := 0; i < 3; i++ {
		f[i] = z.Springs[i] * (z.ue[3+i] - z.ue[i])
	}
	return f
}

// ── ZeroLength2D ─────────────────────────────────────────────────────────────

// ZeroLength2D is a 2-node spring for 2D problems (UX, UY per node).
type ZeroLength2D struct {
	ID      int
	Nds     [2]int
	Springs [2]float64 // [kUX, kUY]

	ke *mat.Dense
	ue [4]float64
}

// NewZeroLength2D creates a 2D zero-length spring with translational DOFs.
func NewZeroLength2D(id int, nodes [2]int, springs [2]float64) *ZeroLength2D {
	z := &ZeroLength2D{ID: id, Nds: nodes, Springs: springs}
	z.formKe()
	return z
}

func (z *ZeroLength2D) formKe() {
	z.ke = mat.NewDense(4, 4, nil)
	for i := 0; i < 2; i++ {
		k := z.Springs[i]
		z.ke.Set(i, i, k)
		z.ke.Set(i+2, i+2, k)
		z.ke.Set(i, i+2, -k)
		z.ke.Set(i+2, i, -k)
	}
}

func (z *ZeroLength2D) GetTangentStiffness() *mat.Dense { return z.ke }

func (z *ZeroLength2D) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(4, nil)
	f.MulVec(z.ke, mat.NewVecDense(4, z.ue[:]))
	return f
}

func (z *ZeroLength2D) NodeIDs() []int       { return z.Nds[:] }
func (z *ZeroLength2D) NumDOF() int          { return 4 }
func (z *ZeroLength2D) DOFPerNode() int      { return 2 }
func (z *ZeroLength2D) DOFTypes() []dof.Type { return dof.Translational2D(2) }

func (z *ZeroLength2D) Update(disp []float64) error { copy(z.ue[:], disp); return nil }
func (z *ZeroLength2D) CommitState() error          { return nil }
func (z *ZeroLength2D) RevertToStart() error        { z.ue = [4]float64{}; return nil }

// SpringForce returns [Fx, Fy] — spring force (tension positive).
func (z *ZeroLength2D) SpringForce() [2]float64 {
	return [2]float64{
		z.Springs[0] * (z.ue[2] - z.ue[0]),
		z.Springs[1] * (z.ue[3] - z.ue[1]),
	}
}

// ── ZeroLength2DFrame ─────────────────────────────────────────────────────────

// ZeroLength2DFrame is a 2-node spring for plane-frame problems (UX, UY, RZ per node).
type ZeroLength2DFrame struct {
	ID      int
	Nds     [2]int
	Springs [3]float64 // [kUX, kUY, kRZ]

	ke *mat.Dense
	ue [6]float64
}

// NewZeroLength2DFrame creates a 2D zero-length spring with UX, UY, RZ DOFs.
func NewZeroLength2DFrame(id int, nodes [2]int, springs [3]float64) *ZeroLength2DFrame {
	z := &ZeroLength2DFrame{ID: id, Nds: nodes, Springs: springs}
	z.formKe()
	return z
}

func (z *ZeroLength2DFrame) formKe() {
	z.ke = mat.NewDense(6, 6, nil)
	for i := 0; i < 3; i++ {
		k := z.Springs[i]
		z.ke.Set(i, i, k)
		z.ke.Set(i+3, i+3, k)
		z.ke.Set(i, i+3, -k)
		z.ke.Set(i+3, i, -k)
	}
}

func (z *ZeroLength2DFrame) GetTangentStiffness() *mat.Dense { return z.ke }

func (z *ZeroLength2DFrame) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(6, nil)
	f.MulVec(z.ke, mat.NewVecDense(6, z.ue[:]))
	return f
}

func (z *ZeroLength2DFrame) NodeIDs() []int       { return z.Nds[:] }
func (z *ZeroLength2DFrame) NumDOF() int          { return 6 }
func (z *ZeroLength2DFrame) DOFPerNode() int      { return 3 }
func (z *ZeroLength2DFrame) DOFTypes() []dof.Type { return dof.PlaneFrame(2) }

func (z *ZeroLength2DFrame) Update(disp []float64) error { copy(z.ue[:], disp); return nil }
func (z *ZeroLength2DFrame) CommitState() error          { return nil }
func (z *ZeroLength2DFrame) RevertToStart() error        { z.ue = [6]float64{}; return nil }

// SpringForce returns [Fx, Fy, Mz] — spring force/moment (tension/CCW positive).
func (z *ZeroLength2DFrame) SpringForce() [3]float64 {
	return [3]float64{
		z.Springs[0] * (z.ue[3] - z.ue[0]),
		z.Springs[1] * (z.ue[4] - z.ue[1]),
		z.Springs[2] * (z.ue[5] - z.ue[2]),
	}
}
