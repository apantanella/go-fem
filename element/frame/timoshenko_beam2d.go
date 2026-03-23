package frame

import (
	"math"

	"go-fem/dof"
	"go-fem/section"

	"gonum.org/v1/gonum/mat"
)

// TimoshenkoBeam2D is a 2-node 2D shear-deformable (Timoshenko) beam element
// for plane frames.
// 3 DOFs per node (UX, UY, RZ), 6 DOFs total — same layout as ElasticBeam2D.
//
// Compared to Euler-Bernoulli, the Timoshenko formulation adds a shear
// flexibility parameter:
//
//	Φ = 12·E·Iz / (G·Asy·L²)
//
// When Φ → 0 the stiffness reduces to the Euler-Bernoulli matrix.
type TimoshenkoBeam2D struct {
	ID     int
	Nds    [2]int
	Coords [2][2]float64
	E      float64 // Young's modulus
	G      float64 // Shear modulus
	Sec    section.BeamSection2D

	ke     *mat.Dense
	kl     *mat.Dense
	length float64
	cos    float64
	sin    float64
	ue     [6]float64
}

// NewTimoshenkoBeam2D creates a 2D Timoshenko beam element.
func NewTimoshenkoBeam2D(id int, nodes [2]int, coords [2][2]float64,
	e, g float64, sec section.BeamSection2D) *TimoshenkoBeam2D {

	b := &TimoshenkoBeam2D{
		ID:     id,
		Nds:    nodes,
		Coords: coords,
		E:      e,
		G:      g,
		Sec:    sec,
	}
	b.computeGeometry()
	b.formKe()
	return b
}

func (b *TimoshenkoBeam2D) computeGeometry() {
	dx := b.Coords[1][0] - b.Coords[0][0]
	dy := b.Coords[1][1] - b.Coords[0][1]
	b.length = math.Sqrt(dx*dx + dy*dy)
	b.cos = dx / b.length
	b.sin = dy / b.length
}

// formKe builds the 6×6 global stiffness: Ke = Tᵀ·Klocal·T
func (b *TimoshenkoBeam2D) formKe() {
	L := b.length
	E := b.E
	G := b.G
	A := b.Sec.A
	Iz := b.Sec.Iz

	const defaultKappa = 5.0 / 6.0
	Asy := b.Sec.Asy
	if Asy == 0 {
		Asy = defaultKappa * A
	}

	L2 := L * L
	L3 := L2 * L

	Phi := 12 * E * Iz / (G * Asy * L2)

	kl := mat.NewDense(6, 6, nil)

	// Axial (DOFs 0, 3)
	ea := E * A / L
	kl.Set(0, 0, ea)
	kl.Set(0, 3, -ea)
	kl.Set(3, 0, -ea)
	kl.Set(3, 3, ea)

	// Bending with shear flexibility (DOFs: v at 1,4; θz at 2,5)
	cz := E * Iz / ((1 + Phi) * L3)
	v1 := 12 * cz
	v2 := 6 * cz * L
	v3 := (4 + Phi) * cz * L2
	v4 := (2 - Phi) * cz * L2

	kl.Set(1, 1, v1)
	kl.Set(1, 2, v2)
	kl.Set(1, 4, -v1)
	kl.Set(1, 5, v2)

	kl.Set(2, 1, v2)
	kl.Set(2, 2, v3)
	kl.Set(2, 4, -v2)
	kl.Set(2, 5, v4)

	kl.Set(4, 1, -v1)
	kl.Set(4, 2, -v2)
	kl.Set(4, 4, v1)
	kl.Set(4, 5, -v2)

	kl.Set(5, 1, v2)
	kl.Set(5, 2, v4)
	kl.Set(5, 4, -v2)
	kl.Set(5, 5, v3)

	// Transformation T (6×6 block diagonal: [R₂, R₂])
	c, s := b.cos, b.sin
	T := mat.NewDense(6, 6, nil)
	for blk := 0; blk < 2; blk++ {
		o := blk * 3
		T.Set(o+0, o+0, c)
		T.Set(o+0, o+1, s)
		T.Set(o+1, o+0, -s)
		T.Set(o+1, o+1, c)
		T.Set(o+2, o+2, 1)
	}

	tmp := mat.NewDense(6, 6, nil)
	tmp.Mul(kl, T)
	b.ke = mat.NewDense(6, 6, nil)
	b.ke.Mul(T.T(), tmp)

	b.kl = kl
}

// ---------- Element interface ----------

func (b *TimoshenkoBeam2D) GetTangentStiffness() *mat.Dense { return b.ke }

func (b *TimoshenkoBeam2D) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(6, nil)
	f.MulVec(b.ke, mat.NewVecDense(6, b.ue[:]))
	return f
}

func (b *TimoshenkoBeam2D) NodeIDs() []int       { return b.Nds[:] }
func (b *TimoshenkoBeam2D) NumDOF() int          { return 6 }
func (b *TimoshenkoBeam2D) DOFPerNode() int      { return 3 }
func (b *TimoshenkoBeam2D) DOFTypes() []dof.Type { return dof.PlaneFrame(2) }

func (b *TimoshenkoBeam2D) Update(disp []float64) error {
	copy(b.ue[:], disp)
	return nil
}

func (b *TimoshenkoBeam2D) CommitState() error   { return nil }
func (b *TimoshenkoBeam2D) RevertToStart() error { b.ue = [6]float64{}; return nil }

// BodyForceLoad computes work-equivalent nodal forces due to a body force
// (ρ·A per unit length). Delegates to EquivalentNodalLoad.
func (b *TimoshenkoBeam2D) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	return b.EquivalentNodalLoad(g, rho*b.Sec.A)
}

// Length returns the beam length.
func (b *TimoshenkoBeam2D) Length() float64 { return b.length }

// EndForces computes the section forces at both beam ends in local coordinates.
func (b *TimoshenkoBeam2D) EndForces() BeamEndForces2D {
	c, s := b.cos, b.sin
	var uloc [6]float64
	for blk := 0; blk < 2; blk++ {
		o := blk * 3
		uloc[o+0] = c*b.ue[o+0] + s*b.ue[o+1]
		uloc[o+1] = -s*b.ue[o+0] + c*b.ue[o+1]
		uloc[o+2] = b.ue[o+2]
	}
	f := mat.NewVecDense(6, nil)
	f.MulVec(b.kl, mat.NewVecDense(6, uloc[:]))
	var ef BeamEndForces2D
	for i := 0; i < 3; i++ {
		ef.I[i] = f.AtVec(i)
		ef.J[i] = f.AtVec(i + 3)
	}
	return ef
}

// EquivalentNodalLoad returns work-equivalent nodal forces for a uniformly
// distributed load.  Same as ElasticBeam2D since for a straight uniform load
// the fixed-end forces are identical (shear effects only change displacements,
// not the equivalent nodal loads).
func (b *TimoshenkoBeam2D) EquivalentNodalLoad(globalDir [3]float64, intensity float64) *mat.VecDense {
	L := b.length
	L2 := L * L
	c, s := b.cos, b.sin

	qx := (c*globalDir[0] + s*globalDir[1]) * intensity
	qy := (-s*globalDir[0] + c*globalDir[1]) * intensity

	fLoc := mat.NewVecDense(6, nil)
	fLoc.SetVec(0, qx*L/2)
	fLoc.SetVec(1, qy*L/2)
	fLoc.SetVec(2, qy*L2/12)
	fLoc.SetVec(3, qx*L/2)
	fLoc.SetVec(4, qy*L/2)
	fLoc.SetVec(5, -qy*L2/12)

	fGlob := mat.NewVecDense(6, nil)
	for blk := 0; blk < 2; blk++ {
		o := 3 * blk
		fx := fLoc.AtVec(o + 0)
		fy := fLoc.AtVec(o + 1)
		mz := fLoc.AtVec(o + 2)
		fGlob.SetVec(o+0, c*fx-s*fy)
		fGlob.SetVec(o+1, s*fx+c*fy)
		fGlob.SetVec(o+2, mz)
	}
	return fGlob
}

// EquivalentNodalLoadLinear returns work-equivalent nodal forces for a
// linearly varying (trapezoidal) distributed load.
func (b *TimoshenkoBeam2D) EquivalentNodalLoadLinear(globalDir [3]float64, intensityI, intensityJ float64) *mat.VecDense {
	L := b.length
	L2 := L * L
	c, s := b.cos, b.sin

	qxi := (c*globalDir[0] + s*globalDir[1]) * intensityI
	qyi := (-s*globalDir[0] + c*globalDir[1]) * intensityI
	qxj := (c*globalDir[0] + s*globalDir[1]) * intensityJ
	qyj := (-s*globalDir[0] + c*globalDir[1]) * intensityJ

	fLoc := mat.NewVecDense(6, nil)
	fLoc.SetVec(0, L/6*(2*qxi+qxj))
	fLoc.SetVec(1, L/20*(7*qyi+3*qyj))
	fLoc.SetVec(2, L2/60*(3*qyi+2*qyj))
	fLoc.SetVec(3, L/6*(qxi+2*qxj))
	fLoc.SetVec(4, L/20*(3*qyi+7*qyj))
	fLoc.SetVec(5, -L2/60*(2*qyi+3*qyj))

	fGlob := mat.NewVecDense(6, nil)
	for blk := 0; blk < 2; blk++ {
		o := 3 * blk
		fx := fLoc.AtVec(o + 0)
		fy := fLoc.AtVec(o + 1)
		mz := fLoc.AtVec(o + 2)
		fGlob.SetVec(o+0, c*fx-s*fy)
		fGlob.SetVec(o+1, s*fx+c*fy)
		fGlob.SetVec(o+2, mz)
	}
	return fGlob
}
