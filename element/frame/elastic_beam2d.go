package frame

import (
	"math"

	"go-fem/dof"
	"go-fem/section"

	"gonum.org/v1/gonum/mat"
)

// BeamEndForces2D holds the local-coordinate section forces at both ends of a
// 2D beam.  Convention per end: [N, V, M].
//
//	N > 0 = tension
//	V = shear in local y (transverse)
//	M = bending moment about local z (out-of-plane)
type BeamEndForces2D struct {
	I [3]float64 // forces at node i (start)
	J [3]float64 // forces at node j (end)
}

// ElasticBeam2D is a 2-node 2D Euler-Bernoulli beam element for plane frames.
// 3 DOFs per node (UX, UY, RZ), 6 DOFs total.
type ElasticBeam2D struct {
	ID     int
	Nds    [2]int
	Coords [2][2]float64
	E      float64 // Young's modulus
	Sec    section.BeamSection2D

	ke     *mat.Dense
	kl     *mat.Dense // local stiffness for EndForces
	length float64
	cos    float64 // cos(θ) — beam angle
	sin    float64 // sin(θ)
	ue     [6]float64
}

// NewElasticBeam2D creates a 2D elastic beam element.
func NewElasticBeam2D(id int, nodes [2]int, coords [2][2]float64,
	e float64, sec section.BeamSection2D) *ElasticBeam2D {

	b := &ElasticBeam2D{
		ID:     id,
		Nds:    nodes,
		Coords: coords,
		E:      e,
		Sec:    sec,
	}
	b.computeGeometry()
	b.formKe()
	return b
}

func (b *ElasticBeam2D) computeGeometry() {
	dx := b.Coords[1][0] - b.Coords[0][0]
	dy := b.Coords[1][1] - b.Coords[0][1]
	b.length = math.Sqrt(dx*dx + dy*dy)
	b.cos = dx / b.length
	b.sin = dy / b.length
}

// formKe builds the 6×6 global stiffness: Ke = Tᵀ·Klocal·T
//
// Local DOF order per node: [u, v, θz]
//
//	u = axial displacement (along beam)
//	v = transverse displacement (perpendicular to beam in the XY plane)
//	θz = rotation about z (out-of-plane)
func (b *ElasticBeam2D) formKe() {
	L := b.length
	E := b.E
	A := b.Sec.A
	Iz := b.Sec.Iz

	L2 := L * L
	L3 := L2 * L

	kl := mat.NewDense(6, 6, nil)

	// Axial (DOFs 0, 3)
	ea := E * A / L
	kl.Set(0, 0, ea)
	kl.Set(0, 3, -ea)
	kl.Set(3, 0, -ea)
	kl.Set(3, 3, ea)

	// Bending in x-y plane (DOFs: v at 1,4; θz at 2,5)
	v1 := 12 * E * Iz / L3
	v2 := 6 * E * Iz / L2
	v3 := 4 * E * Iz / L
	v4 := 2 * E * Iz / L

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
	// R₂ = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
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

	// Ke = Tᵀ · Klocal · T
	tmp := mat.NewDense(6, 6, nil)
	tmp.Mul(kl, T)
	b.ke = mat.NewDense(6, 6, nil)
	b.ke.Mul(T.T(), tmp)

	b.kl = kl
}

// ---------- Element interface ----------

func (b *ElasticBeam2D) GetTangentStiffness() *mat.Dense { return b.ke }

func (b *ElasticBeam2D) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(6, nil)
	f.MulVec(b.ke, mat.NewVecDense(6, b.ue[:]))
	return f
}

func (b *ElasticBeam2D) NodeIDs() []int       { return b.Nds[:] }
func (b *ElasticBeam2D) NumDOF() int          { return 6 }
func (b *ElasticBeam2D) DOFPerNode() int      { return 3 }
func (b *ElasticBeam2D) DOFTypes() []dof.Type { return dof.PlaneFrame(2) }

func (b *ElasticBeam2D) Update(disp []float64) error {
	copy(b.ue[:], disp)
	return nil
}

func (b *ElasticBeam2D) CommitState() error   { return nil }
func (b *ElasticBeam2D) RevertToStart() error { b.ue = [6]float64{}; return nil }

// BodyForceLoad computes work-equivalent nodal forces due to a body force
// (ρ·A per unit length). Delegates to EquivalentNodalLoad.
func (b *ElasticBeam2D) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	return b.EquivalentNodalLoad(g, rho*b.Sec.A)
}

// GetMassMatrix returns the 6×6 consistent mass matrix in global coordinates.
// Local DOF order: [u₁, v₁, θz₁, u₂, v₂, θz₂]
// Axial:   ρAL/6 · [2,1;1,2]
// Bending: ρAL/420 · Hermitian matrix (Euler-Bernoulli shape functions)
// The local matrix is rotated: Mₑ = Tᵀ · Mₗₒc · T
func (b *ElasticBeam2D) GetMassMatrix(rho float64) *mat.Dense {
	L := b.length
	L2 := L * L
	A := b.Sec.A

	// Local consistent mass matrix (6×6)
	mLoc := mat.NewDense(6, 6, nil)

	// Axial (DOFs 0, 3): ρAL/6 · [2,1;1,2]
	ca := rho * A * L / 6.0
	mLoc.Set(0, 0, 2*ca)
	mLoc.Set(3, 3, 2*ca)
	mLoc.Set(0, 3, ca)
	mLoc.Set(3, 0, ca)

	// Bending (DOFs 1,2,4,5): ρAL/420 · Hermitian
	cb := rho * A * L / 420.0
	mLoc.Set(1, 1, 156*cb)
	mLoc.Set(1, 4, 54*cb)
	mLoc.Set(4, 1, 54*cb)
	mLoc.Set(4, 4, 156*cb)

	mLoc.Set(1, 2, 22*L*cb)
	mLoc.Set(2, 1, 22*L*cb)
	mLoc.Set(1, 5, -13*L*cb)
	mLoc.Set(5, 1, -13*L*cb)

	mLoc.Set(4, 2, 13*L*cb)
	mLoc.Set(2, 4, 13*L*cb)
	mLoc.Set(4, 5, -22*L*cb)
	mLoc.Set(5, 4, -22*L*cb)

	mLoc.Set(2, 2, 4*L2*cb)
	mLoc.Set(5, 5, 4*L2*cb)
	mLoc.Set(2, 5, -3*L2*cb)
	mLoc.Set(5, 2, -3*L2*cb)

	// Transformation T (same as for stiffness)
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

	// Mₑ = Tᵀ · Mₗₒc · T
	tmp := mat.NewDense(6, 6, nil)
	tmp.Mul(mLoc, T)
	me := mat.NewDense(6, 6, nil)
	me.Mul(T.T(), tmp)
	return me
}

// Length returns the beam length.
func (b *ElasticBeam2D) Length() float64 { return b.length }

// EndForces computes the section forces at both beam ends in local coordinates.
// Requires Update() to have been called first.
func (b *ElasticBeam2D) EndForces() BeamEndForces2D {
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

// EquivalentNodalLoad returns work-equivalent nodal forces (in global coords)
// for a uniformly distributed load. globalDir is a 3D direction vector
// (only X and Y components are used). intensity is load per unit length.
func (b *ElasticBeam2D) EquivalentNodalLoad(globalDir [3]float64, intensity float64) *mat.VecDense {
	L := b.length
	L2 := L * L
	c, s := b.cos, b.sin

	// Transform global direction to local (2D rotation)
	qx := (c*globalDir[0] + s*globalDir[1]) * intensity
	qy := (-s*globalDir[0] + c*globalDir[1]) * intensity

	// Fixed-end reactions in local coords [u₁,v₁,θ₁, u₂,v₂,θ₂]
	fLoc := mat.NewVecDense(6, nil)
	fLoc.SetVec(0, qx*L/2)
	fLoc.SetVec(1, qy*L/2)
	fLoc.SetVec(2, qy*L2/12)
	fLoc.SetVec(3, qx*L/2)
	fLoc.SetVec(4, qy*L/2)
	fLoc.SetVec(5, -qy*L2/12)

	// Transform to global: f_global = R₂ᵀ · f_local per block
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
// linearly varying (trapezoidal) distributed load. intensityI is the load per
// unit length at node i and intensityJ at node j. The consistent fixed-end
// reactions use cubic Hermite shape functions:
//
//	Fy_i = L/20·(7·qi + 3·qj),  Mz_i = +L²/60·(3·qi + 2·qj)
//	Fy_j = L/20·(3·qi + 7·qj),  Mz_j = -L²/60·(2·qi + 3·qj)
func (b *ElasticBeam2D) EquivalentNodalLoadLinear(globalDir [3]float64, intensityI, intensityJ float64) *mat.VecDense {
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
