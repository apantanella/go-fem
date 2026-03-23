package frame

import (
	"math"

	"go-fem/dof"
	"go-fem/section"

	"gonum.org/v1/gonum/mat"
)

// TimoshenkoBeam3D is a 2-node 3D shear-deformable (Timoshenko) beam element.
// 6 DOFs per node (UX, UY, UZ, RX, RY, RZ), 12 DOFs total — same layout as
// ElasticBeam3D.
//
// Compared to Euler-Bernoulli, the Timoshenko formulation introduces the
// shear-flexibility parameter:
//
//	Φz = 12·E·Iz / (G·Asy·L²)   (bending in local x-y plane)
//	Φy = 12·E·Iy / (G·Asz·L²)   (bending in local x-z plane)
//
// where Asy = κy·A and Asz = κz·A are the shear areas (κ = shear correction
// factor).  When Φ → 0 the stiffness matrix reduces to the Euler-Bernoulli
// matrix.  Typical shear correction factors:
//
//	κ ≈ 5/6  rectangular solid cross-section (default when Asy/Asz = 0)
//	κ ≈ 0.9  circular solid cross-section
//	κ ≈ 0.4  thin-walled I-section (weak axis)
type TimoshenkoBeam3D struct {
	ID     int
	Nds    [2]int
	Coords [2][3]float64
	E      float64 // Young's modulus
	G      float64 // Shear modulus
	Sec    section.BeamSection3D

	// VecXZ is a vector in the local x-z plane (defines element orientation).
	// If zero it is auto-computed (same convention as ElasticBeam3D).
	VecXZ [3]float64

	ke     *mat.Dense
	kl     *mat.Dense // local stiffness, used for EndForces post-processing
	length float64
	R      [3][3]float64 // rotation matrix: rows are local axes in global coords
	ue     [12]float64   // element displacements in global coords (set by Update)
}

// NewTimoshenkoBeam3D creates a 3D Timoshenko beam element.
func NewTimoshenkoBeam3D(id int, nodes [2]int, coords [2][3]float64,
	e, g float64, sec section.BeamSection3D, vecXZ [3]float64) *TimoshenkoBeam3D {

	b := &TimoshenkoBeam3D{
		ID:     id,
		Nds:    nodes,
		Coords: coords,
		E:      e,
		G:      g,
		Sec:    sec,
		VecXZ:  vecXZ,
	}
	b.computeGeometry()
	b.formKe()
	return b
}

func (b *TimoshenkoBeam3D) computeGeometry() {
	dx := b.Coords[1][0] - b.Coords[0][0]
	dy := b.Coords[1][1] - b.Coords[0][1]
	dz := b.Coords[1][2] - b.Coords[0][2]
	b.length = math.Sqrt(dx*dx + dy*dy + dz*dz)

	xAxis := [3]float64{dx / b.length, dy / b.length, dz / b.length}

	vxz := b.VecXZ
	if vxz[0] == 0 && vxz[1] == 0 && vxz[2] == 0 {
		if math.Abs(xAxis[2]) > 0.9 {
			vxz = [3]float64{1, 0, 0}
		} else {
			vxz = [3]float64{0, 0, 1}
		}
	}

	yAxis := cross(xAxis, vxz)
	ny := norm(yAxis)
	yAxis[0] /= ny
	yAxis[1] /= ny
	yAxis[2] /= ny

	zAxis := cross(xAxis, yAxis)
	b.R = [3][3]float64{xAxis, yAxis, zAxis}
}

// formKe builds the 12×12 global stiffness: Ke = Tᵀ·Klocal·T
func (b *TimoshenkoBeam3D) formKe() {
	L := b.length
	E := b.E
	G := b.G
	A := b.Sec.A
	Iy := b.Sec.Iy
	Iz := b.Sec.Iz
	J := b.Sec.J

	// Resolve shear areas; default to 5/6·A if unset.
	const defaultKappa = 5.0 / 6.0
	Asy := b.Sec.Asy
	if Asy == 0 {
		Asy = defaultKappa * A
	}
	Asz := b.Sec.Asz
	if Asz == 0 {
		Asz = defaultKappa * A
	}

	L2 := L * L
	L3 := L2 * L

	// Timoshenko shear-flexibility parameters
	Phiz := 12 * E * Iz / (G * Asy * L2) // for bending in x-y plane
	Phiy := 12 * E * Iy / (G * Asz * L2) // for bending in x-z plane

	kl := mat.NewDense(12, 12, nil)

	// Axial (DOFs 0, 6)
	ea := E * A / L
	kl.Set(0, 0, ea)
	kl.Set(0, 6, -ea)
	kl.Set(6, 0, -ea)
	kl.Set(6, 6, ea)

	// Torsion (DOFs 3, 9)
	gj := G * J / L
	kl.Set(3, 3, gj)
	kl.Set(3, 9, -gj)
	kl.Set(9, 3, -gj)
	kl.Set(9, 9, gj)

	// Bending in x-y plane (uses Iz, shear area Asy)
	// DOFs: v (1,7), θz (5,11)
	cz := E * Iz / ((1 + Phiz) * L3)
	v1 := 12 * cz
	v2 := 6 * cz * L
	v3 := (4 + Phiz) * cz * L2
	v4 := (2 - Phiz) * cz * L2
	kl.Set(1, 1, v1)
	kl.Set(1, 5, v2)
	kl.Set(1, 7, -v1)
	kl.Set(1, 11, v2)
	kl.Set(5, 1, v2)
	kl.Set(5, 5, v3)
	kl.Set(5, 7, -v2)
	kl.Set(5, 11, v4)
	kl.Set(7, 1, -v1)
	kl.Set(7, 5, -v2)
	kl.Set(7, 7, v1)
	kl.Set(7, 11, -v2)
	kl.Set(11, 1, v2)
	kl.Set(11, 5, v4)
	kl.Set(11, 7, -v2)
	kl.Set(11, 11, v3)

	// Bending in x-z plane (uses Iy, shear area Asz)
	// DOFs: w (2,8), θy (4,10)
	// Note: θy sign convention gives negative coupling for (w, θy) — same as EB.
	cy := E * Iy / ((1 + Phiy) * L3)
	w1 := 12 * cy
	w2 := 6 * cy * L
	w3 := (4 + Phiy) * cy * L2
	w4 := (2 - Phiy) * cy * L2
	kl.Set(2, 2, w1)
	kl.Set(2, 4, -w2)
	kl.Set(2, 8, -w1)
	kl.Set(2, 10, -w2)
	kl.Set(4, 2, -w2)
	kl.Set(4, 4, w3)
	kl.Set(4, 8, w2)
	kl.Set(4, 10, w4)
	kl.Set(8, 2, -w1)
	kl.Set(8, 4, w2)
	kl.Set(8, 8, w1)
	kl.Set(8, 10, w2)
	kl.Set(10, 2, -w2)
	kl.Set(10, 4, w4)
	kl.Set(10, 8, w2)
	kl.Set(10, 10, w3)

	// Transformation T (12×12 block diagonal: [R, R, R, R])
	T := mat.NewDense(12, 12, nil)
	for block := 0; block < 4; block++ {
		off := block * 3
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				T.Set(off+i, off+j, b.R[i][j])
			}
		}
	}

	// Ke = Tᵀ · Klocal · T
	tmp := mat.NewDense(12, 12, nil)
	tmp.Mul(kl, T)
	b.ke = mat.NewDense(12, 12, nil)
	b.ke.Mul(T.T(), tmp)

	b.kl = kl
}

// ---------- Element interface ----------

func (b *TimoshenkoBeam3D) GetTangentStiffness() *mat.Dense { return b.ke }

func (b *TimoshenkoBeam3D) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(12, nil)
	f.MulVec(b.ke, mat.NewVecDense(12, b.ue[:]))
	return f
}

func (b *TimoshenkoBeam3D) NodeIDs() []int       { return b.Nds[:] }
func (b *TimoshenkoBeam3D) NumDOF() int          { return 12 }
func (b *TimoshenkoBeam3D) DOFPerNode() int      { return 6 }
func (b *TimoshenkoBeam3D) DOFTypes() []dof.Type { return dof.Full6D(2) }

func (b *TimoshenkoBeam3D) Update(disp []float64) error {
	copy(b.ue[:], disp)
	return nil
}

func (b *TimoshenkoBeam3D) CommitState() error   { return nil }
func (b *TimoshenkoBeam3D) RevertToStart() error { b.ue = [12]float64{}; return nil }

// BodyForceLoad computes work-equivalent nodal forces due to a body force
// (ρ·A per unit length). Delegates to EquivalentNodalLoad.
func (b *TimoshenkoBeam3D) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	return b.EquivalentNodalLoad(g, rho*b.Sec.A)
}

// EquivalentNodalLoad returns work-equivalent nodal forces for a uniformly
// distributed load.  Fixed-end reactions for Timoshenko beam:
//
//	Fy = qy·L/2,  Mz = ±qy·L²/12·(1-Φz/2)/(1+Φz) × correction
//
// For the mid-span loaded case the consistent load vector (from virtual work)
// gives the same Fy=qL/2 as EB; only the end moments differ:
//
//	Mz_i = +qy·L²/12,  Mz_j = −qy·L²/12   (identical to EB for uniform load)
//
// Note: for a Timoshenko beam under UDL the fixed-end moments are actually
// identical to EB because the shear term does not affect the equivalent nodal
// forces for a straight uniform load.  The difference appears in the
// displacement field, not in the load vector.
func (b *TimoshenkoBeam3D) EquivalentNodalLoad(globalDir [3]float64, intensity float64) *mat.VecDense {
	L := b.length
	L2 := L * L

	var qLoc [3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			qLoc[i] += b.R[i][j] * globalDir[j]
		}
	}
	qx := qLoc[0] * intensity
	qy := qLoc[1] * intensity
	qz := qLoc[2] * intensity

	fLoc := mat.NewVecDense(12, nil)
	fLoc.SetVec(0, qx*L/2)
	fLoc.SetVec(1, qy*L/2)
	fLoc.SetVec(2, qz*L/2)
	fLoc.SetVec(4, -qz*L2/12)
	fLoc.SetVec(5, qy*L2/12)
	fLoc.SetVec(6, qx*L/2)
	fLoc.SetVec(7, qy*L/2)
	fLoc.SetVec(8, qz*L/2)
	fLoc.SetVec(10, qz*L2/12)
	fLoc.SetVec(11, -qy*L2/12)

	fGlob := mat.NewVecDense(12, nil)
	for blk := 0; blk < 4; blk++ {
		off := 3 * blk
		for i := 0; i < 3; i++ {
			var sum float64
			for j := 0; j < 3; j++ {
				sum += b.R[j][i] * fLoc.AtVec(off+j)
			}
			fGlob.SetVec(off+i, sum)
		}
	}
	return fGlob
}

// Length returns the beam length.
func (b *TimoshenkoBeam3D) Length() float64 { return b.length }

// EndForces computes section forces at both ends in local element coordinates.
// Requires Update() to have been called first.
func (b *TimoshenkoBeam3D) EndForces() BeamEndForces {
	var uloc [12]float64
	for blk := 0; blk < 4; blk++ {
		off := 3 * blk
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				uloc[off+i] += b.R[i][j] * b.ue[off+j]
			}
		}
	}
	f := mat.NewVecDense(12, nil)
	f.MulVec(b.kl, mat.NewVecDense(12, uloc[:]))
	var ef BeamEndForces
	for i := 0; i < 6; i++ {
		ef.I[i] = f.AtVec(i)
		ef.J[i] = f.AtVec(i + 6)
	}
	return ef
}

// EquivalentNodalLoadLinear returns work-equivalent nodal forces for a
// linearly varying (trapezoidal) distributed load.
func (b *TimoshenkoBeam3D) EquivalentNodalLoadLinear(globalDir [3]float64, intensityI, intensityJ float64) *mat.VecDense {
	L := b.length
	L2 := L * L

	var qLocI, qLocJ [3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			qLocI[i] += b.R[i][j] * globalDir[j]
			qLocJ[i] += b.R[i][j] * globalDir[j]
		}
		qLocI[i] *= intensityI
		qLocJ[i] *= intensityJ
	}
	qxi, qyi, qzi := qLocI[0], qLocI[1], qLocI[2]
	qxj, qyj, qzj := qLocJ[0], qLocJ[1], qLocJ[2]

	fLoc := mat.NewVecDense(12, nil)
	fLoc.SetVec(0, L/6*(2*qxi+qxj))
	fLoc.SetVec(1, L/20*(7*qyi+3*qyj))
	fLoc.SetVec(2, L/20*(7*qzi+3*qzj))
	fLoc.SetVec(4, -L2/60*(3*qzi+2*qzj))
	fLoc.SetVec(5, L2/60*(3*qyi+2*qyj))
	fLoc.SetVec(6, L/6*(qxi+2*qxj))
	fLoc.SetVec(7, L/20*(3*qyi+7*qyj))
	fLoc.SetVec(8, L/20*(3*qzi+7*qzj))
	fLoc.SetVec(10, L2/60*(2*qzi+3*qzj))
	fLoc.SetVec(11, -L2/60*(2*qyi+3*qyj))

	fGlob := mat.NewVecDense(12, nil)
	for blk := 0; blk < 4; blk++ {
		off := 3 * blk
		for i := 0; i < 3; i++ {
			var sum float64
			for j := 0; j < 3; j++ {
				sum += b.R[j][i] * fLoc.AtVec(off+j)
			}
			fGlob.SetVec(off+i, sum)
		}
	}
	return fGlob
}
