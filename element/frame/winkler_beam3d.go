package frame

import (
	"fmt"
	"math"

	"go-fem/dof"
	"go-fem/section"

	"gonum.org/v1/gonum/mat"
)

// WinklerBeam3D is a 2-node 3D Euler-Bernoulli beam element resting on a
// distributed Winkler elastic foundation.
//
// Two independent spring stiffnesses can be provided:
//
//	Ksy: subgrade modulus in the local Y direction [N/mm³]
//	Ksz: subgrade modulus in the local Z direction [N/mm³]
//
// The total element stiffness is:
//
//	Ke = K_beam3D + K_winkler_Y + K_winkler_Z
//
// K_winkler_Y (for [v_i, θz_i, v_j, θz_j], positive slope dv/dx = θz):
//
//	K_wy = Ksy·B·L/420 · [156, 22L, 54, -13L; ...]
//
// K_winkler_Z (for [w_i, θy_i, w_j, θy_j], sign convention dw/dx = -θy):
//
//	K_wz = Ksz·B·L/420 · [156, -22L, 54, 13L; ...]
//
// Note the negated off-diagonal rotation terms in K_wz vs K_wy: this reflects
// the right-hand rule convention dw/dx = -θy used in the 3D beam element.
//
// DOF layout: 6 per node (UX, UY, UZ, RX, RY, RZ), total 12, same as ElasticBeam3D.
// B is the effective foundation width [mm] (defaults to 1 if ≤ 0).
// Units: consistent (e.g. N-mm-MPa).
type WinklerBeam3D struct {
	ID     int
	Nds    [2]int
	Coords [2][3]float64
	E      float64 // Young's modulus
	G      float64 // Shear modulus
	Sec    section.BeamSection3D
	VecXZ  [3]float64 // orientation vector in local x-z plane

	Ksy float64 // Winkler modulus local Y [N/mm³]
	Ksz float64 // Winkler modulus local Z [N/mm³]
	B   float64 // effective width [mm] (default 1 if ≤ 0)

	ke     *mat.Dense // 12×12 global stiffness
	kl     *mat.Dense // 12×12 beam-only local stiffness (for EndForces)
	length float64
	R      [3][3]float64  // rotation matrix: rows are local axes in global coords
	ue     [12]float64    // element displacements (set by Update)
	fFixed [12]float64    // accumulated fixed-end forces in local coords (from distributed loads)
}

// NewWinklerBeam3D creates a 3D Euler-Bernoulli beam on Winkler elastic foundation.
// ksy, ksz are the subgrade reaction moduli for the local Y and Z directions [N/mm³].
// b is the effective width [mm]; set b ≤ 0 to use unit width.
func NewWinklerBeam3D(id int, nodes [2]int, coords [2][3]float64,
	e, g float64, sec section.BeamSection3D, vecXZ [3]float64,
	ksy, ksz, b float64) (*WinklerBeam3D, error) {

	if e <= 0 {
		return nil, fmt.Errorf("WinklerBeam3D: E must be > 0, got %g", e)
	}
	if ksy < 0 {
		return nil, fmt.Errorf("WinklerBeam3D: Ksy must be ≥ 0, got %g", ksy)
	}
	if ksz < 0 {
		return nil, fmt.Errorf("WinklerBeam3D: Ksz must be ≥ 0, got %g", ksz)
	}
	if b <= 0 {
		b = 1
	}
	wb := &WinklerBeam3D{
		ID: id, Nds: nodes, Coords: coords,
		E: e, G: g, Sec: sec, VecXZ: vecXZ,
		Ksy: ksy, Ksz: ksz, B: b,
	}
	wb.computeGeometry()
	wb.formKe()
	return wb, nil
}

func (wb *WinklerBeam3D) computeGeometry() {
	dx := wb.Coords[1][0] - wb.Coords[0][0]
	dy := wb.Coords[1][1] - wb.Coords[0][1]
	dz := wb.Coords[1][2] - wb.Coords[0][2]
	wb.length = math.Sqrt(dx*dx + dy*dy + dz*dz)

	xAxis := [3]float64{dx / wb.length, dy / wb.length, dz / wb.length}

	vxz := wb.VecXZ
	if vxz[0] == 0 && vxz[1] == 0 && vxz[2] == 0 {
		if math.Abs(xAxis[2]) > 0.9 {
			vxz = [3]float64{1, 0, 0}
		} else {
			vxz = [3]float64{0, 0, 1}
		}
	}

	// local y = normalise(xAxis × vxz)
	yAxis := cross(xAxis, vxz)
	ny := norm(yAxis)
	yAxis[0] /= ny
	yAxis[1] /= ny
	yAxis[2] /= ny

	// local z = xAxis × yAxis
	zAxis := cross(xAxis, yAxis)
	wb.R = [3][3]float64{xAxis, yAxis, zAxis}
}

// formKe builds Ke = Tᵀ·(K_beam_local + K_winkler_Y + K_winkler_Z)·T
//
// DOF order per node: [u, v, w, θx, θy, θz]
// Node i: DOFs 0..5 ; node j: DOFs 6..11
//
// Winkler Y acts on: v_i=1, θz_i=5, v_j=7, θz_j=11
// Winkler Z acts on: w_i=2, θy_i=4, w_j=8, θy_j=10
func (wb *WinklerBeam3D) formKe() {
	L := wb.length
	E := wb.E
	G := wb.G
	A := wb.Sec.A
	Iy := wb.Sec.Iy
	Iz := wb.Sec.Iz
	J := wb.Sec.J

	L2 := L * L
	L3 := L2 * L

	// --- Build beam-only local stiffness (saved for EndForces) ---
	klBeam := mat.NewDense(12, 12, nil)

	// Axial
	ea := E * A / L
	klBeam.Set(0, 0, ea)
	klBeam.Set(0, 6, -ea)
	klBeam.Set(6, 0, -ea)
	klBeam.Set(6, 6, ea)

	// Torsion
	gj := G * J / L
	klBeam.Set(3, 3, gj)
	klBeam.Set(3, 9, -gj)
	klBeam.Set(9, 3, -gj)
	klBeam.Set(9, 9, gj)

	// Bending in x-y plane (DOFs: v=1,θz=5; v=7,θz=11)  dv/dx = +θz
	v1 := 12 * E * Iz / L3
	v2 := 6 * E * Iz / L2
	v3 := 4 * E * Iz / L
	v4 := 2 * E * Iz / L
	klBeam.Set(1, 1, v1)
	klBeam.Set(1, 5, v2)
	klBeam.Set(1, 7, -v1)
	klBeam.Set(1, 11, v2)
	klBeam.Set(5, 1, v2)
	klBeam.Set(5, 5, v3)
	klBeam.Set(5, 7, -v2)
	klBeam.Set(5, 11, v4)
	klBeam.Set(7, 1, -v1)
	klBeam.Set(7, 5, -v2)
	klBeam.Set(7, 7, v1)
	klBeam.Set(7, 11, -v2)
	klBeam.Set(11, 1, v2)
	klBeam.Set(11, 5, v4)
	klBeam.Set(11, 7, -v2)
	klBeam.Set(11, 11, v3)

	// Bending in x-z plane (DOFs: w=2,θy=4; w=8,θy=10)  dw/dx = -θy
	w1 := 12 * E * Iy / L3
	w2 := 6 * E * Iy / L2
	w3 := 4 * E * Iy / L
	w4 := 2 * E * Iy / L
	klBeam.Set(2, 2, w1)
	klBeam.Set(2, 4, -w2)
	klBeam.Set(2, 8, -w1)
	klBeam.Set(2, 10, -w2)
	klBeam.Set(4, 2, -w2)
	klBeam.Set(4, 4, w3)
	klBeam.Set(4, 8, w2)
	klBeam.Set(4, 10, w4)
	klBeam.Set(8, 2, -w1)
	klBeam.Set(8, 4, w2)
	klBeam.Set(8, 8, w1)
	klBeam.Set(8, 10, w2)
	klBeam.Set(10, 2, -w2)
	klBeam.Set(10, 4, w4)
	klBeam.Set(10, 8, w2)
	klBeam.Set(10, 10, w3)

	wb.kl = klBeam // beam section forces only

	// --- Build total local stiffness = beam + Winkler ---
	klFull := mat.NewDense(12, 12, nil)
	klFull.Copy(klBeam)

	// Winkler Y: phi = [N1, N2, N3, N4] (positive dv/dx = θz)
	// K_wy = alpha_y · [[156,22L,54,-13L], [22L,4L²,13L,-3L²], [54,13L,156,-22L], [-13L,-3L²,-22L,4L²]]
	// DOF positions:   [1,   5,  7,  11  ]
	if wb.Ksy > 0 {
		ay := wb.Ksy * wb.B * L / 420.0
		addWinklerY3D(klFull, ay, L)
	}

	// Winkler Z: phi = [N1, -N2, N3, -N4] (dw/dx = -θy)
	// K_wz = alpha_z · [[156,-22L,54,13L], [-22L,4L²,-13L,-3L²], [54,-13L,156,22L], [13L,-3L²,22L,4L²]]
	// DOF positions:   [2,   4,   8, 10  ]
	if wb.Ksz > 0 {
		az := wb.Ksz * wb.B * L / 420.0
		addWinklerZ3D(klFull, az, L)
	}

	// Build 12×12 block-diagonal rotation T
	T := mat.NewDense(12, 12, nil)
	for block := 0; block < 4; block++ {
		off := block * 3
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				T.Set(off+i, off+j, wb.R[i][j])
			}
		}
	}

	// Ke = Tᵀ · klFull · T
	tmp := mat.NewDense(12, 12, nil)
	tmp.Mul(klFull, T)
	wb.ke = mat.NewDense(12, 12, nil)
	wb.ke.Mul(T.T(), tmp)
}

// addWinklerY3D adds the consistent Winkler spring stiffness for the local Y
// direction to kl. DOFs involved: v_i=1, θz_i=5, v_j=7, θz_j=11.
// Convention: dv/dx = +θz (standard positive slope).
func addWinklerY3D(kl *mat.Dense, alpha, L float64) {
	L2 := L * L
	// [v_i=1, θz_i=5, v_j=7, θz_j=11]
	ii := [4]int{1, 5, 7, 11}
	m := [4][4]float64{
		{156, 22 * L, 54, -13 * L},
		{22 * L, 4 * L2, 13 * L, -3 * L2},
		{54, 13 * L, 156, -22 * L},
		{-13 * L, -3 * L2, -22 * L, 4 * L2},
	}
	for a := 0; a < 4; a++ {
		for b := 0; b < 4; b++ {
			kl.Set(ii[a], ii[b], kl.At(ii[a], ii[b])+alpha*m[a][b])
		}
	}
}

// addWinklerZ3D adds the consistent Winkler spring stiffness for the local Z
// direction to kl. DOFs involved: w_i=2, θy_i=4, w_j=8, θy_j=10.
// Convention: dw/dx = -θy (right-hand rule). The rotation off-diagonals are
// negated compared to the Y direction due to this sign convention.
func addWinklerZ3D(kl *mat.Dense, alpha, L float64) {
	L2 := L * L
	// [w_i=2, θy_i=4, w_j=8, θy_j=10]
	ii := [4]int{2, 4, 8, 10}
	m := [4][4]float64{
		{156, -22 * L, 54, 13 * L},
		{-22 * L, 4 * L2, -13 * L, -3 * L2},
		{54, -13 * L, 156, 22 * L},
		{13 * L, -3 * L2, 22 * L, 4 * L2},
	}
	for a := 0; a < 4; a++ {
		for b := 0; b < 4; b++ {
			kl.Set(ii[a], ii[b], kl.At(ii[a], ii[b])+alpha*m[a][b])
		}
	}
}

// ---------- Element interface ----------

func (wb *WinklerBeam3D) GetTangentStiffness() *mat.Dense { return wb.ke }

// GetResistingForce returns Ke·ue (in global coordinates).
func (wb *WinklerBeam3D) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(12, nil)
	f.MulVec(wb.ke, mat.NewVecDense(12, wb.ue[:]))
	return f
}

func (wb *WinklerBeam3D) NodeIDs() []int       { return wb.Nds[:] }
func (wb *WinklerBeam3D) NumDOF() int          { return 12 }
func (wb *WinklerBeam3D) DOFPerNode() int      { return 6 }
func (wb *WinklerBeam3D) DOFTypes() []dof.Type { return dof.Full6D(2) }

func (wb *WinklerBeam3D) Update(disp []float64) error {
	copy(wb.ue[:], disp)
	return nil
}

func (wb *WinklerBeam3D) CommitState() error { return nil }
func (wb *WinklerBeam3D) RevertToStart() error {
	wb.ue = [12]float64{}
	wb.fFixed = [12]float64{}
	return nil
}

// ResetFixedEndForces clears accumulated fixed-end forces (for re-assembly).
func (wb *WinklerBeam3D) ResetFixedEndForces() { wb.fFixed = [12]float64{} }

// EndForces returns the beam section forces at both ends in local coordinates.
// Convention per end: [N, Vy, Vz, Mx, My, Mz].
// Uses beam-only stiffness (Winkler spring reactions are distributed, not section forces).
func (wb *WinklerBeam3D) EndForces() BeamEndForces {
	var uloc [12]float64
	for blk := 0; blk < 4; blk++ {
		off := 3 * blk
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				uloc[off+i] += wb.R[i][j] * wb.ue[off+j]
			}
		}
	}
	f := mat.NewVecDense(12, nil)
	f.MulVec(wb.kl, mat.NewVecDense(12, uloc[:]))
	// Subtract equivalent nodal loads to recover true section forces:
	// f_section = Kl·uloc - f_equiv_nodal_local
	var ef BeamEndForces
	for i := 0; i < 6; i++ {
		ef.I[i] = f.AtVec(i) - wb.fFixed[i]
		ef.J[i] = f.AtVec(i+6) - wb.fFixed[i+6]
	}
	return ef
}

// EquivalentNodalLoad returns work-equivalent nodal forces (global coords) for a
// UDL of given intensity in globalDir. Identical formula to ElasticBeam3D.
func (wb *WinklerBeam3D) EquivalentNodalLoad(globalDir [3]float64, intensity float64) *mat.VecDense {
	L := wb.length
	L2 := L * L

	var qLoc [3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			qLoc[i] += wb.R[i][j] * globalDir[j]
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

	// Accumulate local fixed-end forces for EndForces post-processing
	for i := 0; i < 12; i++ {
		wb.fFixed[i] += fLoc.AtVec(i)
	}

	fGlob := mat.NewVecDense(12, nil)
	for blk := 0; blk < 4; blk++ {
		off := 3 * blk
		for i := 0; i < 3; i++ {
			var sum float64
			for j := 0; j < 3; j++ {
				sum += wb.R[j][i] * fLoc.AtVec(off+j)
			}
			fGlob.SetVec(off+i, sum)
		}
	}
	return fGlob
}

// BodyForceLoad computes work-equivalent nodal forces due to self-weight.
func (wb *WinklerBeam3D) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	return wb.EquivalentNodalLoad(g, rho*wb.Sec.A)
}

// EquivalentNodalLoadLinear returns work-equivalent nodal forces for a
// linearly varying (trapezoidal) distributed load.
func (wb *WinklerBeam3D) EquivalentNodalLoadLinear(globalDir [3]float64, intensityI, intensityJ float64) *mat.VecDense {
	L := wb.length
	L2 := L * L

	var qLocI, qLocJ [3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			qLocI[i] += wb.R[i][j] * globalDir[j]
			qLocJ[i] += wb.R[i][j] * globalDir[j]
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

	// Accumulate local fixed-end forces for EndForces post-processing
	for i := 0; i < 12; i++ {
		wb.fFixed[i] += fLoc.AtVec(i)
	}

	fGlob := mat.NewVecDense(12, nil)
	for blk := 0; blk < 4; blk++ {
		off := 3 * blk
		for i := 0; i < 3; i++ {
			var sum float64
			for j := 0; j < 3; j++ {
				sum += wb.R[j][i] * fLoc.AtVec(off+j)
			}
			fGlob.SetVec(off+i, sum)
		}
	}
	return fGlob
}
