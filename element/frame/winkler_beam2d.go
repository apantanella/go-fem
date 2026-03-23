package frame

import (
	"fmt"
	"math"

	"go-fem/dof"
	"go-fem/section"

	"gonum.org/v1/gonum/mat"
)

// WinklerBeam2D is a 2-node 2D Euler-Bernoulli beam element resting on a
// distributed Winkler elastic foundation (beam on elastic foundation).
//
// The total element stiffness is:
//
//	Ke = K_beam + K_winkler
//
// where K_winkler is the consistent spring stiffness matrix for the
// transverse (Y) direction, derived by integrating the product of Hermite
// shape functions:
//
//	K_w,ij = ks · b · ∫₀ᴸ Nᵢ(x)·Nⱼ(x) dx
//
// Analytical result (Hermite basis on [v_i, θz_i, v_j, θz_j]):
//
//	K_w = ks·b·L/420 · [156, 22L, 54, -13L; 22L, 4L², 13L, -3L²;
//	                     54, 13L, 156, -22L; -13L, -3L², -22L, 4L²]
//
// Parameters:
//   - Ks: Winkler subgrade reaction modulus [N/mm³] (force per area per displacement)
//   - B:  effective foundation width [mm] — often equal to beam flange width.
//     Combined stiffness per unit length q = Ks · B · v [N/mm per mm].
//     If B ≤ 0 it defaults to 1 (unit-width model or Ks already in N/mm²).
//
// DOF layout: 3 per node (UX, UY, RZ), total 6, same as ElasticBeam2D.
// The Winkler springs act only in the local Y (transverse) direction.
//
// Units: consistent (e.g. N-mm-MPa).
type WinklerBeam2D struct {
	ID     int
	Nds    [2]int
	Coords [2][2]float64
	E      float64 // Young's modulus
	Sec    section.BeamSection2D
	Ks     float64 // Winkler modulus [N/mm³]
	B      float64 // effective width [mm] (default 1 if ≤ 0)

	ke     *mat.Dense
	kl     *mat.Dense // local stiffness (for EndForces)
	length float64
	cos    float64
	sin    float64
	ue     [6]float64
}

// NewWinklerBeam2D creates a 2D beam on Winkler elastic foundation.
// ks is the subgrade reaction modulus (N/mm³); b is the effective width (mm).
// Set b ≤ 0 to use unit width (ks then interpreted as N/mm²).
func NewWinklerBeam2D(id int, nodes [2]int, coords [2][2]float64,
	e float64, sec section.BeamSection2D, ks, b float64) (*WinklerBeam2D, error) {

	if e <= 0 {
		return nil, fmt.Errorf("WinklerBeam2D: E must be > 0, got %g", e)
	}
	if ks < 0 {
		return nil, fmt.Errorf("WinklerBeam2D: Ks must be ≥ 0, got %g", ks)
	}
	if b <= 0 {
		b = 1
	}
	wb := &WinklerBeam2D{
		ID: id, Nds: nodes, Coords: coords,
		E: e, Sec: sec, Ks: ks, B: b,
	}
	wb.computeGeometry()
	wb.formKe()
	return wb, nil
}

func (wb *WinklerBeam2D) computeGeometry() {
	dx := wb.Coords[1][0] - wb.Coords[0][0]
	dy := wb.Coords[1][1] - wb.Coords[0][1]
	wb.length = math.Sqrt(dx*dx + dy*dy)
	wb.cos = dx / wb.length
	wb.sin = dy / wb.length
}

// formKe assembles Ke = Tᵀ·(K_beam_local + K_winkler_local)·T
func (wb *WinklerBeam2D) formKe() {
	L := wb.length
	E := wb.E
	A := wb.Sec.A
	Iz := wb.Sec.Iz
	L2 := L * L
	L3 := L2 * L

	kl := mat.NewDense(6, 6, nil)

	// --- Beam axial (DOFs 0, 3) ---
	ea := E * A / L
	kl.Set(0, 0, ea)
	kl.Set(0, 3, -ea)
	kl.Set(3, 0, -ea)
	kl.Set(3, 3, ea)

	// --- Beam bending in local x-y (DOFs: v=1,4; θz=2,5) ---
	v1 := 12 * E * Iz / L3
	v2 := 6 * E * Iz / L2
	v3 := 4 * E * Iz / L
	v4 := 2 * E * Iz / L

	addBeamBending2D(kl, v1, v2, v3, v4)

	// --- Winkler consistent spring (DOFs: v=1,θ=2,v=4,θ=5) ---
	// K_w = ks·b·L/420 · M where M is the Hermite mass matrix
	alpha := wb.Ks * wb.B * L / 420.0
	kl.Set(1, 1, kl.At(1, 1)+alpha*156)
	kl.Set(1, 2, kl.At(1, 2)+alpha*22*L)
	kl.Set(1, 4, kl.At(1, 4)+alpha*54)
	kl.Set(1, 5, kl.At(1, 5)+alpha*(-13*L))

	kl.Set(2, 1, kl.At(2, 1)+alpha*22*L)
	kl.Set(2, 2, kl.At(2, 2)+alpha*4*L2)
	kl.Set(2, 4, kl.At(2, 4)+alpha*13*L)
	kl.Set(2, 5, kl.At(2, 5)+alpha*(-3*L2))

	kl.Set(4, 1, kl.At(4, 1)+alpha*54)
	kl.Set(4, 2, kl.At(4, 2)+alpha*13*L)
	kl.Set(4, 4, kl.At(4, 4)+alpha*156)
	kl.Set(4, 5, kl.At(4, 5)+alpha*(-22*L))

	kl.Set(5, 1, kl.At(5, 1)+alpha*(-13*L))
	kl.Set(5, 2, kl.At(5, 2)+alpha*(-3*L2))
	kl.Set(5, 4, kl.At(5, 4)+alpha*(-22*L))
	kl.Set(5, 5, kl.At(5, 5)+alpha*4*L2)

	wb.kl = kl

	// --- Rotation to global ---
	c, s := wb.cos, wb.sin
	T := mat.NewDense(6, 6, nil)
	for blk := 0; blk < 2; blk++ {
		o := blk * 3
		T.Set(o, o, c)
		T.Set(o, o+1, s)
		T.Set(o+1, o, -s)
		T.Set(o+1, o+1, c)
		T.Set(o+2, o+2, 1)
	}
	tmp := mat.NewDense(6, 6, nil)
	tmp.Mul(kl, T)
	wb.ke = mat.NewDense(6, 6, nil)
	wb.ke.Mul(T.T(), tmp)
}

// addBeamBending2D inserts the standard E-B bending terms into kl.
func addBeamBending2D(kl *mat.Dense, v1, v2, v3, v4 float64) {
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
}

// ---------- Element interface ----------

func (wb *WinklerBeam2D) GetTangentStiffness() *mat.Dense { return wb.ke }

func (wb *WinklerBeam2D) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(6, nil)
	f.MulVec(wb.ke, mat.NewVecDense(6, wb.ue[:]))
	return f
}

func (wb *WinklerBeam2D) NodeIDs() []int       { return wb.Nds[:] }
func (wb *WinklerBeam2D) NumDOF() int          { return 6 }
func (wb *WinklerBeam2D) DOFPerNode() int      { return 3 }
func (wb *WinklerBeam2D) DOFTypes() []dof.Type { return dof.PlaneFrame(2) }

func (wb *WinklerBeam2D) Update(disp []float64) error {
	copy(wb.ue[:], disp)
	return nil
}

func (wb *WinklerBeam2D) CommitState() error   { return nil }
func (wb *WinklerBeam2D) RevertToStart() error { wb.ue = [6]float64{}; return nil }

// EndForces returns the section forces at both ends in local coordinates.
// [N, V, M] convention per end.
func (wb *WinklerBeam2D) EndForces() BeamEndForces2D {
	c, s := wb.cos, wb.sin
	// Transform global ue to local
	ul := [6]float64{
		c*wb.ue[0] + s*wb.ue[1],
		-s*wb.ue[0] + c*wb.ue[1],
		wb.ue[2],
		c*wb.ue[3] + s*wb.ue[4],
		-s*wb.ue[3] + c*wb.ue[4],
		wb.ue[5],
	}
	fl := mat.NewVecDense(6, nil)
	fl.MulVec(wb.kl, mat.NewVecDense(6, ul[:]))
	return BeamEndForces2D{
		I: [3]float64{fl.AtVec(0), fl.AtVec(1), fl.AtVec(2)},
		J: [3]float64{fl.AtVec(3), fl.AtVec(4), fl.AtVec(5)},
	}
}

// EquivalentNodalLoad converts a UDL (global direction, intensity per unit length)
// to work-equivalent nodal forces in global coordinates.
// Identical formula to ElasticBeam2D.
func (wb *WinklerBeam2D) EquivalentNodalLoad(globalDir [3]float64, intensity float64) *mat.VecDense {
	L := wb.length
	c, s := wb.cos, wb.sin
	// Project load direction onto local axes
	px := globalDir[0]*c + globalDir[1]*s  // local axial component
	py := -globalDir[0]*s + globalDir[1]*c // local transverse component
	px *= intensity
	py *= intensity
	// Fixed-end reactions in local coords
	floc := [6]float64{
		px * L / 2, py * L / 2, py * L * L / 12,
		px * L / 2, py * L / 2, -py * L * L / 12,
	}
	// Rotate to global
	fglob := mat.NewVecDense(6, nil)
	for n := 0; n < 2; n++ {
		o := n * 3
		fglob.SetVec(o, c*floc[o]-s*floc[o+1])
		fglob.SetVec(o+1, s*floc[o]+c*floc[o+1])
		fglob.SetVec(o+2, floc[o+2])
	}
	return fglob
}

// BodyForceLoad computes work-equivalent nodal forces due to self-weight or
// inertial body force (rho·A per unit length distributed uniformly).
func (wb *WinklerBeam2D) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	return wb.EquivalentNodalLoad(g, rho*wb.Sec.A)
}
