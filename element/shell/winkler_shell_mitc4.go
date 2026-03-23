package shell

import (
	"math"

	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// WinklerShellMITC4 is a 4-node flat shell element with Winkler elastic foundation.
// It combines the full MITC4 shell stiffness (membrane + bending + shear) with
// a distributed vertical spring representing the soil reaction (Winkler model).
//
// The Winkler spring contribution is:
//
//	K_w_ij = Ks · ∫∫ N_i(ξ,η) · N_j(ξ,η) · |J| dξ dη
//
// computed by 2×2 Gauss quadrature over the element. The springs act on the
// UZ (transverse) DOF of each node, corresponding to the shell normal direction.
//
// Parameters:
//   - Ks: Winkler subgrade reaction modulus [N/mm³]
//     (force per area per normal displacement)
//
// DOF layout: 6 per node (UX, UY, UZ, RX, RY, RZ), total 24, same as ShellMITC4.
// Units: consistent (e.g. N-mm-MPa).
type WinklerShellMITC4 struct {
	ID     int
	Nds    [4]int
	Coords [4][3]float64
	E      float64
	Nu     float64
	Thick  float64
	Ks     float64 // Winkler subgrade reaction modulus [N/mm³]

	ke *mat.Dense  // 24×24 global stiffness
	ue [24]float64 // element displacements (set by Update)
}

// NewWinklerShellMITC4 creates a 4-node flat shell element on Winkler foundation.
// ks is the subgrade reaction modulus [N/mm³]; set ks = 0 for no foundation.
func NewWinklerShellMITC4(id int, nodes [4]int, coords [4][3]float64,
	e, nu, thick, ks float64) *WinklerShellMITC4 {
	s := &WinklerShellMITC4{
		ID: id, Nds: nodes, Coords: coords,
		E: e, Nu: nu, Thick: thick, Ks: ks,
	}
	s.formKe()
	return s
}

// formKe assembles the shell + Winkler stiffness in global coordinates.
func (s *WinklerShellMITC4) formKe() {
	// 1. Local coordinate system
	e1, e2, e3 := s.localAxes()
	R := [3][3]float64{e1, e2, e3}

	// 2. Node coordinates in local 2D
	var cx, cy, cz float64
	for i := 0; i < 4; i++ {
		cx += s.Coords[i][0]
		cy += s.Coords[i][1]
		cz += s.Coords[i][2]
	}
	cx /= 4
	cy /= 4
	cz /= 4

	var xl [4][2]float64
	for i := 0; i < 4; i++ {
		dx := s.Coords[i][0] - cx
		dy := s.Coords[i][1] - cy
		dz := s.Coords[i][2] - cz
		xl[i][0] = R[0][0]*dx + R[0][1]*dy + R[0][2]*dz
		xl[i][1] = R[1][0]*dx + R[1][1]*dy + R[1][2]*dz
	}

	// 3. Assemble 24×24 local stiffness (reuse ShellMITC4 sub-routines via helper)
	klocal := s.shellKlocal(xl)

	// 4. Add Winkler contribution at UZ DOF positions (2, 8, 14, 20 in local frame)
	if s.Ks != 0 {
		kw := s.winklerKe(xl)
		uzmap := [4]int{2, 8, 14, 20}
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				klocal.Set(uzmap[i], uzmap[j], klocal.At(uzmap[i], uzmap[j])+kw.At(i, j))
			}
		}
	}

	// 5. Transform to global: Ke = Tᵀ · klocal · T
	T := s.buildTransform(R)
	tmp := mat.NewDense(24, 24, nil)
	tmp.Mul(klocal, T)
	s.ke = mat.NewDense(24, 24, nil)
	s.ke.Mul(T.T(), tmp)
}

// shellKlocal assembles the 24×24 local shell stiffness matrix (same as ShellMITC4).
func (s *WinklerShellMITC4) shellKlocal(xl [4][2]float64) *mat.Dense {
	Km := s.membraneKe(xl)
	Kb := s.bendingKe(xl)

	klocal := mat.NewDense(24, 24, nil)

	// Membrane DOFs: u=0, v=1 per node → [0,1, 6,7, 12,13, 18,19]
	mmap := [8]int{0, 1, 6, 7, 12, 13, 18, 19}
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			klocal.Set(mmap[i], mmap[j], Km.At(i, j))
		}
	}

	// Bending DOFs: w=2, θx=3, θy=4 per node → [2,3,4, 8,9,10, 14,15,16, 20,21,22]
	bmap := [12]int{2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22}
	for i := 0; i < 12; i++ {
		for j := 0; j < 12; j++ {
			klocal.Set(bmap[i], bmap[j], Kb.At(i, j))
		}
	}

	// Drilling DOF penalty: θz=5 → [5, 11, 17, 23]
	drillPenalty := s.E * s.Thick * 1e-4
	for _, idx := range [4]int{5, 11, 17, 23} {
		klocal.Set(idx, idx, drillPenalty)
	}
	return klocal
}

// winklerKe computes the 4×4 UZ Winkler spring stiffness using 2×2 Gauss integration.
//
//	K_w_ij = Ks · ∫∫ N_i · N_j · |J| dξ dη
//
// where N_k are bilinear shape functions. Node order: (-1,-1),(1,-1),(1,1),(-1,1).
func (s *WinklerShellMITC4) winklerKe(xl [4][2]float64) *mat.Dense {
	kw := mat.NewDense(4, 4, nil)
	gp := 1.0 / math.Sqrt(3.0)
	pts := [2]float64{-gp, gp}
	ref := [4][2]float64{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}

	for _, xi := range pts {
		for _, eta := range pts {
			// Bilinear shape functions
			var N [4]float64
			for k := 0; k < 4; k++ {
				sk, ek := ref[k][0], ref[k][1]
				N[k] = (1 + sk*xi) * (1 + ek*eta) / 4
			}
			// Jacobian
			var j00, j01, j10, j11 float64
			for k := 0; k < 4; k++ {
				sk, ek := ref[k][0], ref[k][1]
				dxi := sk * (1 + ek*eta) / 4
				deta := (1 + sk*xi) * ek / 4
				j00 += dxi * xl[k][0]
				j01 += dxi * xl[k][1]
				j10 += deta * xl[k][0]
				j11 += deta * xl[k][1]
			}
			detJ := j00*j11 - j01*j10
			scale := s.Ks * math.Abs(detJ)
			for i := 0; i < 4; i++ {
				for j := 0; j < 4; j++ {
					kw.Set(i, j, kw.At(i, j)+scale*N[i]*N[j])
				}
			}
		}
	}
	return kw
}

// membraneKe returns the 8×8 plane-stress membrane stiffness (2×2 Gauss).
func (s *WinklerShellMITC4) membraneKe(xl [4][2]float64) *mat.Dense {
	t := s.Thick
	E, nu := s.E, s.Nu
	c := E / (1 - nu*nu)
	D := mat.NewDense(3, 3, []float64{
		c, c * nu, 0,
		c * nu, c, 0,
		0, 0, c * (1 - nu) / 2,
	})

	Km := mat.NewDense(8, 8, nil)
	gp := 1.0 / math.Sqrt(3.0)
	pts := [2]float64{-gp, gp}
	ref := [4][2]float64{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}

	X := mat.NewDense(4, 2, nil)
	for i := 0; i < 4; i++ {
		X.Set(i, 0, xl[i][0])
		X.Set(i, 1, xl[i][1])
	}

	dNnat := mat.NewDense(2, 4, nil)
	J := mat.NewDense(2, 2, nil)
	dN := mat.NewDense(2, 4, nil)
	B := mat.NewDense(3, 8, nil)
	DB := mat.NewDense(3, 8, nil)
	BtDB := mat.NewDense(8, 8, nil)
	var Jinv mat.Dense

	for _, xi := range pts {
		for _, eta := range pts {
			for i := 0; i < 4; i++ {
				si, ei := ref[i][0], ref[i][1]
				dNnat.Set(0, i, si*(1+ei*eta)/4)
				dNnat.Set(1, i, (1+si*xi)*ei/4)
			}
			J.Mul(dNnat, X)
			detJ := J.At(0, 0)*J.At(1, 1) - J.At(0, 1)*J.At(1, 0)
			Jinv.Inverse(J)
			dN.Mul(&Jinv, dNnat)

			B.Zero()
			for n := 0; n < 4; n++ {
				dx, dy := dN.At(0, n), dN.At(1, n)
				col := 2 * n
				B.Set(0, col, dx)
				B.Set(1, col+1, dy)
				B.Set(2, col, dy)
				B.Set(2, col+1, dx)
			}

			DB.Mul(D, B)
			BtDB.Mul(B.T(), DB)
			BtDB.Scale(t*math.Abs(detJ), BtDB)
			Km.Add(Km, BtDB)
		}
	}
	return Km
}

// bendingKe returns the 12×12 Mindlin plate bending stiffness.
func (s *WinklerShellMITC4) bendingKe(xl [4][2]float64) *mat.Dense {
	t := s.Thick
	E, nu := s.E, s.Nu

	cb := E * t * t * t / (12 * (1 - nu*nu))
	Db := mat.NewDense(3, 3, []float64{
		cb, cb * nu, 0,
		cb * nu, cb, 0,
		0, 0, cb * (1 - nu) / 2,
	})

	kappa := 5.0 / 6.0
	G := E / (2 * (1 + nu))
	cs := kappa * G * t
	Ds := mat.NewDense(2, 2, []float64{cs, 0, 0, cs})

	Kbend := mat.NewDense(12, 12, nil)
	ref := [4][2]float64{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}

	X := mat.NewDense(4, 2, nil)
	for i := 0; i < 4; i++ {
		X.Set(i, 0, xl[i][0])
		X.Set(i, 1, xl[i][1])
	}

	dNnat := mat.NewDense(2, 4, nil)
	J := mat.NewDense(2, 2, nil)
	dN := mat.NewDense(2, 4, nil)
	var Jinv mat.Dense

	gp := 1.0 / math.Sqrt(3.0)
	pts := [2]float64{-gp, gp}

	Bb := mat.NewDense(3, 12, nil)
	DBb := mat.NewDense(3, 12, nil)
	BtDBb := mat.NewDense(12, 12, nil)

	for _, xi := range pts {
		for _, eta := range pts {
			s.shapeDeriv2D(xi, eta, ref[:], dNnat)
			J.Mul(dNnat, X)
			detJ := J.At(0, 0)*J.At(1, 1) - J.At(0, 1)*J.At(1, 0)
			Jinv.Inverse(J)
			dN.Mul(&Jinv, dNnat)

			Bb.Zero()
			for n := 0; n < 4; n++ {
				dx, dy := dN.At(0, n), dN.At(1, n)
				col := 3 * n
				Bb.Set(0, col+2, dx)
				Bb.Set(1, col+1, -dy)
				Bb.Set(2, col+1, -dx)
				Bb.Set(2, col+2, dy)
			}

			DBb.Mul(Db, Bb)
			BtDBb.Mul(Bb.T(), DBb)
			BtDBb.Scale(math.Abs(detJ), BtDBb)
			Kbend.Add(Kbend, BtDBb)
		}
	}

	// Shear part: 1-point reduced integration
	Bs := mat.NewDense(2, 12, nil)
	DBs := mat.NewDense(2, 12, nil)
	BtDBs := mat.NewDense(12, 12, nil)
	{
		xi, eta := 0.0, 0.0
		s.shapeDeriv2D(xi, eta, ref[:], dNnat)
		J.Mul(dNnat, X)
		detJ := J.At(0, 0)*J.At(1, 1) - J.At(0, 1)*J.At(1, 0)
		Jinv.Inverse(J)
		dN.Mul(&Jinv, dNnat)

		var N [4]float64
		for i := 0; i < 4; i++ {
			si, ei := ref[i][0], ref[i][1]
			N[i] = (1 + si*xi) * (1 + ei*eta) / 4
		}
		for n := 0; n < 4; n++ {
			dx, dy := dN.At(0, n), dN.At(1, n)
			col := 3 * n
			Bs.Set(0, col, dx)
			Bs.Set(0, col+2, N[n])
			Bs.Set(1, col, dy)
			Bs.Set(1, col+1, -N[n])
		}
		DBs.Mul(Ds, Bs)
		BtDBs.Mul(Bs.T(), DBs)
		BtDBs.Scale(4.0*math.Abs(detJ), BtDBs)
		Kbend.Add(Kbend, BtDBs)
	}
	return Kbend
}

func (s *WinklerShellMITC4) shapeDeriv2D(xi, eta float64, ref [][2]float64, dNnat *mat.Dense) {
	for i := 0; i < 4; i++ {
		si, ei := ref[i][0], ref[i][1]
		dNnat.Set(0, i, si*(1+ei*eta)/4)
		dNnat.Set(1, i, (1+si*xi)*ei/4)
	}
}

// localAxes computes the local coordinate system from the node positions.
func (s *WinklerShellMITC4) localAxes() (e1, e2, e3 [3]float64) {
	var g1, g2 [3]float64
	for k := 0; k < 3; k++ {
		g1[k] = 0.5 * ((s.Coords[1][k] + s.Coords[2][k]) - (s.Coords[0][k] + s.Coords[3][k]))
		g2[k] = 0.5 * ((s.Coords[2][k] + s.Coords[3][k]) - (s.Coords[0][k] + s.Coords[1][k]))
	}
	e3 = crossV(g1, g2)
	n3 := normV(e3)
	e3[0] /= n3
	e3[1] /= n3
	e3[2] /= n3
	n1 := normV(g1)
	e1 = [3]float64{g1[0] / n1, g1[1] / n1, g1[2] / n1}
	e2 = crossV(e3, e1)
	return
}

// buildTransform returns the 24×24 rotation matrix (block diagonal).
func (s *WinklerShellMITC4) buildTransform(R [3][3]float64) *mat.Dense {
	T := mat.NewDense(24, 24, nil)
	for block := 0; block < 4; block++ {
		for sub := 0; sub < 2; sub++ {
			off := block*6 + sub*3
			for i := 0; i < 3; i++ {
				for j := 0; j < 3; j++ {
					T.Set(off+i, off+j, R[i][j])
				}
			}
		}
	}
	return T
}

// ---------- Element interface ----------

func (s *WinklerShellMITC4) GetTangentStiffness() *mat.Dense { return s.ke }

// GetResistingForce returns Ke·ue (internal nodal force vector in global coords).
func (s *WinklerShellMITC4) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(24, nil)
	f.MulVec(s.ke, mat.NewVecDense(24, s.ue[:]))
	return f
}

func (s *WinklerShellMITC4) NodeIDs() []int       { return s.Nds[:] }
func (s *WinklerShellMITC4) NumDOF() int          { return 24 }
func (s *WinklerShellMITC4) DOFPerNode() int      { return 6 }
func (s *WinklerShellMITC4) DOFTypes() []dof.Type { return dof.Full6D(4) }

func (s *WinklerShellMITC4) Update(disp []float64) error { copy(s.ue[:], disp); return nil }

func (s *WinklerShellMITC4) CommitState() error   { return nil }
func (s *WinklerShellMITC4) RevertToStart() error { s.ue = [24]float64{}; return nil }

// LocalForces computes the section forces and moments at the element centroid
// in local shell coordinates. Winkler spring reactions are distributed and not
// included in the section forces (Nx, Ny, Nxy, Mx, My, Mxy).
func (s *WinklerShellMITC4) LocalForces() ShellForces {
	e1, e2, e3 := s.localAxes()
	Rm := [3][3]float64{e1, e2, e3}

	var cx, cy, cz float64
	for i := 0; i < 4; i++ {
		cx += s.Coords[i][0]
		cy += s.Coords[i][1]
		cz += s.Coords[i][2]
	}
	cx /= 4
	cy /= 4
	cz /= 4

	var xl [4][2]float64
	for i := 0; i < 4; i++ {
		dx := s.Coords[i][0] - cx
		dy := s.Coords[i][1] - cy
		dz := s.Coords[i][2] - cz
		xl[i][0] = Rm[0][0]*dx + Rm[0][1]*dy + Rm[0][2]*dz
		xl[i][1] = Rm[1][0]*dx + Rm[1][1]*dy + Rm[1][2]*dz
	}

	var uloc [24]float64
	for blk := 0; blk < 4; blk++ {
		for sub := 0; sub < 2; sub++ {
			off := blk*6 + sub*3
			for i := 0; i < 3; i++ {
				for j := 0; j < 3; j++ {
					uloc[off+i] += Rm[i][j] * s.ue[off+j]
				}
			}
		}
	}

	ref := [4][2]float64{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}
	X := mat.NewDense(4, 2, nil)
	for i := 0; i < 4; i++ {
		X.Set(i, 0, xl[i][0])
		X.Set(i, 1, xl[i][1])
	}
	dNnat := mat.NewDense(2, 4, nil)
	s.shapeDeriv2D(0, 0, ref[:], dNnat)
	J2 := mat.NewDense(2, 2, nil)
	J2.Mul(dNnat, X)
	var Jinv2 mat.Dense
	Jinv2.Inverse(J2)
	dN := mat.NewDense(2, 4, nil)
	dN.Mul(&Jinv2, dNnat)

	var umem [8]float64
	for n := 0; n < 4; n++ {
		umem[2*n] = uloc[n*6+0]
		umem[2*n+1] = uloc[n*6+1]
	}
	var ubend [12]float64
	for n := 0; n < 4; n++ {
		ubend[3*n] = uloc[n*6+2]
		ubend[3*n+1] = uloc[n*6+3]
		ubend[3*n+2] = uloc[n*6+4]
	}

	Bm := mat.NewDense(3, 8, nil)
	for n := 0; n < 4; n++ {
		dx, dy := dN.At(0, n), dN.At(1, n)
		col := 2 * n
		Bm.Set(0, col, dx)
		Bm.Set(1, col+1, dy)
		Bm.Set(2, col, dy)
		Bm.Set(2, col+1, dx)
	}
	E, nu, t := s.E, s.Nu, s.Thick
	cp := E / (1 - nu*nu)
	Dm := mat.NewDense(3, 3, []float64{cp, cp * nu, 0, cp * nu, cp, 0, 0, 0, cp * (1 - nu) / 2})
	eps := mat.NewVecDense(3, nil)
	eps.MulVec(Bm, mat.NewVecDense(8, umem[:]))
	Nf := mat.NewVecDense(3, nil)
	Nf.MulVec(Dm, eps)

	Bb := mat.NewDense(3, 12, nil)
	for n := 0; n < 4; n++ {
		dx, dy := dN.At(0, n), dN.At(1, n)
		col := 3 * n
		Bb.Set(0, col+2, dx)
		Bb.Set(1, col+1, -dy)
		Bb.Set(2, col+1, -dx)
		Bb.Set(2, col+2, dy)
	}
	cb := E * t * t * t / (12 * (1 - nu*nu))
	Db := mat.NewDense(3, 3, []float64{cb, cb * nu, 0, cb * nu, cb, 0, 0, 0, cb * (1 - nu) / 2})
	kappa := mat.NewVecDense(3, nil)
	kappa.MulVec(Bb, mat.NewVecDense(12, ubend[:]))
	Mf := mat.NewVecDense(3, nil)
	Mf.MulVec(Db, kappa)

	return ShellForces{
		Nx: t * Nf.AtVec(0), Ny: t * Nf.AtVec(1), Nxy: t * Nf.AtVec(2),
		Mx: Mf.AtVec(0), My: Mf.AtVec(1), Mxy: Mf.AtVec(2),
	}
}

// BodyForceLoad computes work-equivalent nodal forces due to a body force.
// Uses 2×2 Gauss integration. Translational DOFs only; rotational DOFs = 0.
func (s *WinklerShellMITC4) BodyForceLoad(g [3]float64, rho float64) *mat.VecDense {
	f := mat.NewVecDense(24, nil)

	e1, e2, _ := s.localAxes()

	var cx, cy, cz float64
	for i := 0; i < 4; i++ {
		cx += s.Coords[i][0]
		cy += s.Coords[i][1]
		cz += s.Coords[i][2]
	}
	cx /= 4
	cy /= 4
	cz /= 4

	var xl [4][2]float64
	for i := 0; i < 4; i++ {
		dx := s.Coords[i][0] - cx
		dy := s.Coords[i][1] - cy
		dz := s.Coords[i][2] - cz
		xl[i][0] = e1[0]*dx + e1[1]*dy + e1[2]*dz
		xl[i][1] = e2[0]*dx + e2[1]*dy + e2[2]*dz
	}

	gp := 1.0 / math.Sqrt(3.0)
	ref := [4][2]float64{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}
	pts := [2]float64{-gp, gp}

	for _, xi := range pts {
		for _, eta := range pts {
			var N [4]float64
			var j00, j01, j10, j11 float64
			for i := 0; i < 4; i++ {
				si, ei := ref[i][0], ref[i][1]
				N[i] = (1 + si*xi) * (1 + ei*eta) / 4
				dxi := si * (1 + ei*eta) / 4
				deta := (1 + si*xi) * ei / 4
				j00 += dxi * xl[i][0]
				j01 += dxi * xl[i][1]
				j10 += deta * xl[i][0]
				j11 += deta * xl[i][1]
			}
			detJ := j00*j11 - j01*j10
			scale := rho * s.Thick * math.Abs(detJ)
			for n := 0; n < 4; n++ {
				f.SetVec(6*n, f.AtVec(6*n)+scale*N[n]*g[0])
				f.SetVec(6*n+1, f.AtVec(6*n+1)+scale*N[n]*g[1])
				f.SetVec(6*n+2, f.AtVec(6*n+2)+scale*N[n]*g[2])
			}
		}
	}
	return f
}
