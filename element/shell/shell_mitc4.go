// Package shell implements plate and shell elements.
package shell

import (
	"math"

	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// ShellMITC4 is a 4-node flat shell element with MITC interpolation for transverse shear.
// Combines Quad4 membrane with Mindlin plate bending.
// 6 DOFs per node (UX, UY, UZ, RX, RY, RZ), 24 DOFs total.
// Drilling DOF (RZ in local coords) handled with penalty stiffness.
type ShellMITC4 struct {
	ID     int
	Nds    [4]int
	Coords [4][3]float64 // 3D node coordinates
	E      float64
	Nu     float64
	Thick  float64

	ke *mat.Dense // 24×24 global stiffness
}

// NewShellMITC4 creates a 4-node flat shell element.
func NewShellMITC4(id int, nodes [4]int, coords [4][3]float64, e, nu, thick float64) *ShellMITC4 {
	s := &ShellMITC4{
		ID: id, Nds: nodes, Coords: coords,
		E: e, Nu: nu, Thick: thick,
	}
	s.formKe()
	return s
}

// formKe assembles the shell stiffness in global coordinates.
func (s *ShellMITC4) formKe() {
	// 1. Build local coordinate system
	e1, e2, e3 := s.localAxes()
	R := [3][3]float64{e1, e2, e3} // rows = local axes

	// 2. Transform node coords to local 2D (x,y)
	// Use centroid as origin
	var cx, cy, cz float64
	for i := 0; i < 4; i++ {
		cx += s.Coords[i][0]
		cy += s.Coords[i][1]
		cz += s.Coords[i][2]
	}
	cx /= 4
	cy /= 4
	cz /= 4

	var xl [4][2]float64 // local 2D coords
	for i := 0; i < 4; i++ {
		dx := s.Coords[i][0] - cx
		dy := s.Coords[i][1] - cy
		dz := s.Coords[i][2] - cz
		xl[i][0] = R[0][0]*dx + R[0][1]*dy + R[0][2]*dz // local x
		xl[i][1] = R[1][0]*dx + R[1][1]*dy + R[1][2]*dz // local y
	}

	// 3. Compute membrane + bending + shear stiffness in local coords
	Km := s.membraneKe(xl) // 8×8 (u,v per node)
	Kb := s.bendingKe(xl)  // 12×12 (w, θx, θy per node)

	// 4. Assemble into 24×24 local stiffness
	// Local DOF order per node: [u, v, w, θx, θy, θz]
	klocal := mat.NewDense(24, 24, nil)

	// Membrane DOFs: u=0, v=1 → local indices [0,1, 6,7, 12,13, 18,19]
	mmap := [8]int{0, 1, 6, 7, 12, 13, 18, 19}
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			klocal.Set(mmap[i], mmap[j], Km.At(i, j))
		}
	}

	// Bending DOFs: w=2, θx=3, θy=4 → local indices [2,3,4, 8,9,10, 14,15,16, 20,21,22]
	bmap := [12]int{2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22}
	for i := 0; i < 12; i++ {
		for j := 0; j < 12; j++ {
			klocal.Set(bmap[i], bmap[j], Kb.At(i, j))
		}
	}

	// Drilling DOF penalty: θz=5 → local indices [5, 11, 17, 23]
	drillPenalty := s.E * s.Thick * 1e-4 // small penalty
	for _, idx := range [4]int{5, 11, 17, 23} {
		klocal.Set(idx, idx, drillPenalty)
	}

	// 5. Transform to global: Ke = Tᵀ · Klocal · T
	T := s.buildTransform(R)
	tmp := mat.NewDense(24, 24, nil)
	tmp.Mul(klocal, T)
	s.ke = mat.NewDense(24, 24, nil)
	s.ke.Mul(T.T(), tmp)
}

// membraneKe returns the 8×8 plane-stress membrane stiffness (2×2 Gauss).
func (s *ShellMITC4) membraneKe(xl [4][2]float64) *mat.Dense {
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
				c := 2 * n
				B.Set(0, c, dx)
				B.Set(1, c+1, dy)
				B.Set(2, c, dy)
				B.Set(2, c+1, dx)
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
// Uses 2×2 Gauss for bending + selective 1-point reduced integration for shear.
func (s *ShellMITC4) bendingKe(xl [4][2]float64) *mat.Dense {
	t := s.Thick
	E, nu := s.E, s.Nu

	// Bending constitutive (3×3)
	cb := E * t * t * t / (12 * (1 - nu*nu))
	Db := mat.NewDense(3, 3, []float64{
		cb, cb * nu, 0,
		cb * nu, cb, 0,
		0, 0, cb * (1 - nu) / 2,
	})

	// Shear constitutive (2×2)
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

	// --- Bending part: 2×2 Gauss ---
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

			// Bending B matrix: DOFs [w, θx, θy] per node
			// κxx = ∂θy/∂x,  κyy = -∂θx/∂y,  κxy = ∂θy/∂y - ∂θx/∂x
			Bb.Zero()
			for n := 0; n < 4; n++ {
				dx, dy := dN.At(0, n), dN.At(1, n)
				c := 3 * n
				Bb.Set(0, c+2, dx)  // κxx = ∂θy/∂x
				Bb.Set(1, c+1, -dy) // κyy = -∂θx/∂y
				Bb.Set(2, c+1, -dx) // κxy = -∂θx/∂x + ∂θy/∂y
				Bb.Set(2, c+2, dy)
			}

			DBb.Mul(Db, Bb)
			BtDBb.Mul(Bb.T(), DBb)
			BtDBb.Scale(math.Abs(detJ), BtDBb)
			Kbend.Add(Kbend, BtDBb)
		}
	}

	// --- Shear part: 1×1 Gauss (selective reduced integration) ---
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

		// Shape functions at center
		var N [4]float64
		for i := 0; i < 4; i++ {
			si, ei := ref[i][0], ref[i][1]
			N[i] = (1 + si*xi) * (1 + ei*eta) / 4
		}

		// Shear B: γxz = ∂w/∂x + θy,  γyz = ∂w/∂y - θx
		for n := 0; n < 4; n++ {
			dx, dy := dN.At(0, n), dN.At(1, n)
			c := 3 * n
			Bs.Set(0, c, dx)     // ∂w/∂x
			Bs.Set(0, c+2, N[n]) // θy
			Bs.Set(1, c, dy)     // ∂w/∂y
			Bs.Set(1, c+1, -N[n]) // -θx
		}

		DBs.Mul(Ds, Bs)
		BtDBs.Mul(Bs.T(), DBs)
		// 1-point Gauss weight = 4.0 (2×2 domain)
		BtDBs.Scale(4.0*math.Abs(detJ), BtDBs)
		Kbend.Add(Kbend, BtDBs)
	}

	return Kbend
}

func (s *ShellMITC4) shapeDeriv2D(xi, eta float64, ref [][2]float64, dNnat *mat.Dense) {
	for i := 0; i < 4; i++ {
		si, ei := ref[i][0], ref[i][1]
		dNnat.Set(0, i, si*(1+ei*eta)/4)
		dNnat.Set(1, i, (1+si*xi)*ei/4)
	}
}

// localAxes computes the shell local coordinate system from the 4 node positions.
func (s *ShellMITC4) localAxes() (e1, e2, e3 [3]float64) {
	// g1 = ∂X/∂ξ at center ≈ 0.5*((x1+x2)-(x0+x3))
	// g2 = ∂X/∂η at center ≈ 0.5*((x2+x3)-(x0+x1))
	var g1, g2 [3]float64
	for k := 0; k < 3; k++ {
		g1[k] = 0.5 * ((s.Coords[1][k] + s.Coords[2][k]) - (s.Coords[0][k] + s.Coords[3][k]))
		g2[k] = 0.5 * ((s.Coords[2][k] + s.Coords[3][k]) - (s.Coords[0][k] + s.Coords[1][k]))
	}

	// e3 = g1 × g2 (normal)
	e3 = crossV(g1, g2)
	n3 := normV(e3)
	e3[0] /= n3
	e3[1] /= n3
	e3[2] /= n3

	// e1 = g1 normalised
	n1 := normV(g1)
	e1 = [3]float64{g1[0] / n1, g1[1] / n1, g1[2] / n1}

	// e2 = e3 × e1
	e2 = crossV(e3, e1)
	return
}

// buildTransform returns the 24×24 rotation matrix (block diagonal 4×[R3×3, R3×3]).
func (s *ShellMITC4) buildTransform(R [3][3]float64) *mat.Dense {
	T := mat.NewDense(24, 24, nil)
	for block := 0; block < 4; block++ {
		// Each node has 6 DOFs: 3 translations + 3 rotations
		// Both transformed by same R
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

func crossV(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

func normV(v [3]float64) float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}

// ---------- Element interface ----------

func (s *ShellMITC4) GetTangentStiffness() *mat.Dense  { return s.ke }
func (s *ShellMITC4) GetResistingForce() *mat.VecDense { return mat.NewVecDense(24, nil) }
func (s *ShellMITC4) NodeIDs() []int                   { return s.Nds[:] }
func (s *ShellMITC4) NumDOF() int                      { return 24 }
func (s *ShellMITC4) DOFPerNode() int                  { return 6 }
func (s *ShellMITC4) DOFTypes() []dof.Type             { return dof.Full6D(4) }
func (s *ShellMITC4) Update(_ []float64) error         { return nil }
func (s *ShellMITC4) CommitState() error               { return nil }
func (s *ShellMITC4) RevertToStart() error             { return nil }
