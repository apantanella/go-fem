// Package shell implements plate and shell elements.
package shell

import (
	"fmt"
	"math"

	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// DiscreteKirchhoffTriangle is a 3-node plate bending triangle (DKT-style).
//
// Active nodal DOFs are [UZ, RX, RY] per node (9 total). The element is embedded
// in the global 6-DOF-per-node layout by assembling a full 18×18 matrix and using
// small stabilization terms on inactive [UX, UY, RZ] DOFs.
type DiscreteKirchhoffTriangle struct {
	ID     int
	Nds    [3]int
	Coords [3][3]float64
	E      float64
	Nu     float64
	Thick  float64

	ke *mat.Dense  // 18×18 global stiffness
	ue [18]float64 // element displacement vector (global, set by Update)
}

// NewDiscreteKirchhoffTriangle creates a 3-node DKT-style plate element.
func NewDiscreteKirchhoffTriangle(id int, nodes [3]int, coords [3][3]float64, e, nu, thick float64) *DiscreteKirchhoffTriangle {
	d := &DiscreteKirchhoffTriangle{
		ID: id, Nds: nodes, Coords: coords,
		E: e, Nu: nu, Thick: thick,
	}
	d.formKe()
	return d
}

func (d *DiscreteKirchhoffTriangle) formKe() {
	e1, e2, e3, err := d.localAxes()
	if err != nil {
		panic(fmt.Sprintf("dkt3: %v", err))
	}
	R := [3][3]float64{e1, e2, e3}

	// Local 2D coordinates (origin at node 0)
	origin := d.Coords[0]
	var xl [3][2]float64
	for i := 0; i < 3; i++ {
		dx := d.Coords[i][0] - origin[0]
		dy := d.Coords[i][1] - origin[1]
		dz := d.Coords[i][2] - origin[2]
		xl[i][0] = R[0][0]*dx + R[0][1]*dy + R[0][2]*dz
		xl[i][1] = R[1][0]*dx + R[1][1]*dy + R[1][2]*dz
	}

	K9 := d.localPlateKe(xl)

	// Expand 9×9 active plate stiffness into full 18×18 local shell-like layout.
	// Per node local DOFs: [u, v, w, rx, ry, rz].
	klocal := mat.NewDense(18, 18, nil)
	map9to18 := [9]int{2, 3, 4, 8, 9, 10, 14, 15, 16}
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			klocal.Set(map9to18[i], map9to18[j], K9.At(i, j))
		}
	}

	// Small penalties on inactive in-plane/drilling DOFs to avoid singular free modes
	// when the model contains only plate-bending elements.
	stab := d.E * d.Thick * 1e-6
	for _, idx := range [9]int{0, 1, 5, 6, 7, 11, 12, 13, 17} {
		klocal.Set(idx, idx, klocal.At(idx, idx)+stab)
	}

	T := d.buildTransform(R)
	tmp := mat.NewDense(18, 18, nil)
	tmp.Mul(klocal, T)
	d.ke = mat.NewDense(18, 18, nil)
	d.ke.Mul(T.T(), tmp)
}

func (d *DiscreteKirchhoffTriangle) localPlateKe(xl [3][2]float64) *mat.Dense {
	x1, y1 := xl[0][0], xl[0][1]
	x2, y2 := xl[1][0], xl[1][1]
	x3, y3 := xl[2][0], xl[2][1]

	twoA := (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
	A := 0.5 * twoA
	if math.Abs(A) < 1e-14 {
		panic("dkt3: degenerate triangle area")
	}

	b := [3]float64{y2 - y3, y3 - y1, y1 - y2}
	c := [3]float64{x3 - x2, x1 - x3, x2 - x1}

	dNdx := [3]float64{b[0] / twoA, b[1] / twoA, b[2] / twoA}
	dNdy := [3]float64{c[0] / twoA, c[1] / twoA, c[2] / twoA}

	E, nu, t := d.E, d.Nu, d.Thick
	cb := E * t * t * t / (12 * (1 - nu*nu))
	Db := mat.NewDense(3, 3, []float64{
		cb, cb * nu, 0,
		cb * nu, cb, 0,
		0, 0, cb * (1 - nu) / 2,
	})

	// Bending curvatures from nodal rotations (constant over linear triangle).
	// Local active ordering per node: [w, rx, ry].
	Bb := mat.NewDense(3, 9, nil)
	for i := 0; i < 3; i++ {
		col := 3 * i
		Bb.Set(0, col+2, dNdx[i])  // kxx = d(ry)/dx
		Bb.Set(1, col+1, -dNdy[i]) // kyy = -d(rx)/dy
		Bb.Set(2, col+1, -dNdx[i]) // kxy = d(ry)/dy - d(rx)/dx
		Bb.Set(2, col+2, dNdy[i])
	}

	DBb := mat.NewDense(3, 9, nil)
	DBb.Mul(Db, Bb)
	Kb := mat.NewDense(9, 9, nil)
	Kb.Mul(Bb.T(), DBb)
	Kb.Scale(math.Abs(A), Kb)

	// Discrete Kirchhoff constraints (penalty):
	// gamma_xz = dw/dx + ry = 0
	// gamma_yz = dw/dy - rx = 0
	alpha := E * t
	wp := math.Abs(A) / 3.0
	for i := 0; i < 3; i++ {
		gx := make([]float64, 9)
		gy := make([]float64, 9)
		for j := 0; j < 3; j++ {
			gx[3*j] = dNdx[j]
			gy[3*j] = dNdy[j]
		}
		gx[3*i+2] += 1.0
		gy[3*i+1] += -1.0

		for r := 0; r < 9; r++ {
			for c2 := 0; c2 < 9; c2++ {
				v := Kb.At(r, c2) + alpha*wp*(gx[r]*gx[c2]+gy[r]*gy[c2])
				Kb.Set(r, c2, v)
			}
		}
	}

	return Kb
}

func (d *DiscreteKirchhoffTriangle) localAxes() (e1, e2, e3 [3]float64, err error) {
	v1 := [3]float64{
		d.Coords[1][0] - d.Coords[0][0],
		d.Coords[1][1] - d.Coords[0][1],
		d.Coords[1][2] - d.Coords[0][2],
	}
	v2 := [3]float64{
		d.Coords[2][0] - d.Coords[0][0],
		d.Coords[2][1] - d.Coords[0][1],
		d.Coords[2][2] - d.Coords[0][2],
	}
	n1 := normV(v1)
	if n1 < 1e-14 {
		return e1, e2, e3, fmt.Errorf("zero-length edge")
	}
	e1 = [3]float64{v1[0] / n1, v1[1] / n1, v1[2] / n1}

	e3 = crossV(v1, v2)
	n3 := normV(e3)
	if n3 < 1e-14 {
		return e1, e2, e3, fmt.Errorf("collinear triangle nodes")
	}
	e3 = [3]float64{e3[0] / n3, e3[1] / n3, e3[2] / n3}

	e2 = crossV(e3, e1)
	return e1, e2, e3, nil
}

func (d *DiscreteKirchhoffTriangle) buildTransform(R [3][3]float64) *mat.Dense {
	T := mat.NewDense(18, 18, nil)
	for n := 0; n < 3; n++ {
		for sub := 0; sub < 2; sub++ {
			off := n*6 + sub*3
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

func (d *DiscreteKirchhoffTriangle) GetTangentStiffness() *mat.Dense { return d.ke }

func (d *DiscreteKirchhoffTriangle) GetResistingForce() *mat.VecDense {
	f := mat.NewVecDense(18, nil)
	f.MulVec(d.ke, mat.NewVecDense(18, d.ue[:]))
	return f
}

func (d *DiscreteKirchhoffTriangle) NodeIDs() []int       { return d.Nds[:] }
func (d *DiscreteKirchhoffTriangle) NumDOF() int          { return 18 }
func (d *DiscreteKirchhoffTriangle) DOFPerNode() int      { return 6 }
func (d *DiscreteKirchhoffTriangle) DOFTypes() []dof.Type { return dof.Full6D(3) }

func (d *DiscreteKirchhoffTriangle) Update(disp []float64) error { copy(d.ue[:], disp); return nil }

func (d *DiscreteKirchhoffTriangle) CommitState() error   { return nil }
func (d *DiscreteKirchhoffTriangle) RevertToStart() error { d.ue = [18]float64{}; return nil }

// LocalMoments returns centroidal bending moments in the local plate frame.
func (d *DiscreteKirchhoffTriangle) LocalMoments() (mx, my, mxy float64) {
	e1, e2, e3, err := d.localAxes()
	if err != nil {
		return 0, 0, 0
	}
	R := [3][3]float64{e1, e2, e3}

	var uloc [18]float64
	for n := 0; n < 3; n++ {
		for sub := 0; sub < 2; sub++ {
			off := n*6 + sub*3
			for i := 0; i < 3; i++ {
				for j := 0; j < 3; j++ {
					uloc[off+i] += R[i][j] * d.ue[off+j]
				}
			}
		}
	}

	origin := d.Coords[0]
	var xl [3][2]float64
	for i := 0; i < 3; i++ {
		dx := d.Coords[i][0] - origin[0]
		dy := d.Coords[i][1] - origin[1]
		dz := d.Coords[i][2] - origin[2]
		xl[i][0] = R[0][0]*dx + R[0][1]*dy + R[0][2]*dz
		xl[i][1] = R[1][0]*dx + R[1][1]*dy + R[1][2]*dz
	}

	twoA := (xl[1][0]-xl[0][0])*(xl[2][1]-xl[0][1]) - (xl[2][0]-xl[0][0])*(xl[1][1]-xl[0][1])
	if math.Abs(twoA) < 1e-14 {
		return 0, 0, 0
	}
	b := [3]float64{xl[1][1] - xl[2][1], xl[2][1] - xl[0][1], xl[0][1] - xl[1][1]}
	c := [3]float64{xl[2][0] - xl[1][0], xl[0][0] - xl[2][0], xl[1][0] - xl[0][0]}
	dNdx := [3]float64{b[0] / twoA, b[1] / twoA, b[2] / twoA}
	dNdy := [3]float64{c[0] / twoA, c[1] / twoA, c[2] / twoA}

	Bb := mat.NewDense(3, 9, nil)
	for i := 0; i < 3; i++ {
		col := 3 * i
		Bb.Set(0, col+2, dNdx[i])
		Bb.Set(1, col+1, -dNdy[i])
		Bb.Set(2, col+1, -dNdx[i])
		Bb.Set(2, col+2, dNdy[i])
	}

	u9 := [9]float64{}
	for i := 0; i < 3; i++ {
		u9[3*i+0] = uloc[6*i+2]
		u9[3*i+1] = uloc[6*i+3]
		u9[3*i+2] = uloc[6*i+4]
	}

	kappa := mat.NewVecDense(3, nil)
	kappa.MulVec(Bb, mat.NewVecDense(9, u9[:]))

	E, nu, t := d.E, d.Nu, d.Thick
	cb := E * t * t * t / (12 * (1 - nu*nu))
	Db := mat.NewDense(3, 3, []float64{
		cb, cb * nu, 0,
		cb * nu, cb, 0,
		0, 0, cb * (1 - nu) / 2,
	})
	M := mat.NewVecDense(3, nil)
	M.MulVec(Db, kappa)
	return M.AtVec(0), M.AtVec(1), M.AtVec(2)
}
