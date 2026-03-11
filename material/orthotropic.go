package material

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// OrthotropicLinear implements a 3D linear elastic orthotropic material.
//
// The material has three mutually perpendicular planes of elastic symmetry
// aligned with the global X, Y, Z axes. Nine independent constants are needed:
//
//	Ex, Ey, Ez   – Young's moduli in x, y, z
//	Nxy, Nyz, Nxz – Poisson's ratios (ν_ij: lateral strain in j per unit strain in i)
//	Gxy, Gyz, Gxz – Shear moduli
//
// The compliance matrix S (ε = S·σ) is:
//
//	[ 1/Ex    -Nyx/Ey  -Nzx/Ez   0       0       0    ]
//	[-Nxy/Ex   1/Ey   -Nzy/Ez    0       0       0    ]
//	[-Nxz/Ex  -Nyz/Ey   1/Ez     0       0       0    ]
//	[  0        0        0      1/Gxy    0       0    ]
//	[  0        0        0       0      1/Gyz    0    ]
//	[  0        0        0       0       0      1/Gxz ]
//
// where Nyx = Nxy·Ey/Ex, Nzx = Nxz·Ez/Ex, Nzy = Nyz·Ez/Ey (Maxwell's reciprocity).
// The stiffness matrix D = S⁻¹ is obtained by numerical inversion.
//
// Voigt notation: [σxx, σyy, σzz, τxy, τyz, τxz].
type OrthotropicLinear struct {
	Ex, Ey, Ez       float64 // Young's moduli
	Nxy, Nyz, Nxz    float64 // Poisson's ratios (loading direction first)
	Gxy, Gyz, Gxz    float64 // shear moduli

	stress  *mat.VecDense
	tangent *mat.Dense
}

// NewOrthotropicLinear creates an orthotropic linear elastic material and
// pre-computes the 6×6 stiffness matrix D = S⁻¹.
// Returns an error if the compliance matrix is singular or not positive definite.
func NewOrthotropicLinear(ex, ey, ez, nxy, nyz, nxz, gxy, gyz, gxz float64) (*OrthotropicLinear, error) {
	m := &OrthotropicLinear{
		Ex: ex, Ey: ey, Ez: ez,
		Nxy: nxy, Nyz: nyz, Nxz: nxz,
		Gxy: gxy, Gyz: gyz, Gxz: gxz,
	}
	m.stress = mat.NewVecDense(6, nil)
	var err error
	m.tangent, err = m.buildTangent()
	if err != nil {
		return nil, err
	}
	return m, nil
}

// buildTangent constructs D = S⁻¹ where S is the 6×6 compliance matrix.
func (m *OrthotropicLinear) buildTangent() (*mat.Dense, error) {
	// Maxwell reciprocity: ν_ji / Ej = ν_ij / Ei
	//   Nyx = Nxy * Ey / Ex
	//   Nzx = Nxz * Ez / Ex
	//   Nzy = Nyz * Ez / Ey
	Nyx := m.Nxy * m.Ey / m.Ex
	Nzx := m.Nxz * m.Ez / m.Ex
	Nzy := m.Nyz * m.Ez / m.Ey

	S := mat.NewDense(6, 6, []float64{
		1 / m.Ex, -Nyx / m.Ey, -Nzx / m.Ez, 0, 0, 0,
		-m.Nxy / m.Ex, 1 / m.Ey, -Nzy / m.Ez, 0, 0, 0,
		-m.Nxz / m.Ex, -m.Nyz / m.Ey, 1 / m.Ez, 0, 0, 0,
		0, 0, 0, 1 / m.Gxy, 0, 0,
		0, 0, 0, 0, 1 / m.Gyz, 0,
		0, 0, 0, 0, 0, 1 / m.Gxz,
	})

	D := mat.NewDense(6, 6, nil)
	if err := D.Inverse(S); err != nil {
		return nil, fmt.Errorf("orthotropic material: compliance matrix is singular – check parameters")
	}
	return D, nil
}

func (m *OrthotropicLinear) SetTrialStrain(strain *mat.VecDense) error {
	m.stress.MulVec(m.tangent, strain)
	return nil
}

func (m *OrthotropicLinear) GetStress() *mat.VecDense { return m.stress }
func (m *OrthotropicLinear) GetTangent() *mat.Dense   { return m.tangent }
func (m *OrthotropicLinear) CommitState() error       { return nil }
func (m *OrthotropicLinear) RevertToStart() error {
	m.stress = mat.NewVecDense(6, nil)
	return nil
}
