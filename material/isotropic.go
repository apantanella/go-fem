package material

import "gonum.org/v1/gonum/mat"

// IsotropicLinear implements a 3D linear elastic isotropic material.
type IsotropicLinear struct {
	E  float64 // Young's modulus
	Nu float64 // Poisson's ratio

	stress  *mat.VecDense
	tangent *mat.Dense
}

// NewIsotropicLinear creates a new isotropic linear elastic material.
func NewIsotropicLinear(e, nu float64) *IsotropicLinear {
	m := &IsotropicLinear{E: e, Nu: nu}
	m.stress = mat.NewVecDense(6, nil)
	m.tangent = m.buildTangent()
	return m
}

// buildTangent constructs the 6×6 elasticity matrix D.
//
//	D = (E / ((1+ν)(1-2ν))) * [ 1-ν   ν    ν   0        0        0       ]
//	                           [  ν  1-ν    ν   0        0        0       ]
//	                           [  ν    ν  1-ν   0        0        0       ]
//	                           [  0    0    0  (1-2ν)/2  0        0       ]
//	                           [  0    0    0   0       (1-2ν)/2  0       ]
//	                           [  0    0    0   0        0       (1-2ν)/2 ]
func (m *IsotropicLinear) buildTangent() *mat.Dense {
	E, nu := m.E, m.Nu
	c := E / ((1 + nu) * (1 - 2*nu))

	D := mat.NewDense(6, 6, nil)
	// Normal terms
	D.Set(0, 0, c*(1-nu))
	D.Set(0, 1, c*nu)
	D.Set(0, 2, c*nu)
	D.Set(1, 0, c*nu)
	D.Set(1, 1, c*(1-nu))
	D.Set(1, 2, c*nu)
	D.Set(2, 0, c*nu)
	D.Set(2, 1, c*nu)
	D.Set(2, 2, c*(1-nu))
	// Shear terms: G = E / (2(1+ν)) = c*(1-2ν)/2
	G := c * (1 - 2*nu) / 2
	D.Set(3, 3, G)
	D.Set(4, 4, G)
	D.Set(5, 5, G)
	return D
}

func (m *IsotropicLinear) SetTrialStrain(strain *mat.VecDense) error {
	// σ = D * ε
	m.stress.MulVec(m.tangent, strain)
	return nil
}

func (m *IsotropicLinear) GetStress() *mat.VecDense {
	return m.stress
}

func (m *IsotropicLinear) GetTangent() *mat.Dense {
	return m.tangent
}

func (m *IsotropicLinear) CommitState() error { return nil }

func (m *IsotropicLinear) RevertToStart() error {
	m.stress = mat.NewVecDense(6, nil)
	return nil
}
