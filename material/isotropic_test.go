package material_test

import (
	"math"
	"testing"

	"go-fem/material"
	"gonum.org/v1/gonum/mat"
)

// TestIsotropicSymmetry verifies that D is symmetric.
func TestIsotropicSymmetry(t *testing.T) {
	m := material.NewIsotropicLinear(210000, 0.3)
	D := m.GetTangent()
	for i := 0; i < 6; i++ {
		for j := i + 1; j < 6; j++ {
			if D.At(i, j) != D.At(j, i) {
				t.Errorf("D not symmetric at [%d,%d]: %.6g vs %.6g", i, j, D.At(i, j), D.At(j, i))
			}
		}
	}
}

// TestIsotropicDiagonalStructure verifies:
//   - D[0,0] = D[1,1] = D[2,2]  (equal normal stiffness)
//   - D[0,1] = D[0,2] = D[1,2]  (equal coupling)
//   - D[3,3] = D[4,4] = D[5,5]  (equal shear stiffness G)
//   - Off-diagonal shear blocks are zero
func TestIsotropicDiagonalStructure(t *testing.T) {
	m := material.NewIsotropicLinear(210000, 0.3)
	D := m.GetTangent()

	// Equal normal stiffness
	if math.Abs(D.At(0, 0)-D.At(1, 1)) > 1e-6 || math.Abs(D.At(1, 1)-D.At(2, 2)) > 1e-6 {
		t.Errorf("normal diagonal not equal: D00=%.6g D11=%.6g D22=%.6g", D.At(0, 0), D.At(1, 1), D.At(2, 2))
	}
	// Equal coupling terms
	if math.Abs(D.At(0, 1)-D.At(0, 2)) > 1e-6 || math.Abs(D.At(0, 1)-D.At(1, 2)) > 1e-6 {
		t.Errorf("coupling not equal: D01=%.6g D02=%.6g D12=%.6g", D.At(0, 1), D.At(0, 2), D.At(1, 2))
	}
	// Equal shear stiffness
	if math.Abs(D.At(3, 3)-D.At(4, 4)) > 1e-6 || math.Abs(D.At(4, 4)-D.At(5, 5)) > 1e-6 {
		t.Errorf("shear diagonal not equal: D33=%.6g D44=%.6g D55=%.6g", D.At(3, 3), D.At(4, 4), D.At(5, 5))
	}
	// Normal–shear blocks are zero
	for i := 0; i < 3; i++ {
		for j := 3; j < 6; j++ {
			if math.Abs(D.At(i, j)) > 1e-12 {
				t.Errorf("D[%d,%d] should be zero, got %.6g", i, j, D.At(i, j))
			}
		}
	}
}

// TestIsotropicShearModulus verifies G = E / (2(1+ν)).
func TestIsotropicShearModulus(t *testing.T) {
	E, nu := 210000.0, 0.3
	m := material.NewIsotropicLinear(E, nu)
	D := m.GetTangent()
	Gexpected := E / (2 * (1 + nu))
	Gactual := D.At(3, 3)
	if math.Abs(Gactual-Gexpected)/Gexpected > 1e-10 {
		t.Errorf("G = %.6g, want %.6g", Gactual, Gexpected)
	}
}

// TestIsotropicUniaxialStress verifies the uniaxial stress state σ = E·ε for
// free lateral expansion (free-body test, not constrained strain).
// Under uniaxial stress σxx = σ0, the strain is:
//
//	εxx = σ0 / E,  εyy = εzz = -ν·σ0 / E
//
// Inverting: under εxx = ε0 (all others free), σxx = E·ε0 only if εyy, εzz
// are the Poisson strains. We test the constrained case (εxx=ε0, rest=0)
// and verify the ratio σxx / σyy = (1-ν) / ν  (from the constitutive law).
func TestIsotropicUniaxialConstrainedStrain(t *testing.T) {
	E, nu := 210000.0, 0.3
	m := material.NewIsotropicLinear(E, nu)

	eps0 := 1e-3
	strain := mat.NewVecDense(6, []float64{eps0, 0, 0, 0, 0, 0})
	_ = m.SetTrialStrain(strain)
	sigma := m.GetStress()

	// Under uniaxial constrained strain, σxx/σyy = (1-ν)/ν
	sxx := sigma.AtVec(0)
	syy := sigma.AtVec(1)
	ratio := sxx / syy
	expected := (1 - nu) / nu
	if math.Abs(ratio-expected)/expected > 1e-10 {
		t.Errorf("σxx/σyy = %.6g, want (1-ν)/ν = %.6g", ratio, expected)
	}
	// Shear stresses must be zero
	for i := 3; i < 6; i++ {
		if math.Abs(sigma.AtVec(i)) > 1e-12 {
			t.Errorf("shear stress[%d] should be zero, got %.6g", i, sigma.AtVec(i))
		}
	}
}

// TestIsotropicPureShear verifies τxy = G·γxy under pure shear strain γxy = γ0.
func TestIsotropicPureShear(t *testing.T) {
	E, nu := 210000.0, 0.3
	m := material.NewIsotropicLinear(E, nu)
	G := E / (2 * (1 + nu))

	gamma := 1e-3
	strain := mat.NewVecDense(6, []float64{0, 0, 0, gamma, 0, 0})
	_ = m.SetTrialStrain(strain)
	sigma := m.GetStress()

	txy := sigma.AtVec(3)
	expected := G * gamma
	if math.Abs(txy-expected)/expected > 1e-10 {
		t.Errorf("τxy = %.6g, want G·γ = %.6g", txy, expected)
	}
	// Normal stresses must be zero
	for i := 0; i < 3; i++ {
		if math.Abs(sigma.AtVec(i)) > 1e-12 {
			t.Errorf("normal stress[%d] should be zero under pure shear, got %.6g", i, sigma.AtVec(i))
		}
	}
}

// TestIsotropicRevertToStart verifies that RevertToStart zeroes the stress.
func TestIsotropicRevertToStart(t *testing.T) {
	m := material.NewIsotropicLinear(210000, 0.3)
	strain := mat.NewVecDense(6, []float64{1e-3, 0, 0, 0, 0, 0})
	_ = m.SetTrialStrain(strain)

	if err := m.RevertToStart(); err != nil {
		t.Fatalf("RevertToStart: %v", err)
	}
	sigma := m.GetStress()
	for i := 0; i < 6; i++ {
		if math.Abs(sigma.AtVec(i)) > 1e-30 {
			t.Errorf("stress[%d] should be zero after RevertToStart, got %.6g", i, sigma.AtVec(i))
		}
	}
}

// TestIsotropicComplianceRoundTrip verifies D·S = I (stiffness × compliance = identity).
func TestIsotropicComplianceRoundTrip(t *testing.T) {
	E, nu := 210000.0, 0.3
	m := material.NewIsotropicLinear(E, nu)
	D := m.GetTangent()

	// Build compliance matrix S manually
	S := mat.NewDense(6, 6, []float64{
		1 / E, -nu / E, -nu / E, 0, 0, 0,
		-nu / E, 1 / E, -nu / E, 0, 0, 0,
		-nu / E, -nu / E, 1 / E, 0, 0, 0,
		0, 0, 0, 2 * (1 + nu) / E, 0, 0,
		0, 0, 0, 0, 2 * (1 + nu) / E, 0,
		0, 0, 0, 0, 0, 2 * (1 + nu) / E,
	})

	DS := mat.NewDense(6, 6, nil)
	DS.Mul(D, S)

	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			diff := math.Abs(DS.At(i, j) - expected)
			if diff > 1e-10 {
				t.Errorf("(D·S)[%d,%d] = %.6g, want %.6g", i, j, DS.At(i, j), expected)
			}
		}
	}
}
