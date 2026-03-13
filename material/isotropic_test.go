package material_test

import (
	"math"
	"testing"

	"go-fem/material"
	"gonum.org/v1/gonum/mat"
)

// TestIsotropicSymmetry verifies that the 6×6 tangent stiffness tensor D of
// an isotropic linear-elastic material is exactly symmetric.
//
// Property: D[i,j] == D[j,i] for all i, j. This follows from Maxwell reciprocity
// and is guaranteed analytically by the Lamé formula; no floating-point tolerance
// is needed because the entries are computed from a single closed-form expression.
//
// Parameters: E=210000, ν=0.3 (steel-like).
//
// Why valuable: any unsymmetry would invalidate the use of symmetric solvers
// (Cholesky, LDLT) and would indicate an error in the Lamé constant computation.
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

// TestIsotropicDiagonalStructure verifies the structural invariants of the
// isotropic tangent stiffness matrix D.
//
// Isotropy requires:
//   - D[0,0] = D[1,1] = D[2,2]  (equal normal stiffness in all three directions)
//   - D[0,1] = D[0,2] = D[1,2]  (equal Poisson coupling)
//   - D[3,3] = D[4,4] = D[5,5] = G  (equal shear stiffness)
//   - D[i,j] = 0 for i ∈ {0,1,2}, j ∈ {3,4,5}  (no normal-shear coupling)
//
// In Voigt notation: D = λ·(1⊗1) + 2μ·I₆ where λ and μ are Lamé parameters.
//
// Parameters: E=210000, ν=0.3.
//
// Why valuable: a violation of any invariant would indicate incorrect handling
// of isotropy (e.g., a Cij entry being swapped between the normal and shear
// blocks), which would produce wrong stresses in any analysis.
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

// TestIsotropicShearModulus verifies that the shear modulus G reported by the
// tangent stiffness matrix equals the analytical formula G = E / (2(1+ν)).
//
// Property: D[3,3] = G = E / (2(1+ν)). For E=210000 and ν=0.3:
//
//	G = 210000 / (2·1.3) = 80769.23...
//
// Parameters: E=210000, ν=0.3.
//
// Why valuable: the shear modulus is a critical parameter for shear-dominated
// problems; an incorrect G (e.g., using E/(2(1-ν)) or E/2) would corrupt all
// shear stress results.
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

// TestIsotropicUniaxialConstrainedStrain verifies the stress ratio under a
// constrained uniaxial strain state (all lateral strains suppressed to zero).
//
// Physical scenario: εxx = ε₀ (nonzero), εyy = εzz = γ = 0.
// This is the plane-strain / triaxially confined state where lateral strains
// are not allowed to develop. From the constitutive law:
//
//	σxx = (λ + 2μ) · ε₀,  σyy = σzz = λ · ε₀
//
// The ratio σxx / σyy = (λ + 2μ) / λ = (1 - ν) / ν.
//
// Shear stresses must be zero because no shear strains are applied.
//
// Parameters: E=210000, ν=0.3, ε₀=1e-3.
//
// Why valuable: confirms the Lamé constant λ is correctly computed from E
// and ν (a common source of bugs when ν is confused with λ directly).
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

// TestIsotropicPureShear verifies that pure shear strain γxy = γ₀ produces
// the correct shear stress τxy = G·γ₀ and zero normal stresses.
//
// Physical property: for an isotropic material, pure shear does not generate
// normal stresses (σxx = σyy = σzz = 0). Only the shear component is activated:
//
//	τxy = G · γxy = E/(2(1+ν)) · γ₀
//
// Parameters: E=210000, ν=0.3, γ₀=1e-3.
// Strain vector (Voigt): [0, 0, 0, γ₀, 0, 0].
//
// Why valuable: confirms that the off-diagonal coupling terms in D correctly
// produce zero normal stresses under shear strain. A wrong sign or coupling
// entry would generate spurious normal stresses.
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

// TestIsotropicRevertToStart verifies that RevertToStart() zeroes all stress
// components after a trial strain has been set.
//
// Property: after calling SetTrialStrain() and then RevertToStart(), the
// material returns to its initial (unstrained, unstressed) state. GetStress()
// must return zero for all 6 components.
//
// Parameters: E=210000, ν=0.3; trial strain ε=[1e-3, 0, 0, 0, 0, 0].
//
// Why valuable: confirms the material state-management API works correctly
// for nonlinear algorithms that require trial-and-commit/revert cycles;
// a failure would corrupt incremental load-step calculations.
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

// TestIsotropicComplianceRoundTrip verifies the inverse relationship between
// the stiffness matrix D and the compliance matrix S: D·S = I.
//
// Property: for a linear elastic material, S = D⁻¹ where S is the 6×6
// compliance matrix:
//
//	S = [ 1/E   -ν/E  -ν/E    0        0        0     ]
//	    [-ν/E    1/E  -ν/E    0        0        0     ]
//	    [-ν/E   -ν/E   1/E    0        0        0     ]
//	    [  0      0     0   2(1+ν)/E   0        0     ]
//	    [  0      0     0     0      2(1+ν)/E   0     ]
//	    [  0      0     0     0        0      2(1+ν)/E]
//
// The round-trip condition D·S = I verifies that neither matrix has been
// incorrectly scaled or transposed.
//
// Parameters: E=210000, ν=0.3.
//
// Expected: |(D·S)[i,j] - δᵢⱼ| < 1e-10 for all i, j.
//
// Why valuable: any error in the Lamé parameter derivation (wrong E/(1+ν)/(1-2ν)
// prefactor, for example) would cause D·S ≠ I and would be detected here.
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
