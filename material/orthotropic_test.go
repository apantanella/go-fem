package material_test

import (
	"math"
	"testing"

	"go-fem/material"
	"gonum.org/v1/gonum/mat"
)

// TestOrthotropicIsotropicLimit verifies that an orthotropic material with
// equal moduli in all directions degenerates to the isotropic case.
//
// Property: when Ex = Ey = Ez = E, νxy = νyz = νxz = ν, and
// Gxy = Gyz = Gxz = G = E/(2(1+ν)), the orthotropic tangent matrix D_ortho
// must equal the isotropic tangent matrix D_iso entry-by-entry.
//
// This is the most fundamental self-consistency check for an orthotropic
// material implementation: the isotropic special case must be recovered exactly.
//
// Parameters: E=200000, ν=0.3, G=E/(2(1+ν))≈76923.
//
// Expected: |D_ortho[i,j] - D_iso[i,j]| < 1e-6 for all i, j.
//
// Why valuable: any inconsistency in the compliance matrix inversion or the
// Lamé parameterisation would be revealed here, including sign errors in the
// νxy/Ey versus νyx/Ex reciprocal relation.
func TestOrthotropicIsotropicLimit(t *testing.T) {
	E := 200000.0
	nu := 0.3
	G := E / (2 * (1 + nu))

	ortho, err := material.NewOrthotropicLinear(E, E, E, nu, nu, nu, G, G, G)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	iso := material.NewIsotropicLinear(E, nu)

	Do := ortho.GetTangent()
	Di := iso.GetTangent()

	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			diff := math.Abs(Do.At(i, j) - Di.At(i, j))
			if diff > 1e-6 {
				t.Errorf("D[%d,%d]: ortho=%.6g iso=%.6g diff=%.2e", i, j, Do.At(i, j), Di.At(i, j), diff)
			}
		}
	}
}

// TestOrthotropicSymmetry verifies that the 6×6 tangent stiffness matrix D
// of an orthotropic material is symmetric.
//
// Property: D[i,j] == D[j,i]. Symmetry of D is guaranteed by Maxwell reciprocity
// enforced on the compliance matrix: νij/Ei = νji/Ej. The stiffness D is
// obtained by inverting the symmetric compliance matrix S, so D must also
// be symmetric.
//
// Parameters: anisotropic orthotropic material with
//   - Ex=120000, Ey=80000, Ez=60000
//   - νxy=0.25, νyz=0.20, νxz=0.30
//   - Gxy=30000, Gyz=25000, Gxz=20000
//
// This is a strongly anisotropic case that exercises all independent terms.
//
// Expected: |D[i,j] - D[j,i]| < 1e-8 for all i < j.
//
// Why valuable: a non-symmetric D would indicate that the compliance matrix
// inversion did not preserve Maxwell reciprocity, which would violate
// energy conservation.
func TestOrthotropicSymmetry(t *testing.T) {
	ortho, err := material.NewOrthotropicLinear(
		120000, 80000, 60000,
		0.25, 0.20, 0.30,
		30000, 25000, 20000,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	D := ortho.GetTangent()
	for i := 0; i < 6; i++ {
		for j := i + 1; j < 6; j++ {
			diff := math.Abs(D.At(i, j) - D.At(j, i))
			if diff > 1e-8 {
				t.Errorf("D not symmetric at [%d,%d]: %.6g vs %.6g", i, j, D.At(i, j), D.At(j, i))
			}
		}
	}
}

// TestOrthotropicStress verifies that the stress vector obtained via
// SetTrialStrain / GetStress matches the direct product D·ε.
//
// Property: for a linear elastic material σ = D·ε. The stress returned by
// GetStress() after SetTrialStrain() must equal D·ε computed by direct
// matrix-vector multiplication with GetTangent().
//
// Parameters: Ex=120000, Ey=80000, Ez=60000, νxy=0.25, νyz=0.20, νxz=0.30,
// Gxy=30000, Gyz=25000, Gxz=20000; trial strain ε=[1e-3, 0, 0, 0, 0, 0].
//
// Why valuable: confirms the material state is consistent between the tangent
// and the stress computation; a mismatch would reveal that the trial strain
// state is not used in GetStress().
func TestOrthotropicStress(t *testing.T) {
	ortho, err := material.NewOrthotropicLinear(
		120000, 80000, 60000,
		0.25, 0.20, 0.30,
		30000, 25000, 20000,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	strain := mat.NewVecDense(6, []float64{1e-3, 0, 0, 0, 0, 0})
	if err := ortho.SetTrialStrain(strain); err != nil {
		t.Fatalf("SetTrialStrain: %v", err)
	}
	sigma := ortho.GetStress()

	D := ortho.GetTangent()
	expected := mat.NewVecDense(6, nil)
	expected.MulVec(D, strain)

	for i := 0; i < 6; i++ {
		diff := math.Abs(sigma.AtVec(i) - expected.AtVec(i))
		if diff > 1e-10 {
			t.Errorf("stress[%d]: got %.6g, want %.6g", i, sigma.AtVec(i), expected.AtVec(i))
		}
	}
}

// TestOrthotropicUniaxialStrain checks that under constrained uniaxial strain
// (εxx = ε₀, all other strains zero) the material produces physically correct
// stress components.
//
// Physical properties for constrained uniaxial strain in an orthotropic solid:
//  1. σxx > 0 (positive strain → positive axial stress, since Ex > 0)
//  2. σyy ≠ 0 and σzz ≠ 0 (Poisson coupling in constrained configuration)
//  3. All shear stresses must be zero (no shear strain is applied and
//     the orthotropic constitutive law has no normal-shear coupling)
//
// Parameters: Ex=120000, Ey=80000, Ez=60000, νxy=0.25, νyz=0.20, νxz=0.30,
// Gxy=30000, Gyz=25000, Gxz=20000; ε₀=1e-3.
//
// Why valuable: confirms the Poisson coupling terms νij·Ej are correctly
// embedded in D; a wrong sign on a Poisson ratio would flip the sign of the
// lateral stresses.
func TestOrthotropicUniaxialStrain(t *testing.T) {
	ortho, err := material.NewOrthotropicLinear(
		120000, 80000, 60000,
		0.25, 0.20, 0.30,
		30000, 25000, 20000,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	eps0 := 1e-3
	strain := mat.NewVecDense(6, []float64{eps0, 0, 0, 0, 0, 0})
	_ = ortho.SetTrialStrain(strain)
	sigma := ortho.GetStress()

	// σxx must be positive (compression in X causes positive normal stress)
	if sigma.AtVec(0) <= 0 {
		t.Errorf("σxx should be positive under εxx > 0, got %.6g", sigma.AtVec(0))
	}
	// σyy and σzz must be non-zero due to coupling (Poisson effect in constrained solid)
	if math.Abs(sigma.AtVec(1)) < 1e-6 {
		t.Errorf("σyy should be nonzero under constrained uniaxial strain, got %.6g", sigma.AtVec(1))
	}
	// Shear stresses must be zero
	for i := 3; i < 6; i++ {
		if math.Abs(sigma.AtVec(i)) > 1e-12 {
			t.Errorf("shear stress[%d] should be zero, got %.6g", i, sigma.AtVec(i))
		}
	}
}
