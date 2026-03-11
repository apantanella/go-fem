package material_test

import (
	"math"
	"testing"

	"go-fem/material"
	"gonum.org/v1/gonum/mat"
)

// TestOrthotropicIsotropicLimit verifies that an orthotropic material with
// equal moduli in all directions degenerates to the isotropic case.
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

// TestOrthotropicSymmetry verifies D is symmetric.
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

// TestOrthotropicStress verifies σ = D·ε matches SetTrialStrain output.
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

// TestOrthotropicUniaxialStrain checks that under uniaxial strain εxx = ε0
// the lateral stresses are nonzero (coupling) and shear stresses are zero.
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
