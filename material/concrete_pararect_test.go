package material_test

import (
	"math"
	"testing"

	"go-fem/material"
)

// TestConcretePararectTensionZero verifies the no-tension assumption:
// any tensile strain ε > 0 must produce σ = 0 and Et = 0.
func TestConcretePararectTensionZero(t *testing.T) {
	m, err := material.NewConcretePararect(30, 0, 0, 0) // fc=30 MPa, defaults
	if err != nil {
		t.Fatalf("NewConcretePararect: %v", err)
	}

	for _, eps := range []float64{0.0, 1e-6, 0.001, 0.1} {
		m.SetTrialStrain(eps)
		if got := m.GetStress(); got != 0 {
			t.Errorf("tension at ε=%.6g: want σ=0, got %.6g", eps, got)
		}
		if got := m.GetTangent(); got != 0 {
			t.Errorf("tension tangent at ε=%.6g: want Et=0, got %.6g", eps, got)
		}
	}
}

// TestConcretePararectParabolicBranch verifies the parabolic ascending branch.
//
// EN 1992-1-1 Eq. 3.17-3.19 in compression-negative convention:
//
//	σ = fc · ξ · (2 + ξ)   where ξ = ε/εc1 ∈ [−1, 0]
//
// At ε = 0:    σ = 0
// At ε = −εc1: σ = fc · (−1) · (2−1) = −fc
func TestConcretePararectParabolicBranch(t *testing.T) {
	const fc = 30.0
	const epsC1 = 0.002
	m, _ := material.NewConcretePararect(fc, epsC1, 0.0035, 0)

	// At ε = 0 (boundary between tension and compression)
	m.SetTrialStrain(0)
	if got := m.GetStress(); got != 0 {
		t.Errorf("σ at ε=0: want 0, got %g", got)
	}

	// At ε = −εc1/2 (mid parabola)
	eps := -epsC1 / 2
	m.SetTrialStrain(eps)
	xi := eps / epsC1 // = −0.5
	wantSig := fc * xi * (2 + xi)
	if got := m.GetStress(); math.Abs(got-wantSig) > 1e-9 {
		t.Errorf("σ at ε=−εc1/2: want %.9g, got %.9g", wantSig, got)
	}

	// At ε = −εc1 (peak): σ must equal −fc
	m.SetTrialStrain(-epsC1)
	if got := m.GetStress(); math.Abs(got+fc) > 1e-9 {
		t.Errorf("σ at ε=−εc1: want %g, got %g", -fc, got)
	}
}

// TestConcretePararectPlateau verifies the constant-stress rectangle branch:
// for −εcu ≤ ε < −εc1  the stress must equal −fc and the tangent must be 0.
func TestConcretePararectPlateau(t *testing.T) {
	const fc, epsC1, epsCU = 30.0, 0.002, 0.0035
	m, _ := material.NewConcretePararect(fc, epsC1, epsCU, 0)

	// At ε slightly past peak
	m.SetTrialStrain(-epsC1 - 0.0001)
	if got := m.GetStress(); math.Abs(got+fc) > 1e-9 {
		t.Errorf("plateau stress: want %g, got %g", -fc, got)
	}
	if got := m.GetTangent(); got != 0 {
		t.Errorf("plateau tangent: want 0, got %g", got)
	}

	// At ε = −εcu (edge of plateau)
	m.SetTrialStrain(-epsCU)
	if got := m.GetStress(); math.Abs(got+fc) > 1e-9 {
		t.Errorf("plateau at ε=−εcu: want %g, got %g", -fc, got)
	}
}

// TestConcretePararectPostCrush verifies that beyond ultimate strain the model
// carries no stress (concrete interpreted as crushed).
func TestConcretePararectPostCrush(t *testing.T) {
	const fc, epsC1, epsCU = 30.0, 0.002, 0.0035
	m, _ := material.NewConcretePararect(fc, epsC1, epsCU, 0)

	m.SetTrialStrain(-epsCU - 0.001) // past crushing
	if got := m.GetStress(); got != 0 {
		t.Errorf("post-crush stress: want 0, got %g", got)
	}
	if got := m.GetTangent(); got != 0 {
		t.Errorf("post-crush tangent: want 0, got %g", got)
	}
}

// TestConcretePararectInitialTangent verifies that the initial tangent modulus
// near ε = 0 is approximately Ec = 2·fc/εc1 (the slope of the parabola at
// the origin).  A small numerical tolerance of 0.01% is used because the
// exact initial tangent formula Et = 2fc/εc1·(1 + ε/εc1) slightly
// deviates from Ec for any non-zero ε.
func TestConcretePararectInitialTangent(t *testing.T) {
	const fc, epsC1 = 30.0, 0.002
	m, _ := material.NewConcretePararect(fc, epsC1, 0.0035, 0)

	// Apply a very small compressive strain near the origin.
	eps := -1e-8
	m.SetTrialStrain(eps)

	// dσ/dε at ε≈0: Et = 2·fc/εc1·(1 + ε/εc1) ≈ Ec
	Ec := 2 * fc / epsC1
	const tol = 0.001 // 0.1% relative tolerance for the near-zero evaluation
	if got := m.GetTangent(); math.Abs(got-Ec)/Ec > tol {
		t.Errorf("initial tangent: want Ec=%g, got %g (err %.4f%%)", Ec, got, math.Abs(got-Ec)/Ec*100)
	}
}

// TestConcretePararectTangentAtPeak verifies that the tangent modulus equals
// zero at ε = −εc1 (the peak of the parabola).
func TestConcretePararectTangentAtPeak(t *testing.T) {
	const fc, epsC1 = 30.0, 0.002
	m, _ := material.NewConcretePararect(fc, epsC1, 0.0035, 0)

	m.SetTrialStrain(-epsC1) // at peak
	if got := m.GetTangent(); math.Abs(got) > 1e-9 {
		t.Errorf("tangent at peak: want 0, got %g", got)
	}
}

// TestConcretePararectRevertToStart checks that RevertToStart returns the model
// to its initial stress-free state.
func TestConcretePararectRevertToStart(t *testing.T) {
	m, _ := material.NewConcretePararect(30, 0, 0, 0)

	m.SetTrialStrain(-0.003) // plateau
	m.CommitState()
	m.RevertToStart()

	m.SetTrialStrain(0)
	if got := m.GetStress(); got != 0 {
		t.Errorf("after RevertToStart: want σ=0, got %g", got)
	}
}

// TestConcretePararectInvalidInputs verifies that invalid parameters are rejected.
func TestConcretePararectInvalidInputs(t *testing.T) {
	cases := []struct {
		name             string
		fc, ec1, ecu, ec float64
	}{
		{"zero fc", 0, 0, 0, 0},
		{"negative fc", -10, 0, 0, 0},
		{"epsC1 > epsCU", 30, 0.004, 0.003, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := material.NewConcretePararect(tc.fc, tc.ec1, tc.ecu, tc.ec)
			if err == nil {
				t.Errorf("expected error for %s, got nil", tc.name)
			}
		})
	}
}
