package material_test

import (
	"math"
	"testing"

	"go-fem/material"
)

// TestSteelBilinearElastic verifies the elastic branch of SteelBilinear:
// for a strain well below the elastic limit (ε = Fy/E), the stress equals
// E·ε and the tangent modulus equals E.
func TestSteelBilinearElastic(t *testing.T) {
	const E, Fy, Esh = 210000.0, 250.0, 2100.0
	m, err := material.NewSteelBilinear(E, Fy, Esh)
	if err != nil {
		t.Fatalf("NewSteelBilinear: %v", err)
	}

	eps := 0.001 // < Fy/E ≈ 0.00119  → elastic
	m.SetTrialStrain(eps)

	wantSig := E * eps // 210 MPa
	if got := m.GetStress(); math.Abs(got-wantSig) > 1e-9 {
		t.Errorf("elastic stress: want %.6g, got %.6g", wantSig, got)
	}
	if got := m.GetTangent(); math.Abs(got-E) > 1e-9 {
		t.Errorf("elastic tangent: want %g, got %g", E, got)
	}
}

// TestSteelBilinearYieldPoint verifies that the material is entirely elastic
// for a strain just inside the elastic limit (eps = 0.99·Fy/E).
// The test avoids the exact yield strain to prevent floating-point rounding
// artefacts at the f_tr = 0 boundary.
func TestSteelBilinearYieldPoint(t *testing.T) {
	const E, Fy, Esh = 210000.0, 250.0, 2100.0
	m, _ := material.NewSteelBilinear(E, Fy, Esh)

	eps := 0.99 * Fy / E // just below yield → elastic
	m.SetTrialStrain(eps)

	if got, want := m.GetStress(), E*eps; math.Abs(got-want) > 1e-6 {
		t.Errorf("sub-yield stress: want %g, got %g", want, got)
	}
	if got := m.GetTangent(); math.Abs(got-E) > 1e-9 {
		t.Errorf("sub-yield tangent: want E=%g (elastic), got %g", E, got)
	}
}

// TestSteelBilinearPlastic verifies the plastic branch using the return-mapping
// formulae of Simo & Hughes §2.3 for bilinear kinematic hardening.
//
// Setup: E=210000, Fy=250, Esh=2100.  Apply ε = 0.002 (> Fy/E ≈ 0.00119).
//
// By hand:
//
//	σ_tr = 210000·0.002 = 420 MPa
//	f_tr = 420 − 250 = 170 MPa  > 0  →  plastic
//	Δγ   = 170 / (210000+2100) ≈ 0.0008016
//	σ    = 420 − 210000·Δγ ≈ 251.64 MPa
//	Et   = 210000·2100 / 212100 ≈ 2079.2 MPa
func TestSteelBilinearPlastic(t *testing.T) {
	const E, Fy, Esh = 210000.0, 250.0, 2100.0
	m, _ := material.NewSteelBilinear(E, Fy, Esh)

	eps := 0.002
	m.SetTrialStrain(eps)

	sigTr := E * eps
	fTr := sigTr - Fy
	deltaGamma := fTr / (E + Esh)
	wantSig := sigTr - E*deltaGamma
	wantEt := E * Esh / (E + Esh)

	if got := m.GetStress(); math.Abs(got-wantSig) > 1e-6 {
		t.Errorf("plastic stress: want %.9g, got %.9g", wantSig, got)
	}
	if got := m.GetTangent(); math.Abs(got-wantEt) > 1e-6 {
		t.Errorf("plastic tangent: want %.9g, got %.9g", wantEt, got)
	}
	// Stress must be above Fy (kinematic hardening)
	if m.GetStress() <= Fy {
		t.Errorf("expected stress > Fy=%.g in plastic zone, got %.6g", Fy, m.GetStress())
	}
}

// TestSteelBilinearEPP verifies elastic-perfectly plastic behaviour (Esh = 0):
// in the plastic zone the tangent modulus must be zero and the stress must
// equal ±Fy.
func TestSteelBilinearEPP(t *testing.T) {
	const E, Fy = 210000.0, 250.0
	m, err := material.NewSteelBilinear(E, Fy, 0) // Esh = 0
	if err != nil {
		t.Fatalf("NewSteelBilinear EPP: %v", err)
	}

	// Tension past yield
	m.SetTrialStrain(0.005) // ε >> Fy/E
	if got, want := m.GetTangent(), 0.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("EPP tangent in tension: want 0, got %g", got)
	}
	if got := m.GetStress(); math.Abs(got-Fy) > 1e-6 {
		t.Errorf("EPP stress in tension: want %g, got %g", Fy, got)
	}

	// Compression past yield
	m.RevertToStart()
	m.SetTrialStrain(-0.005)
	if got := m.GetStress(); math.Abs(got+Fy) > 1e-6 {
		t.Errorf("EPP stress in compression: want %g, got %g", -Fy, got)
	}
}

// TestSteelBilinearCommitAndUnload verifies that after committing a yielded
// state and then reverting to a smaller strain, the material unloads elastically
// (kinematic hardening shifts the yield surface).
//
// Procedure:
//  1. Load to ε₁ = 0.002 (plastic), commit.
//  2. Unload to ε₂ = 0 without committing; check elastic unloading.
func TestSteelBilinearCommitAndUnload(t *testing.T) {
	const E, Fy, Esh = 210000.0, 250.0, 2100.0
	m, _ := material.NewSteelBilinear(E, Fy, Esh)

	// Step 1 – load past yield and commit.
	m.SetTrialStrain(0.002)
	m.CommitState()
	sigCommitted := m.GetStress()
	alpNext := m.GetTangent() // will be Et in plastic zone — not the back-stress

	// Step 2 – set trial strain back to 0 (elastic unloading expected).
	m.SetTrialStrain(0.0)
	sigUnload := m.GetStress()
	etUnload := m.GetTangent()

	// After committing ε_c ≈ 0.002, going back to ε = 0 gives
	// σ_tr = σ_c + E*(0 - ε_c) = σ_c - E*ε_c
	// The unloading is elastic if |σ_tr - α_c| < Fy.
	// sigCommitted ≈ 251.6, so σ_tr = 251.6 - 420 ≈ -168.4. |-168.4 - α| < 250 → elastic.
	if math.Abs(etUnload-E) > 1e-9 {
		t.Errorf("unloading tangent: want E=%g (elastic), got %g", E, etUnload)
	}
	// The unloaded stress must be less than the committed stress (moving back).
	if sigUnload >= sigCommitted {
		t.Errorf("unloaded stress %g ≥ committed %g (expected reduction)", sigUnload, sigCommitted)
	}
	_ = alpNext // suppress unused warning
}

// TestSteelBilinearRevertToStart checks that RevertToStart completely resets
// the material to its initial stress-free state.
func TestSteelBilinearRevertToStart(t *testing.T) {
	const E, Fy, Esh = 210000.0, 250.0, 2100.0
	m, _ := material.NewSteelBilinear(E, Fy, Esh)

	// Drive deep into plasticity and commit.
	m.SetTrialStrain(0.01)
	m.CommitState()

	// Revert and apply a small strain: should behave as virgin material.
	m.RevertToStart()
	eps := 0.0005
	m.SetTrialStrain(eps)
	if got, want := m.GetStress(), E*eps; math.Abs(got-want) > 1e-9 {
		t.Errorf("after RevertToStart stress: want %g, got %g", want, got)
	}
	if got := m.GetTangent(); math.Abs(got-E) > 1e-9 {
		t.Errorf("after RevertToStart tangent: want E=%g, got %g", E, got)
	}
}

// TestSteelBilinearNegativeE verifies that NewSteelBilinear rejects invalid inputs.
func TestSteelBilinearInvalidInputs(t *testing.T) {
	cases := []struct {
		name     string
		E, Fy, H float64
	}{
		{"zero E", 0, 250, 0},
		{"negative E", -1, 250, 0},
		{"zero Fy", 210000, 0, 0},
		{"negative Esh", 210000, 250, -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := material.NewSteelBilinear(tc.E, tc.Fy, tc.H)
			if err == nil {
				t.Errorf("expected error for %s, got nil", tc.name)
			}
		})
	}
}
