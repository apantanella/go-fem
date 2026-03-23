package material

import (
	"fmt"
	"math"
)

// errorf is a local helper to create formatted errors.
func errorf(format string, a ...any) error { return fmt.Errorf(format, a...) }

// SteelBilinear implements a uniaxial bilinear kinematic hardening steel model.
//
// The constitutive law is:
//
//	σ = E·ε                             for |σ_trial − α| ≤ Fy  (elastic)
//	σ = sign(ξ)·Fy + α + Esh·Δγ·sign   for |σ_trial − α| > Fy  (plastic)
//
// where ξ = σ_trial − α is the reduced stress and Δγ is the plastic multiplier.
//
// Return-mapping algorithm (Simo & Hughes §2.3):
//
//  1. Compute trial stress: σ_tr = σ_c + E·(ε − ε_c)
//  2. Compute trial yield function: f_tr = |σ_tr − α_c| − Fy
//  3. If f_tr ≤ 0: elastic step, σ = σ_tr, E_t = E
//  4. If f_tr > 0: plastic step:
//     Δγ = f_tr / (E + Esh)
//     σ = σ_tr − sign(ξ) · E · Δγ
//     α = α_c + sign(ξ) · Esh · Δγ
//     E_t = E · Esh / (E + Esh)   (consistent algorithmic tangent)
//
// Special case Esh = 0: elastic-perfectly plastic (E_t = 0 in plastic zone).
//
// State variables:
//
//	ε_c, σ_c, α_c  – committed strain, stress, and back-stress.
//
// Units: consistent with E and Fy (e.g. MPa, N/mm²).
type SteelBilinear struct {
	E   float64 // Young's modulus
	Fy  float64 // initial yield stress (> 0)
	Esh float64 // hardening modulus (≥ 0); Esh=0 → elastic-perfectly plastic

	// committed (converged) state
	epsC float64 // committed strain
	sigC float64 // committed stress
	alpC float64 // committed back-stress (kinematic hardening)

	// trial state
	epsTrial float64
	sigTrial float64
	alpTrial float64
	etTrial  float64 // consistent tangent
}

// NewSteelBilinear creates a bilinear steel material.
// Esh = 0 gives elastic-perfectly plastic behaviour.
// Returns an error if E ≤ 0, Fy ≤ 0, or Esh < 0.
func NewSteelBilinear(E, Fy, Esh float64) (*SteelBilinear, error) {
	if E <= 0 {
		return nil, errorf("SteelBilinear: E must be > 0, got %g", E)
	}
	if Fy <= 0 {
		return nil, errorf("SteelBilinear: Fy must be > 0, got %g", Fy)
	}
	if Esh < 0 {
		return nil, errorf("SteelBilinear: Esh must be ≥ 0, got %g", Esh)
	}
	m := &SteelBilinear{E: E, Fy: Fy, Esh: Esh}
	m.etTrial = E // initial tangent is elastic
	return m, nil
}

// SetTrialStrain applies the return-mapping algorithm for the given trial strain.
func (m *SteelBilinear) SetTrialStrain(eps float64) error {
	m.epsTrial = eps

	// 1. Elastic predictor
	sigTr := m.sigC + m.E*(eps-m.epsC)
	xsi := sigTr - m.alpC // relative (reduced) stress

	// 2. Trial yield function
	fTr := math.Abs(xsi) - m.Fy

	if fTr <= 0 {
		// Elastic step
		m.sigTrial = sigTr
		m.alpTrial = m.alpC
		m.etTrial = m.E
		return nil
	}

	// 3. Plastic corrector (return mapping onto the yield surface)
	sgn := math.Copysign(1, xsi)
	deltaGamma := fTr / (m.E + m.Esh)
	m.sigTrial = sigTr - sgn*m.E*deltaGamma
	m.alpTrial = m.alpC + sgn*m.Esh*deltaGamma

	// Consistent algorithmic tangent (reduces to 0 when Esh = 0)
	if m.E+m.Esh > 0 {
		m.etTrial = m.E * m.Esh / (m.E + m.Esh)
	} else {
		m.etTrial = 0
	}
	return nil
}

func (m *SteelBilinear) GetStress() float64  { return m.sigTrial }
func (m *SteelBilinear) GetTangent() float64 { return m.etTrial }

func (m *SteelBilinear) CommitState() error {
	m.epsC = m.epsTrial
	m.sigC = m.sigTrial
	m.alpC = m.alpTrial
	return nil
}

func (m *SteelBilinear) RevertToStart() error {
	m.epsC = 0
	m.sigC = 0
	m.alpC = 0
	m.epsTrial = 0
	m.sigTrial = 0
	m.alpTrial = 0
	m.etTrial = m.E
	return nil
}
