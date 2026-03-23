package material

// ConcretePararect implements a uniaxial compression-only concrete model
// following the parabola-rectangle constitutive law of EN 1992-1-1 §3.1.7.
//
// Sign convention: compression negative, tension positive (consistent with go-fem).
// The model is monotonic — it evaluates σ directly from ε without return-mapping.
// Tensile stress is always zero (no-tension assumption).
//
// Constitutive law (EN 1992-1-1, Eq. 3.17-3.19, adapted to negative ε convention):
//
//	ε ≥ 0                  → σ = 0               (tension → no stress)
//	-εc1 ≤ ε < 0           → σ = -fc·(1-(1+ε/εc1)²) · 2   (parabolic ascending)
//	Wait — standard formula: σ/fc = 1 − (1 − ε/εc1)^n,  n=2 for parabola
//	Using compression-negative ε: let ξ = ε/εc1 ∈ [-1, 0]
//	σ = -fc · [1 − (1 + ξ)²] ... but at ξ=0 → σ=0, at ξ=-1 → σ=-fc*(1-0)=-fc ✓
//	Wait: 1-(1+ξ)² = 1-1-2ξ-ξ² = -2ξ-ξ² = -ξ(2+ξ)
//	So σ = fc·ξ·(2+ξ)  [negative for negative ξ] ✓
//
//	-εcu1 ≤ ε < -εc1       → σ = -fc               (plateau / rectangle)
//	ε < -εcu1              → σ = 0                  (crushed — beyond ultimate strain)
//
// Parameters:
//   - fc:    design/characteristic compressive strength (positive, MPa)
//   - Ec:    initial tangent modulus (MPa); if ≤ 0 → auto Ec = 2·fc/εc1
//   - EpsC1: strain at peak compression (positive value, default 0.0020)
//   - EpsCU: ultimate (crushing) strain (positive value, default 0.0035)
type ConcretePararect struct {
	Fc    float64 // compressive strength (positive, MPa)
	Ec    float64 // initial tangent modulus (derived from Fc/EpsC1 if not set)
	EpsC1 float64 // strain at peak (positive, default 0.002)
	EpsCU float64 // ultimate strain (positive, default 0.0035)

	// trial state
	epsTrial float64
	sigTrial float64
	etTrial  float64
}

// NewConcretePararect creates a parabola-rectangle concrete model.
// fc must be > 0. epsC1 and epsCU may be 0 (defaults: 0.002, 0.0035).
// Ec ≤ 0 → auto-computed as 2·fc/epsC1 (initial tangent of the parabola).
func NewConcretePararect(fc, epsC1, epsCU, Ec float64) (*ConcretePararect, error) {
	if fc <= 0 {
		return nil, errorf("ConcretePararect: fc must be > 0, got %g", fc)
	}
	if epsC1 <= 0 {
		epsC1 = 0.0020
	}
	if epsCU <= 0 {
		epsCU = 0.0035
	}
	if epsCU < epsC1 {
		return nil, errorf("ConcretePararect: epsCU (%g) must be ≥ epsC1 (%g)", epsCU, epsC1)
	}
	if Ec <= 0 {
		Ec = 2 * fc / epsC1 // tangent at origin of parabola
	}
	m := &ConcretePararect{Fc: fc, Ec: Ec, EpsC1: epsC1, EpsCU: epsCU}
	m.etTrial = Ec // initial tangent
	return m, nil
}

// SetTrialStrain evaluates σ and dσ/dε from the current strain ε.
func (m *ConcretePararect) SetTrialStrain(eps float64) error {
	m.epsTrial = eps

	switch {
	case eps >= 0:
		// Tension — no stress, no tangent.
		m.sigTrial = 0
		m.etTrial = 0

	case eps >= -m.EpsC1:
		// Parabolic ascending branch: σ = fc · ξ · (2+ξ)  with ξ = ε/εc1
		xi := eps / m.EpsC1
		m.sigTrial = m.Fc * xi * (2 + xi)         // negative (compression)
		m.etTrial = m.Fc / m.EpsC1 * 2 * (1 + xi) // dσ/dε ≥ 0 at origin, 0 at peak

	case eps >= -m.EpsCU:
		// Plateau (constant stress = -fc)
		m.sigTrial = -m.Fc
		m.etTrial = 0

	default:
		// Beyond ultimate strain — concrete crushed, carry no stress.
		m.sigTrial = 0
		m.etTrial = 0
	}
	return nil
}

func (m *ConcretePararect) GetStress() float64  { return m.sigTrial }
func (m *ConcretePararect) GetTangent() float64 { return m.etTrial }

// CommitState saves the current trial strain as the committed state.
// For this path-independent model no history state is required, but the
// interface mandates implementation.
func (m *ConcretePararect) CommitState() error { return nil }

func (m *ConcretePararect) RevertToStart() error {
	m.epsTrial = 0
	m.sigTrial = 0
	m.etTrial = m.Ec
	return nil
}
