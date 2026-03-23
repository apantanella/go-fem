package analysis

import (
	"fmt"
	"math"
	"sort"

	"go-fem/domain"
)

// ---------------------------------------------------------------------------
// Elastic response spectrum
// ---------------------------------------------------------------------------

// SpectrumPoint is a (period T [s], spectral acceleration Sa [m/s² or g]) pair.
type SpectrumPoint struct {
	T  float64 // period [s]
	Sa float64 // spectral acceleration (any consistent unit — same as used in masses)
}

// Spectrum is a piecewise-linear elastic response spectrum Sa(T).
// Points must cover T=0 and must be sorted by T in ascending order.
// Interpolation is linear; extrapolation clamps to the nearest endpoint.
type Spectrum struct {
	Points []SpectrumPoint
}

// NewSpectrum creates a Spectrum from parallel slices of periods and accelerations.
// Returns an error when the slices are empty, have different lengths, or T values
// are not strictly ascending.
func NewSpectrum(T, Sa []float64) (*Spectrum, error) {
	if len(T) == 0 || len(T) != len(Sa) {
		return nil, fmt.Errorf("spectrum: T and Sa must be non-empty and equal length")
	}
	pts := make([]SpectrumPoint, len(T))
	for i := range T {
		pts[i] = SpectrumPoint{T: T[i], Sa: Sa[i]}
	}
	// Ensure ascending order.
	sort.Slice(pts, func(i, j int) bool { return pts[i].T < pts[j].T })
	for i := 1; i < len(pts); i++ {
		if pts[i].T <= pts[i-1].T {
			return nil, fmt.Errorf("spectrum: duplicate or non-ascending period at index %d (T=%.4g)", i, pts[i].T)
		}
	}
	return &Spectrum{Points: pts}, nil
}

// SaAt returns the spectral acceleration at period T [s] by linear interpolation.
// Values outside the defined range are clamped to the first/last point.
func (s *Spectrum) SaAt(T float64) float64 {
	pts := s.Points
	if T <= pts[0].T {
		return pts[0].Sa
	}
	if T >= pts[len(pts)-1].T {
		return pts[len(pts)-1].Sa
	}
	// Binary search for the bracketing interval.
	lo, hi := 0, len(pts)-1
	for hi-lo > 1 {
		mid := (lo + hi) / 2
		if pts[mid].T <= T {
			lo = mid
		} else {
			hi = mid
		}
	}
	t0, t1 := pts[lo].T, pts[hi].T
	s0, s1 := pts[lo].Sa, pts[hi].Sa
	alpha := (T - t0) / (t1 - t0)
	return s0 + alpha*(s1-s0)
}

// ---------------------------------------------------------------------------
// CQC correlation coefficient
// ---------------------------------------------------------------------------

// CQCCorrelation returns the Complete Quadratic Combination correlation
// coefficient ρᵢⱼ between modes i and j, using the formula of
// Der Kiureghian (1981) for a white-noise input:
//
//	ρᵢⱼ = 8·ξ²·(1+r)·r^(3/2) / ((1-r²)² + 4·ξ²·r·(1+r)²)
//
// where r = ωᵢ/ωⱼ (ωᵢ ≤ ωⱼ) and ξ is the (uniform) damping ratio.
// Returns 1 when i == j.
func CQCCorrelation(omegaI, omegaJ, xi float64) float64 {
	if omegaI <= 0 || omegaJ <= 0 {
		return 0
	}
	// Ensure r ≤ 1 (swap if needed — formula is symmetric).
	r := omegaI / omegaJ
	if r > 1 {
		r = 1 / r
	}
	num := 8.0 * xi * xi * (1 + r) * math.Pow(r, 1.5)
	denom := (1-r*r)*(1-r*r) + 4*xi*xi*r*(1+r)*(1+r)
	if denom < 1e-30 {
		return 1 // degenerate case (r ≈ 1, nearly identical frequencies)
	}
	return num / denom
}

// ---------------------------------------------------------------------------
// Response Spectrum Analysis
// ---------------------------------------------------------------------------

// RSADirection selects which global directions to excite.
type RSADirection int

const (
	DirX RSADirection = 1 << iota
	DirY
	DirZ
	DirXYZ = DirX | DirY | DirZ
)

// ResponseSpectrumAnalysis performs a linear seismic analysis using the
// Response Spectrum Method (multi-mode, CQC or SRSS combination).
//
// Prerequisites:
//   - Dom must already have nodes, elements, BCs set up.
//   - Masses defines element densities for mass matrix assembly.
//   - Spectrum is the elastic design spectrum Sa(T).
//   - NumModes: number of eigenmodes to include (≥ 1).
//   - DampingRatio ξ: uniform modal damping ratio (default 0.05 → 5 %).
//   - UseSRSS: if true, use SRSS instead of CQC combination.
//   - Directions: bitmask of active excitation directions (default DirXYZ).
type ResponseSpectrumAnalysis struct {
	Dom          *domain.Domain
	Masses       []domain.ElementMass
	Spectrum     *Spectrum
	NumModes     int
	DampingRatio float64
	UseSRSS      bool
	Directions   RSADirection
}

// SpectrumResult holds the output of a response spectrum analysis.
type SpectrumResult struct {
	// Modal contains the underlying modal analysis result.
	Modal *ModalResult

	// ModalDisplacements[k] is the peak displacement vector (ndof values) for mode k
	// scaled by the spectral acceleration of that mode (M-normalised basis).
	ModalDisplacements [][]float64 // [mode][globalDOF]

	// MaxDisplacements[node][dofLocal] = CQC-combined peak displacement for that DOF.
	// Indexed identically to domain.SetDisplacements output ([node][6]).
	MaxDisplacements [][6]float64

	// ModalBaseShear[k][d] = modal base shear for mode k in direction d (d=0:X,1:Y,2:Z).
	ModalBaseShear [][3]float64

	// MaxBaseShear[d] = CQC-combined peak base shear in direction d.
	MaxBaseShear [3]float64

	// NumModes is the number of modes used in the combination.
	NumModes int
}

// Run executes the response spectrum analysis and returns a SpectrumResult.
//
//  1. Runs ModalAnalysis (assembles K, M; solves eigenproblem).
//  2. For each mode, computes the modal spectral displacement sd_k = Sa(Tk)/ωk².
//  3. Peak modal displacement vector: u_k = sd_k · Γ_{k,d} · φ_k  (per direction d).
//  4. Combines over modes with CQC (or SRSS) to obtain the peak displacement envelope.
//  5. Computes peak base shear per direction.
func (a *ResponseSpectrumAnalysis) Run() (*SpectrumResult, error) {
	if a.Spectrum == nil {
		return nil, fmt.Errorf("rsa: Spectrum must not be nil")
	}
	numModes := a.NumModes
	if numModes <= 0 {
		numModes = 10
	}
	xi := a.DampingRatio
	if xi <= 0 {
		xi = 0.05
	}
	dirs := a.Directions
	if dirs == 0 {
		dirs = DirXYZ
	}

	// ── 1. Modal analysis ─────────────────────────────────────────────────
	modal := &ModalAnalysis{
		Dom:      a.Dom,
		Masses:   a.Masses,
		NumModes: numModes,
	}
	mr, err := modal.Run()
	if err != nil {
		return nil, fmt.Errorf("rsa: %w", err)
	}
	nm := mr.NumModes
	ndof := len(mr.ModeShapes[0])

	// ── 2. Spectral displacements Sd_k = Sa(Tk) / ωk²  ───────────────────
	sd := make([]float64, nm)
	for k := 0; k < nm; k++ {
		Tk := mr.Periods[k]
		sa := a.Spectrum.SaAt(Tk)
		if mr.Omega2[k] > 0 {
			sd[k] = sa / mr.Omega2[k]
		}
	}

	// ── 3. Peak modal displacement vectors u_{k,d} = Sd_k · Γ_{k,d} · φ_k ─
	// We compute combined-direction peak: u_k[i] = sqrt(sum_d (Sd_k·Γ_{k,d}·φ_k[i])²)
	// but store per-direction modal contributions for base shear.

	// modalDispDir[k][d][dof] = Sd_k * Γ_{k,d} * φ_k[dof]
	activeDirs := []int{}
	for d, flag := range []RSADirection{DirX, DirY, DirZ} {
		if dirs&flag != 0 {
			activeDirs = append(activeDirs, d)
		}
	}

	// Per-mode envelope displacement (CQC across directions treated independently,
	// then SRSS across orthogonal directions as per EC8 §4.3.3.5.2).
	// modalContrib[k][d][dof] stores the signed modal contribution.
	type modeDir [3][]float64
	contrib := make([]modeDir, nm) // contrib[k][d] = slice of ndof
	for k := 0; k < nm; k++ {
		for d := 0; d < 3; d++ {
			contrib[k][d] = make([]float64, ndof)
		}
		for _, d := range activeDirs {
			gamma := mr.ParticipationFactors[k][d]
			for i := 0; i < ndof; i++ {
				contrib[k][d][i] = sd[k] * gamma * mr.ModeShapes[k][i]
			}
		}
	}

	// ── 4. CQC (or SRSS) combination per direction ────────────────────────
	// For each direction d: r_d[i]² = Σᵢ Σⱼ ρᵢⱼ · r_{k,d}[i] · r_{k,d}[i]  (same dof i)
	// Actually the standard CQC formula for a scalar response quantity r is:
	//   r = sqrt(ΣᵢΣⱼ ρᵢⱼ · rᵢ · rⱼ)
	// Applied DOF-by-DOF: r_d[dof] = sqrt(ΣᵢΣⱼ ρᵢⱼ · u_{i,d}[dof] · u_{j,d}[dof])

	// Pre-compute ρ matrix.
	rho := make([][]float64, nm)
	for i := 0; i < nm; i++ {
		rho[i] = make([]float64, nm)
		for j := 0; j < nm; j++ {
			if a.UseSRSS || i == j {
				if i == j {
					rho[i][j] = 1
				} else {
					rho[i][j] = 0 // SRSS: off-diagonal ignored
				}
			} else {
				oi := math.Sqrt(mr.Omega2[i])
				oj := math.Sqrt(mr.Omega2[j])
				rho[i][j] = CQCCorrelation(oi, oj, xi)
			}
		}
	}

	// peakDisp[d][dof] = CQC peak per direction
	peakDisp := [3][]float64{
		make([]float64, ndof),
		make([]float64, ndof),
		make([]float64, ndof),
	}
	for _, d := range activeDirs {
		for dof := 0; dof < ndof; dof++ {
			var r2 float64
			for i := 0; i < nm; i++ {
				for j := 0; j < nm; j++ {
					r2 += rho[i][j] * contrib[i][d][dof] * contrib[j][d][dof]
				}
			}
			if r2 > 0 {
				peakDisp[d][dof] = math.Sqrt(r2)
			}
		}
	}

	// ── 5. Combine orthogonal directions (SRSS per EC8) ───────────────────
	// totalPeak[dof] = sqrt(peak_X² + peak_Y² + peak_Z²)
	// Also store per-node [6] values.
	dpn := a.Dom.DOFPerNode
	numNodes := len(a.Dom.Nodes)
	maxDisp := make([][6]float64, numNodes)

	for node := 0; node < numNodes; node++ {
		for localDOF := 0; localDOF < dpn && localDOF < 6; localDOF++ {
			gDOF := node*dpn + localDOF
			var sum2 float64
			for d := 0; d < 3; d++ {
				v := peakDisp[d][gDOF]
				sum2 += v * v
			}
			maxDisp[node][localDOF] = math.Sqrt(sum2)
		}
	}

	// ── 6. Modal base shear and CQC combined ──────────────────────────────
	// Base shear in direction d for mode k: Vk_d = Γ_{k,d}² · ωk² · ... no —
	// simpler: V_{k,d} = Sa(Tk) · Γ_{k,d}² · (modal mass = 1 since M-normalised)
	//                  = Sa(Tk) · Γ²_{k,d}
	// CQC combination: V_d = sqrt(Σᵢ Σⱼ ρᵢⱼ · V_{i,d} · V_{j,d})
	// (V_{k,d} signed = Sa(Tk) * Γ_{k,d})

	modalBS := make([][3]float64, nm)
	for k := 0; k < nm; k++ {
		Tk := mr.Periods[k]
		sa := a.Spectrum.SaAt(Tk)
		for _, d := range activeDirs {
			gamma := mr.ParticipationFactors[k][d]
			modalBS[k][d] = sa * gamma // signed modal base shear contribution
		}
	}

	var maxBS [3]float64
	for _, d := range activeDirs {
		var r2 float64
		for i := 0; i < nm; i++ {
			for j := 0; j < nm; j++ {
				r2 += rho[i][j] * modalBS[i][d] * modalBS[j][d]
			}
		}
		if r2 > 0 {
			maxBS[d] = math.Sqrt(r2)
		}
	}

	// ── Build ModalDisplacements (combined direction, per mode) ───────────
	modalDisp := make([][]float64, nm)
	for k := 0; k < nm; k++ {
		md := make([]float64, ndof)
		for dof := 0; dof < ndof; dof++ {
			var sum2 float64
			for _, d := range activeDirs {
				v := contrib[k][d][dof]
				sum2 += v * v
			}
			md[dof] = math.Sqrt(sum2)
		}
		modalDisp[k] = md
	}

	return &SpectrumResult{
		Modal:              mr,
		ModalDisplacements: modalDisp,
		MaxDisplacements:   maxDisp,
		ModalBaseShear:     modalBS,
		MaxBaseShear:       maxBS,
		NumModes:           nm,
	}, nil
}
