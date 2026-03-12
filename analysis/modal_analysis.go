package analysis

import (
	"fmt"
	"math"

	"go-fem/dof"
	"go-fem/domain"
	"go-fem/solver"
)

// ModalResult contains the output of a free-vibration (modal) analysis.
type ModalResult struct {
	// NumModes is the number of modes computed.
	NumModes int

	// Omega2 contains the eigenvalues ω² [rad²/s²] in ascending order.
	Omega2 []float64

	// Frequencies contains the natural frequencies [Hz].
	Frequencies []float64

	// Periods contains the natural periods T = 1/f [s].
	Periods []float64

	// ModeShapes is an (ndof × NumModes) matrix.
	// Columns are M-normalised mode shapes (φᵢᵀ·M·φᵢ = 1).
	ModeShapes [][]float64 // indexed [mode][globalDOF]

	// ParticipationFactors[k][d] = Γ_{k,d} = φₖᵀ · M · r_d
	// d = 0→X, 1→Y, 2→Z
	ParticipationFactors [][3]float64

	// EffectiveMass[k][d] = Γ²_{k,d} / m_total_d  (fraction 0–1)
	EffectiveMass [][3]float64

	// CumulativeEffectiveMass[k][d] = sum of EffectiveMass for modes 0..k
	CumulativeEffectiveMass [][3]float64
}

// ModalAnalysis performs a free-vibration (eigenvalue) analysis.
//
// Usage:
//
//	a := &ModalAnalysis{Dom: dom, Masses: masses, NumModes: 10}
//	result, err := a.Run()
type ModalAnalysis struct {
	// Dom is the FEM domain (nodes, elements, BCs must already be set up).
	Dom *domain.Domain

	// Masses associates a density with each element index.
	Masses []domain.ElementMass

	// NumModes is the number of modes to extract (default 10).
	NumModes int
}

// Run executes the modal analysis and returns the result.
//
//  1. Assembles K and M (full, without BCs).
//  2. Identifies free DOFs (not constrained by Dirichlet BCs).
//  3. Extracts reduced (free-DOF) K and M submatrices.
//  4. Solves the generalized eigenvalue problem K·φ = ω²·M·φ.
//  5. Expands modes to full DOF space and computes participation factors.
func (a *ModalAnalysis) Run() (*ModalResult, error) {
	numModes := a.NumModes
	if numModes <= 0 {
		numModes = 10
	}

	// ── 1. Assemble K and M ───────────────────────────────────────────────
	a.Dom.Assemble()
	a.Dom.AssembleMassMatrix(a.Masses)

	// ── 2. Free DOFs ──────────────────────────────────────────────────────
	freeDOFs := a.Dom.FreeDOFs()
	nFree := len(freeDOFs)
	if nFree == 0 {
		return nil, fmt.Errorf("modal: no free DOFs after applying boundary conditions")
	}
	if numModes > nFree {
		numModes = nFree
	}

	// ── 3. Reduced K and M ────────────────────────────────────────────────
	Kred := solver.ExtractSubmatrix(a.Dom.K, freeDOFs)
	Mred := solver.ExtractSubmatrix(a.Dom.M, freeDOFs)

	// ── 4. Solve generalized eigenvalue problem ────────────────────────────
	eigRes, err := solver.SolveGeneralizedEigen(Kred, Mred, numModes)
	if err != nil {
		return nil, fmt.Errorf("modal: %w", err)
	}

	// ── 5. Expand modes to full DOF space ─────────────────────────────────
	ndof := len(a.Dom.Nodes) * a.Dom.DOFPerNode
	fullModes := solver.ExpandModes(eigRes.Modes, freeDOFs, ndof)

	// Build result.
	res := &ModalResult{
		NumModes:                numModes,
		Omega2:                  eigRes.Omega2,
		Frequencies:             make([]float64, numModes),
		Periods:                 make([]float64, numModes),
		ModeShapes:              make([][]float64, numModes),
		ParticipationFactors:    make([][3]float64, numModes),
		EffectiveMass:           make([][3]float64, numModes),
		CumulativeEffectiveMass: make([][3]float64, numModes),
	}

	for k := 0; k < numModes; k++ {
		res.Frequencies[k] = solver.FrequencyHz(eigRes.Omega2[k])
		res.Periods[k] = solver.PeriodSeconds(eigRes.Omega2[k])
		col := make([]float64, ndof)
		for i := 0; i < ndof; i++ {
			col[i] = fullModes.At(i, k)
		}
		res.ModeShapes[k] = col
	}

	// ── 6. Participation factors and effective masses ──────────────────────
	// All quantities are computed on the REDUCED system (free DOFs only).
	// This guarantees the completeness condition:
	//   Σₖ Γ²_{k,d} = r_free^T · M_red · r_free  (= 1 when normalised)
	//
	// Direction vectors r_d (restricted to free DOFs):
	//   r_free_d[i] = 1 if freeDOFs[i] is a translation in direction d, else 0
	translDOFs := [3]dof.Type{dof.UX, dof.UY, dof.UZ}
	rFree := [3][]float64{
		make([]float64, nFree),
		make([]float64, nFree),
		make([]float64, nFree),
	}
	for i, g := range freeDOFs {
		dt := a.Dom.DOFTypeAt(g)
		for d := 0; d < 3; d++ {
			if dt == translDOFs[d] {
				rFree[d][i] = 1.0
			}
		}
	}

	// m_total_d = r_free_d^T · M_red · r_free_d  (excited mass in direction d)
	totalMass := [3]float64{}
	for d := 0; d < 3; d++ {
		for i := 0; i < nFree; i++ {
			if rFree[d][i] == 0 {
				continue
			}
			for j := 0; j < nFree; j++ {
				totalMass[d] += rFree[d][i] * Mred.At(i, j) * rFree[d][j]
			}
		}
	}

	// Participation factor: Γ_{k,d} = φₖ_free^T · M_red · r_free_d
	// φₖ_free is the mode shape restricted to free DOFs.
	for k := 0; k < numModes; k++ {
		phiFree := make([]float64, nFree)
		for i, g := range freeDOFs {
			phiFree[i] = res.ModeShapes[k][g]
		}
		for d := 0; d < 3; d++ {
			// Γ = φ_free^T · (M_red · r_free_d)
			var gamma float64
			for i := 0; i < nFree; i++ {
				var Mr float64
				for j := 0; j < nFree; j++ {
					Mr += Mred.At(i, j) * rFree[d][j]
				}
				gamma += phiFree[i] * Mr
			}
			res.ParticipationFactors[k][d] = gamma
			// Effective mass ratio = Γ² / m_total  (φ is M-normalised → modal mass = 1)
			if totalMass[d] > 0 {
				res.EffectiveMass[k][d] = (gamma * gamma) / totalMass[d]
			}
		}
	}

	// Cumulative effective mass.
	for k := 0; k < numModes; k++ {
		for d := 0; d < 3; d++ {
			if k == 0 {
				res.CumulativeEffectiveMass[k][d] = res.EffectiveMass[k][d]
			} else {
				res.CumulativeEffectiveMass[k][d] = res.CumulativeEffectiveMass[k-1][d] + res.EffectiveMass[k][d]
			}
		}
	}

	return res, nil
}

// AngularFrequency returns ω [rad/s] for mode k.
func (r *ModalResult) AngularFrequency(k int) float64 {
	if r.Omega2[k] <= 0 {
		return 0
	}
	return math.Sqrt(r.Omega2[k])
}
