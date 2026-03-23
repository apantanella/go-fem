package analysis

import (
	"fmt"
	"math"

	"go-fem/domain"
	"go-fem/solver"

	"gonum.org/v1/gonum/mat"
)

// NLResult holds the output of a nonlinear static analysis.
type NLResult struct {
	// U is the converged displacement vector.
	U *mat.VecDense

	// Converged reports whether the Newton-Raphson loop reached the tolerance.
	Converged bool

	// Iterations is the number of iterations performed.
	Iterations int

	// ResidualHistory contains ||R||/||F_ext|| at the end of each iteration.
	ResidualHistory []float64
}

// StaticNonlinearAnalysis performs a materially nonlinear static analysis
// using the Newton-Raphson (full Newton) method.
//
// Algorithm:
//
//  1. Assemble external force vector F_ext and initialise DOF mapping.
//  2. Initialise U to prescribed boundary values (non-zero settlements).
//  3. NR loop (max MaxIter iterations):
//     a. Update all elements with current U.
//     b. Assemble tangent K_T and internal forces R_int.
//     c. Compute unbalanced force R = F_ext − R_int.
//     d. Apply incremental BCs: K_T·ΔU = R with ΔU[constrained] = 0.
//     e. Check convergence: ||R|| / ||F_ext|| < Tol.
//     f. Solve K_T · ΔU = R; update U += ΔU.
//  4. Commit all elements on convergence.
//
// Notes:
//   - If MaxIter ≤ 0 it defaults to 50.
//   - If Tol ≤ 0 it defaults to 1e-6 (relative residual).
//   - For linear problems the algorithm converges in exactly one iteration.
//   - Uses LU solver by default if Solver is nil (handles non-SPD tangents
//     near full plasticity).
type StaticNonlinearAnalysis struct {
	Dom     *domain.Domain
	Solver  solver.LinearSolver
	MaxIter int
	Tol     float64
}

// Run executes the Newton-Raphson iterations and returns the converged
// displacement vector together with convergence details.
func (a *StaticNonlinearAnalysis) Run() (*NLResult, error) {
	maxIter := a.MaxIter
	if maxIter <= 0 {
		maxIter = 50
	}
	tol := a.Tol
	if tol <= 0 {
		tol = 1e-6
	}
	slv := a.Solver
	if slv == nil {
		slv = solver.LU{}
	}

	dom := a.Dom

	// ── 1. Assemble external loads and DOF layout ─────────────────────────
	// Assemble() builds K (linear), F_ext, and sets DOFPerNode / dofOffset.
	dom.Assemble()

	// Save F_ext before BCs are applied.
	ndof := dom.K.RawMatrix().Rows
	Fext := mat.NewVecDense(ndof, nil)
	Fext.CloneFromVec(dom.F)

	// Norm of external force (used for relative convergence check).
	normFext := vecNorm(Fext)
	if normFext < 1e-30 {
		// Zero external load → zero solution.
		U := mat.NewVecDense(ndof, nil)
		// Apply prescribed non-zero displacements.
		setPrescribed(U, dom)
		return &NLResult{U: U, Converged: true, Iterations: 0}, nil
	}

	// ── 2. Initialise U with prescribed displacements ─────────────────────
	U := mat.NewVecDense(ndof, nil)
	setPrescribed(U, dom)

	res := &NLResult{ResidualHistory: make([]float64, 0, maxIter)}

	// ── 3. Newton-Raphson loop ────────────────────────────────────────────
	for iter := 0; iter < maxIter; iter++ {

		// (a) Update all elements with current displacements.
		for _, elem := range dom.Elements {
			ue := dom.ElementDisp(elem, U)
			if err := elem.Update(ue); err != nil {
				return nil, fmt.Errorf("NR iter %d: element update: %w", iter+1, err)
			}
		}

		// (b) Assemble tangent K_T and internal resisting force R_int.
		KT := dom.AssembleTangent()
		Rint := dom.AssembleResisting()

		// (c) Unbalanced force R = F_ext − R_int.
		R := mat.NewVecDense(ndof, nil)
		R.SubVec(Fext, Rint)

		// (d) Apply incremental BCs (sets R[constrained]=0, enforces ΔU=0).
		dom.ApplyBCsIncremental(KT, R)

		// (e) Convergence check on the unbalanced force.
		normR := vecNorm(R)
		relRes := normR / normFext
		res.ResidualHistory = append(res.ResidualHistory, relRes)

		if relRes < tol {
			res.Converged = true
			res.Iterations = iter + 1
			break
		}

		// (f) Solve K_T · ΔU = R.
		dU, err := slv.Solve(KT, R)
		if err != nil {
			return nil, fmt.Errorf("NR iter %d: solver: %w", iter+1, err)
		}
		U.AddVec(U, dU)
	}

	if !res.Converged {
		res.Iterations = maxIter
		// Return non-fatal; caller can inspect residual history.
		// Still commit the last state so post-processing is possible.
	}

	// ── 4. Re-update elements with converged U and commit state ───────────
	for _, elem := range dom.Elements {
		ue := dom.ElementDisp(elem, U)
		elem.Update(ue) //nolint:errcheck — final update
		elem.CommitState()
	}

	res.U = U
	return res, nil
}

// setPrescribed initialises constrained DOFs in U to their prescribed values.
func setPrescribed(U *mat.VecDense, dom *domain.Domain) {
	dpn := dom.DOFPerNode
	for _, bc := range dom.BCs {
		off := dom.DOFOffset(bc.DOF)
		if off < 0 {
			continue
		}
		gdof := bc.NodeID*dpn + off
		if gdof < U.Len() {
			U.SetVec(gdof, bc.Value)
		}
	}
}

// vecNorm returns the Euclidean norm of a vector.
func vecNorm(v *mat.VecDense) float64 {
	n := v.Len()
	var s float64
	for i := 0; i < n; i++ {
		x := v.AtVec(i)
		s += x * x
	}
	return math.Sqrt(s)
}
