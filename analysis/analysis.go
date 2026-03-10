// Package analysis orchestrates the FEM solution process (Layer 4).
// Inspired by the OpenSees Analysis architecture.
package analysis

import (
	"fmt"

	"go-fem/domain"
	"go-fem/solver"

	"gonum.org/v1/gonum/mat"
)

// StaticLinearAnalysis performs a single-step linear static analysis.
// This is equivalent to a degenerate Newton-Raphson with one iteration
// (since the tangent is constant for linear problems).
type StaticLinearAnalysis struct {
	Dom    *domain.Domain
	Solver solver.LinearSolver
}

// Run executes the analysis and returns the displacement vector.
func (a *StaticLinearAnalysis) Run() (*mat.VecDense, error) {
	// 1. Assemble global K and F.
	a.Dom.Assemble()

	// 2. Apply Dirichlet boundary conditions.
	a.Dom.ApplyDirichletBC()

	// 3. Solve K·U = F.
	U, err := a.Solver.Solve(a.Dom.K, a.Dom.F)
	if err != nil {
		return nil, fmt.Errorf("analysis: %w", err)
	}

	// 4. Update each element with its displacements for post-processing.
	for _, elem := range a.Dom.Elements {
		ue := a.Dom.ElementDisp(elem, U)
		elem.Update(ue)
	}

	return U, nil
}
