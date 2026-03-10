// Package element defines finite element types (Layer 2).
// Inspired by the OpenSees Element architecture.
package element

import (
	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// Element is the interface for all finite elements.
type Element interface {
	// GetTangentStiffness returns the element stiffness matrix Ke.
	GetTangentStiffness() *mat.Dense

	// GetResistingForce returns the element internal force vector.
	GetResistingForce() *mat.VecDense

	// NodeIDs returns the global node IDs for this element.
	NodeIDs() []int

	// NumDOF returns the total number of DOFs for this element.
	NumDOF() int

	// DOFPerNode returns the number of DOFs per node (3 for solids/truss, 6 for beams/shells).
	DOFPerNode() int

	// DOFTypes returns the flat list of DOF types, one per local DOF.
	// Length must equal NumDOF(). Used for global assembly mapping.
	DOFTypes() []dof.Type

	// Update recomputes element state for the given element displacements.
	Update(disp []float64) error

	// CommitState commits the current trial state (for nonlinear analysis).
	CommitState() error

	// RevertToStart reverts the element to its initial state.
	RevertToStart() error
}
