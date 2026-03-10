// Package element defines finite element types (Layer 2).
// Inspired by the OpenSees Element architecture.
package element

import "gonum.org/v1/gonum/mat"

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

	// Update recomputes element state for the given global displacements.
	Update(disp []float64) error
}
