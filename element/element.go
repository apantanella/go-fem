// Package element defines finite element types (Layer 2).
// Inspired by the OpenSees Element architecture.
package element

import (
	"go-fem/dof"

	"gonum.org/v1/gonum/mat"
)

// EquivalentNodalLoader is implemented by elements that convert a uniformly
// distributed load into work-equivalent nodal forces.
// globalDir is a unit vector in the load direction (e.g. [0,-1,0] for gravity).
// intensity is the load per unit length (N/m).
// Returns a vector of length NumDOF() in global coordinates.
type EquivalentNodalLoader interface {
	EquivalentNodalLoad(globalDir [3]float64, intensity float64) *mat.VecDense
}

// LinearDistLoader is implemented by elements that convert a linearly varying
// (trapezoidal / triangular) distributed load into work-equivalent nodal forces.
// intensityI is the load per unit length at node i; intensityJ at node j.
// Returns a vector of length NumDOF() in global coordinates.
type LinearDistLoader interface {
	EquivalentNodalLoadLinear(globalDir [3]float64, intensityI, intensityJ float64) *mat.VecDense
}

// BodyForceLoader is implemented by elements that can compute nodal forces
// due to a body force (gravity, centrifugal, etc.).
// g is the body-force acceleration vector (e.g. [0,-9.81,0]).
// rho is the mass density.
// Returns a vector of length NumDOF() in global coordinates.
type BodyForceLoader interface {
	BodyForceLoad(g [3]float64, rho float64) *mat.VecDense
}

// MassMatrixAssembler is implemented by elements that can compute their
// consistent mass matrix. rho is the mass density (kg/m³ or consistent units).
// Returns a square matrix of size NumDOF() × NumDOF().
type MassMatrixAssembler interface {
	GetMassMatrix(rho float64) *mat.Dense
}

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
