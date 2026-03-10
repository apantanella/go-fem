// Package material defines 3D constitutive models (Layer 1).
// Inspired by the OpenSees NDMaterial architecture.
package material

import "gonum.org/v1/gonum/mat"

// Material3D is the interface for 3D constitutive laws.
// Strain and stress vectors use Voigt notation (6 components):
//
//	[σxx, σyy, σzz, τxy, τyz, τxz]
type Material3D interface {
	// SetTrialStrain sets the trial strain and computes stress.
	SetTrialStrain(strain *mat.VecDense) error

	// GetStress returns the current stress vector (6 components).
	GetStress() *mat.VecDense

	// GetTangent returns the 6×6 material tangent matrix D.
	GetTangent() *mat.Dense

	// CommitState commits the current trial state.
	CommitState() error

	// RevertToStart resets the material to its initial state.
	RevertToStart() error
}
