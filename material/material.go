// Package material defines 3D and uniaxial constitutive models (Layer 1).
// Inspired by the OpenSees NDMaterial / UniaxialMaterial architecture.
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

// UniaxialMaterial is the interface for 1D (uniaxial) constitutive laws.
// It models the σ–ε relationship along a single axis.
//
// Sign convention (consistent with go-fem throughout):
//
//	tension positive, compression negative.
type UniaxialMaterial interface {
	// SetTrialStrain sets the trial axial strain and updates the trial state.
	SetTrialStrain(eps float64) error

	// GetStress returns the current trial stress [MPa or consistent units].
	GetStress() float64

	// GetTangent returns the current consistent algorithmic tangent dσ/dε.
	GetTangent() float64

	// CommitState commits the current trial state as the converged state.
	CommitState() error

	// RevertToStart resets the material to its initial (stress-free) state.
	RevertToStart() error
}
