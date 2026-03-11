// Package section defines cross-section properties for 1D/2D structural elements.
package section

// BeamSection3D holds properties for a 3D Euler-Bernoulli or Timoshenko beam.
type BeamSection3D struct {
	A  float64 // Cross-sectional area
	Iy float64 // Moment of inertia about local y-axis (bending in xz plane)
	Iz float64 // Moment of inertia about local z-axis (bending in xy plane)
	J  float64 // Torsion constant (polar moment)

	// Shear areas for Timoshenko beam (Asy = κy·A, Asz = κz·A).
	// If zero, defaults to 5/6·A (rectangular section approximation).
	Asy float64 // Effective shear area in local y (for bending in xy plane)
	Asz float64 // Effective shear area in local z (for bending in xz plane)
}

// ShellSection holds properties for shell/plate elements.
type ShellSection struct {
	Thickness float64 // Shell thickness
}
