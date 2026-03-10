// Package section defines cross-section properties for 1D/2D structural elements.
package section

// BeamSection3D holds properties for a 3D Euler-Bernoulli or Timoshenko beam.
type BeamSection3D struct {
	A  float64 // Cross-sectional area
	Iy float64 // Moment of inertia about local y-axis (bending in xz plane)
	Iz float64 // Moment of inertia about local z-axis (bending in xy plane)
	J  float64 // Torsion constant (polar moment)
}

// ShellSection holds properties for shell/plate elements.
type ShellSection struct {
	Thickness float64 // Shell thickness
}
