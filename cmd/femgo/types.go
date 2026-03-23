package main

// ---------------------------------------------------------------------------
// JSON input types
// ---------------------------------------------------------------------------

type ProblemInput struct {
	Dimensions         string          `json:"dimensions,omitempty"`    // "2D" | "3D" (optional – triggers compatibility check)
	AnalysisType       string          `json:"analysis_type,omitempty"` // "" | "static" | "response_spectrum"
	Materials          []MaterialInput `json:"materials"`
	Nodes              [][3]float64    `json:"nodes"`
	Elements           []ElementInput  `json:"elements"`
	BoundaryConditions []BCInput       `json:"boundary_conditions"`
	Loads              []LoadInput     `json:"loads"`
	Solver             string          `json:"solver,omitempty"`         // "cholesky" (default) | "lu" | "skyline" | "cg" | "gmres"
	SolverOptions      SolverOptions   `json:"solver_options,omitempty"` // optional tuning for iterative/sparse solvers

	// Response spectrum analysis — populated when analysis_type = "response_spectrum"
	Modal    *ModalInput  `json:"modal,omitempty"`
	Spectrum [][2]float64 `json:"spectrum,omitempty"` // [[T, Sa], ...] — piecewise-linear elastic spectrum
	RSA      *RSAOptions  `json:"rsa,omitempty"`
}

// ModalInput configures the modal (eigenvalue) analysis step for RSA.
type ModalInput struct {
	Masses   []MassInput `json:"masses"`    // element mass definitions
	NumModes int         `json:"num_modes"` // number of modes to extract
}

// MassInput associates a mass density with an element index.
type MassInput struct {
	Element int     `json:"element"` // element index (0-based)
	Rho     float64 `json:"rho"`     // mass density (consistent units, e.g. kg/mm³)
}

// RSAOptions controls the response spectrum combination.
type RSAOptions struct {
	DampingRatio float64 `json:"damping_ratio,omitempty"` // uniform modal damping ξ (default 0.05)
	Combination  string  `json:"combination,omitempty"`   // "cqc" (default) | "srss"
	Directions   string  `json:"directions,omitempty"`    // "xyz" (default) | "x" | "y" | "z" | "xy" | "xz" | "yz"
}

// SolverOptions holds optional tuning parameters for iterative and sparse solvers.
// Zero values fall back to solver-specific defaults.
type SolverOptions struct {
	Tol     float64 `json:"tol,omitempty"`      // relative residual tolerance (CG, GMRES)
	MaxIter int     `json:"max_iter,omitempty"` // maximum iterations / outer restarts (CG, GMRES)
	Restart int     `json:"restart,omitempty"`  // Krylov subspace dimension per restart (GMRES only)
	ZeroTol float64 `json:"zero_tol,omitempty"` // sparsity threshold for pattern detection (SkylineLDL)
}

type MaterialInput struct {
	ID   string `json:"id"`
	Type string `json:"type"` // "isotropic_linear" | "orthotropic_linear"

	// Isotropic parameters
	E  float64 `json:"E,omitempty"`
	Nu float64 `json:"nu,omitempty"`

	// Orthotropic parameters (all required for orthotropic_linear)
	Ex  float64 `json:"Ex,omitempty"`
	Ey  float64 `json:"Ey,omitempty"`
	Ez  float64 `json:"Ez,omitempty"`
	Nxy float64 `json:"nxy,omitempty"` // Poisson's ratio ν_xy (loading x, lateral y)
	Nyz float64 `json:"nyz,omitempty"` // ν_yz
	Nxz float64 `json:"nxz,omitempty"` // ν_xz
	Gxy float64 `json:"Gxy,omitempty"`
	Gyz float64 `json:"Gyz,omitempty"`
	Gxz float64 `json:"Gxz,omitempty"`
}

type ElementInput struct {
	Type     string `json:"type"`     // element type (e.g. tet4, hexa8, shell_mitc4, dkt3)
	Material string `json:"material"` // material id (for solid elements)
	Nodes    []int  `json:"nodes"`

	// Truss parameters
	E float64 `json:"E,omitempty"` // Young's modulus (truss/beam direct)
	A float64 `json:"A,omitempty"` // Cross-sectional area

	// Beam parameters
	G     float64    `json:"G,omitempty"`      // Shear modulus
	Iy    float64    `json:"Iy,omitempty"`     // Moment of inertia y
	Iz    float64    `json:"Iz,omitempty"`     // Moment of inertia z
	J     float64    `json:"J,omitempty"`      // Torsion constant
	VecXZ [3]float64 `json:"vec_xz,omitempty"` // Orientation vector
	// Timoshenko shear areas (Asy = κy·A, Asz = κz·A). Default: 5/6·A if omitted.
	Asy float64 `json:"Asy,omitempty"` // Effective shear area in local y
	Asz float64 `json:"Asz,omitempty"` // Effective shear area in local z

	// Shell/Quad parameters
	Nu        float64 `json:"nu,omitempty"`         // Poisson's ratio (shell/quad direct)
	Thickness float64 `json:"thickness,omitempty"`  // Shell/plate thickness
	PlaneType string  `json:"plane_type,omitempty"` // "stress" | "strain"

	// ZeroLength parameters
	Springs [6]float64 `json:"springs,omitempty"` // Spring stiffnesses [kUX..kRZ]
}

type BCInput struct {
	Node   int       `json:"node"`
	DOFs   []int     `json:"dofs"`             // e.g. [0,1,2] for translations, [0,1,2,3,4,5] for all
	Values []float64 `json:"values,omitempty"` // prescribed displacements (default 0); len must match DOFs if provided
}

// LoadInput covers all load types via a "type" discriminator.
// For nodal loads, omit "type" or set it to "nodal".
//
// Load types:
//
//	nodal          – concentrated force/moment on a node (node, dof, value)
//	surface_pressure – uniform pressure on a 4-node face (face_nodes[4], pressure)
//	beam_dist      – UDL on a beam element (element, dir[3], intensity)
//	body_force     – gravity/body force on a solid element (element, rho, g[3])
type LoadInput struct {
	// Nodal load fields
	Node  int     `json:"node,omitempty"`
	DOF   int     `json:"dof,omitempty"` // 0=UX,1=UY,2=UZ,3=RX,4=RY,5=RZ
	Value float64 `json:"value,omitempty"`

	// Load type discriminator (omit or "nodal" for concentrated nodal loads)
	Type string `json:"type,omitempty"` // "nodal"|"surface_pressure"|"beam_dist"|"body_force"

	// Surface pressure
	FaceNodes [4]int  `json:"face_nodes,omitempty"` // global node IDs (CCW from outside)
	Pressure  float64 `json:"pressure,omitempty"`

	// Beam distributed load
	Element   int        `json:"element,omitempty"`   // element index
	Dir       [3]float64 `json:"dir,omitempty"`       // global direction unit vector
	Intensity float64    `json:"intensity,omitempty"` // load per unit length

	// Body force
	Rho float64    `json:"rho,omitempty"` // mass density
	G   [3]float64 `json:"g,omitempty"`   // gravitational acceleration vector
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

type ProblemOutput struct {
	Success       bool                  `json:"success"`
	Error         string                `json:"error,omitempty"`
	Info          *InfoOutput           `json:"info,omitempty"`
	Displacements []DisplacementOutput  `json:"displacements,omitempty"`
	ElementForces []ElementForcesOutput `json:"element_forces,omitempty"`
	Summary       *SummaryOutput        `json:"summary,omitempty"`
	ElapsedMs     float64               `json:"elapsed_ms,omitempty"`

	// Response spectrum analysis output (only when analysis_type = "response_spectrum")
	Modal   *ModalOutput   `json:"modal,omitempty"`
	Seismic *SeismicOutput `json:"seismic,omitempty"`
}

// ModalOutput contains the modal analysis results echoed in the RSA response.
type ModalOutput struct {
	NumModes int              `json:"num_modes"`
	Modes    []ModeInfoOutput `json:"modes"`
}

// ModeInfoOutput describes one natural mode.
type ModeInfoOutput struct {
	Mode                 int     `json:"mode"` // 1-based
	FrequencyHz          float64 `json:"frequency_hz"`
	PeriodS              float64 `json:"period_s"`
	EffectiveMassX       float64 `json:"effective_mass_x"` // fraction 0–1
	EffectiveMassY       float64 `json:"effective_mass_y"`
	EffectiveMassZ       float64 `json:"effective_mass_z"`
	CumulativeEffMassX   float64 `json:"cumulative_eff_mass_x"`
	CumulativeEffMassY   float64 `json:"cumulative_eff_mass_y"`
	CumulativeEffMassZ   float64 `json:"cumulative_eff_mass_z"`
	SpectralAcceleration float64 `json:"sa"` // Sa(Tk) from spectrum
	ModalBaseShearX      float64 `json:"modal_base_shear_x"`
	ModalBaseShearY      float64 `json:"modal_base_shear_y"`
	ModalBaseShearZ      float64 `json:"modal_base_shear_z"`
}

// SeismicOutput contains the CQC/SRSS combined peak seismic results.
type SeismicOutput struct {
	Combination        string               `json:"combination"` // "cqc" | "srss"
	MaxBaseShearX      float64              `json:"max_base_shear_x"`
	MaxBaseShearY      float64              `json:"max_base_shear_y"`
	MaxBaseShearZ      float64              `json:"max_base_shear_z"`
	MaxDisplacements   []DisplacementOutput `json:"max_displacements"` // peak envelope per node
	MaxAbsDisplacement MaxDispOutput        `json:"max_abs_displacement"`
}

// ---------------------------------------------------------------------------
// Element forces / stress output types
// ---------------------------------------------------------------------------

// ElementForcesOutput holds post-processing results for one element.
// Fields are populated depending on element type; unused fields are omitted.
type ElementForcesOutput struct {
	ID   int    `json:"id"`
	Type string `json:"type"`

	// Truss (truss3d, corot_truss)
	N     *float64 `json:"N,omitempty"`     // axial force (positive = tension)
	Sigma *float64 `json:"sigma,omitempty"` // axial stress N/A

	// Beam (elastic_beam3d) — end forces in local element frame
	EndI *BeamEndOutput `json:"end_i,omitempty"`
	EndJ *BeamEndOutput `json:"end_j,omitempty"`

	// Solid (tet4, hexa8, tet10, brick20) and Quad (quad4) — centroidal stress
	Stress *StressOutput `json:"stress,omitempty"`

	// Shell (shell_mitc4) — section resultants in local shell frame
	ShellForces *ShellForcesOutput `json:"shell_forces,omitempty"`

	// Spring elements (zerolength_*) — spring forces/moments (tension/CCW positive)
	SpringForces *SpringForcesOutput `json:"spring_forces,omitempty"`
}

// BeamEndOutput holds the six cross-section forces at one beam end in local coords.
// N > 0 tension; Vy/Vz shear; Mx torsion; My/Mz bending.
type BeamEndOutput struct {
	N  float64 `json:"N"`
	Vy float64 `json:"Vy"`
	Vz float64 `json:"Vz"`
	Mx float64 `json:"Mx"`
	My float64 `json:"My"`
	Mz float64 `json:"Mz"`
}

// StressOutput holds the Cauchy stress components and von Mises equivalent.
// For plane elements (quad4) szz, tyz, txz are zero.
type StressOutput struct {
	Sxx      float64 `json:"sxx"`
	Syy      float64 `json:"syy"`
	Szz      float64 `json:"szz"`
	Txy      float64 `json:"txy"`
	Tyz      float64 `json:"tyz"`
	Txz      float64 `json:"txz"`
	VonMises float64 `json:"von_mises"`
}

// ShellForcesOutput holds in-plane force and bending moment resultants per unit length
// in the local shell coordinate system.
type ShellForcesOutput struct {
	Nx  float64 `json:"Nx"`  // membrane force/unit length local x
	Ny  float64 `json:"Ny"`  // membrane force/unit length local y
	Nxy float64 `json:"Nxy"` // membrane shear/unit length
	Mx  float64 `json:"Mx"`  // bending moment/unit length about local x
	My  float64 `json:"My"`  // bending moment/unit length about local y
	Mxy float64 `json:"Mxy"` // twisting moment/unit length
}

// SpringForcesOutput holds spring/connector forces and moments (tension/CCW positive).
// Only the components active for the specific element type are populated.
type SpringForcesOutput struct {
	Fx *float64 `json:"Fx,omitempty"`
	Fy *float64 `json:"Fy,omitempty"`
	Fz *float64 `json:"Fz,omitempty"`
	Mx *float64 `json:"Mx,omitempty"`
	My *float64 `json:"My,omitempty"`
	Mz *float64 `json:"Mz,omitempty"`
}

type InfoOutput struct {
	NumNodes    int    `json:"num_nodes"`
	NumElements int    `json:"num_elements"`
	NumDOFs     int    `json:"num_dofs"`
	DOFPerNode  int    `json:"dof_per_node"`
	Solver      string `json:"solver"`
	Dimensions  string `json:"dimensions,omitempty"` // "2D" | "3D" as declared in the request
}

type DisplacementOutput struct {
	Node int     `json:"node"`
	Ux   float64 `json:"ux"`
	Uy   float64 `json:"uy"`
	Uz   float64 `json:"uz"`
	Rx   float64 `json:"rx,omitempty"`
	Ry   float64 `json:"ry,omitempty"`
	Rz   float64 `json:"rz,omitempty"`
}

type SummaryOutput struct {
	MaxAbsDisplacement MaxDispOutput `json:"max_abs_displacement"`
}

type MaxDispOutput struct {
	Node      int     `json:"node"`
	Component string  `json:"component"`
	Value     float64 `json:"value"`
}
