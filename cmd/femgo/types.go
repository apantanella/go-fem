package main

// ---------------------------------------------------------------------------
// JSON input types
// ---------------------------------------------------------------------------

type ProblemInput struct {
	Materials          []MaterialInput `json:"materials"`
	Nodes              [][3]float64    `json:"nodes"`
	Elements           []ElementInput  `json:"elements"`
	BoundaryConditions []BCInput       `json:"boundary_conditions"`
	Loads              []LoadInput     `json:"loads"`
	Solver             string          `json:"solver,omitempty"` // "cholesky" (default) | "lu"
}

type MaterialInput struct {
	ID   string  `json:"id"`
	Type string  `json:"type"` // "isotropic_linear"
	E    float64 `json:"E"`
	Nu   float64 `json:"nu"`
}

type ElementInput struct {
	Type     string `json:"type"`     // element type (see below)
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

	// Shell/Quad parameters
	Nu        float64 `json:"nu,omitempty"`         // Poisson's ratio (shell/quad direct)
	Thickness float64 `json:"thickness,omitempty"`  // Shell/plate thickness
	PlaneType string  `json:"plane_type,omitempty"` // "stress" | "strain"

	// ZeroLength parameters
	Springs [6]float64 `json:"springs,omitempty"` // Spring stiffnesses [kUX..kRZ]
}

type BCInput struct {
	Node int   `json:"node"`
	DOFs []int `json:"dofs"` // e.g. [0,1,2] for translations, [0,1,2,3,4,5] for all
}

type LoadInput struct {
	Node  int     `json:"node"`
	DOF   int     `json:"dof"` // 0=UX,1=UY,2=UZ,3=RX,4=RY,5=RZ
	Value float64 `json:"value"`
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

type InfoOutput struct {
	NumNodes    int    `json:"num_nodes"`
	NumElements int    `json:"num_elements"`
	NumDOFs     int    `json:"num_dofs"`
	DOFPerNode  int    `json:"dof_per_node"`
	Solver      string `json:"solver"`
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
