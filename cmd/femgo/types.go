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
	Success       bool                 `json:"success"`
	Error         string               `json:"error,omitempty"`
	Info          *InfoOutput          `json:"info,omitempty"`
	Displacements []DisplacementOutput `json:"displacements,omitempty"`
	Summary       *SummaryOutput       `json:"summary,omitempty"`
	ElapsedMs     float64              `json:"elapsed_ms,omitempty"`
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
