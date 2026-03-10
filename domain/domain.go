// Package domain manages the FEM mesh, assembly, and boundary conditions (Layer 3).
// Inspired by the OpenSees Domain architecture.
package domain

import (
	"go-fem/element"

	"gonum.org/v1/gonum/mat"
)

// Node3D represents a node with 3D coordinates.
type Node3D struct {
	ID    int
	Coord [3]float64 // (X, Y, Z)
}

// BC represents a Dirichlet (fixed) boundary condition.
type BC struct {
	NodeID int
	DOF    int     // 0=UX, 1=UY, 2=UZ, 3=RX, 4=RY, 5=RZ
	Value  float64 // prescribed displacement (usually 0)
}

// NodalLoad represents a concentrated force or moment on a node.
type NodalLoad struct {
	NodeID int
	DOF    int     // 0=UX, 1=UY, 2=UZ, 3=RX, 4=RY, 5=RZ
	Value  float64 // force/moment magnitude
}

// Domain holds the complete FEM model.
type Domain struct {
	Nodes    []Node3D
	Elements []element.Element
	BCs      []BC
	Loads    []NodalLoad

	// DOFPerNode is auto-detected during Assemble:
	// 3 for pure solid/truss, 6 when beams/shells are present.
	DOFPerNode int

	K *mat.Dense    // global stiffness (assembled)
	F *mat.VecDense // global force vector
}

// NewDomain creates an empty domain.
func NewDomain() *Domain {
	return &Domain{}
}

// AddNode appends a node and returns its ID.
func (d *Domain) AddNode(x, y, z float64) int {
	id := len(d.Nodes)
	d.Nodes = append(d.Nodes, Node3D{ID: id, Coord: [3]float64{x, y, z}})
	return id
}

// AddElement appends an element.
func (d *Domain) AddElement(e element.Element) {
	d.Elements = append(d.Elements, e)
}

// FixDOF adds a zero-displacement Dirichlet BC.
func (d *Domain) FixDOF(nodeID, dof int) {
	d.BCs = append(d.BCs, BC{NodeID: nodeID, DOF: dof, Value: 0})
}

// FixNode fixes all translational DOFs (UX, UY, UZ) of a node.
func (d *Domain) FixNode(nodeID int) {
	for dof := 0; dof < 3; dof++ {
		d.FixDOF(nodeID, dof)
	}
}

// FixNodeAll fixes all 6 DOFs (translations + rotations) of a node.
func (d *Domain) FixNodeAll(nodeID int) {
	for dof := 0; dof < 6; dof++ {
		d.FixDOF(nodeID, dof)
	}
}

// ApplyLoad adds a nodal force or moment.
func (d *Domain) ApplyLoad(nodeID, dof int, value float64) {
	d.Loads = append(d.Loads, NodalLoad{NodeID: nodeID, DOF: dof, Value: value})
}

// NumDOF returns the total number of DOFs.
func (d *Domain) NumDOF() int {
	dpn := d.DOFPerNode
	if dpn == 0 {
		dpn = 3
	}
	return len(d.Nodes) * dpn
}

// Assemble constructs the global K and F.
// Auto-detects DOFPerNode from the maximum across all elements.
func (d *Domain) Assemble() {
	// Auto-detect DOFs per node
	d.DOFPerNode = 3
	for _, elem := range d.Elements {
		if dpn := elem.DOFPerNode(); dpn > d.DOFPerNode {
			d.DOFPerNode = dpn
		}
	}

	dpn := d.DOFPerNode
	ndof := len(d.Nodes) * dpn
	d.K = mat.NewDense(ndof, ndof, nil)
	d.F = mat.NewVecDense(ndof, nil)

	// --- Scatter element stiffness into global K ---
	for _, elem := range d.Elements {
		ke := elem.GetTangentStiffness()
		nids := elem.NodeIDs()
		dofTypes := elem.DOFTypes()
		eldof := elem.NumDOF()
		elemDPN := elem.DOFPerNode()

		// Build local → global DOF map using DOF types
		globalDOFs := make([]int, eldof)
		for i, dt := range dofTypes {
			nodeIdx := i / elemDPN
			nodeID := nids[nodeIdx]
			globalDOFs[i] = nodeID*dpn + int(dt)
		}

		for i := 0; i < eldof; i++ {
			gi := globalDOFs[i]
			for j := 0; j < eldof; j++ {
				gj := globalDOFs[j]
				d.K.Set(gi, gj, d.K.At(gi, gj)+ke.At(i, j))
			}
		}
	}

	// --- Assemble load vector ---
	for _, load := range d.Loads {
		gdof := load.NodeID*dpn + load.DOF
		d.F.SetVec(gdof, d.F.AtVec(gdof)+load.Value)
	}
}

// ApplyDirichletBC modifies K and F for prescribed displacements.
func (d *Domain) ApplyDirichletBC() {
	dpn := d.DOFPerNode
	ndof := len(d.Nodes) * dpn
	for _, bc := range d.BCs {
		gdof := bc.NodeID*dpn + bc.DOF
		if gdof >= ndof {
			continue
		}
		for j := 0; j < ndof; j++ {
			d.K.Set(gdof, j, 0)
			d.K.Set(j, gdof, 0)
		}
		d.K.Set(gdof, gdof, 1)
		d.F.SetVec(gdof, bc.Value)
	}
}

// SetDisplacements extracts per-node displacements from the global solution vector.
// Returns a slice indexed by node ID with up to 6 components [ux,uy,uz,rx,ry,rz].
func (d *Domain) SetDisplacements(U *mat.VecDense) [][6]float64 {
	dpn := d.DOFPerNode
	disp := make([][6]float64, len(d.Nodes))
	for i := range d.Nodes {
		for j := 0; j < dpn; j++ {
			disp[i][j] = U.AtVec(i*dpn + j)
		}
	}
	return disp
}

// ElementDisp extracts the displacement vector for a single element from the
// global solution U, using the same DOF-type mapping as Assemble().
// Must be called after Assemble() has set DOFPerNode.
func (d *Domain) ElementDisp(elem element.Element, U *mat.VecDense) []float64 {
	dpn := d.DOFPerNode
	nids := elem.NodeIDs()
	dofTypes := elem.DOFTypes()
	eldof := elem.NumDOF()
	elemDPN := elem.DOFPerNode()
	disp := make([]float64, eldof)
	for i, dt := range dofTypes {
		nodeIdx := i / elemDPN
		nodeID := nids[nodeIdx]
		disp[i] = U.AtVec(nodeID*dpn + int(dt))
	}
	return disp
}
