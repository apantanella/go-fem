// Package domain manages the FEM mesh, assembly, and boundary conditions (Layer 3).
// Inspired by the OpenSees Domain architecture.
package domain

import (
	"go-fem/element"

	"gonum.org/v1/gonum/mat"
)

// Node3D represents a node with 3 translational DOFs.
type Node3D struct {
	ID    int
	Coord [3]float64 // (X, Y, Z)
}

// BC represents a Dirichlet (fixed) boundary condition.
type BC struct {
	NodeID int
	DOF    int     // 0 = X, 1 = Y, 2 = Z
	Value  float64 // prescribed displacement (usually 0)
}

// NodalLoad represents a concentrated force on a node.
type NodalLoad struct {
	NodeID int
	DOF    int     // 0 = X, 1 = Y, 2 = Z
	Value  float64 // force magnitude
}

// Domain holds the complete FEM model.
type Domain struct {
	Nodes    []Node3D
	Elements []element.Element
	BCs      []BC
	Loads    []NodalLoad

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

// FixNode fixes all 3 DOFs of a node.
func (d *Domain) FixNode(nodeID int) {
	for dof := 0; dof < 3; dof++ {
		d.FixDOF(nodeID, dof)
	}
}

// ApplyLoad adds a nodal force.
func (d *Domain) ApplyLoad(nodeID, dof int, value float64) {
	d.Loads = append(d.Loads, NodalLoad{NodeID: nodeID, DOF: dof, Value: value})
}

// NumDOF returns the total number of DOFs (3 per node).
func (d *Domain) NumDOF() int {
	return len(d.Nodes) * 3
}

// Assemble constructs the global K and F.
func (d *Domain) Assemble() {
	ndof := d.NumDOF()
	d.K = mat.NewDense(ndof, ndof, nil)
	d.F = mat.NewVecDense(ndof, nil)

	// --- Scatter element stiffness into global K ---
	for _, elem := range d.Elements {
		ke := elem.GetTangentStiffness()
		nids := elem.NodeIDs()
		eldof := elem.NumDOF()

		// Build local → global DOF map.
		dofs := make([]int, eldof)
		for i, nid := range nids {
			dofs[3*i+0] = 3*nid + 0
			dofs[3*i+1] = 3*nid + 1
			dofs[3*i+2] = 3*nid + 2
		}

		for i := 0; i < eldof; i++ {
			gi := dofs[i]
			for j := 0; j < eldof; j++ {
				gj := dofs[j]
				d.K.Set(gi, gj, d.K.At(gi, gj)+ke.At(i, j))
			}
		}
	}

	// --- Assemble load vector ---
	for _, load := range d.Loads {
		gdof := 3*load.NodeID + load.DOF
		d.F.SetVec(gdof, d.F.AtVec(gdof)+load.Value)
	}
}

// ApplyDirichletBC modifies K and F for prescribed displacements.
// Uses the row/column zeroing technique: for each fixed DOF i,
//
//	K[i,:] = 0,  K[:,i] = 0,  K[i,i] = 1,  F[i] = prescribed value.
func (d *Domain) ApplyDirichletBC() {
	ndof := d.NumDOF()
	for _, bc := range d.BCs {
		gdof := 3*bc.NodeID + bc.DOF
		for j := 0; j < ndof; j++ {
			d.K.Set(gdof, j, 0)
			d.K.Set(j, gdof, 0)
		}
		d.K.Set(gdof, gdof, 1)
		d.F.SetVec(gdof, bc.Value)
	}
}

// SetDisplacements stores the computed displacement for each node.
// Returns a slice indexed by node ID with [ux, uy, uz].
func (d *Domain) SetDisplacements(U *mat.VecDense) [][3]float64 {
	disp := make([][3]float64, len(d.Nodes))
	for i := range d.Nodes {
		disp[i][0] = U.AtVec(3*i + 0)
		disp[i][1] = U.AtVec(3*i + 1)
		disp[i][2] = U.AtVec(3*i + 2)
	}
	return disp
}
