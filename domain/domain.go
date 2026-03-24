// Package domain manages the FEM mesh, assembly, and boundary conditions (Layer 3).
// Inspired by the OpenSees Domain architecture.
package domain

import (
	"math"

	"go-fem/dof"
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

// SurfacePressure applies a uniform pressure on a 4-node (quad) face.
// FaceNodes must be ordered counter-clockwise when viewed from outside
// so that the natural outward normal equals g_ξ × g_η.
// P > 0 is pressure acting inward (compressive); the equivalent nodal
// forces are in the direction −normal.
type SurfacePressure struct {
	FaceNodes [4]int  // global node IDs
	P         float64 // pressure magnitude
}

// BeamLoad represents a uniformly distributed load on a single beam element.
type BeamLoad struct {
	ElemIdx   int        // index into Domain.Elements
	Dir       [3]float64 // global load direction (should be unit vector)
	Intensity float64    // load per unit length (N/m)
}

// BeamLinearLoad represents a linearly varying (trapezoidal / triangular)
// distributed load on a single beam element.
// IntensityI is the load per unit length at node i; IntensityJ at node j.
type BeamLinearLoad struct {
	ElemIdx    int
	Dir        [3]float64
	IntensityI float64
	IntensityJ float64
}

// BodyForce represents a gravitational / body-force load on a single element.
type BodyForce struct {
	ElemIdx int
	Rho     float64    // mass density (kg/m³ or consistent units)
	G       [3]float64 // acceleration vector (e.g. [0, -9.81, 0])
}

// Domain holds the complete FEM model.
type Domain struct {
	Nodes    []Node3D
	Elements []element.Element
	BCs      []BC
	Loads    []NodalLoad

	// Distributed load types
	SurfaceLoads    []SurfacePressure
	BeamLoads       []BeamLoad
	BeamLinearLoads []BeamLinearLoad
	BodyForces      []BodyForce

	// DOFPerNode is auto-detected during Assemble:
	// 2 for pure plane stress/strain, 3 for solids/truss or plane frames,
	// 6 when 3D beams/shells are present.
	DOFPerNode int

	// dofOffset maps each dof.Type enum value (0–5) to a consecutive offset
	// within the node's DOF block. A value of -1 means the DOF type is
	// inactive. Built during Assemble.
	dofOffset [6]int
	// reverseDOF maps consecutive offset back to dof.Type enum value.
	reverseDOF []int

	K *mat.Dense    // global stiffness (assembled)
	F *mat.VecDense // global force vector
	M *mat.Dense    // global mass matrix (assembled by AssembleMassMatrix)
}

// ElementMass pairs an element index with its mass density for mass assembly.
type ElementMass struct {
	ElemIdx int
	Rho     float64 // mass density (kg/m³ or consistent units)
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

// AddSurfacePressure registers a pressure on a 4-node face.
func (d *Domain) AddSurfacePressure(faceNodes [4]int, p float64) {
	d.SurfaceLoads = append(d.SurfaceLoads, SurfacePressure{FaceNodes: faceNodes, P: p})
}

// AddBeamDistLoad registers a uniformly distributed load on element elemIdx.
func (d *Domain) AddBeamDistLoad(elemIdx int, dir [3]float64, intensity float64) {
	d.BeamLoads = append(d.BeamLoads, BeamLoad{ElemIdx: elemIdx, Dir: dir, Intensity: intensity})
}

// AddBeamLinearLoad registers a linearly varying distributed load on element elemIdx.
// intensityI is the load per unit length at node i; intensityJ at node j.
func (d *Domain) AddBeamLinearLoad(elemIdx int, dir [3]float64, intensityI, intensityJ float64) {
	d.BeamLinearLoads = append(d.BeamLinearLoads, BeamLinearLoad{ElemIdx: elemIdx, Dir: dir, IntensityI: intensityI, IntensityJ: intensityJ})
}

// AddBodyForce registers a body-force load on element elemIdx.
func (d *Domain) AddBodyForce(elemIdx int, rho float64, g [3]float64) {
	d.BodyForces = append(d.BodyForces, BodyForce{ElemIdx: elemIdx, Rho: rho, G: g})
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
// Auto-detects DOFPerNode from the active DOF types across all elements.
func (d *Domain) Assemble() {
	// Collect active DOF types across all elements.
	var active [6]bool
	for _, elem := range d.Elements {
		for _, dt := range elem.DOFTypes() {
			active[int(dt)] = true
		}
	}

	// Build consecutive mapping: dof.Type enum → offset 0..n-1
	for i := range d.dofOffset {
		d.dofOffset[i] = -1
	}
	idx := 0
	d.reverseDOF = nil
	for i := 0; i < 6; i++ {
		if active[i] {
			d.dofOffset[i] = idx
			d.reverseDOF = append(d.reverseDOF, i)
			idx++
		}
	}
	d.DOFPerNode = idx
	if d.DOFPerNode == 0 {
		d.DOFPerNode = 3
		d.dofOffset = [6]int{0, 1, 2, -1, -1, -1}
		d.reverseDOF = []int{0, 1, 2}
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

		// Build local → global DOF map using DOF type offsets
		globalDOFs := make([]int, eldof)
		for i, dt := range dofTypes {
			nodeIdx := i / elemDPN
			nodeID := nids[nodeIdx]
			globalDOFs[i] = nodeID*dpn + d.dofOffset[int(dt)]
		}

		for i := 0; i < eldof; i++ {
			gi := globalDOFs[i]
			for j := 0; j < eldof; j++ {
				gj := globalDOFs[j]
				d.K.Set(gi, gj, d.K.At(gi, gj)+ke.At(i, j))
			}
		}
	}

	// --- Nodal loads ---
	for _, load := range d.Loads {
		off := d.dofOffset[load.DOF]
		if off < 0 {
			continue
		}
		gdof := load.NodeID*dpn + off
		d.F.SetVec(gdof, d.F.AtVec(gdof)+load.Value)
	}

	// --- Surface pressure loads (4-node quad faces, 2×2 Gauss) ---
	gp := 1.0 / math.Sqrt(3.0)
	gpPts := [2]float64{-gp, gp}
	for _, sp := range d.SurfaceLoads {
		var xf [4][3]float64
		for i, nid := range sp.FaceNodes {
			xf[i] = d.Nodes[nid].Coord
		}
		for _, s := range gpPts {
			for _, t := range gpPts {
				N := [4]float64{
					(1 - s) * (1 - t) / 4,
					(1 + s) * (1 - t) / 4,
					(1 + s) * (1 + t) / 4,
					(1 - s) * (1 + t) / 4,
				}
				dNds := [4]float64{-(1 - t) / 4, (1 - t) / 4, (1 + t) / 4, -(1 + t) / 4}
				dNdt := [4]float64{-(1 - s) / 4, -(1 + s) / 4, (1 + s) / 4, (1 - s) / 4}

				var gs, gt [3]float64
				for i := 0; i < 4; i++ {
					gs[0] += dNds[i] * xf[i][0]
					gs[1] += dNds[i] * xf[i][1]
					gs[2] += dNds[i] * xf[i][2]
					gt[0] += dNdt[i] * xf[i][0]
					gt[1] += dNdt[i] * xf[i][1]
					gt[2] += dNdt[i] * xf[i][2]
				}
				nx := gs[1]*gt[2] - gs[2]*gt[1]
				ny := gs[2]*gt[0] - gs[0]*gt[2]
				nz := gs[0]*gt[1] - gs[1]*gt[0]

				for i, nid := range sp.FaceNodes {
					d.F.SetVec(nid*dpn+d.dofOffset[int(dof.UX)], d.F.AtVec(nid*dpn+d.dofOffset[int(dof.UX)])-sp.P*N[i]*nx)
					d.F.SetVec(nid*dpn+d.dofOffset[int(dof.UY)], d.F.AtVec(nid*dpn+d.dofOffset[int(dof.UY)])-sp.P*N[i]*ny)
					d.F.SetVec(nid*dpn+d.dofOffset[int(dof.UZ)], d.F.AtVec(nid*dpn+d.dofOffset[int(dof.UZ)])-sp.P*N[i]*nz)
				}
			}
		}
	}

	// --- Beam distributed loads ---
	for _, bl := range d.BeamLoads {
		if bl.ElemIdx < 0 || bl.ElemIdx >= len(d.Elements) {
			continue
		}
		elem := d.Elements[bl.ElemIdx]
		loader, ok := elem.(element.EquivalentNodalLoader)
		if !ok {
			continue
		}
		fe := loader.EquivalentNodalLoad(bl.Dir, bl.Intensity)
		nids := elem.NodeIDs()
		dofTypes := elem.DOFTypes()
		elemDPN := elem.DOFPerNode()
		for i, dt := range dofTypes {
			nodeIdx := i / elemDPN
			nodeID := nids[nodeIdx]
			gdof := nodeID*dpn + d.dofOffset[int(dt)]
			d.F.SetVec(gdof, d.F.AtVec(gdof)+fe.AtVec(i))
		}
	}

	// --- Linearly varying (trapezoidal/triangular) beam loads ---
	for _, bl := range d.BeamLinearLoads {
		if bl.ElemIdx < 0 || bl.ElemIdx >= len(d.Elements) {
			continue
		}
		elem := d.Elements[bl.ElemIdx]
		loader, ok := elem.(element.LinearDistLoader)
		if !ok {
			continue
		}
		fe := loader.EquivalentNodalLoadLinear(bl.Dir, bl.IntensityI, bl.IntensityJ)
		nids := elem.NodeIDs()
		dofTypes := elem.DOFTypes()
		elemDPN := elem.DOFPerNode()
		for i, dt := range dofTypes {
			nodeIdx := i / elemDPN
			nodeID := nids[nodeIdx]
			gdof := nodeID*dpn + d.dofOffset[int(dt)]
			d.F.SetVec(gdof, d.F.AtVec(gdof)+fe.AtVec(i))
		}
	}

	// --- Body forces ---
	for _, bf := range d.BodyForces {
		if bf.ElemIdx < 0 || bf.ElemIdx >= len(d.Elements) {
			continue
		}
		elem := d.Elements[bf.ElemIdx]
		loader, ok := elem.(element.BodyForceLoader)
		if !ok {
			continue
		}
		fe := loader.BodyForceLoad(bf.G, bf.Rho)
		nids := elem.NodeIDs()
		dofTypes := elem.DOFTypes()
		elemDPN := elem.DOFPerNode()
		for i, dt := range dofTypes {
			nodeIdx := i / elemDPN
			nodeID := nids[nodeIdx]
			gdof := nodeID*dpn + d.dofOffset[int(dt)]
			d.F.SetVec(gdof, d.F.AtVec(gdof)+fe.AtVec(i))
		}
	}
}

// ApplyDirichletBC modifies K and F for prescribed displacements.
func (d *Domain) ApplyDirichletBC() {
	dpn := d.DOFPerNode
	ndof := len(d.Nodes) * dpn
	for _, bc := range d.BCs {
		off := d.dofOffset[bc.DOF]
		if off < 0 {
			continue
		}
		gdof := bc.NodeID*dpn + off
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
			enumIdx := d.reverseDOF[j]
			disp[i][enumIdx] = U.AtVec(i*dpn + j)
		}
	}
	return disp
}

// DOFTypeAt returns the DOF type (dof.UX … dof.RZ) for a global DOF index.
// Must be called after Assemble() has set DOFPerNode and reverseDOF.
func (d *Domain) DOFTypeAt(globalDOF int) dof.Type {
	return dof.Type(d.reverseDOF[globalDOF%d.DOFPerNode])
}

// FreeDOFs returns the list of global DOF indices that are not constrained by
// any Dirichlet BC. The slice is sorted in ascending order.
// Must be called after Assemble().
func (d *Domain) FreeDOFs() []int {
	dpn := d.DOFPerNode
	ndof := len(d.Nodes) * dpn
	constrained := make([]bool, ndof)
	for _, bc := range d.BCs {
		off := d.dofOffset[bc.DOF]
		if off < 0 {
			continue
		}
		gdof := bc.NodeID*dpn + off
		if gdof < ndof {
			constrained[gdof] = true
		}
	}
	free := make([]int, 0, ndof)
	for i := 0; i < ndof; i++ {
		if !constrained[i] {
			free = append(free, i)
		}
	}
	return free
}

// DOFOffsetOf returns the within-node offset of a DOF type index (0–5).
// Returns -1 if that DOF type is not active in the current problem.
// Must be called after Assemble().
func (d *Domain) DOFOffsetOf(dofType int) int {
	if dofType < 0 || dofType > 5 {
		return -1
	}
	return d.dofOffset[dofType]
}

// AssembleMassMatrix constructs the global consistent mass matrix M.
// Must be called after Assemble() (which sets DOFPerNode and dofOffset).
// Elements that do not implement element.MassMatrixAssembler are skipped.
func (d *Domain) AssembleMassMatrix(masses []ElementMass) {
	dpn := d.DOFPerNode
	ndof := len(d.Nodes) * dpn
	d.M = mat.NewDense(ndof, ndof, nil)

	for _, em := range masses {
		if em.ElemIdx < 0 || em.ElemIdx >= len(d.Elements) {
			continue
		}
		elem := d.Elements[em.ElemIdx]
		assembler, ok := elem.(element.MassMatrixAssembler)
		if !ok {
			continue
		}
		me := assembler.GetMassMatrix(em.Rho)
		nids := elem.NodeIDs()
		dofTypes := elem.DOFTypes()
		eldof := elem.NumDOF()
		elemDPN := elem.DOFPerNode()

		globalDOFs := make([]int, eldof)
		for i, dt := range dofTypes {
			nodeIdx := i / elemDPN
			nodeID := nids[nodeIdx]
			globalDOFs[i] = nodeID*dpn + d.dofOffset[int(dt)]
		}

		for i := 0; i < eldof; i++ {
			gi := globalDOFs[i]
			for j := 0; j < eldof; j++ {
				gj := globalDOFs[j]
				d.M.Set(gi, gj, d.M.At(gi, gj)+me.At(i, j))
			}
		}
	}
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
		disp[i] = U.AtVec(nodeID*dpn + d.dofOffset[int(dt)])
	}
	return disp
}

// DOFOffset returns the consecutive offset (0-based within the node's DOF block)
// for the given DOF type enum value (0=UX..5=RZ).
// Returns -1 if the DOF type is inactive in this model.
// Must be called after Assemble() has been invoked.
func (d *Domain) DOFOffset(dofEnum int) int {
	if dofEnum < 0 || dofEnum >= 6 {
		return -1
	}
	return d.dofOffset[dofEnum]
}

// AssembleTangent reassembles the global tangent stiffness matrix K_T from the
// current element tangent stiffnesses (after elements have been Updated).
// Must be called after Assemble() has set DOFPerNode and dofOffset.
// Loads and BCs are NOT applied; call ApplyBCsIncremental to enforce constraints.
func (d *Domain) AssembleTangent() *mat.Dense {
	dpn := d.DOFPerNode
	ndof := len(d.Nodes) * dpn
	K := mat.NewDense(ndof, ndof, nil)
	for _, elem := range d.Elements {
		ke := elem.GetTangentStiffness()
		nids := elem.NodeIDs()
		dofTypes := elem.DOFTypes()
		eldof := elem.NumDOF()
		elemDPN := elem.DOFPerNode()
		gDOFs := make([]int, eldof)
		for i, dt := range dofTypes {
			nodeIdx := i / elemDPN
			gDOFs[i] = nids[nodeIdx]*dpn + d.dofOffset[int(dt)]
		}
		for i := 0; i < eldof; i++ {
			for j := 0; j < eldof; j++ {
				K.Set(gDOFs[i], gDOFs[j], K.At(gDOFs[i], gDOFs[j])+ke.At(i, j))
			}
		}
	}
	return K
}

// AssembleResisting assembles the global internal resisting force vector R_int
// from element resisting forces. Must be called after elements have been Updated.
func (d *Domain) AssembleResisting() *mat.VecDense {
	dpn := d.DOFPerNode
	ndof := len(d.Nodes) * dpn
	Rint := mat.NewVecDense(ndof, nil)
	for _, elem := range d.Elements {
		fe := elem.GetResistingForce()
		nids := elem.NodeIDs()
		dofTypes := elem.DOFTypes()
		elemDPN := elem.DOFPerNode()
		for i, dt := range dofTypes {
			nodeIdx := i / elemDPN
			gdof := nids[nodeIdx]*dpn + d.dofOffset[int(dt)]
			Rint.SetVec(gdof, Rint.AtVec(gdof)+fe.AtVec(i))
		}
	}
	return Rint
}

// ApplyBCsIncremental modifies K_T and R for the Newton-Raphson incremental
// system K_T · ΔU = R:
//   - For each constrained DOF gdof: K_T[gdof,*] = 0, K_T[*,gdof] = 0,
//     K_T[gdof,gdof] = 1, R[gdof] = 0
//
// This enforces ΔU[gdof] = 0 so that prescribed displacements do not change
// between NR iterations.
func (d *Domain) ApplyBCsIncremental(K *mat.Dense, R *mat.VecDense) {
	dpn := d.DOFPerNode
	ndof := len(d.Nodes) * dpn
	for _, bc := range d.BCs {
		off := d.dofOffset[bc.DOF]
		if off < 0 {
			continue
		}
		gdof := bc.NodeID*dpn + off
		if gdof >= ndof {
			continue
		}
		for j := 0; j < ndof; j++ {
			K.Set(gdof, j, 0)
			K.Set(j, gdof, 0)
		}
		K.Set(gdof, gdof, 1)
		R.SetVec(gdof, 0)
	}
}
