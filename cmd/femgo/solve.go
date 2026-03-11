package main

import (
	"fmt"
	"math"
	"time"

	"go-fem/analysis"
	"go-fem/domain"
	"go-fem/element"
	"go-fem/element/frame"
	"go-fem/element/quad"
	"go-fem/element/shell"
	"go-fem/element/solid"
	"go-fem/element/truss"
	"go-fem/element/zerolength"
	"go-fem/material"
	"go-fem/section"
	"go-fem/solver"
)

var dofNames = [6]string{"ux", "uy", "uz", "rx", "ry", "rz"}

func solveProblem(input ProblemInput) ProblemOutput {
	t0 := time.Now()

	// --- Build materials map ---
	mats := make(map[string]material.Material3D, len(input.Materials))
	for _, mi := range input.Materials {
		switch mi.Type {
		case "isotropic_linear":
			mats[mi.ID] = material.NewIsotropicLinear(mi.E, mi.Nu)
		case "orthotropic_linear":
			m, err := material.NewOrthotropicLinear(
				mi.Ex, mi.Ey, mi.Ez,
				mi.Nxy, mi.Nyz, mi.Nxz,
				mi.Gxy, mi.Gyz, mi.Gxz,
			)
			if err != nil {
				return errorResponse("material %q: %v", mi.ID, err)
			}
			mats[mi.ID] = m
		default:
			return errorResponse("unknown material type: %s", mi.Type)
		}
	}

	// --- Build domain ---
	dom := domain.NewDomain()

	for _, n := range input.Nodes {
		dom.AddNode(n[0], n[1], n[2])
	}

	for eid, ei := range input.Elements {
		elem, err := createElement(eid, ei, dom, mats)
		if err != nil {
			return errorResponse("element %d: %v", eid, err)
		}
		dom.AddElement(elem)
	}

	// --- Boundary conditions ---
	for _, bc := range input.BoundaryConditions {
		if bc.Node < 0 || bc.Node >= len(dom.Nodes) {
			return errorResponse("BC: node %d out of range", bc.Node)
		}
		for i, dofIdx := range bc.DOFs {
			if dofIdx < 0 || dofIdx > 5 {
				return errorResponse("BC: invalid dof %d (must be 0–5)", dofIdx)
			}
			val := 0.0
			if i < len(bc.Values) {
				val = bc.Values[i]
			}
			dom.BCs = append(dom.BCs, domain.BC{NodeID: bc.Node, DOF: dofIdx, Value: val})
		}
	}

	// --- Loads ---
	for i, ld := range input.Loads {
		switch ld.Type {
		case "", "nodal":
			if ld.Node < 0 || ld.Node >= len(dom.Nodes) {
				return errorResponse("load[%d]: node %d out of range", i, ld.Node)
			}
			if ld.DOF < 0 || ld.DOF > 5 {
				return errorResponse("load[%d]: invalid dof %d", i, ld.DOF)
			}
			dom.ApplyLoad(ld.Node, ld.DOF, ld.Value)

		case "surface_pressure":
			for _, nid := range ld.FaceNodes {
				if nid < 0 || nid >= len(dom.Nodes) {
					return errorResponse("load[%d] surface_pressure: node %d out of range", i, nid)
				}
			}
			dom.AddSurfacePressure(ld.FaceNodes, ld.Pressure)

		case "beam_dist":
			if ld.Element < 0 || ld.Element >= len(dom.Elements) {
				return errorResponse("load[%d] beam_dist: element %d out of range", i, ld.Element)
			}
			dom.AddBeamDistLoad(ld.Element, ld.Dir, ld.Intensity)

		case "body_force":
			if ld.Element < 0 || ld.Element >= len(dom.Elements) {
				return errorResponse("load[%d] body_force: element %d out of range", i, ld.Element)
			}
			dom.AddBodyForce(ld.Element, ld.Rho, ld.G)

		default:
			return errorResponse("load[%d]: unknown load type %q", i, ld.Type)
		}
	}

	// --- Pick solver ---
	solverName := "cholesky"
	var slv solver.LinearSolver
	switch input.Solver {
	case "", "cholesky":
		slv = solver.Cholesky{}
	case "lu":
		slv = solver.LU{}
		solverName = "lu"
	default:
		return errorResponse("unknown solver: %s", input.Solver)
	}

	// --- Solve ---
	ana := analysis.StaticLinearAnalysis{Dom: dom, Solver: slv}
	U, err := ana.Run()
	if err != nil {
		return errorResponse("analysis failed: %v", err)
	}

	disp := dom.SetDisplacements(U)
	dpn := dom.DOFPerNode

	// --- Build displacement output ---
	disps := make([]DisplacementOutput, len(disp))
	var maxVal float64
	var maxNode int
	var maxComp string
	for i, d := range disp {
		disps[i] = DisplacementOutput{
			Node: i,
			Ux:   d[0], Uy: d[1], Uz: d[2],
		}
		if dpn > 3 {
			disps[i].Rx = d[3]
			disps[i].Ry = d[4]
			disps[i].Rz = d[5]
		}
		for c := 0; c < dpn; c++ {
			if math.Abs(d[c]) > math.Abs(maxVal) {
				maxVal = d[c]
				maxNode = i
				maxComp = dofNames[c]
			}
		}
	}

	elapsed := time.Since(t0).Seconds() * 1000

	return ProblemOutput{
		Success: true,
		Info: &InfoOutput{
			NumNodes:    len(dom.Nodes),
			NumElements: len(dom.Elements),
			NumDOFs:     dom.NumDOF(),
			DOFPerNode:  dpn,
			Solver:      solverName,
		},
		Displacements: disps,
		ElementForces: extractElementForces(dom.Elements, input.Elements),
		Summary: &SummaryOutput{
			MaxAbsDisplacement: MaxDispOutput{
				Node:      maxNode,
				Component: maxComp,
				Value:     maxVal,
			},
		},
		ElapsedMs: elapsed,
	}
}

// createElement builds the appropriate element from JSON input.
func createElement(eid int, ei ElementInput, dom *domain.Domain, mats map[string]material.Material3D) (element.Element, error) {
	nn := len(dom.Nodes)

	// Helper: validate and extract 3D node coordinates
	getCoords3 := func(nids []int, expected int) ([]int, [][3]float64, error) {
		if len(nids) != expected {
			return nil, nil, fmt.Errorf("%s requires %d nodes, got %d", ei.Type, expected, len(nids))
		}
		coords := make([][3]float64, expected)
		for i, nid := range nids {
			if nid < 0 || nid >= nn {
				return nil, nil, fmt.Errorf("node %d out of range", nid)
			}
			coords[i] = dom.Nodes[nid].Coord
		}
		return nids, coords, nil
	}

	switch ei.Type {
	case "tet4":
		nids, coords, err := getCoords3(ei.Nodes, 4)
		if err != nil {
			return nil, err
		}
		mat3d, ok := mats[ei.Material]
		if !ok {
			return nil, fmt.Errorf("unknown material: %s", ei.Material)
		}
		var n4 [4]int
		var c4 [4][3]float64
		copy(n4[:], nids)
		copy(c4[:], coords)
		return solid.NewTet4(eid, n4, c4, mat3d), nil

	case "hexa8":
		nids, coords, err := getCoords3(ei.Nodes, 8)
		if err != nil {
			return nil, err
		}
		mat3d, ok := mats[ei.Material]
		if !ok {
			return nil, fmt.Errorf("unknown material: %s", ei.Material)
		}
		var n8 [8]int
		var c8 [8][3]float64
		copy(n8[:], nids)
		copy(c8[:], coords)
		return solid.NewHexa8(eid, n8, c8, mat3d), nil

	case "tet10":
		nids, coords, err := getCoords3(ei.Nodes, 10)
		if err != nil {
			return nil, err
		}
		mat3d, ok := mats[ei.Material]
		if !ok {
			return nil, fmt.Errorf("unknown material: %s", ei.Material)
		}
		var n10 [10]int
		var c10 [10][3]float64
		copy(n10[:], nids)
		copy(c10[:], coords)
		return solid.NewTet10(eid, n10, c10, mat3d), nil

	case "brick20":
		nids, coords, err := getCoords3(ei.Nodes, 20)
		if err != nil {
			return nil, err
		}
		mat3d, ok := mats[ei.Material]
		if !ok {
			return nil, fmt.Errorf("unknown material: %s", ei.Material)
		}
		var n20 [20]int
		var c20 [20][3]float64
		copy(n20[:], nids)
		copy(c20[:], coords)
		return solid.NewBrick20(eid, n20, c20, mat3d), nil

	case "truss3d":
		nids, coords, err := getCoords3(ei.Nodes, 2)
		if err != nil {
			return nil, err
		}
		if ei.E <= 0 || ei.A <= 0 {
			return nil, fmt.Errorf("truss3d requires E > 0 and A > 0")
		}
		var n2 [2]int
		var c2 [2][3]float64
		copy(n2[:], nids)
		copy(c2[:], coords)
		return truss.NewTruss3D(eid, n2, c2, ei.E, ei.A), nil

	case "corot_truss":
		nids, coords, err := getCoords3(ei.Nodes, 2)
		if err != nil {
			return nil, err
		}
		if ei.E <= 0 || ei.A <= 0 {
			return nil, fmt.Errorf("corot_truss requires E > 0 and A > 0")
		}
		var n2 [2]int
		var c2 [2][3]float64
		copy(n2[:], nids)
		copy(c2[:], coords)
		return truss.NewCorotTruss(eid, n2, c2, ei.E, ei.A), nil

	case "elastic_beam3d":
		nids, coords, err := getCoords3(ei.Nodes, 2)
		if err != nil {
			return nil, err
		}
		if ei.E <= 0 || ei.G <= 0 || ei.A <= 0 {
			return nil, fmt.Errorf("elastic_beam3d requires E, G, A > 0")
		}
		var n2 [2]int
		var c2 [2][3]float64
		copy(n2[:], nids)
		copy(c2[:], coords)
		sec := section.BeamSection3D{A: ei.A, Iy: ei.Iy, Iz: ei.Iz, J: ei.J}
		return frame.NewElasticBeam3D(eid, n2, c2, ei.E, ei.G, sec, ei.VecXZ), nil

	case "timoshenko_beam3d":
		nids, coords, err := getCoords3(ei.Nodes, 2)
		if err != nil {
			return nil, err
		}
		if ei.E <= 0 || ei.G <= 0 || ei.A <= 0 {
			return nil, fmt.Errorf("timoshenko_beam3d requires E, G, A > 0")
		}
		var n2 [2]int
		var c2 [2][3]float64
		copy(n2[:], nids)
		copy(c2[:], coords)
		sec := section.BeamSection3D{A: ei.A, Iy: ei.Iy, Iz: ei.Iz, J: ei.J, Asy: ei.Asy, Asz: ei.Asz}
		return frame.NewTimoshenkoBeam3D(eid, n2, c2, ei.E, ei.G, sec, ei.VecXZ), nil

	case "shell_mitc4":
		nids, coords, err := getCoords3(ei.Nodes, 4)
		if err != nil {
			return nil, err
		}
		if ei.E <= 0 || ei.Thickness <= 0 {
			return nil, fmt.Errorf("shell_mitc4 requires E > 0 and thickness > 0")
		}
		var n4 [4]int
		var c4 [4][3]float64
		copy(n4[:], nids)
		copy(c4[:], coords)
		return shell.NewShellMITC4(eid, n4, c4, ei.E, ei.Nu, ei.Thickness), nil

	case "dkt3", "discrete_kirchhoff_triangle":
		nids, coords, err := getCoords3(ei.Nodes, 3)
		if err != nil {
			return nil, err
		}
		if ei.E <= 0 || ei.Thickness <= 0 {
			return nil, fmt.Errorf("dkt3 requires E > 0 and thickness > 0")
		}
		var n3 [3]int
		var c3 [3][3]float64
		copy(n3[:], nids)
		copy(c3[:], coords)
		return shell.NewDiscreteKirchhoffTriangle(eid, n3, c3, ei.E, ei.Nu, ei.Thickness), nil

	case "quad4":
		if len(ei.Nodes) != 4 {
			return nil, fmt.Errorf("quad4 requires 4 nodes, got %d", len(ei.Nodes))
		}
		for _, nid := range ei.Nodes {
			if nid < 0 || nid >= nn {
				return nil, fmt.Errorf("node %d out of range", nid)
			}
		}
		if ei.E <= 0 || ei.Thickness <= 0 {
			return nil, fmt.Errorf("quad4 requires E > 0 and thickness > 0")
		}
		var n4q [4]int
		var c4q [4][2]float64
		copy(n4q[:], ei.Nodes)
		for k, nid := range ei.Nodes {
			c4q[k][0] = dom.Nodes[nid].Coord[0]
			c4q[k][1] = dom.Nodes[nid].Coord[1]
		}
		ptype := quad.PlaneStress
		if ei.PlaneType == "strain" {
			ptype = quad.PlaneStrain
		}
		return quad.NewQuad4(eid, n4q, c4q, ei.E, ei.Nu, ei.Thickness, ptype), nil

	case "quad8":
		if len(ei.Nodes) != 8 {
			return nil, fmt.Errorf("quad8 requires 8 nodes, got %d", len(ei.Nodes))
		}
		for _, nid := range ei.Nodes {
			if nid < 0 || nid >= nn {
				return nil, fmt.Errorf("node %d out of range", nid)
			}
		}
		if ei.E <= 0 || ei.Thickness <= 0 {
			return nil, fmt.Errorf("quad8 requires E > 0 and thickness > 0")
		}
		var n8q [8]int
		var c8q [8][2]float64
		copy(n8q[:], ei.Nodes)
		for k, nid := range ei.Nodes {
			c8q[k][0] = dom.Nodes[nid].Coord[0]
			c8q[k][1] = dom.Nodes[nid].Coord[1]
		}
		ptype8 := quad.PlaneStress
		if ei.PlaneType == "strain" {
			ptype8 = quad.PlaneStrain
		}
		return quad.NewQuad8(eid, n8q, c8q, ei.E, ei.Nu, ei.Thickness, ptype8), nil

	case "zerolength":
		if len(ei.Nodes) != 2 {
			return nil, fmt.Errorf("zerolength requires 2 nodes")
		}
		for _, nid := range ei.Nodes {
			if nid < 0 || nid >= nn {
				return nil, fmt.Errorf("node %d out of range", nid)
			}
		}
		var n2 [2]int
		copy(n2[:], ei.Nodes)
		return zerolength.NewZeroLength(eid, n2, ei.Springs), nil

	default:
		return nil, fmt.Errorf("unknown element type: %s", ei.Type)
	}
}

func errorResponse(format string, args ...any) ProblemOutput {
	return ProblemOutput{
		Success: false,
		Error:   fmt.Sprintf(format, args...),
	}
}

// extractElementForces builds the per-element post-processing results.
// It uses type assertions to call element-specific methods.
func extractElementForces(elems []element.Element, inputs []ElementInput) []ElementForcesOutput {
	out := make([]ElementForcesOutput, len(elems))
	for i, elem := range elems {
		ef := ElementForcesOutput{ID: i}
		if i < len(inputs) {
			ef.Type = inputs[i].Type
		}
		switch e := elem.(type) {
		case *truss.Truss3D:
			N := e.AxialForce()
			sigma := e.AxialStress()
			ef.N = &N
			ef.Sigma = &sigma
		case *truss.CorotTruss:
			N := e.AxialForce()
			sigma := e.AxialStress()
			ef.N = &N
			ef.Sigma = &sigma
		case *frame.ElasticBeam3D:
			ef2 := e.EndForces()
			ef.EndI = &BeamEndOutput{N: ef2.I[0], Vy: ef2.I[1], Vz: ef2.I[2], Mx: ef2.I[3], My: ef2.I[4], Mz: ef2.I[5]}
			ef.EndJ = &BeamEndOutput{N: ef2.J[0], Vy: ef2.J[1], Vz: ef2.J[2], Mx: ef2.J[3], My: ef2.J[4], Mz: ef2.J[5]}
		case *frame.TimoshenkoBeam3D:
			ef2 := e.EndForces()
			ef.EndI = &BeamEndOutput{N: ef2.I[0], Vy: ef2.I[1], Vz: ef2.I[2], Mx: ef2.I[3], My: ef2.I[4], Mz: ef2.I[5]}
			ef.EndJ = &BeamEndOutput{N: ef2.J[0], Vy: ef2.J[1], Vz: ef2.J[2], Mx: ef2.J[3], My: ef2.J[4], Mz: ef2.J[5]}
		case *solid.Tet4:
			s := e.StressCentroid()
			ef.Stress = &StressOutput{Sxx: s[0], Syy: s[1], Szz: s[2], Txy: s[3], Tyz: s[4], Txz: s[5], VonMises: solid.VonMises(s)}
		case *solid.Hexa8:
			s := e.StressCentroid()
			ef.Stress = &StressOutput{Sxx: s[0], Syy: s[1], Szz: s[2], Txy: s[3], Tyz: s[4], Txz: s[5], VonMises: solid.VonMises(s)}
		case *solid.Tet10:
			s := e.StressCentroid()
			ef.Stress = &StressOutput{Sxx: s[0], Syy: s[1], Szz: s[2], Txy: s[3], Tyz: s[4], Txz: s[5], VonMises: solid.VonMises(s)}
		case *solid.Brick20:
			s := e.StressCentroid()
			ef.Stress = &StressOutput{Sxx: s[0], Syy: s[1], Szz: s[2], Txy: s[3], Tyz: s[4], Txz: s[5], VonMises: solid.VonMises(s)}
		case *quad.Quad4:
			s := e.StressCentroid()
			vm := solid.VonMises([6]float64{s[0], s[1], 0, s[2], 0, 0})
			ef.Stress = &StressOutput{Sxx: s[0], Syy: s[1], Txy: s[2], VonMises: vm}
		case *quad.Quad8:
			s := e.StressCentroid()
			vm := solid.VonMises([6]float64{s[0], s[1], 0, s[2], 0, 0})
			ef.Stress = &StressOutput{Sxx: s[0], Syy: s[1], Txy: s[2], VonMises: vm}
		case *shell.ShellMITC4:
			sf := e.LocalForces()
			ef.ShellForces = &ShellForcesOutput{Nx: sf.Nx, Ny: sf.Ny, Nxy: sf.Nxy, Mx: sf.Mx, My: sf.My, Mxy: sf.Mxy}
		case *shell.DiscreteKirchhoffTriangle:
			mx, my, mxy := e.LocalMoments()
			ef.ShellForces = &ShellForcesOutput{Mx: mx, My: my, Mxy: mxy}
		}
		out[i] = ef
	}
	return out
}
