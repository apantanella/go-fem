package main

import (
	"fmt"
	"math"
	"strings"
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

// elementDimension maps every accepted element type name (canonical and old aliases)
// to its spatial dimension ("2D" or "3D").
var elementDimension = map[string]string{
	// ── 3D elements ──────────────────────────────────────────────────────────
	"tet4_3d": "3D", "tet4": "3D",
	"hexa8_3d": "3D", "hexa8": "3D",
	"tet10_3d": "3D", "tet10": "3D",
	"brick20_3d": "3D", "brick20": "3D",
	"truss_3d": "3D", "truss3d": "3D",
	"corot_truss_3d": "3D", "corot_truss": "3D",
	"elastic_beam_3d": "3D", "elastic_beam3d": "3D",
	"timoshenko_beam_3d": "3D", "timoshenko_beam3d": "3D",
	"shell_mitc4_3d": "3D", "shell_mitc4": "3D",
	"dkt3_3d": "3D", "dkt3": "3D", "discrete_kirchhoff_triangle": "3D",
	"zerolength_3d": "3D", "zerolength": "3D",
	"zerolength_trans_3d": "3D",
	// ── 2D elements ──────────────────────────────────────────────────────────
	"truss_2d": "2D", "truss2d": "2D",
	"elastic_beam_2d": "2D", "elastic_beam2d": "2D",
	"timoshenko_beam_2d": "2D", "timoshenko_beam2d": "2D",
	"quad4_2d": "2D", "quad4": "2D",
	"quad8_2d": "2D", "quad8": "2D",
	"tri3_2d": "2D", "tri3": "2D",
	"tri6_2d": "2D", "tri6": "2D",
	"zerolength_2d":       "2D",
	"zerolength_frame_2d": "2D",
}

// elementCanonical maps any element type name (including old-style aliases) to the
// canonical name with an explicit _2d / _3d suffix.
var elementCanonical = map[string]string{
	"tet4": "tet4_3d", "tet4_3d": "tet4_3d",
	"hexa8": "hexa8_3d", "hexa8_3d": "hexa8_3d",
	"tet10": "tet10_3d", "tet10_3d": "tet10_3d",
	"brick20": "brick20_3d", "brick20_3d": "brick20_3d",
	"truss3d": "truss_3d", "truss_3d": "truss_3d",
	"corot_truss": "corot_truss_3d", "corot_truss_3d": "corot_truss_3d",
	"elastic_beam3d": "elastic_beam_3d", "elastic_beam_3d": "elastic_beam_3d",
	"timoshenko_beam3d": "timoshenko_beam_3d", "timoshenko_beam_3d": "timoshenko_beam_3d",
	"shell_mitc4": "shell_mitc4_3d", "shell_mitc4_3d": "shell_mitc4_3d",
	"dkt3": "dkt3_3d", "dkt3_3d": "dkt3_3d", "discrete_kirchhoff_triangle": "dkt3_3d",
	"zerolength": "zerolength_3d", "zerolength_3d": "zerolength_3d",
	"zerolength_trans_3d": "zerolength_trans_3d",
	"truss2d":             "truss_2d", "truss_2d": "truss_2d",
	"elastic_beam2d": "elastic_beam_2d", "elastic_beam_2d": "elastic_beam_2d",
	"timoshenko_beam2d": "timoshenko_beam_2d", "timoshenko_beam_2d": "timoshenko_beam_2d",
	"quad4": "quad4_2d", "quad4_2d": "quad4_2d",
	"quad8": "quad8_2d", "quad8_2d": "quad8_2d",
	"tri3": "tri3_2d", "tri3_2d": "tri3_2d",
	"tri6": "tri6_2d", "tri6_2d": "tri6_2d",
	"zerolength_2d":       "zerolength_2d",
	"zerolength_frame_2d": "zerolength_frame_2d",
}

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

	// Validate problem dimensions before building elements.
	declaredDim := strings.ToUpper(strings.TrimSpace(input.Dimensions))
	if declaredDim != "" {
		if declaredDim != "2D" && declaredDim != "3D" {
			return errorResponse("'dimensions' must be '2D' or '3D', got %q", input.Dimensions)
		}
		for eid, ei := range input.Elements {
			edim, known := elementDimension[ei.Type]
			if !known {
				continue // unknown type will produce a clear error in createElement
			}
			if edim != declaredDim {
				canon, ok := elementCanonical[ei.Type]
				if !ok {
					canon = ei.Type
				}
				return errorResponse("element[%d] %q is a %s element, incompatible with problem dimensions %s",
					eid, canon, edim, declaredDim)
			}
		}
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
			Rx: d[3], Ry: d[4], Rz: d[5],
		}
		for c := 0; c < 6; c++ {
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
			Dimensions:  declaredDim,
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

	// Helper: validate and extract 2D node coordinates (XY only)
	getCoords2 := func(nids []int, expected int) ([]int, [][2]float64, error) {
		if len(nids) != expected {
			return nil, nil, fmt.Errorf("%s requires %d nodes, got %d", ei.Type, expected, len(nids))
		}
		coords := make([][2]float64, expected)
		for i, nid := range nids {
			if nid < 0 || nid >= nn {
				return nil, nil, fmt.Errorf("node %d out of range", nid)
			}
			coords[i][0] = dom.Nodes[nid].Coord[0]
			coords[i][1] = dom.Nodes[nid].Coord[1]
		}
		return nids, coords, nil
	}

	switch ei.Type {
	case "tet4", "tet4_3d":
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

	case "hexa8", "hexa8_3d":
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

	case "tet10", "tet10_3d":
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

	case "brick20", "brick20_3d":
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

	case "truss3d", "truss_3d":
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

	case "corot_truss", "corot_truss_3d":
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

	case "truss2d", "truss_2d":
		nids, coords, err := getCoords2(ei.Nodes, 2)
		if err != nil {
			return nil, err
		}
		if ei.E <= 0 || ei.A <= 0 {
			return nil, fmt.Errorf("truss2d requires E > 0 and A > 0")
		}
		var n2t [2]int
		var c2t [2][2]float64
		copy(n2t[:], nids)
		copy(c2t[:], coords)
		return truss.NewTruss2D(eid, n2t, c2t, ei.E, ei.A), nil

	case "elastic_beam3d", "elastic_beam_3d":
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

	case "timoshenko_beam3d", "timoshenko_beam_3d":
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

	case "elastic_beam2d", "elastic_beam_2d":
		nids, coords, err := getCoords2(ei.Nodes, 2)
		if err != nil {
			return nil, err
		}
		if ei.E <= 0 || ei.A <= 0 {
			return nil, fmt.Errorf("elastic_beam2d requires E, A > 0")
		}
		var n2b [2]int
		var c2b [2][2]float64
		copy(n2b[:], nids)
		copy(c2b[:], coords)
		sec2d := section.BeamSection2D{A: ei.A, Iz: ei.Iz}
		return frame.NewElasticBeam2D(eid, n2b, c2b, ei.E, sec2d), nil

	case "timoshenko_beam2d", "timoshenko_beam_2d":
		nids, coords, err := getCoords2(ei.Nodes, 2)
		if err != nil {
			return nil, err
		}
		if ei.E <= 0 || ei.G <= 0 || ei.A <= 0 {
			return nil, fmt.Errorf("timoshenko_beam2d requires E, G, A > 0")
		}
		var n2b [2]int
		var c2b [2][2]float64
		copy(n2b[:], nids)
		copy(c2b[:], coords)
		sec2d := section.BeamSection2D{A: ei.A, Iz: ei.Iz, Asy: ei.Asy}
		return frame.NewTimoshenkoBeam2D(eid, n2b, c2b, ei.E, ei.G, sec2d), nil

	case "shell_mitc4", "shell_mitc4_3d":
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

	case "dkt3", "dkt3_3d", "discrete_kirchhoff_triangle":
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

	case "tri3", "tri3_2d":
		if len(ei.Nodes) != 3 {
			return nil, fmt.Errorf("tri3 requires 3 nodes, got %d", len(ei.Nodes))
		}
		for _, nid := range ei.Nodes {
			if nid < 0 || nid >= nn {
				return nil, fmt.Errorf("node %d out of range", nid)
			}
		}
		if ei.E <= 0 || ei.Thickness <= 0 {
			return nil, fmt.Errorf("tri3 requires E > 0 and thickness > 0")
		}
		var n3t [3]int
		var c3t [3][2]float64
		copy(n3t[:], ei.Nodes)
		for k, nid := range ei.Nodes {
			c3t[k][0] = dom.Nodes[nid].Coord[0]
			c3t[k][1] = dom.Nodes[nid].Coord[1]
		}
		ptypeTri3 := quad.PlaneStress
		if ei.PlaneType == "strain" {
			ptypeTri3 = quad.PlaneStrain
		}
		return quad.NewTri3(eid, n3t, c3t, ei.E, ei.Nu, ei.Thickness, ptypeTri3), nil

	case "tri6", "tri6_2d":
		if len(ei.Nodes) != 6 {
			return nil, fmt.Errorf("tri6 requires 6 nodes, got %d", len(ei.Nodes))
		}
		for _, nid := range ei.Nodes {
			if nid < 0 || nid >= nn {
				return nil, fmt.Errorf("node %d out of range", nid)
			}
		}
		if ei.E <= 0 || ei.Thickness <= 0 {
			return nil, fmt.Errorf("tri6 requires E > 0 and thickness > 0")
		}
		var n6t [6]int
		var c6t [6][2]float64
		copy(n6t[:], ei.Nodes)
		for k, nid := range ei.Nodes {
			c6t[k][0] = dom.Nodes[nid].Coord[0]
			c6t[k][1] = dom.Nodes[nid].Coord[1]
		}
		ptypeTri6 := quad.PlaneStress
		if ei.PlaneType == "strain" {
			ptypeTri6 = quad.PlaneStrain
		}
		return quad.NewTri6(eid, n6t, c6t, ei.E, ei.Nu, ei.Thickness, ptypeTri6), nil

	case "quad4", "quad4_2d":
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

	case "quad8", "quad8_2d":
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

	case "zerolength", "zerolength_3d":
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

	case "zerolength_trans_3d":
		if len(ei.Nodes) != 2 {
			return nil, fmt.Errorf("zerolength_trans_3d requires 2 nodes")
		}
		for _, nid := range ei.Nodes {
			if nid < 0 || nid >= nn {
				return nil, fmt.Errorf("node %d out of range", nid)
			}
		}
		var n2 [2]int
		copy(n2[:], ei.Nodes)
		var s3 [3]float64
		copy(s3[:], ei.Springs[:3])
		return zerolength.NewZeroLength3DOF(eid, n2, s3), nil

	case "zerolength_2d":
		if len(ei.Nodes) != 2 {
			return nil, fmt.Errorf("zerolength_2d requires 2 nodes")
		}
		for _, nid := range ei.Nodes {
			if nid < 0 || nid >= nn {
				return nil, fmt.Errorf("node %d out of range", nid)
			}
		}
		var n2 [2]int
		copy(n2[:], ei.Nodes)
		var s2 [2]float64
		s2[0] = ei.Springs[0]
		s2[1] = ei.Springs[1]
		return zerolength.NewZeroLength2D(eid, n2, s2), nil

	case "zerolength_frame_2d":
		if len(ei.Nodes) != 2 {
			return nil, fmt.Errorf("zerolength_frame_2d requires 2 nodes")
		}
		for _, nid := range ei.Nodes {
			if nid < 0 || nid >= nn {
				return nil, fmt.Errorf("node %d out of range", nid)
			}
		}
		var n2 [2]int
		copy(n2[:], ei.Nodes)
		var s3 [3]float64
		copy(s3[:], ei.Springs[:3])
		return zerolength.NewZeroLength2DFrame(eid, n2, s3), nil

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
			if canon, ok := elementCanonical[inputs[i].Type]; ok {
				ef.Type = canon
			} else {
				ef.Type = inputs[i].Type
			}
		}
		switch e := elem.(type) {
		case *truss.Truss3D:
			N := e.AxialForce()
			sigma := e.AxialStress()
			ef.N = &N
			ef.Sigma = &sigma
		case *truss.Truss2D:
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
		case *frame.ElasticBeam2D:
			ef2 := e.EndForces()
			ef.EndI = &BeamEndOutput{N: ef2.I[0], Vy: ef2.I[1], Mz: ef2.I[2]}
			ef.EndJ = &BeamEndOutput{N: ef2.J[0], Vy: ef2.J[1], Mz: ef2.J[2]}
		case *frame.TimoshenkoBeam2D:
			ef2 := e.EndForces()
			ef.EndI = &BeamEndOutput{N: ef2.I[0], Vy: ef2.I[1], Mz: ef2.I[2]}
			ef.EndJ = &BeamEndOutput{N: ef2.J[0], Vy: ef2.J[1], Mz: ef2.J[2]}
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
		case *quad.Tri3:
			s := e.StressCentroid()
			vm := solid.VonMises([6]float64{s[0], s[1], 0, s[2], 0, 0})
			ef.Stress = &StressOutput{Sxx: s[0], Syy: s[1], Txy: s[2], VonMises: vm}
		case *quad.Tri6:
			s := e.StressCentroid()
			vm := solid.VonMises([6]float64{s[0], s[1], 0, s[2], 0, 0})
			ef.Stress = &StressOutput{Sxx: s[0], Syy: s[1], Txy: s[2], VonMises: vm}
		case *shell.ShellMITC4:
			sf := e.LocalForces()
			ef.ShellForces = &ShellForcesOutput{Nx: sf.Nx, Ny: sf.Ny, Nxy: sf.Nxy, Mx: sf.Mx, My: sf.My, Mxy: sf.Mxy}
		case *shell.DiscreteKirchhoffTriangle:
			mx, my, mxy := e.LocalMoments()
			ef.ShellForces = &ShellForcesOutput{Mx: mx, My: my, Mxy: mxy}
		case *zerolength.ZeroLength:
			f := e.SpringForce()
			ef.SpringForces = &SpringForcesOutput{
				Fx: &f[0], Fy: &f[1], Fz: &f[2],
				Mx: &f[3], My: &f[4], Mz: &f[5],
			}
		case *zerolength.ZeroLength3DOF:
			f := e.SpringForce()
			ef.SpringForces = &SpringForcesOutput{Fx: &f[0], Fy: &f[1], Fz: &f[2]}
		case *zerolength.ZeroLength2D:
			f := e.SpringForce()
			ef.SpringForces = &SpringForcesOutput{Fx: &f[0], Fy: &f[1]}
		case *zerolength.ZeroLength2DFrame:
			f := e.SpringForce()
			ef.SpringForces = &SpringForcesOutput{Fx: &f[0], Fy: &f[1], Mz: &f[2]}
		}
		out[i] = ef
	}
	return out
}
