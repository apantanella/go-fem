package main

import (
	"strings"
	"testing"

	"go-fem/domain"
)

// TestResolveElementPropsGWithNuZero verifies that G is auto-derived for an
// isotropic material with Nu = 0.  Formula: G = E / (2*(1+Nu)) = E/2.
func TestResolveElementPropsGWithNuZero(t *testing.T) {
	matsRaw := map[string]MaterialInput{
		"m1": {ID: "m1", Type: "isotropic_linear", E: 200000, Nu: 0},
	}
	ei := ElementInput{
		Type:     "elastic_beam_3d",
		Material: "m1",
	}
	resolved, err := resolveElementProps(ei, matsRaw, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := 200000.0 / (2.0 * (1.0 + 0.0)) // = 100000
	if resolved.G != want {
		t.Errorf("G = %v, want %v", resolved.G, want)
	}
}

// TestResolveElementPropsGInlinePrecedence verifies that an explicit inline G
// on the element is not overwritten by the auto-derived value.
func TestResolveElementPropsGInlinePrecedence(t *testing.T) {
	matsRaw := map[string]MaterialInput{
		"m1": {ID: "m1", Type: "isotropic_linear", E: 200000, Nu: 0.3},
	}
	ei := ElementInput{
		Type:     "elastic_beam_3d",
		Material: "m1",
		G:        99999, // explicit inline value
	}
	resolved, err := resolveElementProps(ei, matsRaw, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resolved.G != 99999 {
		t.Errorf("G = %v, want 99999 (inline takes precedence)", resolved.G)
	}
}

// TestDuplicateMaterialID verifies that solveProblem rejects inputs that
// contain two materials sharing the same ID.
func TestDuplicateMaterialID(t *testing.T) {
	out := solveProblem(ProblemInput{
		Materials: []MaterialInput{
			{ID: "m1", Type: "isotropic_linear", E: 200000, Nu: 0.3},
			{ID: "m1", Type: "isotropic_linear", E: 100000, Nu: 0.2},
		},
	})
	if out.Success {
		t.Fatal("expected failure for duplicate material ID, got success")
	}
	if !strings.Contains(out.Error, "duplicate material") {
		t.Errorf("error message %q does not mention duplicate material", out.Error)
	}
}

// TestDuplicateSectionID verifies that solveProblem rejects inputs that
// contain two sections sharing the same ID.
func TestDuplicateSectionID(t *testing.T) {
	out := solveProblem(ProblemInput{
		Materials: []MaterialInput{
			{ID: "m1", Type: "isotropic_linear", E: 200000, Nu: 0.3},
		},
		Sections: []SectionInput{
			{ID: "s1", A: 100},
			{ID: "s1", A: 200},
		},
	})
	if out.Success {
		t.Fatal("expected failure for duplicate section ID, got success")
	}
	if !strings.Contains(out.Error, "duplicate section") {
		t.Errorf("error message %q does not mention duplicate section", out.Error)
	}
}

// TestMaxAbsDisplacementTranslationOnly verifies that buildDispOutput picks the
// maximum from translational components (ux/uy/uz) only, ignoring rotations
// even when rotation magnitudes exceed the translational ones.
func TestMaxAbsDisplacementTranslationOnly(t *testing.T) {
	disp := [][6]float64{
		{1.0, 2.0, 3.0, 0.0, 0.0, 0.0}, // node 0: max translation = uz = 3.0
		{0.0, 0.0, 0.0, 5.0, 6.0, 7.0}, // node 1: rotations larger than any translation
	}
	_, maxDisp := buildDispOutput(disp)
	if maxDisp.Value != 3.0 {
		t.Errorf("MaxAbsDisplacement.Value = %v, want 3.0 (translation only)", maxDisp.Value)
	}
	if maxDisp.Component != "uz" {
		t.Errorf("MaxAbsDisplacement.Component = %q, want %q", maxDisp.Component, "uz")
	}
	if maxDisp.Node != 0 {
		t.Errorf("MaxAbsDisplacement.Node = %d, want 0", maxDisp.Node)
	}
}

// TestBCValuesLengthMismatch verifies that applyBCs returns a non-nil error
// when len(bc.Values) > 0 but len(bc.Values) != len(bc.DOFs).
func TestBCValuesLengthMismatch(t *testing.T) {
	dom := domain.NewDomain()
	dom.AddNode(0, 0, 0)
	bcs := []BCInput{
		{Node: 0, DOFs: []int{0, 1, 2}, Values: []float64{0.0, 0.0}}, // 3 dofs, 2 values
	}
	err := applyBCs(dom, bcs)
	if err == nil {
		t.Fatal("expected error for values/dofs length mismatch, got nil")
	}
}

// TestBCValuesOmittedMeansZero verifies that omitting bc.Values entirely is
// valid and results in zero prescribed displacement for every constrained DOF.
func TestBCValuesOmittedMeansZero(t *testing.T) {
	dom := domain.NewDomain()
	dom.AddNode(0, 0, 0)
	bcs := []BCInput{
		{Node: 0, DOFs: []int{0, 1, 2}}, // no Values field
	}
	if err := applyBCs(dom, bcs); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, bc := range dom.BCs {
		if bc.Value != 0.0 {
			t.Errorf("BC Value = %v, want 0.0 for omitted values", bc.Value)
		}
	}
}
