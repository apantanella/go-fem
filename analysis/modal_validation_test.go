package analysis_test

// Validation tests for the modal analysis implementation.
//
// Each test checks a property that is either analytically exact or a
// fundamental mathematical property of the generalized eigenvalue problem:
//
//  1. Mass-matrix properties:  symmetry, positive diagonal, analytical trace
//  2. Rayleigh quotient:       ω²ₖ  =  φₖᵀ · K_red · φₖ
//  3. M-orthonormality:        φᵢᵀ · M_red · φⱼ  =  δᵢⱼ
//  4. K-orthogonality:         φᵢᵀ · K_red · φⱼ  =  0  for i ≠ j
//  5. Effective mass sum:       Σₖ meff_{k,d}  →  1  as numModes → nFree
//  6. Cantilever convergence:  4-element mesh, < 0.2 % error vs analytical

import (
	"math"
	"testing"

	"go-fem/domain"
	"go-fem/element/frame"
	"go-fem/element/solid"
	"go-fem/element/truss"
	"go-fem/material"
	"go-fem/analysis"
	"go-fem/dof"
	"go-fem/section"
	"go-fem/solver"
)

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Mass matrix properties
// ─────────────────────────────────────────────────────────────────────────────

// TestMassMatrixSymmetry checks that GetMassMatrix returns a symmetric matrix
// for every element type that implements MassMatrixAssembler.
//
// Property: Me = Meᵀ. The consistent mass matrix Me = ρ·∫ Nᵀ·N dV is symmetric
// by construction (it is an outer product of the shape-function matrix N with
// itself). The test covers six element types:
//   - Truss3D (6×6), Truss2D (4×4)
//   - ElasticBeam2D (6×6), ElasticBeam3D (12×12)
//   - Hexa8 (24×24), Tet4 (12×12)
//
// Each element is tested with a unit-length or unit-cube geometry so that
// the expected mass is known analytically.
//
// Why valuable: an unsymmetric mass matrix would cause the eigenvalue solver
// to produce complex or non-physical eigenvalues and would corrupt participation
// factor calculations that rely on M-orthonormality.
func TestMassMatrixSymmetry(t *testing.T) {
	rho := 7850.0

	type namedMassMatrix struct {
		name string
		me   interface{ At(int, int) float64 }
		n    int
	}

	// --- Truss3D ---
	{
		e := truss.NewTruss3D(0, [2]int{0, 1},
			[2][3]float64{{0, 0, 0}, {1, 0, 0}}, 200e9, 1e-4)
		me := e.GetMassMatrix(rho)
		checkSymmetry(t, "Truss3D", me, 6)
		checkPositiveDiag(t, "Truss3D", me, 6)
	}
	// --- Truss2D ---
	{
		e := truss.NewTruss2D(0, [2]int{0, 1},
			[2][2]float64{{0, 0}, {1, 0}}, 200e9, 1e-4)
		me := e.GetMassMatrix(rho)
		checkSymmetry(t, "Truss2D", me, 4)
		checkPositiveDiag(t, "Truss2D", me, 4)
	}
	// --- ElasticBeam2D ---
	{
		sec := section.BeamSection2D{A: 0.01, Iz: 8.33e-6}
		e := frame.NewElasticBeam2D(0, [2]int{0, 1},
			[2][2]float64{{0, 0}, {2, 0}}, 200e9, sec)
		me := e.GetMassMatrix(rho)
		checkSymmetry(t, "ElasticBeam2D", me, 6)
		checkPositiveDiag(t, "ElasticBeam2D", me, 6)
	}
	// --- ElasticBeam3D ---
	{
		sec := section.BeamSection3D{A: 0.01, Iy: 8.33e-6, Iz: 8.33e-6, J: 1.4e-5}
		e := frame.NewElasticBeam3D(0, [2]int{0, 1},
			[2][3]float64{{0, 0, 0}, {2, 0, 0}}, 200e9, 80e9, sec,
			[3]float64{0, 0, 1})
		me := e.GetMassMatrix(rho)
		checkSymmetry(t, "ElasticBeam3D", me, 12)
		checkPositiveDiag(t, "ElasticBeam3D", me, 12)
	}
	// --- Hexa8 ---
	{
		mat3d := material.NewIsotropicLinear(200e9, 0.3)
		coords := [8][3]float64{
			{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
			{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
		}
		e := solid.NewHexa8(0, [8]int{0, 1, 2, 3, 4, 5, 6, 7}, coords, mat3d)
		me := e.GetMassMatrix(rho)
		checkSymmetry(t, "Hexa8", me, 24)
		checkPositiveDiag(t, "Hexa8", me, 24)
	}
	// --- Tet4 ---
	{
		mat3d := material.NewIsotropicLinear(200e9, 0.3)
		coords := [4][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
		e := solid.NewTet4(0, [4]int{0, 1, 2, 3}, coords, mat3d)
		me := e.GetMassMatrix(rho)
		checkSymmetry(t, "Tet4", me, 12)
		checkPositiveDiag(t, "Tet4", me, 12)
	}
}

// checkSymmetry asserts me[i,j] == me[j,i] up to numerical noise.
//
// The tolerance is adaptive: |me[i,j] - me[j,i]| < 1e-10·(|a|+|b|+1e-30)
// to handle both large and near-zero entries correctly.
func checkSymmetry(t *testing.T, name string, me interface{ At(int, int) float64 }, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			a, b := me.At(i, j), me.At(j, i)
			if math.Abs(a-b) > 1e-10*(math.Abs(a)+math.Abs(b)+1e-30) {
				t.Errorf("%s: me[%d,%d]=%.6e ≠ me[%d,%d]=%.6e", name, i, j, a, j, i, b)
			}
		}
	}
}

// checkPositiveDiag asserts all diagonal entries of the mass matrix are positive.
//
// Property: a consistent mass matrix Me = ρ·∫ Nᵀ·N dV is positive semi-definite,
// and for well-formed elements each diagonal entry Me[i,i] = ρ·∫ Nᵢ² dV > 0.
func checkPositiveDiag(t *testing.T, name string, me interface{ At(int, int) float64 }, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		if me.At(i, i) <= 0 {
			t.Errorf("%s: non-positive diagonal me[%d,%d]=%.6e", name, i, i, me.At(i, i))
		}
	}
}

// TestMassMatrixTraceTruss3D verifies the analytical trace of the Truss3D
// consistent mass matrix.
//
// Analytical formula: for a bar element with length L, cross-section A and
// density ρ, the consistent mass matrix is:
//
//	Me = (ρAL/6) · [2I₃, I₃; I₃, 2I₃]
//
// where I₃ is the 3×3 identity matrix. The trace (sum of diagonal entries) is:
//
//	trace(Me) = (ρAL/6) · (2·3 + 2·3) = (ρAL/6)·12 = 2·ρAL
//
// Parameters: ρ=7850, A=1e-4, L=2.0.
//
// Expected: trace = 2·ρAL = 2·7850·1e-4·2 = 3.14.
//
// Why valuable: confirms that the total element mass is ρAL and that it is
// correctly split (2/3 to near-end, 1/3 to far-end, combining to 2·ρAL total).
func TestMassMatrixTraceTruss3D(t *testing.T) {
	rho, A, L := 7850.0, 1e-4, 2.0
	e := truss.NewTruss3D(0, [2]int{0, 1},
		[2][3]float64{{0, 0, 0}, {L, 0, 0}}, 200e9, A)
	me := e.GetMassMatrix(rho)

	tr := 0.0
	for i := 0; i < 6; i++ {
		tr += me.At(i, i)
	}
	want := 2 * rho * A * L
	if relErr(tr, want) > 1e-10 {
		t.Errorf("Truss3D trace = %.6e, want %.6e (err %.2e)", tr, want, relErr(tr, want))
	}
}

// TestMassMatrixTraceTet4 verifies the analytical trace of the Tet4 consistent
// mass matrix.
//
// Analytical formula: for the linear tetrahedron with volume V, the consistent
// mass matrix has the structure:
//
//	Me[3n+k, 3m+k] = ρV/20 · (1 + δₙₘ)   for n,m ∈ {0,1,2,3}, k ∈ {0,1,2}
//
// The diagonal entries are Me[i,i] = ρV/10, giving a trace of:
//
//	trace = 12 · ρV/10 = 6ρV/5
//
// Parameters: unit right-angle tet (V=1/6), ρ=7850.
//
// Expected: trace = 6·7850·(1/6)/5 = 1570.
//
// Why valuable: confirms the Tet4 mass integration correctly captures all 12
// DOFs (4 nodes × 3 directions) with the right shape-function integrals.
func TestMassMatrixTraceTet4(t *testing.T) {
	rho := 7850.0
	mat3d := material.NewIsotropicLinear(200e9, 0.3)
	// Regular tetrahedron with known volume V = 1/6 (base triangle area 1/2, height 1)
	coords := [4][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	e := solid.NewTet4(0, [4]int{0, 1, 2, 3}, coords, mat3d)
	me := e.GetMassMatrix(rho)

	tr := 0.0
	for i := 0; i < 12; i++ {
		tr += me.At(i, i)
	}
	V := e.Volume()
	want := 6.0 / 5.0 * rho * V
	if relErr(tr, want) > 1e-10 {
		t.Errorf("Tet4 trace = %.6e, want %.6e (err %.2e)", tr, want, relErr(tr, want))
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Rayleigh quotient:  ω²ₖ = φₖᵀ · K_red · φₖ
//
// This holds exactly when modes are M-normalised (φₖᵀ · M_red · φₖ = 1).
// It is the most direct validation of the eigenvalue solve.
// ─────────────────────────────────────────────────────────────────────────────

// TestModalRayleighQuotient verifies that the Rayleigh quotient of each mode
// shape equals the corresponding eigenvalue ω²ₖ.
//
// Property: for M-normalised mode shapes (φₖᵀ·M·φₖ = 1) the Rayleigh quotient is:
//
//	RQ(φₖ) = φₖᵀ · K_red · φₖ = ω²ₖ
//
// This is the most direct check on the eigenvalue solver: any error in the
// eigenvector or eigenvalue would produce a mismatch.
//
// Setup: 3-element 2D cantilever with 6 free modes extracted.
//
// Tolerance: 1e-8 relative error on ω² for each mode.
//
// Why valuable: if the eigenvectors are not M-normalised or the solver returns
// incorrect eigenvalues, the Rayleigh quotient would differ from ω²ₖ.
func TestModalRayleighQuotient(t *testing.T) {
	a, res := buildCantilever2D(t, 3, 6) // 3 elements, extract 6 modes (all free DOFs)

	freeDOFs := a.Dom.FreeDOFs()
	Kred := solver.ExtractSubmatrix(a.Dom.K, freeDOFs)

	for k := 0; k < res.NumModes; k++ {
		phiRed := extractFree(res.ModeShapes[k], freeDOFs)
		rq := quadForm(Kred, phiRed)
		if relErr(rq, res.Omega2[k]) > 1e-8 {
			t.Errorf("mode %d: Rayleigh ω² = %.6e, eigenvalue = %.6e (err %.2e)",
				k+1, rq, res.Omega2[k], relErr(rq, res.Omega2[k]))
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  M-orthonormality:  φᵢᵀ · M_red · φⱼ = δᵢⱼ
// ─────────────────────────────────────────────────────────────────────────────

// TestModalMOrthonormality verifies that the extracted mode shapes are
// orthonormal with respect to the mass matrix.
//
// Property: for the generalised eigenvalue problem K·φ = ω²·M·φ, the mass
// matrix provides an inner product. The solution modes are M-orthonormal:
//
//	φᵢᵀ · M · φⱼ = δᵢⱼ   (Kronecker delta)
//
// This means diagonal terms (i=j) must be exactly 1.0 (normalisation), and
// off-diagonal terms (i≠j) must be zero (orthogonality).
//
// Setup: 3-element 2D cantilever with all 6 free modes extracted.
//
// Tolerance: |φᵢᵀ·M·φⱼ - δᵢⱼ| < 1e-8.
//
// Why valuable: a failure indicates the eigenvectors are not properly normalised
// or that numerical round-off has corrupted orthogonality, which would invalidate
// modal superposition.
func TestModalMOrthonormality(t *testing.T) {
	a, res := buildCantilever2D(t, 3, 6)

	freeDOFs := a.Dom.FreeDOFs()
	Mred := solver.ExtractSubmatrix(a.Dom.M, freeDOFs)

	for i := 0; i < res.NumModes; i++ {
		phi_i := extractFree(res.ModeShapes[i], freeDOFs)
		for j := 0; j < res.NumModes; j++ {
			phi_j := extractFree(res.ModeShapes[j], freeDOFs)
			mij := bilinearForm(Mred, phi_i, phi_j)
			want := 0.0
			if i == j {
				want = 1.0
			}
			if math.Abs(mij-want) > 1e-8 {
				t.Errorf("φ%dᵀ·M·φ%d = %.6e, want %.6f", i+1, j+1, mij, want)
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  K-orthogonality:  φᵢᵀ · K_red · φⱼ = ω²ᵢ · δᵢⱼ
// ─────────────────────────────────────────────────────────────────────────────

// TestModalKOrthogonality verifies that the extracted mode shapes are
// orthogonal with respect to the stiffness matrix.
//
// Property: for M-normalised modes of the generalised eigenvalue problem:
//
//	φᵢᵀ · K · φⱼ = ω²ᵢ · δᵢⱼ
//
// Diagonal terms give the eigenvalue (Rayleigh quotient), and off-diagonal
// terms must be zero. This is the modal stiffness matrix being diagonal.
//
// Setup: 3-element 2D cantilever with all 6 free modes.
//
// Tolerance: |φᵢᵀ·K·φⱼ - ω²ᵢ·δᵢⱼ| / (ω²ᵢ + 1) < 1e-8.
//
// Why valuable: K-non-orthogonality would indicate that the eigenvectors
// are not truly decoupling the equations of motion, which would require
// off-diagonal terms in the modal equations.
func TestModalKOrthogonality(t *testing.T) {
	a, res := buildCantilever2D(t, 3, 6)

	freeDOFs := a.Dom.FreeDOFs()
	Kred := solver.ExtractSubmatrix(a.Dom.K, freeDOFs)

	for i := 0; i < res.NumModes; i++ {
		phi_i := extractFree(res.ModeShapes[i], freeDOFs)
		for j := 0; j < res.NumModes; j++ {
			phi_j := extractFree(res.ModeShapes[j], freeDOFs)
			kij := bilinearForm(Kred, phi_i, phi_j)
			var want float64
			if i == j {
				want = res.Omega2[i]
			}
			scale := res.Omega2[i] + 1.0 // avoid divide-by-zero for small ω²
			if math.Abs(kij-want)/scale > 1e-8 {
				t.Errorf("φ%dᵀ·K·φ%d = %.6e, want %.6e", i+1, j+1, kij, want)
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  Effective mass sum
//
// When all free modes are extracted, the sum of effective masses in each
// direction must equal 1 (100 %) — this is the completeness condition.
// ─────────────────────────────────────────────────────────────────────────────

// TestModalEffectiveMassSum verifies that the cumulative effective mass in
// each direction reaches 100% when all free modes are extracted.
//
// Property (completeness): for a structure with n_free free DOFs, when all
// n_free modes are extracted, the sum of effective masses in each direction d
// must equal the total structural mass:
//
//	Σₖ meff_{k,d} = M_total_d = 1  (as a fraction of total mass)
//
// This follows from the completeness of the eigenvector basis.
//
// Setup: 2-element 2D cantilever (2·3 = 6 free DOFs), extract all 6 modes.
//
// Tolerance: |cumulative_meff - 1.0| < 1e-6 for X and Y directions.
//
// Why valuable: if the participation factors or effective mass computation
// has a scaling error, the sum would differ from 1, which would cause modal
// response spectra calculations to miss part of the response.
func TestModalEffectiveMassSum(t *testing.T) {
	// Use 2 beam elements → 2*3 = 6 free DOFs → extract all 6 modes.
	a, res := buildCantilever2D(t, 2, 6)
	_ = a

	// Summing over all 6 modes, the effective mass in Y (bending) and X (axial)
	// must reach 100 %. Z is unused in 2D.
	for d, name := range []string{"X", "Y"} {
		total := res.CumulativeEffectiveMass[res.NumModes-1][d]
		if math.Abs(total-1.0) > 1e-6 {
			t.Errorf("cumulative effective mass direction %s = %.6f%%, want 100%%",
				name, total*100)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  Cantilever convergence
//
// 4-element 2D cantilever — fundamental bending frequency.
// Analytical (Euler-Bernoulli): ω₁² = (1.8751)⁴ · EI/(ρAL⁴)
// FEM with consistent mass converges from above; at n=4 the error is < 0.2 %.
// ─────────────────────────────────────────────────────────────────────────────

// TestModalCantileverConvergence verifies that the fundamental bending frequency
// of a 4-element 2D cantilever matches the Euler-Bernoulli analytical value
// within 0.2%, and that the FEM result is an upper bound.
//
// Analytical formula (Euler-Bernoulli continuum):
//
//	ω₁² = (β₁L)⁴ · EI/(ρAL⁴)   with β₁L = 1.8751 rad
//
// Upper-bound property: FEM with consistent mass is a Ritz method that
// overestimates all natural frequencies. Therefore ω²_FEM > ω²_exact.
//
// Parameters: E=200e9, A=0.01 m², Iz=A²/12, L=2.0 m, ρ=7850 kg/m³.
// Mesh: 4 equal beam elements.
//
// Why valuable: if the mass matrix or stiffness matrix has a systematic error
// (e.g., wrong length scaling), the convergence rate and/or the upper-bound
// property would be violated.
func TestModalCantileverConvergence(t *testing.T) {
	const (
		E   = 200e9  // Pa
		A   = 0.01   // m²
		Lmm = 2.0    // m  total length
		rho = 7850.0 // kg/m³
	)
	Iz := A * A / 12 // I for square section = a⁴/12

	_, res := buildCantilever2D(t, 4, 3)

	EI := E * Iz
	rhoAL4 := rho * A * math.Pow(Lmm, 4)
	beta1L := 1.8751
	omega2Exact := math.Pow(beta1L, 4) * EI / rhoAL4

	got := res.Omega2[0]
	err := relErr(got, omega2Exact)
	t.Logf("n=4: ω²_FEM=%.6e  ω²_exact=%.6e  err=%.4f%%  f₁=%.4f Hz",
		got, omega2Exact, err*100, res.Frequencies[0])

	// FEM with consistent mass overestimates — should be within 0.2 % of exact.
	if err > 0.002 {
		t.Errorf("convergence error = %.4f%%, want < 0.2%%", err*100)
	}
	// FEM should overestimate (upper bound property of Ritz method).
	if got < omega2Exact {
		t.Errorf("FEM should overestimate: ω²_FEM=%.6e < ω²_exact=%.6e", got, omega2Exact)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 7.  Truss2D: total mass conservation
//
// The assembled global mass matrix M should satisfy rᵀ · M · r = ρAL
// in each translational direction, where r is the unit direction vector (1s
// on all UX or all UY DOFs).  This is the total structural mass.
// ─────────────────────────────────────────────────────────────────────────────

// TestModalTotalMassConservation verifies that the assembled global mass matrix
// conserves the total structural mass in each translational direction.
//
// Property: for a single-element model, rᵀ·M·r = ρAL where r is the
// rigid-body vector (all DOFs = 1 in one direction). This is equivalent to:
//
//	Σᵢ Σⱼ M[i,j] · rᵢ · rⱼ = total mass
//
// For a bar element the total mass is ρAL regardless of the mass matrix type
// (consistent or lumped).
//
// Parameters: single Truss2D from (0,0) to (L=1,0), E=200e9, A=1e-4, ρ=7850.
// Total mass = 7850·1e-4·1.0 = 0.785 kg.
//
// Why valuable: a wrong volume or density factor in the mass assembly would
// cause the total mass to be incorrect, leading to wrong natural frequencies
// and wrong effective mass fractions.
func TestModalTotalMassConservation(t *testing.T) {
	E, A, L, rho := 200e9, 1e-4, 1.0, 7850.0

	dom := domain.NewDomain()
	n0 := dom.AddNode(0, 0, 0)
	n1 := dom.AddNode(L, 0, 0)
	e := truss.NewTruss2D(0, [2]int{n0, n1}, [2][2]float64{{0, 0}, {L, 0}}, E, A)
	dom.AddElement(e)

	dom.Assemble()
	dom.AssembleMassMatrix([]domain.ElementMass{{ElemIdx: 0, Rho: rho}})

	M := dom.M
	ndof := 4
	totalMass := rho * A * L

	// rᵀ · M · r  where r[i]=1 for UX DOFs (0, 2) gives total mass in X.
	rX := []float64{1, 0, 1, 0}
	rY := []float64{0, 1, 0, 1}

	massX := quadFormSlice(M, rX, ndof)
	massY := quadFormSlice(M, rY, ndof)

	for dir, got := range []float64{massX, massY} {
		name := [2]string{"X", "Y"}[dir]
		if relErr(got, totalMass) > 1e-10 {
			t.Errorf("total mass dir %s = %.6e, want %.6e (err %.2e)",
				name, got, totalMass, relErr(got, totalMass))
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 8.  3D cantilever beam — bending frequency matches 2D result
//
// A 3D beam oriented along the X-axis should give the same fundamental
// bending frequency as the equivalent 2D beam.
// ─────────────────────────────────────────────────────────────────────────────

// TestModalBeam3DMatchesBeam2D verifies that the fundamental bending frequency
// of a 3D beam cantilever equals that of the equivalent 2D beam.
//
// Property: for a 3D beam with Iy = Iz, the two bending planes (XY and XZ)
// have identical stiffness and mass distributions. The lowest bending mode of
// the 3D beam must therefore equal the single bending mode of the 2D beam.
//
// Setup:
//   - 2D: ElasticBeam2D, single element, clamped at node 0, L=2 m
//   - 3D: ElasticBeam3D with Iy=Iz, same L, clamped all 6 DOFs at node 0
//
// Expected: |f_3D - f_2D| / f_2D < 1e-6.
//
// Why valuable: a discrepancy would indicate that the 3D beam mass matrix
// uses a different rotational inertia formulation than the 2D beam, or that
// the mass contribution from the out-of-plane bending DOFs is incorrect.
func TestModalBeam3DMatchesBeam2D(t *testing.T) {
	const (
		E   = 200e9
		A   = 0.01
		L   = 2.0
		rho = 7850.0
	)
	Iz := A * A / 12

	// ── 2D result (single element, 3 free DOFs) ───────────────────────────
	sec2d := section.BeamSection2D{A: A, Iz: Iz}
	dom2d := domain.NewDomain()
	n0_2d := dom2d.AddNode(0, 0, 0)
	n1_2d := dom2d.AddNode(L, 0, 0)
	e2d := frame.NewElasticBeam2D(0, [2]int{n0_2d, n1_2d},
		[2][2]float64{{0, 0}, {L, 0}}, E, sec2d)
	dom2d.AddElement(e2d)
	dom2d.FixDOF(n0_2d, int(dof.UX))
	dom2d.FixDOF(n0_2d, int(dof.UY))
	dom2d.FixDOF(n0_2d, int(dof.RZ))
	res2d, err := (&analysis.ModalAnalysis{
		Dom:      dom2d,
		Masses:   []domain.ElementMass{{ElemIdx: 0, Rho: rho}},
		NumModes: 1,
	}).Run()
	if err != nil {
		t.Fatalf("2D: %v", err)
	}

	// ── 3D result (single element, 6 free DOFs) ───────────────────────────
	sec3d := section.BeamSection3D{A: A, Iy: Iz, Iz: Iz, J: 2 * Iz}
	dom3d := domain.NewDomain()
	n0_3d := dom3d.AddNode(0, 0, 0)
	n1_3d := dom3d.AddNode(L, 0, 0)
	e3d := frame.NewElasticBeam3D(0, [2]int{n0_3d, n1_3d},
		[2][3]float64{{0, 0, 0}, {L, 0, 0}}, E, 80e9, sec3d, [3]float64{0, 0, 1})
	dom3d.AddElement(e3d)
	// Clamp all 6 DOFs at node 0
	for _, d := range []int{int(dof.UX), int(dof.UY), int(dof.UZ),
		int(dof.RX), int(dof.RY), int(dof.RZ)} {
		dom3d.FixDOF(n0_3d, d)
	}
	res3d, err := (&analysis.ModalAnalysis{
		Dom:      dom3d,
		Masses:   []domain.ElementMass{{ElemIdx: 0, Rho: rho}},
		NumModes: 6,
	}).Run()
	if err != nil {
		t.Fatalf("3D: %v", err)
	}

	// The lowest mode of the 3D beam is in the X-Y or X-Z plane.
	// Because Iy = Iz, both bending planes have the same frequency.
	f2d := res2d.Frequencies[0]
	f3d := res3d.Frequencies[0] // degenerate pair, either bending plane

	t.Logf("2D f₁ = %.6f Hz,  3D f₁ = %.6f Hz", f2d, f3d)

	if relErr(f3d, f2d) > 1e-6 {
		t.Errorf("3D bending frequency %.6f Hz ≠ 2D %.6f Hz (err %.2e)",
			f3d, f2d, relErr(f3d, f2d))
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// buildCantilever2D builds a 2D cantilever with nElem ElasticBeam2D elements
// of total length 2 m, runs ModalAnalysis, and returns both the analysis
// object (for DOM access) and the result.
//
// The cantilever is clamped at node 0 (UX, UY, RZ fixed) and free at all
// other nodes. The element section is a 100mm×100mm square: A=0.01 m²,
// Iz=A²/12. Steel material: E=200e9 Pa, ρ=7850 kg/m³.
func buildCantilever2D(t *testing.T, nElem, numModes int) (*analysis.ModalAnalysis, *analysis.ModalResult) {
	t.Helper()
	const (
		E   = 200e9
		A   = 0.01
		L   = 2.0
		rho = 7850.0
	)
	Iz := A * A / 12
	sec := section.BeamSection2D{A: A, Iz: Iz}
	Le := L / float64(nElem)

	dom := domain.NewDomain()
	nodes := make([]int, nElem+1)
	for i := 0; i <= nElem; i++ {
		nodes[i] = dom.AddNode(float64(i)*Le, 0, 0)
	}
	masses := make([]domain.ElementMass, nElem)
	for i := 0; i < nElem; i++ {
		coords := [2][2]float64{
			{float64(i) * Le, 0},
			{float64(i+1) * Le, 0},
		}
		elem := frame.NewElasticBeam2D(i, [2]int{nodes[i], nodes[i+1]}, coords, E, sec)
		dom.AddElement(elem)
		masses[i] = domain.ElementMass{ElemIdx: i, Rho: rho}
	}
	// Clamp node 0
	dom.FixDOF(nodes[0], int(dof.UX))
	dom.FixDOF(nodes[0], int(dof.UY))
	dom.FixDOF(nodes[0], int(dof.RZ))

	a := &analysis.ModalAnalysis{Dom: dom, Masses: masses, NumModes: numModes}
	res, err := a.Run()
	if err != nil {
		t.Fatalf("ModalAnalysis.Run (nElem=%d): %v", nElem, err)
	}
	return a, res
}

// extractFree extracts the sub-vector of phi at the free DOF indices.
//
// Used to reduce a full-size mode shape vector to the free-DOF subspace
// for Rayleigh quotient and orthogonality checks.
func extractFree(phi []float64, freeDOFs []int) []float64 {
	v := make([]float64, len(freeDOFs))
	for i, g := range freeDOFs {
		v[i] = phi[g]
	}
	return v
}

// quadForm computes the quadratic form vᵀ · A · v.
//
// Used to evaluate Rayleigh quotients φᵀ·K·φ and φᵀ·M·φ.
func quadForm(A interface{ At(int, int) float64 }, v []float64) float64 {
	n := len(v)
	s := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			s += v[i] * A.At(i, j) * v[j]
		}
	}
	return s
}

// bilinearForm computes the bilinear form uᵀ · A · v.
//
// Used to evaluate cross-mode inner products φᵢᵀ·M·φⱼ and φᵢᵀ·K·φⱼ
// for orthogonality checks.
func bilinearForm(A interface{ At(int, int) float64 }, u, v []float64) float64 {
	n := len(u)
	s := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			s += u[i] * A.At(i, j) * v[j]
		}
	}
	return s
}

// quadFormSlice computes the quadratic form rᵀ · M · r for the total-mass
// conservation check.
//
// The matrix M is accessed via the At(i,j) interface, and the vector r is
// provided as a plain float64 slice.
func quadFormSlice(M interface{ At(int, int) float64 }, r []float64, n int) float64 {
	s := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			s += r[i] * M.At(i, j) * r[j]
		}
	}
	return s
}
