// Package integration provides Gauss quadrature rules for various element topologies.
package integration

import "math"

// Point holds a quadrature point in natural coordinates with its weight.
type Point struct {
	Xi, Eta, Zeta float64
	Weight        float64
}

// Line returns n-point Gauss-Legendre quadrature on [-1, 1].
func Line(n int) []Point {
	switch n {
	case 1:
		return []Point{{Xi: 0, Weight: 2}}
	case 2:
		g := 1.0 / math.Sqrt(3.0)
		return []Point{
			{Xi: -g, Weight: 1},
			{Xi: g, Weight: 1},
		}
	case 3:
		g := math.Sqrt(3.0 / 5.0)
		return []Point{
			{Xi: -g, Weight: 5.0 / 9.0},
			{Xi: 0, Weight: 8.0 / 9.0},
			{Xi: g, Weight: 5.0 / 9.0},
		}
	case 4:
		a := math.Sqrt(3.0/7.0 - 2.0/7.0*math.Sqrt(6.0/5.0))
		b := math.Sqrt(3.0/7.0 + 2.0/7.0*math.Sqrt(6.0/5.0))
		wa := (18.0 + math.Sqrt(30.0)) / 36.0
		wb := (18.0 - math.Sqrt(30.0)) / 36.0
		return []Point{
			{Xi: -b, Weight: wb},
			{Xi: -a, Weight: wa},
			{Xi: a, Weight: wa},
			{Xi: b, Weight: wb},
		}
	}
	return nil
}

// Quad returns n×n product Gauss points for a quadrilateral [-1,1]².
func Quad(n int) []Point {
	p1d := Line(n)
	pts := make([]Point, 0, n*n)
	for _, a := range p1d {
		for _, b := range p1d {
			pts = append(pts, Point{
				Xi:     a.Xi,
				Eta:    b.Xi,
				Weight: a.Weight * b.Weight,
			})
		}
	}
	return pts
}

// Hex returns n×n×n product Gauss points for a hexahedron [-1,1]³.
func Hex(n int) []Point {
	p1d := Line(n)
	pts := make([]Point, 0, n*n*n)
	for _, a := range p1d {
		for _, b := range p1d {
			for _, c := range p1d {
				pts = append(pts, Point{
					Xi:     a.Xi,
					Eta:    b.Xi,
					Zeta:   c.Xi,
					Weight: a.Weight * b.Weight * c.Weight,
				})
			}
		}
	}
	return pts
}

// Tet1 returns 1-point quadrature for a tetrahedron (volume = 1/6 in natural coords).
func Tet1() []Point {
	return []Point{
		{Xi: 0.25, Eta: 0.25, Zeta: 0.25, Weight: 1.0 / 6.0},
	}
}

// Tet4 returns 4-point quadrature for a tetrahedron.
func Tet4() []Point {
	a := 0.5854101966249685
	b := 0.1381966011250105
	w := 1.0 / 24.0
	return []Point{
		{Xi: a, Eta: b, Zeta: b, Weight: w},
		{Xi: b, Eta: a, Zeta: b, Weight: w},
		{Xi: b, Eta: b, Zeta: a, Weight: w},
		{Xi: b, Eta: b, Zeta: b, Weight: w},
	}
}

// Tet5 returns 5-point quadrature for a tetrahedron (degree 4).
func Tet5() []Point {
	a := 0.25
	b := 1.0 / 6.0
	c := 0.5
	w0 := -4.0 / 30.0
	w1 := 3.0 / 40.0
	return []Point{
		{Xi: a, Eta: a, Zeta: a, Weight: w0},
		{Xi: b, Eta: b, Zeta: b, Weight: w1},
		{Xi: c, Eta: b, Zeta: b, Weight: w1},
		{Xi: b, Eta: c, Zeta: b, Weight: w1},
		{Xi: b, Eta: b, Zeta: c, Weight: w1},
	}
}

// Tri1 returns 1-point quadrature for a triangle (area = 0.5 in natural coords).
func Tri1() []Point {
	return []Point{
		{Xi: 1.0 / 3.0, Eta: 1.0 / 3.0, Weight: 0.5},
	}
}

// Tri3 returns 3-point quadrature for a triangle.
func Tri3() []Point {
	return []Point{
		{Xi: 1.0 / 6.0, Eta: 1.0 / 6.0, Weight: 1.0 / 6.0},
		{Xi: 2.0 / 3.0, Eta: 1.0 / 6.0, Weight: 1.0 / 6.0},
		{Xi: 1.0 / 6.0, Eta: 2.0 / 3.0, Weight: 1.0 / 6.0},
	}
}
