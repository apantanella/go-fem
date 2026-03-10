// Package dof defines degrees-of-freedom types for finite elements.
package dof

// Type represents a single degree of freedom at a node.
type Type int

const (
	UX Type = iota // Translation X
	UY             // Translation Y
	UZ             // Translation Z
	RX             // Rotation X
	RY             // Rotation Y
	RZ             // Rotation Z
)

func (t Type) String() string {
	switch t {
	case UX:
		return "UX"
	case UY:
		return "UY"
	case UZ:
		return "UZ"
	case RX:
		return "RX"
	case RY:
		return "RY"
	case RZ:
		return "RZ"
	default:
		return "?"
	}
}

// Translational3D returns [UX, UY, UZ] repeated n times (for n nodes).
func Translational3D(nNodes int) []Type {
	out := make([]Type, 3*nNodes)
	for i := 0; i < nNodes; i++ {
		out[3*i+0] = UX
		out[3*i+1] = UY
		out[3*i+2] = UZ
	}
	return out
}

// Full6D returns [UX, UY, UZ, RX, RY, RZ] repeated n times (for n nodes).
func Full6D(nNodes int) []Type {
	out := make([]Type, 6*nNodes)
	for i := 0; i < nNodes; i++ {
		out[6*i+0] = UX
		out[6*i+1] = UY
		out[6*i+2] = UZ
		out[6*i+3] = RX
		out[6*i+4] = RY
		out[6*i+5] = RZ
	}
	return out
}
