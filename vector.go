package sparse

import (
	"fmt"
	"math"
)

// Vector is an indexed representation of a multidimensional vector.
type Vector struct {
	dim  int
	data map[int]float64
}

// Size is the dimensionality of the vector.
func (v Vector) Size() int {
	return v.dim
}

// Grow Vector to n-dimensions, if n is greater than the current
// number of dimensions.
func (v Vector) Grow(n int) Vector {
	if n > v.dim {
		v.dim = n
	}

	return v
}

// Set data on the n'th dimension.
func (v Vector) Set(n int, data float64) {
	v.data[n] = data
}

// Get data from the n'th dimension.
func (v Vector) Get(n int) float64 {
	return v.data[n]
}

// Load data from an array of floats.
func (v Vector) Load(data []float64) {
	for i, f := range data {
		if f != 0 {
			v.Set(i, f)
		}
	}
}

// Magnitude (scalar) of the vector.
func (v Vector) Magnitude() float64 {
	ret := float64(0)
	for _, val := range v.data {
		ret += math.Pow(val, 2)
	}

	return math.Sqrt(ret)
}

// Times a scalar, means multiple this vector with a scalar.
func (v Vector) Times(scalar float64) Vector {
	ret := v.clone()
	for n, d := range ret.data {
		ret.data[n] = d * scalar
	}

	return ret
}

func (v Vector) String() string {
	return fmt.Sprintf("%v", v.data)
}

// Append one Vector to another.
func Append(v1 Vector, v2 Vector) Vector {
	baseDim := v1.Size()
	v1.dim = baseDim + v2.Size()
	for n, d := range v2.data {
		v1.Set(baseDim+n, d)
	}

	return v1
}

// clone a Vector to a new instance to avoid side-effects.
func (v Vector) clone() Vector {
	clone := v
	clone.data = make(map[int]float64)
	for n, d := range v.data {
		clone.data[n] = d
	}

	return clone
}

// smallerBigger is a private helper that helps sort two vector based
// on which one is less sparse (smaller) than the other (bigger)
func smallerBigger(v1 Vector, v2 Vector) (*Vector, *Vector) {
	var (
		smaller *Vector
		bigger  *Vector
	)

	smaller = &v1
	bigger = &v1
	if len(v2.data) >= len(v1.data) {
		bigger = &v2
	} else {
		smaller = &v2
	}

	return smaller, bigger
}

// Add two Vectors.
func Add(v1 Vector, v2 Vector) Vector {
	smaller, bigger := smallerBigger(v1, v2)
	biggerClone := (*bigger).clone()
	for n, d := range (*smaller).data {
		biggerClone.data[n] += d
	}

	return biggerClone
}

// Dot product of two Vectors.
func Dot(v1 Vector, v2 Vector) float64 {
	ret := float64(0)
	smaller, bigger := smallerBigger(v1, v2)

	for n := range (*smaller).data {
		ret += (*smaller).Get(n) * (*bigger).Get(n)
	}

	return ret
}

// Acos is a measure of similarity between vectors.
func Acos(v1 Vector, v2 Vector) float64 {
	dotProduct := Dot(v1, v2)
	scalarProduct := v1.Magnitude() * v2.Magnitude()
	return math.Acos(dotProduct / scalarProduct)
}

// NormAcos is a normalised measure of similarity between vectors.
func NormAcos(v1 Vector, v2 Vector) float64 {
	v1 = v1.Times(1 / v1.Magnitude())
	v2 = v2.Times(1 / v2.Magnitude())
	return math.Acos(Dot(v1, v2))
}

// Similarity is a convenience function for Cos(Acos(v1, v2)).
func Similarity(v1 Vector, v2 Vector) float64 {
	return math.Cos(Acos(v1, v2))
}

// NormalizedSimilarity is Similarity except normalized.
func NormalizedSimilarity(v1 Vector, v2 Vector) float64 {
	return math.Cos(NormAcos(v1, v2))
}

// NewVector constructs a blank Vector with dim number of dimensions.
func NewVector(dim int) Vector {
	return Vector{
		dim:  dim,
		data: map[int]float64{},
	}
}

// NewVectorFromArray maps an array to a Vector.
func NewVectorFromArray(arr []float64) Vector {
	ret := NewVector(len(arr))
	ret.Load(arr)
	return ret
}
