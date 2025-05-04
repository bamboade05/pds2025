# ----- Working with NumPy -----

import numpy as np

# Array Creation
a = np.array([2, 3, 4])
print(a)
print(a.dtype)  # datatype
print("\n")

b = np.array([1.2, 3.5, 5.1])
print(b.dtype)  # datatype
print("\n")

# a = np.array(1, 2, 3, 4) # WRONG
a = np.array([1, 2, 3, 4])  # RIGHT

b = np.array([(1.5, 2, 3), (4, 5, 6)])
print(b)

print("\n")
# Can also be explicitly specified at creation time:
c = np.array([[1, 2], [3, 4]], dtype=complex)
print(c)
print("\n")

# function zero creates an array full of zeros
np.zeros((3, 4))
print(np.zeros((3, 4)))
print("\n")

# function ones creates an array full of ones
np.ones((2, 3, 4), dtype=np.int16)
print(np.ones((2, 3, 4), dtype=np.int16))
print("\n")

np.empty((2, 3))
print(np.empty((2, 3)))  # may vary
print("\n")

# NumPy provides the arrange functon which is analogous to the Python
# built-in range, but returns an array
np.arange(10, 30, 5)
print(np.arange(10, 30, 5))  # create sequences of numbers
print("\n")

np.arange(0, 2, 0.3)
print(np.arange(0, 2, 0.3))  # it accepts float arguments
print("\n")

# use function linspace that receives as an argument the number
# of elements we want, instead of the step:

from numpy import pi

np.linspace(0, 2, 9)  # 9 numbers from 0 to 2
print(np.linspace(0, 2, 9))
print("\n")

x = np.linspace(0, 2 * pi, 100) # useful to evaluate function at lots of points
f = np.sin(x)
print(f)
print("\n")

# --- Basic Operations ---

# Arithmetic operators on arrays apply elementwise. A new array is created and filled with the
# result.

a = np.array([20, 30, 40, 50])
b = np.arange(4)
print(b)
print("\n")

c = a - b
print(c)
print("\n")

print(b**2)
print("\n")

print(10 * np.sin(a))
print("\n")

print(a < 35)
print("\n")

# operator * operates elementwise in NumPy arrays

# matrix product can be performed using the @ operator (python >=3.5)
# or dot function or method

A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])
print(A * B) # elementwise product
print("\n")

print(A @ B) # matrix product
print("\n")

# Some operations, such as += and *=, act in place to modify an existing array rather than
# create a new one.

rg = np.random.default_rng(1) # create instance of default random number generator
a = np.ones((2, 3), dtype=int)
b = rg.random((2, 3))
a *= 3
print(a)
print("\n")

b += a
print(b)
print("\n")

# a += b # b is not automatically converted to integer type

# When operating with arrays of different types, the type of the resulting array corresponds to
# the more general or precise one (a behavior known as upcasting).

a = np.ones(3, dtype=np.int32)
b = np.linspace(0, pi, 3)
print(b.dtype.name)
print("\n")

c = a + b
print(c)
print(c.dtype.name)
print("\n")

d = np.exp(c * 1j)
print(d)
print(d.dtype.name)
print("\n")

a = rg.random((2, 3))
print(a)
print("\n")

print(a.sum())
print(a.min())
print(a.max())
print("\n")

b = np.arange(12).reshape(3, 4)
print(b)
print("\n")

print(b.sum(axis=0)) # sum of each column
print(b.min(axis=1)) # min of each row
print("\n")

print(b.cumsum(axis=1)) # cumulative sum along each row
print("\n")

# --- Universal Functions ---

# ufunc function operates elementwise on an array, producing an array as an output.

B = np.arange(3)
print(B)
print("\n")

print(np.sqrt(B))
print("\n")

C = np.array([2., -1., 4.])
print(np.add(B, C))
print("\n")

# --- Indexing, Slicing and Iterating ---

# One-dimensional arrays can be indexed,sliced and iterated over, much like lists
# and other Python sequences

a = np.arange(10)**3
print(a)
print(a[2])
print(a[2:5])
print("\n")

a[:6:2] = 1000
print(a)
print("\n")

print(a[::-1]) # reversed a
print("\n")

for i in a:
    print(i ** (1 / 3.))
print("\n")

# Multidimensional arrays can have one index per axis. These indices are given in a
# tuple separated by commas:

def f(x, y):
    return 10 * x + y

b = np.fromfunction(f, (5, 4), dtype=int)
print(b)
print("\n")

print(b[2, 3])
print("\n")

print(b[0:5, 1]) # each row in the second column of b
print("\n")

print(b[:, 1]) # equivalent to the previous example
print("\n")

print(b[1:3, :]) # each column in the second and third row of b
print("\n")

# When fewer indices are provided than the number of axes, the missing indices are
# considered complete slices:

print(b[-1]) # the last row. Equivalent to b[-1, :]
print("\n")

# The expression within brackets in b [i] is treated as an i followed by as many
# instances of : as needed to represent the remaining axes.

# NumPy also allows you to write this using dots as b[i, ...].

# The dots (...) represent as many colons as needed to produce a complete indexing
# tuple. For example, if x is an array with 5 axes, then:

# • x[1, 2, ...] is equivalent to x[1, 2, :, :, :],
# • x[..., 3] to x[:, :, :, :, 3] and
# • x[4, ..., 5, :] to x[4, :, :, 5, :].

c = np.array([[[ 0, 1, 2], # a 3D array (two stacked 2D arrays)
 [ 10, 12, 13]],
 [[100, 101, 102],
 [110, 112, 113]]])

print(c.shape)
print("\n")

print(c[1, ...]) # same as c[1, :, :] or c[1]
print("\n")

print(c[..., 2]) # same as c[:, :, 2]
print("\n")

# Iterating over multidimensional arrays is done with respect to the first axis:
for row in b:
    print(row)

# However, if one wants to perform an operation on each element in the array, one can
# use the flat attribute which is an iterator over all the elements of the array:

for element in b.flat:
    print(element)

print("\n")

# --- Shape Manipulation ---
# Changing the shape of an array
# An array has a shape given by the number of elements along each axis:

a = np.floor(10 * rg.random((3, 4)))
print(a)
print("\n")

print(a.shape)
print("\n")

# The shape of an array can be changed with various commands. Note that the following
# three commands all return a modified array, but do not change the original array:

print(a.ravel()) # returns the array, flattened
print("\n")

print(a.reshape(6, 2)) # returns the array with a modified shape
print("\n")

print(a.T) # returns the array, transposed
print("\n")

print(a.T.shape)
print("\n")

print(a)
print("\n")

a.resize((2, 6))
print(a)
print("\n")

# --- Shape Manipulation ---

a = np.floor(10 * rg.random((2, 2)))
print(a)
print("\n")

b = np.floor(10 * rg.random((2, 2)))
print(b)
print("\n")

print(np.vstack((a, b)))
print("\n")

print(np.hstack((a, b)))
print("\n")

# --- Splitting one array into several smaller ones ---
a = np.floor(10 * rg.random((2, 12)))
print(a)
print("\n")

print(np.hsplit(a, 3))
print("\n")

print(np.hsplit(a, (3, 4)))
print("\n")




