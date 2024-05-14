import numpy as np

# NOTE THAT NP ARRAYS ARE N X K, where N is the amount of rows, K is amout of cols
# ALSO NOTE that for ML, N means number of samples, K means number of features per sample

""" a_mul = np.array([[1,2,3], 
                  [4,5,6],
                  [7,8,9]])

print(a_mul.shape) # n x k 
print(a_mul.ndim) # highest depth of lists
print(a_mul.size) # number of elements
print(a_mul.dtype) # int32 """

""" a = np.array([[1,2,3],
              [4,"Hello",6],
              [7,8,9]])

d = {'1': 'A'}

c = np.array([1,d,3])
print(c.dtype) # object

print(a.dtype) # <U11 - basically a string datatype
print(type(a[0][0])) # prints integer as a numpy string, one string changes other elements in np arrays """

""" # Filling arrays
a = np.full((2,3,4),9) # a list of two 3 x 4 lists 

a = np.zeros((10,5,2)) # or ones 

a = np.empty((5,5,5)) # allocates memory without placing a value; more cpp-based

x_values = np.arange(0,1000,5) # from 0 to 1000, step size of 5
print(x_values)

x_val = np.linspace(0, 1000, 2) # Evenly distributes values given the amount specified

print(np.nan) # not a number
print(np.inf) # infinity, occurs with division by zero
print(np.isnan(np.sqrt(-1))) # false, complex number  """

""" # Mathematical Operations
l1 = [1,2,3,4,5]
l2 = [6,7,8,9,0]

a1 = np.array(l1)
a2 = np.array(l2)

print(l1 * 5) # Repeats list 5 times
print(a1 * 5) # Scalar mult. on a matrix

#print(l1 + l2) # Doesn't work
print(a1 + a2) # Matrix addition, have to be the same n x k dimensions of course

a = np.array([[1,2,3],
             [4,5,6]])

print(np.sqrt(a)) # sin, cos, tan, arctan, arcsin, arccos, exp, log, log2, log10, etc. check docs """

""" # Array Methods

a = np.array([1,2,3])

a = np.append(a, [7,8,9]) # have to put a equal to this, or it won't persist
a = np.insert(a, 3, [4,5,6]) # provide index of insertion

#np.delete(a, 1) # No axis, the item in the 1st index is deleted
b = np.array([[1,2,3],
              [4,5,6]])

b = np.delete(b, 1, 0) # Gets rid of second row, 0 is row, 1 is column, so row 1 is deleted 

print(b) """

""" # Structuring Methods
a = np.array([[1,2,3,4,5], # 4 x 5
              [6,7,8,9,10],
              [11,12,13,14,15],
              [16,17,18,19,20]])

# Reshape needs to be assigned, resize persists
print(a.reshape((5,4))) # Has to be compatible, same number of elements
print(a.reshape((20,))) # One list, 20 elements
print(a.reshape((20, 1))) # 20 ists, 1 element
print(a.reshape((2, 2, 5))) # Two 2 x 5 lists, still 20 elements

print(a.flatten()) # puts arrays side by size, changes array
print(a.ravel()) # does not change array

var = [v for v in a.flat]
print(var)

print(a.transpose()) # Matrix transposition, rows become columns and vice versa
print(a.swapaxes(0,1)) # Can specify the exact axes you want to transpose """

""" # More Methods
a1 = np.array([[1,2,3,4,5], 
              [6,7,8,9,10]])

a2 = np.array([[11,12,13,14,15],
              [16,17,18,19,20]])

a = np.concatenate((a1, a2), axis=1) # 0 for rows, 1 for columns

a3 = np.stack((a1, a2)) # combines by adding a new dimension

b = np.array([[1,2,3,4,5], 
              [6,7,8,9,10],
              [11,12,13,14,15],
              [16,17,18,19,20]])

print(np.split(b, 2, axis=0)) # split rows into 2 arrays
print(np.split(b, 4, axis=0)) # split rows into 4 arrays

c = np.array([[1,2,3,4,5], 
              [6,7,8,9,10],
              [11,12,13,14,15],
              [16,17,18,19,20]])

print(c.min())
print(c.max())
print(c.mean())
print(c.std())
print(c.sum())
print(np.median(a)) """


""" # NP Random
numbers = np.random.randint(0, 100, size=(2,3,4))
numbers2 = np.random.binomial(10, p=0.5, size=(5,10))
numbers2 = np.random.normal(loc=170, scale=15, size=(5,10))
numbers2 = np.random.choice([10,20,30,40,50], size=(5,10)) # choose one of these numbers
print(numbers) """

# Importing/Exporting
# a = np.array([[1,2,3,4,5], 
#               [6,7,8,9,10],
#               [11,12,13,14,15],
#               [16,17,18,19,20]])

# np.save('myarray.npy', a)

# a = np.load('myarray.npy')
# print(a)

# np.savetxt('myarray.csv', a, delimiter=',')

a = np.loadtxt('myarray.csv', delimiter=',') # good for loading other people's datasets
print(a)