import numpy as np 

# create an array 
a = np.array([1,2,3,4,5,6])

# print dimension and shape of array 
print('the dimension of array: ',a.ndim, '\n')
print('the shape of array: ',a.shape,'\n')
print('data type of array: ', a.dtype)

# transform the array a into a 2D array
b = a.reshape((3,2))
print("After transformation: \n", b)

# change the value of b[1][0] and a is also changed
b[1][0] = 77
print("After change the value b[1][0]:\n",b)

print("clip values (0,4)", a.clip(0,4))

# elements with different data types
c = np.array([1,"ssdfsdf",set([12,3])])

print(c)

