#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[11]:


import numpy as np

arr_1d = np.array([0,1,2,3,4,5,6,7,8,9])
arr_2d = np.reshape(arr_1d,(2,5))
print(arr_2d)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[6]:


arr_a=np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr_b=np.array([[1,1,1,1,1],[1,1,1,1,1]])
arr_v=np.vstack((arr_a,arr_b))
print(arr_v)


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[7]:


#In which i can use above array
arr_h=np.hstack((arr_a,arr_b))
print(arr_h)


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[8]:


#in which i can use above Q 1 resultant array arr_2d
arr_farr1d=arr_2d.flatten()
print(arr_farr1d)


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[26]:


hd_arr = np.array([[ 0, 1, 2],
               [ 3, 4, 5],
               [ 6, 7, 8],
               [9, 10, 11],
               [12, 13, 14]])
print(hd_arr.ndim)
d1_arr = hd_arr.ravel()
print(d1_arr)
print(d1_arr.ndim)


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[28]:


newarr=d1_arr.reshape(5,3)
newarr


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[32]:


arr1=np.random.uniform(1,15, size=(5,5))
arr2=np.square(arr1)
arr2


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[34]:


arr3=np.random.uniform(1,15, size=(5,6))
arr4=np.mean(arr3)
arr4


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[37]:


arr4=np.std(arr3)
arr4


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[35]:


arr6=np.median(arr3)
arr6


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[36]:


print(arr3)
print("transpose")
arr7=np.transpose(arr3)
arr7


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[3]:


arr8=np.random.uniform(1,15, size=(4,4))
print(arr8)
print("diagonals \n")
arr9=np.diagonal(arr8)
print(arr9)
print("sum of diagonals \n")
sum=arr9.sum()
print(sum)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[5]:


dtmnt_arr=np.linalg.det(arr8)
print(dtmnt_arr)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[7]:


prc_arr=np.arange(10)
prc_5=np.percentile(prc_arr,5)
prc_95=np.percentile(prc_arr,95)
print("Array:\n",prc_arr)
print("5th percentile :",prc_5)
print("95th percentile :",prc_95)


# ## Question:15

# ### How to find if a given array has any null values?

# In[10]:


Array=np.array([1,2,np.nan,3])
chk_null=np.isnan(Array)
print("Array :\n",Array)
print("Checking if array contain null value :\n",chk_null)

