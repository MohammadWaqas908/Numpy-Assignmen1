#!/usr/bin/env python
# coding: utf-8

# # Task no.1

# In[2]:


import numpy as np
def function1():
    #create 3D array (3,3,3)
    #must data type should have float64
    #array value should be satart from 10 and end with 36 (both included)
    # Hint: dtype, reshape
    x=np.arange(1,13).reshape(6,2) #code
    return x 
function1()


# # Task no.2

# In[16]:


def function2():
    x=np.arange(10,37, dtype="float64").reshape(3,3,3)
    print(x.dtype)
    return x
    """
    Expected: output
array([[[10., 11., 12.],
        [13., 14., 15.],
        [16., 17., 18.]],

       [[19., 20., 21.],
        [22., 23., 24.],
        [25., 26., 27.]],

       [[28., 29., 30.],
        [31., 32., 33.],
        [34., 35., 36.]]])    
    """
function2()


# # Task 3

# In[19]:


def function3():
    #extract those numbers from given array. those are must exist in 5,7 Table
    #example [35,70,105,..]
    a = np.arange(1, 100*10+1).reshape((100,10))
    condition=(np.mod(a,5)==0) & (np.mod(a,7)==0)
    x = a[condition] #code
    return x
    """
    Expected Output:
     [35,  70, 105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455,
       490, 525, 560, 595, 630, 665, 700, 735, 770, 805, 840, 875, 910,
       945, 980] 
    """
function3()


# # Task 4

# In[33]:


def function4():
    #Swap columns 1 and 2 in the array arr.
    arr = np.arange(9).reshape(3,3)
    return arr[:,[1, 0,2]]
    """
    Expected Output:
          array([[1, 0, 2],
                [4, 3, 5],
                [7, 6, 8]])
    """
function4()


# # Task 5

# In[35]:


def function5():
    #Create a null vector of size 20 with 4 rows and 5 columns with numpy function
    z=np.zeros(20,dtype="int").reshape(4,5) #code
    return z
    """
    Expected Output:
          array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]])
    """
function5()


# # Task 6

# In[38]:


def function6():
    # Create a null vector of size 10 but the fifth and eighth value which is 10,20 respectively
    arr=np.zeros(10, dtype="int")
    arr[5]=10
    arr[8]=20
    return arr #code
function6()


# # Task 7

# In[41]:


def function7():
    #  Create an array of zeros with the same shape and type as X. Dont use reshape method
    x = np.arange(4, dtype=np.int64)
    return np.zeros(4, dtype="int64") #code
    """
    Expected Output:
          array([0, 0, 0, 0], dtype=int64)
    """
function7()


# # Task 8

# In[53]:


def function8():
    # Create a new array of 2x5 uints, filled with 6.
    x=np.full((2,5),6).astype("uint32") #code 
    return x
    """
     Expected Output:
              array([[6, 6, 6, 6, 6],
                     [6, 6, 6, 6, 6]], dtype=uint32)
     """
function8()


# # Task 9

# In[56]:


def function9():
    # Create an array of 2, 4, 6, 8, ..., 100.
    a=np.arange(2,101,2) #code
    return a
     """
     Expected Output:
              array([  2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,
                    28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,
                    54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,
                    80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100])
     """
function9()


# # Task 10

# In[61]:


def function10():
    # Subtract the 1d array brr from the 2d array arr, such that each item of brr subtracts from respective row of arr.
    arr = np.array([[3,3,3],[4,4,4],[5,5,5]])
    brr = np.array([1,2,3])
    subt =arr-brr[:,None] #code
    return subt
    """
     Expected Output:
               array([[2 2 2]
                      [2 2 2]
                      [2 2 2]])
     """
function10()


# # Task 11

# In[8]:


def function11():
    # Replace all odd numbers in arr with -1 without changing arr.
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    arr[arr%2==1]= -1 #code
    ans=arr #code
    return ans
     """
     Expected Output:
              array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
     """ 
function11()


# # Task 12

# In[9]:


def function12():
    # Create the following pattern without hardcoding. Use only numpy functions and the below input array arr.
    # HINT: use stacking concept
    arr = np.array([1,2,3])
    ans = np.hstack((np.repeat(arr,3),arr,arr,arr)) #code
    return ans
    """
     Expected Output:
              array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
     """ 
function12()


# # Task 13

# In[11]:


def function13():
    # Set a condition which gets all items between 5 and 10 from arr.
    arr = np.array([2, 6, 1, 9, 10, 3, 27])
    ans = arr[(arr>5)&(arr<10)] #code
    return ans
    """
     Expected Output:
              array([6, 9])
     """
function13()


# # Task 14

# In[15]:


def function14():
        # Create an 8X3 integer array from a range between 10 to 34 such that the difference between each element is 1 and then Split the array into four equal-sized sub-arrays.
    # Hint use split method
    arr = np.arange(10, 34, 1).reshape(8,3) # reshape code
    ans = np.split(arr,4) #code
    return ans
    """
     Expected Output:
       [array([[10, 11, 12],[13, 14, 15]]), 
        array([[16, 17, 18],[19, 20, 21]]), 
        array([[22, 23, 24],[25, 26, 27]]), 
        array([[28, 29, 30],[31, 32, 33]])]
     """ 
function14()


# # Task 15

# In[16]:


def function15():
    #Sort following NumPy array by the second column
    arr = np.array([[ 8,  2, -2],[-4,  1,  7],[ 6,  3,  9]])
    arr[:3,1:2] = np.sort(arr[:3,1:2],axis=0)
    ans=arr #code
    return ans 
    """
     Expected Output:
           array([[-4,  1,  7],
                   [ 8,  2, -2],
                   [ 6,  3,  9]])
     """ 
function15()


# # Task 16

# In[17]:


def function16():
    #Write a NumPy program to join a sequence of arrays along depth.
    x = np.array([[1], [2], [3]])
    y = np.array([[2], [3], [4]])
    ans = np.hstack([x,y]) #code
    return ans
    """
     Expected Output:
                [[[1 2]]

                 [[2 3]]

                 [[3 4]]]
    """ 
function16()


# # Task 17

# In[21]:


def function17():
    # replace numbers with "YES" if it divided by 3 and 5
    # otherwise it will be replaced with "NO"
    # Hint: np.where
    arr = np.arange(1,10*10+1).reshape((10,10))
    return np.where((arr%3==0)&(arr%5==0),"YES","NO")
#Excpected Out
"""
array([['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO']],
      dtype='<U3')
"""
function17()


# # Task 18

# In[23]:


def function18():
    # count values of "students" are exist in "piaic"

    piaic = np.arange(100)
    students = np.array([5,20,50,200,301,7001])
    x = len(set(piaic)&set(students)) #code
    return x

    #Expected output: 3
function18()


# # Task 19

# In[24]:


def function19():
    #Create variable "X" from 1,25 (both are included) range values
    #Convert "X" variable dimension into 5 rows and 5 columns
    #Create one more variable "W" copy of "X" 
    #Swap "W" row and column axis (like transpose)
    # then create variable "b" with value equal to 5
    # Now return output as "(X*W)+b:
    
    X = np.arange(1,26).reshape(5,5) #code
    W = np.copy(X)   #code
    W = W.T          #code
    b = 5            #code
    output = (X*W)+b #code
    return output
    #expected output
    """
    array([[  6,  17,  38,  69, 110],
       [ 17,  54, 101, 158, 225],
       [ 38, 101, 174, 257, 350],
       [ 69, 158, 257, 366, 485],
       [110, 225, 350, 485, 630]])
    """
function19()


# # Task 20

# In[26]:


def function20():
    #apply fuction "abc" on each value of Array "X"
    x = np.arange(1,11)
    def xyz(x):
        return x*2+3-2

    return xyz(x) #code
#Expected Output: array([ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21])
function20()


# In[ ]:




