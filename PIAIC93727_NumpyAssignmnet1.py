{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NgUeapcvmIRh"
   },
   "source": [
    "# **Assignment For Numpy**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yhr3JicwnA-4"
   },
   "source": [
    "Difficulty Level **Beginner**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "hVPqDJ1SnKSh"
   },
   "source": [
    "1. Import the numpy package under the name np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "SePu31zKmgS-"
   },
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "d0gO81krnL7g"
   },
   "source": [
    "2. Create a null vector of size 10 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "2s6o2JPgnSBr"
   },
   "outputs": [],
   "source": [
    "x=np.zeros(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "BJTMK03fnS1w"
   },
   "source": [
    "3. Create a vector with values ranging from 10 to 49"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "qRXxXuWdnj1M"
   },
   "outputs": [],
   "source": [
    "x=np.arange(10,49)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "n63lzg5Onkcn"
   },
   "source": [
    "4. Find the shape of previous array in question 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "1FueBdqPn_q7"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(39,)\n"
     ]
    }
   ],
   "source": [
    "print(x.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "AzMlQj1MoAVe"
   },
   "source": [
    "5. Print the type of the previous array in question 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "AO9PYmdWoE1L"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "int32\n"
     ]
    }
   ],
   "source": [
    "print(x.dtype)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0qaKS13Yon-U"
   },
   "source": [
    "6. Print the numpy version and the configuration\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "T3wY24e1oorh"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.16.5\n",
      "mkl_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/MWAQAS/Anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/MWAQAS/Anaconda3\\\\Library\\\\include']\n",
      "blas_mkl_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/MWAQAS/Anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/MWAQAS/Anaconda3\\\\Library\\\\include']\n",
      "blas_opt_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/MWAQAS/Anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/MWAQAS/Anaconda3\\\\Library\\\\include']\n",
      "lapack_mkl_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/MWAQAS/Anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/MWAQAS/Anaconda3\\\\Library\\\\include']\n",
      "lapack_opt_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/MWAQAS/Anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/MWAQAS/Anaconda3\\\\Library\\\\include']\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(np.__version__)\n",
    "print(np.show_config())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZyUUiQf5oyvE"
   },
   "source": [
    "7. Print the dimension of the array in question 3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "id": "pLXfuIqIo0vq"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1\n"
     ]
    }
   ],
   "source": [
    "print(x.ndim)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JYVMuFrqpBdV"
   },
   "source": [
    "8. Create a boolean array with all the True values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "id": "3apZsISzpFKR"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ True  True]\n",
      " [ True  True]]\n"
     ]
    }
   ],
   "source": [
    "x=np.array([[True, True],\n",
    "            [True, True]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4zbBooWZpPBU"
   },
   "source": [
    "9. Create a two dimensional array\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "KfPQEiVIpdTo"
   },
   "outputs": [],
   "source": [
    "x1=np.array([[1, 2],\n",
    "            [3, 4]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "do9wPAFbpqC7"
   },
   "source": [
    "10. Create a three dimensional array\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "id": "ZCO_q-KZqAeW"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3\n",
      "[[[0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0.]]\n",
      "\n",
      " [[0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0.]]]\n"
     ]
    }
   ],
   "source": [
    "x2=np.array([[[ 0,  1,  2],\n",
    "        [ 3,  4,  5],\n",
    "        [ 6,  7,  8]],\n",
    "       [[ 9, 10, 11],\n",
    "        [12, 13, 14],\n",
    "        [15, 16, 17]]])\n",
    "#print(x2.ndim)\n",
    "\n",
    "#or\n",
    "x3=np.zeros((2,3,4))\n",
    "\n",
    "# print(x3.ndim)\n",
    "# print(x3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "a4iysJ87qEeb"
   },
   "source": [
    "Difficulty Level **Easy**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "DehgqszSqjY6"
   },
   "source": [
    "11. Reverse a vector (first element becomes last)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "id": "tfevmc4VqkWu"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Orignal vector: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14]\n",
      "Reverse vector: [14 13 12 11 10  9  8  7  6  5  4  3  2  1]\n"
     ]
    }
   ],
   "source": [
    "x4=np.arange(1,15)\n",
    "print(\"Orignal vector:\",x4)\n",
    "x4=x4[::-1]\n",
    "print(\"Reverse vector:\",x4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yoATXS-qqk_q"
   },
   "source": [
    "12. Create a null vector of size 10 but the fifth value which is 1 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "id": "AheZ32Owqpte"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]\n"
     ]
    }
   ],
   "source": [
    "x5=np.zeros(10)\n",
    "x5[5]=1\n",
    "print(x5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RZRzOBbFsY0w"
   },
   "source": [
    "13. Create a 3x3 identity matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "id": "va2ou0BHsvEj"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1. 0. 0.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]]\n"
     ]
    }
   ],
   "source": [
    "x6=np.identity(3)\n",
    "print(x6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lnN5drkUs6o2"
   },
   "source": [
    "14. arr = np.array([1, 2, 3, 4, 5]) \n",
    "\n",
    "---\n",
    "\n",
    " Convert the data type of the given array from int to float "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "id": "59EIQt6otKUB"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "int32\n",
      "float32\n"
     ]
    }
   ],
   "source": [
    "x7 = np.array([1, 2, 3, 4, 5])\n",
    "print(x7.dtype)\n",
    "x7=x7.astype(np.float32)\n",
    "print(x7.dtype)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fSL2AKJetTes"
   },
   "source": [
    "15. arr1 = np.array([[1., 2., 3.],\n",
    "\n",
    "                    [4., 5., 6.]])  \n",
    "                      \n",
    "    arr2 = np.array([[0., 4., 1.],\n",
    "     \n",
    "                   [7., 2., 12.]])\n",
    "\n",
    "---\n",
    "\n",
    "\n",
    "Multiply arr1 with arr2\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "id": "7VDXgRQDuJNV"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.  8.  3.]\n",
      " [28. 10. 72.]]\n"
     ]
    }
   ],
   "source": [
    "arr1 = np.array([[1., 2., 3.],\n",
    "                 [4., 5., 6.]])  \n",
    "arr2 = np.array([[0., 4., 1.],\n",
    "                 [7., 2., 12.]])\n",
    "arr3=np.multiply(arr2, arr1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "a8yJ438-uKni"
   },
   "source": [
    "16. arr1 = np.array([[1., 2., 3.],\n",
    "                    [4., 5., 6.]]) \n",
    "                    \n",
    "    arr2 = np.array([[0., 4., 1.], \n",
    "                    [7., 2., 12.]])\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "Make an array by comparing both the arrays provided above"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {
    "id": "3MPDaAlYueF3"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "False\n"
     ]
    }
   ],
   "source": [
    "arr1 = np.array([[1., 2., 3.],\n",
    "                 [4., 5., 6.]]) \n",
    "arr2 = np.array([[0., 4., 1.],\n",
    "                 [7., 2., 12.]])\n",
    "arr3=np.array_equal(arr1, arr2)\n",
    "print(arr3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3xiD0eJluewk"
   },
   "source": [
    "17. Extract all odd numbers from arr with values(0-9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {
    "id": "J_qbt_EuvM7l"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 3 5 7 9]\n"
     ]
    }
   ],
   "source": [
    "arr4=np.arange(0,10)\n",
    "arr5=arr4[arr4%2==1]\n",
    "print(arr5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zpg5cyLsvPoZ"
   },
   "source": [
    "18. Replace all odd numbers to -1 from previous array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {
    "id": "rXUV1-ULvQdd"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-1\n"
     ]
    }
   ],
   "source": [
    "arr6 = arr5[arr5%2==1]= -1\n",
    "print(arr6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FWpHTWTDvjvi"
   },
   "source": [
    "19. arr = np.arange(10)\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "Replace the values of indexes 5,6,7 and 8 to **12**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {
    "id": "ensZ3lqwvlSb"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0  1  2  3  4 12 12 12 12  9]\n"
     ]
    }
   ],
   "source": [
    "arr7 = np.arange(10)\n",
    "arr8=arr7[5:9]= 12\n",
    "print(arr7)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ib-vBffAv0zy"
   },
   "source": [
    "20. Create a 2d array with 1 on the border and 0 inside"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {
    "id": "RmzcjNYrwDrM"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original array:\n",
      "[[0. 0. 0.]\n",
      " [0. 0. 0.]\n",
      " [0. 0. 0.]]\n",
      "array with border 1 and zero inside\n",
      "[[1. 1. 1. 1. 1.]\n",
      " [1. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 1.]\n",
      " [1. 1. 1. 1. 1.]]\n"
     ]
    }
   ],
   "source": [
    "x8 = np.zeros((3,3))\n",
    "print(\"Original array:\")\n",
    "print(x8)\n",
    "print(\"array with border 1 and zero inside\")\n",
    "x = np.pad(x8, pad_width=1, mode='constant', constant_values=1)\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "E1diIFSEwEmh"
   },
   "source": [
    "Difficulty Level **Medium**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "eGLZmNb_wKb4"
   },
   "source": [
    "21. arr2d = np.array([[1, 2, 3],\n",
    "\n",
    "                    [4, 5, 6], \n",
    "\n",
    "                    [7, 8, 9]])\n",
    "\n",
    "---\n",
    "\n",
    "Replace the value 5 to 12"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {
    "id": "VdUnsUOZwJFz"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1  2  3]\n",
      " [ 4 12  6]\n",
      " [ 7  8  9]]\n"
     ]
    }
   ],
   "source": [
    "arr2d = np.array([[1, 2, 3],\n",
    "                  [4, 5, 6], \n",
    "                  [7, 8, 9]])\n",
    "print(np.where(arr2d==5,12,arr2d))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TEbFhlPZxaKO"
   },
   "source": [
    "22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])\n",
    "\n",
    "---\n",
    "Convert all the values of 1st array to 64\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {
    "id": "eYzephU8xmsg"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[64 64 64]\n",
      "  [ 4  5  6]]\n",
      "\n",
      " [[ 7  8  9]\n",
      "  [10 11 12]]]\n"
     ]
    }
   ],
   "source": [
    "arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])\n",
    "print(np.where(arr3d<4,64,arr3d))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vZj0Ndvnx6f0"
   },
   "source": [
    "23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {
    "id": "pM4e9YfdyHSv"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1]]\n"
     ]
    }
   ],
   "source": [
    "array1 = np.array([[0, 1],\n",
    "                [2, 3],\n",
    "                [4, 5],\n",
    "                [6, 7],\n",
    "                [8, 9]])\n",
    "slice=array1[:1,:2]\n",
    "print(slice)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "C3dkW0lZyVps"
   },
   "source": [
    "24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {
    "id": "IrrzK2xgygqV"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[3]\n"
     ]
    }
   ],
   "source": [
    "array1 = np.array([[0, 1],\n",
    "                [2, 3],\n",
    "                [4, 5],\n",
    "                [6, 7],\n",
    "                [8, 9]])\n",
    "slice2=array1[1:2, 1]\n",
    "print(slice2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "G3dLwOO9yyrE"
   },
   "source": [
    "25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {
    "id": "5Bx4fqsLyqGD"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[4 5]]\n"
     ]
    }
   ],
   "source": [
    "array1 = np.array([[0, 1],\n",
    "                [2, 3],\n",
    "                [4, 5],\n",
    "                [6, 7],\n",
    "                [8, 9]])\n",
    "slice3=array1[2:3, :2]\n",
    "print(slice3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CQ9YST5jy0IL"
   },
   "source": [
    "26. Create a 10x10 array with random values and find the minimum and maximum values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {
    "id": "lhJPugjQzG6W"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original Array:\n",
      "[[0.19031819 0.5546566  0.61582667 0.80587212 0.15273602 0.83005206\n",
      "  0.06928739 0.39087319 0.89841196 0.186443  ]\n",
      " [0.87643748 0.3952413  0.51213069 0.25267678 0.61854962 0.35229381\n",
      "  0.84048222 0.71225098 0.79530917 0.93526598]\n",
      " [0.92817497 0.02105157 0.50491061 0.63875775 0.49173032 0.02158417\n",
      "  0.42546111 0.29042077 0.49652479 0.97608001]\n",
      " [0.2799285  0.35255808 0.64297774 0.64507418 0.29480809 0.08566505\n",
      "  0.004164   0.8386548  0.94876123 0.85769557]\n",
      " [0.95060138 0.79218974 0.92974204 0.97169203 0.73364245 0.14323869\n",
      "  0.51914057 0.98226194 0.89015714 0.78044058]\n",
      " [0.25087852 0.44088033 0.74116689 0.35133588 0.90821601 0.94099232\n",
      "  0.39368764 0.25372756 0.00960172 0.70486421]\n",
      " [0.08387545 0.96877967 0.65766413 0.90369753 0.25673892 0.29638571\n",
      "  0.63554086 0.49089858 0.14394583 0.31308731]\n",
      " [0.81906402 0.20974763 0.3606962  0.7712671  0.84892728 0.19537777\n",
      "  0.42057339 0.18616368 0.05648483 0.09213821]\n",
      " [0.64288697 0.53470593 0.30864165 0.40758565 0.31729778 0.91164581\n",
      "  0.32210826 0.11045542 0.32076209 0.33584862]\n",
      " [0.85272437 0.03019491 0.49692218 0.59983631 0.59780872 0.98245679\n",
      "  0.46139379 0.20477165 0.77146154 0.23327494]]\n",
      "Minimum and Maximum Values:\n",
      "0.004163997970374256 0.9824567943284307\n"
     ]
    }
   ],
   "source": [
    "X = np.random.random((10,10))\n",
    "print(\"Original Array:\")\n",
    "print(X) \n",
    "xmin, xmax = X.min(), X.max()\n",
    "print(\"Minimum and Maximum Values:\")\n",
    "print(xmin, xmax)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cAgDt6KazHk6"
   },
   "source": [
    "27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "---\n",
    "Find the common items between a and b\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "kBXZanxIzs_F"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2 4]\n"
     ]
    }
   ],
   "source": [
    "a = np.array([1,2,3,2,3,4,3,4,5,6]) \n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "print(np.intersect1d(a, b))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lF9IzWOAztpD"
   },
   "source": [
    "28. a = np.array([1,2,3,2,3,4,3,4,5,6])\n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "\n",
    "---\n",
    "Find the positions where elements of a and b match\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {
    "id": "JR-mHQLH0HY_"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 5]\n",
      "[1 5]\n",
      "[2 4]\n"
     ]
    }
   ],
   "source": [
    "a = np.array([1,2,3,2,3,4,3,4,5,6]) \n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "ab, a_ind, b_ind=np.intersect1d(a, b, return_indices=True)\n",
    "print(a_ind)\n",
    "print(b_ind)\n",
    "print(ab)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mMDiefpH0H1x"
   },
   "source": [
    "29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)\n",
    "\n",
    "---\n",
    "Find all the values from array **data** where the values from array **names** are not equal to **Will**\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "gfc1Lpf81ZSR"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.17268794  0.82007511 -1.23919306  0.7839802 ]\n",
      " [ 2.57879489  0.92526884  1.55303369 -0.62627541]\n",
      " [ 0.67362153 -1.08431519 -2.05555466 -0.22339212]\n",
      " [ 0.43684443 -1.2033156   0.63855659  0.55071161]\n",
      " [-0.70872105 -1.09573012  1.41041293 -0.84962444]]\n"
     ]
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "data = np.random.randn(7, 4)\n",
    "res = data[names != 'Will']\n",
    "print(res)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "52zEjDMf1aXS"
   },
   "source": [
    "30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)\n",
    "\n",
    "---\n",
    "Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "8RYG0kLO1mHw"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-2.55002595  0.51142918 -1.3355147  -0.42981955]\n",
      " [ 1.39725185  0.46750126 -0.05062167  0.97766302]]\n"
     ]
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "data = np.random.randn(7, 4)\n",
    "nam = (names != 'Will') & (names != 'Joe')\n",
    "res2 = data[nam]\n",
    "print(res2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Q7hjf2bd2dCY"
   },
   "source": [
    "Difficulty Level **Hard**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b48g7aRA2jVM"
   },
   "source": [
    "31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "EbwfSCrW2f9_"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 7.32469598  1.47771407  2.85003553]\n",
      " [ 7.59904773 10.19603908  7.47107425]\n",
      " [ 9.0360369  12.08800499  7.46093794]\n",
      " [ 4.43448652 11.04198702  8.56351297]\n",
      " [ 6.94437921  9.18459039  5.62771748]]\n"
     ]
    }
   ],
   "source": [
    "rand_arr = np.random.uniform(1,15, size=(5,3))\n",
    "print(rand_arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Ei_nDHtO3qBk"
   },
   "source": [
    "32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "id": "phOxVpBM4M6D"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[14.62951368  8.63195438 10.18688898  4.00419377]\n",
      "  [14.2312554   2.17749268 12.93691621 13.29726867]]\n",
      "\n",
      " [[14.3082953   4.96787257  3.49129191 12.66089232]\n",
      "  [ 5.28173388  1.39924567 15.591163   14.91612123]]]\n"
     ]
    }
   ],
   "source": [
    "rand_arr1 = np.random.uniform(1,16, size=(2,2,4))\n",
    "print(rand_arr1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uyiQaMjA4d_x"
   },
   "source": [
    "33. Swap axes of the array you created in Question 32"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "w2m9p8064VtR"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[14.62951368, 14.3082953 ],\n",
       "        [14.2312554 ,  5.28173388]],\n",
       "\n",
       "       [[ 8.63195438,  4.96787257],\n",
       "        [ 2.17749268,  1.39924567]],\n",
       "\n",
       "       [[10.18688898,  3.49129191],\n",
       "        [12.93691621, 15.591163  ]],\n",
       "\n",
       "       [[ 4.00419377, 12.66089232],\n",
       "        [13.29726867, 14.91612123]]])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.swapaxes(rand_arr1,0,2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2dlU_yUR4mVZ"
   },
   "source": [
    "34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "id": "tmBabEJ-5JgB"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.61591606, 0.        , 0.78832072, 0.78093563, 0.        ,\n",
       "       0.6733114 , 0.81405127, 0.73821938, 0.66897194, 0.        ])"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Z = np.random.random((10))\n",
    "sqtz=np.sqrt(Z)\n",
    "sqtz[sqtz<0.5]=0\n",
    "sqtz"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SNAM4dpu5RKA"
   },
   "source": [
    "35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "id": "3pxebU2b5q9Z"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "array one:\n",
      "[0.37554911 0.38672534 0.49691766 0.18295417 0.87409008 0.64137116\n",
      " 0.75542867 0.45864117 0.62249531 0.08240682 0.52935637 0.94297748]\n",
      "array two:\n",
      "[0.68327925 0.59274549 0.96552279 0.04517573 0.98296941 0.9208245\n",
      " 0.00234394 0.28238712 0.11607695 0.77201773 0.31792778 0.63663404]\n",
      "array three maximum:\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(0.942977480574871, 0.9829694132076576)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "array1 = np.random.random(12)\n",
    "array2 = np.random.random(12)\n",
    "print(\"array one:\")\n",
    "print(array1)\n",
    "print(\"array two:\")\n",
    "print(array2)\n",
    "print(\"array three maximum:\")\n",
    "array3 = np.max(array1), np.max(array2)  \n",
    "array3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fz_bpqm28qmD"
   },
   "source": [
    "36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "\n",
    "---\n",
    "Find the unique names and sort them out!\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "id": "hiSNMgcY87Ey"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Bob' 'Joe' 'Will']\n"
     ]
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "resnames=np.unique(names)\n",
    "sortnames=np.sort(resnames)\n",
    "print(sortnames)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dguMNzob870H"
   },
   "source": [
    "37. a = np.array([1,2,3,4,5])\n",
    "b = np.array([5,6,7,8,9])\n",
    "\n",
    "---\n",
    "From array a remove all items present in array b\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "id": "eCNJXPvK9IAr"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([1,2,3,4,5]) \n",
    "b = np.array([5,6,7,8,9])\n",
    "c=np.setdiff1d(a,b)\n",
    "c"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gz4Dvd4b_Akh"
   },
   "source": [
    "38.  Following is the input NumPy array delete column two and insert following new column in its place.\n",
    "\n",
    "---\n",
    "sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) \n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "newColumn = numpy.array([[10,10,10]])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "id": "lLWAJWJJ_I3d"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[34 10 73]\n",
      " [82 10 12]\n",
      " [53 10 66]]\n"
     ]
    }
   ],
   "source": [
    "sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])\n",
    "newColumn = np.array([[10,10,10]]).reshape(3,1)\n",
    "res1=np.delete(sampleArray, 1, 1)\n",
    "res = np.insert(res1, [1], newColumn, axis=1)\n",
    "print(res)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1lIfmbQt_J_W"
   },
   "source": [
    "39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])\n",
    "\n",
    "\n",
    "---\n",
    "Find the dot product of the above two matrix\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "id": "fV2-vXcR_vmu"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 28.  64.]\n",
      " [ 67. 181.]]\n"
     ]
    }
   ],
   "source": [
    "x = np.array([[1., 2., 3.], [4., 5., 6.]]) \n",
    "y = np.array([[6., 23.], [-1, 7], [8, 9]])\n",
    "z = np.dot(x,y)\n",
    "print(z)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5U-4odUw_wP0"
   },
   "source": [
    "40. Generate a matrix of 20 random values and find its cumulative sum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "id": "_oOHdiYEAF8N"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.07624187 0.41384912 1.04118907 1.71999155 1.92346842 2.36469676\n",
      " 2.85014835 2.93724531 3.92935358]\n"
     ]
    }
   ],
   "source": [
    "matrix=np.random.rand(3,3)\n",
    "csum=np.cumsum(matrix)\n",
    "print(csum)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "name": "Numpy-Assignment-PIAIC-Batch35-Q2.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
