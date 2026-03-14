import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A=np.array(A)
    n,m= A.shape


    a=np.zeros((m,n), dtype=A.dtype)
    for i in range(n):
        for j in range(m):
            
            a[j,i]=A[i,j]

    return a
    
    
