
# coding: utf-8

# In[17]:

get_ipython().magic('pylab')


# In[24]:

def C_z(z_values, f_values, z): #z_values are the points at which we know the value of our function, f_values the values of our function at those points, z the point where we want to evaluate the continued fraction C_z;

    N = z_values.size
    
    g_mat = np.zeros([N,N]) #this is the matrix whose elements are: g_mat[i,j] = g_i(z_j), cfr eqn. (A2). Note that only diagonal and upper diagonal elements matter. The coefficients a_i will be found on the diagonal: a_i = g_mat[i,i];
    """
    for i in range(N):     #filling the first row with the initial data;
        g_mat[0,i] = f_values[i]"""
    #can be done diretly:
    g_mat[0]=f_values
    
    for i in range(N-1): #filling the rest of the matrix using the recursion relation;
        for j in range(N-(i+1)):
            g_mat[i+1,i+1+j] = (g_mat[i,i]-g_mat[i,j+i+1])/((z_values[j+1+i]-z_values[i])*g_mat[i,j+i+1])

    """
    a_coeff = np.zeros(N)
    
    for i in range (N):
        a_coeff[i] = g_mat[i,i]
    """
    #also more effenciently:
    a_coeff = np.diag(g_mat)

    #now we evaluate the contiued fraction at point z with previously computed coeeficients a_i;

    A = np.zeros(N+1)
    B = np.zeros(N+1)
    A[1] = a_coeff[0]
    B[0] = 1
    B[1] = 1
    
    for i in range(N-1):
        A[i+2] = A[i+1] + (z-z_values[i])*a_coeff[i+1]*A[i]
        B[i+2] = B[i+1] + (z-z_values[i])*a_coeff[i+1]*B[i]

    return A[N]/B[N]

#try: if C_z is computed for one of the given points, it returns the expected result: the code could be actually working properly!


points = np.array([11,2,3,12,22,123,9,15,55])
values = np.array([11,21,3,7,1,12,43,54,88])

print(C_z(points,values,2))



# In[ ]:



