
# coding: utf-8

# In[1]:

get_ipython().magic('pylab inline')
#loading ipython cython extension
get_ipython().magic('load_ext Cython')


# Reimplementing all in cython for speed optimisation:
# For this every function has to be in the same block for cython magic (%%cython) to work

# In[60]:

get_ipython().run_cell_magic('cython', '', "\ncdef double beta = 100. # inverse temperature\ncdef double t = 1. # energy scale for cubic lattice\ncdef double D = 1. # energy scale for bethe lattice\nimport numpy as np\ncimport numpy as np\n#definition of FourierTransforms\ncdef matsubara_fft(np.ndarray G_tau):\n    cdef int N = G_tau.shape[0]\n    k = np.arange(N, dtype='float')\n    return beta/N*np.fft.fft(G_tau*np.exp(1j*np.pi*k/N))\ncdef np.ndarray matsubara_fft_trick(np.ndarray G_tau):\n    cdef int N = G_tau.shape[0]\n    cdef np.ndarray freq = matsubara_freq(N)\n    cdef np.ndarray k = np.arange(N, dtype=np.float128)\n    return beta/N*np.fft.fft( (G_tau+0.5) *np.exp(1j*np.pi*k/N)) + 1./(1j*freq)\n#numpy orders frequencies differently so one has to convert frequencies\ncdef np.ndarray matsubara_freq(int N):\n    return np.pi/beta *(-2.*N*( np.array(np.fft.fftfreq(N),dtype=np.float128) ) +1.)\ncdef np.ndarray matsubara_ifft(np.ndarray G_omega):\n    N = G_omega.shape[0]\n    k = np.arange(N,dtype=np.float128)\n    return N/beta*np.exp(-1j*np.pi*k/N)*np.fft.ifft(G_omega)\n#following has to be improved:\ncdef np.ndarray matsubara_ifft_trick(G_omega):\n    cdef int N = G_omega.shape[0]\n    cdef np.ndarray freq = matsubara_freq(N)\n    k = np.arange(N,dtype=np.float128)\n    return -1/2+N/beta*np.exp(-1j*np.pi*k/N)*np.fft.ifft(G_omega-1./(1j*freq) )\n\ncdef int N = 500# number of imaginary frequencies\ntau = np.linspace(0,beta*(1.-1./N),N) # discrete points in imaginary time\ncdef double dtau = beta/N\nfreq = matsubara_freq(N) # matsubara frequencies, ordered according to np.fft\n\n\n# second order self energy diagram\ncdef np.ndarray self_energy(np.ndarray G0_omega,double U):\n    cdef np.ndarray G0_tau = matsubara_ifft_trick(G0_omega)\n    cdef np.ndarray integrand = np.zeros_like(G0_tau)\n\n    integrand[0]= -U*U*G0_tau[0]**3\n    cdef int N = G0_tau.shape[0]\n    \n    #if we define the index like this in cython, for loops are reduced to c-for loops and therefore way faster\n    cdef int i\n    for i in range(1,N):\n        integrand[i] = U*U*G0_tau[i]**2*G0_tau[N-i]\n    \n    return matsubara_fft_trick(integrand)\n\n# compute local Green's function from self energy given a disperion / density of states\n# we can still use all normal python libraries (although possibly slower)\nfrom scipy.integrate import simps\nNe = N\ndef dos_bethe(e):\n    return 2./(np.pi*D)*np.sqrt(1.-(e/D)**2)\ndef dos_cubic(e):\n    return 1./(t*np.sqrt(2.*np.pi))*np.exp(-e**2/(2.*t))\ndef Gloc_omega(self_energy_omega):\n    #return 1./Nk*np.sum(1./(1j*freq-epsilon_k[:, np.newaxis]-self_energy_omega),axis=0)\n    ##\n    energies_bethe = np.linspace(-D,D,Ne)\n    denergies_bethe = energies_bethe[2] - energies_bethe[1]\n    return simps( dos_bethe(energies_bethe)/(1j*freq[:,np.newaxis]-energies_bethe-self_energy_omega[:,np.newaxis]) ,dx=denergies_bethe)\n    ##\n    #energies_cubic = np.linspace(-10.*t,10*t,Ne)\n    #denergies_cubic = energies_cubic[2] - energies_cubic[1]\n    #return simps( dos_cubic(energies_cubic)/(1j*freq[:,np.newaxis]-energies_cubic-self_energy_omega[:,np.newaxis]) ,dx=denergies_cubic)\n\n# perform one loop from G0_initial to get new G0 and Gloc\ndef DMFT_loop(np.ndarray G0_initial_omega,double U,double frac_new, int iterations=1):\n    cdef np.ndarray G0_omega= G0_initial_omega\n    #use faster c for loop for iterations\n    cdef int i\n    for i in range(iterations):\n        self_e = self_energy(G0_omega,U)\n        Gloc = Gloc_omega(self_e)\n        G0_omega = frac_new * 1./(1./Gloc+self_e+U/2.) + (1.-frac_new) * G0_omega\n    return G0_omega, Gloc")


# In[65]:

get_ipython().run_cell_magic('cython', '', "import numpy as np\ncimport numpy as np\n# use the Padé approximation to interpolate the Green's function\ndef get_a(np.ndarray points,np.ndarray values):  \n    cdef int n = points.shape[0]\n    g_mat = np.zeros((n,n),dtype=np.complex256)\n    g_mat[0] = values\n    \n    cdef int i\n    cdef int j\n    for i in range(n-1): #filling the rest of the matrix using the recursion relation;\n        for j in range(n-(i+1)):\n            g_mat[i+1,i+1+j] = (g_mat[i,i]-g_mat[i,j+i+1])/((values[j+1+i]-values[i])*g_mat[i,j+i+1])\n\n    return g_mat.diagonal()\ndef get_C(np.ndarray z,np.ndarray points,np.ndarray a_coeff):\n    cdef int Na = points.shape[0]\n    cdef int Nz = z.shape[0]\n    A = np.zeros((Na+1,Nz),dtype=np.complex256)\n    B = np.zeros((Na+1,Nz),dtype=np.complex256)\n    A[1] = a_coeff[0]\n    B[0] = 1.\n    B[1] = 1.\n    cdef int i\n    for i in range(2,Na+1):\n        A[i] = A[i-1] + (z-points[i-2]) * a_coeff[i-1] * A[i-2]\n        B[i] = B[i-1] + (z-points[i-2]) * a_coeff[i-1] * B[i-2]\n    return A[Na]/B[Na]")


# In[66]:

# fit various Green's functions and plot the spectral function
G0 = 1. / ( 1j*freq + 2. )
Glocs = []
U_list = np.array([0.2,0.6,1.,1.45,1.9])
numiter = 1
for U in U_list:
    g_0 = G0
    g_0, g_loc = DMFT_loop(g_0,U,0.7,iterations=numiter)
    Glocs.append(g_loc)


# In[70]:

def shift_cut_symmetric(array, lower,upper,k):
    n = array.shape[0]
    interstep = 2*k+1
    a1 = np.fft.fftshift(array)[n/2+k+1:upper:interstep]
    a2 = np.fft.fftshift(array)[n/2-k:lower:-interstep]
    return np.concatenate((a2,a1))


# In[71]:

#step = 160
cut = 0.1
k = 2
#interstep = 2*k+1
eta = 0.1
plot_freq = np.linspace(-5.,5.,100)
my_freq = eta - 1j*plot_freq
#gf = 1./(1j*freq+2.)
fig, ax = plt.subplots(nrows=len(Glocs),ncols=2,figsize=(15,15))
fig.tight_layout()
for index in range(len(Glocs)):
    gf = Glocs[index]
    n = freq.size
    lower = 0
    upper = n
    freq_points = shift_cut_symmetric(freq,lower,upper,k)
    gf_points = shift_cut_symmetric(gf,lower,upper,k)
    # freq_points = freq[::step]
    #freq_points = np.fft.fftshift(freq)[lower:upper:interstep]
    # gf_points = gf[::step]
    #gf_points = np.fft.fftshift(gf)[lower:upper:interstep]
    a_list = get_a(freq_points, gf_points)
    spectral_func = -get_C(my_freq,freq_points,a_list).imag
    ordered_mats_freq = np.fft.fftshift(freq)
    fit_plot_freq = np.linspace(ordered_mats_freq[0],ordered_mats_freq[-1],5000)
    fit_func = get_C(fit_plot_freq,freq_points,a_list)
    ax[index,0].set_title("U = {}, GF on imaginary frequencies".format(U_list[index]))
    ax[index,0].plot(freq,Glocs[index].imag,'y+',label="GF imag")
    ax[index,0].plot(fit_plot_freq,fit_func.imag,'g-',label="Fit imag")
    ax[index,0].plot(freq,Glocs[index].real,'b+',label="GF real")
    ax[index,0].plot(fit_plot_freq,fit_func.real,'r-',label="Fit imag")
    ax[index,0].legend()  
    ax[index,1].set_title("U = {}, Spectral Function".format(U_list[index]))
    ax[index,1].plot(plot_freq,spectral_func,'-')


# In[72]:

# check the interpolation for an easy example
step = 10
index = 2
gf = 1./(1j*freq+2.)
a_list = get_a(freq[::step], gf[::step])
CC = get_C(freq,freq[::step],a_list)
plt.plot(freq,gf.imag,'b+')
plt.plot(np.fft.fftshift(freq),np.fft.fftshift(CC.imag),'g-')
plt.plot(freq,gf.real,'r+')
plt.plot(np.fft.fftshift(freq),np.fft.fftshift(CC.real),'y-')
#plt.axis([-10,-9.5,0.015,0.025])
plt.figure()
plt.plot(freq, gf.real-CC.real, 'b.')
plt.plot(freq, gf.imag-CC.imag, 'g.')


# In[73]:

# fit various Green's functions and plot the spectral function
G0 = 1. / ( 1j*freq + 2. )
g_0 = G0
Glocs = []
U_list = np.array([0.2,0.6,1.,1.45,1.9])
for U in U_list:
    for i in range(9):
        g_0, g_loc = DMFT_loop(g_0,U,0.3)
    Glocs.append(g_loc)


# In[75]:

#step = 160
cut = 0.1
interstep = 5
eta = 0.1
plot_freq = np.linspace(-5.,5.,100)
my_freq = eta - 1j*plot_freq
#gf = 1./(1j*freq+2.)
fig, ax = plt.subplots(nrows=len(Glocs),ncols=2,figsize=(15,15))
fig.tight_layout()
for index in range(len(Glocs)):
    gf = Glocs[index]
    n = freq.size
    # freq_points = freq[::step]
    lower = int(np.floor(cut*n))
    upper = int(np.ceil((1.-cut)*n))
    freq_points = np.fft.fftshift(freq)[lower:upper:interstep]
    # gf_points = gf[::step]
    gf_points = np.fft.fftshift(gf)[lower:upper:interstep]
    a_list = get_a(freq_points, gf_points)
    spectral_func = -get_C(my_freq,freq_points,a_list).imag
    ordered_mats_freq = np.fft.fftshift(freq)
    fit_plot_freq = np.linspace(ordered_mats_freq[0],ordered_mats_freq[-1],100)
    fit_func = get_C(fit_plot_freq,freq_points,a_list)
    ax[index,0].set_title("U = {}, GF on imaginary frequencies".format(U_list[index]))
    ax[index,0].plot(freq,Glocs[index].imag,'y+',label="GF imag")
    ax[index,0].plot(fit_plot_freq,fit_func.imag,'g-',label="Fit imag")
    ax[index,0].plot(freq,Glocs[index].real,'b+',label="GF real")
    ax[index,0].plot(fit_plot_freq,fit_func.real,'r-',label="Fit imag")
    ax[index,0].legend()  
    ax[index,1].set_title("U = {}, Spectral Function".format(U_list[index]))
    ax[index,1].plot(plot_freq,spectral_func,'-')


# In[ ]:




# In[ ]:




# In[ ]:



