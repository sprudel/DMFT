
# coding: utf-8

# This file contains the whole DMFT loop with its helper functions and the Padé approximation for analytic continuation

# In[1]:

import numpy as np


# In[2]:

def initialize(beta_value=100.,bethe_lattice_D=1.):
    global beta
    global D
    beta = beta_value
    D = bethe_lattice_D


# ## Matsubara Fouriertransforms

# In[3]:

def matsubara_time(N):
    return np.linspace(0,beta*(1.-1./N),N) # discrete points in imaginary time
#definition of FourierTransforms
def matsubara_fft_naiv(G_tau):
    global beta
    N = G_tau.shape[0]
    k = np.arange(N, dtype='float')
    return beta/N*np.fft.fft(G_tau*np.exp(1j*np.pi*k/N))
def matsubara_fft(G_tau):
    global beta
    N = G_tau.shape[0]
    freq = matsubara_freq(N)
    k = np.arange(N, dtype='float')
    return beta/N*np.fft.fft( (G_tau+0.5) *np.exp(1j*np.pi*k/N)) + 1./(1j*freq)
#numpy orders frequencies differently so one has to convert frequencies
def matsubara_freq(N):
    global beta
    return np.pi/beta *( np.array(-2*N*np.fft.fftfreq(N),dtype=np.float128) +1.)
def matsubara_ifft_naiv(G_omega):
    global beta
    N = G_omega.shape[0]
    k = np.arange(N,dtype='float')
    return N/beta*np.exp(-1j*np.pi*k/N)*np.fft.ifft(G_omega)
def matsubara_ifft(G_omega):
    global beta
    N = G_omega.shape[0]
    freq = matsubara_freq(N)
    k = np.arange(N,dtype='float')
    return -1/2+N/beta*np.exp(-1j*np.pi*k/N)*np.fft.ifft(G_omega-1./(1j*freq) )


# ## Second order perturbation theory
# Self energy:

# In[4]:

# second order self energy diagram
def self_energy(G0_omega,U):
    G0_tau = matsubara_ifft(G0_omega)
    G0_minus_tau = -np.roll(G0_tau[::-1],1)
    G0_minus_tau[0] = -G0_minus_tau[0]
    sigma_tau = -U*U * G0_tau*G0_tau * G0_minus_tau
    return matsubara_fft(sigma_tau)


# ## Local Greensfunction
# compute local Green's function from self energy given a disperion / density of states.
# Here we use dispersion relation of Bethe lattice.
# $$G_{loc}(iω) = \int\limits_{-D}^{D} dε \frac{ρ(ε)}{iω-ε-\Sigma(iω)} \quad ρ(ε)= \frac{2}{π D} \sqrt{1-\frac{ε^2}{D^2}}$$
# This can be calculated analytically (Mathematica):
# $$G_{loc}(iω) = \frac{2}{π D} \left( B π + \sqrt{1-B^2}\left[\log{(1-B)}-\log{(B-1)}\right] \right)$$
# with $B=\frac{iω-\Sigma(iω)}{D}$ and only for $\Im{B} \neq 0 $ and $-1\leq\Re{B}\leq1$.

# In[5]:

from scipy.integrate import simps
#density of states for Bethe lattice
def dos_bethe(e): 
    return 2./(np.pi*D)*np.sqrt(1.-(e/D)**2)

def Gloc_omega(self_energy_omega):
    #we use same resultion of energies as for matsubara frequencies
    N = self_energy_omega.shape[0]
    #energies_bethe = np.linspace(-D,D,N)
    #denergies_bethe = energies_bethe[2] - energies_bethe[1]
    freq = matsubara_freq(N)
    B=(1j*freq-self_energy_omega)/D
    return 2/np.pi/D*(B*np.pi+np.sqrt(1-B**2)*(np.log(1-B)-np.log(B-1)))
    #return simps( dos_bethe(energies_bethe)/(1j*freq[:,np.newaxis]-energies_bethe-self_energy_omega[:,np.newaxis]) ,dx=denergies_bethe)


# ## DMFT Loop

# In[6]:

def DMFT_loop(G0_initial_omega,U,iterations=1,frac_new=1.):
    G0_omega= G0_initial_omega
    N = G0_initial_omega.shape[0]
    for i in range(iterations):
        self_e = self_energy(G0_omega,U)
        Gloc = Gloc_omega(self_e)
        freq = matsubara_freq(N)
        #G0_omega = frac_new * 1./(1./Gloc+self_e) + (1.-frac_new) * G0_omega
        G0_omega = frac_new/( 1j*freq - 0.25*D*D* Gloc ) + (1.-frac_new) * G0_omega # only for Bethe lattice
    return G0_omega, Gloc


# ## Padé Approximation
# For the Padé approxmitaion we use only positive frequencies and use the observed symmetry:
# $$\Im{G(-iω)} = -\Im{G(iω)}$$
# $$\Re{G(-iω)} = \Re{G(iω)}$$
# i.e.
# $$G(-i ω) = G(i ω)^*$$

# In[1]:

class PadeApproximation:
    def __init__(self, points, values,cut_freq=1.,use_every=1):
        
        #first sort frequencies (points = i omega), sort can also sort complex arrays (real first, then complex)
        #np.arg gives index to sort array
        index_sorted = np.argsort(points)
        self.points = points[index_sorted]
        self.values = values[index_sorted]
        
        #only positive freqencies below cat and only use every [use_every] points
        index_pos = np.logical_and(np.imag(self.points)>0,np.imag(self.points)<cut_freq*np.imag(self.points).max())
        self.points = self.points[index_pos][::use_every]
        self.values = self.values[index_pos][::use_every]
        
        self.N = self.points.shape[0]
        
        g_matrix = np.zeros((self.N,self.N),dtype=np.complex256)
        g_matrix[0] = self.values
        for i in range(1,self.N):
            g_matrix[i,i:] = ( g_matrix[i-1,i-1] - g_matrix[i-1,i:] ) / ( ( self.points[i:] - self.points[i-1] ) * g_matrix[i-1,i:] )
        self.a_coeff = np.diag(g_matrix)
    def __call__(self,z,norm=1.):
        positive_imag_index = np.imag(z)>0
        tmp = np.zeros_like(z)
        #for positive imaginary index do normal procedure
        tmp[positive_imag_index] = self._evaluate_fit(z[positive_imag_index],norm=norm)
        #for negative imaginary index use symmetriy
        tmp[np.logical_not(positive_imag_index)] = np.conjugate(self._evaluate_fit(-z[np.logical_not(positive_imag_index)],norm=norm))
        return tmp
    def _evaluate_fit_old(self,z):
        Nz = z.shape[0]
        A = np.zeros((self.N+1,Nz),dtype=np.complex256)
        B = np.zeros((self.N+1,Nz),dtype=np.complex256)
        A[1] = self.a_coeff[0]
        B[0] = 1.
        B[1] = 1.
        for i in range(2,self.N+1):
            A[i] = A[i-1] + (z-self.points[i-2]) * self.a_coeff[i-1] * A[i-2]
            B[i] = B[i-1] + (z-self.points[i-2]) * self.a_coeff[i-1] * B[i-2]
        return A[self.N]/B[self.N]
    def _evaluate_fit(self,z,norm=1.):
        Nz = z.shape[0]
        Am1 = np.zeros_like(z,dtype=np.complex256)
        Am2 = np.zeros_like(z,dtype=np.complex256)
        Bm1 = np.ones_like(z,dtype=np.complex256)
        Bm2 = np.ones_like(z,dtype=np.complex256)
        tmpA = np.zeros_like(z,dtype=np.complex256)
        tmpB = np.zeros_like(z,dtype=np.complex256)
        factor = np.zeros_like(z,dtype=np.complex256)
        np.copyto(Am1, self.a_coeff[0])
        for i in range(2,self.N+1):
            np.copyto(factor,(z-self.points[i-2])* self.a_coeff[i-1])
            #normalisation = 1/np.sqrt(1+np.abs(factor)**2-np.sqrt(1+1/4*np.abs(factor)**4))
            np.copyto(tmpA,norm*Am1)
            np.copyto(Am1, norm*(Am1 + factor* Am2))
            np.copyto(Am2,tmpA)
            np.copyto(tmpB,norm*Bm1)
            np.copyto(Bm1, norm*(Bm1 + factor* Bm2))
            np.copyto(Bm2,tmpB)
        return Am1/Bm1
        


# In[ ]:



