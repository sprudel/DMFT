
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

# In[5]:

from scipy.integrate import simps
#density of states for Bethe lattice
def dos_bethe(e): 
    return 2./(np.pi*D)*np.sqrt(1.-(e/D)**2)

def Gloc_omega(self_energy_omega):
    #we use same resultion of energies as for matsubara frequencies
    N = self_energy_omega.shape[0]
    energies_bethe = np.linspace(-D,D,N)
    denergies_bethe = energies_bethe[2] - energies_bethe[1]
    freq = matsubara_freq(N)
    return simps( dos_bethe(energies_bethe)/(1j*freq[:,np.newaxis]-energies_bethe-self_energy_omega[:,np.newaxis]) ,dx=denergies_bethe)


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

# In[7]:

class PadeApproximation:
    def __init__(self, points, values,filter_index=None):
        #possible to filter specific values eg filter_index = points>0 to filter positive values
        if filter_index is not None:
            self.N = points[filter_index].shape[0]
            self.points = points[filter_index]
            self.values = values[filter_index]
        else:
            self.N = points.shape[0]
            self.points = points
            self.values = values
        
        g_matrix = np.zeros((self.N,self.N),dtype=np.complex256)
        g_matrix[0] = self.values
        for i in range(1,self.N):
            g_matrix[i,i:] = ( g_matrix[i-1,i-1] - g_matrix[i-1,i:] ) / ( ( self.points[i:] - self.points[i-1] ) * g_matrix[i-1,i:] )
        self.a_coeff = np.diag(g_matrix)
    def __call__(self,z):
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

