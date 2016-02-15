
# coding: utf-8

# In[ ]:

get_ipython().magic('pylab inline')


# In[ ]:

beta = 100. # inverse temperature
t = 1. # energy scale for cubic lattice
D = 1. # energy scale for bethe lattice


# Loading DMFTlib module:

# In[ ]:

import DMFTlib
DMFTlib.initialize(beta_value=beta, bethe_lattice_D=D)


# In[ ]:

N = 10000 # number of imaginary frequencies
tau = DMFTlib.matsubara_time(N) # discrete points in imaginary time
dtau = beta/N
freq = DMFTlib.matsubara_freq(N) # matsubara frequencies, ordered according to np.fft


# In[ ]:

U_list = np.linspace(1,4,7)
numiter = 3200

from multiprocessing import Pool
def calculate(U):
    if U>2.5 and U<3.5: #increase iteration near phase transition
        return DMFTlib.DMFT_loop(G0,U,iterations=numiter*1000,frac_new=0.99)[1]
    else:
        return DMFTlib.DMFT_loop(G0,U,iterations=numiter,frac_new=0.99)[1]
    
# fit various Green's functions and plot the spectral function
G0 = 1. / ( 1j*freq + 2. )

p = Pool(8)


Glocs = p.map(calculate,U_list)
p.terminate()


# In[ ]:

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
eta = 1e-16
plot_freq = np.linspace(-5.,5.,1001)
my_freq = plot_freq +  eta*1j

n = len(Glocs)
fig, ax = plt.subplots(n, sharex=True, sharey=True, figsize=(10,2*n))
for index, gf in enumerate(Glocs):
    padeapp = DMFTlib.PadeApproximation(1j*freq,gf,cut_freq=1.,use_every=3)
    spectral_func = -padeapp(my_freq,norm=0.1).imag/np.pi
 
    ax[index].plot(plot_freq,spectral_func,'-',label="U/D={}".format(U_list[index]))
    ax[index].set_ylim(0,0.75)
    ax[index].set_xlim(-5.,5.)
    ax[index].legend(frameon=False)
    ax[index].set_xlabel("$\omega$")
    ax[index].set_ylabel("$A(\omega)$")
fig.subplots_adjust(hspace=0)
ax[0].set_title("Spectral Function")

plt.savefig("Mott_transition.pdf",bbox_inches='tight')

