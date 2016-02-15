
# coding: utf-8

# In[2]:

get_ipython().magic('pylab inline')


# In[3]:

x = linspace(-1,1,100)
y = linspace(-1,1,100)
z0 = zeros_like(y)
freq = linspace(0.2,1.2,10)


# In[17]:

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
figure(figsize=(6,6))
ylim(-1,1)
title("Domain of the Matsubara and retarded Green's function")
plot(x,z0, color='black')
plot(z0,y, color="black")
plot(x,z0+0.05, "--", lw=2, color='red', label="dom $G(\omega + i \eta)$")
fill_between(x,0,z0+1,color='blue', alpha=0.2)
plot(zeros_like(freq), freq, "o", color="green", label="dom $G(i\omega)$")
plot(zeros_like(freq), -freq, "o", color="green")
xlabel("$\Re{}$")
ylabel("$\Im{}$")
annotate("Jump in Green's function", (0,0), (-0.8,-0.2),arrowprops=dict(facecolor='black', shrink=0.001))
legend()
plt.savefig("analytic_continuation.pdf",bbox_inches='tight')


# In[ ]:



