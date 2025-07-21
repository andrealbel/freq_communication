from math import *
from pylab import *
import matplotlib.pyplot as pyplot
import numpy 
import matplotlib.patches as patches
pyplot.rcParams["text.usetex"] =True
import time
from scipy.signal import argrelextrema
from scipy.fftpack import fft, ifft
import cmath
from scipy.optimize import fsolve
import importlib
mpl_toolkits = importlib.import_module('mpl_toolkits')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


inicio=time.time()

# Linear networks of two 1D cells
# Fig 3 and 4
# ej = 1 -> A, ej = 2 -> B, ej = 3 -> C

ej = 3

gL1 = 0.1
gL2 = 0.1
gsyn11 = -0.0
gsyn22 = -0.0
gsyn12 = -0.05
gsyn21 = -0.05
if ej == 2:
   gsyn21 = 0.05
if ej == 3:
   gsyn12 = -0.2
   gsyn21 = 0.2
g1 =  0. ## 
g2 = 0. ##  
tau1 = 1.
tau2 = 1.


ain = 1
ain2 = 0
f = [0.2*fx for fx in range(0,5001)]
fmax = 200

print(f[-1])


#####################################################
# natural and resonant frequencies 

def Ss(v):
	hh = 1./(np.exp(-v)+1)
	return hh

def fresna(gl,gg1,tau):
    if gg1>0:
       ffres = 500*np.sqrt(-1+tau*np.sqrt(gg1**2+2*gl*gg1+2*gg1/tau))/(pi*tau)
       return ffres

def zmax(gl,gg1,tau):
    if gg1>0:
    	rara = np.sqrt(gg1*(2+gg1*tau+2*gl*tau)/tau)
    	zz = np.sqrt(1/(gl**2-1/tau**2-2*gg1/tau+2*rara/tau))
    	return zz
    	

delta1 = (gL1*tau1-1)**2-4*g1*tau1
delta2 = (gL2*tau2-1)**2-4*g2*tau2

if delta1<0 and delta2<0:
   print('ambos nodos osc')
   print('f nat 1 =', 500*np.sqrt(-delta1)/(pi*tau1))
   print('f nat 2 =', 500*np.sqrt(-delta2)/(pi*tau2))
elif delta1<0:
   print('nodo 1 osc')
   print('f nat 1 =', 500*np.sqrt(-delta1)/(pi*tau1))
elif delta2<0:
   print('nodo 2 osc')
   print('f nat 2 =', 500*np.sqrt(-delta2)/(pi*tau2))
else:
   print('ninguno oscila')

# resonant frequency for each isolated cell

fresn1 = fresna(gL1,g1,tau1)
fresn2 = fresna(gL2,g2,tau2)

print('f res 1 =', fresna(gL1,g1,tau1))
print('Z res 1 =', zmax(gL1,g1,tau1))
print()
print('f res 2 =', fresna(gL2,g2,tau2))
print('Z res 2 =', zmax(gL2,g2,tau2))
print()


#####################################################
# impedances and K coefficient

Z1=numpy.zeros(len(f),dtype=np.complex_)
Z2=numpy.zeros(len(f),dtype=np.complex_)
Zn1=numpy.zeros(len(f))
Zn2=numpy.zeros(len(f))
dZn1=numpy.zeros(len(f))
dZn2=numpy.zeros(len(f))
Pn1=numpy.zeros(len(f))
Pn2=numpy.zeros(len(f))
K=numpy.zeros(len(f))
DP=numpy.zeros(len(f))

sigma = 0

# impedances for each isolated cell

for jj in range(len(f)):
     ome = 2*np.pi*f[jj]/1000
     if g1>-1:
     	Z1[jj] = (1+tau1*ome*1j)/((gL1+ome*1j)*(1+tau1*ome*1j)+g1)
     if g2>-1:
     	Z2[jj] = (1+tau2*ome*1j)/((gL2+ome*1j)*(1+tau2*ome*1j)+g2)

Zi1 = abs(Z1)
Zi2 = abs(Z2)

# network impedance (Z_{ntwk,i}) for each cell of the network 

for jj in range(len(f)):
     detj =  1-gsyn12*gsyn21*Z1[jj]*Z2[jj]
     if g1>-1:
     	Zn1[jj] = abs(Z1[jj]/detj)
     	Pn1[jj] = -np.angle(Z1[jj]/detj)
     if g2>-1:
     	Zn2[jj] = abs((gsyn21*Z1[jj]*Z2[jj])/detj)
     	Pn2[jj] = -np.angle((gsyn21*Z1[jj]*Z2[jj])/detj)
     DP[jj] = Pn2[jj]-Pn1[jj]


for i in range(0,len(Pn1)):
    if DP[i]>pi/2:
       DP[i]=DP[i]-2*pi

# communication coefficient K

for jj in range(len(f)):
	K[jj] = Zn2[jj]/Zn1[jj]


#####################################################
# angle between position and tangent vectors

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    producto_cruzado = np.cross(v1_u, v2_u)
    signo = 1 # np.sign(producto_cruzado)    
    return signo*np.arccos(np.dot(v1_u, v2_u))

# numerical derivatives of network impedances

for k in range(len(f)-1):
   dZn1[k] = (Zn1[k+1]-Zn1[k])/(f[k+1]-f[k])
   dZn2[k] = (Zn2[k+1]-Zn2[k])/(f[k+1]-f[k])

angulo = numpy.zeros(len(f))

for k in range(len(f)-1):
   angulo[k] = angle_between([Zn1[k],Zn2[k]], [dZn1[k],dZn2[k]])


#####################################################
# equilibria and nullclines

def func(x):
    return [-gL1*x[0]-g1*x[0]-gsyn12*x[1],-gL2*x[1]-g2*x[1]-gsyn21*x[0]]

root = fsolve(func,[0,-5])

print(root)

v01 = root[0]
v02 = root[1]

Xx = np.arange(-15, 15, .1)
  
Nc1 = (gL1*Xx+g1*Xx)/gsyn12
Nc11 = (gL1*Xx+g1*Xx+1)/gsyn12
Nc12 = (gL1*Xx+g1*Xx-1)/gsyn12
Nc2 = gsyn21*Xx/(gL2+g2)


#####################################################
# trajectories for representative values of f

ff = [1,20,100]
if ej == 3:
    ff = [1,35,100]

T = 20000
dt = 0.01
cant = int(1/dt)
t = [tx*dt for tx in range(0, T*cant)]


solx1=[]
solx2=[]
ffs=[]
ttt = []

for  k in range(0,len(ff)):
  tt = t[:(int(4*1000./ff[k]))*cant]
  X = numpy.zeros((2,len(tt)))
  Y = numpy.zeros((2,len(tt)))
  Fsin = [0]*len(tt)
  ax = numpy.zeros((2,len(tt)))
  ay = numpy.zeros((2,len(tt)))
  X[0,0]= v01
  ax[0,0]= v01
  X[1,0]= v02
  ax[1,0]= v02
  for iii in range(2):
    if iii >0:
      X[0,0]= X[0,-1]
      ax[0,0]= ax[0,-1]
      X[1,0]= X[1,-1]
      ax[1,0]= ax[1,-1]      
      Y[0,0]= Y[0,-1]
      ay[0,0]= ay[0,-1]
      Y[1,0]= Y[1,-1]
      ay[1,0]= ay[1,-1]      
    for j in range(0,len(tt)-1):
      Fsin[j] = ain*sin(2*pi*ff[k]*tt[j]/1000)
#
      k1x1 = -gL1*X[0,j] -g1*Y[0,j] + gsyn12*X[1,j] + Fsin[j]
      k1y1 = (X[0,j]-Y[0,j])/tau1 
      k1x2 = -gL2*X[1,j] -g2*Y[1,j] + gsyn21*X[0,j] +ain2*Fsin[j] 
      k1y2 = (X[1,j]-Y[1,j])/tau2 
#
      ax[0,j+1] = X[0,j]+k1x1*dt
      ay[0,j+1] = Y[0,j]+k1y1*dt
      ax[1,j+1] = X[1,j]+k1x2*dt
      ay[1,j+1] = Y[1,j]+k1y2*dt
#
      k2x1 = -gL1*ax[0,j+1] - g1*ay[0,j+1] + gsyn12*ax[1,j+1] + Fsin[j]
      k2y1 = (ax[0,j+1]-ay[0,j+1])/tau1 
      k2x2 = -gL2*ax[1,j+1] - g2*ay[1,j+1] + gsyn21*ax[0,j+1] + ain2*Fsin[j]
      k2y2 = (ax[1,j+1]-ay[1,j+1])/tau2
#
      X[0,j+1] = X[0,j] + (k1x1 + k2x1)*dt/2
      Y[0,j+1] = Y[0,j] + (k1y1 + k2y1)*dt/2
      X[1,j+1] = X[1,j] + (k1x2 + k2x2)*dt/2
      Y[1,j+1] = Y[1,j] + (k1y2 + k2y2)*dt/2
  solx1.append(X[0,int(len(tt)/2):])
  solx2.append(X[1,int(len(tt)/2):])
  ffs.append(Fsin[int(len(tt)/2):])
  ttt.append(tt[int(len(tt)/2):])


puntosK = []
for i in range(len(ff)):
   for ii in range(len(f)):
      if ff[i]==f[ii]:
         puntosK.append([Zn1[ii],Zn2[ii],ii])

for i in range(len(puntosK)):
    print(puntosK[i][2])


fint=time.time()

print('tiempo=', fint-inicio)

#################################################

font1 = 24
font2 = 22

fig, axs = plt.subplots(2,1,figsize=(6,5.5),dpi=100)
axs[0].plot([fresn1,fresn1],[0,10],'--',color='lightgray')
axs[0].plot([fresn2,fresn2],[0,10],'--',color='gray')
axs[0].plot(f,Zn1,c= 'blue', label=r'$ Z_{ntwk,1}$')
axs[0].plot(f,Zn2,c='red',label=r'$ Z_{ntwk,2}$')
axs[0].plot(f,K,lw='2',c= 'k', label=r'$ K$')
axs[0].xaxis.set_tick_params(labelsize=font1)
axs[0].yaxis.set_tick_params(labelsize=font1)
axs[0].legend(loc=1,fontsize=font2)
if ej ==1 :
    axs[0].axis([0,fmax,0,15])
if ej == 2:
    axs[0].axis([0,fmax,0,9])
if ej == 3:
    axs[0].axis([0,fmax,0,6])
axs[1].plot([fresn1,fresn1],[0,10],'--',color='lightgray')
axs[1].plot([fresn2,fresn2],[0,10],'--',color='gray')
axs[1].plot(f,Zn1/Zn1[0],c= 'blue', label=r'$ Z^*_{ntwk,1}$')
axs[1].plot(f,Zn2/Zn2[0],c='red',label=r'$ Z^*_{ntwk,2}$')
axs[1].plot(f,K/K[0],lw='2',c= 'k', label=r'$ K^*$')
axs[1].set_xlabel(r'$f$ [Hz]', fontsize =font1)
axs[1].xaxis.set_tick_params(labelsize=font1)
axs[1].yaxis.set_tick_params(labelsize=font1)
axs[1].legend(loc=1,fontsize=font2)
axs[1].axis([0,fmax,0,1.1])
axs[1].set_title('Normalized', fontdict={'fontsize': font1+2, 'fontweight': 'medium'},loc='left')
if ej==3:
   axs[1].axis([0,fmax,0,3.1])
fig.tight_layout()


if ej == 1:
   savefig('L1_1.pdf')
if ej == 2:
   savefig('L2_1.pdf')
if ej == 3:
   savefig('L3_1.pdf')


pyplot.figure(figsize=(6,5.5),dpi=100)
pyplot.plot(Xx, Nc1, c = 'orange', label=r'$v_1$-nullcline')
pyplot.plot(Xx, Nc11,'--', c = 'orange')
pyplot.plot(Xx, Nc12, '--', c = 'orange')
pyplot.plot(Xx, Nc2,  c = 'green', label=r'$v_2$-nullcline')
pyplot.plot(solx1[0],solx2[0],'b', label='trajectories')
for i in range(1,len(ff)):
    pyplot.plot(solx1[i],solx2[i],'b')
pyplot.plot(Zn1+v01,Zn2+v02,lw=2,c='black')
for i in range(len(puntosK)):
   pyplot.scatter(puntosK[i][0]+v01,puntosK[i][1]+v02,s=80,marker='*',c='gray',zorder=3,alpha=1)
if ej ==1:
   pyplot.text(-13,7.5, r'$f=1$', fontsize =font1)
   pyplot.text(6.5,-2, r'$f=20$', fontsize =font1)
   pyplot.text(-5,5, r'$f=100$', fontsize =font1)
   pyplot.text(6.5,7, r'$K$-curve', fontsize =font1)
   pyplot.arrow(-1.3,0.15,-1,3.8,width=0.02,head_width=0.4,fc='k', ec='k',zorder=2)
if ej ==2:
   pyplot.text(2,4.5, r'$f=1$', fontsize =font1)
   pyplot.text(-3.5,-3.5, r'$f=20$', fontsize =font1)
   pyplot.text(-10,3, r'$f=100$', fontsize =font1)
   pyplot.text(5,-4, r'$K$-curve', fontsize =font1)
   pyplot.arrow(-1.4,0.1,-4,2,width=0.02,head_width=0.4,fc='k', ec='k',zorder=2)
   pyplot.arrow(7,2.5,0.4,-4,width=0.02,head_width=0.4,fc='k', ec='k',zorder=2)
if ej ==3:
   pyplot.text(-4,7, r'$f=1$', fontsize =font1)
   pyplot.text(4,-3, r'$f=35$', fontsize =font1)
   pyplot.text(-13,3, r'$f=100$', fontsize =font1)
   pyplot.text(4.5,6, r'$K$-curve', fontsize =font1)
   pyplot.arrow(0.8,1.8,-1,4,width=0.02,head_width=0.4,fc='k', ec='k',zorder=2)
   pyplot.arrow(-1.8,0.,-8,2.25,width=0.02,head_width=0.4,fc='k', ec='k',zorder=2)
pyplot.xlabel(r'$v_1$', fontsize =font1)
pyplot.ylabel(r'$v_2$', fontsize =font1)
pyplot.legend(loc=3,fontsize=font2)
pyplot.xticks(fontsize=font1)
pyplot.yticks(fontsize=font1)
pyplot.axis([-15,14,-15,14])
if ej >1 :
   pyplot.axis([-15,12,-15,12])
pyplot.tight_layout()

if ej == 1:
   savefig('L1_3.pdf')
if ej == 2:
   savefig('L2_3.pdf')
if ej == 3:
   savefig('L3_3.pdf')


prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",color='b',lw=1.5,
            shrinkA=0,shrinkB=0)
prop2 = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",color='r',lw=1.5,
            shrinkA=0,shrinkB=0)

fig, ax1 = plt.subplots(figsize=(6,5.5),dpi=100)
ax1.plot(Xx, Nc1, c = 'orange')
ax1.plot(Xx, Nc2,  c = 'green')
ax1.plot(Zn1+v01,Zn2+v02,lw=2,c='black')
for i in range(len(puntosK)):
   ax1.scatter(puntosK[i][0]+v01,puntosK[i][1]+v02,s=80,marker='*',c='gray',zorder=3,alpha=1)
for i in range(len(puntosK)):
   ccual = puntosK[i][2]
   norm = np.sqrt(dZn1[ccual]**2+dZn2[ccual]**2)
   ax1.annotate("", xy=(Zn1[ccual],Zn2[ccual]), xytext=(v01,v02), arrowprops=prop)
   tv = [dZn1[ccual]/norm,dZn2[ccual]/norm]
   ax1.annotate("", xy=(tv[0]+Zn1[ccual],tv[1]+Zn2[ccual]), xytext=(Zn1[ccual],Zn2[ccual]), arrowprops=prop2)
ax1.set_xlabel(r'$v_1$', fontsize =font1)
ax1.set_ylabel(r'$v_2$', fontsize =font1)
ax1.xaxis.set_tick_params(labelsize=font1)
ax1.yaxis.set_tick_params(labelsize=font1)
if ej ==1:
   ax1.text(10,3, r'$K$-curve', fontsize =font1)
   ax1.axis([-.5,14,-.5,9])
if ej ==2:
   ax1.text(5,.5, r'$K$-curve', fontsize =font1)
   ax1.axis([-.5,8.5,-.5,6])
if ej ==3:
   ax1.text(3.5,.8, r'$K$-curve', fontsize =font1)
   ax1.axis([-.5,6.1,-.5,8])
ax2 = inset_axes(ax1, width="70%", height="22%", loc="upper center")
ax2.plot(f,angulo,c='darkgray',lw='2',label=r'$ang(\vec{P},\vec{T})$')
ax2.plot([0,fmax],[pi,pi],'--', c = 'lightgrey')
for i in range(len(puntosK)):
   ccual = puntosK[i][2]
   ax2.scatter(f[ccual],angulo[ccual],s=80,marker='*',c='gray',zorder=3,alpha=1)
ax2.axis([0,fmax,1.5,3.2])
if ej == 3:
   ax2.axis([0,fmax,0,3.2])
ax2.set_xlabel(r'$f$', fontsize =20)
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.legend(loc=4,fontsize=20)
ax2.set_xticks([0,50,100,150,200])
ax2.set_yticks([np.pi/2,np.pi],labels=[r'$\frac{\pi}{2}$', r'$\pi$'])
if ej ==3 :
   ax2.set_yticks([0,np.pi/2,np.pi],labels=[r'$0$',r'$\frac{\pi}{2}$', r'$\pi$'])
fig.tight_layout()

if ej == 1:
   savefig('L1_4.pdf')
if ej == 2:
   savefig('L2_4.pdf')
if ej == 3:
   savefig('L3_4.pdf')


pyplot.figure(figsize=(6,4),dpi=100)
pyplot.plot([0,fmax],[-pi/2,-pi/2],'--', c = 'lightgrey')
pyplot.plot([0,fmax],[0,0],'--', c = 'lightgrey')
pyplot.plot([0,fmax],[pi/2,pi/2],'--', c = 'lightgrey')
pyplot.plot([0,fmax],[pi,pi],'--', c = 'lightgrey')
pyplot.plot(f,Pn1, c = 'blue',lw='2', label = r'$ \Phi_{ntwk,1}$')
pyplot.plot(f,Pn2, c='red',lw='2',label = r'$ \Phi_{ntwk,2}$')
pyplot.plot(f,DP, c='black',lw='2',label = r'$ \Delta\Phi$')
pyplot.xlabel(r'$f$ [Hz]',fontsize=24)
pyplot.axis([0,fmax,-1.6,np.pi+.1])
pyplot.legend(loc=4,fontsize=22)
pyplot.xticks([0,50,100,150,200],fontsize=24)
pyplot.yticks([-np.pi/2,0,np.pi/2,np.pi],labels=[  r'$-\frac{\pi}{2}$','$0$',  r'$\frac{\pi}{2}$',r'$\pi$'],fontsize=24)
if ej == 1:
     pyplot.axis([0,fmax,-3.2,1.7])
     pyplot.yticks([-np.pi,-np.pi/2,0,np.pi/2],labels=[ r'$-\pi$', r'$-\frac{\pi}{2}$','$0$',  r'$\frac{\pi}{2}$'],fontsize=24)
pyplot.tight_layout()

if ej == 1:
   savefig('L1_5.pdf')
if ej == 2:
   savefig('L2_5.pdf')
if ej == 3:
   savefig('L3_5.pdf')

pyplot.show()





