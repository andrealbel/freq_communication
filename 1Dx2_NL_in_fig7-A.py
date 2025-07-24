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
from matplotlib import gridspec


# Nonlinear inhibitory networks of two 1D cells
# Fig 7 A


inicio=time.time()


gL1 = 0.2
gL2 = 0.2
gsyn12 = 0.0
gsyn21 = 0.01
e12 = -20.
e21 = -20.
g1 =  0. ## 
g2 = 0. ##  
tau1 = 1.
tau2 = 1.


ain = 1.
f = [.3+.5*fx for fx in range(1,40)]+[20+fx for fx in range(0,30)]+[50+2*fx for fx in range(0,40)]+[130+5*fx for fx in range(0,14)]+[200,300,400,500,600,800,1000,2000]
fmax = 200


def Ss(v):
	hh = 1./(np.exp(-v)+1)
	return hh

def Ssinv(v):
	hh = -np.log(1/v-1)
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
  
#####################################################
# equilibria and nullclines

def func(x):
    return [-gL1*x[0]-g1*x[0]-gsyn12*Ss(x[1])*(x[0]-e12),-gL2*x[1]-g2*x[1]-gsyn21*Ss(x[0])*(x[1]-e21)]
 

root = fsolve(func,[0,-5])

print(root)

v010 = root[0]
v020 = root[1]

Xx = np.arange(-10, 10, .02)

Nc1 = Ssinv((-gL1*Xx-g1*Xx)/(gsyn12*(Xx-e12))) 
Nc11 = Ssinv((-gL1*Xx-g1*Xx+ain)/(gsyn12*(Xx-e12)))
Nc12 = Ssinv((-gL1*Xx-g1*Xx-ain)/(gsyn12*(Xx-e12)))
Nc20 = -gsyn21*Ss(Xx)*e21/(-gL2-g2-gsyn21*Ss(Xx))

####################################
# impedances and phases isolated cells

Zi1=numpy.zeros(len(f))
Zi2=numpy.zeros(len(f))
Pi1=numpy.zeros(len(f))
Pi2=numpy.zeros(len(f))

sigma = 0

for jj in range(len(f)):
     ome = 2*np.pi*f[jj]/1000
     if g1>-1:
     	deno = np.sqrt((gL1/tau1-ome**2+g1/tau1)**2+(gL1+1/tau1)**2*ome**2) 
     	Zi1[jj] = np.sqrt((1/tau1)**2+ome**2)/deno
     	Pi1[jj] = arctan(ome/gL1)
     if g2>-1:
     	deno2 = np.sqrt((gL2/tau2-(ome)**2+g2/tau2)**2+(gL2+1/tau2)**2*ome**2) 
     	Zi2[jj] = np.sqrt((1/tau2)**2+ome**2)/deno2
     	Pi2[jj] = arctan(ome/gL2)
	

#####################################################
# impedances and K coefficient


T = 20000
dt = 0.01
cant = int(1/dt)
t = [tx*dt for tx in range(0, T*cant)]


Z1=numpy.zeros(len(f))
dZ1=numpy.zeros(len(f))
Z2=numpy.zeros(len(f))
dZ2=numpy.zeros(len(f))
Mv1=numpy.zeros(len(f))
mv1=numpy.zeros(len(f))
dMv1=numpy.zeros(len(f))
Mv2=numpy.zeros(len(f))
mv2=numpy.zeros(len(f))
dMv2=numpy.zeros(len(f))
phase1=numpy.zeros(len(f))
phase2=numpy.zeros(len(f))
K=numpy.zeros(len(f))
KM=numpy.zeros(len(f))

Z10=numpy.zeros(len(f))
Z20=numpy.zeros(len(f))
Mv10=numpy.zeros(len(f))
mv10=numpy.zeros(len(f))
Mv20=numpy.zeros(len(f))
mv20=numpy.zeros(len(f))
phase10=numpy.zeros(len(f))
phase20=numpy.zeros(len(f))
K0=numpy.zeros(len(f))
KM0=numpy.zeros(len(f))


print()

for jj in range(2):
 if jj == 1:
    gsyn21 = 0.1
    Nc2 = -gsyn21*Ss(Xx)*e21/(-gL2-g2-gsyn21*Ss(Xx))
    root = fsolve(func,[0,-5])
    v01 = root[0]
    v02 = root[1]
 solx1=[] 
 solx2=[]
 ffs=[]
 ttt = []
 for  k in range(0,len(f)):
   tt = t[:(int(4*1000./f[k]))*cant]
   if f[k] > 30:
     tt = [tx*dt for tx in range(0, 500*cant)]     
   X = numpy.zeros((2,len(tt)))
   Y = numpy.zeros((2,len(tt)))
   Fsin = [0]*len(tt)
   ax = numpy.zeros((2,len(tt)))
   ay = numpy.zeros((2,len(tt)))
   X[0,0]= v010
   ax[0,0]= v010
   X[1,0]= v020
   ax[1,0]= v020
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
       Fsin[j] = ain*sin(2*pi*f[k]*tt[j]/1000)
#
       k1x1 = -gL1*X[0,j] -g1*Y[0,j] - gsyn12*Ss(X[1,j])*(X[0,j]-e12) + Fsin[j]
       k1y1 = (X[0,j]-Y[0,j])/tau1 
       k1x2 = -gL2*X[1,j] -g2*Y[1,j] - gsyn21*Ss(X[0,j])*(X[1,j]-e21) 
       k1y2 = (X[1,j]-Y[1,j])/tau2 
#
       ax[0,j+1] = X[0,j]+k1x1*dt
       ay[0,j+1] = Y[0,j]+k1y1*dt
       ax[1,j+1] = X[1,j]+k1x2*dt
       ay[1,j+1] = Y[1,j]+k1y2*dt
#
       k2x1 = -gL1*ax[0,j+1] - g1*ay[0,j+1] - gsyn12*Ss(ax[1,j+1])*(ax[0,j+1]-e12) + Fsin[j]
       k2y1 = (ax[0,j+1]-ay[0,j+1])/tau1 
       k2x2 = -gL2*ax[1,j+1] - g2*ay[1,j+1] - gsyn21*Ss(ax[0,j+1])*(ax[1,j+1]-e21) 
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
   max11=[]
   min11=[]
   max22=[]
   min22=[]
   for kk in range(1,len(solx1[k])-1):
      if solx1[k][kk-1]<solx1[k][kk] and solx1[k][kk]>solx1[k][kk+1]:
         max11.append([solx1[k][kk],ttt[k][kk]])
      if solx1[k][kk-1]>solx1[k][kk] and solx1[k][kk]<solx1[k][kk+1]:
         min11.append([solx1[k][kk],ttt[k][kk]])
      if solx2[k][kk-1]<solx2[k][kk] and solx2[k][kk]>solx2[k][kk+1]:
         max22.append([solx2[k][kk],ttt[k][kk]])
      if solx2[k][kk-1]>solx2[k][kk] and solx2[k][kk]<solx2[k][kk+1]:
         min22.append([solx2[k][kk],ttt[k][kk]])
      if ffs[k][kk-1]<ffs[k][kk] and ffs[k][kk]>ffs[k][kk+1]:
         maxf = [ffs[k][kk],ttt[k][kk]]
   max1 = sorted(max11, reverse=True)
   min1 = sorted(min11)
   max2 = sorted(max22, reverse=True)
   min2 = sorted(min22)
   if jj == 0:
     if len(max1)>0:
       phase10[k] = ((max1[0][1]-maxf[1])*pi*f[k]/500.)
     if len(max2)>0:
       phase20[k] = ((max2[0][1]-maxf[1])*pi*f[k]/500.)
     Z10[k] = (max1[0][0] - min1[0][0])/(2*ain)
     Z20[k] = (max2[0][0] - min2[0][0])/(2*ain)
     K0[k] = Z20[k]/Z10[k]
     Mv10[k] = max1[0][0]
     Mv20[k] = max2[0][0]
     mv10[k] = min1[0][0]
     mv20[k] = min2[0][0]
     KM0[k] = (Mv20[k]-v020)/(Mv10[k]-v010)
   if jj == 1:
     Z1[k] = (max1[0][0] - min1[0][0])/(2*ain)
     Z2[k] = (max2[0][0] - min2[0][0])/(2*ain)
     K[k] = Z2[k]/Z1[k]
     if len(max1)>0:
       phase1[k] = ((max1[0][1]-maxf[1])*pi*f[k]/500.)
     if len(max2)>0:
       phase2[k] = ((max2[0][1]-maxf[1])*pi*f[k]/500.)
     Mv1[k] = max1[0][0]
     Mv2[k] = max2[0][0]
     mv1[k] = min1[0][0]
     mv2[k] = min2[0][0]
     KM[k] = (Mv2[k]-v02)/(Mv1[k]-v01)


Dphase0=numpy.zeros(len(f))
for i in range(0,len(phase10)):
    phase10[i] = np.mod(phase10[i], 2*pi)
    phase20[i] = np.mod(phase20[i], 2*pi)
    Dphase0[i] = phase20[i]-phase10[i]

for i in range(0,len(phase10)):
    if phase10[i]>pi:
       phase10[i]=phase10[i]-2*pi
    if phase20[i]>pi:
       phase20[i]=phase20[i]-2*pi
    if Dphase0[i]>pi:
       Dphase0[i]=Dphase0[i]-2*pi


Dphase=numpy.zeros(len(f))
for i in range(0,len(phase1)):
    phase1[i] = np.mod(phase1[i], 2*pi)
    phase2[i] = np.mod(phase2[i], 2*pi)
    Dphase[i] = phase2[i]-phase1[i]

for i in range(0,len(phase1)):
    if phase1[i]>pi:
       phase1[i]=phase1[i]-2*pi
    if phase2[i]>pi:
       phase2[i]=phase2[i]-2*pi
    if Dphase[i]>pi:
       Dphase[i]=Dphase[i]-2*pi


#####################################################
# angle between position and tangent vectors

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    producto_cruzado = np.cross(v1_u, v2_u)
    signo = 1 #np.sign(producto_cruzado)    
    return signo*np.arccos(np.dot(v1_u, v2_u))


for k in range(len(f)-1):
   dZ1[k] = (Z1[k+1]-Z1[k])/(f[k+1]-f[k])
   dZ2[k] = (Z2[k+1]-Z2[k])/(f[k+1]-f[k])
   dMv1[k] = (Mv1[k+1]-Mv1[k])/(f[k+1]-f[k])
   dMv2[k] = (Mv2[k+1]-Mv2[k])/(f[k+1]-f[k])

angulo = numpy.zeros(len(f))

for k in range(len(f)-1):
   angulo[k] = angle_between([Mv1[k]-v01,Mv2[k]-v02], [dMv1[k],dMv2[k]])


#####################################################
# K max and min

puntosM0 = []

for k in range(1,len(f)-1):
   if KM0[k]<KM0[k-1] and KM0[k]<KM0[k+1]:
      puntosM0.append([k,Mv10[k],Mv20[k],KM0[k]])
   if KM0[k]>KM0[k-1] and KM0[k]>KM0[k+1]:
      puntosM0.append([k,Mv10[k],Mv20[k],KM0[k]])

puntos = []

for k in range(1,len(f)-1):
   if K[k]<K[k-1] and K[k]<K[k+1]:
      puntos.append([k,Z1[k],Z2[k],K[k]])
   if K[k]>K[k-1] and K[k]>K[k+1]:
      puntos.append([k,Z1[k],Z2[k],K[k]])


print('puntosM0',puntosM0)
print('puntos',puntos)

puntosM = []

for k in range(1,len(f)-1):
   if KM[k]<KM[k-1] and KM[k]<KM[k+1]:
      puntosM.append([k,Mv1[k],Mv2[k],KM[k]])
   if KM[k]>KM[k-1] and KM[k]>KM[k+1]:
      puntosM.append([k,Mv1[k],Mv2[k],KM[k]])

print('puntosM',puntosM)


fint=time.time()

print('tiempo=', fint-inicio)

########################################################

font1 = 24
font2 = 24

alp0 = 0.3

pyplot.figure(figsize=(6,5.5),dpi=100)
pyplot.plot(f,Mv10,lw='2',c= 'blue',alpha=alp0)
pyplot.plot(f,Mv20,lw='2',c='red',alpha=alp0)
pyplot.plot(f,KM0,lw='2',c= 'k',alpha=alp0)
pyplot.plot(f,Mv1,lw='2',c= 'blue', label=r'$ V_{max,1}$')
pyplot.plot(f,Mv2,lw='2',c='red',label=r'$ V_{max,2}$')
pyplot.plot(f,KM,lw='2',c= 'k', label=r'$ K$')
for i in range(len(puntosM0)):
   ccual2 = puntosM0[i][0]
   pyplot.scatter(f[ccual2],KM0[ccual2],s=70,linewidth=1.5,c='white', ec='black',zorder=3,alpha=alp0)
for i in range(len(puntosM)):
   ccual2 = puntosM[i][0]
   pyplot.scatter(f[ccual2],KM[ccual2],s=70,linewidth=1.5,c='white', ec='black',zorder=3)
pyplot.xlabel(r'$f$ [Hz]',fontsize=font1)
pyplot.legend(loc=1,fontsize=font1)
pyplot.xticks([0,50,100,150,200],fontsize=font1)
pyplot.axis([0,fmax,-4,5.5])
pyplot.yticks([-3,-1,1,3,5],fontsize=font1)
pyplot.tight_layout()

savefig('1Dx2_in1_1_kmax.pdf')

pyplot.figure(figsize=(6,5.5),dpi=100)
pyplot.plot(f,(Mv10-v010)/(Mv10[0]-v010),lw='2',c= 'blue',alpha=alp0)
pyplot.plot(f,(Mv20-v020)/(Mv20[0]-v020),lw='2',c='red',alpha=alp0)
pyplot.plot(f,KM0/KM0[0],lw='2',c= 'k',alpha=alp0)
pyplot.plot(f,(Mv1-v01)/(Mv1[0]-v01),lw='2',c= 'blue', label=r'$ V^*_{max,1}$')
pyplot.plot(f,(Mv2-v02)/(Mv2[0]-v02),lw='2',c='red',label=r'$ V^*_{max,2}$')
pyplot.plot(f,KM/KM[0],lw='2',c= 'k', label=r'$ K^*$')
for i in range(len(puntosM0)):
   ccual2 = puntosM0[i][0]
   pyplot.scatter(f[ccual2],KM0[ccual2]/KM0[0],s=70,linewidth=1.5,c='white', ec='black',zorder=3,alpha=alp0)
for i in range(len(puntosM)):
   ccual2 = puntosM[i][0]
   pyplot.scatter(f[ccual2],KM[ccual2]/KM[0],s=70,linewidth=1.5,c='white', ec='black',zorder=3)
pyplot.xlabel(r'$f$ [Hz]', fontsize =font1)
pyplot.xticks(fontsize=font1)
pyplot.yticks(fontsize=font1)
pyplot.legend(loc=1,fontsize=font2)
pyplot.axis([0,fmax,0,1.45])
pyplot.tight_layout()

savefig('1Dx2_in1_2_kmax.pdf')



pyplot.figure(figsize=(6,4),dpi=100)
pyplot.plot([0,fmax],[pi/2,pi/2],'--', c = 'lightgrey')
pyplot.plot([0,fmax],[0,0],'--', c = 'lightgrey')
pyplot.plot([0,fmax],[-pi/2,-pi/2],'--', c = 'lightgrey')
pyplot.plot([0,fmax],[-pi,-pi],'--', c = 'lightgrey')
pyplot.plot(f,phase10, c = 'blue',lw='2',alpha=alp0)
pyplot.plot(f,phase20, c = 'red',lw='2',alpha=alp0)
pyplot.plot(f,Dphase0, c='black',lw='2',alpha=alp0)
pyplot.plot(f,phase1, c = 'blue',lw='2', label = r'$ \Phi_{ntwk,1}$')
pyplot.plot(f,phase2, c = 'red',lw='2', label = r'$ \Phi_{ntwk,2}$')
pyplot.plot(f,Dphase, c='black',lw='2',label = r'$ \Delta\Phi$')
pyplot.xlabel(r'$f$ [Hz]', fontsize =font1)
pyplot.xticks([0,50,100,150,200],fontsize=font1)
pyplot.yticks([-np.pi,-np.pi/2,0,np.pi/2],labels=[ r'$-\pi$', r'$-\frac{\pi}{2}$','$0$',  r'$\frac{\pi}{2}$'],fontsize=font1)
pyplot.legend(loc=1,fontsize=font2)
pyplot.axis([0,fmax,-3.2,1.7])
pyplot.tight_layout()

savefig('1Dx2_in1_2_1_kmax.pdf')



pyplot.figure(figsize=(6,5.5),dpi=100)
pyplot.plot(0*Xx, Xx, c = 'orange', label=r'$v_1$-nullcline')
pyplot.plot(1/gL1+0*Xx, Xx,'--', c = 'orange')
pyplot.plot(-1/gL1+0*Xx,Xx, '--', c = 'orange')
pyplot.plot(Xx, Nc20,  c = 'green', alpha=alp0)
pyplot.plot(Mv10,Mv20,lw=2,c='black',alpha=alp0)
pyplot.plot(Xx, Nc2,  c = 'green', label=r'$v_2$-nullcline')
pyplot.plot(Mv1,Mv2,lw=2,c='black',label=r'$K$-curve')
for i in range(len(puntosM0)):
   ccual2 = puntosM0[i][0]
   pyplot.scatter(Mv10[ccual2],Mv20[ccual2],s=70,linewidth=1.5,c='white', ec='black',zorder=3,alpha=alp0)
for i in range(len(puntosM)):
   ccual2 = puntosM[i][0]
   pyplot.scatter(Mv1[ccual2],Mv2[ccual2],s=70,linewidth=1.5,c='white', ec='black',zorder=3)
pyplot.xlabel(r'$v_1$', fontsize =font1)
pyplot.ylabel(r'$v_2$', fontsize =font1)
pyplot.legend(loc=4,fontsize=font1)
pyplot.xticks(fontsize=font1)
pyplot.yticks(fontsize=font1)
pyplot.axis([-.9,5.2,-6.2,.2])
pyplot.tight_layout()

savefig('1Dx2_in1_3_kmax.pdf')



pyplot.figure(figsize=(6,5.5),dpi=100)
pyplot.plot(f,angulo,c='darkgray',lw='2',label=r'$ang(\vec{P},\vec{T})$')
pyplot.plot([0,fmax],[pi,pi],'--', c = 'lightgrey')
for i in range(len(puntosM)):
   ccual2 = puntosM[i][0]
   pyplot.scatter(f[ccual2],angulo[ccual2],linewidth=1.5,c='white', ec='black',zorder=3,alpha=1)
pyplot.axis([0,fmax,1.5,3.2])
pyplot.xlabel(r'$f$', fontsize =font1)
pyplot.legend(loc=4 , fontsize =font1)
pyplot.xticks([0,50,100,150,200],fontsize =font1)
pyplot.yticks([np.pi/2,np.pi],labels=[r'$\frac{\pi}{2}$', r'$\pi$'],fontsize =font1)
pyplot.tight_layout()

savefig('1Dx2_in1_4_kmax.pdf')


pyplot.show()


