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

# Nonlinear inhibitory networks of two 1D cells
# Fig 5
# ej = 1 -> A, ej = 2 -> B, ej = 3 -> C

ej = 1

gL1 = 0.1
gL2 = 0.1
gsyn12 = 0.02
gsyn21 = 0.02
e12 = -20.
e21 = -20.
g1 =  0. ## 
g2 = 0. ##  
tau1 = 1.
tau2 = 1.


if ej == 1:
   ain = 0.1
if ej == 2:
   ain = 0.2
if ej == 3:
   ain = 1
f = [.5*fx for fx in range(1,40)]+[20+fx for fx in range(0,30)]+[50+2*fx for fx in range(0,40)]+[130+5*fx for fx in range(0,14)]+[200,210,250,300,400,500,600,800,1000]
fmax = 200


ff = [1,20,100]
if ej == 3:
    ff = [1,15,100]

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

v01 = root[0]
v02 = root[1]

Xx = np.arange(-15, 15, .1)

Nc1 = Ssinv((-gL1*Xx-g1*Xx)/(gsyn12*(Xx-e12))) 
Nc11 = Ssinv((-gL1*Xx-g1*Xx+ain)/(gsyn12*(Xx-e12)))
Nc12 = Ssinv((-gL1*Xx-g1*Xx-ain)/(gsyn12*(Xx-e12)))
Nc2 = -gsyn21*Ss(Xx)*e21/(-gL2-g2-gsyn21*Ss(Xx))

	
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

solx1=[]
solx2=[]
ffs=[]
ttt = []

print()

for  k in range(0,len(f)):
  tt = t[:(int(4*1000./f[k]))*cant]
  if f[k] > 25:
    tt = [tx*dt for tx in range(0, 800*cant)]     
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
  if len(max1)>0:
     phase1[k] = ((max1[0][1]-maxf[1])*pi*f[k]/500.)
  if len(max2)>0:
     phase2[k] = ((max2[0][1]-maxf[1])*pi*f[k]/500.)
  Z1[k] = (max1[0][0] - min1[0][0])/(2*ain)
  Z2[k] = (max2[0][0] - min2[0][0])/(2*ain)
  Mv1[k] = max1[0][0]
  Mv2[k] = max2[0][0]
  mv1[k] = min1[0][0]
  mv2[k] = min2[0][0]
  K[k] = Z2[k]/Z1[k]
  KM[k] = (Mv2[k]-v02)/(Mv1[k]-v01)


Dphase=numpy.zeros(len(f))
for i in range(0,len(phase1)):
    Dphase[i] = phase2[i]-phase1[i]

for i in range(0,len(phase1)):
    if phase1[i]>pi:
       phase1[i]=phase1[i]-2*pi
    if phase2[i]<0:
       phase2[i]=phase2[i]+2*pi
    if Dphase[i]<0:
       Dphase[i]=Dphase[i]+2*pi



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

puntos = []

for k in range(1,len(f)-1):
   if K[k]<K[k-1] and K[k]<K[k+1]:
      puntos.append([k,Z1[k],Z2[k],K[k]])
   if K[k]>K[k-1] and K[k]>K[k+1]:
      puntos.append([k,Z1[k],Z2[k],K[k]])

print('puntos',puntos)

puntosM = []

for k in range(1,len(f)-1):
   if KM[k]<KM[k-1] and KM[k]<KM[k+1]:
      puntosM.append([k,Mv1[k],Mv2[k],KM[k]])
   if KM[k]>KM[k-1] and KM[k]>KM[k+1]:
      puntosM.append([k,Mv1[k],Mv2[k],KM[k]])

print('puntos',puntosM)


#####################################################
# trajectories for representative values of f


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
  for iii in range(3):
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


puntosK = []
for i in range(len(ff)):
   for ii in range(len(f)):
      if ff[i]==f[ii]:
         puntosK.append([ii,Mv1[ii],Mv2[ii]])

for i in range(len(puntosK)):
    print(puntosK[i][2])


fint=time.time()

print('tiempo=', fint-inicio)

########################################################

font1 = 24
font2 = 22

fig, axs = plt.subplots(2,1,figsize=(6,5.5),dpi=100)
axs[0].plot(f,Mv1,c= 'blue', label=r'$ V_{max,1}$')
axs[0].plot(f,Mv2,c='red',label=r'$ V_{max,2}$')
axs[0].plot(f,KM,lw='2',c= 'k', label=r'$ K$')
for i in range(len(puntosM)):
   ccual = puntosM[i][0]
   axs[0].scatter(f[ccual],KM[ccual],s=70,linewidth=1.5,c='white', ec='black',zorder=3,alpha=1)
axs[0].xaxis.set_tick_params(labelsize=font1)
axs[0].yaxis.set_tick_params(labelsize=font1)
axs[0].legend(loc=1,fontsize=font2)
if ej ==1 :
    axs[0].axis([0,fmax,-1.05,.75])
if ej == 2:
    axs[0].axis([0,fmax,-1.05,1.85])
if ej == 3:
    axs[0].axis([0,fmax,-1.25,10.5])
axs[1].plot(f,(Mv1-v01)/(Mv1[0]-v01),c= 'blue', label=r'$ V^*_{max,1}$')
axs[1].plot(f,(Mv2-v02)/(Mv2[0]-v02),c='red',label=r'$ V^*_{max,2}$')
axs[1].plot(f,KM/KM[0],lw='2',c= 'k',label=r'$ K^*$')
axs[1].set_xlabel(r'$f$ [Hz]', fontsize =font1)
for i in range(len(puntosM)):
   ccual = puntosM[i][0]
   axs[1].scatter(f[ccual],KM[ccual]/KM[0],s=70,linewidth=1.5,c='white', ec='black',zorder=3,alpha=1)
axs[1].xaxis.set_tick_params(labelsize=font1)
axs[1].yaxis.set_tick_params(labelsize=font1)
axs[1].legend(loc=1,fontsize=font2)
axs[1].axis([0,fmax,0,1.2])
axs[1].set_title('Normalized', fontdict={'fontsize': font1+2, 'fontweight': 'medium'},loc='left')
if ej==3:
   axs[1].axis([0,fmax,-.72,1.4])
fig.tight_layout()



if ej == 1:
   savefig('NL1_1_kmax.pdf')
if ej == 2:
   savefig('NL2_1_kmax.pdf')
if ej == 3:
   savefig('NL3_1_kmax.pdf')




pyplot.figure(figsize=(6,5.5),dpi=100)
pyplot.plot(Xx, Nc1, c = 'orange', label=r'$v_1$-nullcline')
pyplot.plot(Xx, Nc11,'--', c = 'orange')
pyplot.plot(Xx, Nc12, '--', c = 'orange')
pyplot.plot(Xx, Nc2,  c = 'green', label=r'$v_2$-nullcline')
pyplot.plot(solx1[0],solx2[0],'b', label='trajectories')
for i in range(1,len(ff)):
    pyplot.plot(solx1[i],solx2[i],'b')
pyplot.plot(Mv1+0*v01,Mv2+0*v02,lw=2,c='black')
#pyplot.plot(mv1+0*v01,mv2+0*v02,lw=2,c='black')
for i in range(len(puntosK)):
   pyplot.scatter(puntosK[i][1]+0*v01,puntosK[i][2]+0*v02,s=80,marker='*',c='gray',zorder=3,alpha=1)
for i in range(len(puntosM)):
   ccual2 = puntosM[i][0]
   pyplot.scatter(Mv1[ccual2],Mv2[ccual2],s=70,linewidth=1.5,c='white', ec='black',zorder=3,alpha=1)
if ej ==1:
   pyplot.text(.75,-2.25, r'$f=1$', fontsize =font1)
   pyplot.text(.5,-1, r'$f=20$', fontsize =font1)
   pyplot.text(-4,-2.1, r'$f=100$', fontsize =font1)
   pyplot.text(.25,0, r'$K$-curve', fontsize =font1)
   pyplot.text(-4.75,0.5, r'$A_{in}=0.1$', fontsize =font1)
   pyplot.arrow(-1.16,-1.02,-2,-.75,width=0.01,head_width=0.1,fc='k', ec='k',zorder=2)
   pyplot.arrow(-0.4,-1.2,.7,0.25,width=0.01,head_width=0.1,fc='k', ec='k',zorder=2)
if ej ==2:
   pyplot.text(0,-3, r'$f=1$', fontsize =font1)
   pyplot.text(.6,-1.5, r'$f=20$', fontsize =font1)
   pyplot.text(-4,-2.1, r'$f=100$', fontsize =font1)
   pyplot.text(.25,0.15, r'$K$-curve', fontsize =font1)
   pyplot.text(-4.75,0.5, r'$A_{in}=0.2$', fontsize =font1)
   pyplot.arrow(-1.3,-1.02,-2,-.75,width=0.01,head_width=0.1,fc='k', ec='k',zorder=2)
if ej ==3:
   pyplot.text(4,-3.75, r'$f=1$', fontsize =font1)
   pyplot.text(4,-1.5, r'$f=15$', fontsize =font1)
   pyplot.text(-10,-2.5, r'$f=100$', fontsize =font1)
   pyplot.text(4,.2, r'$K$-curve', fontsize =font1)
   pyplot.text(-10.5,.5, r'$A_{in}=1$', fontsize =font1)
   pyplot.arrow(-2.3,-1.3,-5,-.75,width=0.01,head_width=0.1,fc='k', ec='k',zorder=2)
pyplot.xlabel(r'$v_1$', fontsize =font1)
pyplot.ylabel(r'$v_2$', fontsize =font1)
pyplot.legend(loc=3,fontsize=font2)
pyplot.xticks(fontsize=font1)
pyplot.yticks(fontsize=font1)
if ej == 1:
    pyplot.axis([-5,3,-4,1])
if ej == 2:
    pyplot.axis([-5,3,-4,1])
if ej == 3:
    pyplot.axis([-12,11,-5,1.])
pyplot.tight_layout()

if ej == 1:
   savefig('NL1_3_kmax.pdf')
if ej == 2:
   savefig('NL2_3_kmax.pdf')
if ej == 3:
   savefig('NL3_3_kmax.pdf')


prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",color='b',lw=1.5,
            shrinkA=0,shrinkB=0)
prop2 = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",color='r',lw=1.5,
            shrinkA=0,shrinkB=0)


fig, ax1 = plt.subplots(figsize=(6,5.5),dpi=100)
ax1.plot(Xx, Nc1, c = 'orange')
ax1.plot(Xx, Nc2,  c = 'green')
ax1.plot(Mv1,Mv2,lw=2,c='black')
for i in range(len(puntosK)):
   ax1.scatter(puntosK[i][1]+0*v01,puntosK[i][2]+0*v02,s=80,marker='*',c='gray',zorder=3,alpha=1)
for i in range(len(puntosK)):
   ccual = puntosK[i][0]
   norm = np.sqrt(dMv1[ccual]**2+dMv2[ccual]**2)
   if ej < 3:
      norm = 3*norm
   if ej == 3:
      norm = .5*norm
   ax1.annotate("", xy=(puntosK[i][1]+0*v01,puntosK[i][2]+0*v02), xytext=(v01,v02), arrowprops=prop)
   tv = [dMv1[ccual]/norm,dMv2[ccual]/norm]
   ax1.annotate("", xy=(tv[0]+Mv1[ccual],tv[1]+Mv2[ccual]), xytext=(Mv1[ccual],Mv2[ccual]), arrowprops=prop2)
for i in range(len(puntosM)):
   ccual2 = puntosM[i][0]
   ax1.scatter(Mv1[ccual2],Mv2[ccual2],s=70,linewidth=1.5,c='white', ec='black',zorder=3,alpha=1)
ax1.set_xlabel(r'$v_1$', fontsize =font1)
ax1.set_ylabel(r'$v_2$', fontsize =font1)
ax1.xaxis.set_tick_params(labelsize=font1)
ax1.yaxis.set_tick_params(labelsize=font1)
if ej ==1:
   ax1.text(-.25,-1, r'$K$-curve', fontsize =font1)
   ax1.axis([-1.5,.75,-1.25,.5])
if ej ==2:
   ax1.text(.25,-1, r'$K$-curve', fontsize =font1)
   ax1.axis([-1.5,2,-1.25,.5])
if ej ==3:
   ax1.text(3.5,-1, r'$K$-curve', fontsize =font1)
   ax1.axis([-2,10.5,-1.25,1])
ax2 = inset_axes(ax1, width="70%", height="22%", loc="upper center")
ax2.plot(f,angulo,c='darkgray',lw='2')
ax2.text(120,2.5, r'$ang(\vec{P},\vec{T})$', fontsize =20)
ax2.plot([0,fmax],[pi,pi],'--', c = 'lightgrey')
for i in range(len(puntosK)):
   ccual = puntosK[i][0]
   ax2.scatter(f[ccual],angulo[ccual],s=80,marker='*',c='gray', zorder=3,alpha=1)
for i in range(len(puntosM)):
   ccual2 = puntosM[i][0]
   ax2.scatter(f[ccual2],angulo[ccual2],linewidth=1.5,c='white', ec='black',zorder=3,alpha=1)
ax2.axis([0,fmax,2.2,3.2])
ax2.set_xlabel(r'$f$ [Hz]', fontsize =20)
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.set_xticks([0,50,100,150,200])
ax2.set_yticks([3*np.pi/4,np.pi],labels=[r'$\frac{3}{4}\pi$', r'$\pi$'])
fig.tight_layout()


if ej == 1:
   savefig('NL1_4_kmax.pdf')
if ej == 2:
   savefig('NL2_4_kmax.pdf')
if ej == 3:
   savefig('NL3_4_kmax.pdf')



pyplot.show()




