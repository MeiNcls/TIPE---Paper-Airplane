##importations de bibliothèques
import sympy as sp
from sympy.matrices import Matrix
from pylab import *


### constantes
g = 9.81 #pesanteur
a = 2e-2 #m
x0 = Matrix([[1,0,0]])
y0 = Matrix([[0,1,0]])
z0 = Matrix([[0,0,1]])

rho = 1.2 #kg/m3
S = 50e-4 #m2
Cx = 0.5 #à changer
Cz = 0.9 #à changer
m = 5e-3 #kg

#matrice d'inertie en G dans la base de l'avion

#A1, B1, C1, D1, E1, F1 = (2e-5,2e-5,2e-5,1e-7,0,1e-7) #à changer plus tard
A1, B1, C1, D1, E1, F1 = sp.symbols("A1, B1, C1, D1, E1, F1")
I = Matrix([[A1,-F1,-E1],
          [-F1,B1,-D1],
          [-E1,-D1,C1]])

### rotations
# vecteurs dans la base de l'avion

def u(psi,theta,phi) :
    a =  Matrix([sp.cos(theta),
                sp.sin(theta)*sp.cos(psi),
                sp.sin(theta)*sp.sin(psi)])
    return a.transpose()

def v(psi,theta,phi) :
    a = Matrix([-sp.cos(phi)*sp.sin(theta),
               sp.cos(phi)*sp.cos(theta)*sp.cos(psi)-sp.sin(phi)*sp.sin(psi),
               sp.cos(phi)*sp.cos(theta)*sp.sin(psi)+sp.sin(phi)*sp.cos(psi)])
    return a.transpose()

def w(psi,theta,phi) :
    a = Matrix([sp.sin(phi)*sp.sin(theta),
               -sp.cos(phi)*sp.sin(psi)-sp.sin(phi)*sp.cos(theta)*sp.cos(psi),
               sp.cos(phi)*sp.cos(psi)-sp.sin(phi)*sp.cos(theta)*sp.sin(psi)])
    return a.transpose()

def omega(psi,theta,phi) : # vecteur rotation de l'avion dans la base B de l'avion
    o = Matrix([phi.diff()+psi.diff()*sp.cos(theta),
               theta.diff()*sp.sin(phi)-psi.diff()*sp.sin(theta)*sp.cos(phi),
               theta.diff()*sp.cos(phi) + psi.diff()*sp.sin(theta)*sp.sin(phi)])
    return o

t = sp.Symbol('t')
psi = sp.Function("psi")(t)
phi = sp.Function("phi")(t)
theta = sp.Function("theta")(t)
OmegaB = omega(psi,theta,phi)
u,v,w = u(psi,theta,phi), v(psi,theta,phi), w(psi,theta,phi)


SigmaB = I*OmegaB
X,Y,Z = SigmaB[0],SigmaB[1],SigmaB[2]
usx0 = u.dot(x0)
usy0 = u.dot(y0)
usz0 = u.dot(z0)
vsx0 = v.dot(x0)
vsy0 = v.dot(y0)
vsz0 = v.dot(z0)
wsx0 = w.dot(x0)
wsy0 = w.dot(y0)
wsz0 = w.dot(z0)

Deltax0 = X.diff(t)*usx0 + Y.diff(t)*vsx0 + Z.diff(t)*wsx0 + X*usx0.diff(t) + Y*vsx0.diff(t) + Z*wsx0.diff(t)
Deltay0 = X.diff(t)*usy0 + Y.diff(t)*vsy0 + Z.diff(t)*wsy0 + X*usy0.diff(t) + Y*vsy0.diff(t) + Z*wsy0.diff(t)
Deltaz0 = X.diff(t)*usz0 + Y.diff(t)*vsz0 + Z.diff(t)*wsz0 + X*usz0.diff(t) + Y*vsz0.diff(t) + Z*wsz0.diff(t)
Delta0 = Matrix([[Deltax0,Deltay0,Deltaz0]]).T


def f(Y,temps) :
    #depactage
    P, Q, dP, dQ = Y[0], Y[1], Y[2], Y[3]
    x_t,y_t,z_t,dx_t,dy_t,dz_t = P[0],P[1],P[2],dP[0],dP[1],dP[2]
    psi_t,theta_t,phi_t,dpsi_t,dtheta_t,dphi_t = Q[0],Q[1],Q[2],dQ[0],dQ[1],dQ[2]
    #Sigmax, Sigmay,Sigmaz = S[0],S[1],S[2]
    #usx,usy,usz,vsx,vsy,vsz,wsx,wsy,wsz = V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7],V[8]
    
    x1,x2,x3 = sp.symbols('x1 x2 x3')
    D = Delta0.subs({psi:psi_t , theta:theta_t , phi:phi_t , 
                     psi.diff(t):dpsi_t , theta.diff(t):dtheta_t , phi.diff(t):dphi_t,
                     psi.diff(t,2):x1 , theta.diff(t,2):x2, phi.diff(t,2):x3,
                     A1:2e-5, B1:2e-2, C1:2e-5, D1:1e-7, E1:0, F1:1e-7})
    vx = vsx0.subs({psi:psi_t , theta:theta_t , phi:phi_t }) #pas la vitesse obviously
    vy = vsy0.subs({psi:psi_t , theta:theta_t , phi:phi_t })
    vz = vsz0.subs({psi:psi_t , theta:theta_t , phi:phi_t }) 
    #peur que ça marche pas car je remplace une fonction par une valeur numérique
    
    dx0, dy0, dz0 = D[0], D[1], D[2]
    
    vitesse_carre = dx_t**2 + dy_t**2 + dz_t**2
    Fx = (1/2)*rho*S*Cx*vitesse_carre
    Fz = (1/2)*rho*S*Cz*vitesse_carre
    
    eqx = sp.Eq(Fz*a*vx, dx0)
    eqy = sp.Eq(Fz*a*vy, dy0)
    eqz = sp.Eq(Fz*a*vz, dz0)
    sol = list(sp.linsolve([eqx,eqy,eqz],(x1,x2,x3)))
    ddQ = list(sol)[0]
    ddQ = array(ddQ)
    
    ddP_symb = (1/m)*(-m*g*z0 -Fx*u + Fz*w)
    ddP = list(ddP_symb.subs({psi:psi_t , theta:theta_t , phi:phi_t }))
    ddP = array(ddP)
    print(ddQ)
    dY = array([dP,dQ,ddP,ddQ])
    return dY

def Euler_explicite(f,Y0,dt,n):
    temps = linspace(0,n*dt,n)
    P = zeros((3,n))
    dP = zeros((3,n))
    Q = zeros((3,n))
    dQ = zeros((3,n))
    Y = array([P,Q,dP,dQ])
    Y[:,:,0] = Y0
    
    for i in range (n-1) :
        Y[:,:,i+1] = Y[:,:,i] + f(Y[:,:,i],temps)*dt
    
    return Y

dtr = pi/180
Y0 = array([[0,0,0],[0,0,0],[1,0.1,0.01],[2*dtr,10*dtr,5*dtr]])
dt = 0.1
n = 1000
Y_de_l_espoir = Euler_explicite(f,Y0,dt,n)
