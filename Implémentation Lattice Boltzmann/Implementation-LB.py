## Méthode de Lattice Boltzmann (2D)
# Code fourni
#Site d'origine : https://readmedium.com/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c

#Bibliothèques 
import numpy as np
from matplotlib import pyplot as plt

#défintion des contours

img=plt.imread("wing_foil.png")
plt.close(1)
plt.figure(1)
plt.imshow(img)
plt.show()

def contour(img):
    n,p=n,p=img.shape[0],img.shape[1]
    L=[[0]*p for i in range (n)]
    for i in range (n):
        for j in range(p):
            c=(img[i,j,1],img[i,j,2],img[i,j,3])
            if c==(1,1,1):
                L[i][j]=1
    return L


C=contour(img)

# Simulation parameters
Nx          = 400    # resolution x-dir
Ny          = 100    # resolution y-dir
rho0        = 100    # average density
tau         = 0.6    # collision timescale
Nt          = 6000   # number of timesteps

# Lattice speeds / weights
NL = 9
idxs = np.arange(NL)
cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
X, Y = np.meshgrid(range(Nx), range(Ny))

# Initial Conditions - flow to the right with some perturbations
F = np.ones((Ny,Nx,NL)) + 0.01*np.random.randn(Ny,Nx,NL)
F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
rho = np.sum(F,2)
for i in idxs:
  F[:,:,i] *= rho0 / rho

# Cylinder boundary
cylinder2=np.full((Ny,Nx),False)

for i in range(Ny):
    for j in range(Nx):
        if C[i][j]==0:
            cylinder2[i][j]=True


cylinder = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/4)**2


# Simulation Main Loop
for it in range(Nt):

  # Drift
  for i, cx, cy in zip(idxs, cxs, cys):
    F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
    F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)

  # Set reflective boundaries
  bndryF = F[cylinder,:]
  bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]

  # Calculate fluid variables
  rho = np.sum(F,2)
  ux  = np.sum(F*cxs,2) / rho
  uy  = np.sum(F*cys,2) / rho

  # Apply Collision
  Feq = np.zeros(F.shape)
  for i, cx, cy, w in zip(idxs, cxs, cys, weights):
    Feq[:,:,i] = rho*w* (1 + 3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2)

  F += -(1.0/tau) * (F - Feq)

  # Apply boundary
  F[cylinder,:] = bndryF

  if (it%10==0):
      plt.imshow(np.sqrt(ux**2+uy**2))
      plt.pause(0.005)
      plt.cla()

