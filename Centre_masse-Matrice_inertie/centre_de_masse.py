#Bibliothèques 
from math import *
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

img=plt.imread("IMG_0534.png")
im13c=plt.imread("IMG_0543.png")
im13a=plt.imread("IMG_0544.png")
print(len(im13a),len(im13a[0]))

r=80 #80g/m2
rho={(234,64,143):r, (255,200,67): 2*r, (164,129,247):4*r, (143,162,248):12*r, (157,201,166):24*r}
rho2={(234,64,143):r, (255,200,67): r, (164,129,247):r, (143,162,248):r, (157,201,166):r}
e13c=(4.8/100)/len(im13c)
e13a=(10/100)/len(im13a[0])

plt.close(1)
plt.figure(1)
plt.imshow(im13a)
plt.show()
img.shape
img[10][10]
b=img[0,0,3]
print(b, b*255)
print(e13a*len(im13a))

#Fonctions annexes 

#produit matriciel d'une ligne par une colonne 
def produit_matcl(L,C):
    s=0
    for i in range(len(L)):
        s+=L[i]*C[i]
    return s

#fonction qui vérifie si chaque pixel est bien dans le bon interval
def clamp(pixel):
    for i in range(len(pixel)):
        if pixel[i]<0:
            pixel[i] = 0
        else:
            if pixel[i]>1:
                pixel[i] = 1
                
#fonction qui permet de rechercher si un tupple rgb de couleur approximé est bien dans un dictionnaire
def tupple_approx(d:dict, t:(int) ): 
    a,b,c=t[0],t[1],t[2]
    for i in range(-1,2):
        for j in range (-1,2):
            for k in range(-1,2):
                e,f,g= a+i, b+j, c+k
                if 0<=e<=255 and 0<=f<=255 and 0<=g<=255:
                    if (e,f,g) in d :
                        return (True, d[(e,f,g)])
    return (False,())

#Matrice de l'image avec des valeurs entières entre 0 et 255
def img_255(img):
    n,p=img.shape[0],img.shape[1]
    imgCopie=[[[0,0,0] for j in range(p)] for i in range (n)]
    for i in range(n):
        for j in range(p):
            for k in range(3):
                imgCopie[i][j][k]=int(img[i,j,k]*255)
    return imgCopie

im13a255=img_255(im13a)
print(im13a255[0][0])
print(im13a255[100][200])

im13a255=img_255(im13a)
im13c255=img_255(im13c)

def centre_inertie(imgf255, G, echelle):
    """
    IN : image avec les couleurs entre 0 et 255 et le grammage du papier utilisé 
    sous forme de dictionnaire, où chaque clé est une couleur sous forme (r,g,b) 
    et la valeur associé est le grammage (g/cm)d'une couche de la couleur associée 
    ainsique l'échelle : le gain en pt/cm de la photo (ie 1 pixel= ? mm dans la 
    réalité)
    OUT : renvoie un vecteur numy qui donne la position du vecteur AG (A origine)"""
    x=np.array([1.,0.])
    y=np.array([0.,1.])
    AG=np.array([0.,0.])
    m=0.0
    dS=echelle**2
    S=0
    n,p=len(imgf255),len(imgf255[0])
    for i in range (n):
        for j in range(p):
            b,gm=tupple_approx(G, tuple(imgf255[i][j]))
            if b :
                dm=dS*gm
                m+=dm
                S+=dS
                AG+=dm*(i*x+j*y)
    return AG/m,m,S

OG,m,S=centre_inertie(im13a255,rho2, e13a)
print([int(OG[0]),int(OG[1])], m,S)

OG,m1,S=centre_inertie(im13a255,rho, e13a)
print([int(OG[0]),int(OG[1])], m1,S)

OG,m2,S=centre_inertie(im13c255,rho, e13c)
print([int(OG[0]),int(OG[1])], m2,S)

mtot= m1+m2
print(mtot)

# Calcul de la matrice d'inertie d'un plan de l'avion 
#On considère que le plan a pour normale z

def matrice_inertie_plan(imgf255, G, echelle):
    """
    IN : image avec les couleurs entre 0 et 255 et le grammage du papier utilisé 
    sous forme de dictionnaire, où chaque clé est une couleur sous forme (r,g,b) 
    et la valeur associé est le grammage (g/cm)d'une couche de la couleur associée 
    ainsique l'échelle : le gain en pt/m de la photo (ie 1 pixel= ? m dans la 
    réalité)
    OUT : Renvoie la matrce d'inertie du plan de l'avion """
    A,B,C,D,E,F=0,0,0,0,0,0
    X=np.array([1.,0.])
    Y=np.array([0.,1.])
    OG,m=centre_inertie(imgf255, G, echelle)
    dS=echelle**2
    m=0
    n,p=len(imgf255),len(imgf255[0])
    for i in range (n):
        for j in range(p):
            b,gm=tupple_approx(G, tuple(imgf255[i][j]))
            AG=OG - i*echelle*X+j*echelle*Y
            x,y=AG[0],AG[1]
            if b :
                dm=dS*gm
                m+=dm
                A+=(y**2)*dm
                B+=(x**2)*dm
                C+=(x**2+y**2)*dm
                F+=x*y*dm
    return [[A,-F,-E],[-F,B,-D],[-E,-D,C]]

print(matrice_inertie_plan(im13c255,rho,e13c))