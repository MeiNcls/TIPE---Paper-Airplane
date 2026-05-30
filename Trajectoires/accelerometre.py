### Traitement des données d'accéléromètre

from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumtrapz


### Import des données



### Extraction des données
def extrait_ligne(ligne,div):
    L=[]
    mot=""
    for e in ligne:
        if e!=div:
            mot+=e
        else:        
            L.append(mot)
            mot=""
    return L

def extrait_donnees(nomFichier):
    f = open(nomFichier, "r") # ouverture en mode lecture
    fichier = f.read() # lecture du contenu du fichier
    lines = fichier.split("\n") # séparation des lignes
    f.close()
    
    div=lines[1][23]
    data = [0]*(len(lines)-1)
    for i in range(1, len(lines)):
        line = lines[i]
        data[i-1]=extrait_ligne(line, div)
    return data

donnees = extrait_donnees("Testacc.txt")
#print(donnees[0:2])

def extraction_colonne(data, j):
    return [float(data[i][j]) for i in range(len(data)-1)]

def extraction_temps(data):
    t=[0]*(len(data)-1)
    t0=float(data[0][0][11:13])*3600+float(data[0][0][14:16])*60+float(data[0][0][17:19])+float(data[0][0][20:23])/1000
    for i in range(1,len(data)-1):
        t[i]=float(data[i][0][11:13])*3600+float(data[i][0][14:16])*60+float(data[i][0][17:19])+float(data[i][0][20:23])/1000
        t[i]-=t0
        t[i]=round(t[i],3)
    return t

cvradiant=np.pi/180

acc_x = np.array(extraction_colonne(donnees, 2))
acc_y = np.array(extraction_colonne(donnees, 3))
acc_z = np.array(extraction_colonne(donnees, 4))
anglex = np.array(extraction_colonne(donnees, 8))*cvradiant
angley = np.array(extraction_colonne(donnees, 9))*cvradiant
anglez = np.array(extraction_colonne(donnees, 10))*cvradiant
time = np.array(extraction_temps(donnees))
g=9.81


### Traitement de l'acceleration pour trouver la trajectoire
def acceleration(acc_X,acc_Y, acc_Z, roll, pitch):
    for i in range(len(acc_x)):
        acc_X[i] = acc_X[i] + g*np.sin(pitch[i])
        acc_Y[i] = acc_Y[i] - g*np.sin(roll[i])*np.cos(pitch[i])
        acc_Z[i] = acc_Z[i] - g*np.cos(roll[i])*np.cos(pitch[i])
    return acc_X, acc_Y, acc_Z

acc_x, acc_y, acc_z = acceleration(acc_x, acc_y, acc_z, anglex, angley)

def moyenne(liste):
    m = 0
    for i in range(len(liste)-1):
        m += (liste[i+1]-liste[i])
    m /= (len(liste)-1)
    return round(m, 4)

dt=moyenne(time)

def integration(signal, dt):
    return cumtrapz(signal, dx=dt, initial=0)

vx = integration(acc_x, dt)
vy = integration(acc_y, dt)
vz = integration(acc_z, dt)
x = integration(vx, dt)
y = integration(vy, dt)
z = integration(vz, dt)

