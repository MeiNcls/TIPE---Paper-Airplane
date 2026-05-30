### Traitement des données d'accéléromètre

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

f = open("Testacc.txt", "r") # ouverture en mode lecture
data = f.read() # lecture du contenu du fichier
lines = data.split("\n") # séparation des lignes
f.close()
print(len(lines))