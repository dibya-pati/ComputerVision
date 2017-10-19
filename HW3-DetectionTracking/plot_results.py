import pandas as pd
from matplotlib import pyplot as plt

CAM = pd.read_csv('output_camshift.txt')
CAM.columns = ['index', 'x', 'y']
Kalman = pd.read_csv('output_kalman.txt')
Kalman.columns = ['index', 'x', 'y']
particle = pd.read_csv('output_particle.txt')
particle.columns = ['index', 'x', 'y']
optical = pd.read_csv('output_of.txt')
optical.columns = ['index', 'x', 'y']
plt.scatter(x='x', y='y', data=CAM)
plt.scatter(x='x', y='y', data=Kalman)
plt.scatter(x='x', y='y', data=particle)
plt.scatter(x='x', y='y', data=optical)
plt.legend(['CAM', 'Kalman', 'Particle', 'Optical'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()