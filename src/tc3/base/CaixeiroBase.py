import numpy as np
import matplotlib.pyplot as plt

def distance(p1,p2):    
    return np.sqrt(np.sum((p1-p2)**2))

def generate_points(N): 
    x_partition = np.random.uniform(-10, 10, size=(N,3))
    y_partition = np.random.uniform(0, 20, size=(N,3))
    z_partition = np.random.uniform(-20, 0, size=(N,3))
    w_partition = np.random.uniform(0, 20, size=(N,3))

    x1 = np.array([[20,-20,-20]])
    x1 = np.tile(x1,(N,1))
    x_partition = x_partition+x1

    x1 = np.array([[-20,20,20]])
    x1 = np.tile(x1,(N,1))
    y_partition = y_partition+x1

    x1 = np.array([[-20,20,-20]])
    x1 = np.tile(x1,(N,1))
    z_partition = z_partition+x1

    x1 = np.array([[20,20,-20]])
    x1 = np.tile(x1,(N,1))
    w_partition = w_partition+x1   
    return np.concatenate((x_partition,y_partition,z_partition,w_partition), axis=0)

# 
K = 20
points = generate_points(K)
I = np.random.permutation(K*4)
inicial = I[0]
p_origem = points[inicial,:].reshape(1,3)
points = np.delete(points,inicial,axis=0)

Populacao = np.empty((0,points.shape[0]))
N = 50 #tamanho da pop.
for i in range(N):
    individuo = np.random.permutation(points.shape[0]).reshape(1,points.shape[0])
    Populacao = np.concatenate((Populacao,individuo))

vetor_origem= np.tile(np.array([[int(inicial)]]),(N,1))

#essa matriz pode ser utilizado para aptidao:
caminhos = np.concatenate((vetor_origem,Populacao,vetor_origem),axis=1)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], c='#248DD2', marker='o')
ax.scatter(p_origem[0:,0], p_origem[0:,1], p_origem[0:,2], c='green', marker='x',linewidth=3,s=30)

#exemplo caminho a partir da origem.
p2 = points[0,:].reshape(1,3)
line, = ax.plot([p_origem[0,0],p2[0,0]],[p_origem[0,1],p2[0,1]],[p_origem[0,2],p2[0,2]],color='k')

plt.tight_layout()
plt.show()

