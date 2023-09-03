import numpy as np
import matplotlib.pyplot as plt

x = np.array([480,500,380,1100,1100,230,490,250,300,510 ])
x.shape = (len(x),1)
y = np.array([180,150,170,350,460,60,240,90,110,250])
y.shape = (len(y),1) 
plt.scatter(x,y,color='orange')
X = np.concatenate((np.ones((len(x),1)),x),axis=1)

#Estimação do modelo:
B = np.linalg.pinv(X.T@X)@X.T@y

x_axis = np.linspace (0,1200,1200) 
x_axis.shape = (len(x_axis),1)
ones = np.ones((len(x_axis),1))
X_new = np.concatenate((ones,x_axis), axis=1)
Y_pred = X_new@B
plt.plot(x_axis,Y_pred, color='blue')
plt.show()