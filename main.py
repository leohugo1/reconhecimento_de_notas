from layer import  Activation,Dense,Layer,Tanh
import numpy as np


X=np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y=np.reshape([[0],[1],[1],[0]],(4,1,1))

network=[
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]
epocas=1000
learning_rate=0.1

for e in range(epocas):
    error=0

    for x,y in zip(X,Y):
        output=x 
        for layer in network:
            output=layer.foward(output)
        
        error += Tanh.mse(y,output)

        grad=Tanh.mse_prime(y,output)

        for layer in reversed(network):
            grad=layer.backward(grad,learning_rate)
    
    error /= len(X)

    print('%d/%d, error=%f' % (e + 1,epocas,error))
