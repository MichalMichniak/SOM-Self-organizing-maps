
from cmath import inf
import numpy as np
from random import random

class SOM:
    def __init__(self,n,m,l,starting_order = 1e-8,ni_0 = 1,sigma_0 = 3,T1 = 3, T2 = 1, dt = 0.02,neigh_mul = 8):
        self.weights_ = np.array([[[random() * starting_order for k in range(n)] for j in range(m)] for i in range(l)])
        self.weights_ = np.array([[[float(j) * starting_order if i ==0 else float(k) * starting_order for k in range(n)] for j in range(m)] for i in range(l)])
        self.time = 0
        self.dt = dt
        self.neighbours = max(n,m)*neigh_mul
        self.neighbours_max = max(n,m)*neigh_mul
        self.T1 = T1
        self.T2 = T2
        self.sigma_0 = sigma_0
        self.ni_0 = ni_0

    def find_min(self,x):
        """
        find nearest vector of x in SOM
        """
        err = np.zeros(self.weights_[0].shape)
        minimum = inf
        indx = (0,0)
        for i in range(len(err)):
            for j in range(len(err[0])):
                err[i,j] = np.sqrt(np.sum((self.weights_[:,i,j] - x)*(self.weights_[:,i,j] - x)))
                if err[i,j]<minimum:
                    indx = (i,j)
                    minimum = err[i,j]
        return indx
    
    def get_neurons(self):
        return self.weights_.copy()

    def update_weights(self,v,x):
        """
        update weights (vectors) of SOM

        args:
            v : Tuple[int,int] - coordinates of nearest vector to x in weight matrix
            x : numpy.ndarray - vector to learn
        """
        X,Y = self.weights_[0].shape
        ni = self.ni_0*np.exp(-self.time/self.T1)
        sigma_square = (self.sigma_0*np.exp(-self.time/self.T2))**2
        for i in range(X):
            for j in range(Y):
                if (abs(v[0] - i) < self.neighbours and abs(v[1] - j) < self.neighbours):
                    di = abs(v[0] - i) + abs(v[1] - j)
                    hi = (b if b != None else 100) if (b:=np.exp(-di**2/(2*sigma_square))) <100 else 100
                    self.weights_[:,i,j] += ni*hi*(x - self.weights_[:,i,j])
        self.time+=self.dt
        self.neighbours = max(self.neighbours_max - int(self.time*10),1)
        
    
    def teach(self,x):
        """
        one iteration of learning algorythm

        args:
            x : numpy.ndarray - vector to learn
        """
        v = self.find_min(x)
        self.update_weights(v,x)


