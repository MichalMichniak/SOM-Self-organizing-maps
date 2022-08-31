from random import random
import numpy as np
import matplotlib.pyplot as plt
from SOM import SOM

def draw(som,x):
    som_weights = som.get_neurons()#np.array([[[ i if k == 0 else j for i in range(10)]for j in range(10)]for k in range(2)])
    plt.plot(x[:,0],x[:,1],"r.")
    for i in range(len(som_weights[0])):
        for j in range(len(som_weights[0,0,:])):
            plt.plot(som_weights[0,i,j],som_weights[1,i,j],"bo")
            if i>0:
                plt.plot([som_weights[0,i-1,j],som_weights[0,i,j]],[som_weights[1,i-1,j],som_weights[1,i,j]],"b-")
            if j>0:
                plt.plot([som_weights[0,i,j-1],som_weights[0,i,j]],[som_weights[1,i,j-1],som_weights[1,i,j]],"b-")
            if j>0 and i>0:
                plt.plot([som_weights[0,i-1,j-1],som_weights[0,i,j]],[som_weights[1,i-1,j-1],som_weights[1,i,j]],"b-")
    
    plt.show()
def main():
    x = [[np.random.uniform()+0.5,np.random.uniform()+0.8] for i in range(1000)]
    #x += [[np.random.uniform()+0.5,np.random.uniform()-0.8] for i in range(1000)]
    x = np.array(x)
    plt.plot(x[:,0],x[:,1],"r.")
    som = SOM(10,10,2)#20,20
    #draw(som,x)
    for i in range(2):
        for j in x:
            #draw(som,x)
            som.teach(j)#x[np.random.randint(0,1999)]
    draw(som,x)
    
    pass


if __name__ == '__main__':
    main()