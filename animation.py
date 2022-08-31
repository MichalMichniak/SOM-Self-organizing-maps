from email.policy import default
from random import random
import numpy as np
import matplotlib.pyplot as plt
from SOM import SOM
import pygame as pg

WIDTH = 800
HEIGH = 800

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

class MainWindow:
    def __init__(self,som : SOM, x) -> None:
        pg.init()
        pg.display.set_caption("game")
        self.font = pg.font.SysFont(None, 24)
        self.win = pg.display.set_mode((WIDTH, HEIGH))
        self.som = som
        self.zoom = 200
        self.x = x
        self.x_ref = WIDTH//2
        self.y_ref = HEIGH//2
        pass
    
    def draw_som(self):
        som_weights = self.som.get_neurons()
        for i in range(len(som_weights[0])):
            for j in range(len(som_weights[0,0,:])):
                pg.draw.circle(self.win,(0,0,255),(som_weights[0,i,j]*self.zoom + self.x_ref,som_weights[1,i,j]*self.zoom + self.y_ref),3)
                if i>0:
                    pg.draw.line(self.win,(0,0,255),(som_weights[0,i-1,j]*self.zoom + self.x_ref,som_weights[1,i-1,j]*self.zoom+ self.y_ref),(som_weights[0,i,j]*self.zoom + self.x_ref,som_weights[1,i,j]*self.zoom+ self.y_ref))
                if j>0:
                    pg.draw.line(self.win,(0,0,255),(som_weights[0,i,j-1]*self.zoom + self.x_ref,som_weights[1,i,j-1]*self.zoom+ self.y_ref),(som_weights[0,i,j]*self.zoom + self.x_ref,som_weights[1,i,j]*self.zoom+ self.y_ref))
                if j>0 and i>0:
                    pg.draw.line(self.win,(0,0,255),(som_weights[0,i-1,j-1]*self.zoom + self.x_ref,som_weights[1,i-1,j-1]*self.zoom+ self.y_ref),(som_weights[0,i,j]*self.zoom + self.x_ref,som_weights[1,i,j]*self.zoom+ self.y_ref))
        pass
    
    def draw_x(self):
        for i in self.x:
            pg.draw.circle(self.win,(255,0,0),(i[0]*self.zoom + self.x_ref,i[1]*self.zoom + self.y_ref),1)
        pass

    def run(self):
        run = True
        i = (k for k in range(2))
        temp = next(i,None)
        x_iter = (j for j in self.x)
        while run:
            self.win.fill((0,0,0))
            ##### event section #####
            ##### 
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
            if temp != None:
                j = next(x_iter,None)
                if j is None:
                    temp = next(i,None)
                    x_iter = (j for j in self.x)
                    if temp == None:
                        continue
                    j = next(x_iter,None)
                self.som.teach(j)
            self.draw_x()
            self.draw_som()
            pg.display.update()
            #### delays ####
            pg.time.delay(60)

def main():
    x = [[np.random.uniform()+0.5,np.random.uniform()+0.8] for i in range(1000)]
    x = np.array(x)
    x = [[((np.sin(t:=(np.random.uniform()*2*np.pi))))*(r:=np.random.uniform())+0.5,(np.cos(t)*r)+0.8] for i in range(1000)]
    x = np.array(x)

    som = SOM(10,10,2,ni_0 = 1,T2=16,sigma_0=2)
    win = MainWindow(som,x)
    win.run()
    pass


if __name__ == '__main__':
    main()