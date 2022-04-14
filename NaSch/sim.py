import numpy as np
import random
from time import sleep

def sim(nCars=40, nLength=50, fDawdleProb=0.5, tMax=100):
    pos = random.sample(range(0, nLength), nCars)
    cars = [[p, 0] for p in np.sort(pos)]
    t = 0

    while t < tMax:
        for i in range(len(cars)):
            cars[i][1] = np.min([cars[i][1] + 1, 4])

            if cars[(i+1) % nCars][0] < cars[i][0] :
                nDistance2Front = nLength - cars[i][0] + cars[(i+1) % nCars][0] - 1
            else:
                nDistance2Front = cars[(i+1) % nCars][0] - cars[i][0]
            cars[i][1] = np.min([cars[i][1], nDistance2Front])

            if cars[i][1] > 0 and random.uniform(0, 1) < fDawdleProb:
                cars[i][1] -= 1

        for i in range(len(cars)):
            cars[i][0] = (cars[i][0] + cars[i][1]) % nLength


        print(cars)
        sleep(1)

if __name__ == "__main__":
    sim()









