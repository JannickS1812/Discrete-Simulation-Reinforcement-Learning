import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

def sim(num_cars=5, num_cells=8, dawdle_prob=0.4, t_max=100, v_max=2):
    cars = [[p, 0] for p in np.sort(random.sample(range(0, num_cells), num_cars))]
    history = [0] * t_max

    for t in range(t_max):
        for i in range(len(cars)):
            cars[i][1] = np.min([cars[i][1] + 1, v_max]) # increment velocity

            # determine distance to front
            if cars[(i+1) % num_cars][0] < cars[i][0]:  # car in front has 'looped around'
                distance_front = num_cells - cars[i][0] + cars[(i+1) % num_cars][0] - 1
            else:
                distance_front = cars[(i+1) % num_cars][0] - cars[i][0] - 1
            cars[i][1] = np.min([cars[i][1], distance_front])  # slow down to avoid crash

            if cars[i][1] > 0 and random.uniform(0, 1) < dawdle_prob: # decrement on occasion
                cars[i][1] -= 1

        cars = [[(c[0] + c[1]) % num_cells, c[1]] for c in cars]  # add velocity to position
        history[t] = copy.deepcopy(cars)

    return history

if __name__ == "__main__":
    t_max = 100
    num_cells = 100
    num_cars = 50
    dawdle_prob = 0.5
    v_max = 4
    cars_over_time = sim(num_cars=num_cars, num_cells=num_cells, dawdle_prob=dawdle_prob, t_max=t_max, v_max=v_max)

    # show the cars over time in a grey scale image & colored image (where the velocity determines the color of the cell)
    cells_c = np.ones((t_max, num_cells)) * -1
    cells = np.ones((t_max, num_cells))
    for t in range(t_max):
        for n in range(num_cars):
            cells_c[t, cars_over_time[t][n][0]] = cars_over_time[t][n][1]
            cells[t, cars_over_time[t][n][0]] = 0

    # colored
    plt.figure(1)
    im = plt.imshow(cells_c, interpolation='nearest')
    plt.xlabel('Road')
    plt.ylabel('Time')

    # add legend
    patches = [Patch(color=im.cmap(im.norm(-1)), label="Empty")]
    plt.legend(handles=patches, bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
    plt.colorbar(im, boundaries=np.linspace(0, v_max), ticks=np.arange(v_max+1))
    plt.gca().set_aspect(1/plt.gca().get_data_ratio())  # squares the plot
    title = r'$\frac{N_{cars}}{N_{cells}}=' + f'{num_cars / num_cells:.2f}' + ', P_{dawdle}=' + f'{dawdle_prob:.2f}' + '$'
    plt.title('Colored: ' + title)

    # colored
    plt.figure(2)
    im = plt.imshow(cells, interpolation='nearest', cmap=cm.Greys_r)
    plt.xlabel('Road')
    plt.ylabel('Time')

    # add legend
    patches = [Patch(color=im.cmap(im.norm(0)), label="Empty"), Patch(facecolor=im.cmap(im.norm(1)), edgecolor='k', linewidth=1, label="Car")]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.gca().set_aspect(1 / plt.gca().get_data_ratio())  # squares the plot
    plt.title('Gray-Scale: ' + title)

    plt.show()











