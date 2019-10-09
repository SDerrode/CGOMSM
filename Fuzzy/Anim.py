import numpy as np
import random
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


from APrioriFuzzyLaw_Series2 import LoiAPrioriSeries2
from APrioriFuzzyLaw_Series4 import LoiAPrioriSeries4

fontS = 16 # fontSize
mpl.rc('xtick', labelsize=fontS)
mpl.rc('ytick', labelsize=fontS)
dpi = 100


if __name__ == '__main__':

    discretization = 200

    #seed = random.randrange(sys.maxsize)
    seed = 5039309497922655937
    random.seed(seed)
    print("Seed was:", seed)
    # 4887634503081723624 8132925864540000067 5039309497922655937

    # P = LoiAPrioriSeries2(alpha=0.10, eta=0.21, delta=0.076)
    P = LoiAPrioriSeries4(alpha=0.15, gamma = 0.60, delta_d=0.20, delta_u=0.20)

    fig = plt.figure(figsize=(7, 10))
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    P.plotR1R2(discretization, None, ax1, dpi=dpi)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.grid(True)

    #############################
    # Premier dessin (3D)
    #############################
    # Les données markoviennes à dessiner par dessus le fond (qui est la densité 2D de la loi floue)
    N = 50
    chain = np.empty((1, N))
    chain[:, 0] = P.tirageR1()
    # les suivantes...
    for i in range(1, N):
        chain[0, i] = P.tirageRnp1CondRn(chain[0, i-1])

    x, y, z = chain[:, 0], chain[:, 1], 0.
    line1, = ax1.plot(x, y, z, 'm*', alpha=1., linewidth=3.0)


    #############################
    # Deuxième dessin (2D)
    #############################
    x = np.arange(0.0, N, 1)
    #line2 = ax2.plot(x, np.reshape(chain, newshape=(N)), 'b*', alpha=1., linewidth=3.0)
    line2 = ax2.plot([], [], 'b*', alpha=1., linewidth=3.0) # on ne dessine rien
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlim(0, N)

    def animate(num, chain, lines): 
        if num>3:
            # animation dessin 3D
            x, y, z = np.reshape(chain[:, 0:num], newshape=(num)), np.reshape(chain[:, 1:num+1], newshape=(num)), np.zeros((num))
            lines[0].set_data(x,y)
            lines[0].set_3d_properties(z)

            # animation dessin 2D
            y1 = np.reshape(chain[:, 0:num], newshape=(num))
            x1 = list(range(0, num))
            # print('Array y1 = ', y1)
            # print('Array x1 = ', x1)
            ax2.plot(x1, y1, 'b-', alpha=1., linewidth=1.0)
            y1 = np.reshape(chain[:, num-2:num], newshape=(2))
            x1 = list(range(num-2, num))
            # print('Array y1 = ', y1)
            # print('Array x1 = ', x1)
            ax2.plot(x1, y1, 'm*', alpha=1., linewidth=3.0)
            # input('pause')

        return lines,

    lines = [line1, line2]
    #ani     = animation.FuncAnimation(fig, animate, init_func=init, frames=100, blit=True, interval=20, repeat=False)
    line_ani = animation.FuncAnimation(fig, animate, frames=(N), fargs=(chain, lines), interval=150, blit=False, repeat=False)
    # print('Array line_ani = ', line_ani)
    line_ani.save('mymovie.mp4')

    # input('pause')
    plt.show(block=False)


