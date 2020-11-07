import voting_systems as vs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

np.random.seed(0)

DIM = 2
N_CANDIDATES = 5
N_POPULATION = 1000
CANDIDATE_COLORS = ['red', 'blue', 'orange', 'cyan', 'green']

def get_rankings(population, candidates):
    return np.argsort(((population - candidates) ** 2).sum(axis=2))

def plot_population_and_candidates(population, candidates):
    plt.title("Voter Population and Candidates")
    plt.scatter(population[:, 0, 0], population[:, 0, 1], alpha=0.05, color='black')
    for i in range(N_CANDIDATES):
        plt.scatter(candidates[0, i, 0], candidates[0, i, 1], alpha=1, color=CANDIDATE_COLORS[i])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

def plot_population_votes(population, candidates):
    rankings = get_rankings(population, candidates)
    print(np.bincount(rankings[:, 0]))

    plt.title("Voter Population and Candidates")
    for i in range(N_POPULATION):
        plt.scatter(population[i, 0, 0], population[i, 0, 1], alpha=0.1, color=CANDIDATE_COLORS[rankings[i][0]])
    for i in range(N_CANDIDATES):
        plt.scatter(candidates[0, i, 0], candidates[0, i, 1], alpha=1, color=CANDIDATE_COLORS[i])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

if __name__ == '__main__':

    population = np.random.normal(0, 1, (N_POPULATION, 1, DIM))
    candidates_begin = np.random.normal(0, 1, (1, N_CANDIDATES, DIM))
    candidates_end = np.random.normal(0, 1, (1, N_CANDIDATES, DIM))

    # plot_population_and_candidates(population, candidates)
    # plot_population_votes(population, candidates)

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')
    plt.scatter(population[:, 0, 0], population[:, 0, 1], alpha=0.05, color='black')

    def init():
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        return ln,

    def update(frame):
        print(frame)
        candidates = (candidates_begin * (1 - frame)) + (candidates_end * frame)
        xdata.append(frame)
        ydata.append(np.sin(frame))
        ln.set_data(candidates[0, :, 0], candidates[0, :, 1])
        return ln,

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, 30 * 3), init_func=init, interval=1000 / 30)

    plt.show()
    # writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('im.mp4', writer=writer)
