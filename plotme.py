import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import imageio

class TrainingPlot:
    def __init__(self):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.scores_line, = self.ax.plot([], label='Scores')
        self.mean_scores_line, = self.ax.plot([], label='Mean Scores')
        self.ax.set_title('Training...')
        self.ax.set_xlabel('Number of Games')
        self.ax.set_ylabel('Score')
        self.ax.legend()
        self.scores = []
        self.mean_scores = []

    def update(self, scores, mean_scores):
        self.scores.append(scores[-1])
        self.mean_scores.append(mean_scores[-1])

        self.scores_line.set_xdata(range(len(self.scores)))
        self.scores_line.set_ydata(self.scores)

        self.mean_scores_line.set_xdata(range(len(self.mean_scores)))
        self.mean_scores_line.set_ydata(self.mean_scores)

        self.ax.relim()
        self.ax.autoscale_view()
        display.clear_output(wait=True)
        display.display(self.fig)
        plt.pause(0.1)

