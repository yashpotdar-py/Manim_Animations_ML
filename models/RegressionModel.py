import numpy as np
from manim import *
import warnings
from manim import Line, DashedLine, Text

warnings.filterwarnings("ignore")


class LinearRegressionModel:
    def __init__(self, x_coords, y_coords):
        self.slope = None
        self.intercept = None
        self.x_coords = x_coords
        self.y_coords = y_coords

    def fit(self):
        """Fit the linear regression model to the data."""
        x_mean = np.mean(self.x_coords)
        y_mean = np.mean(self.y_coords)
        numerator = np.sum((self.x_coords - x_mean) * (self.y_coords - y_mean))
        denominator = np.sum((self.x_coords - x_mean) ** 2)
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, x):
        """Make predictions using the fitted model."""
        return self.slope * x + self.intercept

    def plot(self, ax, line_color=RED):
        """Plot the linear regression line."""
        x_line = np.array([0, 10])
        y_line = self.predict(x_line)
        line = Line(ax.c2p(x_line[0], y_line[0]), ax.c2p(x_line[1], y_line[1]), color=line_color)
        return line

    def residuals(self, ax):
        """Return the residual lines for the data points."""
        residual_lines = []
        for x, y in zip(self.x_coords, self.y_coords):
            predicted_y = self.predict(x)
            line = DashedLine(ax.c2p(x, y), ax.c2p(x, predicted_y), color=GREY)
            residual_lines.append((line, y - predicted_y))
        return residual_lines
