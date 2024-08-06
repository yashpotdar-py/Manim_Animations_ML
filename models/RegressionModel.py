import numpy as np
from manim import *
import warnings
warnings.filterwarnings("ignore")


class RegressionModel:
    def __init__(self, x_coords, y_coords):
        self.x_coords = x_coords
        self.y_coords = y_coords

    def fit(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError


class LinearRegressionModel(RegressionModel):
    pass


class PolynomialRegressionModel(RegressionModel):
    pass
