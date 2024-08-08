from manim import *
import numpy as np
from models.RegressionModel import LinearRegressionModel


class AnimateSlopeChangeWithResidualsAndErrors(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 20, 1],
            y_length=7
        )
        self.play(Write(ax))

        x_coords = np.arange(1, 10, 1, dtype=float)
        y_coords = np.array([1.52246106, 6.2760993, 3.80538691, 6.19722876,
                             7.21630459, 11.43284789, 11.8475795, 10.04278697, 16.15779999])

        # Create and fit the regression model
        model = LinearRegressionModel(x_coords, y_coords)
        model.fit()

        # Plot scatter points
        dots = [Dot(ax.c2p(x, y), color=BLUE) for x, y in zip(x_coords, y_coords)]
        self.play(LaggedStart(*[Write(dot) for dot in dots], lag_ratio=0.05))

        # Create initial regression line and residuals
        line = model.plot(ax)
        self.play(Write(line))
        residual_lines_with_errors = model.residuals(ax)
        residual_lines = [residual for residual, _ in residual_lines_with_errors]
        error_values = [error for _, error in residual_lines_with_errors]

        # Draw residual lines
        self.play(LaggedStart(*[Write(residual) for residual in residual_lines], lag_ratio=0.1))

        # Display errors for each residual
        error_texts = []
        for (residual, error), (x, y) in zip(residual_lines_with_errors, zip(x_coords, y_coords)):
            # Position text at the end of each residual line
            error_text = Text(f"{error:.2f}", font_size=18).shift(RIGHT * 0.5)
            error_text.move_to(ax.c2p(x, (y + (y - model.predict(x))) / 2))  # Center it vertically along the residual
            error_texts.append(error_text)
            self.play(Write(error_text))

        # Create text for displaying slope value
        slope_text = Text(f"Slope (m) = {model.slope:.2f}", font_size=24).shift(UP * 2)
        self.play(Write(slope_text))

        # Animate changing slope
        for slope in np.linspace(-2, 2, 20):  # Range of slopes for the animation
            model.slope = slope
            model.intercept = np.mean(y_coords) - model.slope * np.mean(x_coords)
            new_line = model.plot(ax)

            # Update residuals and errors
            new_residual_lines_with_errors = model.residuals(ax)
            new_residual_lines = [residual for residual, _ in new_residual_lines_with_errors]
            new_error_values = [error for _, error in new_residual_lines_with_errors]

            # Update line and residuals
            self.play(Transform(line, new_line))
            self.play(LaggedStart(*[Transform(old_residual, new_residual) for old_residual, new_residual in
                                    zip(residual_lines, new_residual_lines)], lag_ratio=0.1))

            # Update error texts
            for error_text, new_error, (x, y) in zip(error_texts, new_error_values, zip(x_coords, y_coords)):
                error_text.become(Text(f"{new_error:.2f}", font_size=18))
                error_text.move_to(ax.c2p(x, (y + (y - model.predict(x))) / 2))
                self.play(Transform(error_text, error_text))

            # Update slope text
            new_slope_text = Text(f"Slope (m) = {model.slope:.2f}", font_size=24).shift(UP * 2)
            self.play(Transform(slope_text, new_slope_text))

            self.wait(0.1)

        self.wait()
