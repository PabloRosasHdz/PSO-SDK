import numpy as np
import random
from Function import Function


class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.evaluation = float("inf")
        self.particle_best_position = position
        self.particle_best_value = float("inf")

    def update_velocity(
        self, global_best_positions: np.ndarray, b1: float, b2: float, a: float
    ):
        for i in range(len(self.velocity)):
            r_coefficients = np.random.rand(2)
            self.velocity[i] = (
                (a * self.velocity[i])
                + (b1 * r_coefficients[0])
                * (self.particle_best_position[i] - self.position[i])
                + (b2 * r_coefficients[1])
                * (global_best_positions[i] - self.position[i])
            )

    def update_values(self):
        # update_velocity(global_best_positions,b1,b2)
        self.position = self.position + self.velocity
        self.evaluation = Function.evaluate(self.position)
        if self.evaluation < self.particle_best_value:
            self.particle_best_value = self.evaluation
            self.particle_best_position = self.position

    def __str__(self):
        return (
            f"Posicion: {self.position} \n"
            + f"Velocidad: {self.velocity} \n"
            + f"Mejor posicion: {self.particle_best_position} \n"
        )
