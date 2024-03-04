from Particle import Particle
import numpy as np


class ParticleSwarmIntelligence:
    particles: list[Particle]
    number_iterations: int
    number_particles: int
    inertia: float
    b1: float
    b2: float
    global_best_positions: np.ndarray
    global_best_value: float

    def __init__(self, number_particles, number_iterations, inertia, b1, b2):
        self.number_particles = number_particles
        self.number_iterations = number_iterations
        self.inertia = inertia
        self.b1 = b1
        self.b2 = b2
        self.particles = []

    def print_particles(self, message):
        print(message)
        print(f"Mejor posicion global: {self.global_best_positions}")
        for i, particle in enumerate(self.particles):
            print(f"Particula {i}")
            print(particle)

    def execute(self):
        # Step 1: Initialize particles
        for i in range(self.number_particles):
            # Let's initialize the particle with random values between [-5,5]
            p = Particle(((np.random.rand(2) * 5) - 5), np.zeros(2))
            self.particles.append(p)

        # Step 2: Initialize global_best
        self.global_best_positions = np.zeros(2)
        self.global_best_value = float("inf")

        # Step 3: Evaluate particles and assign best values
        for j, particle in enumerate(self.particles):
            particle.update_values()
            if particle.evaluation < self.global_best_value:
                self.global_best_value = particle.evaluation
                self.global_best_positions = particle.position

        self.print_particles("Valores iniciales")

        # Step 4: Update particles during n iterations
        for i in range(self.number_iterations):
            iteration_best_value = float("inf")
            iteration_best_positions = np.zeros(2)
            for j, particle in enumerate(self.particles):
                particle.update_velocity(
                    self.global_best_positions, self.b1, self.b2, self.inertia
                )
                particle.update_values()
                if particle.evaluation < iteration_best_value:
                    iteration_best_value = particle.evaluation
                    iteration_best_positions = particle.position

            if iteration_best_value < self.global_best_value:
                self.global_best_positions = iteration_best_positions
                self.global_best_value = iteration_best_value
            self.print_particles(f"Iteracion {i+1}")

        return [self.global_best_positions, self.global_best_value]
