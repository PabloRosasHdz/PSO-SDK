from Particle import Particle
from Swarm import Swarm
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def ejemploParticula():
    # Ejemplo creación partícula.
    part = Particle(
            num_variables = 3,
            lower_limits = [4,10,20],
            upper_limits = [-1,2,5],
            verbose     = True
            )
    # Ejemplo evaluar partícula con una función objetivo
    def funcion_objetivo(x_0, x_1, x_2):
        f= x_0**2 + x_1**2 + x_2**2
        return(f)

    part.evaluate_particle(
        objective_function = funcion_objetivo,
        optimization = "maximizar",
        verbose     = True
        )

    # Hasta que la partícula se mueva, el valor actual y mejor valor es el mismo.
    part

    # Ejemplo mover partícula
    part.move_particle(
        best_swarm_position = np.array([-1000,-1000,+1000]),
        inertia          = 0.8,
        cognitive_weight = 2,
        social_weight    = 2,
        verbose          = True
        )

    part

def ejemploEnjambre():
    # Ejemplo crear enjambre
    enjambre = Swarm(
                n_particles = 4,
                num_variables = 3,
                lower_limits  = [-5,-5,-5],
                upper_limits  = [5,5,5],
                verbose      = False
                )
    # Ejemplo evaluar enjambre
    def funcion_objetivo(x_0, x_1, x_2):
        f= x_0**2 + x_1**2 + x_2**2
        return(f)

    enjambre.evaluate_swarm(
        objective_function = funcion_objetivo,
        optimization     = "minimizar",
        verbose          = False
        )
    # Ejemplo mover enjambre
    enjambre.move_swarm(
        inertia          = 0.8,
        cognitive_weight = 2,
        social_weight    = 2,
        verbose          = True
    )

def ejemploOptimizacion():
# Ejemplo optimización
    def funcion_objetivo(x_0, x_1):
        '''
        Para la región acotada entre −10<=x_0<=0 y −6.5<=x_1<=0 la función tiene
        múltiples mínimos locales y un único minimo global que se encuentra en
        f(−3.1302468,−1.5821422) = −106.7645367
        '''
        f = np.sin(x_1)*np.exp(1-np.cos(x_0))**2 \
            + np.cos(x_0)*np.exp(1-np.sin(x_1))**2 \
            + (x_0-x_1)**2
        return(f)

    # Contour plot función objetivo
    x_0 = np.linspace(start = -10, stop = 0, num = 100)
    x_1 = np.linspace(start = -6.5, stop = 0, num = 100)
    x_0, x_1 = np.meshgrid(x_0, x_1)
    z = funcion_objetivo(x_0, x_1)
    plt.contour(x_0, x_1, z, 35, cmap='RdGy')

    enjambre = Swarm(
               n_particles = 50,
               num_variables  = 2,
               lower_limits  = [-10, -6.5],
               upper_limits  = [0, 0],
               verbose      = False
            )

    enjambre.optimize(
        objective_function = funcion_objetivo,
        optimization     = "minimizar",
        n_iterations    = 250,
        inertia          = 0.8,
        reduce_inertia    = False,
        cognitive_weight   = 1,
        social_weight      = 2,
        early_stopping  = True,
        stopping_rounds    = 5,
        stopping_tolerance = 10**-3,
        verbose          = False
    )
    # print(enjambre)

    # Evolución de la optimización
    fig = plt.figure(figsize=(6,4))
    enjambre.resultados_df['mejor_valor_enjambre'].plot()
    plt.show()

def ejemploInercia():
    # Ejemplo optimización con una función de inercia propia
    def inertia_Lineal(inertia, social_weight, cognitive_weight, n_iterations, i):
        Weightmin = 0.9
        Weightmax = 0.4
        inertia =  Weightmax - ((Weightmax - Weightmin)  * (i / n_iterations))
        return inertia
    
    def inertia_NoLineal(inertia, social_weight, cognitive_weight, n_iterations, i):
        inertia =  cognitive_weight + social_weight
        return inertia
    
    def inertia_Senoidal(inertia, social_weight, cognitive_weight, n_iterations, i):
        Wieightrange = 0.9
        inertia =  inertia + (Wieightrange)/2 * math.sin((2*math.pi*i)/n_iterations)
        return inertia
    
    def inertia_RandomInertia(inertia, social_weight, cognitive_weight, n_iterations, i):
        inertia = 0.5 + random.uniform(0, 1)/2
        return inertia
    
    def inertia_ExponencialDecreciente(inertia, social_weight, cognitive_weight, n_iterations, i):
        alpha = 0.7
        inertia = inertia * math.e**(-alpha*i)
        return inertia
    
    def inertia_NonlinearImproved(inertia, social_weight, cognitive_weight, n_iterations, i):
        Unl =  1.0002 # Unl ∈ [1.0001,1.005].
        inertia = inertia * Unl**(-i)
        return inertia

    def inertia_DecresingInertiaWeight(inertia, social_weight, cognitive_weight, n_iterations, i):
        inertia = (2/i)**(0.3)
        return inertia
    #natural exponent inertia weigh
    def inertia_PSONEIW(inertia, social_weight, cognitive_weight, n_iterations, i):
        Weightmin = 0.9
        Weightmax = 0.4
        inertia = Weightmin + (Weightmax - Weightmin)*math.e**(-(10*i)/n_iterations)
        return inertia
#    Ejemplo PARA NUESTRA PROPIA FUNCION INERCIAL
#    def inertia_NOMBRE(inertia, social_weight, cognitive_weight, n_iterations, i):
#        inertia = 
#        return inertia

    def funcion_objetivo(x_0, x_1):
        '''
        Para la región acotada entre −10<=x_0<=0 y −6.5<=x_1<=0 la función tiene
        múltiples mínimos locales y un único minimo global que se encuentra en
        f(−3.1302468,−1.5821422) = −106.7645367
        '''
        f = np.sin(x_1)*np.exp(1-np.cos(x_0))**2 \
            + np.cos(x_0)*np.exp(1-np.sin(x_1))**2 \
            + (x_0-x_1)**2
        return(f)

    # Contour plot función objetivo
    x_0 = np.linspace(start = -10, stop = 0, num = 100)
    x_1 = np.linspace(start = -6.5, stop = 0, num = 100)
    x_0, x_1 = np.meshgrid(x_0, x_1)
    z = funcion_objetivo(x_0, x_1)
    # plt.contour(x_0, x_1, z, 35, cmap='RdGy')

    enjambre = Swarm(
               n_particles = 50,
               num_variables  = 2,
               lower_limits  = [-10, -6.5],
               upper_limits  = [0, 0],
               verbose      = False
            )

    enjambre.optimize(
        objective_function = funcion_objetivo,
        optimization     = "maximizar",
        n_iterations    = 250,
        inertia          = 0.8,
        reduce_inertia    = True,
        inertia_function=  inertia_Lineal,
        cognitive_weight   = 1,
        social_weight      = 2,
        early_stopping  = True,
        stopping_rounds    = 5,
        stopping_tolerance = 10**-3,
        verbose          = False
    )
    # print(enjambre)

    # Evolución de la optimización
    fig = plt.figure(figsize=(6,4))
    enjambre.resultados_df['mejor_valor_enjambre'].plot()
    plt.show()
if __name__ == "__main__":
    ejemploInercia()