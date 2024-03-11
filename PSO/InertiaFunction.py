import math
import random

class InertiaFuc:
    """
    Clase que incluye las estrategias para el calculo del coeficiente del peso inercial. 
    """
    def RandomInertia(inertia, social_weight, cognitive_weight, n_iterations, i):
        """
        Es una función que selecciona aleatoriamente el coeficiente de inercia 
        de modo que la inercia estará entre (0.5 , 1.0), exhibirá un comportamiento
        convergente cuando inertia < 0,7854 lo que es un 57.08% del tiempo.
        """
        inertia = 0.5 + random.uniform(0, 1)/2
        return inertia

    def LinealDecresingIW(inertia, social_weight, cognitive_weight, n_iterations, i, 
                                Weightmin = 0.4, Weightmax = 0.9 ):
        """
        Es una fución propuesta por Shi y Eberhart (1998) se propuso como
        método para disminuir linealmente el peso de la inercia a lo largo
        del tiempo, basándose en el consenso general de que la exploración 
        se favorece al principio de un proceso de búsqueda, mientras que la 
        explotación se favorece más tarde.
        Esta función no hace uso del 'social_weight' y 'cognitive_weight'
            Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. In Proceedings of the 1998 IEEE
        international conference on evolutionary computation, (pp. 69–73).
        """
        inertia =  Weightmax - ((Weightmax - Weightmin)  * (i / n_iterations))
        return inertia

    def NoLinearIW(inertia, social_weight, cognitive_weight, n_iterations, i, 
                                Weightmin = 0.4, Weightmax = 0.9, alpha = 1/math.pi**2):
        """
        Es una función no-lineal, varible en el tiempo del peso incercial
        que demuestra un rendimiento superior a la lineal, donde alpha es 
        una constante dada por el usuario, usualmente 1/pi^2, exhibe un comportamiento convergente
            Yang, C., Gao, W., Liu, N., & Song, C. (2015). Low-discrepancy sequence initialized particle swarm opti
        mization algorithm with high-order nonlinear time-varying inertia weight. Applied Soft Computing, 29,
        386–394.
        """
        inertia = Weightmax - (Weightmax - Weightmin) * (i/n_iterations)**alpha
        return inertia

    def NonlinearImproved(inertia, social_weight, cognitive_weight, n_iterations, i,
                                Unl =  1.0002 ):
        """
        Es una función no lineal donde el parametro Unl ∈ [1.0001,1.005], 
        Se utiliza por defecto Unl=1.0002 y Se comienda inercia inicial = 0.3.
        Este algoritmo simpre exhibira un comportamiento convergente pero se 
        espera que sea prematuro
            Jiao, B., Lian, Z., & Gu, X. (2008). A dynamic inertia weight particle swarm optimization algorithm. Chaos,
        ractals, 37(3), 698–705.
        """
        inertia = inertia * Unl**(-i)
        return inertia

    def DecresingInertiaWeight(inertia, social_weight, cognitive_weight, n_iterations, i):
        """
        Es una función decreciente de Fan y Chiu (2007) que emplea una
        estrategia de peso de inercia dependiente del tiempo de modo que
        el coeficiente disminuye con el tiempo de forma no lineal, exhibirá
        un comportamiento convergente general después de que hayan pasado 
        cinco iteraciones.
        """
        inertia = (2/i)**(0.3)
        return inertia

    def ChaoticDescendingIW(inertia, social_weight, cognitive_weight, n_iterations, i,
                                    Weightmin = 0.4, Weightmax = 0.9 ):
        """
        Es una función descendente caótica por Feng et al. (2007) que adopta 
        el uso de dinámica caótica para adaptar el peso de incercia a lo 
        largo del tiempo. Exhibirá un comportamiento mayormente convergente
        """
        def functZ(i):
            if i == 0:
                return random.uniform(0, 1)
            else:
                return 4 * functZ(i-1) * (1-functZ(i-1))
        inertia = functZ(i) *  Weightmin + (Weightmax - Weightmin) * (n_iterations - i)/n_iterations
        return inertia

    def NaturalExponentIW(inertia, social_weight, cognitive_weight, n_iterations, i, 
                                Weightmin = 0.4, Weightmax = 0.9 ):
        """
        Es una función de exponente natural de Chen et al. (2006) que utiliza 
        un coeficiente inercial basado en un exponencial, exhibe un comportamiento
        convergente despues de un 2,6% de la busqueda.
        """
        inertia = Weightmin + (Weightmax - Weightmin)*math.e**(-(10*i)/n_iterations)
        return inertia

    def OscillatingIW(inertia, social_weight, cognitive_weight, n_iterations, i,
                            Weightmin = 0.3, Weightmax = 0.9, parK = 7):
        """
        Es una función oscilante de Kentzoglanakis y Poole (2009) que no disminuía
        monótonamente, sino que porporcionaba un peso de inercia oscilante durante
        la optimización.
        """
        if i < (3*n_iterations)/4:
            inertia = (Weightmin+Weightmax)/2 + (Weightmax-Weightmin)/2 * math.cos(2*math.pi * (4*parK+6))
        else:
            inertia = Weightmin
        return inertia

    def SugenoIW(inertia, social_weight, cognitive_weight, n_iterations, i,
                        parS = 2):
        """
        El algoritmo propuesta por Lei et al. (2006) emplea el uso de la función
        Sugeno que proporciona un peso de inercia monótonamente decreciente.
        """
        def beta(i):
            return i/n_iterations

        inertia = (1-beta(i))/(1+parS*beta(i))
        return inertia 

    def LogarithmDecreasingIW(inertia, social_weight, cognitive_weight, n_iterations, i,
                                    Weightmin = 0.4, Weightmax = 0.9, parA = 1 ):
        """
        La función propuesta por Gao et al. (2008) emplea un peso de inercia 
        logarítmitmicamente decreciente, exhibirá un comportamiento convergente 
        cuando i/n_iterations > 0.02576 o solo despues del 2.6% de la busqueda.
        """
        inertia = Weightmax + (Weightmin - Weightmax) * math.log10(parA+(10*i)/n_iterations)
        return inertia

    def SocialCognitive(inertia, social_weight, cognitive_weight, n_iterations, i):
        inertia =  cognitive_weight + social_weight
        return inertia

    def Senoidal(inertia, social_weight, cognitive_weight, n_iterations, i):
        Wieightrange = 0.9
        inertia =  inertia + (Wieightrange)/2 * math.sin((2*math.pi*i)/n_iterations)
        return inertia

    #    Ejemplo PARA NUESTRA PROPIA FUNCION INERCIAL
    #    def NOMBRE(inertia, social_weight, cognitive_weight, n_iterations, i):
    #        inertia = FUCTION
    #        return inertia