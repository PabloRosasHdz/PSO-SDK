import numpy as np
import copy
import time
from datetime import datetime
import pandas as pd
from Particle import Particle

class Swarm:
    """
    Esta clase crea un swarm de n partículas. El rango de posibles valores
    para cada variable (posición) puede estar acotado.

    Parameters
    ----------
    n_particles : `int`
        número de partículas del swarm.

    num_variables : `int`
        número de variables que definen la posición de las partículas.

    lower_limits : `list` or `numpy.ndarray`
        límite inferior de cada variable. Si solo se quiere predefinir límites
        de alguna variable, emplear ``None``. Los ``None`` serán remplazados por
        el valor (-10**3).

    upper_limits : `list` or `numpy.ndarray`
        límite superior de cada variable. Si solo se quiere predefinir límites
        de alguna variable, emplear ``None``. Los``None`` serán remplazados por
        el valor (+10**3).

    verbose : `bool`, optional
        mostrar información de la partícula creada. (default is ``False``)

    Attributes
    ----------
    particles : `list`
        lista con todas las partículas del swarm.
    
    n_particles :`int`
        número de partículas del swarm.

    num_variables : `int`
        número de variables que definen la posición de las partículas.

    lower_limits : `list` or `numpy.ndarray`
        límite inferior de cada variable.

    upper_limits : `list` or `numpy.ndarray`
        límite superior de cada variable.

    best_particle : `object particle`
        la mejor partícula del swarm en su estado actual.

    best_value : `floar`
        el mejor valor del swarm en su estado actual.

    particle_history : `list`
        lista con el estado de las partículas en cada una de las iteraciones que
        ha tenido el swarm.

    best_position_history : `list`
        lista con la mejor posición en cada una de las iteraciones que ha tenido
        el swarm.

    best_value_history : `list`
        lista con el mejor valor en cada una de las iteraciones que ha tenido el
        swarm.

    absolute_difference : `list`
        diferencia absoluta entre el mejor valor de iteraciones consecutivas.

    results_df : `pandas.core.frame.DataFrame`
        dataframe con la información del mejor valor y posición encontrado en
        cada iteración, así como la mejora respecto a la iteración anterior.

    optimal_value : `float`
        mejor valor encontrado en todas las iteraciones.

    optimal_position : `numpy.narray`
        posición donde se ha encontrado el valor óptimo.

    optimized : `bool`
        si el swarm ha sido optimizado.

    optimization_iterations : `int`
        número de iteraciones de optimizacion.

    verbose : `bool`, optional
        mostrar información de la partícula creada. (default is ``False``)

    Examples
    --------
    Ejemplo crear swarm

    >>> swarm = Swarm(
               n_particles = 5,
               num_variables  = 3,
               lower_limits  = [-5,-5,-5],
               upper_limits  = [5,5,5],
               verbose      = True
            )

    """

    def __init__(self, n_particles, num_variables, lower_limits=None,
                 upper_limits=None, verbose=False):

        # Número de partículas del swarm
        self.n_particles = n_particles
        # Número de variables de cada partícula
        self.num_variables = num_variables
        # Límite inferior de cada variable
        self.lower_limits = lower_limits
        # Límite superior de cada variable
        self.upper_limits = upper_limits
        # Verbose
        self.verbose = verbose
        # Lista de las partículas del swarm
        self.particles = []
        # Etiqueta para saber si el swarm ha sido optimizado
        self.optimized = False
        # Número de iteraciones de optimización llevadas a cabo
        self.optimization_iterations = None
        # Mejor partícula del swarm
        self.best_particle = None
        # Mejor valor del swarm
        self.best_value = None
        # Posición del mejor valor del swarm.
        self.best_position = None
        # Estado de todas las partículas del swarm en cada iteración.
        self.particle_history = []
        # Mejor posición en cada iteración.
        self.best_position_history = []
        # Mejor valor en cada iteración.
        self.best_value_history = []
        # Diferencia absoluta entre el mejor valor de iteraciones consecutivas.
        self.absolute_difference = []
        # data.frame con la información del mejor valor y posición encontrado en
        # cada iteración, así como la mejora respecto a la iteración anterior.
        self.results_df = None
        # Mejor valor de todas las iteraciones
        self.optimal_value = None
        # Mejor posición de todas las iteraciones
        self.optimal_position = None

        # CONVERSIONES DE TIPO INICIALES
        # ----------------------------------------------------------------------
        # Si lower_limits o upper_limits no son un array numpy, se convierten en
        # ello.
        if self.lower_limits is not None \
        and not isinstance(self.lower_limits, np.ndarray):
            self.lower_limits = np.array(self.lower_limits)

        if self.upper_limits is not None \
        and not isinstance(self.upper_limits, np.ndarray):
            self.upper_limits = np.array(self.upper_limits)

        # SE CREAN LAS PARTÍCULAS DEL SWARM Y SE ALMACENAN
        # ----------------------------------------------------------------------
        for i in np.arange(n_particles):
            particle_i = Particle(
                            num_variables=self.num_variables,
                            lower_limits=self.lower_limits,
                            upper_limits=self.upper_limits,
                            verbose=self.verbose
                          )
            self.particles.append(particle_i)

        # INFORMACIÓN DEL PROCESO (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("---------------")
            print("Swarm creado")
            print("---------------")
            print("Número de partículas: " + str(self.n_particles))
            print("Límites inferiores de cada variable: " \
                  + np.array2string(self.lower_limits))
            print("Límites superiores de cada variable: " \
                  + np.array2string(self.upper_limits))
            print("")

    def __repr__(self):
        """
        Información que se muestra cuando se imprime un objeto swarm.
        """

        text = "============================" \
                + "\n" \
                + "         Swarm" \
                + "\n" \
                + "============================" \
                + "\n" \
                + "Número de partículas: " + str(self.n_particles) \
                + "\n" \
                + "Límites inferiores de cada variable: " + str(self.lower_limits) \
                + "\n" \
                + "Límites superiores de cada variable: " + str(self.upper_limits) \
                + "\n" \
                + "Optimizado: " + str(self.optimized) \
                + "\n" \
                + "Iteraciones optimización: " + str(self.optimization_iterations) \
                + "\n" \
                + "\n" \
                + "Información mejor partícula:" \
                + "\n" \
                + "----------------------------" \
                + "\n" \
                + "Mejor posición actual: " + str(self.best_position) \
                + "\n" \
                + "Mejor valor actual: " + str(self.best_value) \
                + "\n" \
                + "\n" \
                + "Resultados tras optimizar:" \
                + "\n" \
                + "----------------------------" \
                + "\n" \
                + "Posición óptima: " + str(self.optimal_position) \
                + "\n" \
                + "Valor óptimo: " + str(self.optimal_value)
                
        return(text)

    def show_particles(self, n=None):
        """
        Este método muestra la información de cada una de las n primeras 
        partículas del swarm.

        Parameters
        ----------

        n : `int`
            número de particulas que se muestran. Si no se indica el valor
            (por defecto ``None``), se muestran todas. Si el valor es mayor
            que `self.n_particles` se muestran todas.
        
        Examples
        --------
        >>> swarm = Swarm(
               n_particles = 5,
               num_variables  = 3,
               lower_limits  = [-5,-5,-5],
               upper_limits  = [5,5,5],
               verbose      = True
            )

        >>> swarm.show_particles(n=1)

        """

        if n is None:
            n = self.n_particles
        elif n > self.n_particles:
            n = self.n_particles

        for i in np.arange(n):
            print(self.particles[i])
        return(None)

    def evaluate_swarm(self, objective_function, optimization, verbose=False):
        """
        Este método evalúa todas las partículas del swarm, actualiza sus
        valores e identifica la mejor partícula.

        Parameters
        ----------
        objective_function : `function`
            función que se quiere optimizar.

        optimization : {maximizar o minimizar}
            Dependiendo de esto, el mejor valor histórico de la partícula será
            el mayor o el menor valorque ha tenido hasta el momento.

        verbose : `bool`, optional
            mostrar información de la partícula creada. (default is ``False``)
        
        Examples
        --------
        Ejemplo evaluar swarm

        >>> swarm = Swarm(
               n_particles = 5,
               num_variables  = 3,
               lower_limits  = [-5,-5,-5],
               upper_limits  = [5,5,5],
               verbose      = True
            )

        >>> def objective_function(x_0, x_1, x_2):
                f= x_0**2 + x_1**2 + x_2**2
                return(f)

        >>> swarm.evaluate_swarm(
                objective_function = objective_function,
                optimization     = "minimizar",
                verbose          = True
                )
        
        """

        # SE EVALÚA CADA PARTÍCULA DEL SWARM
        # ----------------------------------------------------------------------
        for i in np.arange(self.n_particles):
            self.particles[i].evaluate_particle(
                objective_function=objective_function,
                optimization=optimization,
                verbose=verbose
                )

        # MEJOR PARTÍCULA DEL SWARM
        # ----------------------------------------------------------------------
        # Se identifica la mejor partícula de todo el swarm. Si se está
        # maximizando, la mejor partícula es aquella con mayor valor.
        # Lo contrario si se está minimizando.

        # Se selecciona inicialmente como mejor partícula la primera.
        self.best_particle = copy.deepcopy(self.particles[0])
        # Se comparan todas las partículas del swarm.
        for i in np.arange(self.n_particles):
            if optimization == "minimizar":
                if self.particles[i].value < self.best_particle.value:
                    self.best_particle = copy.deepcopy(self.particles[i])
            else:
                if self.particles[i].value > self.best_particle.value:
                    self.best_particle = copy.deepcopy(self.particles[i])

        # Se extrae la posición y valor de la mejor partícula y se almacenan
        # como mejor valor y posición del swarm.
        self.best_value = self.best_particle.value
        self.best_position = self.best_particle.position

        # INFORMACIÓN DEL PROCESO (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("-----------------")
            print("Swarm evaluado")
            print("-----------------")
            print("Mejor posición encontrada : "
                  + np.array2string(self.best_position))
            print("Mejor valor encontrado : " + str(self.best_value))
            print("")

    def move_swarm(self, inertia, cognitive_weight, social_weight,
                       verbose=False):
        """
        Este método mueve todas las partículas del swarm.

        Parameters
        ----------
        optimization : {maximizar o minimizar}
            si se desea maximizar o minimizar la función.

        inertia : `float` or `int`
            coeficiente de inercia.

        cognitive_weight : `float` or `int`
            coeficiente cognitivo.

        social_weight : `float` or `int`
            coeficiente social.

        verbose : `bool`, optional
            mostrar información de la partícula creada. (default is ``False``)
        
        """

        # Se actualiza la posición de cada una de las partículas que forman el
        # swarm.
        for i in np.arange(self.n_particles):
            self.particles[i].move_particle(
                best_swarm_position=self.best_position,
                inertia=inertia,
                cognitive_weight=cognitive_weight,
                social_weight=social_weight,
                verbose=verbose
            )

        # Información del proceso (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("---------------------------------------------------------" \
                  "------------")
            print("La posición de todas las partículas del swarm ha sido " \
                  "actualizada.")
            print("---------------------------------------------------------" \
            "------------")
            print("")


    def optimize(self, objective_function, optimization, n_iterations=50,
                  inertia=0.8, reduce_inertia=False, inertia_function = None, cognitive_weight=2, social_weight=2,
                  early_stopping=False, stopping_rounds=None,
                  stopping_tolerance=None, verbose=False):
        """
        Este método realiza el proceso de optimización de un swarm.

        Parameters
        ----------
        objective_function : `function`
            función que se quiere optimizar.

        optimization : {maximizar o minimizar}
            si se desea maximizar o minimizar la función.

        n_iterations : `int` , optional
            numero de iteraciones de optimización. (default is ``50``)

        inertia : `float` or `int`, optional
            coeficiente de inercia. (default is ``0.8``)

        cognitive_weight : `float` or `int`, optional
            coeficiente cognitivo. (default is ``2``)

        social_weight : `float` or `int`, optional
            coeficiente social. (default is ``2``)

        reduce_inertia: `bool`, optional
           activar la reducción del coeficiente de inercia. En tal caso, el
           argumento `inertia` es ignorado. (default is ``True``)

        inertia_function: `function`
            función para el calculo de la inercia

        early_stopping : `bool`, optional
            si durante las últimas `stopping_rounds` generaciones la diferencia
            absoluta entre mejores individuos no es superior al valor de 
            `stopping_tolerance`, se detiene el algoritmo y no se crean nuevas
            generaciones. (default is ``False``)

        stopping_rounds : `int`, optional
            número de generaciones consecutivas sin mejora mínima para que se
            active el early stopping. (default is ``None``)

        stopping_tolerance : `float` or `int`, optional
            valor mínimo que debe tener la diferencia de generaciones consecutivas
            para considerar que hay cambio. (default is ``None``)

         verbose : `bool`, optional
            mostrar información de la partícula creada. (default is ``False``)
        
        Raises
        ------
        raise Exception
            si se indica `early_stopping = True` y los argumentos `stopping_rounds`
            o `stopping_tolerance` son ``None``.


        Examples
        --------
        Ejemplo optimización

        >>> def objective_function(x_0, x_1):
                # Para la región acotada entre −10<=x_0<=0 y −6.5<=x_1<=0 la 
                # función tiene múltiples mínimos locales y un único minimo 
                # global en f(−3.1302468,−1.5821422)= −106.7645367.
                f = np.sin(x_1)*np.exp(1-np.cos(x_0))**2 \
                    + np.cos(x_0)*np.exp(1-np.sin(x_1))**2 \
                    + (x_0-x_1)**2
                return(f)

        >>> swarm = Swarm(
                        n_particles = 50,
                        num_variables  = 2,
                        lower_limits  = [-10, -6.5],
                        upper_limits  = [0, 0],
                        verbose      = False
                        )

        >>> swarm.optimize(
                objective_function = objective_function,
                optimization     = "minimizar",
                n_iterations    = 250,
                inertia          = 0.8,
                reduce_inertia    = True,
                cognitive_weight   = 1,
                social_weight      = 2,
                early_stopping  = True,
                stopping_rounds    = 5,
                stopping_tolerance = 10**-3,
                verbose          = False
            )

        """

        # COMPROBACIONES INICIALES: EXCEPTIONS Y WARNINGS
        # ----------------------------------------------------------------------
        # Si se activa el early stopping, hay que especificar los argumentos
        # stopping_rounds y stopping_tolerance.
        if early_stopping \
        and (stopping_rounds is None or stopping_tolerance is None):
            raise Exception(
                "Para activar el early stopping es necesario indicar un " \
                + " valor de stopping_rounds y de stopping_tolerance."
                )
        
        # ITERACIONES
        # ----------------------------------------------------------------------
        start = time.time()

        for i in np.arange(n_iterations):
            if verbose:
                print("-------------")
                print("Iteracion: " + str(i))
                print("-------------")
            
            # EVALUAR PARTÍCULAS DEL SWARM
            # ------------------------------------------------------------------
            Swarm.evaluate_swarm(
                self,
                objective_function=objective_function,
                optimization=optimization,
                verbose=verbose
                )

            # SE ALMACENA LA INFORMACIÓN DE LA ITERACIÓN EN LOS HISTÓRICOS
            # ------------------------------------------------------------------
            self.particle_history.append(copy.deepcopy(self.particles))
            self.best_position_history.append(copy.deepcopy(self.best_position))
            self.best_value_history.append(copy.deepcopy(self.best_value))
            
            # SE CALCULA LA DIFERENCIA ABSOLUTA RESPECTO A LA ITERACION ANTERIOR
            # ------------------------------------------------------------------
            # La diferencia solo puede calcularse a partir de la segunda
            # iteración.
            if i == 0:
                self.absolute_difference.append(None)
            else:
                diference = abs(self.best_value_history[i] \
                                 - self.best_value_history[i-1])
                self.absolute_difference.append(diference)

            # DETECCIÓN EARLY STOPPING
            # ------------------------------------------------------------------
            # Si la diferencia absoluta entre los dos mejores valores de 
            # las últimas `stopping_rounds` generaciones no es superior al 
            # valor de `stopping_tolerance`, el algoritmo termina.
            if early_stopping and i > stopping_rounds:
                ultimos_n = np.array(self.absolute_difference[-(stopping_rounds): ])
                if all(ultimos_n < stopping_tolerance):
                    print("Algoritmo detenido en la iteracion " 
                          + str(i) \
                          + " por falta cambio absoluto mínimo de " \
                          + str(stopping_tolerance) \
                          + " durante " \
                          + str(stopping_rounds) \
                          + " iteraciones consecutivas.")
                    break
            
            # SE MUEVEN LAS PARTÍCULAS DEL SWARM
            # ------------------------------------------------------------------
            # Si se ha activado la reducción de inercia, se recalcula su valor 
            # para la iteración actual.
                        # SE ACTUALIZA EL COEFICIENTE DE INERCIA
            # ------------------------------------------------------------------
            if reduce_inertia:
                inertia = inertia_function(inertia, social_weight, cognitive_weight, n_iterations, i)
            Swarm.move_swarm(
               self,
               inertia        = inertia,
               cognitive_weight = cognitive_weight,
               social_weight    = social_weight,
               verbose        = False
            )
                
        end = time.time()
        self.optimized = True
        self.optimization_iterations = i
        
        # IDENTIFICACIÓN DEL MEJOR INDIVIDUO DE TO-DO EL PROCESO
        # ----------------------------------------------------------------------
        indice_valor_optimo  = np.argmin(np.array(self.best_value_history))
        self.optimal_value    = self.best_value_history[indice_valor_optimo]
        self.optimal_position = self.best_position_history[indice_valor_optimo]
        
        # CREACIÓN DE UN DATAFRAME CON LOS RESULTADOS
        # ----------------------------------------------------------------------
        self.resultados_df = pd.DataFrame(
            {
            "mejor_valor_enjambre"   : self.best_value_history,
            "mejor_posicion_enjambre": self.best_position_history,
            "diferencia_abs"         : self.absolute_difference
            }
        )
        self.resultados_df["iteracion"] = self.resultados_df.index
        
        print("-------------------------------------------")
        print("Optimización finalizada " \
              + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("-------------------------------------------")
        print("Duración optimización: " + str(end - start))
        print("Número de iteraciones: " + str(self.optimization_iterations))