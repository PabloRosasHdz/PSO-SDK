from ParticleSwarmIntelligence import ParticleSwarmIntelligence
from Function import Function

if __name__ == "__main__":
    obj = ParticleSwarmIntelligence(20, 50, 0.8, 0.7, 1)
    best_position, best_value = obj.execute()
    print(f"Mejor posicion obtenida: {best_position}")
    print(f"Mejor valor obtenido: {best_value}")
