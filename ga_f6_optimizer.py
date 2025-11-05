import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random

# ============================================================================
# Parâmetros
# ============================================================================
POPULATION_SIZE = 100
BITS_PER_VARIABLE = 22
TOTAL_BITS = BITS_PER_VARIABLE * 2  # 44 bits totais (22 do x, 22 do y)
CROSSOVER_RATE = 0.65
MUTATION_RATE = 0.008
GENERATIONS = 200
VARIABLE_RANGE = (-100, 100)  # Range for x and y

# ============================================================================
# AVALIAÇÃO
# ============================================================================

def f6_function(x: float, y: float) -> float:
    """
    Calcula F6
    F6(x, y) = 0.5 - ((sin(sqrt(x² + y²)))² - 0.5) / (1 + 0.001 * (x² + y²))²   
    """
    numerator = np.sin(np.sqrt(x**2 + y**2))**2 - 0.5
    denominator = (1 + 0.001 * (x**2 + y**2))**2
    return 0.5 - (numerator / denominator)


def binary_to_decimal(binary_string: str, min_val: float, max_val: float) -> float:
    """
    Converte binario para decimal
    Xreal = Xbin * ((max - min)/2**k-1) + min
    """

    # Converte o binario em string para int
    decimal_value = int(binary_string, 2)

    # 2**k - 1
    max_binary = 2**len(binary_string) - 1

    # Decodificação final
    scaled_value = min_val + (decimal_value / max_binary) * (max_val - min_val)
    return scaled_value


def decode_individual(individual: List[int]) -> Tuple[float, float]:
    """
    Transformal o resultado binario em x (primeiros 22 bits) e y (ultimos 22 bits)
    """
    # Divide os bits
    x_bits = ''.join(map(str, individual[:BITS_PER_VARIABLE]))
    y_bits = ''.join(map(str, individual[BITS_PER_VARIABLE:]))
    
    # Decodifica
    x = binary_to_decimal(x_bits, VARIABLE_RANGE[0], VARIABLE_RANGE[1])
    y = binary_to_decimal(y_bits, VARIABLE_RANGE[0], VARIABLE_RANGE[1])
    
    return x, y


def evaluate_individual(individual: List[int]) -> float:
    """
    Avaliação dos individuos. 
    """
    x, y = decode_individual(individual)
    fitness = f6_function(x, y)
    return fitness


# ============================================================================
# POPULAÇÃO
# ============================================================================

def initialize_population(pop_size: int, chromosome_length: int) -> List[List[int]]:
    """
    Inicializa uma população aleatoria de individos
    """
    population = []
    for _ in range(pop_size):
        individual = [random.randint(0, 1) for _ in range(chromosome_length)]
        population.append(individual)
    return population


def evaluate_population(population: List[List[int]]) -> List[float]:
    """
    Avaliação da População com base na F(x)
    """
    fitness_values = [evaluate_individual(ind) for ind in population]
    return fitness_values


def roulette_wheel_selection(population: List[List[int]], 
                             fitness_values: List[float]) -> List[int]:
    """
    Roleta para seleçao dos individuos
    """
    # Como é um problema de maximizacao quanto maior melhor.
    min_fitness = min(fitness_values)
    
    # joga valores para positivo para montar a roleta
    if min_fitness < 0:
        shifted_fitness = [f - min_fitness + 0.0001 for f in fitness_values]
    else:
        shifted_fitness = [f + 0.0001 for f in fitness_values]
    
    # Soma dos valores
    total_fitness = sum(shifted_fitness)
    
    # Probabilidades
    probabilities = [f / total_fitness for f in shifted_fitness]
    
    # Roleta
    r = random.random()
    cumulative_prob = 0
    for i, prob in enumerate(probabilities):
        cumulative_prob += prob
        if r <= cumulative_prob:
            return population[i].copy()
    
    # Fallback
    return population[-1].copy()


def get_best_individual(population: List[List[int]], 
                        fitness_values: List[float]) -> Tuple[List[int], float]:
    """
    Pega o melhor individuo para o elitismo
    """
    best_idx = np.argmax(fitness_values) # Melhor individuo nesse caso é o maior
    return population[best_idx].copy(), fitness_values[best_idx]


# ============================================================================
# REPRODUÇÃO
# ============================================================================

def one_point_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Cruza os pais com 1 ponto de crossover aleatorio
    """
    # Cria o ponto aleatorio
    crossover_point = random.randint(1, len(parent1) - 1)
    
    # Reproduz
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return offspring1, offspring2


def bit_flip_mutation(individual: List[int], mutation_rate: float) -> List[int]:
    """
    Calcula uma possivel mutacao (Flip Bit)
    """
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Flip bit
            mutated[i] = 1 - mutated[i]
    return mutated


def reproduce_population(population: List[List[int]], 
                        fitness_values: List[float],
                        elite_individual: List[int]) -> List[List[int]]:
    """
    Cria a nova geracao de populacao, Fazendo o cruzamento mutacao e mantendo o melhor (Elitismo)
    """
    new_population = [elite_individual]  # Elitismo
    
    while len(new_population) < POPULATION_SIZE:
        # Seleciona 2 pais com a rolete
        parent1 = roulette_wheel_selection(population, fitness_values)
        parent2 = roulette_wheel_selection(population, fitness_values)
        
        # Cruzamento
        if random.random() < CROSSOVER_RATE:
            offspring1, offspring2 = one_point_crossover(parent1, parent2)
        else:
            offspring1, offspring2 = parent1.copy(), parent2.copy()
        
        # Mutacao
        offspring1 = bit_flip_mutation(offspring1, MUTATION_RATE)
        offspring2 = bit_flip_mutation(offspring2, MUTATION_RATE)
        
        # Adiciona na nova populacao
        new_population.append(offspring1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(offspring2)
    
    return new_population


# ============================================================================
# MAIN LOOP
# ============================================================================

def run_genetic_algorithm(generations: int = GENERATIONS, 
                         verbose: bool = True) -> Tuple[List[int], float, List[float]]:
    """
    Roda o algoritimo
    """
    # Inicia a populacao
    population = initialize_population(POPULATION_SIZE, TOTAL_BITS)
    
    # Historico de melhores
    best_fitness_history = []
    
    # Loop
    for generation in range(generations):
        # Avaliacao
        fitness_values = evaluate_population(population)
        
        # Pega o melhor individuo - elitismo
        best_individual, best_fitness = get_best_individual(population, fitness_values)
        best_fitness_history.append(best_fitness)
        
        #if verbose and (generation % 10 == 0 or generation == generations - 1):
        x, y = decode_individual(best_individual)
        print(f"Generation {generation:3d}: Best Fitness = {best_fitness:.10f} "
              f"(x={x:.4f}, y={y:.4f})")
        
        # Nova populacao
        if generation < generations - 1:
            population = reproduce_population(population, fitness_values, best_individual)
    
    # Os melhores individuos
    fitness_values = evaluate_population(population)
    best_individual, best_fitness = get_best_individual(population, fitness_values)
    
    return best_individual, best_fitness, best_fitness_history


def plot_convergence(fitness_history: List[float]):
    """
    Plota os melhores individuos historicos
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness (F6 value)', fontsize=12)
    plt.title('Genetic Algorithm Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GENETIC ALGORITHM FOR F6 FUNCTION MAXIMIZATION")
    print("=" * 70)
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Chromosome Length: {TOTAL_BITS} bits ({BITS_PER_VARIABLE} per variable)")
    print(f"Crossover Rate: {CROSSOVER_RATE}")
    print(f"Mutation Rate: {MUTATION_RATE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Variable Range: {VARIABLE_RANGE}")
    print("=" * 70)
    print()
    
    # Roda o algoritiomo genetico
    best_individual, best_fitness, fitness_history = run_genetic_algorithm(
        generations=GENERATIONS, 
        verbose=True
    )
    
    # Decodifica os melhores
    x_best, y_best = decode_individual(best_individual)
    
    # Printa os resultados
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best Individual (binary):")
    print(f"  X bits: {''.join(map(str, best_individual[:BITS_PER_VARIABLE]))}")
    print(f"  Y bits: {''.join(map(str, best_individual[BITS_PER_VARIABLE:]))}")
    print()
    print(f"Decoded Values:")
    print(f"  x = {x_best:.10f}")
    print(f"  y = {y_best:.10f}")
    print()
    print(f"F6(x, y) = {best_fitness:.10f}")
    print("=" * 70)
    
    # Plota
    plot_convergence(fitness_history)
    
    # Estatisticas extras
    print()
    print("Convergence Statistics:")
    print(f"  Initial Best Fitness: {fitness_history[0]:.10f}")
    print(f"  Final Best Fitness: {fitness_history[-1]:.10f}")
    print(f"  Improvement: {fitness_history[-1] - fitness_history[0]:.10f}")
    print(f"  Improvement %: {((fitness_history[-1] - fitness_history[0]) / fitness_history[0] * 100):.2f}%")