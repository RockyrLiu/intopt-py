import random
import math
from deap import tools, algorithms

def simulated_annealing_acceptance(new_fitness, old_fitness, temperature):
    """模拟退火接受准则"""
    if new_fitness < old_fitness:
        return True
    return math.exp((old_fitness - new_fitness) / temperature) > random.random()

def gasa_algorithm(population, toolbox, cxpb, mutpb, ngen, 
                 t_start=1000.0, t_end=0.01, alpha=0.99, 
                 stats=None, halloffame=None, verbose=__debug__):
    """遗传模拟退火算法主函数"""
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # 评价初始种群
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    if halloffame is not None:
        halloffame.update(population)
    
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    
    # 初始温度
    temperature = t_start
    
    # 开始进化
    for gen in range(1, ngen + 1):
        # 选择下一代个体
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # 交叉操作
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # 变异操作 + 模拟退火
        for i in range(len(offspring)):
            if random.random() < mutpb:
                mutant = offspring[i]
                # 保存原始个体和适应度
                old_mutant = toolbox.clone(mutant)
                old_fitness = mutant.fitness.values[0] if mutant.fitness.valid else toolbox.evaluate(mutant)[0]
                
                # 执行变异
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
                # 评估新个体
                new_fitness = toolbox.evaluate(mutant)[0]
                
                # 模拟退火接受准则
                if simulated_annealing_acceptance(new_fitness, old_fitness, temperature):
                    mutant.fitness.values = (new_fitness,)
                else:
                    # 拒绝变异，恢复为原来的个体
                    offspring[i] = old_mutant
        
        # 评估所有适应度值无效的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # 更新种群
        population[:] = offspring
        
        # 更新名人堂
        if halloffame is not None:
            halloffame.update(population)
        
        # 降温
        temperature *= alpha
        if temperature < t_end:
            temperature = t_end
        
        # 记录统计信息
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    
    return population, logbook


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

