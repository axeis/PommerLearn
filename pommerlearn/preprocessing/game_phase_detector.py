"""
@file: game-phase-detector.py
Created on 15.04.24
@project: PommerLearn
@author: axeis

"""
import numpy as np
from tqdm import tqdm


def get_phase_vector(z, phase_definition="mixedness"):
    samples = z['obs'].shape[0]
    phases = np.zeros(shape=samples, dtype=int)

    for i, observation in enumerate(tqdm(z['obs'])):
        phases[i] =get_game_phase(
            observation=observation,
            phase_definition=phase_definition
            )

        # assert phases[i] == z['phase'][i], (f"calculated {phases[i]}, while previously it was {z['phase'][i]} at {i}")


    print (np.sum(phases == z['phase']))


def get_game_phase(observation, phase_definition):
    
    
    if phase_definition == "steps":
        if observation[22][10][10]*799.0 <= 40:
            return 0
        else:
            return 1
    
    
    
    elif phase_definition == "living_opponent":
        living_count = np.sum(observation[18:22,0,0])
        return 4 - living_count
    

    elif phase_definition == "mixedness":
        minManhattanDis = 11+11
        agents = np.empty(shape=(0,2), dtype=int)
        for i in range(4):
            arr = np.array(np.where(observation[10+i] == 1))
            agents = np.append(agents, arr.transpose(), axis=0)
        
        # handle dead agents
        for i in range(1,agents.shape[0]):
            manhattanDis = int(np.sum(np.abs(agents[0] - agents[i])))
            minManhattanDis = min(minManhattanDis, manhattanDis)
        
        if minManhattanDis < 3 : return 2
        if minManhattanDis < 6 : return 1
        return 0


    else:
        pass


