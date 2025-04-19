import gym
import numpy as np
import matplotlib as plt

# follow: https://aleksandarhaber.com/monte-carlo-method-for-learning-state-value-functions-first-visit-method-reinforcement-learning-tutorial/
'''
1. Loop in every episode - create total returns vector + visited state vector
2. In every episode, compute return from every visited state
3. In every iteration, update returns vector + total number of visits
4. Divide total returns of a particular state with total number of visits
'''

def MonteCarloStateValueFunction(env, stateNumber, episodes, discountRate):
    sumReturnForEveryState = np.zeros(stateNumber)
    numberVisitsEveryState = np.zeros(stateNumber)
    valueFunctionEstimate = np.zeros(stateNumber)

    for _ in range(episodes):
        visitedStateInEpisode = []
        returnsInEpisode = []

        (currentState, prob) = env.reset()
        visitedStateInEpisode.append(currentState)

        while True:
            # not end until reach terminalState
            randomAction = env.action_space.sample()

            (currentState, currentReward, terminalState, _, _) = env.step(randomAction)
            returnsInEpisode.append(currentReward)
            if not terminalState:
                visitedStateInEpisode.append(currentState)
            else:
                break
        
        lenStates = len(visitedStateInEpisode)
        Gt=0

        for idx in range(lenStates - 1, -1, -1):
            stateTmp = visitedStateInEpisode[idx]
            returns = returnsInEpisode[idx]

            Gt = discountRate*Gt + returns

            if stateTmp not in visitedStateInEpisode[0:idx]:
                numberVisitsEveryState[stateTmp] = numberVisitsEveryState[stateTmp] + 1
                sumReturnForEveryState[stateTmp] = sumReturnForEveryState[stateTmp] + Gt

    for idx in range(stateNumber):
        if numberVisitsEveryState[idx] != 0:
            valueFunctionEstimate[idx] = sumReturnForEveryState[idx] / numberVisitsEveryState[idx]

    return valueFunctionEstimate
        
        
            

    