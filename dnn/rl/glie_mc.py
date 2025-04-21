import numpy as np
import matplotlib as plt
import gym

# follow: https://aleksandarhaber.com/python-implementation-of-the-greedy-in-the-limit-with-infinite-exploration-glie-monte-carlo-control-method-reinforcement-learning-tutorial/

def GLIEMonteCarlo(env, numEpisodes, discountRate, initialEpsilon):
    stateNumber = env.observation_space.n
    actionNumber = env.action_space.n
    epsilon = initialEpsilon
    visitsForEveryStateAction = np.zeros((stateNumber, actionNumber))
    actionValueMatrixEstimate = np.zeros((stateNumber, actionNumber))
    finalPolicy = np.zeros(stateNumber)

    def selectAction(state, index):
        
        randomNumber = np.random.random()
        if randomNumber < epsilon or index < 5:
            return np.random.choice(actionNumber)
        
        else:
            return np.random.choice(np.where(actionValueMatrixEstimate[state,:] == np.max(actionValueMatrixEstimate[state,:]))[0])
        
    for i in range(numEpisodes):
        visitsInEachEpisodes = []
        rewardInEachEpsidoes = []
        actionInSpace = []

        (currentState, prob) = env.reset()
        visitsInEachEpisodes.append(currentState)

        # not end until reach terminalState
        terminalState = False
        while not terminalState:
            # e-greedy
            action = selectAction(currentState, i)
            actionInSpace.append(action)

            (currentState, currentReward, terminalState, _, _) = env.step(action)
            rewardInEachEpsidoes.append(currentReward)

            if not terminalState:
                visitsInEachEpisodes.append(currentState)

        numVisited = len(visitsInEachEpisodes)
        Gt = 0
        # compute MC reward
        for idx in range(numVisited -1, -1, -1):
            stateTmp = visitsInEachEpisodes[idx]
            reward = rewardInEachEpsidoes[idx]
            actionTmp = actionInSpace[idx]

            Gt = discountRate * Gt + reward
            if stateTmp not in visitsInEachEpisodes[0: idx]:
                visitsForEveryStateAction[stateTmp][actionTmp] += 1
                # Q(S,A) <- Q(S,A) + 1/N(S,A)(G - Q(S,A))
                actionValueMatrixEstimate[stateTmp][actionTmp] = actionValueMatrixEstimate[stateTmp][actionTmp] + (1/visitsForEveryStateAction[stateTmp][actionTmp])*(Gt - actionValueMatrixEstimate[stateTmp][actionTmp])

        # decrease epsilon
        if i < 100:
            epsilon = initialEpsilon
        else:
            epsilon = 0.8 * epsilon
        
    
    for idx in range(stateNumber):
        # at each state, select the policy - take action that maximize q-value
        finalPolicy[idx] = np.random.choice(np.where(actionValueMatrixEstimate[idx,:] == np.max(actionValueMatrixEstimate[idx,:]))[0])

    return finalPolicy, actionValueMatrixEstimate








    