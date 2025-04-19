import numpy as np
import matplotlib as plt
import gym

# follow: https://aleksandarhaber.com/policy-iteration-algorithm-in-python-and-tests-with-frozen-lake-openai-gym-environment-reinforcement-learning-tutorial/

def policyEvaluation(env, valueFunctionVector, policy, numIterations, convergenceTolerance, discountRate):
    convergenceTrack = []
    for _ in range(numIterations):
        convergenceTrack.append(np.linalg.norm(valueFunctionVector, 2))
        valueFunctionVectorNext = np.zeros(env.observation_space.n)

        for state in env.P:
            outerSum =  0
            for action in env.P[state]:
                innerSum = 0
                for prob, nextState, reward, isTerminalState in env.P[state][action]:
                    innerSum = innerSum + prob * (reward + discountRate * valueFunctionVector[nextState])
                outerSum = outerSum + policy[state, action] * innerSum
            
            valueFunctionVectorNext[state] = outerSum

        if np.max(np.abs(valueFunctionVectorNext - valueFunctionVector)) < convergenceTolerance:
            valueFunctionVector = valueFunctionVectorNext
            break
        valueFunctionVector = valueFunctionVectorNext

    return valueFunctionVector, convergenceTrack

def policyImprovement(env, valueFunctionVector, numAction, numState, discountRate):

    qvalueMatrix = np.zeros((numState, numAction))
    improvedPolicy = np.zeros((numState, numAction))

    for state in range(numState):
        for action in range(numAction):
            for prob, nextState, reward, _ in env.P[state][action]:
                qvalueMatrix[state][action] = qvalueMatrix[state][action] + prob * (reward + discountRate * valueFunctionVector[nextState])

        bestAction = np.where(qvalueMatrix[state,:] == np.max(qvalueMatrix[state, :]))

        improvedPolicy[state][bestAction] = 1/np.size(bestAction)

    return improvedPolicy, qvalueMatrix

def policyIteration(env, valueFunctionVector, initPolicy, numPolicyIteration, numPolicyEvaluation, convergenceTolerance, discountRate):
    numAction = env.action_space.n
    numState = env.observation_space.n
    for idx in range(numPolicyIteration):
        if idx == 0:
            currentPolicy = initPolicy
        valueFunctionVectorComputed, _ = policyEvaluation(env, valueFunctionVector, currentPolicy, numPolicyEvaluation, convergenceTolerance, discountRate)
        improvedPolicy, qvalueMatrix = policyImprovement(env, valueFunctionVectorComputed, numAction, numState, discountRate)

        if np.allclose(currentPolicy, improvedPolicy):
            currentPolicy = improvedPolicy
            break
        currentPolicy = improvedPolicy   

    return currentPolicy