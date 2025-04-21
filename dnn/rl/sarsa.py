import numpy as np
import gymnasium as gym

"""
Main idea
1. initialize S
2. choose action A using e-greedy approach computed on current value of Q(S,A)
3. loop through each episode
- take action A, observe env
- choose action S' from e-greedy approach computed on current value of Q(S',A)
- set S<-S', A<-A'
"""

# follow: https://aleksandarhaber.com/explanation-and-python-implementation-of-on-policy-sarsa-temporal-difference-learning-reinforcement-learning-tutorial/
class SARSA:
    def __init__(self, env, alpha, gamma, epsilon, episodes):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.stateNumber = env.observation_space.n
        self.actionNumber = env.action_space.n
        self.learnedPolicy = np.zeros(env.observation_space.n)
        self.Qmatrix=np.zeros((self.stateNumber,self.actionNumber))

    def selectAction(self, state, index):
        
        if index < 100:
            return np.random.choice(self.actionNumber)
             
        # Returns a random real number in the half-open interval [0.0, 1.0)
        randomNumber=np.random.random()
           
        if index > 1000:
            self.epsilon=0.9*self.epsilon
         
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionNumber)
        else:
            return np.random.choice(np.where(self.Qmatrix[state,:] == np.max(self.Qmatrix[state,:]))[0])

    def simulateEpisodes(self):
        for episode in range(self.episodes):
            # reset env at beginning of every episode
            (stateS, prob) = self.env.reset()

            actionA = self.selectAction(stateS, episode)
            
            terminalState = False
            while not terminalState:
                (stateSPrime, rewardPrime, terminalState,_,_) = self.env.step(actionA)
                actionAPrime = self.selectAction(stateSPrime, episode)

                if not terminalState:
                    error = rewardPrime + self.gamma*self.Qmatrix[stateSPrime, actionAPrime] - self.Qmatrix[stateS, actionA]
                    self.Qmatrix[stateS, actionA] = self.Qmatrix[stateS, actionA] + self.alpha*error

                else:
                    error = rewardPrime - self.Qmatrix[stateS, actionA]
                    self.Qmatrix[stateS, actionA] = self.Qmatrix[stateS, actionA] + self.alpha * error

                stateS = stateSPrime
                actionA = actionAPrime
            
    def finalPolicy(self):
        for idx in range(self.stateNumber):
            self.learnedPolicy[idx] = np.random.choice(np.where(self.Qmatrix[idx] == np.max(self.Qmatrix[idx]))[0])
