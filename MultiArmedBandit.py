import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random


class Question1(object):
    def __init__(self, k, runs):
        self.k = k
        self.runs = runs

        self.q_star = np.random.normal(0, 1, (runs,k))
        self.arms = [0] * k

    def testbed(self):
        for i in range(self.k):
            self.arms[i] = np.random.normal(self.q_star[1, i], 1, self.runs) # first problem as a sample
            
        plt.figure(figsize=(12,8))
        plt.ylabel('Rewards distribution')
        plt.xlabel('Actions')
        plt.xticks(range(1,11))
        plt.yticks(np.arange(-5,5,0.5))

        plt.violinplot(self.arms, positions=range(1,11), showmedians=True)
        plt.show()
        
    def simple_max(Q, N, t):
    #     return np.argmax(Q)
        return np.random.choice(np.flatnonzero(Q == Q.max())) # ties are removed in this step

    def simple_bandit(self, epsilon, steps, initial_Q, alpha, argmax_func=simple_max, c = 0):
        self.c = c
        rewards = np.zeros(steps)
        actions = np.zeros(steps)
        
        for i in range(self.runs):
            Q = np.ones(self.k) * initial_Q # initial Q
            N = np.zeros(self.k)  # initalize number of rewards given
            best_action = np.argmax(self.q_star[i])
            for t in range(steps):

                if (t==0) :
                    a=np.random.randint(self.k) 
                else :
                    a=np.random.choice([(np.random.randint(self.k)),(argmax_func(Q, N, t))], 1, p=[epsilon,1-epsilon]).item()

                reward = np.random.normal(self.q_star[i, a], 1)

                N[a] += 1
                if alpha > 0:
                    Q[a] = Q[a] + (reward - Q[a]) * alpha
                else:
                    Q[a] = Q[a] + (reward - Q[a]) / N[a]

                rewards[t] += reward
                if a == best_action:
                    actions[t] += 1
        
        return np.divide(rewards,self.runs), np.divide(actions,self.runs)

    def ucb(self, Q, N, t):
        if N.min() == 0:
            return np.random.choice(np.flatnonzero(N == N.min()))
        
        M = Q + self.c * np.sqrt(np.divide(np.log(t),N))
        return np.argmax(M) # breaking ties randomly


    def graphs(self):
        ep_0, ac_0 = self.simple_bandit(0, 1000, 0,0)
        ep_01, ac_01 = self.simple_bandit(0.01, 1000, 0,0)
        ep_1, ac_1 = self.simple_bandit(0.1, 1000, 0, 0)
        ep_5, ac_5 = self.simple_bandit(0.5, 1000, 0, 0)
        ep_9, ac_9 = self.simple_bandit(0.9, 1000, 0, 0)

        plt.figure(figsize=(12,6))
        plt.plot(ep_0, 'g', label='epsilon = 0')
        plt.plot(ep_01, 'r', label='epsilon = 0.01')
        plt.plot(ep_1, 'b', label='epsilon = 0.1')
        plt.plot(ep_5, 'k', label='epsilon = 0.5')
        plt.plot(ep_9, 'c', label='epsilon = 0.9')
        plt.legend() 
        plt.show()

        plt.figure(figsize=(12,6))
        plt.yticks(np.arange(0,1,0.1))
        plt.plot(ac_0, 'g', label='epsilon = 0')
        plt.plot(ac_01, 'r', label='epsilon = 0.01')
        plt.plot(ac_1, 'b', label='epsilon = 0.1')
        plt.plot(ac_5, 'k', label='epsilon = 0.5')
        plt.plot(ac_9, 'c', label='epsilon = 0.9')
        plt.legend() 
        plt.show()

        opt_0, ac_opt_0 = self.simple_bandit(0, 1000, 5, 0.1)
        opt_1, ac_opt_1 = self.simple_bandit(0, 1000, 10, 0.1)
        opt_2, ac_opt_2 = self.simple_bandit(0.1, 1000, 5, 0.1)

        plt.figure(figsize=(12,6))
        plt.yticks(np.arange(0,1,0.1))
        plt.plot(ac_1, 'r', label='Q = 0, e = 0.1')
        plt.plot(ac_opt_0, 'b', label='Q = 5, e = 0')
        plt.plot(ac_opt_1, 'g', label='Q = 10, e = 0')
        plt.plot(ac_opt_2, 'y', label='Q = 5, e = 0.1')
        
        plt.legend() 
        plt.show()

        ucb_2, ac_ucb_2 = self.simple_bandit(0, 1000, 0 ,0, self.ucb,2)
        ucb_3, ac_ucb_3 = self.simple_bandit(0, 1000, 0 ,0, self.ucb,1)
        ucb_4, ac_ucb_4 = self.simple_bandit(0, 1000, 0 ,0, self.ucb,0.1)
        ucb_5, ac_ucb_5 = self.simple_bandit(0, 1000, 0 ,0, self.ucb,4)
        plt.figure(figsize=(12,6))
        plt.plot(ep_1, 'g', label='e-greedy e=0.1')
        plt.plot(ucb_2, 'b', label='ucb c=2')
        plt.plot(ucb_3, 'r', label='ucb c=1')
        plt.plot(ucb_4, 'c', label='ucb c=0.1')
        plt.plot(ucb_5, 'k', label='ucb c=4')
        plt.legend() 
        plt.show()
        
        
if __name__ == '__main__':
    q1 = Question1(10,2000)
    q1.testbed()
    q1.graphs()




