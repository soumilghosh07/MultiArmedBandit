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





class   Question2:
    

    # Terminal state Left=0, A=1, B=2, C=3, D=4, E=5, Terminal state Right=6


    def __init__(self):
        self.all_states = np.arange(7) # This initiliases the environment by making an array [0,1,2,3,4,5,6,7]
        # Terminal state Left=0, A=1, B=2, C=3, D=4, E=5, Terminal state Right=6
        self.start_state = 3 # all episodes start at the state 3
        self.reset_state()

    def get_states(self):
        return self.all_states # returns the array [0,1,2,3,4,5,6,7]

    def get_reward(self, state):
        # if it's right terminal state we return reward as 1
        return int(state == self.all_states[-1])

    def step(self):
        action = [-1, 1][np.random.rand() >= 0.5]  # choosing 1 or -1 for going right or left respectively with equal probability
        next_state = self.state + action
        reward = self.get_reward(next_state)
        self.rewards_received.append(reward) # 

        if not self.is_terminal(next_state):      # if state is not right most terminal or left most terminal we move to next state
            self.state = next_state
            self.states_visited.append(next_state)

        return next_state, reward

    def is_terminal(self, state):
        # returns the terminal states
        return (state == self.all_states[0]) or (state == self.all_states[-1])
    
    def reset_state(self):
        self.state = self.start_state
        self.states_visited = [self.state]
        self.rewards_received = []
        return self.state


def game(env, n_episodes, algo, alpha=0.1):

    
    vals = 0.5*np.ones(len(env.get_states()))
    vals[0] = vals[-1] = 0  #setting value of terminal states to zero.
    v_over_episodes = np.empty((n_episodes+1, len(vals))) # creates an empty 2D NumPy array v_over_episodes to store the estimated state values over episodes.
    v_over_episodes[0] = vals.copy() #storing initial values for first row

    
    for episode in range(1, n_episodes+1):
        
        state = env.reset_state() #resetting environment to initial values
        episode_reward = 0
        # loop until state is terminal
        while not env.is_terminal(state):
            next_state, step_reward = env.step()
            episode_reward += step_reward
             # performing the  td(0) algorithm
            if algo == 'td':
                vals[state] += alpha * (step_reward + vals[next_state] - vals[state])
            state = next_state
        

        # after every episode we add the values of that episode to the row belonging to that episode
        v_over_episodes[episode] = vals.copy()

    # return only the non-terminal states
    print(v_over_episodes[:,1:-1])
    return v_over_episodes[:,1:-1]


def experiment():
     
        env = Question2() #creating an instance of the randomwalk algorithm
        true_values = [1/6,2/6,3/6,4/6,5/6]
     
        fig, axs = plt.subplots(1,2, figsize=(8,5))
        x = np.arange(1,6)  # This makes the x axis for the 5 states A,B,C,D,E
     
        
     
        estimated_v = game(env, n_episodes=100, algo='td')
        for ep in [0,1,10,100]:
            axs[0].plot(x, estimated_v[ep], marker='o', markersize=4, label='{} episodes'.format(ep), color='#{:02x}{:02x}{:02x}'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))) 
     
        axs[0].plot(x, true_values, label='True values', marker='o', markersize=5)
        axs[0].set_title('Estimated value')
        axs[0].set_xlabel('State')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(['A', 'B', 'C', 'D', 'E'])
        axs[0].legend(loc='lower right')
     
        
     
        alphavals = [ 0.05, 0.1, 0.15]  #taking values for various td alphas 
        runs = 100
        n_episodes = 100
     
        rmse = np.zeros((len(alphavals), n_episodes+1)) #2D NumPy array with a shape where the number of rows is equal to the number of different alpha values
                                                                # and the number of columns is equal to the total number of episodes plus one (n_episodes+1).
     
        for r in range(runs):  #looping over the runs
            # performing td
            for a, alpha in enumerate(alphavals):
                v = game(env, n_episodes, 'td', alpha)
                # calculate rms
                rmse[a] += np.sqrt(np.mean((v - true_values)**2, axis=1)) #TD(0) learning run with a specific alpha value,  the root mean square
                                                                                  #(RMS) error between the estimated value function v and the true values  for each episode is calculated
     
        rmse /= runs #taking average over all runs
     
        for i, a in enumerate(alphavals):
            axs[1].plot(np.arange(n_episodes+1), rmse[i], label=r'TD(0), $\alpha$ = {}'.format(a))
     
        axs[1].set_xlabel('Walks / Episodes')
        axs[1].set_title('Empirical RMS error, averaged over states')
        axs[1].legend(loc='upper right')
     
        plt.show()



if __name__ == '__main__':
    experiment()