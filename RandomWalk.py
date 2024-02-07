import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

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