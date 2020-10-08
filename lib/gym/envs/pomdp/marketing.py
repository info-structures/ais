import numpy as np
import copy

class MarketingEnv():
    def __init__(self):
        self.start_state=0
        self.current_state=copy.deepcopy(self.start_state)
        self.name="Marketing"
    def step(self,action):
        if action==0:
            transition_probability=np.array([[0.8,0.2],[0.5,0.5]])
            observation_probability=np.array([[0.8,0.2],[0.6,0.4]])
            rewards=np.array([4,-4])
            next_state=np.random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(1,observation_probability[next_state,:]).argmax()
            reward=rewards[next_state]
        if action==1:
            transition_probability=np.array([[0.5,0.5],[0.4,0.6]])
            observation_probability=np.array([[0.9,0.1],[0.4,0.6]])
            rewards=np.array([0,3])
            next_state=np.random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(1,observation_probability[next_state,:]).argmax()
            reward=rewards[next_state]
            
        self.current_state=next_state
        
        return observation,reward
    def number_of_actions(self):
        actions=2
        return actions
    def number_of_observations(self):
        observations=2
        return observations
    def reset(self):
        self.current_state=copy.deepcopy(self.start_state)
        return self.current_state
