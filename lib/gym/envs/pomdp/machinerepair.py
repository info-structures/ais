import numpy as np
import copy

class MachineRepairEnv():
    def __init__(self):
        self.start_state=0
        self.name="Machine Repair"
        
        self.current_state=copy.deepcopy(self.start_state)
        
    def step(self,action):
        
        
        if action==0:##Manufacture with out observation
            reward=0
            transition_probability=np.array([[0.5,0.3,0.15,0.05],[0,0.6,0.3,0.1],[0,0,0.4,0.6],[0,0,0,1]])
            observation_probability=np.array([[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5]])
            next_state=np.random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=np.random.choice(len(observation_probability[next_state,:]),observation_probability[next_state,:]).argmax()
            
        if action==1:##Manufacture with observation 
            reward=-1
            transition_probability=np.array([[0.5,0.3,0.15,0.05],[0,0.6,0.3,0.1],[0,0,0.4,0.6],[0,0,0,1]])
            observation_probability=np.array([[0.3,0.7],[0.6,0.4],[0.8,0.2],[0.9,0.1]])
            next_state=np.random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(1,observation_probability[next_state,:]).argmax()
            
        if action==2:
            
            transition_probability=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
            observation_probability=np.array([[0,1],[0,1],[0,1],[0,1]])
            next_state=np.random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(1,observation_probability[next_state,:]).argmax()
            reward=-1-2*self.current_state 
            
        if action==3:
            transition_probability=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
            observation_probability=np.array([[0,1],[0,1],[0,1],[0,1]])
            next_state=np.random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=np.random.multinomial(1,observation_probability[next_state,:]).argmax()
            reward=-3 

            
        self.current_state=next_state
        
        return observation,reward
    def number_of_actions(self):
        actions=4
        return actions
    def number_of_observations(self):
        observations=2
        return observations
    def reset(self):
        self.current_state=copy.deepcopy(self.start_state)
        return self.current_state
    