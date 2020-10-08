import numpy as np
import copy
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register
import gym

class CheeseMazeEnv(gym.Env):
    def __init__(self):
        self.start_state_prob=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0])
        self.start_state=None
        self.current_state=copy.deepcopy(self.start_state)
        self.name="CheeseMaze"
        self.discount=0.7
        self.renewal=False
        self.observation_space=spaces.Discrete(7)
        self.action_space=spaces.Discrete(4)
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        observation_probability=np.array([[1,0,0,0,0,0,0],
                                              [0,1,0,0,0,0,0],
                                              [0,0,1,0,0,0,0],
                                              [0,1,0,0,0,0,0],
                                              [0,0,0,1,0,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,1,0,0],
                                              [0,0,0,0,0,1,0],
                                              [0,0,0,0,0,1,0],
                                              [0,0,0,0,0,0,1]])
        if action==0: #actions =0,1,2,3 which stands for N,S,E,W
            transition_probability=np.array([[1,0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0],
                                             [1,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0],
                                             [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]])
            

            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=self.np_random.multinomial(1,observation_probability[next_state,:]).argmax()
            self.renewal=self.current_state==next_state
        if action==1:
            transition_probability=np.array([[0,0,0,0,0,1,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,1],
                                             [0,0,0,0,0,0,0,0,0,1,0],
                                             [0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0],
                                             [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]])
            
            
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=self.np_random.multinomial(1,observation_probability[next_state,:]).argmax()
            self.renewal=self.current_state==next_state
        if action==2:
            transition_probability=np.array([[0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0],
                                             [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]])
            
            
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=self.np_random.multinomial(1,observation_probability[next_state,:]).argmax()
            self.renewal=self.current_state==next_state
        if action==3:
            transition_probability=np.array([[1,0,0,0,0,0,0,0,0,0,0],
                                             [1,0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0],
                                             [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]])
            
            
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=self.np_random.multinomial(1,observation_probability[next_state,:]).argmax()
            self.renewal=self.current_state==next_state
        if next_state==10:
          reward=1.
        else:
          reward=0.
        self.current_state=next_state

        return observation,reward,False,{}
    
    def reset(self):
        observation_probability=np.array([[1,0,0,0,0,0,0],
                                          [0,1,0,0,0,0,0],
                                          [0,0,1,0,0,0,0],
                                          [0,1,0,0,0,0,0],
                                          [0,0,0,1,0,0,0],
                                          [0,0,0,0,1,0,0],
                                          [0,0,0,0,1,0,0],
                                          [0,0,0,0,1,0,0],
                                          [0,0,0,0,0,1,0],
                                          [0,0,0,0,0,1,0],
                                          [0,0,0,0,0,0,1]])
        self.start_state=self.np_random.multinomial(1,self.start_state_prob).argmax()
        self.current_state=copy.deepcopy(self.start_state)
        
        observation=self.np_random.multinomial(1,observation_probability[self.current_state,:]).argmax()
        return observation
