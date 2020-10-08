import numpy as np
import copy
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register
import gym



class FourByFourMazeEnv(gym.Env):
    def __init__(self):
        self.start_state_prob=np.array([1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,0])
        self.start_state=None
        self.current_state=copy.deepcopy(self.start_state)
        self.name="4x4 Maze"
        self.discount=0.95
        self.renewal=False
        self.action_space=spaces.Discrete(4)
        self.observation_space=spaces.Discrete(2)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self,action):
        if action==0:
            transition_probability=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                             [1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,0]])
            
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            self.renewal=self.current_state==next_state

        if action==1:
            transition_probability=np.array([[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                             [1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,0]])
        
           
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            self.renewal=self.current_state==next_state

        if action==2:
            transition_probability=np.array([[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                             [1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,0]])
        
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            self.renewal=self.current_state==next_state

        if action==3:
            transition_probability=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                             [1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,1/15,0]])
        
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            self.renewal=self.current_state==next_state

        if next_state==15:
            observation=1
            reward=1
        else:
            observation=0
            reward=0
        self.current_state=next_state
        return observation,reward,False,{}
    
    def reset(self):
        self.start_state=self.np_random.multinomial(1,self.start_state_prob).argmax()
        self.current_state=copy.deepcopy(self.start_state)
        observation=0
        return observation