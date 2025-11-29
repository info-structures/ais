import numpy as np
import copy
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gymnasium.envs.registration import register


class TigerEnv(gym.Env):
    def __init__(self):
        self.start_state_prob=np.array([0.5,0.5])
        self.start_state=None
        self.current_state=copy.deepcopy(self.start_state)
        self.name="Tiger"
        self.discount=0.95
        self.renewal=False
        self.action_space=spaces.Discrete(3)
        self.observation_space=spaces.Discrete(2)

    def step(self,action):
        if action==0: ##corresponds to listen action
            reward=-1
            transition_probability=np.array([[1,0],[0,1]])
            observation_probability=np.array([[0.85,0.15],[0.15,0.85]])
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=self.np_random.multinomial(1,observation_probability[next_state,:]).argmax()
            self.renewal=self.current_state==next_state
           
        if action==1: ##open left
            rewards=[-100,10]
            transition_probability=np.array([[0.5,0.5],[0.5,0.5]])
            observation_probability=np.array([[0.5,0.5],[0.5,0.5]])
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=self.np_random.multinomial(1,observation_probability[next_state,:]).argmax()
            reward=rewards[self.current_state]
            self.renewal=self.current_state==next_state

        if action==2: ##open right 
            rewards=[10,-100]
            transition_probability=np.array([[0.5,0.5],[0.5,0.5]])
            observation_probability=np.array([[0.5,0.5],[0.5,0.5]])
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=self.np_random.multinomial(1,observation_probability[next_state,:]).argmax()
            reward=rewards[self.current_state]
            self.renewal=self.current_state==next_state

        self.current_state=next_state
        return observation,reward,False,False,{}
   
    def reset(self, seed=None, options=None):
        self.start_state=self.np_random.multinomial(1,self.start_state_prob).argmax()
        self.current_state=copy.deepcopy(self.start_state)

        observation_probability=np.array([0.5,0.5])
        observation=self.np_random.multinomial(1,observation_probability).argmax()
        return observation
