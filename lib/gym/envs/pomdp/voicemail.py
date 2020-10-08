import numpy as np
import copy
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register
import gym



class VoicemailEnv(gym.Env):
    def __init__(self):
        self.start_state_probs=np.array([0.65,0.35])#assuming user wants the message to be saved
        self.start_state=None
        self.current_state=copy.deepcopy(self.start_state)
        self.name="Voicemail"
        self.discount=0.95
        self.renewal=False
        self.action_space=spaces.Discrete(3)
        self.observation_space=spaces.Discrete(2)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        if action==0:#refers to asking the user
            transition_probability=np.array([[1,0],[0,1]])
            reward=-1
            observation_probability=np.array([[0.8,0.2],[0.3,0.7]])
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=self.np_random.multinomial(1,observation_probability[next_state,:]).argmax()
            self.renewal=self.current_state==next_state

        if action==1:#refers to Saving the data
            transition_probability=np.array([[0.65,0.35],[0.65,0.35]])
            rewards=[5,-10]
            observation_probability=np.array([[0.5,0.5],[0.5,0.5]])
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=self.np_random.multinomial(1,observation_probability[next_state,:]).argmax()
            reward=rewards[self.current_state]
            self.renewal=self.current_state==next_state

        if action==2:#refers to deleting the data
            transition_probability=np.array([[0.65,0.35],[0.65,0.35]])
            rewards=[-20,5]
            observation_probability=np.array([[0.5,0.5],[0.5,0.5]])
            next_state=self.np_random.multinomial(1,transition_probability[self.current_state,:]).argmax()
            observation=self.np_random.multinomial(1,observation_probability[next_state,:]).argmax()
            reward=rewards[self.current_state]
            self.renewal=self.current_state==next_state
            
        self.current_state=next_state
        return observation,reward,False,{}
    
    def reset(self):
        self.start_state=self.np_random.multinomial(1,self.start_state_probs).argmax()
        self.current_state=copy.deepcopy(self.start_state)

        observation_probability=np.array([0.5,0.5])
        observation=self.np_random.multinomial(1,observation_probability).argmax()
        return observation



            
            
