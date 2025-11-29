import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils


#state space: {0, 1}
#0 means item not learned
#1 means item is learned

#observation space: {0, 1}
#0 means student gives a wrong answer
#1 means that student gives a right answer

#action space: {0, 1}
#0 means continue the instruction
#1 means terminate the instruction
class InstructionEnv(gym.Env):
    def __init__(self):
        self.start_state_probs = np.array([1.,0.]) #always start in unlearned state
        self.start_state = np.random.choice(2, 1, p=self.start_state_probs)
        self.current_state = self.start_state
        self.name = "Instruction"

        self.state_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)

        self.t = 0.6 #probability of learning (transition from unlearned to learned state) -> dynamics
        self.c = 0.8 #probability of consistency (item is learned and answer is correct) -> obs
        self.l = 0.1 #probability of getting lucky (item is not learned but answer is correct) -> obs
        self.I = -1.#reward for continuing instruction -> -cost
        self.C = -20.#reward for not learning after termination -> -cost
        self.R = 20. #reward for teaching correctly

    def step(self, action):
        if action==0: #continue the instruction
            reward = self.I
            transition_probability = np.array([[1.-self.t, self.t],[0.,1.]])
            observation_probability = np.array([[1-self.l, self.l],[1-self.c, self.c]])

            next_state = np.random.choice(2, 1, p=transition_probability[self.current_state.item(), :])
            observation = np.random.choice(2, 1, p=observation_probability[next_state.item(), :])

            self.current_state = next_state

            return observation.item(), reward, False, {}
        
        if action==1: #terminate the instruction
            # print ('Final state is:', self.current_state.item())
            if self.current_state.item() == 0:
                reward = self.C
            else:
                reward = self.R

            return self.reset(), reward, False, {}
        
    
    def reset(self):
        observation_probability = np.array([[1-self.l, self.l],[1-self.c, self.c]])
        self.start_state = np.random.choice(2, 1, p=self.start_state_probs)
        self.current_state = self.start_state
        observation = np.random.choice(2, 1, p=observation_probability[self.current_state.item(), :])
        return observation.item()



            
            
