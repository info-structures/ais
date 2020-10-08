# POMDP
register(
    id='FourByFourMaze-v0',
    entry_point='gym.envs.pomdp.fourbyfourmaze:FourByFourMazeEnv',
    max_episode_steps=300,
)

register(
    id='Tiger-v0',
    entry_point='gym.envs.pomdp.tiger:TigerEnv',
    max_episode_steps=300,
)

register(
    id='Voicemail-v0',
    entry_point='gym.envs.pomdp.voicemail:VoicemailEnv',
    max_episode_steps=300,
)

register(
    id='CheeseMaze-v0',
    entry_point='gym.envs.pomdp.cheesemaze:CheeseMazeEnv',
    max_episode_steps=300,
)

register(
    id='Instruction-v0',
    entry_point='gym.envs.pomdp.instruction:InstructionEnv',
    max_episode_steps=300,
)

register(
    id='DroneSurveillance-v0',
    entry_point='gym.envs.pomdp.dronesurveillance:DroneSurveillanceEnv',
    max_episode_steps=200,
)

register(
    id='RockSampling-v0',
    entry_point='gym.envs.pomdp.rocksampling:RockSamplingEnv',
    max_episode_steps=200,
)
