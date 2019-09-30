class experiment:
    UPDATE_RATE = 1
    MAX_EPISODES = 5000
    MAX_STEPS = 20000
    ENV = 'Life-v0' # 'CartPole-v0'
    EVAL_REWARDS_WINDOW = 10

class pg:
    CONTINUOUS_EPISODE_LENGTH = 10

backprop_through_input = True
experiment_name = 'initial'
results_path = 'logs/LifeSimulationWorld/' + experiment_name + '/'

# running continuous
iterations = 2000
rewards_eval_window = 10

# running episodic
MAX_EPISODES = 5000
max_steps = 10000
episode_eval_window = 1

# network parameters
LR = .1
max_ep_window = 5
discount_factor = .9
clip_value = 5.0

# agent parameters
reward_update_min = .1
adjacent_reward_list = [0, .1, .1, -1, -1, 2.3, 0, -1]  # adjacent to Feels
reward_prediction_discount = .2

# compuation differences
self_reward_update = True

# network
# 'SGD', 'Adam'
class life:
    LR = .001
    OPTIMIZER = 'Adam'
    EXPLOITATION_PENALTY = -.3

class gym:
    LR = .001
    OPTIMIZER = 'Adam'
    _atari7 = ['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders']
    # gym_env = 'Pong-v0'
    # gym_env = 'MountainCar-v0'
    gym_env = 'CartPole-v0'
    hidden_size = 128


