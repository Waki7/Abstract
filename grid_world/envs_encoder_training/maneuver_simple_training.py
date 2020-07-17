import gym
import grid_world.envs.maneuver_simple as grid_world

class StateEncodingProtocol():
    def __init__(self):
        self.env: grid_world.ManeuverSimple = gym.make('Grid-v0')

    def get_in_shapes(self):
        pass
    
    def get_out_shapes(self):
        pass

    def generate_batch(self):
        self.env.reset()