
class ManualAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # try to make the encoding part separate
    def __init__(self):
        pass

    def step(self, env_input):
        pass

    def reset_buffers(self):
        self.rewards = []
        self.probs = []
        self.action_probs = []
        self.actions = []
        self.value_estimates = []

    def should_update(self, episode_end, reward):
        steps_since_update = len(self.rewards) + 1
        td_update = self.td_step != -1 and steps_since_update % self.td_step == 0
        if self.update_threshold == -1:  # not trying the threshold updater
            return episode_end or td_update
        update = episode_end or np.abs(reward) >= self.update_threshold
        return update


def case_1():
    pass

def interactive():
    pass

def main():
    case_1()
    interactive()

if __name__ == "__main__":
    main()