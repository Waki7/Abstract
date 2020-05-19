class NaivePolicy():
    def __init__(self):
        pass


class RandomPolicy(NaivePolicy):
    def __init__(self):
        super(RandomPolicy, self).__init__()


class FollowPolicy(NaivePolicy):
    def __init__(self):
        super(FollowPolicy, self).__init__()


class AvoidPolicy(NaivePolicy):
    def __init__(self):
        super(AvoidPolicy, self).__init__()
