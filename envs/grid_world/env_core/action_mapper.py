class ActionMapper():
    def __init__(self, in_space, out_space, encode_func):
        self.in_space = in_space
        self.out_space = out_space
        self.encode_func = encode_func

    def encode(self, action):
        return self.encode_func(action)
