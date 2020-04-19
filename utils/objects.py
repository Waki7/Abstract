

class ActionNode():
    def __init__(self, space, next):
        self.results_path = ''
        self.writer = None
        self.progress_values_sum = {}
        self.progress_values_mean = {}
        self.counts = {}
