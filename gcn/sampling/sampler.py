from datetime import datetime


class Sampler:
    def __init__(self, initial_train_mask, adj):
        self.initial_train_mask = initial_train_mask
        self.adj = adj
        self.label_percent = 0
        self.sampling_config_index = 0
        self.num_nodes = 0

    def precomputations(self):
        pass

    # By default, only one configuration of the sampling algo is tried out
    def next_parameter(self):
        if self.sampling_config_index == 0:
            self.sampling_config_index += 1
            return True
        return False

    def set_info(self, settings, result):
        self.info = {
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'params': settings
        }
        self.dict_output = {'results': result, 'info': self.info}
