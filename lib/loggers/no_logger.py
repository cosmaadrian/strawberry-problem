from collections import defaultdict
from collections import deque


class NoLogger(object):
    def __init__(self):
        self.on_step_metrics = defaultdict(lambda: deque(maxlen = 16))
        self.trainer = None

        self.metrics = dict()

        self.min_values = defaultdict(lambda: 10000)
        self.max_values = defaultdict(lambda: -1)

    def watch(self, model):
        pass

    def log_dict(self, *args, **kwargs):
        pass

    def log(self, key, value, on_step = True, force_log = False, log_min = False, log_max = False, log_instant = True):
        self.metrics[key] = value

        if on_step:
            self.on_step_metrics[key] = value

        if (self.trainer.global_step % self.trainer.args.log_every == 0) or force_log:
            log_dict = dict()

            if log_instant:
                log_dict[key] = value

            if log_min:
                self.min_values[key] = min(self.min_values[key], value)
                log_dict[key + ':min'] = self.min_values[key]

            if log_max:
                self.max_values[key] = max(self.max_values[key], value)
                log_dict[key + ':max'] = self.max_values[key]
