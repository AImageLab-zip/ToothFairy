class DummyExperiment:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_parameters(self, *args, **kwargs):
        pass

    def log_metric(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_table(self, *args, **kwargs):
        pass

    def log_current_epoch(self, *args, **kwargs):
        pass

    def log_figure(self, *args, **kwargs):
        pass

    def log_image(self, *args, **kwargs):
            pass

    def log_scene(self, *args, **kwargs):
        pass

    def train(self):
        return self