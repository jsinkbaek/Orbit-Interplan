class CallbackInfo:
    def __init__(self, ftol):
        self.ftol = ftol
        self.current_diff = None
        self.current_params = None

    def update(self, **kwargs):
        self.current_params = kwargs.get('current_params', self.current_params)
        self.current_diff = kwargs.get('current_diff', self.current_diff)

