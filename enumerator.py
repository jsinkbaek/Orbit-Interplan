class Enumerator:
    def __init__(self):
        self.counter = 0

    def __call__(self):
        self.counter = self.counter + 1
        return self.counter
