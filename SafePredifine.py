class SafeCounter:
    def __init__(self, min_val=-10, max_val=10):
        self.value = 0
        self.min_val = min_val
        self.max_val = max_val

    def increment(self, amount=1):
        self.value = min(self.max_val, self.value + amount)

    def decrement(self, amount=1):
        self.value = max(self.min_val, self.value - amount)

    def reset(self):
        self.value = 0