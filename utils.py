import numpy as np

class EMA:
    """Exponential moving average over class logits."""
    def __init__(self, n_classes, alpha=0.6):
        self.alpha = alpha
        self.value = np.zeros(n_classes, dtype=np.float32)

    def reset(self):
        self.value[:] = 0

    def update(self, logits):
        self.value = self.alpha * logits + (1 - self.alpha) * self.value
        return self.value

class Debouncer:
    """Emit a class only if confidence stays above τ for N consecutive updates."""
    def __init__(self, threshold=0.65, hold=4):
        self.threshold = threshold
        self.hold = hold
        self.count = 0
        self.current = None

    def step(self, cls_id, conf):
        if conf >= self.threshold:
            if cls_id == self.current:
                self.count += 1
            else:
                self.current = cls_id
                self.count = 1
        else:
            self.current = None
            self.count = 0

        return (self.current if self.count >= self.hold else None)
