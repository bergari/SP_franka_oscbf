import numpy as np

class OneEuroFilter:
    """ Implements the One Euro Filter for smoothing 3D joint positions over time."""
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.1, beta=0.5, d_cutoff=1.0):
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = np.full(x0.shape, dx0) if isinstance(dx0, float) else dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def alpha(self, t, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0: return x
        a_d = self.alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        speed = np.linalg.norm(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        a = self.alpha(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.t_prev = t
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
