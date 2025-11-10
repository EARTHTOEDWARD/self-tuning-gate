from dataclasses import dataclass

@dataclass
class ShockGatePolicy:
    tau: float = 0.3
    high: float = 0.6

    def step(self, valence_t, Gamma_t):
        v = float(valence_t)
        g = float(Gamma_t)
        if v < -self.tau and g > self.high:
            return "correct"
        if v > self.tau and g > self.high:
            return "consolidate"
        return "coast"

    def state_dict(self):
        return {"tau": self.tau, "high": self.high}

    def metrics(self):
        return {}
