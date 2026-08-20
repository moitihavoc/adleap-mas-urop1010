import numpy as np

class TraceField:
    """
    Trace Field Class: Manage fields of trace components.
    Components:
            intensity: float - how much attention should we give to the signal 
            age: how many steps ago was this trace created
            level: the level of the agent creating this signal
            help signal: either 0 or 1; 1 for requires help
            claim signal: either 0 or 1; 1 for having claimed a task in this cell  
    Methods:
        decay: decay the intensity and age of all traces in the field
        diffuse: diffuse the trace to 4 neighboring cells in the field
        emit: emit a trace in the field
        fusion: resolve overlapping traces in the field
        reset: clear all traces
    """

    def __init__(self, dim: int, decay_rate:float = 0.2, diffusion_rate: float = 0.3, num_fields:int = 5):
        self.dim = dim
        self.decay_rate = decay_rate
        self.diffusion_rate = diffusion_rate
        self.fields = np.zeros((num_fields, dim, dim), dtype=Trace)  # 5 fields: intensity, age, level, help, claim

    def decay(self):
        self.fields[0] = np.maximum(0.0, self.fields[0] - self.decay_rate)  # Decay intensity
        self.fields[1] += 1.0  # Increment age
        self.fields[:,self.fields <= 0] = 0.0  # Remove traces with zero intensity

    def diffuse(self):
        """
        Diffuse the trace to 4 neighboring cells in the field with 10% intensity.
        """
        mask = self.fields[0] == 1
        # Shift Down (targets the top neighbor of each 1)
        self.fields[0][1:, :] += mask[:-1, :] * self.diffusion_rate

        # Shift Up (targets the bottom neighbor of each 1)
        self.fields[0][:-1, :] += mask[1:, :] * self.diffusion_rate

        # Shift Right (targets the left neighbor of each 1)
        self.fields[0][:, 1:] += mask[:, :-1] * self.diffusion_rate

        # Shift Left (targets the right neighbor of each 1)
        self.fields[0][:, :-1] += mask[:, 1:] * self.diffusion_rate
        return

    def emit(self, level:float, help:int, claim:int, position:tuple):
        """
        Emit a trace in the field at the given position.
        """
        x, y = position
        self.fields[0, x, y] = 1.0  
        self.fields[1, x, y] = 1.0
        self.fields[2, x, y] = level
        self.fields[3, x, y] = help
        self.fields[4, x, y] = claim

    def fusion(self, level:float, help:int, claim:int, position:tuple, intensity:float = 1.0):
        """
        Resolve overlapping traces in the field.
        Priority: help signal > intensity.
        """
        x, y = position
        if self.fields[3, x, y]  == 1:
            pass
        elif help == 1:
            self.emit(level, help, claim, position)
        else:
            if self.fields[0, x, y] < intensity:
                self.emit(level, help, claim, position)

    def reset(self):
        self.fields.fill(0.0)  # Clear all traces
