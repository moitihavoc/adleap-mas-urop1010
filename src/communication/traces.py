import numpy as np

class Trace:
    """
    Trace class: Represent traces in fields.
    Components:
        intensity: float - how much attention should we give to the signal 
        age: how many steps ago was this trace created
        level: the level of the agent creating this signal
        help signal: either 0 or 1; 1 for requires help
        claim signal: either 0 or 1; 1 for having claimed a task in this cell    
    """

    def __init__(self, level:float, help: int, claim:int, intensity:float = 1.0, age: float = 0.0):
        self.intensity = intensity
        self.age = age
        self.level = level
        self.help = help
        self.claim = claim

    