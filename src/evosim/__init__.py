"""evosim -- an individual-based, scientifically grounded evolution sandbox.

Layering (dependencies point one way only):

    config -> world -> life -> evolution -> history -> sim -> server -> ui

`sim.Simulation` is pure and headless. It never reads the wall clock and never imports from
`server` or `ui`, which is what makes a run reproducible from (config, seed) alone.
"""

from evosim.config import Config, ConfigError
from evosim.rng import RngBundle

__all__ = ["Config", "ConfigError", "RngBundle", "__version__"]

__version__ = "0.0.1"
