"""HTTP boundary around a live :class:`~evosim.sim.Simulation`.

This layer may read the simulation and advance it; it must never be imported by it.  Time comes
from step requests rather than the wall clock, so a session driven from the browser produces the
same run as ``evosim --ticks N`` with the same ``(config, seed)``.
"""

from evosim.server.app import Session, create_app

__all__ = ["Session", "create_app"]
