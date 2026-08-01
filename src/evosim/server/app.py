"""A single live simulation, exposed over HTTP.

The browser drives time: there is no background thread advancing the world, so the server is
idle unless a step is requested and a paused session is genuinely frozen rather than merely
unwatched.  Stepping is serialized under a lock because a browser can easily have two step
requests in flight, and two overlapping ``Simulation.step`` calls would corrupt the population
arrays.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from evosim import __version__
from evosim.config import DEFAULT_CONFIG_DIR, Config, ConfigError
from evosim.life import HabitatError
from evosim.sim import Simulation, TickStats

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

#: Field name in ``World.arrays()`` (or the derived ``population``) → label shown in the UI.
LAYERS: dict[str, str] = {
    "population": "Population density",
    "elevation_km": "Elevation",
    "depth_km": "Water depth",
    "temperature_c": "Temperature",
    "nutrients": "Nutrients",
    "detritus": "Detritus",
    "moisture": "Moisture",
    "insolation": "Insolation (surface)",
    "light": "Light (at depth)",
    "toxicity": "Toxicity",
}

#: How many ticks of history a session keeps for the charts.
HISTORY_LENGTH = 1200

#: Upper bound on ticks per request, so one call cannot hang the server for minutes.
MAX_TICKS_PER_REQUEST = 1000


@dataclass(slots=True)
class Session:
    """One simulation plus the recent history the UI plots."""

    config: Config
    simulation: Simulation
    history: deque[dict[str, float | int]] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LENGTH)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(cls, config: Config) -> Session:
        return cls(config=config, simulation=Simulation.create(config))

    def step(self, ticks: int) -> None:
        """Advance ``ticks`` days, recording one history sample per day."""

        for _ in range(ticks):
            stats = self.simulation.step()
            self.history.append(_history_sample(stats))

    def meta(self) -> dict[str, Any]:
        planet = self.config.planet
        return {
            "version": __version__,
            "planet": planet.name,
            "seed": self.config.sim.seed,
            "fingerprint": self.config.fingerprint(),
            "width": planet.grid_width,
            "height": planet.grid_height,
            "capacity": self.simulation.population.capacity,
            "year_length_days": planet.year_length_days,
            "layers": [{"name": name, "label": label} for name, label in LAYERS.items()],
            # The creature viewer builds its mesh from these, which is what makes the animal on
            # screen the same animal the energy model charges: both read one set of constants.
            "morphology": asdict(self.config.genome.morphology),
        }

    def state(self, layer: str, species: int | None = None) -> dict[str, Any]:
        simulation = self.simulation
        return {
            "day": simulation.world.day,
            "population": simulation.population.size,
            "capacity": simulation.population.capacity,
            "extinct": simulation.population.size == 0,
            "species": len(simulation.history.living()),
            "stats": _stats_payload(simulation.last_stats),
            "history": list(self.history),
            "layer": self._layer_payload(layer, species),
        }

    def species(self) -> dict[str, Any]:
        """Every species the run has produced, for the list the UI selects from."""

        simulation = self.simulation
        return {
            "day": simulation.world.day,
            "trait_names": list(simulation.population.phenotypes.trait_names),
            "species": simulation.history.summary(simulation.population),
        }

    def report(self, species: int) -> dict[str, Any]:
        """One species in full, including what has changed in it since it appeared."""

        simulation = self.simulation
        return simulation.history.report(species, simulation.population)

    def _layer_payload(self, layer: str, species: int | None = None) -> dict[str, Any]:
        if layer not in LAYERS:
            raise KeyError(layer)
        if layer == "population":
            # Restricting density to one species turns the map into a range map, which is how
            # you see that a split is geographic rather than merely numerical.
            values = self._density(species)
        elif layer == "elevation_km":
            # Elevation is the one layer drawn relative to sea level, because "is this cell
            # land?" is the question it is actually being read for.
            values = (
                self.simulation.world.terrain.elevation_km
                - self.config.planet.terrain.sea_level
            )
        else:
            values = self.simulation.world.arrays()[layer]
        values = np.asarray(values, dtype=np.float64)
        # A filter only means anything on the layer that counts organisms, so it is reported as
        # unset everywhere else rather than being silently accepted and ignored.
        filtered = species if layer == "population" else None
        label = LAYERS[layer]
        if filtered is not None:
            label = f"{label} — species {filtered}"
        return {
            "name": layer,
            "label": label,
            "species": filtered,
            "min": float(values.min()),
            "max": float(values.max()),
            "values": np.round(values, 4).ravel().tolist(),
            "land": self.simulation.world.terrain.land.ravel().tolist(),
        }

    def _density(self, species: int | None = None) -> np.ndarray:
        population = self.simulation.population
        cells = population.cell[population.active].astype(np.intp)
        if species is not None:
            cells = cells[population.species_id[population.active] == species]
        grid = self.simulation.world.grid
        return np.bincount(cells, minlength=grid.n_cells).reshape(grid.shape)


def _history_sample(stats: TickStats) -> dict[str, float | int]:
    return {
        "day": stats.day,
        "population": stats.population,
        "births": stats.births,
        "deaths": stats.deaths,
        "net_energy": stats.net_energy,
    }


def _stats_payload(stats: TickStats | None) -> dict[str, Any] | None:
    """Flatten the last tick's ledger, keeping the cause breakdowns intact."""

    if stats is None:
        return None
    return {
        "births": stats.births,
        "sexual_births": stats.sexual_births,
        "breeding_parents": stats.breeding_parents,
        "capacity_throttle": stats.capacity_throttle,
        "species": stats.species,
        "species_born": stats.species_born,
        "species_extinct": stats.species_extinct,
        "deaths": stats.deaths,
        "deaths_starvation": stats.deaths_starvation,
        "deaths_hazard": stats.deaths_hazard,
        "deaths_predation": stats.deaths_predation,
        "attacks": stats.attacks,
        "kills": stats.kills,
        "hunters": stats.hunters,
        "energy_intake": stats.energy_intake,
        "energy_cost": stats.energy_cost,
        "net_energy": stats.net_energy,
        "intake_autotrophy": stats.intake_autotrophy,
        "intake_detritivory": stats.intake_detritivory,
        "intake_predation": stats.intake_predation,
        "cost_basal": stats.cost_basal,
        "cost_support": stats.cost_support,
        "cost_locomotion": stats.cost_locomotion,
        "cost_sensory": stats.cost_sensory,
        "cost_thermoregulation": stats.cost_thermoregulation,
        "cost_armor": stats.cost_armor,
        "cells_moved": stats.cells_moved,
        "mean_energy": stats.mean_energy,
        "mean_energy_fullness": stats.mean_energy_fullness,
    }


class StepRequest(BaseModel):
    ticks: int = Field(default=1, ge=0, le=MAX_TICKS_PER_REQUEST)
    layer: str = "population"
    species: int | None = Field(default=None, ge=0)


class ResetRequest(BaseModel):
    seed: int | None = None
    layer: str = "population"


def default_reload(seed: int | None) -> Config:
    """Load the default planet, optionally reseeded."""

    overrides = [] if seed is None else [f"sim.seed={seed}"]
    return Config.load(DEFAULT_CONFIG_DIR, overrides=overrides)


def create_app(
    config: Config | None = None,
    reload: Callable[[int | None], Config] = default_reload,
) -> FastAPI:
    """Build the app around one session, rebuilt in place by ``/api/reset``.

    ``reload`` is how a reseeded reset gets a config: the caller supplies it so that a session
    started with ``--planet`` or ``--set`` is reseeded from *those* files rather than silently
    reverting to the defaults.
    """

    if config is None:
        config = reload(None)
    app = FastAPI(title="evosim", version=__version__)
    app.state.session = Session.create(config)

    def session() -> Session:
        return app.state.session

    def require_species(species: int | None) -> None:
        """Reject an unknown species id.

        A 404 rather than an empty map: a caller asking about a species that never existed has
        made a mistake, and an empty map looks like an answer.  Checked *before* a step runs, so
        a rejected request does not quietly advance the world on its way to failing.
        """

        if species is not None and species not in session().simulation.history.records:
            raise HTTPException(status_code=404, detail=f"no such species: {species}")

    def state_or_404(layer: str, species: int | None = None) -> dict[str, Any]:
        require_species(species)
        try:
            return session().state(layer, species)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no such layer: {layer}") from None

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return session().meta()

    @app.get("/api/state")
    def state(layer: str = "population", species: int | None = None) -> dict[str, Any]:
        return state_or_404(layer, species)

    @app.get("/api/species")
    def species_list() -> dict[str, Any]:
        return session().species()

    @app.get("/api/species/{species}")
    def species_report(species: int) -> dict[str, Any]:
        try:
            return session().report(species)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"no such species: {species}"
            ) from None

    @app.post("/api/step")
    def step(request: StepRequest) -> dict[str, Any]:
        current = session()
        with current.lock:
            require_species(request.species)
            current.step(request.ticks)
            return state_or_404(request.layer, request.species)

    @app.post("/api/reset")
    def reset(request: ResetRequest) -> dict[str, Any]:
        current = session()
        with current.lock:
            config = current.config
            if request.seed is not None and request.seed != config.sim.seed:
                try:
                    config = reload(request.seed)
                except ConfigError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from None
            try:
                app.state.session = Session.create(config)
            except HabitatError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
        return state_or_404(request.layer)

    return app


__all__ = ["LAYERS", "Session", "create_app"]
