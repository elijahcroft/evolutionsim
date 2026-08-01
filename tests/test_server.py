"""The HTTP boundary: it must expose the simulation without changing what the simulation does."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from evosim.config import DEFAULT_CONFIG_DIR, Config
from evosim.server.app import LAYERS, MAX_TICKS_PER_REQUEST, create_app
from evosim.sim import Simulation


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load(DEFAULT_CONFIG_DIR, overrides=["sim.seed=11"])


@pytest.fixture
def client(config: Config) -> TestClient:
    return TestClient(create_app(config))


def test_meta_describes_the_planet(client: TestClient, config: Config) -> None:
    meta = client.get("/api/meta").json()
    assert meta["planet"] == config.planet.name
    assert meta["seed"] == config.sim.seed
    assert meta["fingerprint"] == config.fingerprint()
    assert (meta["width"], meta["height"]) == (
        config.planet.grid_width,
        config.planet.grid_height,
    )
    assert [layer["name"] for layer in meta["layers"]] == list(LAYERS)


def test_initial_state_has_founders_and_no_tick_yet(client: TestClient, config: Config) -> None:
    state = client.get("/api/state").json()
    assert state["day"] == 0
    assert state["population"] == config.sim.initial_population
    assert state["stats"] is None
    assert state["history"] == []


@pytest.mark.parametrize("layer", list(LAYERS))
def test_every_layer_renders_one_value_per_cell(
    client: TestClient, config: Config, layer: str
) -> None:
    payload = client.get(f"/api/state?layer={layer}").json()["layer"]
    n_cells = config.planet.grid_width * config.planet.grid_height
    assert payload["name"] == layer
    assert len(payload["values"]) == n_cells
    assert len(payload["land"]) == n_cells
    assert payload["min"] <= payload["max"]


def test_unknown_layer_is_a_404(client: TestClient) -> None:
    assert client.get("/api/state?layer=phlogiston").status_code == 404


def test_stepping_advances_the_day_and_records_history(client: TestClient) -> None:
    state = client.post("/api/step", json={"ticks": 3}).json()
    assert state["day"] == 3
    assert [sample["day"] for sample in state["history"]] == [1, 2, 3]
    assert state["stats"]["deaths"] == (
        state["stats"]["deaths_starvation"]
        + state["stats"]["deaths_hazard"]
        + state["stats"]["deaths_predation"]
    )


def test_step_ticks_are_bounded(client: TestClient) -> None:
    assert client.post("/api/step", json={"ticks": -1}).status_code == 422
    over = MAX_TICKS_PER_REQUEST + 1
    assert client.post("/api/step", json={"ticks": over}).status_code == 422


def test_reset_restores_day_zero_and_reseeds(client: TestClient, config: Config) -> None:
    client.post("/api/step", json={"ticks": 2})
    state = client.post("/api/reset", json={"seed": 12}).json()
    assert state["day"] == 0
    assert state["history"] == []
    assert client.get("/api/meta").json()["seed"] == 12
    assert config.sim.seed == 11  # the caller's config object is not mutated


def test_serving_matches_a_headless_run(config: Config) -> None:
    """The point of the whole layer: driving from HTTP must not perturb the run.

    A browser session and ``evosim --ticks 20`` on the same ``(config, seed)`` have to land on
    the same state, or the UI would be showing a run nobody else can reproduce.
    """

    headless = Simulation.create(config)
    headless_stats = headless.run(20)

    client = TestClient(create_app(config))
    # Split across requests, since that is how the UI actually steps.
    client.post("/api/step", json={"ticks": 7})
    served = client.post("/api/step", json={"ticks": 13}).json()

    assert served["day"] == headless.world.day
    assert served["population"] == headless.population.size
    assert served["stats"]["births"] == headless_stats.births
    assert served["stats"]["deaths"] == headless_stats.deaths
    assert served["stats"]["energy_intake"] == pytest.approx(headless_stats.energy_intake)


def test_index_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>evosim</title>" in response.text
