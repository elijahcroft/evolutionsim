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

# -- species ---------------------------------------------------------------------------------


def diverged_client(config: Config) -> TestClient:
    """A session whose founders are already two kinds of organism, so a split happens."""
    app = create_app(config)
    population = app.state.session.simulation.population
    schema = population.schema
    for name in ("maturity_age", "repro_threshold", "offspring_count", "sex_bias"):
        index = schema.index_of(name)
        population.genomes[: population.size // 2, index, :] = schema.high[index]
    population.phenotypes.update(0, population.genomes[: population.size], schema)
    return TestClient(app)


def test_species_lists_the_founder_before_anything_splits(client: TestClient) -> None:
    payload = client.get("/api/species").json()
    assert len(payload["trait_names"]) == 28
    assert [entry["species_id"] for entry in payload["species"]] == [0]
    assert payload["species"][0]["parent"] is None
    assert payload["species"][0]["lineage"] == [0]


def test_a_split_appears_in_the_species_list_and_the_state(config: Config) -> None:
    client = diverged_client(
        Config.load(DEFAULT_CONFIG_DIR, overrides=["sim.seed=11", "sim.taxonomy_interval=2"])
    )
    state = client.post("/api/step", json={"ticks": 2}).json()
    assert state["species"] == 2
    assert state["stats"]["species_born"] == 1
    # Newest first: a run is read from its present backwards.
    listing = client.get("/api/species").json()["species"]
    assert [entry["species_id"] for entry in listing] == [1, 0]
    assert listing[0]["lineage"] == [0, 1]


def test_a_species_report_says_what_changed_since_it_appeared(client: TestClient) -> None:
    client.post("/api/step", json={"ticks": 4})
    report = client.get("/api/species/0").json()
    assert report["species_id"] == 0
    assert report["lineage"] == [0]
    assert len(report["current_traits"]) == len(report["trait_names"]) == 28
    assert len(report["drift"]) == 28
    # Ranked by how far each trait moved relative to what its locus can do.
    magnitudes = [abs(entry["span_fraction"]) for entry in report["drift"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert report["drift"][0]["now"] - report["drift"][0]["origin"] == pytest.approx(
        report["drift"][0]["change"]
    )


def test_a_species_report_carries_its_sampled_time_series(config: Config) -> None:
    client = TestClient(
        create_app(
            Config.load(DEFAULT_CONFIG_DIR, overrides=["sim.seed=11", "sim.sample_interval=2"])
        )
    )
    client.post("/api/step", json={"ticks": 6})
    series = client.get("/api/species/0").json()["series"]
    assert [point["day"] for point in series] == [2, 4, 6]
    assert all(point["population"] > 0 for point in series)


def test_an_unknown_species_is_a_404(client: TestClient) -> None:
    assert client.get("/api/species/7").status_code == 404


def test_the_map_can_be_restricted_to_one_species(config: Config) -> None:
    client = diverged_client(
        Config.load(DEFAULT_CONFIG_DIR, overrides=["sim.seed=11", "sim.taxonomy_interval=2"])
    )
    client.post("/api/step", json={"ticks": 2})
    whole = client.get("/api/state?layer=population").json()["layer"]
    part = client.get("/api/state?layer=population&species=1").json()["layer"]
    assert whole["species"] is None and part["species"] == 1
    assert sum(part["values"]) < sum(whole["values"])
    assert sum(part["values"]) > 0
    # A filter is a filter, not a different world: the two must agree cell by cell.
    assert all(a <= b for a, b in zip(part["values"], whole["values"]))


def test_the_species_filter_only_touches_the_population_layer(client: TestClient) -> None:
    plain = client.get("/api/state?layer=temperature_c").json()["layer"]
    filtered = client.get("/api/state?layer=temperature_c&species=0").json()["layer"]
    assert filtered["species"] is None
    assert filtered["values"] == plain["values"]


def test_filtering_by_a_species_that_never_existed_is_a_404(client: TestClient) -> None:
    """An empty map looks like an answer, so a mistaken id must not get one."""
    assert client.get("/api/state?layer=population&species=3").status_code == 404
    assert client.post("/api/step", json={"ticks": 1, "species": 3}).status_code == 404
    # ...and the rejected step did not advance the world on its way to failing.
    assert client.get("/api/state").json()["day"] == 0
