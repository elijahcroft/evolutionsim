"""Grouping organisms by the cell they occupy.

Both mate search and predation ask the same question -- "pick another organism in my cell" --
and both must answer it for tens of thousands of organisms without a Python loop.  Sorting the
candidates by cell once turns that question into a pair of :func:`numpy.searchsorted` bounds,
which is what makes the answer a handful of vectorised operations rather than a scan per asker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CellNeighbourhood:
    """Population rows grouped into contiguous runs, one run per occupied cell."""

    rows: IntArray
    """The candidate rows, ascending. Callers locate a row here with searchsorted."""

    members: IntArray
    """The same rows, reordered so that everyone sharing a cell is contiguous."""

    member_cells: IntArray
    """The cell of each entry in ``members``, ascending."""

    slot: IntArray
    """Where ``rows[k]`` ended up in ``members``; used to skip over an asker itself."""

    @classmethod
    def build(cls, rows: IntArray, cells: NDArray[np.integer]) -> CellNeighbourhood:
        """Group ``rows`` by their cell.

        ``rows`` must be ascending, which is what :func:`numpy.flatnonzero` already returns, so
        a caller can find any row's index by searchsorted rather than by an inverse table.
        """

        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(cells[rows], kind="stable")
        members = rows[order]
        slot = np.empty(rows.size, dtype=np.int64)
        slot[order] = np.arange(rows.size, dtype=np.int64)
        return cls(
            rows=rows,
            members=members,
            member_cells=np.asarray(cells[members], dtype=np.int64),
            slot=slot,
        )

    @property
    def size(self) -> int:
        return int(self.rows.size)

    def occupancy(self, cells: NDArray[np.integer]) -> IntArray:
        """How many candidates share each of the given cells."""

        return np.searchsorted(self.member_cells, cells, side="right") - np.searchsorted(
            self.member_cells, cells, side="left"
        )

    def sample_other(
        self,
        seekers: NDArray[np.integer],
        cells: NDArray[np.integer],
        rng: np.random.Generator,
    ) -> tuple[IntArray, NDArray[np.bool_]]:
        """Draw one uniformly random *other* occupant of each seeker's cell.

        ``seekers`` are population rows that must themselves be in this neighbourhood; a row may
        repeat, which is how one predator makes several independent attacks in a tick.  The
        returned mask is false where a cell holds nobody but the seeker.

        Excluding the seeker is done by drawing from a block one shorter and then stepping over
        the seeker's own slot, so no rejection loop is needed and the draw stays uniform.
        """

        seekers = np.asarray(seekers, dtype=np.int64)
        if seekers.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)

        home = np.asarray(cells[seekers], dtype=np.int64)
        start = np.searchsorted(self.member_cells, home, side="left")
        occupants = np.searchsorted(self.member_cells, home, side="right") - start

        own = self.slot[np.searchsorted(self.rows, seekers)]
        draw = start + (
            rng.random(seekers.size) * np.maximum(occupants - 1, 1)
        ).astype(np.int64)
        draw += draw >= own
        chosen = self.members[np.clip(draw, 0, self.members.size - 1)]
        return chosen, occupants >= 2


__all__ = ["CellNeighbourhood"]
