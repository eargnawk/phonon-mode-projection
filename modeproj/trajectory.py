"""Reading MD trajectories and turning them into displacements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    """A constant-cell MD trajectory in fractional coordinates."""

    cell: np.ndarray  #: (3, 3) lattice vectors in Angstrom, rows are vectors
    symbols: list  #: (nat,) chemical symbols
    scaled_positions: np.ndarray  #: (nframes, nat, 3) fractional coordinates

    @property
    def nframes(self) -> int:
        return self.scaled_positions.shape[0]

    @property
    def natoms(self) -> int:
        return self.scaled_positions.shape[1]

    def displacements(self, reference="first") -> np.ndarray:
        """Cartesian displacements in Angstrom, ``(nframes, nat, 3)``.

        Parameters
        ----------
        reference
            ``"first"`` uses the first frame of the trajectory, ``"mean"`` the
            time-averaged positions; an ``(nat, 3)`` array of fractional
            coordinates (e.g. ``ideal.get_scaled_positions()``) uses the ideal
            lattice sites.

        The fractional difference is folded into ``[-0.5, 0.5)`` first, so atoms
        that cross a periodic boundary do not produce spurious jumps.
        """
        if isinstance(reference, str):
            if reference == "first":
                ref = self.scaled_positions[0]
            elif reference == "mean":
                ref = _mean_scaled_positions(self.scaled_positions)
            else:
                raise ValueError("reference must be 'first', 'mean' or an array")
        else:
            ref = np.asarray(reference, dtype=float)
            if ref.shape != self.scaled_positions.shape[1:]:
                raise ValueError("reference must have shape (%d, 3)" % self.natoms)

        diff = self.scaled_positions - ref
        diff -= np.round(diff)  # minimum image convention
        return diff @ self.cell


def read_xdatcar(path: str, start: int = 0, stop: int | None = None,
                 stride: int = 1) -> Trajectory:
    """Read a VASP ``XDATCAR`` (constant cell) into a :class:`Trajectory`.

    This is a small dedicated parser rather than ``ase.io.read``: an XDATCAR with
    10000 frames is read in a fraction of a second.

    Parameters
    ----------
    start, stop, stride
        Frame slice to keep, applied as ``frames[start:stop:stride]``.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()

    if len(lines) < 8:
        raise ValueError("%s is too short to be an XDATCAR" % path)

    scale = float(lines[1].split()[0])
    cell = np.array([[float(t) for t in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
    symbol_names = lines[5].split()
    counts = [int(t) for t in lines[6].split()]
    if len(symbol_names) != len(counts):
        raise ValueError(
            "Could not read the element names/counts of %s (lines 6 and 7). Only "
            "VASP 5 style XDATCAR files are supported." % path)
    symbols = [s for s, n in zip(symbol_names, counts) for _ in range(n)]
    natoms = sum(counts)

    frames: list[np.ndarray] = []
    index = 7
    nlines = len(lines)
    while index < nlines:
        if not lines[index].lstrip().startswith("Direct configuration"):
            index += 1  # tolerate repeated headers between configurations
            continue
        block = " ".join(lines[index + 1:index + 1 + natoms])
        values = np.fromstring(block, sep=" ")
        if values.size != 3 * natoms:
            break  # truncated last frame, e.g. a still running MD run
        frames.append(values.reshape(natoms, 3))
        index += natoms + 1

    if not frames:
        raise ValueError("No 'Direct configuration' blocks found in %s" % path)

    positions = np.array(frames)[start:stop:stride]
    if positions.size == 0:
        raise ValueError("The frame selection start=%s stop=%s stride=%s keeps no "
                         "frames (%d available)" % (start, stop, stride, len(frames)))
    return Trajectory(cell=cell, symbols=symbols, scaled_positions=positions)


def _mean_scaled_positions(scaled: np.ndarray) -> np.ndarray:
    """Average fractional positions, unwrapped relative to the first frame."""
    diff = scaled - scaled[0]
    diff -= np.round(diff)
    return scaled[0] + diff.mean(axis=0)


def check_consistency(trajectory: Trajectory, symbols: list, cell: np.ndarray,
                      tol: float = 1.0e-3) -> None:
    """Raise/warn if a trajectory does not match the supercell used for the basis."""
    if list(trajectory.symbols) != list(symbols):
        raise ValueError(
            "The atomic order of the trajectory does not match the supercell file:\n"
            "  trajectory : %s\n  supercell  : %s"
            % (_formula(trajectory.symbols), _formula(symbols)))
    if not np.allclose(trajectory.cell, cell, atol=1.0e-2):
        import warnings
        warnings.warn(
            "The trajectory cell differs from the supercell file by more than "
            "0.01 A:\n%s\nvs\n%s" % (trajectory.cell, np.asarray(cell)), stacklevel=2)


def _formula(symbols) -> str:
    out = []
    for symbol in symbols:
        if out and out[-1][0] == symbol:
            out[-1][1] += 1
        else:
            out.append([symbol, 1])
    return " ".join("%s%d" % (s, n) for s, n in out)
