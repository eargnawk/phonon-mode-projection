"""Reader for ALAMODE (``anphon``) eigenvector files, i.e. ``PREFIX.evec``.

A ``.evec`` file written on a dense q-point mesh is easily hundreds of MB, while a
mode projection only needs the handful of q points that are commensurate with the
molecular-dynamics supercell.  :func:`read_eigenvectors` therefore parses only the
requested q points and skips the rest of the file, which makes reading a
20x20x20-mesh file a few seconds instead of a minute.
"""

from __future__ import annotations

import collections
import itertools
import os

import numpy as np

from .modedata import BOHR_TO_ANGSTROM, RY_TO_CM, ModeData


class EvecFormatError(RuntimeError):
    """Raised when a file does not look like an ALAMODE ``.evec`` file."""


def _parse_header(fh) -> tuple[int, int, np.ndarray, np.ndarray]:
    lattice_rows: list[list[float]] = []
    nmode = nq = None
    masses = None
    in_lattice = False

    for line in fh:
        stripped = line.strip()
        if stripped.startswith("# Lattice vectors of the primitive cell"):
            in_lattice = True
            continue
        if stripped.startswith("#"):
            in_lattice = False
            if "Number of phonon modes" in stripped:
                nmode = int(stripped.split(":")[1])
            elif "Number of k points" in stripped:
                nq = int(stripped.split(":")[1])
            elif "Atomic masses" in stripped:
                masses = np.array([float(t) for t in stripped.split(":")[1].split()])
            elif "Eigenvalues and eigenvectors" in stripped:
                break
            continue
        if in_lattice and stripped and len(lattice_rows) < 3:
            lattice_rows.append([float(t) for t in stripped.split()])

    if nmode is None or nq is None or masses is None:
        raise EvecFormatError(
            "Could not read the header of the .evec file. Expected the lines "
            "'# Number of phonon modes:', '# Number of k points:' and "
            "'# Atomic masses:' written by ALAMODE's anphon (PRINTEVEC = 1)."
        )
    lattice = np.array(lattice_rows) if len(lattice_rows) == 3 else np.eye(3)
    return nmode, nq, masses, lattice


def read_header(path: str) -> dict:
    """Header information of an ALAMODE ``.evec`` file, without parsing the modes.

    Returns a dict with ``nbranch``, ``nqpoint``, ``masses`` (amu) and
    ``primitive_lattice`` (Angstrom, rows are vectors).
    """
    with open(path) as fh:
        nmode, nq, masses, lattice = _parse_header(fh)
    return {"nbranch": nmode, "nqpoint": nq, "masses": masses,
            "primitive_lattice": lattice * BOHR_TO_ANGSTROM}


def _skip_lines(fh, count: int) -> None:
    collections.deque(itertools.islice(fh, count), maxlen=0)


def _read_qpoint_line(line: str) -> np.ndarray:
    return np.array([float(t) for t in line.split(":")[1].split()])


def _folded_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    diff = q1 - q2
    diff -= np.round(diff)
    return float(np.linalg.norm(diff))


def read_eigenvectors(
    path: str,
    qpoints: np.ndarray | None = None,
    tol: float = 1.0e-3,
) -> EvecData:
    """Read an ALAMODE ``.evec`` file.

    Parameters
    ----------
    path
        Path to ``PREFIX.evec`` (written by ``anphon`` with ``PRINTEVEC = 1``).
    qpoints
        (n, 3) fractional q points to extract, in the primitive reciprocal basis.
        Blocks for all other q points in the file are skipped without being
        parsed.  ``None`` reads the whole file, which can be slow and memory
        hungry for a dense mesh.
    tol
        Tolerance used when matching q points (modulo a reciprocal lattice
        vector).

    Returns
    -------
    ModeData
        Arrays ordered like ``qpoints`` (or like the file if ``qpoints`` is None), in
        the ``"lattice"`` phase convention.
    """
    wanted = None if qpoints is None else np.asarray(qpoints, dtype=float).reshape(-1, 3)

    with open(path) as fh:
        nmode, nq_file, masses, lattice = _parse_header(fh)
        lines_per_block = nmode * (nmode + 2) + 1  # after the '## kpoint' line

        n_target = nq_file if wanted is None else len(wanted)
        q_found = np.zeros((n_target, 3))
        omega2 = np.full((n_target, nmode), np.nan)
        evec = np.zeros((n_target, nmode, nmode), dtype=np.complex128)
        filled = np.zeros(n_target, dtype=bool)

        line = fh.readline()
        while line:
            if not line.lstrip().startswith("## kpoint"):
                line = fh.readline()
                continue

            q = _read_qpoint_line(line)
            if wanted is None:
                targets = [int(np.count_nonzero(filled))]
            else:
                targets = [
                    i
                    for i in range(len(wanted))
                    if not filled[i] and _folded_distance(q, wanted[i]) < tol
                ]

            if not targets:
                _skip_lines(fh, lines_per_block)
                line = fh.readline()
                continue

            block_omega2 = np.empty(nmode)
            block_evec = np.empty((nmode, nmode), dtype=np.complex128)
            for imode in range(nmode):
                block_omega2[imode] = float(fh.readline().split(":")[1])
                for jmode in range(nmode):
                    real, imag = fh.readline().split()[:2]
                    block_evec[imode, jmode] = complex(float(real), float(imag))
                fh.readline()  # blank line after each mode
            fh.readline()  # blank line after each q point

            for i in targets:
                q_found[i] = q
                omega2[i] = block_omega2
                evec[i] = block_evec
                filled[i] = True

            if filled.all():
                break
            line = fh.readline()

    if not filled.all():
        missing = np.asarray(wanted)[~filled] if wanted is not None else []
        raise EvecFormatError(
            "The following q points were not found in %s:\n%s\n"
            "Make sure the anphon mesh contains the q points commensurate with "
            "the supercell (e.g. an even mesh for a 2x2x2 supercell)."
            % (path, "\n".join("  %8.4f %8.4f %8.4f" % tuple(q) for q in missing))
        )

    frequencies_cm = np.sign(omega2) * np.sqrt(np.abs(omega2)) * RY_TO_CM
    return ModeData(
        qpoints=q_found,
        frequencies_cm=frequencies_cm,
        eigenvectors=evec,
        convention="lattice",
        source="ALAMODE eigenvector file %s" % os.path.basename(path),
        masses=masses,
        primitive_lattice=lattice * BOHR_TO_ANGSTROM,
    )
