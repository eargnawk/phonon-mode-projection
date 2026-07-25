"""Primitive-cell/supercell bookkeeping needed for a mode projection.

The functions here reproduce what ALAMODE's ``displace.py --pes`` does internally
(``_find_commensurate_q`` and ``_generate_mapping_s2p``), but with plain numpy and
ASE structures so that no ALAMODE code has to be imported.
"""

from __future__ import annotations

import numpy as np

#: Labels used for the high-symmetry q points of a simple-cubic primitive cell,
#: keyed by the number of half-integer components of the folded q point.
_CUBIC_LABELS = {0: "G", 1: "X", 2: "M", 3: "R"}

#: Mathtext version of the labels above, for figure titles.
MATH_LABEL = {"G": "\\Gamma"}


def supercell_matrix(primitive_cell: np.ndarray, super_cell: np.ndarray,
                     tol: float = 1.0e-3) -> np.ndarray:
    """Integer matrix ``M`` with ``super_cell = M @ primitive_cell`` (rows = vectors)."""
    matrix = np.asarray(super_cell) @ np.linalg.inv(np.asarray(primitive_cell))
    rounded = np.round(matrix)
    if np.max(np.abs(matrix - rounded)) > tol:
        raise ValueError(
            "The supercell is not an integer multiple of the primitive cell:\n"
            "supercell = M @ primitive with M =\n%s\n"
            "Check that the two structure files describe the same lattice."
            % np.array2string(matrix, precision=4)
        )
    return rounded.astype(int)


def commensurate_qpoints(matrix: np.ndarray) -> np.ndarray:
    """q points commensurate with a supercell, in the primitive reciprocal basis.

    These are the q for which ``exp(2*pi*i*q.R) = 1`` for every supercell lattice
    translation ``R``; there are ``det(M)`` of them.  Each is folded into
    ``[-0.5, 0.5)`` and the list is sorted so that Gamma comes first.
    """
    matrix = np.asarray(matrix, dtype=int)
    ncells = int(round(abs(np.linalg.det(matrix))))
    inv = np.linalg.inv(matrix)

    # q = inv(M) @ n with integer n; scanning n over one supercell repetition of
    # the reciprocal lattice is enough to find all det(M) distinct q points.
    span = int(np.max(np.abs(matrix))) * 2 + 1
    grid = range(-span, span + 1)
    found: list[np.ndarray] = []
    for n in np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T:
        q = inv @ n
        q -= np.round(q)  # fold into [-0.5, 0.5)
        q[np.abs(q) < 1.0e-8] = 0.0
        if not any(np.linalg.norm((q - other) - np.round(q - other)) < 1.0e-6
                   for other in found):
            found.append(q)
        if len(found) == ncells:
            break

    if len(found) != ncells:  # pragma: no cover - defensive
        raise RuntimeError("Found %d of %d commensurate q points" % (len(found), ncells))

    order = sorted(range(ncells), key=lambda i: (np.linalg.norm(found[i]), tuple(found[i])))
    return np.array([found[i] for i in order])


def map_supercell_to_primitive(primitive, supercell, tol: float = 1.0e-3):
    """Map every supercell atom onto a primitive-cell atom plus a lattice point.

    Parameters
    ----------
    primitive, supercell
        ASE ``Atoms`` objects for the primitive cell and the MD supercell.

    Returns
    -------
    map_s2p : (nat_super,) int
        Index of the equivalent primitive-cell atom.
    lattice_points : (nat_super, 3) float
        Lattice translation ``R`` of each supercell atom, in primitive-cell
        fractional coordinates (integers stored as floats).
    """
    matrix = supercell_matrix(primitive.cell[:], supercell.cell[:], tol=tol)
    x_prim = primitive.get_scaled_positions()
    # supercell fractional -> primitive fractional
    x_super = supercell.get_scaled_positions() @ matrix

    nat_super = len(supercell)
    map_s2p = np.full(nat_super, -1, dtype=int)
    lattice_points = np.zeros((nat_super, 3))

    for iat in range(nat_super):
        diff = x_super[iat] - x_prim  # (nat_prim, 3)
        shift = np.round(diff)
        residual = np.linalg.norm(diff - shift, axis=1)
        jat = int(np.argmin(residual))
        if residual[jat] > tol:
            raise ValueError(
                "Supercell atom %d (%s) has no equivalent atom in the primitive "
                "cell. Are the two structures consistent (same origin, same "
                "atomic order convention)?" % (iat, supercell[iat].symbol)
            )
        if primitive[jat].symbol != supercell[iat].symbol:
            raise ValueError(
                "Supercell atom %d is %s but maps onto primitive atom %d (%s)."
                % (iat, supercell[iat].symbol, jat, primitive[jat].symbol)
            )
        map_s2p[iat] = jat
        lattice_points[iat] = shift[jat]

    return map_s2p, lattice_points


def is_cubic(cell: np.ndarray, tol: float = 1.0e-4) -> bool:
    cell = np.asarray(cell)
    lengths = np.linalg.norm(cell, axis=1)
    orthogonal = np.allclose(cell @ cell.T - np.diag(lengths ** 2), 0.0,
                             atol=tol * lengths.max() ** 2)
    return bool(orthogonal and np.ptp(lengths) < tol * lengths.max())


def label_qpoints(qpoints: np.ndarray, primitive_cell: np.ndarray) -> list[str]:
    """Guess high-symmetry labels (G, X, M, R) for commensurate q points.

    Only meaningful for a simple-cubic primitive cell with q components in
    ``{0, +-1/2}``; anything else gets a positional name (``q3``, ...).  Labels are
    made unique by appending a star index (``X``, ``X2``, ``X3``).
    """
    qpoints = np.asarray(qpoints)
    cubic = is_cubic(primitive_cell)
    labels: list[str] = []
    counts: dict[str, int] = {}

    for iq, q in enumerate(qpoints):
        twice = 2.0 * q
        base = None
        if cubic and np.allclose(twice, np.round(twice), atol=1.0e-6):
            nhalf = int(np.sum(np.round(twice).astype(int) % 2 != 0))
            base = _CUBIC_LABELS.get(nhalf)
        if base is None:
            labels.append("q%d" % iq)
            continue
        counts[base] = counts.get(base, 0) + 1
        labels.append(base if counts[base] == 1 else "%s%d" % (base, counts[base]))

    return labels
