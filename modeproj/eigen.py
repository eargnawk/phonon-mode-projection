"""Bringing eigenvectors of different codes onto a common footing.

Two things have to happen before eigenvectors from an arbitrary phonon code can be
used for a real-valued mode projection:

1. **Phase convention.** phonopy carries the basis positions ``tau`` in the phase of
   its dynamical matrix, ALAMODE does not (see :data:`modeproj.modedata.CONVENTIONS`).
   :func:`to_lattice_convention` removes the ``tau`` phase so that everything
   downstream works in the lattice convention.
2. **Global phase.** In the lattice convention the dynamical matrix is real symmetric
   at every q with ``q == -q`` (all components 0 or +-1/2), so a real eigenvector
   basis exists.  A diagonalisation done in another convention can still hand back
   complex vectors, whose real part is meaningless on its own (multiplying an
   eigenvector by ``i`` would silently give zero amplitude).
   :func:`realify` rotates each degenerate group back to a real basis.
"""

from __future__ import annotations

import numpy as np


def to_lattice_convention(eigenvectors: np.ndarray, qpoints: np.ndarray,
                          scaled_positions: np.ndarray) -> np.ndarray:
    """Convert eigenvectors from the ``"atom"`` to the ``"lattice"`` convention.

    ``e_lattice[a] = e_atom[a] * exp(2 pi i q.tau_a)``, which leaves the physical
    displacement pattern ``e_a exp(2 pi i q.(R + tau_a))`` unchanged.

    Parameters
    ----------
    eigenvectors
        (nq, nbranch, nbranch) complex, atom-major component order.
    qpoints
        (nq, 3) fractional q points.
    scaled_positions
        (natom, 3) fractional coordinates of the primitive-cell atoms.
    """
    eigenvectors = np.asarray(eigenvectors)
    nq, nbranch = eigenvectors.shape[:2]
    natom = nbranch // 3
    phases = np.exp(2j * np.pi * (np.asarray(qpoints) @ np.asarray(scaled_positions).T))
    shaped = eigenvectors.reshape(nq, nbranch, natom, 3)
    return (shaped * phases[:, None, :, None]).reshape(nq, nbranch, nbranch)


def realify(eigenvectors: np.ndarray, frequencies: np.ndarray,
            tol_imaginary: float = 1.0e-6, tol_degenerate: float = 1.0e-3):
    """Return an equivalent set of real eigenvectors, one q point at a time.

    Vectors that are already real (up to a global phase) are only cleaned up; a
    degenerate group whose vectors are genuinely complex mixtures is rotated to a
    real orthonormal basis spanning the same subspace.  Amplitudes of individual
    modes inside a degenerate group are basis dependent either way - the invariant
    quantity is ``sqrt(sum of Q^2)`` over the group.

    Parameters
    ----------
    eigenvectors
        (nq, nbranch, nbranch) complex, eigenvectors along the second axis.
    frequencies
        (nq, nbranch) frequencies, used to detect degenerate groups.
    tol_imaginary
        Largest imaginary component that is treated as numerical noise.
    tol_degenerate
        Frequency difference (same unit as ``frequencies``) below which two branches
        count as degenerate.

    Returns
    -------
    (real_eigenvectors, rotated_qpoints)
        ``real_eigenvectors`` has the same shape but a real dtype;
        ``rotated_qpoints`` lists the q-point indices where a degenerate group had to
        be rotated, i.e. where individual amplitudes are not comparable with another
        code's.
    """
    eigenvectors = np.asarray(eigenvectors, dtype=np.complex128)
    frequencies = np.asarray(frequencies, dtype=float)
    nq, nbranch = frequencies.shape
    out = np.zeros((nq, nbranch, nbranch))
    rotated: list[int] = []

    for iq in range(nq):
        vectors = eigenvectors[iq].copy()
        if np.max(np.abs(vectors.imag)) < tol_imaginary:
            out[iq] = vectors.real
            continue

        for group in _degenerate_groups(frequencies[iq], tol_degenerate):
            block = vectors[group]
            # remove the arbitrary global phase of each vector
            for row in range(len(block)):
                pivot = np.argmax(np.abs(block[row]))
                block[row] *= np.exp(-1j * np.angle(block[row][pivot]))
            if np.max(np.abs(block.imag)) < tol_imaginary:
                out[iq, group] = block.real
                continue
            out[iq, group] = _real_basis_of_span(block)
            rotated.append(iq)

    return out, sorted(set(rotated))


def _degenerate_groups(frequencies: np.ndarray, tol: float):
    """Indices of branches grouped by (nearly) equal frequency."""
    order = np.argsort(frequencies, kind="stable")
    groups: list[list[int]] = []
    for index in order:
        if groups and abs(frequencies[index] - frequencies[groups[-1][0]]) < tol:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return [np.array(group) for group in groups]


def _real_basis_of_span(block: np.ndarray) -> np.ndarray:
    """Real orthonormal basis of the space spanned by complex vectors.

    The eigenspace of a real symmetric matrix is closed under complex conjugation, so
    the real and imaginary parts of the vectors span the very same space and an SVD
    of ``[Re; Im]`` recovers a real basis of it.
    """
    count = len(block)
    candidates = np.vstack([block.real, block.imag])
    _, singular, right = np.linalg.svd(candidates, full_matrices=False)
    basis = right[:count]

    if singular[count - 1] < 1.0e-8 * max(singular[0], 1.0e-30):
        raise ValueError(
            "Could not build a real basis for a degenerate group of %d modes; the "
            "eigenvectors do not span a real subspace. This should not happen for a "
            "q point with q == -q." % count)

    projector_before = block.conj().T @ block
    projector_after = basis.T @ basis
    if not np.allclose(projector_before, projector_after, atol=1.0e-6):
        raise ValueError(
            "Real-basis construction changed the subspace of a degenerate group; "
            "please report this together with the input files.")
    return basis
