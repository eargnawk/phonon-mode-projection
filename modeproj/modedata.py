"""Container for phonon eigenvectors, independent of which code produced them."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: 1 Rydberg expressed in cm^-1 (ALAMODE stores omega^2 in Rydberg atomic units).
RY_TO_CM = 109737.31568

#: 1 THz expressed in cm^-1 (phonopy reports frequencies in THz).
THZ_TO_CM = 33.35641

BOHR_TO_ANGSTROM = 0.5291772108

#: Phase convention of the eigenvectors.
#:
#: ``"lattice"``
#:     ``D_ab(q) = sum_l Phi_ab(l) exp(2 pi i q.R_l) / sqrt(m_a m_b)``, so the
#:     displacement pattern of a mode is ``e_a exp(2 pi i q.R_l)``. Used by ALAMODE
#:     and internally by this package.
#: ``"atom"``
#:     the phase also contains the basis positions, ``exp(2 pi i q.(R_l + tau_b -
#:     tau_a))``, so the displacement pattern is ``e_a exp(2 pi i q.(R_l + tau_a))``.
#:     Used by phonopy.
CONVENTIONS = ("lattice", "atom")


@dataclass
class ModeData:
    """Phonon frequencies and eigenvectors at a set of q points.

    Eigenvectors are stored as ``eigenvectors[iq, ibranch, iatom * 3 + idir]`` with
    ``iatom`` running over the atoms of the primitive cell *in the order of the
    primitive structure file given by the user*.
    """

    qpoints: np.ndarray  #: (nq, 3) fractional coordinates, primitive reciprocal basis
    frequencies_cm: np.ndarray  #: (nq, nbranch) in cm^-1, negative = imaginary
    eigenvectors: np.ndarray  #: (nq, nbranch, nbranch) complex
    convention: str = "lattice"  #: see :data:`CONVENTIONS`
    source: str = ""  #: short description of where the data came from, for messages
    masses: np.ndarray | None = None  #: (nkind,) or (natom,) masses in amu, if known
    primitive_lattice: np.ndarray | None = None  #: (3, 3) in Angstrom, rows
    #: ``(lattice, scaled_positions, symbols)`` of the primitive cell the source used,
    #: when the file provides it.  Needed to convert the ``"atom"`` convention and to
    #: reorder the atoms into the order of the user's primitive cell.
    structure: tuple | None = None
    warnings: list = field(default_factory=list)  #: messages to show the user

    def __post_init__(self):
        if self.convention not in CONVENTIONS:
            raise ValueError("convention must be one of %s, got %r"
                             % (CONVENTIONS, self.convention))
        self.qpoints = np.asarray(self.qpoints, dtype=float)
        self.frequencies_cm = np.asarray(self.frequencies_cm, dtype=float)
        self.eigenvectors = np.asarray(self.eigenvectors, dtype=np.complex128)
        nq, nbranch = self.frequencies_cm.shape
        if self.eigenvectors.shape != (nq, nbranch, nbranch):
            raise ValueError(
                "Inconsistent shapes in %s: %d q points, %d branches, but "
                "eigenvectors have shape %s"
                % (self.source or "mode data", nq, nbranch,
                   (self.eigenvectors.shape,)))

    @property
    def nbranch(self) -> int:
        return self.frequencies_cm.shape[1]

    def select_qpoints(self, indices) -> "ModeData":
        """A copy holding only the q points ``indices``, in that order."""
        indices = np.asarray(indices, dtype=int)
        return ModeData(
            qpoints=self.qpoints[indices],
            frequencies_cm=self.frequencies_cm[indices],
            eigenvectors=self.eigenvectors[indices],
            convention=self.convention,
            source=self.source,
            masses=self.masses,
            primitive_lattice=self.primitive_lattice,
            structure=self.structure,
            warnings=list(self.warnings),
        )

    def permute_atoms(self, permutation) -> None:
        """Reorder the primitive-cell atoms of the eigenvectors in place.

        ``permutation[i]`` is the index, in the source data, of the atom that comes
        ``i``-th in the user's primitive cell.
        """
        permutation = np.asarray(permutation, dtype=int)
        nq, nbranch = self.frequencies_cm.shape
        shaped = self.eigenvectors.reshape(nq, nbranch, nbranch // 3, 3)
        self.eigenvectors = shaped[:, :, permutation, :].reshape(nq, nbranch, nbranch)
