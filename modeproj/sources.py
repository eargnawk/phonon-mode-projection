"""One entry point for every supported phonon code.

:func:`load_modes` takes whatever eigenvector file the user has, reads the q points
that are commensurate with the MD supercell, and returns them in a single common
form: the ``"lattice"`` phase convention, real eigenvectors, and the atoms of the
primitive cell in the order of the user's primitive structure file.
"""

from __future__ import annotations

import os

import numpy as np

from . import evec as alamode
from . import phonopy_io
from .eigen import realify, to_lattice_convention
from .geometry import atom_permutation, reciprocal_transform

#: Supported values of the ``format`` argument.
FORMATS = ("auto", "alamode", "phonopy-fc", "phonopy-yaml", "phonopy-hdf5")

FORMAT_HELP = {
    "alamode": "ALAMODE anphon eigenvector file, PREFIX.evec (PRINTEVEC = 1)",
    "phonopy-fc": "phonopy.yaml plus force constants; phonopy computes the "
                  "eigenvectors at the required q points (needs phonopy installed)",
    "phonopy-yaml": "qpoints.yaml / mesh.yaml / band.yaml written by phonopy with "
                    "EIGENVECTORS = .TRUE.",
    "phonopy-hdf5": "qpoints.hdf5 / mesh.hdf5 written by phonopy with "
                    "EIGENVECTORS = .TRUE. (needs h5py)",
}


def detect_format(path: str) -> str:
    """Guess the format of an eigenvector file from its name and first lines."""
    name = os.path.basename(path)
    if name.endswith(".evec"):
        return "alamode"
    if name.endswith((".hdf5", ".h5")):
        return "phonopy-hdf5"
    if name in phonopy_io.YAML_NAMES:
        return "phonopy-yaml"
    if name.endswith((".yaml", ".yml")):
        if phonopy_io.looks_like_phonopy_yaml_with_force_constants(path):
            return "phonopy-fc"
        return "phonopy-yaml"
    raise ValueError(
        "Cannot tell the format of '%s' from its name. Pass an explicit format:\n%s"
        % (path, "\n".join("  %-14s %s" % (key, text)
                           for key, text in FORMAT_HELP.items())))


def load_modes(path: str, qpoints: np.ndarray, primitive, source_format: str = "auto",
               force_constants: str | None = None, force_sets: str | None = None,
               born: str | None = None, nac: bool = False, nac_direction=None,
               tol: float = 1.0e-3):
    """Read eigenvectors for ``qpoints`` from any supported phonon code.

    Parameters
    ----------
    path
        Eigenvector file (see :data:`FORMATS`).
    qpoints
        (nq, 3) fractional q points in the reciprocal basis of ``primitive``.
    primitive
        ASE ``Atoms`` of the primitive cell the user gave us; it defines both the
        q-point basis and the atom order of the returned eigenvectors.
    source_format
        One of :data:`FORMATS`; ``"auto"`` guesses from the file name.
    force_constants, force_sets, born, nac, nac_direction
        Only used by ``phonopy-fc``.

    Returns
    -------
    ModeData
        ``convention="lattice"``, real eigenvectors, atoms in the order of
        ``primitive``, ``qpoints`` as given.
    """
    qpoints = np.asarray(qpoints, dtype=float)
    if source_format in (None, "auto"):
        source_format = detect_format(path)
    if source_format not in FORMATS:
        raise ValueError("Unknown format %r; choose one of %s"
                         % (source_format, ", ".join(FORMATS)))

    if source_format == "alamode":
        header = alamode.read_header(path)
        transform = reciprocal_transform(primitive.cell[:],
                                         header["primitive_lattice"], tol=1.0e-2,
                                         what="primitive cell of the .evec file")
        data = alamode.read_eigenvectors(path, qpoints @ transform, tol=tol)

    elif source_format == "phonopy-fc":
        calculation = phonopy_io.load_phonopy(
            path, force_constants=force_constants, force_sets=force_sets,
            born=born, nac=nac)
        transform = reciprocal_transform(primitive.cell[:],
                                         np.asarray(calculation.primitive.cell),
                                         what="primitive cell of the phonopy "
                                              "calculation")
        data = phonopy_io.modes_from_phonopy(
            calculation, qpoints @ transform, nac_q_direction=nac_direction,
            source="phonopy (%s + force constants)" % os.path.basename(path))

    elif source_format in ("phonopy-yaml", "phonopy-hdf5"):
        reader = (phonopy_io.read_phonopy_yaml_modes if source_format == "phonopy-yaml"
                  else phonopy_io.read_phonopy_hdf5_modes)
        available = reader(path, qpoints=None)
        transform = (np.eye(3) if available.structure is None else
                     reciprocal_transform(primitive.cell[:], available.structure[0],
                                          what="primitive cell of %s"
                                               % os.path.basename(path)))
        indices = phonopy_io.match_qpoints(available.qpoints, qpoints @ transform,
                                           tol, os.path.basename(path))
        data = available.select_qpoints(indices)

    else:  # pragma: no cover - guarded above
        raise ValueError(source_format)

    return _to_common_form(data, qpoints, primitive, tol=tol)


def _to_common_form(data, qpoints: np.ndarray, primitive, tol: float = 1.0e-3):
    """Atom order, phase convention and real eigenvectors, in place."""
    if 3 * len(primitive) != data.nbranch:
        raise ValueError(
            "%s has %d branches, which does not match the %d atoms of your primitive "
            "cell (%d branches)." % (data.source, data.nbranch, len(primitive),
                                     3 * len(primitive)))

    if data.convention == "atom":
        positions = (np.asarray(data.structure[1]) if data.structure is not None
                     else primitive.get_scaled_positions())
        data.eigenvectors = to_lattice_convention(data.eigenvectors, data.qpoints,
                                                  positions)
        data.convention = "lattice"

    if data.structure is not None:
        lattice, positions, symbols = data.structure
        permutation = atom_permutation(primitive, lattice, positions, symbols, tol=tol)
        if not np.array_equal(permutation, np.arange(len(permutation))):
            data.permute_atoms(permutation)
            data.warnings.append(
                "The atom order of the phonon calculation differs from your primitive "
                "structure file; the eigenvectors were reordered accordingly.")
            if data.masses is not None and len(data.masses) == len(permutation):
                data.masses = np.asarray(data.masses)[permutation]

    real_eigenvectors, rotated = realify(data.eigenvectors, data.frequencies_cm)
    data.eigenvectors = real_eigenvectors
    if rotated:
        labels = ", ".join("%.3f %.3f %.3f" % tuple(data.qpoints[iq]) for iq in rotated)
        data.warnings.append(
            "Complex eigenvectors of degenerate branches were rotated to a real basis "
            "at q = %s. Amplitudes of individual branches inside a degenerate group "
            "depend on that basis; combine them as sqrt(sum of Q^2)." % labels)

    # from here on the q points are expressed in the user's primitive basis
    data.qpoints = np.asarray(qpoints, dtype=float)
    return data
