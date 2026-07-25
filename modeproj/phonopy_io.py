"""Readers for phonopy eigenvectors.

Three ways of getting phonon eigenvectors out of phonopy are supported:

``phonopy.yaml`` + force constants (``phonopy-fc``)
    The recommended route: phonopy is asked for the eigenvectors exactly at the q
    points commensurate with the MD supercell, so no mesh has to match anything and
    no large text file is needed.  Requires phonopy to be importable.

``qpoints.yaml`` / ``mesh.yaml`` / ``band.yaml`` (``phonopy-yaml``)
    Text output of ``phonopy --qpoints``/``--mesh``/``--band`` run with
    ``EIGENVECTORS = .TRUE.``.  No phonopy installation needed.

``qpoints.hdf5`` / ``mesh.hdf5`` (``phonopy-hdf5``)
    The same data in HDF5, which is much faster to read for dense meshes.  Requires
    ``h5py``.  These files carry no structure information, so the primitive cell in
    the file is assumed to be the one given on the command line.

phonopy keeps the basis positions in the phase of its dynamical matrix, so its
eigenvectors come back in the ``"atom"`` convention (see
:data:`modeproj.modedata.CONVENTIONS`); the conversion happens in
:mod:`modeproj.eigen`.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from .modedata import THZ_TO_CM, ModeData

#: Names of phonopy output files that contain eigenvectors.
YAML_NAMES = ("qpoints.yaml", "mesh.yaml", "band.yaml")
HDF5_NAMES = ("qpoints.hdf5", "mesh.hdf5")


class PhonopyFormatError(RuntimeError):
    """Raised when a phonopy output file cannot be used for a projection."""


# --------------------------------------------------------------------- utilities


def _load_yaml(path: str):
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise PhonopyFormatError(
            "Reading %s needs PyYAML (pip install PyYAML)." % path) from error
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path) as handle:
        return yaml.load(handle, Loader=loader)


def match_qpoints(available: np.ndarray, wanted: np.ndarray, tol: float,
                  source: str) -> np.ndarray:
    """Index in ``available`` of each wanted q point (modulo a lattice vector)."""
    indices = np.empty(len(wanted), dtype=int)
    missing = []
    for i, q in enumerate(wanted):
        diff = available - q
        distance = np.linalg.norm(diff - np.round(diff), axis=1)
        best = int(np.argmin(distance))
        if distance[best] > tol:
            missing.append(q)
        indices[i] = best
    if missing:
        raise PhonopyFormatError(
            "%s does not contain the q points commensurate with the supercell:\n%s\n"
            "Use a Gamma-centred mesh that contains them (phonopy: --gc / "
            "GAMMA_CENTER = .TRUE., e.g. --mesh 2 2 2 for a 2x2x2 supercell), or list "
            "them with 'python -m modeproj anphon-input' and hand them to phonopy via "
            "--qpoints / QPOINTS."
            % (source, "\n".join("  %8.4f %8.4f %8.4f" % tuple(q) for q in missing)))
    return indices


# ------------------------------------------------------------------- yaml output


def read_phonopy_yaml_modes(path: str, qpoints: np.ndarray | None = None,
                            tol: float = 1.0e-3) -> ModeData:
    """Read eigenvectors from ``qpoints.yaml``, ``mesh.yaml`` or ``band.yaml``."""
    data = _load_yaml(path)
    if not isinstance(data, dict) or "phonon" not in data:
        raise PhonopyFormatError(
            "%s has no 'phonon' section; it does not look like the output of "
            "phonopy --qpoints/--mesh/--band." % path)

    entries = data["phonon"]
    available = np.array([entry["q-position"] for entry in entries], dtype=float)
    if qpoints is None:
        selection = np.arange(len(entries))
    else:
        selection = match_qpoints(available, np.asarray(qpoints), tol,
                                   os.path.basename(path))

    frequencies = []
    eigenvectors = []
    for index in selection:
        bands = entries[int(index)]["band"]
        if "eigenvector" not in bands[0]:
            raise PhonopyFormatError(
                "%s contains frequencies but no eigenvectors. Re-run phonopy with "
                "EIGENVECTORS = .TRUE. (or --eigenvectors)." % path)
        frequencies.append([band["frequency"] for band in bands])
        eigenvectors.append([
            [complex(component[0], component[1])
             for atom in band["eigenvector"] for component in atom]
            for band in bands])

    lattice = None
    if "lattice" in data:
        lattice = np.array(data["lattice"], dtype=float)
    elif "reciprocal_lattice" in data:
        # qpoints.yaml only stores the reciprocal vectors (rows a*, b*, c*)
        lattice = np.linalg.inv(np.array(data["reciprocal_lattice"], dtype=float).T)

    structure = masses = None
    if "points" in data and lattice is not None:
        points = data["points"]
        structure = (lattice,
                     np.array([point["coordinates"] for point in points], dtype=float),
                     [point["symbol"] for point in points])
        masses = np.array([point.get("mass", np.nan) for point in points])

    result = ModeData(
        qpoints=available[selection],
        frequencies_cm=np.array(frequencies) * THZ_TO_CM,
        eigenvectors=np.array(eigenvectors),
        convention="atom",
        source="phonopy output %s" % os.path.basename(path),
        masses=masses,
        primitive_lattice=lattice,
        structure=structure,
    )
    if structure is None:
        result.warnings.append(
            "%s does not list the atoms of the primitive cell (mesh.yaml does), so "
            "your primitive structure file is assumed to have the same atoms, in the "
            "same order and with the same origin, as the phonopy calculation."
            % os.path.basename(path))
    return result


# ------------------------------------------------------------------- hdf5 output


def read_phonopy_hdf5_modes(path: str, qpoints: np.ndarray | None = None,
                            tol: float = 1.0e-3) -> ModeData:
    """Read eigenvectors from ``qpoints.hdf5`` or ``mesh.hdf5``."""
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise PhonopyFormatError(
            "Reading %s needs h5py (pip install h5py)." % path) from error

    with h5py.File(path, "r") as handle:
        if "eigenvector" not in handle:
            raise PhonopyFormatError(
                "%s has no 'eigenvector' dataset. Re-run phonopy with "
                "EIGENVECTORS = .TRUE. (or --eigenvectors)." % path)
        if "qpoint" not in handle:
            raise PhonopyFormatError(
                "%s has no 'qpoint' dataset, so its q points cannot be identified. "
                "Use qpoints.yaml/mesh.yaml or the phonopy.yaml + force-constants "
                "route instead." % path)
        available = np.array(handle["qpoint"])
        selection = (np.arange(len(available)) if qpoints is None else
                     match_qpoints(available, np.asarray(qpoints), tol,
                                    os.path.basename(path)))
        frequencies = np.array(handle["frequency"])[selection]
        # phonopy stores eigenvectors as columns of a (nq, nband, nband) array
        eigenvectors = np.array(handle["eigenvector"])[selection]

    warnings.warn(
        "%s carries no structure information: the primitive cell is assumed to have "
        "the same atomic order as your primitive structure file. Prefer "
        "qpoints.yaml/mesh.yaml or phonopy.yaml + force constants if you are unsure."
        % os.path.basename(path), stacklevel=2)

    return ModeData(
        qpoints=available[selection],
        frequencies_cm=frequencies * THZ_TO_CM,
        eigenvectors=np.swapaxes(eigenvectors, 1, 2),
        convention="atom",
        source="phonopy output %s" % os.path.basename(path),
    )


# ----------------------------------------------------- phonopy.yaml + force constants


def load_phonopy(path: str, force_constants: str | None = None,
                 force_sets: str | None = None, born: str | None = None,
                 nac: bool = False):
    """Load a phonopy calculation from ``phonopy.yaml`` plus force constants.

    Parameters
    ----------
    path
        ``phonopy.yaml`` or ``phonopy_disp.yaml``.  If it already contains the force
        constants, nothing else is needed.
    force_constants, force_sets
        ``FORCE_CONSTANTS``/``force_constants.hdf5`` or ``FORCE_SETS``, when the yaml
        file does not carry them.
    born, nac
        Non-analytic term correction: ``nac=True`` switches it on, using ``BORN`` next
        to the yaml file or the file given by ``born``.
    """
    try:
        import phonopy
    except ImportError as error:
        raise PhonopyFormatError(
            "Using %s needs phonopy (pip install phonopy). Alternatively run phonopy "
            "yourself and pass its qpoints.yaml / mesh.hdf5 to modeproj." % path
        ) from error

    calculation = phonopy.load(phonopy_yaml=path,
                               force_constants_filename=force_constants,
                               force_sets_filename=force_sets,
                               born_filename=born, is_nac=nac, log_level=0)
    if calculation.force_constants is None:
        raise PhonopyFormatError(
            "%s does not provide force constants. Pass --force-constants "
            "FORCE_CONSTANTS or --force-sets FORCE_SETS." % path)
    return calculation


def modes_from_phonopy(calculation, qpoints: np.ndarray, nac_q_direction=None,
                       source: str = "phonopy") -> ModeData:
    """Eigenvectors of a loaded phonopy calculation at exactly the given q points.

    ``qpoints`` must be fractional coordinates in the reciprocal basis of phonopy's
    own primitive cell.  ``nac_q_direction`` is the direction along which q approaches
    zero for the LO-TO splitting at Gamma; ``None`` evaluates Gamma without the
    non-analytic term.
    """
    calculation.run_qpoints(np.asarray(qpoints, dtype=float), with_eigenvectors=True,
                            nac_q_direction=nac_q_direction)
    result = calculation.get_qpoints_dict()
    primitive = calculation.primitive

    data = ModeData(
        qpoints=np.asarray(qpoints, dtype=float),
        frequencies_cm=np.asarray(result["frequencies"]) * THZ_TO_CM,
        # phonopy returns the eigenvectors of each q point as columns
        eigenvectors=np.swapaxes(np.asarray(result["eigenvectors"]), 1, 2),
        convention="atom",
        source=source,
        masses=np.asarray(primitive.masses),
        primitive_lattice=np.asarray(primitive.cell),
        structure=(np.asarray(primitive.cell), np.asarray(primitive.scaled_positions),
                   [str(symbol) for symbol in primitive.symbols]),
    )
    if calculation.nac_params is not None and nac_q_direction is None:
        data.warnings.append(
            "The non-analytic term correction is on, but no direction was given for "
            "Gamma (--nac-direction), so Gamma is evaluated without it.")
    return data


# ------------------------------------------------------------------------ helpers


def looks_like_phonopy_yaml_with_force_constants(path: str) -> bool:
    """True for ``phonopy.yaml``/``phonopy_disp.yaml`` style files."""
    name = os.path.basename(path)
    if name in YAML_NAMES:
        return False
    if not name.endswith((".yaml", ".yml")):
        return False
    try:
        with open(path) as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return "phonopy:" in head or "supercell_matrix:" in head
