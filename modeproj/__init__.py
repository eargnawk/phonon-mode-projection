"""Phonon normal-mode projection of ab initio molecular dynamics trajectories.

Typical use::

    from modeproj import ModeBasis, read_xdatcar, plot_amplitudes

    basis = ModeBasis.from_files("kno_221.mesh.evec", "PPOSCAR", "SPOSCAR")
    print(basis.qpoint_table())            # which q points and branches exist

    traj = read_xdatcar("XDATCAR")
    modes = [basis.select(spec) for spec in ("G:1", "M:2", "R:1")]
    Q = basis.project_modes(traj.displacements(), modes)

Phonons may come from ALAMODE (``PREFIX.evec``) or phonopy (``qpoints.yaml``,
``mesh.yaml``, ``band.yaml``, ``qpoints.hdf5``, ``mesh.hdf5``, or ``phonopy.yaml``
plus force constants)::

    basis = ModeBasis.from_files("phonopy.yaml", "PPOSCAR", "SPOSCAR",
                                 force_constants="FORCE_CONSTANTS")

See ``python -m modeproj --help`` for the command line interface.
"""

from .basis import ModeBasis, ModeSelection
from .evec import read_eigenvectors
from .geometry import commensurate_qpoints, label_qpoints, supercell_matrix
from .modedata import ModeData
from .phonopy_io import (
    read_phonopy_hdf5_modes,
    read_phonopy_yaml_modes,
)
from .plot import plot_amplitudes, plot_histograms, rolling_mean
from .sources import FORMATS, detect_format, load_modes
from .trajectory import Trajectory, read_xdatcar

__all__ = [
    "FORMATS",
    "ModeBasis",
    "ModeData",
    "ModeSelection",
    "Trajectory",
    "commensurate_qpoints",
    "detect_format",
    "label_qpoints",
    "load_modes",
    "plot_amplitudes",
    "plot_histograms",
    "read_eigenvectors",
    "read_phonopy_hdf5_modes",
    "read_phonopy_yaml_modes",
    "read_xdatcar",
    "rolling_mean",
    "supercell_matrix",
]

__version__ = "1.0.0"
