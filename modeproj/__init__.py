"""Phonon normal-mode projection of ab initio molecular dynamics trajectories.

Typical use::

    from modeproj import ModeBasis, read_xdatcar, plot_amplitudes

    basis = ModeBasis.from_files("kno_221.mesh.evec", "PPOSCAR", "SPOSCAR")
    print(basis.qpoint_table())            # which q points and branches exist

    traj = read_xdatcar("XDATCAR")
    modes = [basis.select(spec) for spec in ("G:1", "M:2", "R:1")]
    Q = basis.project_modes(traj.displacements(), modes)

See ``python -m modeproj --help`` for the command line interface.
"""

from .basis import ModeBasis, ModeSelection
from .evec import EvecData, read_eigenvectors
from .geometry import commensurate_qpoints, label_qpoints, supercell_matrix
from .plot import plot_amplitudes, plot_histograms, rolling_mean
from .trajectory import Trajectory, read_xdatcar

__all__ = [
    "EvecData",
    "ModeBasis",
    "ModeSelection",
    "Trajectory",
    "commensurate_qpoints",
    "label_qpoints",
    "plot_amplitudes",
    "plot_histograms",
    "read_eigenvectors",
    "read_xdatcar",
    "rolling_mean",
    "supercell_matrix",
]

__version__ = "1.0.0"
