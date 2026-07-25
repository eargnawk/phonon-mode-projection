"""Projection basis: phonon modes of the supercell, ready to project onto."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
from ase.io import read as ase_read

from .geometry import (
    MATH_LABEL,
    commensurate_qpoints,
    label_qpoints,
    map_supercell_to_primitive,
    supercell_matrix,
)
from .sources import detect_format, load_modes

CACHE_VERSION = 2


@dataclass
class ModeSelection:
    """One selected phonon mode."""

    iq: int  #: index into :attr:`ModeBasis.qpoints`
    imode: int  #: branch index, 0-based (branch 1 of the user interface)
    label: str  #: q-point label, e.g. ``"M"``
    frequency_cm: float  #: frequency in cm^-1, negative when unstable

    @property
    def name(self) -> str:
        """Short name used for output columns, e.g. ``"Q_M_2"``."""
        return "Q_%s_%d" % (self.label, self.imode + 1)

    @property
    def pretty(self) -> str:
        """Label for plots, e.g. ``"$Q_{\\mathrm{M}}$ (branch 2, 154 cm$^{-1}$)"``."""
        symbol = MATH_LABEL.get(self.label, "\\mathrm{%s}" % self.label)
        frequency = ("%.0f cm$^{-1}$" % self.frequency_cm if self.frequency_cm >= 0
                     else "%.0f$i$ cm$^{-1}$" % abs(self.frequency_cm))
        return "$Q_{%s}$ (branch %d, %s)" % (symbol, self.imode + 1, frequency)


class ModeBasis:
    """Phonon eigenvectors of a supercell, arranged for fast mode projection.

    Eigenvectors from any supported code (ALAMODE, phonopy) are stored in the same
    ``"lattice"`` phase convention, so the projection follows the convention of
    ALAMODE's ``displace.py --pes`` no matter where the phonons came from:

    .. math::

        Q_{q\\nu} = \\sum_{a} \\sqrt{m_a}\\, \\mathbf{u}_a \\cdot
                    \\mathrm{Re}\\left[ \\mathbf{e}_{q\\nu}^{p(a)}
                    e^{2\\pi i \\mathbf{q}\\cdot\\mathbf{R}_a} \\right],

    where the sum runs over all atoms ``a`` of the supercell, ``u_a`` is the
    displacement from the reference structure in Angstrom, ``m_a`` the mass in amu,
    ``p(a)`` the equivalent atom in the primitive cell and ``R_a`` the lattice point
    the atom belongs to.  ``Q`` therefore comes out in amu^(1/2) Angstrom and grows
    with the supercell size unless ``normalize=True`` is used.
    """

    def __init__(self, qpoints, labels, frequencies_cm, projectors, mass_sqrt, symbols,
                 primitive_cell, supercell_matrix_, ncells, source=""):
        self.qpoints = np.asarray(qpoints)  #: (nq, 3) fractional q points
        self.labels = list(labels)  #: (nq,) q-point labels
        #: (nq, nmode) frequencies in cm^-1; negative values are imaginary modes
        self.frequencies_cm = np.asarray(frequencies_cm)
        self.projectors = np.asarray(projectors)  #: (nq, nmode, nat, 3)
        self.mass_sqrt = np.asarray(mass_sqrt)  #: (nat, 3) sqrt(mass) in amu^(1/2)
        self.symbols = list(symbols)
        self.primitive_cell = np.asarray(primitive_cell)
        self.supercell_matrix = np.asarray(supercell_matrix_)
        self.ncells = int(ncells)
        self.source = str(source)  #: where the eigenvectors came from

    # ------------------------------------------------------------------ build

    @classmethod
    def from_files(cls, file_evec: str, file_primitive: str, file_supercell: str,
                   source_format: str = "auto", cache: str | None = None,
                   verbose: bool = True, **source_options) -> "ModeBasis":
        """Build a basis from a phonon calculation and two VASP structure files.

        Parameters
        ----------
        file_evec
            Eigenvectors of the phonon calculation. Supported are ALAMODE
            ``PREFIX.evec`` files and phonopy output (``qpoints.yaml``, ``mesh.yaml``,
            ``band.yaml``, ``qpoints.hdf5``, ``mesh.hdf5``) as well as
            ``phonopy.yaml`` together with force constants; see
            :data:`modeproj.sources.FORMATS`.
        file_primitive, file_supercell
            POSCAR-format primitive cell and MD supercell (e.g. ``PPOSCAR`` and
            ``SPOSCAR`` from phonopy). The primitive cell fixes both the q-point basis
            and the atom order of the eigenvectors.
        source_format
            ``"auto"`` (default) guesses the format from the file name.
        cache
            Path of an ``.npz`` cache.  It is used when it is newer than the input
            files and written otherwise.  ``None`` disables caching.
        source_options
            Passed on to :func:`modeproj.sources.load_modes`, e.g.
            ``force_constants=``, ``force_sets=``, ``born=``, ``nac=``,
            ``nac_direction=`` for the phonopy force-constants route.
        """
        sources = [file_evec, file_primitive, file_supercell]
        sources += [value for key, value in sorted(source_options.items())
                    if key in ("force_constants", "force_sets", "born")
                    and isinstance(value, str)]
        if cache and _cache_is_fresh(cache, sources):
            if verbose:
                print("Reading projection basis from cache %s" % cache)
            return cls.load(cache)

        primitive = ase_read(file_primitive, format="vasp")
        supercell = ase_read(file_supercell, format="vasp")

        matrix = supercell_matrix(primitive.cell[:], supercell.cell[:])
        ncells = int(round(abs(np.linalg.det(matrix))))
        if ncells * len(primitive) != len(supercell):
            raise ValueError(
                "The supercell contains %d atoms but %d x %d = %d are expected from "
                "the lattice vectors." % (len(supercell), ncells, len(primitive),
                                          ncells * len(primitive)))

        qpoints = commensurate_qpoints(matrix)
        labels = label_qpoints(qpoints, primitive.cell[:])
        map_s2p, lattice_points = map_supercell_to_primitive(primitive, supercell)

        if source_format in (None, "auto"):
            source_format = detect_format(file_evec)
        if verbose:
            print("Supercell matrix : %s"
                  % np.array2string(matrix).replace("\n", " "))
            print("Reading %d commensurate q points from %s (%s) ..."
                  % (len(qpoints), os.path.basename(file_evec), source_format))

        data = load_modes(file_evec, qpoints, primitive,
                          source_format=source_format, **source_options)
        _check_masses(data.masses, primitive, data.source)
        for message in data.warnings:
            warnings.warn(message, stacklevel=2)

        nq, nmode = data.frequencies_cm.shape
        # (nq, nmode, nat_prim, 3): eigenvector components are ordered atom-major
        evec = data.eigenvectors.reshape(nq, nmode, nmode // 3, 3)
        # exp(2*pi*i*q.R) for every supercell atom (lattice phase convention)
        phases = np.exp(2j * np.pi * (qpoints @ lattice_points.T))  # (nq, nat)
        # projectors[q, nu, a, i] = Re[e_{q,nu}^{p(a),i} * exp(2*pi*i*q.R_a)]
        projectors = (evec[:, :, map_s2p, :] * phases[:, None, :, None]).real

        self_paired = np.all(
            np.abs(2.0 * qpoints - np.round(2.0 * qpoints)) < 1.0e-6, axis=1)
        if not self_paired.all():
            warnings.warn(
                "Some commensurate q points satisfy q != -q (mod G). For those the "
                "real projection used here mixes the +q and -q normal coordinates; "
                "interpret their amplitudes with care.", stacklevel=2)

        mass_sqrt = np.sqrt(supercell.get_masses())[:, None].repeat(3, axis=1)

        basis = cls(qpoints, labels, data.frequencies_cm, projectors, mass_sqrt,
                    supercell.get_chemical_symbols(), primitive.cell[:], matrix, ncells,
                    source=data.source)

        if cache:
            basis.save(cache)
            if verbose:
                print("Wrote projection basis cache %s" % cache)
        return basis

    # ------------------------------------------------------------------- I/O

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            version=CACHE_VERSION,
            qpoints=self.qpoints,
            labels=np.array(self.labels),
            frequencies_cm=self.frequencies_cm,
            projectors=self.projectors,
            mass_sqrt=self.mass_sqrt,
            symbols=np.array(self.symbols),
            primitive_cell=self.primitive_cell,
            supercell_matrix=self.supercell_matrix,
            ncells=self.ncells,
            source=np.array(self.source),
        )

    @classmethod
    def load(cls, path: str) -> "ModeBasis":
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != CACHE_VERSION:
                raise ValueError("Cache %s was written by another version; delete it."
                                 % path)
            return cls(data["qpoints"], [str(s) for s in data["labels"]],
                       data["frequencies_cm"], data["projectors"], data["mass_sqrt"],
                       [str(s) for s in data["symbols"]], data["primitive_cell"],
                       data["supercell_matrix"], int(data["ncells"]),
                       source=str(data["source"]))

    # -------------------------------------------------------------- selection

    @property
    def nmode(self) -> int:
        return self.frequencies_cm.shape[1]

    @property
    def supercell_cell(self) -> np.ndarray:
        """(3, 3) lattice vectors of the supercell in Angstrom, rows are vectors."""
        return self.supercell_matrix @ self.primitive_cell

    def find_qpoint(self, spec: str) -> int:
        """Index of a q point given a label (``"M"``) or coordinates (``"0.5,0.5,0"``)."""
        spec = spec.strip()
        if "," in spec or " " in spec.strip():
            values = [float(t) for t in spec.replace(",", " ").split()]
            if len(values) != 3:
                raise ValueError("Expected three q-point components, got %r" % spec)
            target = np.array(values)
            diff = self.qpoints - target
            distance = np.linalg.norm(diff - np.round(diff), axis=1)
            iq = int(np.argmin(distance))
            if distance[iq] > 1.0e-3:
                raise ValueError(
                    "q point %s is not commensurate with the supercell. Available:\n%s"
                    % (spec, self.qpoint_table()))
            return iq

        for iq, label in enumerate(self.labels):
            if label.lower() == spec.lower():
                return iq
        raise ValueError("Unknown q point %r. Available labels: %s"
                         % (spec, ", ".join(self.labels)))

    def select(self, spec: str) -> ModeSelection:
        """Parse a mode specification such as ``"M:2"``, ``"R"`` or ``"0.5,0.5,0.5:1"``.

        The part after ``:`` is the branch index, counted from 1 in order of
        increasing frequency, exactly like the ``### mode`` numbering of ALAMODE.
        It defaults to 1, the lowest branch.
        """
        qspec, _, mode_spec = spec.rpartition(":")
        if not qspec:  # no colon in spec
            qspec, mode_spec = spec, "1"
        iq = self.find_qpoint(qspec)
        imode = int(mode_spec) - 1
        if not 0 <= imode < self.nmode:
            raise ValueError("Branch index must be between 1 and %d, got %s"
                             % (self.nmode, mode_spec))
        return ModeSelection(iq=iq, imode=imode, label=self.labels[iq],
                             frequency_cm=float(self.frequencies_cm[iq, imode]))

    def degenerate_branches(self, iq: int, imode: int, tol_cm: float = 1.0) -> np.ndarray:
        """Branch indices degenerate with ``imode`` at q point ``iq``."""
        freq = self.frequencies_cm[iq]
        return np.where(np.abs(freq - freq[imode]) < tol_cm)[0]

    def qpoint_table(self) -> str:
        """Human-readable table of commensurate q points and their frequencies."""
        lines = [""]
        if self.source:
            lines.append("Phonons from %s" % self.source)
        lines += ["Commensurate q points (%d cells in the supercell)" % self.ncells,
                 "%-6s %-24s %s" % ("label", "q (primitive recip.)",
                                    "frequencies in cm^-1 (branch 1, 2, ...)"),
                 "-" * 100]
        for iq, q in enumerate(self.qpoints):
            freq = self.frequencies_cm[iq]
            shown = "  ".join("%8.2f" % f for f in freq[:8])
            more = "  ..." if len(freq) > 8 else ""
            lines.append("%-6s %-24s %s%s"
                         % (self.labels[iq], "%6.3f %6.3f %6.3f" % tuple(q), shown, more))
        lines += ["-" * 100,
                  "Negative frequencies are imaginary (unstable) modes.",
                  "Select a mode as LABEL:BRANCH, e.g. 'M:2' (branch numbering starts "
                  "at 1, ordered by frequency).", ""]
        return "\n".join(lines)

    # ------------------------------------------------------------- projection

    def project(self, displacements: np.ndarray, normalize: bool = False) -> np.ndarray:
        """Project displacements onto every mode of the basis.

        Parameters
        ----------
        displacements
            ``(nframes, nat, 3)`` or ``(nat, 3)`` cartesian displacements in Angstrom.
        normalize
            Divide by ``sqrt(ncells)`` so that the amplitude does not depend on the
            supercell size.  ``False`` (default) reproduces the ALAMODE convention.

        Returns
        -------
        numpy.ndarray
            ``(nframes, nq, nmode)`` mode amplitudes in amu^(1/2) Angstrom, or
            ``(nq, nmode)`` for a single frame.
        """
        disp = np.asarray(displacements, dtype=float)
        single = disp.ndim == 2
        if single:
            disp = disp[None]
        if disp.shape[1:] != self.mass_sqrt.shape:
            raise ValueError("Expected displacements with shape (nframes, %d, 3), got %s"
                             % (self.mass_sqrt.shape[0], (disp.shape,)))

        weighted = disp * self.mass_sqrt
        amplitudes = np.einsum("tai,qnai->tqn", weighted, self.projectors,
                               optimize=True)
        if normalize:
            amplitudes = amplitudes / np.sqrt(self.ncells)
        return amplitudes[0] if single else amplitudes

    def project_modes(self, displacements: np.ndarray, modes, normalize: bool = False):
        """Project onto selected modes only.

        Returns a dict ``{mode.name: (nframes,) array}`` preserving the order of
        ``modes``.
        """
        amplitudes = self.project(displacements, normalize=normalize)
        if amplitudes.ndim == 2:
            amplitudes = amplitudes[None]
        return {mode.name: amplitudes[:, mode.iq, mode.imode] for mode in modes}


def _check_masses(masses_source, primitive, source: str = "the phonon data") -> None:
    """Warn when the masses of the phonon calculation and the structure differ.

    Phonon codes list masses either per atomic kind (ALAMODE) or per atom of the
    primitive cell (phonopy); both are compared against the structure files, which are
    what the projection actually uses.
    """
    if masses_source is None:
        return
    masses_source = np.asarray(masses_source, dtype=float)
    if np.isnan(masses_source).any():
        return

    symbols = primitive.get_chemical_symbols()
    masses_structure = primitive.get_masses()
    if len(masses_source) == len(symbols):
        reference = masses_structure
    else:
        kinds = list(dict.fromkeys(symbols))
        if len(masses_source) != len(kinds):
            warnings.warn("%s lists %d masses, which matches neither the %d atoms nor "
                          "the %d atomic kinds of the primitive cell; masses are taken "
                          "from the structure files."
                          % (source, len(masses_source), len(symbols), len(kinds)),
                          stacklevel=3)
            return
        reference = np.array([masses_structure[symbols.index(kind)] for kind in kinds])

    if not np.allclose(reference, masses_source, rtol=1.0e-3):
        warnings.warn(
            "Atomic masses differ between %s (%s) and the structure files (%s). The "
            "structure files are used; check that the atomic order matches."
            % (source, np.array2string(masses_source, precision=3),
               np.array2string(reference, precision=3)), stacklevel=3)


def _cache_is_fresh(cache: str, sources: list[str]) -> bool:
    if not os.path.exists(cache):
        return False
    cache_time = os.path.getmtime(cache)
    return all(os.path.exists(s) and os.path.getmtime(s) <= cache_time for s in sources)
