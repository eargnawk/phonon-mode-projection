"""Tests for the phonopy interface.

The fixtures in ``tests/data/phonopy`` were produced by phonopy itself (see
``generate.py`` there), so these tests check modeproj against phonopy's own output and
its own modulated structure without needing phonopy at test time.

The decisive check is :func:`test_modulated_structure_is_a_single_mode`: ``MPOSCAR`` is
the supercell that phonopy builds for one chosen ``(q, band)``.  Projecting it must give
that mode and nothing else - which only works if the phonopy phase convention
(``exp(2 pi i q.(R + tau))``) is handled correctly.
"""

import json
import os
import warnings

import numpy as np
import pytest
from ase.io import read as ase_read

from modeproj import ModeBasis, detect_format
from modeproj.modedata import THZ_TO_CM

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "phonopy")
FIXTURE = json.load(open(os.path.join(DATA, "fixture.json")))

EIGENVECTOR_FILES = ["qpoints.yaml", "mesh.yaml", "qpoints.hdf5", "mesh.hdf5"]


def build_basis(name, **kwargs):
    with warnings.catch_warnings():
        # files without structure information warn on purpose; not the point here
        warnings.simplefilter("ignore")
        return ModeBasis.from_files(os.path.join(DATA, name),
                                    os.path.join(DATA, "PPOSCAR"),
                                    os.path.join(DATA, "SPOSCAR"),
                                    cache=None, verbose=False, **kwargs)


def modulated_displacements():
    """Displacements of phonopy's modulated supercell, in the SPOSCAR atom order."""
    ideal = ase_read(os.path.join(DATA, "SPOSCAR"), format="vasp")
    modulated = ase_read(os.path.join(DATA, "MPOSCAR"), format="vasp")
    difference = modulated.get_scaled_positions() - ideal.get_scaled_positions()
    difference -= np.round(difference)
    return difference @ ideal.cell[:]


# ------------------------------------------------------------------ format guessing


@pytest.mark.parametrize("name,expected", [
    ("qpoints.yaml", "phonopy-yaml"),
    ("mesh.yaml", "phonopy-yaml"),
    ("qpoints.hdf5", "phonopy-hdf5"),
    ("mesh.hdf5", "phonopy-hdf5"),
    ("phonopy.yaml", "phonopy-fc"),
])
def test_detect_format(name, expected):
    assert detect_format(os.path.join(DATA, name)) == expected


def test_detect_format_alamode(tmp_path):
    path = tmp_path / "prefix.evec"
    path.write_text("")
    assert detect_format(str(path)) == "alamode"


# ---------------------------------------------------------------------- frequencies


@pytest.mark.parametrize("name", EIGENVECTOR_FILES)
def test_frequencies_match_phonopy(name):
    basis = build_basis(name)
    expected = np.array(FIXTURE["frequencies_thz"]) * THZ_TO_CM

    assert basis.labels == ["G", "X", "X2", "M"]
    assert basis.qpoints.shape == (4, 3)
    for iq, q in enumerate(FIXTURE["qpoints"]):
        jq = basis.find_qpoint(",".join(str(component) for component in q))
        assert basis.frequencies_cm[jq] == pytest.approx(expected[iq], abs=1e-3)


# ------------------------------------------------------- the projection basis itself


@pytest.mark.parametrize("name", EIGENVECTOR_FILES)
def test_projectors_are_orthonormal(name):
    """The 3N projectors must be an orthonormal basis (times the number of cells).

    A wrong phase convention or a wrong atom order destroys this.
    """
    basis = build_basis(name)
    flat = basis.projectors.reshape(-1, basis.projectors.shape[2] * 3)
    gram = flat @ flat.T / basis.ncells
    assert flat.shape[0] == flat.shape[1]  # 3N modes for 3N degrees of freedom
    assert np.allclose(gram, np.eye(len(gram)), atol=1e-6)


@pytest.mark.parametrize("name", EIGENVECTOR_FILES)
def test_projection_conserves_the_norm(name):
    """Parseval: sum of Q^2 over all modes = ncells * sum of m_a u_a^2."""
    basis = build_basis(name)
    rng = np.random.default_rng(4)
    displacements = rng.normal(scale=0.05, size=(3, len(basis.symbols), 3))

    amplitudes = basis.project(displacements)
    total = (amplitudes ** 2).sum(axis=(1, 2))
    expected = basis.ncells * ((displacements * basis.mass_sqrt) ** 2).sum(axis=(1, 2))
    assert total == pytest.approx(expected, rel=1e-8)


def test_modulated_structure_is_a_single_mode():
    """phonopy's own modulation of one mode must project onto exactly that mode."""
    basis = build_basis("mesh.yaml")
    modulation = FIXTURE["modulation"]
    iq = basis.find_qpoint(",".join(str(c) for c in modulation["qpoint"]))
    imode = modulation["band"]

    amplitudes = basis.project(modulated_displacements())
    picked = abs(amplitudes[iq, imode])

    # everything that is not the modulated mode (or degenerate with it) must vanish
    degenerate = basis.degenerate_branches(iq, imode)
    others = np.array([[amplitudes[jq, jmode]
                        for jmode in range(basis.nmode)
                        if not (jq == iq and jmode in degenerate)]
                       for jq in range(len(basis.qpoints))], dtype=object)
    largest_other = max(abs(value) for row in others for value in row)

    # phonopy writes u_a = A Re[e_a exp(2 pi i q.r_a)] / (sqrt(m_a) sqrt(N_atoms)), and
    # the projectors have norm^2 = ncells, so Q = A * ncells / sqrt(N_atoms)
    expected = modulation["amplitude"] * basis.ncells / np.sqrt(len(basis.symbols))
    assert picked == pytest.approx(expected, rel=1e-3)
    # the residue is limited by the mass tables: phonopy builds the modulation with its
    # own masses, the projection uses ASE's (they differ in the 5th digit for Cl)
    assert largest_other < 1e-3 * picked


@pytest.mark.parametrize("name", EIGENVECTOR_FILES[1:])
def test_all_files_give_the_same_projection(name):
    """qpoints/mesh, yaml/hdf5: same amplitudes for the same displacements."""
    reference = build_basis("qpoints.yaml")
    other = build_basis(name)
    displacements = modulated_displacements()

    for iq, q in enumerate(reference.qpoints):
        jq = other.find_qpoint(",".join(str(component) for component in q))
        assert other.frequencies_cm[jq] == pytest.approx(reference.frequencies_cm[iq],
                                                         abs=1e-3)
        assert np.abs(other.project(displacements)[jq]) == \
            pytest.approx(np.abs(reference.project(displacements)[iq]), abs=1e-6)


# --------------------------------------------------- phonopy.yaml + force constants


def test_force_constants_route_matches_the_files():
    pytest.importorskip("phonopy", reason="the phonopy-fc route needs phonopy")
    basis = build_basis("phonopy.yaml")
    reference = build_basis("mesh.yaml")

    assert basis.frequencies_cm == pytest.approx(reference.frequencies_cm, abs=1e-3)
    displacements = modulated_displacements()
    assert np.abs(basis.project(displacements)) == \
        pytest.approx(np.abs(reference.project(displacements)), abs=1e-6)


def test_missing_qpoint_reports_gamma_centre(tmp_path):
    """A mesh without the commensurate q points must say what to do about it."""
    import h5py

    with h5py.File(os.path.join(DATA, "mesh.hdf5"), "r") as source:
        qpoints = np.array(source["qpoint"])
        frequencies = np.array(source["frequency"])
        eigenvectors = np.array(source["eigenvector"])
    path = tmp_path / "mesh.hdf5"
    with h5py.File(path, "w") as target:
        target["qpoint"] = qpoints + 0.1          # shifted off the commensurate points
        target["frequency"] = frequencies
        target["eigenvector"] = eigenvectors

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="Gamma-centred"):
            ModeBasis.from_files(str(path), os.path.join(DATA, "PPOSCAR"),
                                 os.path.join(DATA, "SPOSCAR"), cache=None,
                                 verbose=False)


def test_wrong_primitive_cell_is_reported(tmp_path):
    """A primitive cell that is not the one phonopy used must be caught."""
    primitive = ase_read(os.path.join(DATA, "PPOSCAR"), format="vasp")
    primitive.set_cell(primitive.cell[:] * 1.1, scale_atoms=True)
    primitive.write(tmp_path / "PPOSCAR", format="vasp")
    supercell = ase_read(os.path.join(DATA, "SPOSCAR"), format="vasp")
    supercell.set_cell(supercell.cell[:] * 1.1, scale_atoms=True)
    supercell.write(tmp_path / "SPOSCAR", format="vasp")

    with pytest.raises(ValueError, match="same lattice"):
        ModeBasis.from_files(os.path.join(DATA, "mesh.yaml"), str(tmp_path / "PPOSCAR"),
                             str(tmp_path / "SPOSCAR"), cache=None, verbose=False)
