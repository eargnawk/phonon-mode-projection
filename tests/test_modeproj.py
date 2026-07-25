"""Unit tests: ``python -m pytest``.

The tests build small synthetic inputs, so they run without the large example
data.  The last test is an end-to-end check that is skipped unless the unpacked
``kno_221.mesh.evec`` of the example is present.
"""

import os

import numpy as np
import pytest
from ase import Atoms

from modeproj import (
    ModeBasis,
    commensurate_qpoints,
    read_eigenvectors,
    read_xdatcar,
    rolling_mean,
    supercell_matrix,
)
from modeproj.geometry import label_qpoints, map_supercell_to_primitive

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cubic_primitive(a=4.0):
    return Atoms("KNbO3", cell=[a, a, a], pbc=True,
                 scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0],
                                   [0, 0.5, 0.5], [0.5, 0, 0.5]])


# --------------------------------------------------------------------- geometry


def test_supercell_matrix_and_qpoints():
    primitive = cubic_primitive()
    supercell = primitive.repeat((2, 2, 2))
    matrix = supercell_matrix(primitive.cell[:], supercell.cell[:])
    assert np.array_equal(matrix, 2 * np.eye(3, dtype=int))

    qpoints = commensurate_qpoints(matrix)
    assert len(qpoints) == 8
    assert np.allclose(qpoints[0], 0.0)  # Gamma comes first
    # every q must satisfy exp(2 pi i q.R) = 1 for the supercell translations
    for q in qpoints:
        assert np.allclose(matrix @ q, np.round(matrix @ q))
    assert label_qpoints(qpoints, primitive.cell[:]) == \
        ["G", "X", "X2", "X3", "M", "M2", "M3", "R"]


def test_supercell_matrix_rejects_mismatch():
    primitive = cubic_primitive()
    with pytest.raises(ValueError):
        supercell_matrix(primitive.cell[:], primitive.cell[:] * 1.5)


def test_commensurate_qpoints_non_diagonal():
    matrix = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
    qpoints = commensurate_qpoints(matrix)
    assert len(qpoints) == 2
    for q in qpoints:
        assert np.allclose(matrix @ q, np.round(matrix @ q))


def test_mapping_supercell_to_primitive():
    primitive = cubic_primitive()
    supercell = primitive.repeat((2, 1, 1))
    map_s2p, lattice_points = map_supercell_to_primitive(primitive, supercell)
    assert sorted(map_s2p) == sorted(list(range(5)) * 2)
    # ASE's repeat puts the two images of each atom next to each other
    assert np.allclose(sorted(lattice_points[:, 0]), [0] * 5 + [1] * 5)
    for iat, jat in enumerate(map_s2p):
        assert supercell[iat].symbol == primitive[jat].symbol


# ------------------------------------------------------------------- trajectory


def write_xdatcar(path, cell, symbols, counts, frames):
    with open(path, "w") as fh:
        fh.write("test\n           1\n")
        for row in cell:
            fh.write("  %.6f %.6f %.6f\n" % tuple(row))
        fh.write("  " + "  ".join(symbols) + "\n")
        fh.write("  " + "  ".join(str(c) for c in counts) + "\n")
        for index, frame in enumerate(frames, start=1):
            fh.write("Direct configuration=  %d\n" % index)
            for position in frame:
                fh.write("  %.8f %.8f %.8f\n" % tuple(position))


def test_read_xdatcar_and_slicing(tmp_path):
    cell = np.diag([4.0, 4.0, 4.0])
    frames = [np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) + 0.01 * i
              for i in range(5)]
    path = tmp_path / "XDATCAR"
    write_xdatcar(path, cell, ["K", "Nb"], [1, 1], frames)

    trajectory = read_xdatcar(path)
    assert trajectory.nframes == 5 and trajectory.natoms == 2
    assert trajectory.symbols == ["K", "Nb"]
    assert np.allclose(trajectory.cell, cell)

    strided = read_xdatcar(path, start=1, stop=5, stride=2)
    assert strided.nframes == 2
    assert np.allclose(strided.scaled_positions[0], frames[1])


def test_displacements_use_minimum_image(tmp_path):
    cell = np.diag([4.0, 4.0, 4.0])
    # the atom sits at the cell boundary and is wrapped from 0.0 to 0.99
    frames = [np.array([[0.0, 0.0, 0.0]]), np.array([[0.99, 0.0, 0.0]])]
    path = tmp_path / "XDATCAR"
    write_xdatcar(path, cell, ["K"], [1], frames)

    displacements = read_xdatcar(path).displacements("first")
    assert np.allclose(displacements[1], [-0.04, 0.0, 0.0])


def test_displacements_mean_reference(tmp_path):
    cell = np.diag([4.0, 4.0, 4.0])
    frames = [np.array([[0.1, 0.0, 0.0]]), np.array([[0.3, 0.0, 0.0]])]
    path = tmp_path / "XDATCAR"
    write_xdatcar(path, cell, ["K"], [1], frames)

    displacements = read_xdatcar(path).displacements("mean")
    assert np.allclose(displacements[:, 0, 0], [-0.4, 0.4])


# ------------------------------------------------------------------ evec reader


def write_evec(path, qpoints, omega2, eigenvectors, masses):
    nmode = omega2.shape[1]
    with open(path, "w") as fh:
        fh.write("# Lattice vectors of the primitive cell\n")
        for row in np.eye(3) * 7.5:
            fh.write("  %e %e %e\n" % tuple(row))
        fh.write("\n# Number of phonon modes: %d\n" % nmode)
        fh.write("# Number of k points : %d\n" % len(qpoints))
        fh.write("# Number of atomic kinds : %d\n" % len(masses))
        fh.write("# Atomic masses : " + " ".join("%e" % m for m in masses) + "\n\n")
        fh.write("# Eigenvalues and eigenvectors for each phonon modes below:\n\n")
        for iq, q in enumerate(qpoints):
            fh.write("## kpoint %d : %e %e %e\n" % (iq + 1, q[0], q[1], q[2]))
            for imode in range(nmode):
                fh.write("### mode %d : %e\n" % (imode + 1, omega2[iq, imode]))
                for jmode in range(nmode):
                    value = eigenvectors[iq, imode, jmode]
                    fh.write("  %e %e\n" % (value.real, value.imag))
                fh.write("\n")
            fh.write("\n")


def test_read_eigenvectors_selects_requested_qpoints(tmp_path):
    rng = np.random.default_rng(0)
    qpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
    omega2 = rng.normal(size=(3, 6)) * 1e-5
    eigenvectors = rng.normal(size=(3, 6, 6)) + 1j * rng.normal(size=(3, 6, 6))
    path = tmp_path / "test.evec"
    write_evec(path, qpoints, omega2, eigenvectors, [39.1, 92.9, 16.0])

    data = read_eigenvectors(path, [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(data.qpoints, [[0.5, 0, 0], [0, 0, 0]])
    assert np.allclose(data.omega2, omega2[[2, 0]], atol=1e-12)
    assert np.allclose(data.eigenvectors, eigenvectors[[2, 0]], atol=1e-6)
    # -0.5 and +0.5 differ by a reciprocal lattice vector and must both match
    assert np.allclose(read_eigenvectors(path, [[-0.5, 0.0, 0.0]]).qpoints,
                       [[0.5, 0.0, 0.0]])


def test_read_eigenvectors_reports_missing_qpoint(tmp_path):
    rng = np.random.default_rng(1)
    qpoints = np.array([[0.0, 0.0, 0.0]])
    path = tmp_path / "test.evec"
    write_evec(path, qpoints, rng.normal(size=(1, 3)) * 1e-5,
               rng.normal(size=(1, 3, 3)) + 0j, [16.0])
    with pytest.raises(RuntimeError, match="not found"):
        read_eigenvectors(path, [[0.5, 0.5, 0.5]])


# ------------------------------------------------------------------- projection


def synthetic_basis(tmp_path, repeat=(2, 1, 1)):
    """A one-atom primitive cell with hand-made eigenvectors, for exact tests."""
    primitive = Atoms("K", cell=[4.0, 4.0, 4.0], pbc=True, scaled_positions=[[0, 0, 0]])
    supercell = primitive.repeat(repeat)
    primitive.write(tmp_path / "PPOSCAR", format="vasp")
    supercell.write(tmp_path / "SPOSCAR", format="vasp")

    qpoints = commensurate_qpoints(supercell_matrix(primitive.cell[:], supercell.cell[:]))
    nmode = 3
    omega2 = np.tile(np.array([1e-6, 4e-6, 9e-6]), (len(qpoints), 1))
    eigenvectors = np.tile(np.eye(nmode, dtype=complex), (len(qpoints), 1, 1))
    write_evec(tmp_path / "test.evec", qpoints, omega2, eigenvectors, [39.0983])
    return ModeBasis.from_files(str(tmp_path / "test.evec"), str(tmp_path / "PPOSCAR"),
                                str(tmp_path / "SPOSCAR"), cache=None, verbose=False)


def test_projection_of_a_known_pattern(tmp_path):
    basis = synthetic_basis(tmp_path)
    mass = 39.0983

    # A uniform displacement along x is purely the Gamma point, branch 1.
    displacement = np.zeros((len(basis.symbols), 3))
    displacement[:, 0] = 0.1
    amplitudes = basis.project(displacement)
    iq_gamma = basis.find_qpoint("G")
    expected = np.sqrt(mass) * 0.1 * len(basis.symbols)
    assert amplitudes[iq_gamma, 0] == pytest.approx(expected)
    assert abs(amplitudes[iq_gamma, 1]) < 1e-10
    assert abs(amplitudes[basis.find_qpoint("X"), 0]) < 1e-10

    # An alternating pattern is the zone-boundary mode instead.
    displacement[:, 0] = [0.1, -0.1]
    amplitudes = basis.project(displacement)
    assert abs(amplitudes[iq_gamma, 0]) < 1e-10
    assert abs(amplitudes[basis.find_qpoint("X"), 0]) == pytest.approx(expected)

    # normalize=True removes the supercell-size dependence
    normalized = basis.project(displacement, normalize=True)
    assert normalized[basis.find_qpoint("X"), 0] == \
        pytest.approx(amplitudes[basis.find_qpoint("X"), 0] / np.sqrt(basis.ncells))


def test_mode_selection_and_cache(tmp_path):
    basis = synthetic_basis(tmp_path)
    mode = basis.select("X:2")
    assert mode.label == "X" and mode.imode == 1
    assert mode.name == "Q_X_2"
    assert mode.frequency_cm == pytest.approx(basis.frequencies_cm[mode.iq, 1])
    assert basis.select("0.5,0,0:2").iq == mode.iq
    assert basis.select("X").imode == 0  # branch defaults to 1
    with pytest.raises(ValueError):
        basis.select("Z:1")
    with pytest.raises(ValueError):
        basis.select("X:99")

    cache = tmp_path / "basis.npz"
    basis.save(cache)
    restored = ModeBasis.load(cache)
    assert restored.labels == basis.labels
    assert np.allclose(restored.projectors, basis.projectors)


def test_degenerate_branches(tmp_path):
    basis = synthetic_basis(tmp_path)
    assert list(basis.degenerate_branches(0, 0)) == [0]


# ------------------------------------------------------------------------ misc


def test_rolling_mean():
    values = np.arange(5.0)
    assert np.allclose(rolling_mean(values, 1), values)
    # centred window, shrinking at the edges
    assert np.allclose(rolling_mean(values, 3), [0.5, 1.0, 2.0, 3.0, 3.5])


# ------------------------------------------------------------------ integration


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "kno_221.mesh.evec")),
                    reason="unpack kno_221.mesh.evec.tar.gz to run this test")
def test_example_reproduces_reference_amplitudes():
    basis = ModeBasis.from_files(os.path.join(ROOT, "kno_221.mesh.evec"),
                                 os.path.join(ROOT, "PPOSCAR"),
                                 os.path.join(ROOT, "SPOSCAR"),
                                 cache=None, verbose=False)
    trajectory = read_xdatcar(os.path.join(ROOT, "XDATCAR"), stop=6)
    modes = [basis.select(spec) for spec in ("G:1", "M:2", "R:1")]
    amplitudes = basis.project_modes(trajectory.displacements(), modes)

    # values of the original notebook implementation for frame 5
    assert amplitudes["Q_G_1"][5] == pytest.approx(1.26575, abs=1e-4)
    assert amplitudes["Q_M_2"][5] == pytest.approx(-1.72265, abs=1e-4)
    assert amplitudes["Q_R_1"][5] == pytest.approx(-1.14303, abs=1e-4)
