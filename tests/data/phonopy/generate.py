"""Create the small phonopy fixture files used by the tests.

Run with phonopy installed::

    python tests/data/phonopy/generate.py

The system is a made-up two-atom cubic cell with random (but acoustic-sum-rule
obeying) force constants.  The second atom sits at (0.5, 0.25, 0.25) on purpose: with
quarter coordinates the phonopy phase convention ``exp(2 pi i q.(R + tau_j - tau_a))``
gives genuinely complex eigenvectors at the zone boundary, so the fixtures exercise the
conversion in :mod:`modeproj.eigen`.

``MPOSCAR`` is a *modulated* supercell written by phonopy itself: it holds the
displacement pattern of one chosen (q, band) as phonopy builds it.  Projecting it must
give a single non-zero amplitude for that mode, which is how the tests check that
modeproj interprets phonopy eigenvectors correctly without needing phonopy installed.
"""

import json
import os

import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import write_vasp
from phonopy.structure.atoms import PhonopyAtoms

HERE = os.path.dirname(os.path.abspath(__file__))
DIMENSION = [2, 2, 1]
# q point and band (0-based, by increasing frequency) used for MPOSCAR
MODULATION_Q = [0.5, 0.5, 0.0]
MODULATION_BAND = 3
MODULATION_AMPLITUDE = 1.5


def random_force_constants(natom, seed=0):
    """Random force constants that are symmetric and obey the acoustic sum rule."""
    rng = np.random.default_rng(seed)
    force_constants = rng.normal(scale=0.4, size=(natom, natom, 3, 3))
    # each 3x3 block symmetric, and Phi_ij = Phi_ji
    force_constants = 0.5 * (force_constants + force_constants.transpose(0, 1, 3, 2))
    force_constants = 0.5 * (force_constants + force_constants.transpose(1, 0, 2, 3))
    # acoustic sum rule: the self term cancels the sum of all the others
    for i in range(natom):
        force_constants[i, i] = 0.0
        force_constants[i, i] = -force_constants[i].sum(axis=0)
    return force_constants


def main():
    unitcell = PhonopyAtoms(symbols=["Na", "Cl"],
                            cell=np.eye(3) * 4.2,
                            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.25, 0.25]])
    phonon = Phonopy(unitcell, supercell_matrix=np.diag(DIMENSION))
    phonon.force_constants = random_force_constants(len(phonon.supercell))

    write_vasp(os.path.join(HERE, "PPOSCAR"), phonon.primitive)
    write_vasp(os.path.join(HERE, "SPOSCAR"), phonon.supercell)

    qpoints = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]
    phonon.run_qpoints(qpoints, with_eigenvectors=True)
    phonon.write_yaml_qpoints_phonon()
    phonon.write_hdf5_qpoints_phonon()

    # the mesh has to be Gamma centred to contain the commensurate q points
    phonon.run_mesh(DIMENSION, with_eigenvectors=True, is_mesh_symmetry=False,
                    is_gamma_center=True)
    phonon.write_yaml_mesh()
    phonon.write_hdf5_mesh()

    for name in ("qpoints.yaml", "qpoints.hdf5", "mesh.yaml", "mesh.hdf5"):
        os.replace(name, os.path.join(HERE, name))

    phonon.run_qpoints(qpoints, with_eigenvectors=True)

    phonon.save(os.path.join(HERE, "phonopy.yaml"), settings={"force_constants": True})

    phonon.run_modulations(
        DIMENSION, [[MODULATION_Q, MODULATION_BAND, MODULATION_AMPLITUDE, 0.0]])
    modulated = phonon.get_modulated_supercells()[0]
    write_vasp(os.path.join(HERE, "MPOSCAR"), modulated)

    frequencies = phonon.get_qpoints_dict()["frequencies"]
    with open(os.path.join(HERE, "fixture.json"), "w") as handle:
        json.dump({
            "created_with": "phonopy %s" % phonon.version,
            "dimension": DIMENSION,
            "qpoints": qpoints,
            "frequencies_thz": np.asarray(frequencies).tolist(),
            "modulation": {"qpoint": MODULATION_Q, "band": MODULATION_BAND,
                           "amplitude": MODULATION_AMPLITUDE},
        }, handle, indent=1)
        handle.write("\n")
    print("wrote fixtures to %s" % HERE)


if __name__ == "__main__":
    main()
