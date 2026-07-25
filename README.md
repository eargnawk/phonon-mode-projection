# Phonon mode projection

Project an *ab initio* molecular dynamics trajectory onto phonon normal modes, and watch how much
of each mode is present as a function of time.

The example in this repository reproduces **Figure 3** of

> *Exploring anharmonic lattice dynamics and dielectric relations in niobate perovskites from
> first-principles self-consistent phonon calculations*,
> npj Computational Materials **9**, 154 (2023) — <https://doi.org/10.1038/s41524-023-01110-8>

for cubic KNbO<sub>3</sub>: 10 ps of AIMD on a 2×2×2 supercell, projected onto the soft polar
(Γ), and antiferrodistortive (M and R) phonon modes.

![Mode amplitudes of cubic KNbO3](docs/amplitudes.png)

## What is computed

Each MD snapshot is a set of displacements **u**<sub>a</sub> from the ideal structure. Rather than
40 atoms × 3 directions, the mode amplitude asks *how much of one phonon mode* a snapshot contains:

```
Q_qν(t) = Σ_a √m_a · u_a(t) · Re[ e_qν^p(a) · exp(2πi q·R_a) ]
```

with `m_a` the mass in amu, `e_qν` the phonon eigenvector from ALAMODE, `p(a)` the equivalent atom
in the primitive cell and `R_a` the lattice point of atom `a`. `Q` is in amu<sup>1/2</sup> Å. This
is the same convention as ALAMODE's `displace.py --pes`.

## Requirements

Python ≥ 3.8 with `numpy`, `ase` and `matplotlib`:

```bash
pip install -r requirements.txt        # or: pip install -e .
```

## The example data

| file | what it is |
|---|---|
| `XDATCAR` | AIMD trajectory: 10000 frames × 40 atoms, 1 fs steps |
| `PPOSCAR` | primitive cell, 5 atoms |
| `SPOSCAR` | MD supercell, 40 atoms (2×2×2) |
| `kno_221.mesh.evec.tar.gz` | phonon eigenvectors from ALAMODE `anphon` |

Unpack the eigenvectors once (60 MB unpacked, not tracked by git):

```bash
tar xzf kno_221.mesh.evec.tar.gz
```

## Quickstart

```bash
# 1. which q points does the supercell support, and what are their frequencies?
python -m modeproj modes

# 2. project the trajectory onto three modes, write a table and the figure above
python -m modeproj project --mode G:1 --mode M:2 --mode R:1 \
    --timestep 1.0 --out amplitudes.csv --plot amplitudes.png --histogram histogram.png
```

`modes` prints

```
label  q (primitive recip.)     frequencies in cm^-1 (branch 1, 2, ...)
G       0.000  0.000  0.000      -188.66   -188.66   -188.66      0.00      0.00      0.00 ...
X      -0.500  0.000  0.000      -131.84   -131.84    144.76    144.76    161.11    161.11 ...
M      -0.500 -0.500  0.000       -93.83    153.69    156.29    156.29    161.31    260.22 ...
R      -0.500 -0.500 -0.500       144.98    144.98    144.98    175.86    175.86    175.86 ...
```

Negative values are imaginary (unstable) frequencies. Modes are selected as `LABEL:BRANCH`, where
branches are numbered from 1 by increasing frequency, exactly like the `### mode` entries of the
`.evec` file. Explicit coordinates work as well: `--mode 0.5,0.5,0.5:1` is the same as `--mode R:1`.
Members of the same star are symmetry equivalent and get numbered labels (`X`, `X2`, `X3`).

Useful options for `project`:

| option | meaning |
|---|---|
| `--timestep 2.0` | MD time step in fs, sets the time axis |
| `--stride 10`, `--start`, `--stop` | use a subset of frames for a quick look |
| `--reference first \| mean \| supercell` | reference structure for the displacements |
| `--all-modes` | write the amplitude of every q point and branch |
| `--normalize` | divide by √N<sub>cell</sub> so amplitudes do not scale with the supercell |
| `--window 250` | running-average window (in frames) drawn on top of the raw signal |
| `--out amplitudes.csv \| .npz` | save the amplitudes |

`python -m modeproj <command> --help` lists everything.

## Notebook

[`projection.ipynb`](projection.ipynb) walks through the same analysis step by step and adds
amplitude histograms (single-well vibration vs double-well hopping) and mode-resolved
⟨Q²⟩ values.

## Python API

```python
from modeproj import ModeBasis, read_xdatcar, plot_amplitudes

basis = ModeBasis.from_files("kno_221.mesh.evec", "PPOSCAR", "SPOSCAR")
print(basis.qpoint_table())

trajectory = read_xdatcar("XDATCAR", stride=1)
modes = [basis.select(spec) for spec in ("G:1", "M:2", "R:1")]

Q = basis.project_modes(trajectory.displacements(), modes)   # {"Q_G_1": array, ...}
Q_all = basis.project(trajectory.displacements())            # (nframes, nq, nbranch)
```

`ModeBasis.from_files` works out the supercell matrix, the commensurate q points and the
atom→(primitive atom, lattice point) mapping, then reads **only the required q points** from the
`.evec` file and caches the result in `<evec>.basis.npz`. A 60 MB mesh file is read in about a
second instead of a minute, and the whole trajectory is projected with a single `einsum`.

## Using your own data

1. Relax the primitive cell and build the MD supercell (`PPOSCAR`/`SPOSCAR` from phonopy work well).
2. Run an MD simulation in that supercell at fixed cell volume and shape, giving an `XDATCAR`.
3. Compute phonons with ALAMODE `anphon` and print the eigenvectors at the q points commensurate
   with the supercell. The required input blocks are generated for you:

   ```bash
   python -m modeproj anphon-input --primitive PPOSCAR --supercell SPOSCAR
   ```

   Any denser mesh containing those q points is fine too — the example file is a 20×20×20 mesh.
4. Project:

   ```bash
   python -m modeproj project --evec PREFIX.evec --primitive PPOSCAR \
       --supercell SPOSCAR --xdatcar XDATCAR --mode R:1
   ```

## Conventions and limitations

* Amplitudes contain **no** 1/√N<sub>cell</sub> factor (ALAMODE convention), so they grow with the
  supercell size; `--normalize` / `project(..., normalize=True)` divides it out.
* Displacements use the minimum-image convention, so atoms that cross a periodic boundary do not
  produce spurious jumps.
* Only q points with **q** ≡ −**q** (components 0 or ±½) have a purely real normal coordinate. For
  other commensurate q points the real projection mixes +**q** and −**q**; a warning is issued.
* Within a degenerate set the individual eigenvectors are arbitrary up to a rotation, so a single
  `Q` is basis dependent. Use `basis.degenerate_branches(iq, imode)` and combine them as
  √(Σ Q²) when that matters.
* A constant cell (NVE/NVT) is assumed. Frequencies in cm<sup>-1</sup> come from ω² in Rydberg
  atomic units as stored by `anphon`.

## Repository layout

```
modeproj/            the package: evec reader, geometry, projection, plots, CLI
  evec.py            targeted reader for ALAMODE PREFIX.evec files
  geometry.py        supercell matrix, commensurate q points, atom mapping, q labels
  basis.py           ModeBasis: eigenvectors + projection + mode selection
  trajectory.py      fast XDATCAR reader and displacements
  plot.py            amplitude-vs-time and histogram figures
  cli.py             python -m modeproj
tests/               unit tests, run with `python -m pytest`
projection.ipynb     annotated walk-through of the example
tools/               ALAMODE displace.py and its interfaces (MIT, T. Tadano), kept for
                     reference; the projection no longer imports them
```

## Credits

The projection convention and the geometry bookkeeping follow
[ALAMODE](https://github.com/ttadano/alamode) (`displace.py --pes`, © 2014–2020 Terumasa Tadano,
MIT licence); a copy of those scripts is in `tools/`. Please cite the paper above when using this
example.
