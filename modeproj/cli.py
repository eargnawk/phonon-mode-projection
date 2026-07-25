"""Command line interface: ``python -m modeproj ...``"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .basis import ModeBasis
from .plot import plot_amplitudes, plot_histograms
from .sources import FORMAT_HELP, FORMATS
from .trajectory import check_consistency, read_xdatcar

DEFAULT_EVEC = "kno_221.mesh.evec"
DEFAULT_PRIMITIVE = "PPOSCAR"
DEFAULT_SUPERCELL = "SPOSCAR"
DEFAULT_XDATCAR = "XDATCAR"
DEFAULT_MODES = ("G:1", "M:2", "R:1")

EPILOG = """\
examples:
  # list the q points commensurate with the supercell and their frequencies
  python -m modeproj modes

  # reproduce Fig. 3 of npj Comput. Mater. 9, 154 (2023)
  python -m modeproj project --mode G:1 --mode M:2 --mode R:1 \\
      --timestep 1.0 --plot amplitudes.png --out amplitudes.csv

  # a different mode, quick look at every 10th frame
  python -m modeproj project --mode 0.5,0.5,0.5:4 --stride 10

  # phonons from phonopy instead of ALAMODE
  python -m modeproj project --evec phonopy.yaml --force-constants FORCE_CONSTANTS \\
      --primitive PPOSCAR --supercell SPOSCAR --mode R:1
  python -m modeproj project --evec mesh.hdf5 --mode M:2

eigenvector formats (--format, guessed from the file name by default):
"""
EPILOG += "\n".join("  %-14s %s" % (key, text) for key, text in FORMAT_HELP.items())
EPILOG += "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modeproj",
        description="Project an ab initio MD trajectory onto phonon normal modes.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--evec", "--phonons", dest="evec", default=DEFAULT_EVEC,
                       help="eigenvectors of the phonon calculation: an ALAMODE "
                            "PREFIX.evec, phonopy qpoints/mesh/band .yaml or .hdf5, or "
                            "phonopy.yaml together with --force-constants "
                            "(default: %(default)s)")
        p.add_argument("--format", dest="source_format", default="auto", choices=FORMATS,
                       help="format of the eigenvector file (default: %(default)s, "
                            "guessed from the file name)")
        p.add_argument("--force-constants", default=None,
                       help="phonopy FORCE_CONSTANTS or force_constants.hdf5, when "
                            "phonopy.yaml does not contain them")
        p.add_argument("--force-sets", default=None,
                       help="phonopy FORCE_SETS, as an alternative to "
                            "--force-constants")
        p.add_argument("--born", default=None,
                       help="phonopy BORN file, for the non-analytic term correction")
        p.add_argument("--nac", action="store_true",
                       help="switch on phonopy's non-analytic term correction (needs "
                            "BORN)")
        p.add_argument("--nac-direction", default=None, metavar="X,Y,Z",
                       help="direction along which q approaches Gamma for the LO-TO "
                            "splitting, e.g. 1,0,0 (default: Gamma without the "
                            "non-analytic term)")
        p.add_argument("--primitive", default=DEFAULT_PRIMITIVE,
                       help="POSCAR of the primitive cell (default: %(default)s)")
        p.add_argument("--supercell", default=DEFAULT_SUPERCELL,
                       help="POSCAR of the MD supercell (default: %(default)s)")
        p.add_argument("--cache", default=None,
                       help="npz cache for the parsed eigenvectors "
                            "(default: <evec>.basis.npz)")
        p.add_argument("--no-cache", action="store_true",
                       help="do not read or write the eigenvector cache")
        p.add_argument("-q", "--quiet", action="store_true", help="print less")

    modes = sub.add_parser("modes", help="list commensurate q points and frequencies")
    add_common(modes)

    anphon = sub.add_parser(
        "anphon-input",
        help="print the anphon &kpoint block needed to produce the .evec file")
    anphon.add_argument("--primitive", default=DEFAULT_PRIMITIVE,
                        help="POSCAR of the primitive cell (default: %(default)s)")
    anphon.add_argument("--supercell", default=DEFAULT_SUPERCELL,
                        help="POSCAR of the MD supercell (default: %(default)s)")

    project = sub.add_parser("project", help="project a trajectory onto phonon modes")
    add_common(project)
    project.add_argument("--xdatcar", default=DEFAULT_XDATCAR,
                         help="VASP XDATCAR trajectory (default: %(default)s)")
    project.add_argument("--mode", dest="modes", action="append", metavar="SPEC",
                         help="mode to project onto, as LABEL:BRANCH or "
                              "qx,qy,qz:BRANCH, e.g. M:2. Branches are numbered from "
                              "1 by increasing frequency. Repeatable "
                              "(default: %s)" % " ".join(DEFAULT_MODES))
    project.add_argument("--all-modes", action="store_true",
                         help="write the amplitudes of every q point and branch")
    project.add_argument("--timestep", type=float, default=1.0,
                         help="MD time step in fs, for the time axis "
                              "(default: %(default)s)")
    project.add_argument("--start", type=int, default=0, help="first frame to use")
    project.add_argument("--stop", type=int, default=None,
                         help="stop before this frame")
    project.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    project.add_argument("--reference", default="first",
                         choices=("first", "mean", "supercell"),
                         help="reference structure for the displacements: the first "
                              "frame, the time average, or the ideal sites of the "
                              "supercell file (default: %(default)s)")
    project.add_argument("--normalize", action="store_true",
                         help="divide amplitudes by sqrt(number of cells) so they do "
                              "not depend on the supercell size")
    project.add_argument("--window", type=int, default=250,
                         help="running-average window in frames, for the plot "
                              "(default: %(default)s)")
    project.add_argument("--ylim", type=float, nargs=2, default=None,
                         metavar=("LOW", "HIGH"), help="amplitude range of the plot")
    project.add_argument("--out", default=None,
                         help="write the amplitudes to this .csv or .npz file")
    project.add_argument("--plot", default=None,
                         help="write the amplitude-vs-time figure to this file")
    project.add_argument("--histogram", default=None,
                         help="write an amplitude histogram to this file")
    return parser


def _cache_path(args) -> str | None:
    if args.no_cache:
        return None
    return args.cache or (args.evec + ".basis.npz")


def _nac_direction(args):
    if args.nac_direction is None:
        return None
    values = [float(t) for t in args.nac_direction.replace(",", " ").split()]
    if len(values) != 3:
        raise SystemExit("--nac-direction needs three components, got %r"
                         % args.nac_direction)
    return values


def _load_basis(args) -> ModeBasis:
    paths = [(args.evec, "eigenvector file"),
             (args.primitive, "primitive cell"),
             (args.supercell, "supercell")]
    paths += [(path, "force-constants file") for path in
              (args.force_constants, args.force_sets, args.born) if path]
    for path, what in paths:
        if not os.path.exists(path):
            hint = ""
            if path.endswith(".evec") and os.path.exists(path + ".tar.gz"):
                hint = ("\nThe archive %s.tar.gz is there; unpack it first:\n"
                        "  tar xzf %s.tar.gz" % (path, path))
            raise SystemExit("Cannot find the %s '%s'.%s" % (what, path, hint))

    try:
        return ModeBasis.from_files(
            args.evec, args.primitive, args.supercell,
            source_format=args.source_format, cache=_cache_path(args),
            verbose=not args.quiet, force_constants=args.force_constants,
            force_sets=args.force_sets, born=args.born, nac=args.nac,
            nac_direction=_nac_direction(args))
    except (ValueError, RuntimeError) as error:
        raise SystemExit("%s" % error)


def _selected_modes(basis: ModeBasis, args):
    if args.all_modes:
        return [basis.select("%s:%d" % (basis.labels[iq], imode + 1))
                for iq in range(len(basis.qpoints))
                for imode in range(basis.nmode)]
    specs = args.modes or list(DEFAULT_MODES)
    modes = []
    for spec in specs:
        try:
            modes.append(basis.select(spec))
        except ValueError as error:
            raise SystemExit("Bad mode specification %r: %s" % (spec, error))
    return modes


def _write_output(path: str, time, amplitudes: dict) -> None:
    names = list(amplitudes)
    if path.endswith(".npz"):
        np.savez_compressed(path, time_ps=time,
                            **{name: amplitudes[name] for name in names})
    else:
        columns = np.column_stack([time] + [amplitudes[name] for name in names])
        np.savetxt(path, columns, delimiter=",", fmt="%.8g",
                   header="time_ps," + ",".join(names), comments="")


def _print_anphon_input(args) -> int:
    from ase.io import read as ase_read

    from .evec import BOHR_TO_ANGSTROM
    from .geometry import commensurate_qpoints, supercell_matrix

    primitive = ase_read(args.primitive, format="vasp")
    supercell = ase_read(args.supercell, format="vasp")
    qpoints = commensurate_qpoints(
        supercell_matrix(primitive.cell[:], supercell.cell[:]))

    print("# anphon input to write the eigenvectors needed by modeproj.")
    print("# Add these blocks to your anphon input file (mode = phonons).")
    print("&cell")
    print("1.0")
    for vector in primitive.cell[:]:
        print("".join("%20.15f" % (component / BOHR_TO_ANGSTROM) for component in vector))
    print("/")
    print("&kpoint")
    print("0")
    for q in qpoints:
        print("%20.15f %20.15f %20.15f" % tuple(q))
    print("/")
    print("&analysis")
    print(" PRINTEVEC = 1")
    print("/")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "anphon-input":
        return _print_anphon_input(args)

    basis = _load_basis(args)

    if args.command == "modes":
        print(basis.qpoint_table())
        return 0

    if not os.path.exists(args.xdatcar):
        raise SystemExit("Cannot find the trajectory '%s'." % args.xdatcar)

    modes = _selected_modes(basis, args)
    trajectory = read_xdatcar(args.xdatcar, start=args.start, stop=args.stop,
                              stride=args.stride)
    check_consistency(trajectory, basis.symbols, basis.supercell_cell)

    if args.reference == "supercell":
        from ase.io import read as ase_read
        reference = ase_read(args.supercell, format="vasp").get_scaled_positions()
    else:
        reference = args.reference
    displacements = trajectory.displacements(reference)

    amplitudes = basis.project_modes(displacements, modes, normalize=args.normalize)
    time = (args.start + np.arange(trajectory.nframes) * args.stride) \
        * args.timestep / 1000.0

    if not args.quiet:
        print("\n%d frames, %d atoms, time step %.3f fs, %s reference"
              % (trajectory.nframes, trajectory.natoms, args.timestep, args.reference))
        print("%-12s %-10s %10s %10s %10s %10s"
              % ("mode", "q label", "w (cm-1)", "mean Q", "rms Q", "max |Q|"))
        for mode in modes:
            values = amplitudes[mode.name]
            print("%-12s %-10s %10.2f %10.3f %10.3f %10.3f"
                  % (mode.name, mode.label, mode.frequency_cm, values.mean(),
                     np.sqrt(np.mean(values ** 2)), np.max(np.abs(values))))
        print("Amplitudes in amu^(1/2) A%s."
              % (" (normalized per cell)" if args.normalize else ""))

    if args.out:
        _write_output(args.out, time, amplitudes)
        if not args.quiet:
            print("Wrote %s" % args.out)

    titles = {mode.name: mode.pretty for mode in modes}
    if args.plot:
        plot_amplitudes(time, amplitudes, window=args.window, ylim=args.ylim,
                        titles=titles, path=args.plot)
        if not args.quiet:
            print("Wrote %s" % args.plot)
    if args.histogram:
        plot_histograms(amplitudes, titles=titles, path=args.histogram)
        if not args.quiet:
            print("Wrote %s" % args.histogram)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
