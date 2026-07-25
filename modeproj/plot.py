"""Figure-3 style plots of mode amplitudes against time."""

from __future__ import annotations

import numpy as np

#: Colours used in Fig. 3 of npj Comput. Mater. 9, 154 (2023), cycled per panel.
DEFAULT_COLORS = ("green", "orange", "royalblue", "crimson", "purple", "teal")


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred running average with shrinking windows at both ends."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    window = min(window, values.size)
    kernel = np.ones(window)
    total = np.convolve(values, kernel, mode="same")
    count = np.convolve(np.ones_like(values), kernel, mode="same")
    return total / count


def plot_amplitudes(time, series: dict, window: int = 250, ylim=None,
                    colors=DEFAULT_COLORS, time_unit: str = "ps", titles=None,
                    path: str | None = None, figsize=None):
    """Plot one panel per mode: raw amplitude plus a running average.

    Parameters
    ----------
    time
        ``(nframes,)`` time axis.
    series
        ``{name: amplitudes}`` as returned by :meth:`ModeBasis.project_modes`.
    window
        Running-average window in frames (``250`` in the reference figure).
    ylim
        ``(low, high)`` amplitude limits, shared by all panels.  ``None`` makes a
        symmetric range around zero from the data.
    titles
        Optional ``{name: title}`` overrides, e.g. pretty mode labels.
    path
        When given, the figure is saved there instead of being returned only.

    Returns
    -------
    (figure, axes)
    """
    import matplotlib.pyplot as plt

    time = np.asarray(time)
    names = list(series)
    if not names:
        raise ValueError("Nothing to plot: no mode amplitudes given")

    if ylim is None:
        limit = 1.05 * max(np.max(np.abs(series[name])) for name in names)
        ylim = (-limit, limit)

    figsize = figsize or (7.0, 2.2 * len(names) + 0.6)
    fig, axes = plt.subplots(len(names), 1, figsize=figsize, sharex=True,
                             squeeze=False)
    axes = axes[:, 0]

    for index, (name, axis) in enumerate(zip(names, axes)):
        values = np.asarray(series[name], dtype=float)
        color = colors[index % len(colors)]
        axis.plot(time, values, lw=1, alpha=0.25, color=color)
        axis.plot(time, rolling_mean(values, window), lw=1.25, color=color)
        axis.axhline(0.0, lw=0.6, color="0.6", zorder=0)
        axis.set_ylabel("$Q$ (amu$^{1/2}$Å)")
        axis.set_ylim(*ylim)
        axis.set_xlim(time[0], time[-1])
        axis.set_title((titles or {}).get(name, name), fontsize=10, pad=4)

    axes[-1].set_xlabel("Time (%s)" % time_unit)
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig, axes


def plot_histograms(series: dict, bins: int = 80, colors=DEFAULT_COLORS,
                    titles=None, path: str | None = None):
    """Histogram of each mode amplitude: single-well vs double-well behaviour.

    A single peak at ``Q = 0`` means the mode fluctuates around the high-symmetry
    structure, while two peaks indicate an order-disorder (hopping) mode.
    """
    import matplotlib.pyplot as plt

    names = list(series)
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 2.6),
                             squeeze=False)
    axes = axes[0]
    for index, (name, axis) in enumerate(zip(names, axes)):
        values = np.asarray(series[name], dtype=float)
        axis.hist(values, bins=bins, color=colors[index % len(colors)], alpha=0.75)
        axis.set_xlabel("$Q$ (amu$^{1/2}$Å)")
        axis.set_title((titles or {}).get(name, name), fontsize=10)
    axes[0].set_ylabel("Counts")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig, axes
