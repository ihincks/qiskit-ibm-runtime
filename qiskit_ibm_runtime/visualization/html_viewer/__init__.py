# This code is part of Qiskit.
#
# (C) Copyright IBM 2025-2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Python bridge that produces a self-contained HTML viewer for circuit schedule timing data."""

from __future__ import annotations

import importlib.resources
import json
import tempfile
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..circuit_schedule import CircuitSchedule


def draw_circuit_schedule_timing_html(
    circuit_schedule: str | CircuitSchedule,
    included_channels: list | None = None,
    filter_readout_channels: bool = False,
    filter_barriers: bool = False,
    merge_common_instructions: bool = False,
    path: str | Path | None = None,
    open_in_browser: bool = False,
) -> str:
    """Return a self-contained HTML string for the interactive circuit schedule timing viewer.

    The viewer is rendered entirely in the browser using Canvas 2D — no Python runtime
    dependency after this function has been called.  Interactive features include
    mouse-wheel zoom on the x-axis (Shift+Wheel for y), drag-to-pan, double-click to
    reset the view, a collapsible sidebar with per-gate / per-channel / per-branch
    visibility toggles, a text search box with jump-to-next-hit, and an overview
    minimap.

    This function produces an HTML document equivalent to calling the standalone
    ``viewer.html`` file (shipped with the package in
    ``qiskit_ibm_runtime/visualization/html_viewer/``) with the CSV timing data
    and options pre-loaded — no paste/drop step is required when using this function.

    Args:
        circuit_schedule: The circuit schedule as a raw CSV string (as returned by the
            compiler and stored in ``ItemMetadata.scheduler_timing.timing``), or a
            :class:`~qiskit_ibm_runtime.visualization.CircuitSchedule` instance.
        included_channels: A list of channel names to include.  Also controls the
            y-axis display order: the first name in the list appears at the top of
            the plot.
        filter_readout_channels: If ``True``, hide all readout channels (those whose
            name starts with ``AWGR``) on initial render.  The user can toggle them
            back via the sidebar.
        filter_barriers: If ``True``, hide all barrier instructions on initial render.
        merge_common_instructions: If ``True``, merge temporally adjacent instructions
            of the same type (same branch / instruction / channel) into a single span.
        path: Optional filesystem path.  When provided the HTML is written to this
            file.  Useful for saving a snapshot or for Jupyter's ``IFrame`` display.
        open_in_browser: If ``True``, open the resulting HTML in the system default
            web browser.  When *path* is also given, that file is opened; otherwise
            a temporary file is written and opened.

    Returns:
        A self-contained HTML string.

    Example::

        from qiskit_ibm_runtime.visualization import draw_circuit_schedule_timing_html

        timing = result[0].metadata.scheduler_timing.timing
        html = draw_circuit_schedule_timing_html(
            timing,
            filter_readout_channels=True,
            open_in_browser=True,
        )
    """
    # Resolve the raw CSV string
    csv_text = _resolve_csv(circuit_schedule)

    # Build the JS options object
    js_options: dict = {}
    if included_channels is not None:
        js_options["includedChannels"] = included_channels
    if filter_readout_channels:
        js_options["filterReadoutChannels"] = True
    if filter_barriers:
        js_options["filterBarriers"] = True
    if merge_common_instructions:
        js_options["mergeCommonInstructions"] = True

    # Read static assets from package data
    pkg = importlib.resources.files(__name__.rsplit(".", 1)[0] + ".html_viewer")
    html_template = (pkg / "viewer.html").read_text(encoding="utf-8")
    js_source = (pkg / "viewer.js").read_text(encoding="utf-8")
    css_source = (pkg / "viewer.css").read_text(encoding="utf-8")

    # Build inline data block
    escaped_csv = json.dumps(csv_text)  # JSON-encode to safely embed in JS string literal
    data_block = (
        "<script>\n"
        f"  window.__CIRCUIT_SCHEDULE_TIMING__ = {escaped_csv};\n"
        f"  window.__CIRCUIT_SCHEDULE_OPTIONS__ = {json.dumps(js_options)};\n"
        "</script>"
    )

    # Assemble self-contained document
    html = html_template
    html = html.replace("__STYLE_SLOT__", f"<style>\n{css_source}\n</style>")
    html = html.replace("__DATA_SLOT__", data_block)
    html = html.replace("__SCRIPT_SLOT__", f"<script>\n{js_source}\n</script>")

    # Write to file if requested
    if path is not None:
        Path(path).write_text(html, encoding="utf-8")
        if open_in_browser:
            webbrowser.open(Path(path).resolve().as_uri())
    elif open_in_browser:
        # Write to a temp file so webbrowser.open gets a file:// URL
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(html)
            tmp_path = tmp.name
        webbrowser.open(Path(tmp_path).resolve().as_uri())

    return html


def _resolve_csv(circuit_schedule: str | CircuitSchedule) -> str:
    """Return the raw CSV timing string from either a str or CircuitSchedule."""
    # Import here to avoid circular imports at module load time
    from ..circuit_schedule import CircuitSchedule  # noqa: PLC0415

    if isinstance(circuit_schedule, str):
        return circuit_schedule
    if isinstance(circuit_schedule, CircuitSchedule):
        return _csv_from_circuit_schedule(circuit_schedule)
    raise ValueError(
        f"'circuit_schedule' must be a str or CircuitSchedule, got {type(circuit_schedule)}"
    )


def _csv_from_circuit_schedule(schedule: CircuitSchedule) -> str:
    """Re-serialise a CircuitSchedule back to its raw CSV text.

    The CircuitSchedule constructor parses the CSV into a NumPy array (adding a
    computed ``Finish`` column and a ``GateName`` column).  To recover the original
    six-column format we subtract Finish − Start to recover Duration.
    """
    import numpy as np  # noqa: PLC0415

    cs = schedule.circuit_scheduling
    t = schedule.type_to_idx
    rows = []
    for row in cs:
        branch = row[t["Branch"]]
        instruction = row[t["Instruction"]]
        channel = row[t["Channel"]]
        start = row[t["Start"]]
        duration = str(int(row[t["Finish"]]) - int(start))
        pulse = row[t["Pulse"]]
        rows.append(f"{branch},{instruction},{channel},{start},{duration},{pulse}")
    return "\n".join(rows)
