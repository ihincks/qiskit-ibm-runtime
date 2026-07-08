// This code is part of Qiskit.
//
// (C) Copyright IBM 2025-2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

/**
 * Circuit Schedule Timing Viewer
 *
 * Canvas 2D renderer for circuit schedule timing data. No external
 * dependencies — one plain script tag is enough.
 *
 * Data is supplied either inline (window.__CIRCUIT_SCHEDULE_TIMING__ set
 * by the Python bridge) or via the paste / drag-and-drop area shown when
 * the page first loads without data.
 */
(function () {
  "use strict";

  // =========================================================================
  // Constants
  // =========================================================================

  // Plotly qualitative palette — keep in sync with circuit_schedule.py
  const COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
  ];

  const READOUT_PREFIX = "AWGR"; // mirrors circuit_schedule.py:33
  const BARRIER_GATE = "barrier";

  // Canvas padding (pixels)
  const PAD_LEFT = 120; // space for y-axis labels
  const PAD_RIGHT = 20;
  const PAD_TOP = 24;
  const PAD_BOTTOM = 42; // space for x-axis ticks + title

  const ROW_HEIGHT_PX = 60; // pixels per data row at initial zoom
  const CANVAS_MAX_H = 900; // cap height; user pans y if more channels
  const MM_H = 50; // minimap height
  const MIN_LABEL_PX = 32; // minimum rect width to draw a text label
  const MIN_LABEL_CYCLES = 4; // typical instruction length; used for text LOD
  const MIN_STROKE_PX = 2; // minimum rect width to draw a border
  const SIDEBAR_W = 230; // sidebar width in pixels

  // Branch y-offsets (mirrors circuit_schedule.py:244–288)
  const BRANCH_Y = {
    main: { low: -0.4, high: 0.4, annY: 0.0 },
    then: { low: 0.0, high: 0.4, annY: 0.25 },
    else: { low: -0.4, high: 0.0, annY: -0.25 },
  };
  const BARRIER_PAD = 0.05; // mirrors circuit_schedule.py:305–308

  // Zero-duration (shift_phase) y-center per branch (circuit_schedule.py:267–288)
  const ZERO_DUR_CENTER = { main: 0, then: 0.2, else: -0.2 };
  const ZERO_DUR_HALF = 0.2;

  const ZOOM_FACTOR = 1.2; // per wheel tick
  const MIN_X_SPAN = 2; // don't zoom in past 2 cycles

  // =========================================================================
  // Mutable state
  // =========================================================================

  let rawCsv = "";
  let opts = {
    includedChannels: null,
    filterReadoutChannels: false,
    filterBarriers: false,
    mergeCommonInstructions: false,
  };

  // --- Columnar typed-array store ---
  let N = 0; // number of instructions
  let starts; // Int32Array [N]
  let finishes; // Int32Array [N]
  let branchIds; // Uint8Array [N]  0=main 1=then 2=else
  let gateIds; // Uint16Array [N]
  let channelIds; // Uint16Array [N]
  let instructionIds; // Uint16Array [N]
  let pulseNameIds; // Uint16Array [N]
  let isZeroDuration; // Uint8Array [N]  1 = shift_phase glyph

  // String tables
  let gates = []; // gateId -> gate name
  let channels = []; // channelId -> channel name (sorted)
  let instructions = []; // instructionId -> full instruction string
  let pulseNames = []; // pulseNameId -> pulse string
  const BRANCHES = ["main", "then", "else"];

  // Precomputed per-gate instruction index lists
  let gateInstructions = []; // gateId -> Int32Array of instruction indices

  // Per-row buckets (rebuilt lazily when bucketsDirty)
  let rowInstances = []; // row -> Int32Array of instruction indices (sorted by start)
  let rowStarts = []; // row -> Int32Array of starts[idx] parallel to rowInstances
  let rowMaxDur = new Int32Array(0); // row -> max (finishes[i] - starts[i]) in that row
  let bucketsDirty = true;

  // Minimap bitmap cache (rebuilt only when data / filters / size change)
  let mmRectsCache = null; // HTMLCanvasElement offscreen bitmap
  let mmCacheDirty = true;

  // Precomputed lowercase haystack per instruction (for search)
  let searchText = [];

  let colorMap = {}; // gateId -> hex color
  let maxTime = 0;

  // --- Filter / visibility ---
  let gateVisible = new Uint8Array(0); // per gateId
  let branchVisible = new Uint8Array([1, 1, 1]); // main / then / else
  let chanVisible = new Uint8Array(0); // per channelId
  let filterBarrier = false;
  let mergeActive = false;
  let visible = new Uint8Array(0); // per instruction computed bitmask

  // Visual row ordering
  let channelOrder = []; // display order (index 0 = bottom row, last = top row)
  let channelRow = new Int16Array(0); // channelId -> row index (-1 = hidden)

  // --- Viewport ---
  let xMin = 0,
    xMax = 100; // data x range
  let yMin = -0.6,
    yMax = 1.4; // data y range

  // --- Search ---
  let searchQuery = "";
  let searchHits = new Uint8Array(0); // per instruction
  let searchCursor = 0; // index into instructions for jump-to-next

  // --- Hover ---
  let hoveredIdx = -1;
  let hoveredGate = -1; // gateId of currently hovered instruction (or -1)

  // --- Solo gate ---
  let soloGate = -1; // gateId currently soloed (or -1)

  // --- Drag state (main canvas) ---
  // dragMode: "none" | "pan" | "zoom"
  let dragMode = "none";
  let dragStartX = 0; // clientX at mousedown
  let dragXMin0 = 0;
  let dragXMax0 = 0;
  let dragMoved = false;
  let zoomBox = null; // { startCx, endCx } in canvas pixels, or null

  // --- Drag state (minimap) ---
  let mmDragMode = "none"; // "none" | "pan" | "zoom"
  let mmDragStartX = 0;
  let mmXMin0 = 0;
  let mmXMax0 = 0;
  let mmDragMoved = false;
  let mmZoomBox = null; // { startMx, endMx } in minimap pixels, or null

  // --- Keyboard state ---
  let shiftHeld = false;

  // --- Sidebar expand/collapse state (persists across buildSidebar rebuilds) ---
  let expandedSections = new Set(); // keyed by: "qubit"|"readout"|"broadcast"|"other"|"source"|"help"
  let sidebarBuiltOnce = false;     // seeds "help" into expandedSections on first build

  let dirty = false;

  // --- DOM references ---
  let mainCanvas, mainCtx, mmCanvas, mmCtx, tooltip;

  // =========================================================================
  // CSV Parser  (mirrors circuit_schedule.py _parse, lines 103–137)
  //
  // Pass { strict: true } to throw RangeError on malformed rows instead of
  // silently skipping them (used by the Source editor's Apply button).
  // =========================================================================
  function parseCsv(csv, { strict = false } = {}) {
    const rows = [];
    const lines = csv.split("\n");
    for (let lineNo = 0; lineNo < lines.length; lineNo++) {
      const trimmed = lines[lineNo].trim();
      if (!trimmed) continue;
      // Python skips lines where words[0] contains 'shift_phase' (line 117-118)
      if (trimmed.startsWith("shift_phase")) continue;
      const words = trimmed.split(",");
      if (words.length !== 6) {
        if (strict) {
          throw new RangeError(
            `Line ${lineNo + 1}: expected 6 comma-separated fields, got ${words.length}`
          );
        }
        continue;
      }

      const t0 = parseInt(words[3], 10);
      const dur = parseInt(words[4], 10);
      if (strict && (isNaN(t0) || isNaN(dur))) {
        throw new RangeError(
          `Line ${lineNo + 1}: non-numeric start (${words[3]}) or duration (${words[4]})`
        );
      }
      const pulse = words[5].trim();
      rows.push({
        branch: words[0].trim(),
        instruction: words[1].trim(),
        channel: words[2].trim(),
        start: t0,
        finish: t0 + dur,
        pulse,
        gateName: words[1].trim().split("_")[0], // circuit_schedule.py:132
        isZero: pulse === "shift_phase" ? 1 : 0,
      });
    }
    return rows;
  }

  // =========================================================================
  // Merge common consecutive instructions (mirrors circuit_schedule.py:197–242)
  // =========================================================================
  function mergeInstructions(rows) {
    const groups = new Map();
    for (const r of rows) {
      const key = `${r.branch}\0${r.instruction}\0${r.channel}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }
    const merged = [];
    for (const grp of groups.values()) {
      if (grp.length === 1) {
        merged.push(grp[0]);
        continue;
      }
      grp.sort((a, b) => a.start - b.start);
      const acc = [Object.assign({}, grp[0])];
      for (let k = 1; k < grp.length; k++) {
        const prev = acc[acc.length - 1];
        if (grp[k].start === prev.finish) {
          prev.finish = grp[k].finish; // temporally adjacent → merge
        } else {
          acc.push(Object.assign({}, grp[k]));
        }
      }
      merged.push(...acc);
    }
    return merged;
  }

  // =========================================================================
  // Build columnar store from parsed rows
  // =========================================================================
  function buildStore(rows) {
    // Intern strings
    const gateSet = new Set(),
      chanSet = new Set(),
      instrSet = new Set(),
      pulseSet = new Set();
    for (const r of rows) {
      gateSet.add(r.gateName);
      chanSet.add(r.channel);
      instrSet.add(r.instruction);
      pulseSet.add(r.pulse);
    }
    gates = [...gateSet].sort();
    channels = [...chanSet].sort();
    instructions = [...instrSet].sort();
    pulseNames = [...pulseSet].sort();

    const gIdx = Object.fromEntries(gates.map((g, i) => [g, i]));
    const cIdx = Object.fromEntries(channels.map((c, i) => [c, i]));
    const iIdx = Object.fromEntries(instructions.map((s, i) => [s, i]));
    const pIdx = Object.fromEntries(pulseNames.map((p, i) => [p, i]));
    const bIdx = { main: 0, then: 1, else: 2 };

    N = rows.length;
    starts = new Int32Array(N);
    finishes = new Int32Array(N);
    branchIds = new Uint8Array(N);
    gateIds = new Uint16Array(N);
    channelIds = new Uint16Array(N);
    instructionIds = new Uint16Array(N);
    pulseNameIds = new Uint16Array(N);
    isZeroDuration = new Uint8Array(N);

    for (let i = 0; i < N; i++) {
      const r = rows[i];
      starts[i] = r.start;
      finishes[i] = r.finish;
      branchIds[i] = bIdx[r.branch] ?? 0;
      gateIds[i] = gIdx[r.gateName] ?? 0;
      channelIds[i] = cIdx[r.channel] ?? 0;
      instructionIds[i] = iIdx[r.instruction] ?? 0;
      pulseNameIds[i] = pIdx[r.pulse] ?? 0;
      isZeroDuration[i] = r.isZero;
    }

    maxTime = N > 0 ? Math.max(...finishes) : 0;

    // Colour map: one colour per unique gate, cycling through palette
    colorMap = {};
    for (let g = 0; g < gates.length; g++) {
      colorMap[g] = COLORS[g % COLORS.length];
    }

    // Precompute per-gate instruction index lists (avoids O(N*G) in render)
    gateInstructions = gates.map(() => []);
    for (let i = 0; i < N; i++) {
      gateInstructions[gateIds[i]].push(i);
    }
    gateInstructions = gateInstructions.map((arr) => new Int32Array(arr));

    // Reset per-item arrays
    gateVisible = new Uint8Array(gates.length).fill(1);
    chanVisible = new Uint8Array(channels.length).fill(1);
    visible = new Uint8Array(N).fill(1);
    searchHits = new Uint8Array(N).fill(0);
    channelRow = new Int16Array(channels.length).fill(-1);

    // Precomputed lowercase haystack for search (one .toLowerCase per row,
    // not per keystroke). `\0` separator keeps fragments from bleeding.
    searchText = new Array(N);
    for (let i = 0; i < N; i++) {
      searchText[i] = (
        gates[gateIds[i]] +
        "\0" +
        channels[channelIds[i]] +
        "\0" +
        instructions[instructionIds[i]]
      ).toLowerCase();
    }

    // Data changed: invalidate row buckets and minimap cache
    bucketsDirty = true;
    mmCacheDirty = true;

    // Reset interaction state
    soloGate = -1;
  }

  // =========================================================================
  // Apply initial Python-supplied options (called once after buildStore)
  // =========================================================================
  function applyInitialOptions() {
    if (opts.filterReadoutChannels) {
      for (let c = 0; c < channels.length; c++) {
        if (channels[c].startsWith(READOUT_PREFIX)) chanVisible[c] = 0;
      }
    }
    if (opts.filterBarriers) {
      filterBarrier = true;
    }
    // includedChannels channel filtering is enforced in recomputeVisible via chanVisible
    if (opts.includedChannels && opts.includedChannels.length) {
      const allowed = new Set(opts.includedChannels);
      for (let c = 0; c < channels.length; c++) {
        if (!allowed.has(channels[c])) chanVisible[c] = 0;
      }
    }
  }

  // =========================================================================
  // Recompute per-instruction visible flags and channel display order
  // (called after any filter change)
  // =========================================================================
  function recomputeVisible() {
    // Determine display order of visible channels
    // mirrors circuit_schedule.py:187–191 (included_channels ordering, reversed)
    let ordered = [];
    if (opts.includedChannels && opts.includedChannels.length) {
      const rev = [...opts.includedChannels].reverse();
      for (const name of rev) {
        const c = channels.indexOf(name);
        if (c >= 0 && chanVisible[c]) ordered.push(c);
      }
      // Append any remaining visible channels not in includedChannels
      for (let c = 0; c < channels.length; c++) {
        if (chanVisible[c] && !ordered.includes(c)) ordered.push(c);
      }
    } else {
      // Alphabetical, same as Python np.unique sort
      for (let c = 0; c < channels.length; c++) {
        if (chanVisible[c]) ordered.push(c);
      }
    }
    channelOrder = ordered;

    // Row 0 = bottom, row n-1 = top (matches Plotly y-axis convention)
    channelRow.fill(-1);
    for (let r = 0; r < ordered.length; r++) {
      channelRow[ordered[r]] = r;
    }

    // Per-instruction visibility
    for (let i = 0; i < N; i++) {
      const c = channelIds[i];
      const g = gateIds[i];
      const b = branchIds[i];
      const isBarrier = gates[g] === BARRIER_GATE;
      visible[i] =
        chanVisible[c] &&
        gateVisible[g] &&
        branchVisible[b] &&
        channelRow[c] >= 0 &&
        !(filterBarrier && isBarrier)
          ? 1
          : 0;
    }

    // Update y view range to cover all visible rows
    const nRows = channelOrder.length;
    yMin = -0.6;
    yMax = nRows > 0 ? nRows - 0.4 : 1.0;

    // Invalidate row buckets and minimap cache (channel order or visibility
    // may have changed, and either can affect what the minimap shows).
    bucketsDirty = true;
    mmCacheDirty = true;

    markDirty();
  }

  // =========================================================================
  // Coordinate transforms (data ↔ canvas pixels)
  // =========================================================================
  function plotW() {
    return mainCanvas.width - PAD_LEFT - PAD_RIGHT;
  }
  function plotH() {
    return mainCanvas.height - PAD_TOP - PAD_BOTTOM;
  }

  function dataXtoCx(dx) {
    return PAD_LEFT + ((dx - xMin) / (xMax - xMin)) * plotW();
  }
  function dataYtoCy(dy) {
    // y increases upward in data space; flip for canvas
    return PAD_TOP + (1 - (dy - yMin) / (yMax - yMin)) * plotH();
  }
  function cxToDataX(cx) {
    return xMin + ((cx - PAD_LEFT) / plotW()) * (xMax - xMin);
  }
  function cyToDataY(cy) {
    return yMin + (1 - (cy - PAD_TOP) / plotH()) * (yMax - yMin);
  }

  // =========================================================================
  // Per-instruction geometry helpers
  // =========================================================================
  function getRect(i) {
    // Returns {x, y, w, h} in canvas pixels for a finite-duration instruction
    const b = BRANCHES[branchIds[i]];
    const bOff = BRANCH_Y[b] || BRANCH_Y.main;
    const row = channelRow[channelIds[i]];
    let yLow = row + bOff.low;
    let yHigh = row + bOff.high;
    if (gates[gateIds[i]] === BARRIER_GATE) {
      yLow -= BARRIER_PAD;
      yHigh += BARRIER_PAD;
    }
    const px1 = dataXtoCx(starts[i]);
    const px2 = dataXtoCx(finishes[i]);
    const py1 = dataYtoCy(yHigh); // yHigh → smaller cy (nearer canvas top)
    const py2 = dataYtoCy(yLow);
    return { x: px1, y: py1, w: Math.max(px2 - px1, 1), h: py2 - py1 };
  }

  function getDiamond(i) {
    // Returns centre & half-extents (pixels) for a zero-duration instruction
    const b = BRANCHES[branchIds[i]];
    const yCtr = (ZERO_DUR_CENTER[b] ?? 0) + channelRow[channelIds[i]];
    const cx = dataXtoCx(starts[i]);
    const cy = dataYtoCy(yCtr);
    const halfW = Math.max(3, dataXtoCx(starts[i] + 1) - dataXtoCx(starts[i]));
    const halfH = Math.max(
      3,
      dataYtoCy(yCtr - ZERO_DUR_HALF) - dataYtoCy(yCtr + ZERO_DUR_HALF)
    );
    return { cx, cy, halfW, halfH };
  }

  // =========================================================================
  // Row buckets + binary search (viewport culling for O(N) render passes)
  // =========================================================================
  function rebuildRowBuckets() {
    const nRows = channelOrder.length;
    // Bucket ALL instructions per row (do not filter by visible[]) so that
    // per-frame `visible` toggles don't require a bucket rebuild.
    const buckets = new Array(nRows);
    for (let r = 0; r < nRows; r++) buckets[r] = [];
    for (let i = 0; i < N; i++) {
      const row = channelRow[channelIds[i]];
      if (row < 0) continue;
      buckets[row].push(i);
    }
    rowInstances = new Array(nRows);
    rowStarts = new Array(nRows);
    rowMaxDur = new Int32Array(nRows);
    for (let r = 0; r < nRows; r++) {
      const arr = buckets[r];
      arr.sort((a, b) => starts[a] - starts[b]);
      const idxs = new Int32Array(arr.length);
      const sts = new Int32Array(arr.length);
      let maxDur = 0;
      for (let k = 0; k < arr.length; k++) {
        const i = arr[k];
        idxs[k] = i;
        sts[k] = starts[i];
        const dur = finishes[i] - starts[i];
        if (dur > maxDur) maxDur = dur;
      }
      rowInstances[r] = idxs;
      rowStarts[r] = sts;
      rowMaxDur[r] = maxDur;
    }
    bucketsDirty = false;
  }

  // First index k such that arr[k] >= v (or arr.length if none)
  function lowerBound(arr, v) {
    let lo = 0,
      hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (arr[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  // First index k such that arr[k] > v (or arr.length if none)
  function upperBound(arr, v) {
    let lo = 0,
      hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (arr[mid] <= v) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  // =========================================================================
  // Rendering
  // =========================================================================
  function markDirty() {
    if (dirty) return;
    dirty = true;
    requestAnimationFrame(renderFrame);
  }

  function renderFrame() {
    if (!dirty) return;
    dirty = false;

    // Lazy row-bucket rebuild — invalidated by recomputeVisible / buildStore.
    if (bucketsDirty) rebuildRowBuckets();

    const W = mainCanvas.width,
      H = mainCanvas.height;
    mainCtx.clearRect(0, 0, W, H);
    mainCtx.fillStyle = "#ffffff";
    mainCtx.fillRect(0, 0, W, H);

    drawGrid(mainCtx);
    drawInstructions(mainCtx);
    drawAxes(mainCtx);
    renderMinimap();
  }

  function drawGrid(ctx) {
    const nRows = channelOrder.length;
    if (nRows === 0) return;
    // Only iterate rows currently in the y-viewport. `dataYtoCy` flips y, so
    // the row at data-y = yMax is nearer the canvas top; use min/max after
    // conversion in both directions to be safe.
    const dyTop = cyToDataY(PAD_TOP);
    const dyBot = cyToDataY(PAD_TOP + plotH());
    const rFirst = Math.max(0, Math.floor(Math.min(dyTop, dyBot)));
    const rLast = Math.min(nRows - 1, Math.ceil(Math.max(dyTop, dyBot)));
    const xRight = mainCanvas.width - PAD_RIGHT;
    ctx.strokeStyle = "rgba(38,38,38,0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let r = rFirst; r <= rLast; r++) {
      const cy = dataYtoCy(r);
      ctx.moveTo(PAD_LEFT, cy);
      ctx.lineTo(xRight, cy);
    }
    ctx.stroke();
  }

  function drawInstructions(ctx) {
    // Clip all instruction drawing to the plot area so nothing bleeds over
    // the y-axis label strip (left of PAD_LEFT) or past the right edge.
    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD_LEFT, PAD_TOP, plotW(), plotH());
    ctx.clip();

    // Row-viewport prelude: only iterate rows currently on screen. `dataYtoCy`
    // flips y, so use min/max on both converted values.
    const nRows = channelOrder.length;
    const pw = plotW();
    const pxPerCycle = pw / (xMax - xMin);
    let rFirst = 0,
      rLast = -1;
    if (nRows > 0) {
      const dyTop = cyToDataY(PAD_TOP);
      const dyBot = cyToDataY(PAD_TOP + plotH());
      rFirst = Math.max(0, Math.floor(Math.min(dyTop, dyBot)));
      rLast = Math.min(nRows - 1, Math.ceil(Math.max(dyTop, dyBot)));
    }

    // Pass 1 – filled rectangles, grouped by gate to minimise fillStyle changes.
    // Within each gate, iterate row buckets (viewport-culled by binary search).
    // We collect per-gate indices in a first pass so we can preserve the
    // gate-order colour grouping while touching only viewport instances.
    // `xMin - rowMaxDur[r]` widens the low bound to catch instructions whose
    // `start < xMin < finish` (straddling the left edge).
    const perGate = new Array(gates.length);
    for (let g = 0; g < gates.length; g++) perGate[g] = null;
    for (let r = rFirst; r <= rLast; r++) {
      const idxs = rowInstances[r];
      const sts = rowStarts[r];
      if (!idxs || idxs.length === 0) continue;
      const lo = lowerBound(sts, xMin - rowMaxDur[r]);
      const hi = upperBound(sts, xMax);
      for (let k = lo; k < hi; k++) {
        const i = idxs[k];
        if (!visible[i] || isZeroDuration[i]) continue;
        if (finishes[i] < xMin) continue; // exact left-edge cull
        const g = gateIds[i];
        (perGate[g] || (perGate[g] = [])).push(i);
      }
    }
    for (let g = 0; g < gates.length; g++) {
      const arr = perGate[g];
      if (!arr) continue;
      ctx.fillStyle = colorMap[g];
      for (let k = 0; k < arr.length; k++) {
        const r = getRect(arr[k]);
        ctx.fillRect(r.x, r.y, r.w, r.h);
      }
    }

    // Pass 2 – rectangle borders (single colour sweep) using the same
    // per-gate collection so we don't re-scan buckets.
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 0.5;
    for (let g = 0; g < gates.length; g++) {
      const arr = perGate[g];
      if (!arr) continue;
      for (let k = 0; k < arr.length; k++) {
        const rct = getRect(arr[k]);
        if (rct.w < MIN_STROKE_PX) continue;
        ctx.strokeRect(rct.x, rct.y, rct.w, rct.h);
      }
    }

    // Pass 3 – zero-duration diamonds (drawn last so they sit on top).
    // Iterate row buckets with a ±4 cycle slack on the x window.
    ctx.lineWidth = 0.5;
    const perGateDiamonds = new Array(gates.length);
    for (let g = 0; g < gates.length; g++) perGateDiamonds[g] = null;
    for (let r = rFirst; r <= rLast; r++) {
      const idxs = rowInstances[r];
      const sts = rowStarts[r];
      if (!idxs || idxs.length === 0) continue;
      const lo = lowerBound(sts, xMin - 4);
      const hi = upperBound(sts, xMax + 4);
      for (let k = lo; k < hi; k++) {
        const i = idxs[k];
        if (!visible[i] || !isZeroDuration[i]) continue;
        const g = gateIds[i];
        (perGateDiamonds[g] || (perGateDiamonds[g] = [])).push(i);
      }
    }
    for (let g = 0; g < gates.length; g++) {
      const arr = perGateDiamonds[g];
      if (!arr) continue;
      ctx.fillStyle = colorMap[g];
      ctx.strokeStyle = "#000000";
      for (let k = 0; k < arr.length; k++) {
        const { cx, cy, halfW, halfH } = getDiamond(arr[k]);
        ctx.beginPath();
        ctx.moveTo(cx, cy - halfH);
        ctx.lineTo(cx + halfW, cy);
        ctx.lineTo(cx, cy + halfH);
        ctx.lineTo(cx - halfW, cy);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }

    // Pass 4 – text labels. Level-of-detail: when a typical instruction spans
    // fewer than MIN_LABEL_PX pixels at the current zoom, skip the pass. Text
    // rendering is the single most expensive per-frame cost at N=100K.
    if (pxPerCycle * MIN_LABEL_CYCLES >= MIN_LABEL_PX) {
      ctx.fillStyle = "#000000";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let g = 0; g < gates.length; g++) {
        const arr = perGate[g];
        if (!arr) continue;
        for (let k = 0; k < arr.length; k++) {
          const i = arr[k];
          const rct = getRect(i);
          if (rct.w < MIN_LABEL_PX) continue;
          const label = gates[gateIds[i]] + "_" + pulseNames[pulseNameIds[i]];
          ctx.fillText(label, rct.x + rct.w / 2, rct.y + rct.h / 2, rct.w - 4);
        }
      }
    }

    // Pass 5 – search highlights (row-bucket sweep with slack for diamonds)
    if (searchQuery) {
      ctx.strokeStyle = "#FF6600";
      ctx.lineWidth = 2;
      for (let r = rFirst; r <= rLast; r++) {
        const idxs = rowInstances[r];
        const sts = rowStarts[r];
        if (!idxs || idxs.length === 0) continue;
        const lo = lowerBound(sts, xMin - Math.max(4, rowMaxDur[r]));
        const hi = upperBound(sts, xMax + 4);
        for (let k = lo; k < hi; k++) {
          const i = idxs[k];
          if (!searchHits[i] || !visible[i]) continue;
          if (isZeroDuration[i]) {
            if (starts[i] < xMin - 4 || starts[i] > xMax + 4) continue;
            const { cx, cy, halfW, halfH } = getDiamond(i);
            ctx.strokeRect(cx - halfW - 2, cy - halfH - 2, halfW * 2 + 4, halfH * 2 + 4);
          } else {
            if (finishes[i] < xMin || starts[i] > xMax) continue;
            const rct = getRect(i);
            ctx.strokeRect(rct.x - 1, rct.y - 1, rct.w + 2, rct.h + 2);
          }
        }
      }
    }

    // Pass 5.5 – gate-type hover highlights (all visible instances of the
    // hovered gate). Still uses gateInstructions[hoveredGate] since that
    // list is typically small per-gate; add row-viewport check for large gates.
    if (hoveredGate >= 0) {
      ctx.strokeStyle = "#FF6600";
      ctx.lineWidth = 2;
      for (const i of gateInstructions[hoveredGate]) {
        if (!visible[i]) continue;
        const row = channelRow[channelIds[i]];
        if (row < rFirst || row > rLast) continue;
        if (isZeroDuration[i]) {
          if (starts[i] < xMin - 4 || starts[i] > xMax + 4) continue;
          const { cx, cy, halfW, halfH } = getDiamond(i);
          ctx.strokeRect(cx - halfW - 2, cy - halfH - 2, halfW * 2 + 4, halfH * 2 + 4);
        } else {
          if (finishes[i] < xMin || starts[i] > xMax) continue;
          const rct = getRect(i);
          ctx.strokeRect(rct.x - 1, rct.y - 1, rct.w + 2, rct.h + 2);
        }
      }
    }

    // Pass 6 – hovered item highlight (white inner ring on the specific item)
    if (hoveredIdx >= 0 && visible[hoveredIdx]) {
      const i = hoveredIdx;
      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      if (isZeroDuration[i]) {
        const { cx, cy, halfW, halfH } = getDiamond(i);
        ctx.beginPath();
        ctx.moveTo(cx, cy - halfH + 2);
        ctx.lineTo(cx + halfW - 2, cy);
        ctx.lineTo(cx, cy + halfH - 2);
        ctx.lineTo(cx - halfW + 2, cy);
        ctx.closePath();
        ctx.stroke();
      } else {
        const r = getRect(i);
        ctx.strokeRect(r.x + 1, r.y + 1, r.w - 2, r.h - 2);
      }
    }

    // Pass 7 – zoom selection band (drawn last so it sits on top of everything)
    if (zoomBox) {
      const left = Math.min(zoomBox.startCx, zoomBox.endCx);
      const right = Math.max(zoomBox.startCx, zoomBox.endCx);
      if (right > left) {
        ctx.fillStyle = "rgba(80,80,220,0.12)";
        ctx.fillRect(left, PAD_TOP, right - left, plotH());
        ctx.strokeStyle = "rgba(80,80,220,0.75)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(left, PAD_TOP);
        ctx.lineTo(left, PAD_TOP + plotH());
        ctx.moveTo(right, PAD_TOP);
        ctx.lineTo(right, PAD_TOP + plotH());
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    ctx.restore(); // end plot-area clip
  }

  function drawAxes(ctx) {
    const pw = plotW(),
      ph = plotH();

    // Plot border
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD_LEFT, PAD_TOP, pw, ph);

    // X-axis ticks and labels
    ctx.fillStyle = "#333333";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    const xRange = xMax - xMin;
    const targetTicks = Math.max(4, Math.floor(pw / 80));
    const rawStep = xRange / targetTicks;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const nice = [1, 2, 5, 10];
    let step = nice.find((n) => n * mag >= rawStep) * mag;
    if (!step || step <= 0) step = rawStep;

    const firstTick = Math.ceil(xMin / step) * step;
    ctx.beginPath();
    for (let t = firstTick; t <= xMax + step * 0.01; t += step) {
      const cx = dataXtoCx(t);
      ctx.moveTo(cx, PAD_TOP + ph);
      ctx.lineTo(cx, PAD_TOP + ph + 5);
      ctx.fillText(Math.round(t).toString(), cx, PAD_TOP + ph + 7);
    }
    ctx.strokeStyle = "#333333";
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // X-axis title
    ctx.font = "12px sans-serif";
    ctx.textBaseline = "bottom";
    ctx.fillText(
      "Cycles",
      PAD_LEFT + pw / 2,
      mainCanvas.height - 2
    );

    // Y-axis labels (channel names)
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.font = "11px sans-serif";
    ctx.beginPath();
    for (let r = 0; r < channelOrder.length; r++) {
      const name = channels[channelOrder[r]];
      const cy = dataYtoCy(r);
      if (cy < PAD_TOP || cy > PAD_TOP + ph) continue; // outside viewport
      ctx.moveTo(PAD_LEFT, cy);
      ctx.lineTo(PAD_LEFT - 4, cy);
      ctx.fillText(name, PAD_LEFT - 6, cy);
    }
    ctx.strokeStyle = "#333333";
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Y-axis title
    ctx.save();
    ctx.translate(12, PAD_TOP + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.font = "12px sans-serif";
    ctx.fillText("Channels", 0, 0);
    ctx.restore();

    // Plot title
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.font = "bold 15px sans-serif";
    ctx.fillStyle = "#000000";
    ctx.fillText("Payload Schedule", PAD_LEFT + pw / 2, PAD_TOP - 4);
  }

  // =========================================================================
  // Minimap
  // =========================================================================

  // Paint the O(N) rects layer into the offscreen cache. Only re-runs when
  // filters/data/size change; per-frame pan/zoom just blits the cache below.
  function refreshMinimapCache() {
    if (!mmCanvas) return;
    const mw = mmCanvas.width,
      mh = mmCanvas.height;
    if (!mmRectsCache) {
      mmRectsCache = document.createElement("canvas");
    }
    if (mmRectsCache.width !== mw) mmRectsCache.width = mw;
    if (mmRectsCache.height !== mh) mmRectsCache.height = mh;
    const cctx = mmRectsCache.getContext("2d");
    cctx.clearRect(0, 0, mw, mh);
    cctx.fillStyle = "#f5f5f5";
    cctx.fillRect(0, 0, mw, mh);

    const nRows = channelOrder.length;
    if (!nRows || !maxTime) {
      mmCacheDirty = false;
      return;
    }

    const scaleX = mw / (maxTime + 1);
    const scaleY = mh / nRows;

    for (let g = 0; g < gates.length; g++) {
      cctx.fillStyle = colorMap[g];
      for (const i of gateInstructions[g]) {
        if (!visible[i] || isZeroDuration[i]) continue;
        const row = channelRow[channelIds[i]];
        if (row < 0) continue;
        const mx = starts[i] * scaleX;
        const mw2 = Math.max(1, (finishes[i] - starts[i]) * scaleX);
        // Row 0 = bottom; flip for canvas
        const my = (nRows - 1 - row) * scaleY + 0.1 * scaleY;
        const mh2 = Math.max(1, 0.8 * scaleY);
        cctx.fillRect(mx, my, mw2, mh2);
      }
    }
    mmCacheDirty = false;
  }

  function renderMinimap() {
    if (!mmCanvas) return;
    const mw = mmCanvas.width,
      mh = mmCanvas.height;

    // Rebuild the cached rects layer if invalidated or size changed.
    if (
      mmCacheDirty ||
      !mmRectsCache ||
      mmRectsCache.width !== mw ||
      mmRectsCache.height !== mh
    ) {
      refreshMinimapCache();
    }

    // Blit cached rects layer (single drawImage — O(1) per frame).
    mmCtx.clearRect(0, 0, mw, mh);
    if (mmRectsCache) mmCtx.drawImage(mmRectsCache, 0, 0);

    // Viewport indicator
    const vx1 = (xMin / (maxTime + 1)) * mw;
    const vx2 = (xMax / (maxTime + 1)) * mw;
    mmCtx.fillStyle = "rgba(80,80,220,0.18)";
    mmCtx.fillRect(vx1, 0, vx2 - vx1, mh);
    mmCtx.strokeStyle = "#4455cc";
    mmCtx.lineWidth = 1.5;
    mmCtx.strokeRect(vx1, 0, vx2 - vx1, mh);

    // Zoom-selection band on minimap
    if (mmZoomBox) {
      const left = Math.min(mmZoomBox.startMx, mmZoomBox.endMx);
      const right = Math.max(mmZoomBox.startMx, mmZoomBox.endMx);
      if (right > left) {
        mmCtx.fillStyle = "rgba(80,80,220,0.25)";
        mmCtx.fillRect(left, 0, right - left, mh);
        mmCtx.save();
        mmCtx.strokeStyle = "rgba(80,80,220,0.9)";
        mmCtx.lineWidth = 1;
        mmCtx.setLineDash([3, 3]);
        mmCtx.beginPath();
        mmCtx.moveTo(left, 0);
        mmCtx.lineTo(left, mh);
        mmCtx.moveTo(right, 0);
        mmCtx.lineTo(right, mh);
        mmCtx.stroke();
        mmCtx.restore();
      }
    }

    // Border
    mmCtx.strokeStyle = "#888888";
    mmCtx.lineWidth = 1;
    mmCtx.strokeRect(0, 0, mw, mh);
  }

  // =========================================================================
  // Zoom helper — zoom x-axis around a data-space anchor point
  // =========================================================================
  function zoomX(anchorData, factor) {
    const newMin = anchorData + (xMin - anchorData) * factor;
    const newMax = anchorData + (xMax - anchorData) * factor;
    if (newMax - newMin >= MIN_X_SPAN) {
      xMin = newMin;
      xMax = newMax;
      clampX();
      markDirty();
    }
  }

  // =========================================================================
  // Interaction — main canvas
  // =========================================================================
  function onWheel(e) {
    e.preventDefault();
    const rect = mainCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;

    if (e.shiftKey) {
      // Zoom y anchored at cursor
      const factor = e.deltaY > 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
      const cy = e.clientY - rect.top;
      const anchorY = cyToDataY(cy);
      yMin = anchorY + (yMin - anchorY) * factor;
      yMax = anchorY + (yMax - anchorY) * factor;
      markDirty();
    } else {
      // Zoom x anchored at cursor
      const factor = e.deltaY > 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
      zoomX(cxToDataX(cx), factor);
    }
  }

  function clampX() {
    const span = xMax - xMin;
    if (xMin < 0) {
      xMin = 0;
      xMax = Math.max(span, MIN_X_SPAN);
    }
    const limit = maxTime + 1;
    if (xMax > limit) {
      xMax = limit;
      xMin = Math.max(0, limit - span);
    }
  }

  function onMouseDown(e) {
    if (e.button !== 0) return;
    dragMode = e.shiftKey ? "pan" : "zoom";
    dragStartX = e.clientX;
    dragXMin0 = xMin;
    dragXMax0 = xMax;
    dragMoved = false;
    zoomBox = null;
    mainCanvas.style.cursor = dragMode === "pan" ? "grabbing" : "col-resize";
  }

  function onMouseMove(e) {
    if (dragMode !== "none") {
      if (Math.abs(e.clientX - dragStartX) > 3) dragMoved = true;

      if (dragMode === "pan") {
        const dx =
          ((e.clientX - dragStartX) / plotW()) * (dragXMax0 - dragXMin0);
        xMin = dragXMin0 - dx;
        xMax = dragXMax0 - dx;
        clampX();
        markDirty();
        return;
      } else {
        // zoom: update selection band
        const rect = mainCanvas.getBoundingClientRect();
        zoomBox = {
          startCx: dragStartX - rect.left,
          endCx: e.clientX - rect.left,
        };
        markDirty();
        return;
      }
    }

    const rect = mainCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const newHov = hitTest(cx, cy);
    if (newHov !== hoveredIdx) {
      hoveredIdx = newHov;
      markDirty();
    }
    updateTooltip(hoveredIdx, e.clientX, e.clientY);
  }

  function onMouseUp() {
    if (dragMode === "zoom" && dragMoved && zoomBox) {
      // Apply zoom to the selected region
      const left = Math.min(zoomBox.startCx, zoomBox.endCx);
      const right = Math.max(zoomBox.startCx, zoomBox.endCx);
      const newMin = cxToDataX(left);
      const newMax = cxToDataX(right);
      if (newMax - newMin >= MIN_X_SPAN) {
        xMin = newMin;
        xMax = newMax;
        clampX();
        markDirty();
      }
    }
    zoomBox = null;
    dragMode = "none";
    mainCanvas.style.cursor = shiftHeld ? "ew-resize" : "crosshair";
  }

  function onMouseLeave() {
    if (dragMode !== "none") {
      zoomBox = null;
      dragMode = "none";
      mainCanvas.style.cursor = shiftHeld ? "ew-resize" : "crosshair";
    }
    if (hoveredIdx >= 0) {
      hoveredIdx = -1;
      tooltip.style.display = "none";
      markDirty();
    }
  }

  function onDblClick() {
    resetView();
  }

  function hitTest(cx, cy) {
    const nRows = channelOrder.length;
    if (nRows === 0) return -1;
    // Candidate rows: nearest to the cursor plus ±1 for branch offsets (±0.4)
    // and barrier padding (0.05). Buckets get rebuilt lazily in renderFrame,
    // but mousemove fires before render, so ensure they're current.
    if (bucketsDirty) rebuildRowBuckets();
    const dy = cyToDataY(cy);
    const rCtr = Math.max(0, Math.min(nRows - 1, Math.round(dy)));
    const rLow = Math.max(0, rCtr - 1);
    const rHigh = Math.min(nRows - 1, rCtr + 1);
    const dx = cxToDataX(cx);

    // Pass A: zero-duration diamonds first (they render on top).
    for (let r = rLow; r <= rHigh; r++) {
      const idxs = rowInstances[r];
      const sts = rowStarts[r];
      if (!idxs || idxs.length === 0) continue;
      const lo = lowerBound(sts, dx - 4);
      const hi = upperBound(sts, dx + 4);
      for (let k = lo; k < hi; k++) {
        const i = idxs[k];
        if (!visible[i] || !isZeroDuration[i]) continue;
        const { cx: dcx, cy: dcy, halfW, halfH } = getDiamond(i);
        if (
          Math.abs(cx - dcx) <= halfW + 4 &&
          Math.abs(cy - dcy) <= halfH + 4
        ) {
          return i;
        }
      }
    }
    // Pass B: finite-duration rects. Later indices should win, so we sweep
    // the row window and track the last match.
    let lastHit = -1;
    for (let r = rLow; r <= rHigh; r++) {
      const idxs = rowInstances[r];
      const sts = rowStarts[r];
      if (!idxs || idxs.length === 0) continue;
      const lo = lowerBound(sts, dx - rowMaxDur[r]);
      const hi = upperBound(sts, dx);
      for (let k = lo; k < hi; k++) {
        const i = idxs[k];
        if (!visible[i] || isZeroDuration[i]) continue;
        if (finishes[i] < dx) continue;
        const rct = getRect(i);
        if (cx >= rct.x && cx <= rct.x + rct.w && cy >= rct.y && cy <= rct.y + rct.h) {
          if (i > lastHit) lastHit = i;
        }
      }
    }
    return lastHit;
  }

  function updateTooltip(i, clientX, clientY) {
    if (i < 0) {
      tooltip.style.display = "none";
      return;
    }
    const instr = instructions[instructionIds[i]];
    const pulse = pulseNames[pulseNameIds[i]];
    const t0 = starts[i];
    const tf = finishes[i];
    tooltip.innerHTML =
      `<b>${instr}</b><br>` +
      `Pulse: ${pulse}<br>` +
      `Start: ${t0}<br>` +
      `Finish: ${tf}<br>` +
      `Duration: ${tf - t0}`;
    tooltip.style.display = "block";
    // Keep tooltip inside window
    const tw = tooltip.offsetWidth;
    const winW = window.innerWidth;
    const left = clientX + 14 + tw > winW ? clientX - tw - 8 : clientX + 14;
    tooltip.style.left = left + "px";
    tooltip.style.top = clientY + 14 + "px";
  }

  // =========================================================================
  // Interaction — minimap
  // =========================================================================

  // Returns true when offsetX (minimap pixels) is inside the current viewport indicator.
  function isInsideViewport(offsetX) {
    const mw = mmCanvas.width;
    const vx1 = (xMin / (maxTime + 1)) * mw;
    const vx2 = (xMax / (maxTime + 1)) * mw;
    return offsetX >= vx1 && offsetX <= vx2;
  }

  function mmOnMouseDown(e) {
    if (e.button !== 0) return;
    // Context-sensitive: drag inside viewport = pan; outside = zoom-to-region
    mmDragMode = isInsideViewport(e.offsetX) ? "pan" : "zoom";
    mmDragStartX = e.offsetX;
    mmXMin0 = xMin;
    mmXMax0 = xMax;
    mmDragMoved = false;
    mmZoomBox = null;
    mmCanvas.style.cursor = mmDragMode === "pan" ? "grabbing" : "crosshair";
  }

  function mmOnMouseMove(e) {
    if (mmDragMode === "none") {
      // Update cursor to show whether dragging here will pan or zoom
      mmCanvas.style.cursor = isInsideViewport(e.offsetX) ? "grab" : "crosshair";
      return;
    }
    if (Math.abs(e.offsetX - mmDragStartX) > 2) mmDragMoved = true;

    if (mmDragMode === "pan") {
      mmCanvas.style.cursor = "grabbing";
      const dx = ((e.offsetX - mmDragStartX) / mmCanvas.width) * (maxTime + 1);
      const span = mmXMax0 - mmXMin0;
      xMin = Math.max(0, mmXMin0 + dx);
      xMax = xMin + span;
      if (xMax > maxTime + 1) {
        xMax = maxTime + 1;
        xMin = Math.max(0, xMax - span);
      }
      markDirty();
    } else {
      // zoom mode: update selection band
      mmZoomBox = { startMx: mmDragStartX, endMx: e.offsetX };
      markDirty();
    }
  }

  function mmSeekTo(offsetX) {
    const dataX = (offsetX / mmCanvas.width) * (maxTime + 1);
    const span = xMax - xMin;
    xMin = Math.max(0, dataX - span / 2);
    xMax = xMin + span;
    if (xMax > maxTime + 1) {
      xMax = maxTime + 1;
      xMin = Math.max(0, xMax - span);
    }
    markDirty();
  }

  // =========================================================================
  // Window-level mouse-up (handles both main canvas and minimap cleanup)
  // =========================================================================
  function onWindowMouseUp() {
    // Main canvas cleanup
    if (dragMode !== "none") {
      onMouseUp();
    }

    // Minimap cleanup
    if (mmDragMode !== "none") {
      if (mmDragMode === "zoom" && mmDragMoved && mmZoomBox) {
        const left = Math.min(mmZoomBox.startMx, mmZoomBox.endMx);
        const right = Math.max(mmZoomBox.startMx, mmZoomBox.endMx);
        const newMin = (left / mmCanvas.width) * (maxTime + 1);
        const newMax = (right / mmCanvas.width) * (maxTime + 1);
        if (newMax - newMin >= MIN_X_SPAN) {
          xMin = newMin;
          xMax = newMax;
          clampX();
          markDirty();
        }
      } else if (mmDragMode === "zoom" && !mmDragMoved) {
        // Single click on minimap → jump to that position
        mmSeekTo(mmDragStartX);
      }
      mmZoomBox = null;
      mmDragMode = "none";
      mmCanvas.style.cursor = "crosshair";
      markDirty();
    }
  }

  // =========================================================================
  // View reset
  // =========================================================================
  function resetView() {
    xMin = 0;
    xMax = maxTime + 1;
    yMin = -0.6;
    yMax = channelOrder.length > 0 ? channelOrder.length - 0.4 : 1.0;
    markDirty();
  }

  // Full reset: viewport + all sidebar toggles/filters back to initial state.
  // "Reset zoom" (double-click) calls resetView() only; "Reset" calls this.
  function resetAll() {
    if (mergeActive) {
      // reload() re-parses, rebuilds store (resetting all visibility arrays),
      // applies initial options, resets viewport, and rebuilds the sidebar.
      mergeActive = false;
      reload(rawCsv);
      return;
    }
    gateVisible.fill(1);
    branchVisible[0] = 1;
    branchVisible[1] = 1;
    branchVisible[2] = 1;
    chanVisible.fill(1);
    filterBarrier = false;
    soloGate = -1;
    searchQuery = "";
    searchHits.fill(0);
    applyInitialOptions(); // re-apply Python-supplied defaults (readout filter, etc.)
    recomputeVisible();
    resetView();
    buildSidebar();
  }

  // =========================================================================
  // Canvas sizing
  // =========================================================================
  function resizeCanvas() {
    const container = document.getElementById("canvas-container");
    if (!container) return;
    const containerW = container.clientWidth;
    const nRows = channelOrder.length || 1;
    const neededH = PAD_TOP + PAD_BOTTOM + nRows * ROW_HEIGHT_PX;

    mainCanvas.width = Math.max(300, containerW);
    mainCanvas.height = Math.min(CANVAS_MAX_H, Math.max(200, neededH));
    // Size the minimap from its own parent, not from canvas-container.
    // #mm-canvas lives in #minimap-area (full-width) while #main-canvas lives
    // in #canvas-container (full-width minus sidebar), so they differ by ~230 px.
    // Using the wrong width causes e.offsetX to be out of step with drawn pixels.
    mmCanvas.width = Math.max(300, mmCanvas.parentElement.clientWidth);
    mmCanvas.height = MM_H;
    // Canvas dimensions changed — the cached bitmap must be repainted.
    mmCacheDirty = true;
    markDirty();
  }

  // =========================================================================
  // Search
  // =========================================================================
  function runSearch(query) {
    searchQuery = query.trim().toLowerCase();
    searchHits.fill(0);
    if (!searchQuery) {
      markDirty();
      updateSearchStatus();
      return;
    }
    // Precomputed lowercase haystack (built once in buildStore) avoids
    // per-keystroke .toLowerCase() on N strings.
    for (let i = 0; i < N; i++) {
      if (searchText[i].includes(searchQuery)) {
        searchHits[i] = 1;
      }
    }
    searchCursor = -1;
    markDirty();
    updateSearchStatus();
  }

  function jumpToNextHit() {
    let start = searchCursor;
    for (let k = 1; k <= N; k++) {
      const idx = (start + k) % N;
      if (searchHits[idx] && visible[idx]) {
        // Pan x to centre on this instruction
        const mid = (starts[idx] + finishes[idx]) / 2;
        const span = xMax - xMin;
        xMin = Math.max(0, mid - span / 2);
        xMax = xMin + span;
        if (xMax > maxTime + 1) {
          xMax = maxTime + 1;
          xMin = Math.max(0, xMax - span);
        }
        searchCursor = idx;
        markDirty();
        return;
      }
    }
  }

  function updateSearchStatus() {
    const el = document.getElementById("search-status");
    if (!el) return;
    if (!searchQuery) {
      el.textContent = "";
      return;
    }
    const nHits = searchHits.reduce(
      (acc, h, i) => acc + (h && visible[i] ? 1 : 0),
      0
    );
    el.textContent = `${nHits} hit${nHits !== 1 ? "s" : ""} · Enter ↵ to jump`;
  }

  // =========================================================================
  // Channel classification for grouped sidebar sections
  // =========================================================================
  function classifyChannel(name) {
    if (name.startsWith("Qubit ")) return "qubit";
    if (name.startsWith(READOUT_PREFIX)) return "readout";
    if (name === "Hub" || name === "Receive") return "broadcast";
    return "other";
  }

  // =========================================================================
  // Sidebar
  // =========================================================================
  function buildSidebar() {
    const sidebar = document.getElementById("sidebar");
    if (!sidebar) return;
    sidebar.innerHTML = "";

    // --- Seed default expanded sections on first build ---
    if (!sidebarBuiltOnce) {
      expandedSections.add("source");
      expandedSections.add("instructions");
      sidebarBuiltOnce = true;
    }

    // --- Instructions legend (collapsible with master toggle checkbox) ---
    {
      const legSec = document.createElement("div");
      legSec.className = "sb-section";

      const legHeader = document.createElement("div");
      legHeader.className = "sb-section-header";

      const legChevron = document.createElement("span");
      legChevron.className = "sb-chevron";
      const legExpanded = expandedSections.has("instructions");
      legChevron.textContent = legExpanded ? "▼" : "▶";

      const legTitle = document.createElement("span");
      legTitle.className = "sb-section-title";
      legTitle.textContent = "Instructions";

      const legMasterCb = document.createElement("input");
      legMasterCb.type = "checkbox";
      legMasterCb.className = "sb-master-check";
      legMasterCb.title = "Toggle all";

      legHeader.appendChild(legChevron);
      legHeader.appendChild(legTitle);
      legHeader.appendChild(legMasterCb);
      legSec.appendChild(legHeader);

      const legBody = document.createElement("div");
      legBody.className = "sb-section-body" + (legExpanded ? " expanded" : "");

      // Per-gate rows
      for (let g = 0; g < gates.length; g++) {
        const row = document.createElement("div");
        row.className = "sb-row sb-clickable" + (soloGate === g ? " solo" : "");
        row.title = "Click: toggle · Alt+click: solo";
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = colorMap[g];
        const lbl = document.createElement("span");
        lbl.textContent = gates[g];
        if (!gateVisible[g]) {
          lbl.style.textDecoration = "line-through";
          lbl.style.color = "#aaa";
          swatch.style.opacity = "0.35";
        }
        row.appendChild(swatch);
        row.appendChild(lbl);
        const gCopy = g;
        row.addEventListener("click", (e) => {
          if (e.altKey) {
            if (soloGate === gCopy) {
              soloGate = -1;
              gateVisible.fill(1);
            } else {
              soloGate = gCopy;
              for (let gg = 0; gg < gates.length; gg++) {
                gateVisible[gg] = gg === gCopy ? 1 : 0;
              }
            }
          } else {
            soloGate = -1;
            gateVisible[gCopy] = gateVisible[gCopy] ? 0 : 1;
          }
          recomputeVisible();
          buildSidebar();
        });
        row.addEventListener("mouseenter", () => { hoveredGate = gCopy; markDirty(); });
        row.addEventListener("mouseleave", () => { hoveredGate = -1; markDirty(); });
        legBody.appendChild(row);
      }

      // Sync master checkbox to current gateVisible state
      function updateLegMaster() {
        const nTotal = gates.length;
        const nOn = gateVisible.reduce((acc, v) => acc + v, 0);
        if (nOn === nTotal && soloGate < 0) {
          legMasterCb.checked = true;
          legMasterCb.indeterminate = false;
        } else if (nOn === 0) {
          legMasterCb.checked = false;
          legMasterCb.indeterminate = false;
        } else {
          legMasterCb.checked = false;
          legMasterCb.indeterminate = true;
        }
      }
      updateLegMaster();

      legMasterCb.addEventListener("change", () => {
        if (legMasterCb.checked) {
          gateVisible.fill(1);
        } else {
          gateVisible.fill(0);
        }
        soloGate = -1;
        recomputeVisible();
        buildSidebar();
      });

      legHeader.addEventListener("click", (e) => {
        if (e.target === legMasterCb) return;
        const expanded = legBody.classList.toggle("expanded");
        legChevron.textContent = expanded ? "▼" : "▶";
        if (expanded) expandedSections.add("instructions");
        else expandedSections.delete("instructions");
      });

      legSec.appendChild(legBody);
      sidebar.appendChild(legSec);
    }

    // --- Channels (grouped collapsible sections) ---
    buildChannelSections(sidebar);

    // --- Branches ---
    const branchSec = makeSection("Branches");
    for (let b = 0; b < 3; b++) {
      const bCopy = b;
      branchSec.appendChild(
        makeCheckRow(BRANCHES[b], !!branchVisible[b], (checked) => {
          branchVisible[bCopy] = checked ? 1 : 0;
          recomputeVisible();
        })
      );
    }
    sidebar.appendChild(branchSec);

    // --- Filters (barrier + merge; readout is now in channel sections) ---
    const filterSec = makeSection("Filters");
    filterSec.appendChild(
      makeToggleBtn("Hide barriers", filterBarrier, (on) => {
        filterBarrier = on;
        recomputeVisible();
      })
    );
    filterSec.appendChild(
      makeToggleBtn("Merge instructions", mergeActive, (on) => {
        mergeActive = on;
        reload(rawCsv);
      })
    );
    sidebar.appendChild(filterSec);

    // --- Search ---
    const searchSec = makeSection("Search");
    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.placeholder = "gate / channel / instruction";
    searchInput.value = searchQuery;
    searchInput.className = "search-input";
    // Debounce keystrokes (~120ms trailing) so rapid typing doesn't stall
    // the main thread with per-keystroke O(N) scans.
    let searchTimer = null;
    searchInput.addEventListener("input", (e) => {
      const v = e.target.value;
      if (searchTimer) clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        searchTimer = null;
        runSearch(v);
      }, 120);
    });
    searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") jumpToNextHit();
    });
    searchSec.appendChild(searchInput);
    const statusEl = document.createElement("div");
    statusEl.id = "search-status";
    statusEl.className = "search-status";
    searchSec.appendChild(statusEl);
    sidebar.appendChild(searchSec);
    updateSearchStatus();

    // --- Source (collapsible; state persisted in expandedSections) ---
    const sourceSec = document.createElement("div");
    sourceSec.className = "sb-section";

    const sourceHeader = document.createElement("div");
    sourceHeader.className = "sb-section-header";
    const sourceChevron = document.createElement("span");
    sourceChevron.className = "sb-chevron";
    const sourceIsExpanded = expandedSections.has("source");
    sourceChevron.textContent = sourceIsExpanded ? "▼" : "▶";
    const sourceTitleSpan = document.createElement("span");
    sourceTitleSpan.className = "sb-section-title";
    sourceTitleSpan.textContent = "Source";
    sourceHeader.appendChild(sourceChevron);
    sourceHeader.appendChild(sourceTitleSpan);
    sourceSec.appendChild(sourceHeader);

    const sourceBody = document.createElement("div");
    sourceBody.className = "sb-section-body" + (sourceIsExpanded ? " expanded" : "");

    const sourceTA = document.createElement("textarea");
    sourceTA.className = "source-editor";
    sourceTA.spellcheck = false;
    sourceTA.value = rawCsv;
    sourceBody.appendChild(sourceTA);

    const btnRow = document.createElement("div");
    btnRow.className = "source-btn-row";

    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Apply";
    applyBtn.className = "source-btn source-btn-apply";

    const revertBtn = document.createElement("button");
    revertBtn.textContent = "Revert";
    revertBtn.className = "source-btn source-btn-revert";

    btnRow.appendChild(applyBtn);
    btnRow.appendChild(revertBtn);
    sourceBody.appendChild(btnRow);

    const sourceErr = document.createElement("div");
    sourceErr.className = "source-error";
    sourceBody.appendChild(sourceErr);

    applyBtn.addEventListener("click", () => {
      // Capture caret before reload() replaces the DOM
      const selStart = sourceTA.selectionStart;
      const selEnd = sourceTA.selectionEnd;
      const newCsv = sourceTA.value;
      try {
        parseCsv(newCsv, { strict: true }); // validate before committing
        reload(newCsv); // rebuilds sidebar — sourceTA/btnRow/sourceErr are now stale
        // Re-focus the new textarea and restore caret
        const newTA = document.querySelector(".source-editor");
        if (newTA) {
          newTA.focus();
          const maxPos = newTA.value.length;
          newTA.setSelectionRange(
            Math.min(selStart, maxPos),
            Math.min(selEnd, maxPos)
          );
        }
        // Show a brief "Applied ✓" flash in the new button row
        const newBtnRow = document.querySelector(".source-btn-row");
        if (newBtnRow) {
          const flag = document.createElement("span");
          flag.className = "source-applied-flag";
          flag.textContent = "Applied ✓";
          newBtnRow.appendChild(flag);
          setTimeout(() => flag.remove(), 1200);
        }
      } catch (ex) {
        // sourceErr is stale after a failed validation (reload was not called)
        sourceErr.textContent = ex.message;
      }
    });

    revertBtn.addEventListener("click", () => {
      sourceTA.value = rawCsv;
      sourceErr.textContent = "";
    });

    sourceHeader.addEventListener("click", () => {
      const expanded = sourceBody.classList.toggle("expanded");
      sourceChevron.textContent = expanded ? "▼" : "▶";
      if (expanded) expandedSections.add("source");
      else expandedSections.delete("source");
    });

    sourceSec.appendChild(sourceBody);
    sidebar.appendChild(sourceSec);

    // --- Shortcuts (collapsed by default; lives below Source) ---
    {
      const helpSec = document.createElement("div");
      helpSec.className = "sb-section";

      const helpHeader = document.createElement("div");
      helpHeader.className = "sb-section-header";
      const helpChevron = document.createElement("span");
      helpChevron.className = "sb-chevron";
      const helpExpanded = expandedSections.has("help");
      helpChevron.textContent = helpExpanded ? "▼" : "▶";
      const helpTitle = document.createElement("span");
      helpTitle.className = "sb-section-title";
      helpTitle.textContent = "Shortcuts";
      helpHeader.appendChild(helpChevron);
      helpHeader.appendChild(helpTitle);
      helpSec.appendChild(helpHeader);

      const helpBody = document.createElement("div");
      helpBody.className = "sb-section-body" + (helpExpanded ? " expanded" : "");

      const dl = document.createElement("dl");
      dl.className = "shortcuts-list";
      const shortcuts = [
        ["Wheel",              "Zoom X at cursor"],
        ["Shift+Wheel",        "Zoom Y at cursor"],
        ["Drag",               "Zoom to region"],
        ["Shift+Drag",         "Pan"],
        ["Double-click",       "Reset zoom"],
        ["Alt+click legend",   "Solo instruction type"],
        ["Hover legend row",   "Highlight all instances"],
        ["Enter in search",    "Jump to next hit"],
        ["Map wheel",          "Zoom X (centered)"],
        ["Map drag (inside)",  "Pan viewport"],
        ["Map drag (outside)", "Zoom to region"],
        ["Map click",          "Jump to position"],
      ];
      for (const [key, val] of shortcuts) {
        const dt = document.createElement("dt");
        dt.textContent = key;
        const dd = document.createElement("dd");
        dd.textContent = val;
        dl.appendChild(dt);
        dl.appendChild(dd);
      }
      helpBody.appendChild(dl);

      helpHeader.addEventListener("click", () => {
        const expanded = helpBody.classList.toggle("expanded");
        helpChevron.textContent = expanded ? "▼" : "▶";
        if (expanded) expandedSections.add("help");
        else expandedSections.delete("help");
      });

      helpSec.appendChild(helpBody);
      sidebar.appendChild(helpSec);
    }

    // --- Reset buttons (Reset zoom = viewport only; Reset = everything) ---
    const resetRow = document.createElement("div");
    resetRow.style.cssText = "display:flex; gap:6px; margin-top:10px;";

    const resetZoomBtn = document.createElement("button");
    resetZoomBtn.textContent = "Reset zoom";
    resetZoomBtn.className = "reset-btn";
    resetZoomBtn.style.marginTop = "0";
    resetZoomBtn.addEventListener("click", resetView);

    const resetAllBtn = document.createElement("button");
    resetAllBtn.textContent = "Reset";
    resetAllBtn.className = "reset-btn";
    resetAllBtn.style.marginTop = "0";
    resetAllBtn.addEventListener("click", resetAll);

    resetRow.appendChild(resetZoomBtn);
    resetRow.appendChild(resetAllBtn);
    sidebar.appendChild(resetRow);
  }

  // --- Grouped / collapsible channel sections ---
  function buildChannelSections(sidebar) {
    const CATEGORY_DEFS = [
      { key: "qubit",     label: "Qubit channels" },
      { key: "readout",   label: "Readout channels" },
      { key: "broadcast", label: "Broadcast channels" },
      { key: "other",     label: "Other channels" },
    ];

    // Use same display order as the main plot
    const displayChannels =
      opts.includedChannels && opts.includedChannels.length
        ? [...opts.includedChannels]
            .reverse()
            .filter((n) => channels.includes(n))
        : [...channels];

    for (const { key, label } of CATEGORY_DEFS) {
      const members = displayChannels.filter(
        (n) => classifyChannel(n) === key
      );
      if (members.length === 0) continue;

      const secEl = document.createElement("div");
      secEl.className = "sb-section";

      // --- Collapsible header ---
      const header = document.createElement("div");
      header.className = "sb-section-header";

      const chevron = document.createElement("span");
      chevron.className = "sb-chevron";
      chevron.textContent = "▶";

      const titleSpan = document.createElement("span");
      titleSpan.className = "sb-section-title";
      titleSpan.textContent = label;

      const masterCb = document.createElement("input");
      masterCb.type = "checkbox";
      masterCb.className = "sb-master-check";
      masterCb.title = "Toggle all";

      header.appendChild(chevron);
      header.appendChild(titleSpan);
      header.appendChild(masterCb);
      secEl.appendChild(header);

      // --- Collapsible body (restore expanded state from module-level set) ---
      const isExpanded = expandedSections.has(key);
      const body = document.createElement("div");
      body.className = "sb-section-body" + (isExpanded ? " expanded" : "");
      chevron.textContent = isExpanded ? "▼" : "▶";

      // Reflect current visibility state in the master checkbox
      function updateMaster() {
        const indices = members
          .map((n) => channels.indexOf(n))
          .filter((c) => c >= 0);
        const total = indices.length;
        const nChecked = indices.filter((c) => chanVisible[c]).length;
        if (nChecked === 0) {
          masterCb.checked = false;
          masterCb.indeterminate = false;
        } else if (nChecked === total) {
          masterCb.checked = true;
          masterCb.indeterminate = false;
        } else {
          masterCb.checked = false;
          masterCb.indeterminate = true;
        }
      }

      // Individual channel rows
      for (const name of members) {
        const c = channels.indexOf(name);
        if (c < 0) continue;
        const row = makeCheckRow(name, !!chanVisible[c], (checked) => {
          chanVisible[c] = checked ? 1 : 0;
          recomputeVisible();
          updateMaster();
        });
        body.appendChild(row);
      }

      updateMaster();

      // Master checkbox: toggle all members on/off (update checkboxes in-place,
      // no full sidebar rebuild needed — preserves expanded state naturally)
      masterCb.addEventListener("change", () => {
        const indices = members
          .map((n) => channels.indexOf(n))
          .filter((c) => c >= 0);
        const allOn = masterCb.checked;
        for (const c of indices) chanVisible[c] = allOn ? 1 : 0;
        // Sync individual row checkboxes without rebuilding the sidebar
        body.querySelectorAll("input[type=\"checkbox\"]").forEach((cb) => {
          cb.checked = allOn;
        });
        recomputeVisible();
        updateMaster();
      });

      // Chevron / header: toggle expand/collapse; persist in expandedSections
      header.addEventListener("click", (e) => {
        if (e.target === masterCb) return; // handled separately
        const expanded = body.classList.toggle("expanded");
        chevron.textContent = expanded ? "▼" : "▶";
        if (expanded) expandedSections.add(key);
        else expandedSections.delete(key);
      });

      secEl.appendChild(body);
      sidebar.appendChild(secEl);
    }
  }

  // --- Sidebar helpers ---
  function makeSection(title) {
    const sec = document.createElement("div");
    sec.className = "sb-section";
    const h = document.createElement("div");
    h.className = "sb-title";
    h.textContent = title;
    sec.appendChild(h);
    return sec;
  }

  function makeCheckRow(label, checked, onChange) {
    const lbl = document.createElement("label");
    lbl.className = "sb-row sb-check";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = checked;
    cb.addEventListener("change", () => onChange(cb.checked));
    const span = document.createElement("span");
    span.textContent = label;
    lbl.appendChild(cb);
    lbl.appendChild(span);
    return lbl;
  }

  function makeToggleBtn(label, active, onChange) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.className = "toggle-btn" + (active ? " active" : "");
    btn.addEventListener("click", () => {
      const nowActive = !btn.classList.contains("active");
      btn.classList.toggle("active", nowActive);
      onChange(nowActive);
    });
    return btn;
  }

  // =========================================================================
  // Load / reload  (call whenever CSV or merge flag changes)
  // =========================================================================
  function reload(csv) {
    rawCsv = csv;
    let rows = parseCsv(csv);
    if (mergeActive) rows = mergeInstructions(rows);
    buildStore(rows);
    applyInitialOptions();
    recomputeVisible();
    resetView();
    buildSidebar();
    resizeCanvas();
  }

  // =========================================================================
  // Standalone paste / drag-drop area
  // =========================================================================
  function setupPasteArea() {
    const area = document.getElementById("paste-area");
    if (!area) return;

    area.addEventListener("click", () => area.querySelector("textarea")?.focus());

    const ta = area.querySelector("textarea");
    if (ta) {
      ta.addEventListener("input", () => {
        if (ta.value.trim()) {
          area.style.display = "none";
          reload(ta.value);
        }
      });
      ta.addEventListener("paste", (e) => {
        // Let the default paste happen then read value on next tick
        setTimeout(() => {
          if (ta.value.trim()) {
            area.style.display = "none";
            reload(ta.value);
          }
        }, 0);
      });
    }

    area.addEventListener("dragover", (e) => {
      e.preventDefault();
      area.classList.add("drag-over");
    });
    area.addEventListener("dragleave", () =>
      area.classList.remove("drag-over")
    );
    area.addEventListener("drop", (e) => {
      e.preventDefault();
      area.classList.remove("drag-over");
      const file = e.dataTransfer.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => {
          area.style.display = "none";
          reload(ev.target.result);
        };
        reader.readAsText(file);
      } else {
        const text = e.dataTransfer.getData("text/plain");
        if (text) {
          area.style.display = "none";
          reload(text);
        }
      }
    });
  }

  // =========================================================================
  // Initialisation
  // =========================================================================
  function init() {
    mainCanvas = document.getElementById("main-canvas");
    mainCtx = mainCanvas.getContext("2d");
    mmCanvas = document.getElementById("mm-canvas");
    mmCtx = mmCanvas.getContext("2d");
    tooltip = document.getElementById("tooltip");

    // Main canvas events
    mainCanvas.addEventListener("wheel", onWheel, { passive: false });
    mainCanvas.addEventListener("mousedown", onMouseDown);
    mainCanvas.addEventListener("mousemove", onMouseMove);
    mainCanvas.addEventListener("mouseleave", onMouseLeave);
    mainCanvas.addEventListener("dblclick", onDblClick);
    mainCanvas.style.cursor = "crosshair"; // default: zoom-drag mode

    // Minimap events
    mmCanvas.addEventListener("mousedown", mmOnMouseDown);
    mmCanvas.addEventListener("mousemove", mmOnMouseMove);
    mmCanvas.addEventListener("mouseleave", () => {
      if (mmDragMode === "none") mmCanvas.style.cursor = "crosshair";
    });
    mmCanvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
      // Anchor at current viewport centre so wheel-zoom is a pure zoom (no pan)
      const anchorData = (xMin + xMax) / 2;
      zoomX(anchorData, factor);
    }, { passive: false });

    // Shared window-level mouse-up
    window.addEventListener("mouseup", onWindowMouseUp);

    // Shift key tracking for cursor feedback
    window.addEventListener("keydown", (e) => {
      if (e.key === "Shift") {
        shiftHeld = true;
        if (dragMode === "none") mainCanvas.style.cursor = "ew-resize";
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.key === "Shift") {
        shiftHeld = false;
        if (dragMode === "none") mainCanvas.style.cursor = "crosshair";
      }
    });

    window.addEventListener("resize", () => {
      resizeCanvas();
      markDirty();
    });

    setupPasteArea();

    // Check for pre-loaded data from Python bridge
    const pre = window.__CIRCUIT_SCHEDULE_TIMING__;
    if (pre && pre.trim()) {
      document.getElementById("paste-area").style.display = "none";
      const preOpts = window.__CIRCUIT_SCHEDULE_OPTIONS__ || {};
      opts = Object.assign(opts, preOpts);
      reload(pre);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
