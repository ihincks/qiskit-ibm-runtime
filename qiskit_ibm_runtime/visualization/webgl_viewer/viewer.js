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
 * Circuit Schedule Timing Viewer  —  WebGL2 edition
 *
 * A parallel implementation of the Canvas 2D viewer under
 * ../html_viewer/ that renders every instruction with a single instanced
 * WebGL2 draw call, so 100 K+ elements stay interactive.
 *
 * Data is supplied inline by the Python bridge:
 *     window.__CIRCUIT_SCHEDULE_TIMING__ = "<csv>"
 *     window.__CIRCUIT_SCHEDULE_OPTIONS__ = { ... }
 */
(function () {
  "use strict";

  // =========================================================================
  // Constants (must match html_viewer/viewer.js for pixel-parity geometry)
  // =========================================================================

  const COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
  ];

  const READOUT_PREFIX = "AWGR";
  const BARRIER_GATE = "barrier";

  const PAD_LEFT = 120;
  const PAD_RIGHT = 8;
  const PAD_TOP = 8;
  const PAD_BOTTOM = 38;

  const ROW_HEIGHT_PX = 60;
  // Minimap height (CSS pixels). User-adjustable via the resize handle below
  // the minimap; bounded by [MM_H_MIN, MM_H_MAX] (further limited at runtime
  // to half the window height).
  let mmCurrentH = 50;
  const MM_H_MIN = 30;
  const MM_H_MAX = 300;
  // Pickable thickness (CSS pixels) around each viewport-rect edge for
  // drag-to-resize on the minimap.
  const MM_EDGE_PX = 6;
  const MIN_LABEL_PX = 12;
  const MIN_STROKE_PX = 2;
  // Below this pixel-width for the shortest labelled cycle length we skip
  // the entire text pass (LOD cutoff).
  const MIN_LABEL_CYCLES = 3;

  const BRANCH_Y = {
    main: { low: -0.4, high: 0.4, annY: 0.0 },
    then: { low: 0.0, high: 0.4, annY: 0.25 },
    else: { low: -0.4, high: 0.0, annY: -0.25 },
  };
  const BARRIER_PAD = 0.05;
  const ZERO_DUR_CENTER = { main: 0, then: 0.2, else: -0.2 };
  const ZERO_DUR_HALF = 0.2;
  // Diamond glyph (zero-duration event) shape constraints. halfW is capped
  // by (a) DIAMOND_ASPECT * halfH so the diamond is always noticeably
  // taller than wide, and (b) DIAMOND_WIDTH_CYCLES * pxPerCycle so it
  // shrinks along X as you zoom out. Both bounds are gate-independent —
  // RZ, RX shift-phase, SX shift-phase all render identically.
  const DIAMOND_ASPECT = 0.6;
  const DIAMOND_WIDTH_CYCLES = 1.5;

  const ZOOM_FACTOR = 1.2;
  const MIN_X_SPAN = 2;
  // Minimum Y span (data units). Roughly one row (a data row spans 1.0).
  // Prevents zooming past a single row.
  const MIN_Y_SPAN = 0.8;
  // Zoom-to-region gesture: triangular cones emanating from the click
  // position along each axis. A drag within the X cone (|dy| <= |dx| *
  // CONE_SLOPE) is xOnly; within the Y cone is yOnly; otherwise xy.
  // 20° cone → tan(20°) ≈ 0.364.
  const CONE_SLOPE = Math.tan(20 * Math.PI / 180);
  // Minimum drag distance before we consider transitioning out of the
  // default xOnly mode. Below this the gesture is treated as an intended
  // horizontal slice (matches the initial mode).
  const GESTURE_MIN_PX = 24;

  const BRANCHES = ["main", "then", "else"];

  // =========================================================================
  // State
  // =========================================================================

  let rawCsv = "";
  let opts = {
    includedChannels: null,
    filterReadoutChannels: false,
    filterBarriers: false,
    mergeCommonInstructions: false,
  };
  // Sample time in nanoseconds. When > 0, x-axis ticks are shown in real
  // time with auto-scaled units (ns / µs / ms / s) instead of cycles.
  let dtNs = 0;

  // Columnar store
  let N = 0;
  let starts, finishes, branchIds, gateIds, channelIds;
  let instructionIds, pulseNameIds, isZeroDuration;
  let gates = [], channels = [], instructions = [], pulseNames = [];
  let gateInstructions = [];
  let colorMap = {};
  let maxTime = 0;

  // Filter / visibility
  let gateVisible = new Uint8Array(0);
  let branchVisible = new Uint8Array([1, 1, 1]);
  let chanVisible = new Uint8Array(0);
  let filterBarrier = false;
  let mergeActive = false;
  let visible = new Uint8Array(0);

  // Row ordering
  let channelOrder = [];
  let channelRow = new Int16Array(0);

  // Stable per-element reverse indices (built once in buildStore).
  let chanInstances = [];      // chanInstances[c]  : Int32Array of instance indices
  let branchInstances = [];    // branchInstances[b]: Int32Array of instance indices
  // Per-channel start-sorted buckets (stable, never rebuilt after load).
  // channelBuckets[c] = { idxArr: Int32Array, startArr: Int32Array }
  let channelBuckets = [];
  // Precomputed gate index of BARRIER_GATE in gates[] (-1 if none).
  let barrierGateId = -1;
  // Uint16 view of channelRow used as the uChannelRow GPU texture source.
  let channelRowData = new Uint16Array(0);

  // Viewport
  let xMin = 0, xMax = 100;
  let yMin = -0.6, yMax = 1.4;

  // Search
  let searchQuery = "";
  let searchHits = new Uint8Array(0);
  let searchCursor = -1;

  // Hover / solo
  let hoveredIdx = -1;
  let hoveredGate = -1;
  let soloGate = -1;

  // Main-canvas drag
  let dragMode = "none";  // "none" | "pan" | "zoom"
  let dragStartX = 0, dragXMin0 = 0, dragXMax0 = 0;
  let dragStartY = 0, dragYMin0 = 0, dragYMax0 = 0;
  let dragMoved = false;
  // zoomBox shape: { startCx, startCy, endCx, endCy, mode: "xOnly" | "xy" }
  let zoomBox = null;

  // Minimap drag. mmDragMode ∈ {"none", "pan", "zoom", "resize"}. When
  // "resize" is active, mmResizeEdges records which viewport-rect edges
  // are being dragged: { left, right, top, bottom } booleans.
  let mmDragMode = "none";
  let mmDragStartX = 0, mmXMin0 = 0, mmXMax0 = 0;
  let mmDragStartY = 0, mmYMin0 = 0, mmYMax0 = 0;
  let mmDragMoved = false;
  // mmZoomBox shape: { startMx, startMy, endMx, endMy }
  let mmZoomBox = null;
  let mmResizeEdges = null;

  // Minimap resize handle
  let mmResizeDragging = false;
  let mmResizeStartY = 0;
  let mmResizeStartH = 0;

  let shiftHeld = false;

  // Sidebar expand state
  let expandedSections = new Set();
  let sidebarBuiltOnce = false;
  // Sidebar DOM refs (built once per reload, updated in-place on toggle).
  // { gateRows: [{row, swatch, label}], chanRows: [{cb}], branchRows: [{cb}],
  //   groupMasters: {key: cb}, groupMembers: {key: [channelId,...]}, instMaster: cb }
  let sidebarRefs = null;

  let dirty = false;

  // Reusable Float32 buffer for text glyph instances (grow-by-double).
  let textFloats = new Float32Array(4096 * 8);
  let textFloatsUsed = 0;

  // Minimap rects layer cache (offscreen canvas).
  let mmRectsCache = null;
  let mmCacheDirty = true;

  // DOM refs
  let glCanvas, overlayCanvas, overlayCtx, mmCanvas, mmCtx, tooltip, sidebar;
  let container;

  // GL state
  let gl = null;
  let rectsProg = null, diamondsProg = null, textProg = null;
  let paletteTex = null, glyphTex = null;
  let instanceVBO = null;
  let textVBO = null;
  // Small GPU lookup textures (allocated/resized in buildLookupTextures).
  let chanRowTex = null;    // uChannelRow: R16UI, channelId → row index
  let chanVisTex = null;    // uChanVisible: R8,   channelId → 0/1
  let gateVisTex = null;    // uGateVisible: R8,   gateId    → 0/1
  let branchVisTex = null;  // uBranchVisible: R8, branchId  → 0/1
  let glyphMap = null;   // char -> {u, v, w, h, ax, ay}
  let glyphAtlasSize = 512;
  let glyphAtlasReady = false;
  let dpr = 1;

  // =========================================================================
  // CSV parsing (port of html_viewer parseCsv)
  // =========================================================================
  function parseCsv(csv, { strict = false } = {}) {
    const rows = [];
    const lines = csv.split("\n");
    for (let lineNo = 0; lineNo < lines.length; lineNo++) {
      const trimmed = lines[lineNo].trim();
      if (!trimmed) continue;
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
        gateName: words[1].trim().split("_")[0],
        isZero: pulse === "shift_phase" ? 1 : 0,
      });
    }
    return rows;
  }

  function mergeInstructions(rows) {
    const groups = new Map();
    for (const r of rows) {
      const key = `${r.branch}\0${r.instruction}\0${r.channel}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }
    const merged = [];
    for (const grp of groups.values()) {
      if (grp.length === 1) { merged.push(grp[0]); continue; }
      grp.sort((a, b) => a.start - b.start);
      const acc = [Object.assign({}, grp[0])];
      for (let k = 1; k < grp.length; k++) {
        const prev = acc[acc.length - 1];
        if (grp[k].start === prev.finish) {
          prev.finish = grp[k].finish;
        } else {
          acc.push(Object.assign({}, grp[k]));
        }
      }
      merged.push(...acc);
    }
    return merged;
  }

  function buildStore(rows) {
    const gateSet = new Set(), chanSet = new Set(),
          instrSet = new Set(), pulseSet = new Set();
    for (const r of rows) {
      gateSet.add(r.gateName);
      chanSet.add(r.channel);
      instrSet.add(r.instruction);
      pulseSet.add(r.pulse);
    }
    gates = [...gateSet].sort();
    // Natural (numeric-aware) sort for channels so "Qubit 2" sits between
    // "Qubit 1" and "Qubit 10". Applied to all channel names, so AWGR0..N
    // and other numbered channel families also order correctly.
    channels = [...chanSet].sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" })
    );
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

    maxTime = 0;
    for (let i = 0; i < N; i++) if (finishes[i] > maxTime) maxTime = finishes[i];

    colorMap = {};
    for (let g = 0; g < gates.length; g++) colorMap[g] = COLORS[g % COLORS.length];

    gateInstructions = gates.map(() => []);
    for (let i = 0; i < N; i++) gateInstructions[gateIds[i]].push(i);
    gateInstructions = gateInstructions.map((arr) => new Int32Array(arr));

    // Reverse indices by channel and branch (like gateInstructions).
    const _cBuckets = channels.map(() => []);
    const _bBuckets = [[], [], []];
    for (let i = 0; i < N; i++) {
      _cBuckets[channelIds[i]].push(i);
      _bBuckets[branchIds[i]].push(i);
    }
    chanInstances = _cBuckets.map((arr) => new Int32Array(arr));
    branchInstances = _bBuckets.map((arr) => new Int32Array(arr));

    // Stable per-channel buckets sorted by start time (never rebuilt after load).
    channelBuckets = _cBuckets.map((arr) => {
      arr.sort((a, b2) => starts[a] - starts[b2]);
      const idxArr = new Int32Array(arr.length);
      const startArr = new Int32Array(arr.length);
      for (let k = 0; k < arr.length; k++) {
        idxArr[k] = arr[k];
        startArr[k] = starts[arr[k]];
      }
      return { idxArr, startArr };
    });

    // Precompute barrier gate index (avoids per-instance string compare later).
    barrierGateId = gates.indexOf(BARRIER_GATE);

    gateVisible = new Uint8Array(gates.length).fill(1);
    chanVisible = new Uint8Array(channels.length).fill(1);
    visible = new Uint8Array(N).fill(1);
    searchHits = new Uint8Array(N).fill(0);
    channelRow = new Int16Array(channels.length).fill(-1);
    channelRowData = new Uint16Array(channels.length);
    channelOrder = [];
    soloGate = -1;
  }

  // =========================================================================
  // Startup options and visibility recomputation
  // =========================================================================
  function applyInitialOptions() {
    if (opts.filterReadoutChannels) {
      for (let c = 0; c < channels.length; c++) {
        if (channels[c].startsWith(READOUT_PREFIX)) chanVisible[c] = 0;
      }
    }
    if (opts.filterBarriers) filterBarrier = true;
    if (Array.isArray(opts.includedChannels) && opts.includedChannels.length) {
      const inc = new Set(opts.includedChannels);
      for (let c = 0; c < channels.length; c++) {
        if (!inc.has(channels[c])) chanVisible[c] = 0;
      }
    }
    if (opts.mergeCommonInstructions) mergeActive = true;

    // Default-hide non-qubit channel groups (readout / broadcast / other).
    // Only when the user did not supply an explicit includedChannels list —
    // that list is a stronger user preference and wins.
    if (!Array.isArray(opts.includedChannels) || !opts.includedChannels.length) {
      for (let c = 0; c < channels.length; c++) {
        if (classifyChannel(channels[c]) !== "qubit") chanVisible[c] = 0;
      }
    }
  }

  function classifyChannel(name) {
    if (name.startsWith("Qubit ")) return "qubit";
    if (name.startsWith(READOUT_PREFIX)) return "readout";
    if (name === "Hub" || name === "Receive") return "broadcast";
    return "other";
  }

  function computeChannelOrder() {
    channelOrder = [];
    channelRow.fill(-1);
    const inc = Array.isArray(opts.includedChannels) ? opts.includedChannels : null;
    const already = new Set();
    if (inc && inc.length) {
      // Include-list first, in listed order: first name should sit at top.
      for (let k = 0; k < inc.length; k++) {
        const name = inc[k];
        const c = channels.indexOf(name);
        if (c >= 0 && chanVisible[c] && !already.has(c)) {
          channelOrder.push(c);
          already.add(c);
        }
      }
      // Then the rest, in natural (channels[]) order.
      for (let c = 0; c < channels.length; c++) {
        if (chanVisible[c] && !already.has(c)) {
          channelOrder.push(c);
          already.add(c);
        }
      }
    } else {
      for (let c = 0; c < channels.length; c++) {
        if (chanVisible[c]) channelOrder.push(c);
      }
    }
    // channelOrder[0] is now the desired-top channel. dataYtoCy inverts Y
    // (row 0 → bottom, row nRows-1 → top), so reverse the array to put the
    // desired-top channel at the highest row index.
    channelOrder.reverse();
    for (let r = 0; r < channelOrder.length; r++) channelRow[channelOrder[r]] = r;
    // Sync Uint16 copy for the uChannelRow GPU texture (hidden channels get 0;
    // they are gated by uChanVisible so the value doesn't matter).
    for (let c = 0; c < channels.length; c++) {
      channelRowData[c] = channelRow[c] < 0 ? 0 : channelRow[c];
    }
  }

  // ---------------------------------------------------------------------------
  // Visibility helpers
  // ---------------------------------------------------------------------------

  // Per-instance visibility on the CPU side. Uses the same four conditions as
  // the vertex shaders but from the small JS arrays (no O(N) scan needed when
  // only one dimension changes — callers use the reverse-index arrays).
  function isVisibleCpu(i) {
    const c = channelIds[i], g = gateIds[i], b = branchIds[i];
    return (chanVisible[c] && gateVisible[g] && branchVisible[b] &&
            !(filterBarrier && g === barrierGateId)) ? 1 : 0;
  }

  // Delta-update visible[] for instances belonging to one channel.
  function updateVisibleForChannel(c) {
    const arr = chanInstances[c];
    for (let k = 0; k < arr.length; k++) visible[arr[k]] = isVisibleCpu(arr[k]);
    mmCacheDirty = true;
  }

  // Delta-update visible[] for instances belonging to one gate type.
  function updateVisibleForGate(g) {
    const arr = gateInstructions[g];
    for (let k = 0; k < arr.length; k++) visible[arr[k]] = isVisibleCpu(arr[k]);
    mmCacheDirty = true;
  }

  // Delta-update visible[] for instances belonging to one branch.
  function updateVisibleForBranch(b) {
    const arr = branchInstances[b];
    for (let k = 0; k < arr.length; k++) visible[arr[k]] = isVisibleCpu(arr[k]);
    mmCacheDirty = true;
  }

  // Full visible[] rebuild — used when multiple dimensions change at once
  // (e.g. resetAll, filterBarrier toggle, solo mode).
  function updateVisibleAll() {
    for (let i = 0; i < N; i++) visible[i] = isVisibleCpu(i);
    mmCacheDirty = true;
  }

  // Full recompute of channel order + all visibility state; used only from
  // resetAll() / applyInitialOptions() paths that may change many dimensions.
  function recomputeVisible() {
    const prevNRows = channelOrder.length;
    computeChannelOrder();
    updateVisibleAll();
    if (gl) {
      uploadChannelRowTex();
      uploadChanVisTex();
      uploadGateVisTex();
      uploadBranchVisTex();
    }
    if (channelOrder.length !== prevNRows && container) resizeCanvas();
  }

  // Standard lower/upper bound on a sorted Int32Array.
  function lowerBound(arr, v) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (arr[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
  }
  function upperBound(arr, v) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (arr[mid] <= v) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  // =========================================================================
  // Coordinate transforms
  // =========================================================================
  function plotW() { return glCanvas.clientWidth - PAD_LEFT - PAD_RIGHT; }
  function plotH() { return glCanvas.clientHeight - PAD_TOP - PAD_BOTTOM; }
  function dataXtoCx(dx) { return PAD_LEFT + ((dx - xMin) / (xMax - xMin)) * plotW(); }
  function cxToDataX(cx) { return xMin + ((cx - PAD_LEFT) / plotW()) * (xMax - xMin); }
  function dataYtoCy(dy) { return PAD_TOP + (1 - (dy - yMin) / (yMax - yMin)) * plotH(); }
  function cyToDataY(cy) { return yMax - ((cy - PAD_TOP) / plotH()) * (yMax - yMin); }

  function getRect(i) {
    const branch = BRANCHES[branchIds[i]] || "main";
    const bOff = BRANCH_Y[branch] || BRANCH_Y.main;
    const row = channelRow[channelIds[i]];
    let yLow = row + bOff.low;
    let yHigh = row + bOff.high;
    if (gates[gateIds[i]] === BARRIER_GATE) {
      yLow -= BARRIER_PAD;
      yHigh += BARRIER_PAD;
    }
    const x1 = dataXtoCx(starts[i]);
    const x2 = dataXtoCx(finishes[i]);
    const py1 = dataYtoCy(yHigh);
    const py2 = dataYtoCy(yLow);
    return { x: x1, y: py1, w: Math.max(1, x2 - x1), h: py2 - py1 };
  }

  function getDiamond(i) {
    // Mirror DIAMONDS_VS: height matches the rect band for this row/branch,
    // width is min(aspect cap, cycle-based cap) so diamonds have a fixed
    // aspect when zoomed in and narrow with x-zoom-out. All gates share
    // the same formula (no per-gate refDur), so RZ / RX shift-phase / SX
    // shift-phase render identically. Floored at 1 px total width.
    const branch = BRANCHES[branchIds[i]] || "main";
    const bOff = BRANCH_Y[branch] || BRANCH_Y.main;
    const row = channelRow[channelIds[i]];
    const yLow = row + bOff.low;
    const yHigh = row + bOff.high;
    const cx = dataXtoCx(starts[i]);
    const pyTop = dataYtoCy(yHigh);
    const pyBot = dataYtoCy(yLow);
    const cy = (pyTop + pyBot) * 0.5;
    const halfH = Math.max(0.5, (pyBot - pyTop) * 0.5);
    const pxPerCycle = plotW() / (xMax - xMin);
    const halfW = Math.max(
      0.5,
      Math.min(DIAMOND_ASPECT * halfH, DIAMOND_WIDTH_CYCLES * pxPerCycle),
    );
    return { cx, cy, halfW, halfH };
  }

  // =========================================================================
  // Nice-numbers x-axis tick generation
  // =========================================================================
  function niceTicks(min, max, pxWidth) {
    const range = max - min;
    if (range <= 0) return [];
    const target = Math.max(4, Math.floor(pxWidth / 80));
    const raw = range / target;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const nice = [1, 2, 5, 10];
    let step = raw;
    for (const n of nice) {
      if (n * mag >= raw) { step = n * mag; break; }
    }
    if (!step || step <= 0) step = raw;
    const first = Math.ceil(min / step) * step;
    const ticks = [];
    for (let t = first; t <= max + step * 0.01; t += step) ticks.push(t);
    return ticks;
  }

  // Pick a time unit whose scaled value of the given range in seconds is
  // >= 1 — so tick labels come out as e.g. "12.3 µs" rather than
  // "0.0000123 s". Returns { scale (multiplier from seconds), label }.
  function pickTimeUnit(secondsRange) {
    const units = [
      { scale: 1,       label: "s"  },
      { scale: 1e3,     label: "ms" },
      { scale: 1e6,     label: "\u00B5s" },
      { scale: 1e9,     label: "ns" },
      { scale: 1e12,    label: "ps" },
      { scale: 1e15,    label: "fs" },
    ];
    for (const u of units) {
      if (secondsRange * u.scale >= 1) return u;
    }
    return units[units.length - 1];
  }

  // Format a cycle-count tick as a time value in the chosen unit. Trims
  // trailing zeros so "12.30 µs" becomes "12.3 µs".
  function formatTimeTick(cycles, dtNsVal, unitScale) {
    const seconds = cycles * dtNsVal * 1e-9;
    const v = seconds * unitScale;
    let s = v.toPrecision(4);
    // Drop trailing zeros after the decimal point, and the point itself
    // when nothing follows.
    if (s.indexOf(".") >= 0 && s.indexOf("e") < 0) {
      s = s.replace(/0+$/, "").replace(/\.$/, "");
    }
    return s;
  }

  // =========================================================================
  // WebGL2 setup
  // =========================================================================
  function initGL() {
    gl = glCanvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false });
    if (!gl) {
      showGlError();
      return false;
    }
    glCanvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      showGlError("WebGL context lost. Refresh the page to restore rendering.");
    });
    glCanvas.addEventListener("webglcontextrestored", () => {
      hideGlError();
      compilePrograms();
      buildGlyphAtlas();
      buildPaletteTexture();
      rebuildInstanceBuffers();
      buildLookupTextures();
      markDirty();
    });
    compilePrograms();
    buildPaletteTexture();
    buildGlyphAtlas();
    return true;
  }

  function showGlError(msg) {
    let banner = document.getElementById("gl-error-banner");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "gl-error-banner";
      banner.innerHTML =
        "<h2>WebGL2 is required</h2>" +
        "<p>This viewer needs a WebGL2 rendering context, which your browser does not " +
        "appear to provide. Please try a recent version of Chrome, Firefox, Edge or " +
        "Safari 15+.</p>";
      container.appendChild(banner);
    }
    if (msg) banner.querySelector("p").textContent = msg;
    banner.style.display = "block";
  }
  function hideGlError() {
    const b = document.getElementById("gl-error-banner");
    if (b) b.style.display = "none";
  }

  function compileShader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("Shader compile error:", gl.getShaderInfoLog(s), src);
      gl.deleteShader(s);
      return null;
    }
    return s;
  }
  function linkProgram(vsSrc, fsSrc) {
    const vs = compileShader(gl.VERTEX_SHADER, vsSrc);
    const fs = compileShader(gl.FRAGMENT_SHADER, fsSrc);
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.error("Program link error:", gl.getProgramInfoLog(p));
      return null;
    }
    return p;
  }

  const RECTS_VS = `#version 300 es
  precision highp float;
  precision highp usampler2D;
  layout(location=0) in vec2  aStartFinish; // start, finish (data-space x)
  layout(location=1) in float aChannelId;
  layout(location=2) in float aGateId;
  layout(location=3) in float aBranchId;
  layout(location=4) in float aFlags;       // bit0 = isZeroDuration, bit1 = isBarrier

  uniform vec2 uPlotOrigin;
  uniform vec2 uPlotSize;
  uniform vec2 uDataMin;
  uniform vec2 uDataSize;
  uniform vec2 uResolution;

  uniform usampler2D uChannelRow;    // channelId → row index (R16UI)
  uniform usampler2D uChanVisible;   // channelId → 0/1 (R8UI)
  uniform usampler2D uGateVisible;   // gateId    → 0/1 (R8UI)
  uniform usampler2D uBranchVisible; // branchId  → 0/1 (R8UI)
  uniform float uBranchLow[3];
  uniform float uBranchHigh[3];
  uniform float uBarrierPad;
  uniform int   uFilterBarrier;

  out vec2 vLocal;
  out vec2 vSizePx;
  flat out float vGateId;
  flat out float vFlags;

  void main() {
    int iChan   = int(aChannelId);
    int iGate   = int(aGateId);
    int iBranch = int(aBranchId);
    int iFlags  = int(aFlags);
    bool isBarrier      = (iFlags & 2) != 0;
    bool isZeroDuration = (iFlags & 1) != 0;

    uint vChan   = texelFetch(uChanVisible,   ivec2(iChan,   0), 0).r;
    uint vGate   = texelFetch(uGateVisible,   ivec2(iGate,   0), 0).r;
    uint vBranch = texelFetch(uBranchVisible, ivec2(iBranch, 0), 0).r;
    bool visible = vChan > 0u && vGate > 0u && vBranch > 0u
                && !(uFilterBarrier != 0 && isBarrier);

    // Rects program handles non-zero-duration instances only.
    if (!visible || isZeroDuration) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      vLocal = vec2(0.0);
      vSizePx = vec2(0.0);
      vGateId = aGateId;
      vFlags = aFlags;
      return;
    }

    uint  uRow  = texelFetch(uChannelRow, ivec2(iChan, 0), 0).r;
    float rowF  = float(uRow);
    float yLow  = rowF + uBranchLow[iBranch]  - (isBarrier ? uBarrierPad : 0.0);
    float yHigh = rowF + uBranchHigh[iBranch] + (isBarrier ? uBarrierPad : 0.0);

    float px1 = uPlotOrigin.x + (aStartFinish.x - uDataMin.x) / uDataSize.x * uPlotSize.x;
    float px2 = uPlotOrigin.x + (aStartFinish.y - uDataMin.x) / uDataSize.x * uPlotSize.x;
    px2 = max(px2, px1 + 1.0);
    float py1 = uPlotOrigin.y + (1.0 - (yHigh - uDataMin.y) / uDataSize.y) * uPlotSize.y;
    float py2 = uPlotOrigin.y + (1.0 - (yLow  - uDataMin.y) / uDataSize.y) * uPlotSize.y;

    vec2 cornerUV = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    vec2 pxPos = mix(vec2(px1, py1), vec2(px2, py2), cornerUV);

    vec2 clip = pxPos / uResolution * 2.0 - 1.0;
    clip.y = -clip.y;
    gl_Position = vec4(clip, 0.0, 1.0);

    vLocal = cornerUV;
    vSizePx = vec2(px2 - px1, py2 - py1);
    vGateId = aGateId;
    vFlags = aFlags;
  }`;

  const RECTS_FS = `#version 300 es
  precision highp float;
  in vec2 vLocal;
  in vec2 vSizePx;
  flat in float vGateId;
  flat in float vFlags;
  uniform sampler2D uPalette;
  out vec4 outColor;

  void main() {
    ivec2 palCoord = ivec2(int(vGateId) - (int(vGateId) / 10) * 10, 0);
    vec3 baseCol = texelFetch(uPalette, palCoord, 0).rgb;

    // Distance from nearest edge in pixel units
    vec2 edgePx = min(vLocal, 1.0 - vLocal) * vSizePx;
    float edge = min(edgePx.x, edgePx.y);
    float borderWidth = 1.0;
    // Draw border only if rect is wider than MIN_STROKE_PX in the narrower axis
    float minSide = min(vSizePx.x, vSizePx.y);
    if (minSide > 2.0 && edge < borderWidth) {
      // darken edge
      outColor = vec4(baseCol * 0.55, 1.0);
    } else {
      outColor = vec4(baseCol, 1.0);
    }
  }`;

  const DIAMONDS_VS = `#version 300 es
  precision highp float;
  precision highp usampler2D;
  layout(location=0) in vec2  aStartFinish;
  layout(location=1) in float aChannelId;
  layout(location=2) in float aGateId;
  layout(location=3) in float aBranchId;
  layout(location=4) in float aFlags;

  uniform vec2 uPlotOrigin;
  uniform vec2 uPlotSize;
  uniform vec2 uDataMin;
  uniform vec2 uDataSize;
  uniform vec2 uResolution;

  uniform usampler2D uChannelRow;
  uniform usampler2D uChanVisible;
  uniform usampler2D uGateVisible;
  uniform usampler2D uBranchVisible;
  uniform float uBranchLow[3];
  uniform float uBranchHigh[3];
  uniform float uBarrierPad;
  uniform int   uFilterBarrier;

  out vec2 vLocal;
  flat out float vGateId;
  flat out float vHalfW;
  flat out float vHalfH;

  void main() {
    int iChan   = int(aChannelId);
    int iGate   = int(aGateId);
    int iBranch = int(aBranchId);
    int iFlags  = int(aFlags);
    bool isBarrier      = (iFlags & 2) != 0;
    bool isZeroDuration = (iFlags & 1) != 0;

    uint vChan   = texelFetch(uChanVisible,   ivec2(iChan,   0), 0).r;
    uint vGate   = texelFetch(uGateVisible,   ivec2(iGate,   0), 0).r;
    uint vBranch = texelFetch(uBranchVisible, ivec2(iBranch, 0), 0).r;
    bool visible = vChan > 0u && vGate > 0u && vBranch > 0u
                && !(uFilterBarrier != 0 && isBarrier);

    // Diamonds program handles zero-duration instances only.
    if (!visible || !isZeroDuration) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      vLocal = vec2(0.0);
      vGateId = aGateId;
      vHalfW = 0.0;
      vHalfH = 0.0;
      return;
    }

    uint  uRow  = texelFetch(uChannelRow, ivec2(iChan, 0), 0).r;
    float rowF  = float(uRow);
    float yLow  = rowF + uBranchLow[iBranch];
    float yHigh = rowF + uBranchHigh[iBranch];

    float cx   = uPlotOrigin.x + (aStartFinish.x - uDataMin.x) / uDataSize.x * uPlotSize.x;
    float cyHi = uPlotOrigin.y + (1.0 - (yHigh - uDataMin.y) / uDataSize.y) * uPlotSize.y;
    float cyLo = uPlotOrigin.y + (1.0 - (yLow  - uDataMin.y) / uDataSize.y) * uPlotSize.y;
    float halfH = max(0.5, (cyLo - cyHi) * 0.5);
    float cy = (cyLo + cyHi) * 0.5;

    float pxPerCycle = uPlotSize.x / uDataSize.x;
    float halfW = max(0.5, min(
      ${DIAMOND_ASPECT.toFixed(4)} * halfH,
      ${DIAMOND_WIDTH_CYCLES.toFixed(4)} * pxPerCycle
    ));

    int c = gl_VertexID % 4;
    vec2 offset;
    if (c == 0)      offset = vec2(-halfW, 0.0);
    else if (c == 1) offset = vec2(0.0, -halfH);
    else if (c == 2) offset = vec2( halfW, 0.0);
    else             offset = vec2(0.0,  halfH);
    vec2 pxPos = vec2(cx, cy) + offset;
    vec2 clip = pxPos / uResolution * 2.0 - 1.0;
    clip.y = -clip.y;
    gl_Position = vec4(clip, 0.0, 1.0);

    if (c == 0)      vLocal = vec2(-1.0, 0.0);
    else if (c == 1) vLocal = vec2(0.0, -1.0);
    else if (c == 2) vLocal = vec2( 1.0, 0.0);
    else             vLocal = vec2(0.0,  1.0);
    vGateId = aGateId;
    vHalfW = halfW;
    vHalfH = halfH;
  }`;

  const DIAMONDS_FS = `#version 300 es
  precision highp float;
  in vec2 vLocal;
  flat in float vGateId;
  flat in float vHalfW;
  flat in float vHalfH;
  uniform sampler2D uPalette;
  out vec4 outColor;
  void main() {
    ivec2 palCoord = ivec2(int(vGateId) - (int(vGateId) / 10) * 10, 0);
    vec3 baseCol = texelFetch(uPalette, palCoord, 0).rgb;
    // Thin (~1 px) darker rim at the diamond perimeter, anti-aliased via
    // the screen-space derivative of the Manhattan distance so it stays
    // 1-pixel wide regardless of diamond size. Skipped for very small
    // diamonds — matches RECTS_FS's minSide > 2 gate.
    float d = abs(vLocal.x) + abs(vLocal.y);
    float minSidePx = 2.0 * min(vHalfW, vHalfH);
    if (minSidePx > 2.0) {
      float fd = fwidth(d);
      float rim = smoothstep(1.0 - fd, 1.0, d);
      outColor = vec4(mix(baseCol, baseCol * 0.55, rim), 1.0);
    } else {
      outColor = vec4(baseCol, 1.0);
    }
  }`;

  const TEXT_VS = `#version 300 es
  precision highp float;
  layout(location=0) in vec4 aQuad;  // pxX, pxY, pxW, pxH
  layout(location=1) in vec4 aUV;    // u0, v0, u1, v1
  uniform vec2 uResolution;
  out vec2 vUV;
  void main() {
    int corner = gl_VertexID;
    vec2 cornerUV = vec2(float(corner & 1), float((corner >> 1) & 1));
    vec2 px = aQuad.xy + cornerUV * aQuad.zw;
    vec2 clip = px / uResolution * 2.0 - 1.0;
    clip.y = -clip.y;
    gl_Position = vec4(clip, 0.0, 1.0);
    vUV = mix(aUV.xy, aUV.zw, cornerUV);
  }`;

  const TEXT_FS = `#version 300 es
  precision highp float;
  in vec2 vUV;
  uniform sampler2D uGlyphs;
  uniform vec3 uColor;
  out vec4 outColor;
  void main() {
    float a = texture(uGlyphs, vUV).r;
    outColor = vec4(uColor, a);
  }`;

  function compilePrograms() {
    rectsProg = linkProgram(RECTS_VS, RECTS_FS);
    diamondsProg = linkProgram(DIAMONDS_VS, DIAMONDS_FS);
    textProg = linkProgram(TEXT_VS, TEXT_FS);
  }

  function buildPaletteTexture() {
    paletteTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, paletteTex);
    const pixels = new Uint8Array(COLORS.length * 3);
    for (let i = 0; i < COLORS.length; i++) {
      const hex = COLORS[i];
      pixels[i * 3 + 0] = parseInt(hex.substr(1, 2), 16);
      pixels[i * 3 + 1] = parseInt(hex.substr(3, 2), 16);
      pixels[i * 3 + 2] = parseInt(hex.substr(5, 2), 16);
    }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, COLORS.length, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, pixels);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  // Build a bitmap glyph atlas via Canvas 2D. Store per-char {u, v, w, h, ax}.
  function buildGlyphAtlas() {
    const fontPx = 22;
    const pad = 2;
    const off = document.createElement("canvas");
    off.width = glyphAtlasSize;
    off.height = glyphAtlasSize;
    const octx = off.getContext("2d");
    octx.clearRect(0, 0, off.width, off.height);
    octx.font = `${fontPx}px "Segoe UI", system-ui, -apple-system, sans-serif`;
    octx.textBaseline = "alphabetic";
    octx.fillStyle = "#fff";
    glyphMap = new Map();
    let x = pad, y = pad, rowH = 0;
    for (let code = 32; code < 127; code++) {
      const ch = String.fromCharCode(code);
      const m = octx.measureText(ch);
      const w = Math.max(1, Math.ceil(m.width)) + pad * 2;
      const h = fontPx + pad * 2;
      if (x + w > off.width) { x = pad; y += rowH + pad; rowH = 0; }
      octx.fillText(ch, x + pad, y + fontPx);
      glyphMap.set(ch, {
        u: x / off.width,
        v: y / off.height,
        u2: (x + w) / off.width,
        v2: (y + h) / off.height,
        w, h, ax: m.width,
      });
      x += w + pad;
      if (h > rowH) rowH = h;
    }
    // Upload as R8 (extract alpha channel)
    const img = octx.getImageData(0, 0, off.width, off.height).data;
    const r8 = new Uint8Array(off.width * off.height);
    for (let i = 0; i < r8.length; i++) r8[i] = img[i * 4 + 3];
    glyphTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, glyphTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, off.width, off.height, 0, gl.RED, gl.UNSIGNED_BYTE, r8);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    glyphAtlasReady = true;
  }

  // Stable per-instance buffer layout (written once at load, never rebuilt):
  //   [0..7]   vec2 aStartFinish  (start, finish) — data-space x
  //   [8..11]  float aChannelId
  //   [12..15] float aGateId
  //   [16..19] float aBranchId
  //   [20..23] float aFlags       (bit0=zeroDur, bit1=barrier)
  // Total stride = 24 bytes.
  const INSTANCE_STRIDE = 24;

  function rebuildInstanceBuffers() {
    if (!gl) return;
    if (!instanceVBO) instanceVBO = gl.createBuffer();
    if (!textVBO) textVBO = gl.createBuffer();
    const buf = new ArrayBuffer(N * INSTANCE_STRIDE);
    const fv = new Float32Array(buf);
    for (let i = 0; i < N; i++) {
      const base = (i * INSTANCE_STRIDE) >> 2;  // float32 index
      fv[base + 0] = starts[i];
      fv[base + 1] = finishes[i];
      fv[base + 2] = channelIds[i];
      fv[base + 3] = gateIds[i];
      fv[base + 4] = branchIds[i];
      let flags = 0;
      if (isZeroDuration[i]) flags |= 1;
      if (barrierGateId >= 0 && gateIds[i] === barrierGateId) flags |= 2;
      fv[base + 5] = flags;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceVBO);
    gl.bufferData(gl.ARRAY_BUFFER, fv, gl.STATIC_DRAW);
  }

  // ---------------------------------------------------------------------------
  // GPU lookup textures — one per mutable dimension; tiny (<<1 KB each).
  // Allocated/sized in buildLookupTextures(), uploaded on any state change.
  // ---------------------------------------------------------------------------

  function _allocTex1D(oldTex, width, internalFmt, fmt, type) {
    const t = oldTex || gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, width, 1, 0, fmt, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  function buildLookupTextures() {
    if (!gl) return;
    const nc = Math.max(1, channels.length);
    const ng = Math.max(1, gates.length);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    chanRowTex   = _allocTex1D(chanRowTex,   nc, gl.R16UI, gl.RED_INTEGER, gl.UNSIGNED_SHORT);
    chanVisTex   = _allocTex1D(chanVisTex,   nc, gl.R8UI, gl.RED_INTEGER, gl.UNSIGNED_BYTE);
    gateVisTex   = _allocTex1D(gateVisTex,   ng, gl.R8UI, gl.RED_INTEGER, gl.UNSIGNED_BYTE);
    branchVisTex = _allocTex1D(branchVisTex,  3, gl.R8UI, gl.RED_INTEGER, gl.UNSIGNED_BYTE);
    uploadAllLookupTextures();
  }

  function uploadChannelRowTex() {
    if (!gl || !chanRowTex || !channels.length) return;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, chanRowTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, channels.length, 1,
      gl.RED_INTEGER, gl.UNSIGNED_SHORT, channelRowData);
  }

  function uploadChanVisTex() {
    if (!gl || !chanVisTex || !channels.length) return;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, chanVisTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, channels.length, 1,
      gl.RED_INTEGER, gl.UNSIGNED_BYTE, chanVisible);
  }

  function uploadGateVisTex() {
    if (!gl || !gateVisTex || !gates.length) return;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, gateVisTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gates.length, 1,
      gl.RED_INTEGER, gl.UNSIGNED_BYTE, gateVisible);
  }

  function uploadBranchVisTex() {
    if (!gl || !branchVisTex) return;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, branchVisTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 3, 1,
      gl.RED_INTEGER, gl.UNSIGNED_BYTE, branchVisible);
  }

  function uploadAllLookupTextures() {
    uploadChannelRowTex();
    uploadChanVisTex();
    uploadGateVisTex();
    uploadBranchVisTex();
  }

  // Bind the per-instance stable VBO to the fixed attribute locations declared
  // in both vertex shaders (layout(location=N)):
  //   0: vec2  aStartFinish (offset  0)
  //   1: float aChannelId   (offset  8)
  //   2: float aGateId      (offset 12)
  //   3: float aBranchId    (offset 16)
  //   4: float aFlags       (offset 20)
  function bindInstanceAttribs() {
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceVBO);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, INSTANCE_STRIDE, 0);
    gl.vertexAttribDivisor(0, 1);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, INSTANCE_STRIDE, 8);
    gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, INSTANCE_STRIDE, 12);
    gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, INSTANCE_STRIDE, 16);
    gl.vertexAttribDivisor(3, 1);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.FLOAT, false, INSTANCE_STRIDE, 20);
    gl.vertexAttribDivisor(4, 1);
  }

  function setCommonUniforms(prog) {
    gl.uniform2f(gl.getUniformLocation(prog, "uPlotOrigin"), PAD_LEFT, PAD_TOP);
    gl.uniform2f(gl.getUniformLocation(prog, "uPlotSize"), plotW(), plotH());
    gl.uniform2f(gl.getUniformLocation(prog, "uDataMin"), xMin, yMin);
    gl.uniform2f(gl.getUniformLocation(prog, "uDataSize"), xMax - xMin, yMax - yMin);
    gl.uniform2f(gl.getUniformLocation(prog, "uResolution"),
                 glCanvas.clientWidth, glCanvas.clientHeight);
  }

  // Bind the four small lookup textures and upload branch/barrier uniforms.
  // Texture units:  0 = uPalette (bound by caller),  1–4 = lookup tables.
  function setVisibilityUniforms(prog) {
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, chanRowTex);
    gl.uniform1i(gl.getUniformLocation(prog, "uChannelRow"), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, chanVisTex);
    gl.uniform1i(gl.getUniformLocation(prog, "uChanVisible"), 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, gateVisTex);
    gl.uniform1i(gl.getUniformLocation(prog, "uGateVisible"), 3);

    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, branchVisTex);
    gl.uniform1i(gl.getUniformLocation(prog, "uBranchVisible"), 4);

    const lowArr  = BRANCHES.map((b) => (BRANCH_Y[b] || BRANCH_Y.main).low);
    const highArr = BRANCHES.map((b) => (BRANCH_Y[b] || BRANCH_Y.main).high);
    gl.uniform1fv(gl.getUniformLocation(prog, "uBranchLow"),  lowArr);
    gl.uniform1fv(gl.getUniformLocation(prog, "uBranchHigh"), highArr);
    gl.uniform1f(gl.getUniformLocation(prog, "uBarrierPad"), BARRIER_PAD);
    gl.uniform1i(gl.getUniformLocation(prog, "uFilterBarrier"), filterBarrier ? 1 : 0);
  }

  // =========================================================================
  // Render passes
  // =========================================================================
  function renderGL() {
    if (!gl) return;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    // Clear the full canvas (with scissor disabled) so gutters stay white.
    gl.disable(gl.SCISSOR_TEST);
    gl.clearColor(1, 1, 1, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Constrain rects / diamonds / text to the plot rectangle. GL scissor
    // uses drawing-buffer pixels with origin at BOTTOM-left, so the y
    // coordinate is flipped relative to CSS.
    const scX = Math.round(PAD_LEFT * dpr);
    const scY = Math.round(
        (glCanvas.clientHeight - PAD_TOP - plotH()) * dpr);
    const scW = Math.round(plotW() * dpr);
    const scH = Math.round(plotH() * dpr);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(scX, scY, scW, scH);

    // Rects
    if (rectsProg && N > 0) {
      gl.useProgram(rectsProg);
      bindInstanceAttribs();
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, paletteTex);
      gl.uniform1i(gl.getUniformLocation(rectsProg, "uPalette"), 0);
      setCommonUniforms(rectsProg);
      setVisibilityUniforms(rectsProg);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, N);
    }

    // Diamonds (zero-duration events)
    if (diamondsProg && N > 0) {
      gl.useProgram(diamondsProg);
      bindInstanceAttribs();
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, paletteTex);
      gl.uniform1i(gl.getUniformLocation(diamondsProg, "uPalette"), 0);
      setCommonUniforms(diamondsProg);
      setVisibilityUniforms(diamondsProg);
      gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, N);
    }

    // Text labels
    renderText();

    gl.disable(gl.SCISSOR_TEST);
  }

  function ensureTextFloats(needFloats) {
    if (needFloats <= textFloats.length) return;
    let cap = textFloats.length;
    while (cap < needFloats) cap *= 2;
    const next = new Float32Array(cap);
    next.set(textFloats);
    textFloats = next;
  }

  function renderText() {
    if (!textProg || !glyphAtlasReady) return;
    // LOD cutoff: skip the whole pass when even a MIN_LABEL_CYCLES-long
    // instruction can't reach MIN_LABEL_PX.
    const pxPerCycle = plotW() / (xMax - xMin);
    if (pxPerCycle * MIN_LABEL_CYCLES < MIN_LABEL_PX) return;

    textFloatsUsed = 0;
    const nRows = channelOrder.length;
    // Y-viewport row range (inclusive). yMax is top row, yMin is bottom.
    const rowTop = Math.min(nRows - 1, Math.max(0, Math.ceil(cyToDataY(PAD_TOP))));
    const rowBot = Math.min(nRows - 1, Math.max(0, Math.floor(cyToDataY(PAD_TOP + plotH()))));
    const rFirst = Math.min(rowTop, rowBot);
    const rLast  = Math.max(rowTop, rowBot);
    for (let r = rFirst; r <= rLast; r++) {
      const cb = channelBuckets[channelOrder[r]];
      if (!cb || cb.idxArr.length === 0) continue;
      const lo = lowerBound(cb.startArr, xMin);
      const hi = upperBound(cb.startArr, xMax);
      for (let k = lo; k < hi; k++) {
        const i = cb.idxArr[k];
        if (!visible[i] || isZeroDuration[i]) continue;
        if (finishes[i] < xMin) continue;
        const rect = getRect(i);
        if (rect.w < MIN_LABEL_PX) continue;
        const fontH = 14;
        if (rect.h < fontH + 1) continue;   // vertical fit: skip when rect too short
        // Try progressively shorter labels. Falling back from
        // "<gate>_<pulse>" to just "<gate>" keeps moderately-wide rects
        // labeled instead of all-or-nothing on the long name.
        const gate = gates[gateIds[i]];
        const pulse = pulseNames[pulseNameIds[i]];
        const candidates = pulse ? [`${gate}_${pulse}`, gate] : [gate];
        const maxW = rect.w - 6;
        let label = null, w = 0;
        for (let ci = 0; ci < candidates.length; ci++) {
          const cand = candidates[ci];
          let cw = 0;
          for (let m = 0; m < cand.length; m++) {
            const g = glyphMap.get(cand[m]);
            if (g) cw += g.ax;
          }
          if (cw <= maxW) { label = cand; w = cw; break; }
        }
        if (label === null) continue;       // no candidate fits
        let cx = rect.x + (rect.w - w) / 2;
        const cy = rect.y + (rect.h - fontH) / 2;
        ensureTextFloats(textFloatsUsed + label.length * 8);
        for (let m = 0; m < label.length; m++) {
          const g = glyphMap.get(label[m]);
          if (!g) continue;
          const scale = fontH / (g.h - 4);
          const w2 = g.w * scale, h2 = g.h * scale;
          textFloats[textFloatsUsed + 0] = cx;
          textFloats[textFloatsUsed + 1] = cy;
          textFloats[textFloatsUsed + 2] = w2;
          textFloats[textFloatsUsed + 3] = h2;
          textFloats[textFloatsUsed + 4] = g.u;
          textFloats[textFloatsUsed + 5] = g.v;
          textFloats[textFloatsUsed + 6] = g.u2;
          textFloats[textFloatsUsed + 7] = g.v2;
          textFloatsUsed += 8;
          cx += g.ax * scale;
        }
      }
    }
    if (textFloatsUsed === 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, textVBO);
    // Upload only the used prefix (WebGL2 supports srcOffset + length).
    gl.bufferData(gl.ARRAY_BUFFER, textFloats, gl.DYNAMIC_DRAW, 0, textFloatsUsed);

    gl.useProgram(textProg);
    const locQuad = gl.getAttribLocation(textProg, "aQuad");
    const locUV = gl.getAttribLocation(textProg, "aUV");
    gl.enableVertexAttribArray(locQuad);
    gl.vertexAttribPointer(locQuad, 4, gl.FLOAT, false, 32, 0);
    gl.vertexAttribDivisor(locQuad, 1);
    gl.enableVertexAttribArray(locUV);
    gl.vertexAttribPointer(locUV, 4, gl.FLOAT, false, 32, 16);
    gl.vertexAttribDivisor(locUV, 1);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, glyphTex);
    gl.uniform1i(gl.getUniformLocation(textProg, "uGlyphs"), 0);
    gl.uniform2f(gl.getUniformLocation(textProg, "uResolution"),
                 glCanvas.clientWidth, glCanvas.clientHeight);
    gl.uniform3f(gl.getUniformLocation(textProg, "uColor"), 0.05, 0.05, 0.05);
    const nGlyphs = textFloatsUsed / 8;
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, nGlyphs);
    // Disable divisors so future non-instanced draws behave.
    gl.vertexAttribDivisor(locQuad, 0);
    gl.vertexAttribDivisor(locUV, 0);
  }

  // =========================================================================
  // 2D overlay: axes, ticks, y-labels, tooltip position, selection band,
  // highlight strokes.
  // =========================================================================
  function renderOverlay() {
    const w = overlayCanvas.clientWidth;
    const h = overlayCanvas.clientHeight;
    overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    overlayCtx.clearRect(0, 0, w, h);

    // Plot border
    overlayCtx.strokeStyle = "#666";
    overlayCtx.lineWidth = 1;
    overlayCtx.strokeRect(PAD_LEFT, PAD_TOP, plotW(), plotH());

    // FPS counter (top-right of plot area, discrete grey text)
    if (smoothedFps > 0) {
      overlayCtx.save();
      overlayCtx.fillStyle = "#999";
      overlayCtx.font = "10px system-ui, sans-serif";
      overlayCtx.textAlign = "right";
      overlayCtx.textBaseline = "top";
      overlayCtx.fillText(
        Math.round(smoothedFps) + " fps",
        PAD_LEFT + plotW() - 4,
        PAD_TOP + 2,
      );
      overlayCtx.restore();
    }

    // X-axis ticks and labels
    overlayCtx.strokeStyle = "#aaa";
    overlayCtx.fillStyle = "#333";
    overlayCtx.font = "11px system-ui, sans-serif";
    overlayCtx.textAlign = "center";
    overlayCtx.textBaseline = "top";
    const ticks = niceTicks(xMin, xMax, plotW());
    // When dtNs > 0 pick a single unit for all ticks so labels share scale.
    let timeUnit = null;
    if (dtNs > 0) {
      const secondsRange = (xMax - xMin) * dtNs * 1e-9;
      timeUnit = pickTimeUnit(secondsRange);
    }
    for (const t of ticks) {
      const x = dataXtoCx(t);
      if (x < PAD_LEFT - 0.5 || x > PAD_LEFT + plotW() + 0.5) continue;
      overlayCtx.beginPath();
      overlayCtx.moveTo(x, PAD_TOP + plotH());
      overlayCtx.lineTo(x, PAD_TOP + plotH() + 4);
      overlayCtx.stroke();
      const label = timeUnit
        ? formatTimeTick(t, dtNs, timeUnit.scale)
        : String(t);
      overlayCtx.fillText(label, x, PAD_TOP + plotH() + 6);
    }
    // X-axis title
    overlayCtx.font = "12px system-ui, sans-serif";
    overlayCtx.fillText(
      timeUnit ? "Time (" + timeUnit.label + ")" : "Cycles",
      PAD_LEFT + plotW() / 2,
      PAD_TOP + plotH() + 22,
    );

    // Y-axis channel labels — decimate stride so labels don't overlap when
    // channel count is high (or the y-axis is zoomed out).
    overlayCtx.textAlign = "right";
    overlayCtx.textBaseline = "middle";
    overlayCtx.font = "11px system-ui, sans-serif";
    overlayCtx.fillStyle = "#333";
    const labelFontLine = 13;
    const dyPerRow = plotH() / Math.max(1, yMax - yMin);
    const yStride = Math.max(1, Math.ceil(labelFontLine / dyPerRow));
    for (let r = 0; r < channelOrder.length; r += yStride) {
      const c = channelOrder[r];
      const y = dataYtoCy(r);
      if (y < PAD_TOP - 8 || y > PAD_TOP + plotH() + 8) continue;
      overlayCtx.fillText(channels[c], PAD_LEFT - 6, y);
    }

    // Zoom selection band — full-height X slice, full-width Y slice, or 2D box
    if (zoomBox) {
      const plotL = PAD_LEFT;
      const plotR = PAD_LEFT + plotW();
      const plotT = PAD_TOP;
      const plotB = PAD_TOP + plotH();
      let x1, x2, y1, y2;
      if (zoomBox.mode === "yOnly") {
        x1 = plotL; x2 = plotR;
        const rawY1 = Math.min(zoomBox.startCy, zoomBox.endCy);
        const rawY2 = Math.max(zoomBox.startCy, zoomBox.endCy);
        y1 = Math.max(plotT, rawY1);
        y2 = Math.min(plotB, rawY2);
      } else if (zoomBox.mode === "xy") {
        x1 = Math.min(zoomBox.startCx, zoomBox.endCx);
        x2 = Math.max(zoomBox.startCx, zoomBox.endCx);
        const rawY1 = Math.min(zoomBox.startCy, zoomBox.endCy);
        const rawY2 = Math.max(zoomBox.startCy, zoomBox.endCy);
        y1 = Math.max(plotT, rawY1);
        y2 = Math.min(plotB, rawY2);
      } else {  // xOnly
        x1 = Math.min(zoomBox.startCx, zoomBox.endCx);
        x2 = Math.max(zoomBox.startCx, zoomBox.endCx);
        y1 = plotT; y2 = plotB;
      }
      overlayCtx.fillStyle = "rgba(80,80,220,0.12)";
      overlayCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
      overlayCtx.strokeStyle = "rgba(80,80,220,0.75)";
      overlayCtx.setLineDash([4, 4]);
      overlayCtx.beginPath();
      // Vertical edges: drawn for xOnly and xy (they bound X).
      if (zoomBox.mode !== "yOnly") {
        overlayCtx.moveTo(x1, y1);
        overlayCtx.lineTo(x1, y2);
        overlayCtx.moveTo(x2, y1);
        overlayCtx.lineTo(x2, y2);
      }
      // Horizontal edges: drawn for yOnly and xy (they bound Y).
      if (zoomBox.mode !== "xOnly") {
        overlayCtx.moveTo(x1, y1);
        overlayCtx.lineTo(x2, y1);
        overlayCtx.moveTo(x1, y2);
        overlayCtx.lineTo(x2, y2);
      }
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);
    }

    // Highlights: search hits, hovered gate group, hovered item.
    // Both iterate only the row-viewport slice via channelBuckets[channelOrder[r]].
    const nRowsV = channelOrder.length;
    let rFirstV = 0, rLastV = nRowsV - 1;
    if (nRowsV > 0) {
      const yTop = cyToDataY(PAD_TOP);
      const yBot = cyToDataY(PAD_TOP + plotH());
      rFirstV = Math.max(0, Math.floor(Math.min(yTop, yBot)));
      rLastV  = Math.min(nRowsV - 1, Math.ceil(Math.max(yTop, yBot)));
    }
    if (searchQuery) {
      overlayCtx.strokeStyle = "#FF6600";
      overlayCtx.lineWidth = 2;
      for (let r = rFirstV; r <= rLastV; r++) {
        const cb = channelBuckets[channelOrder[r]];
        if (!cb || cb.idxArr.length === 0) continue;
        const lo = lowerBound(cb.startArr, xMin - 4);
        const hi = upperBound(cb.startArr, xMax);
        for (let k = lo; k < hi; k++) {
          const i = cb.idxArr[k];
          if (!searchHits[i] || !visible[i]) continue;
          if (finishes[i] < xMin) continue;
          if (isZeroDuration[i]) {
            const d = getDiamond(i);
            overlayCtx.strokeRect(d.cx - d.halfW - 2, d.cy - d.halfH - 2, (d.halfW + 2) * 2, (d.halfH + 2) * 2);
          } else {
            const rr = getRect(i);
            overlayCtx.strokeRect(rr.x - 1, rr.y - 1, rr.w + 2, rr.h + 2);
          }
        }
      }
    }
    if (hoveredGate >= 0) {
      overlayCtx.strokeStyle = "#FF6600";
      overlayCtx.lineWidth = 2;
      for (let r = rFirstV; r <= rLastV; r++) {
        const cb = channelBuckets[channelOrder[r]];
        if (!cb || cb.idxArr.length === 0) continue;
        const lo = lowerBound(cb.startArr, xMin - 4);
        const hi = upperBound(cb.startArr, xMax);
        for (let k = lo; k < hi; k++) {
          const i = cb.idxArr[k];
          if (gateIds[i] !== hoveredGate) continue;
          if (!visible[i]) continue;
          if (finishes[i] < xMin) continue;
          if (isZeroDuration[i]) {
            const d = getDiamond(i);
            overlayCtx.strokeRect(d.cx - d.halfW - 2, d.cy - d.halfH - 2, (d.halfW + 2) * 2, (d.halfH + 2) * 2);
          } else {
            const rr = getRect(i);
            overlayCtx.strokeRect(rr.x - 1, rr.y - 1, rr.w + 2, rr.h + 2);
          }
        }
      }
    }
    if (hoveredIdx >= 0 && visible[hoveredIdx]) {
      overlayCtx.strokeStyle = "rgba(0,0,0,0.85)";
      overlayCtx.lineWidth = 2;
      if (isZeroDuration[hoveredIdx]) {
        const d = getDiamond(hoveredIdx);
        overlayCtx.strokeRect(d.cx - d.halfW + 1, d.cy - d.halfH + 1, d.halfW * 2 - 2, d.halfH * 2 - 2);
      } else {
        const r = getRect(hoveredIdx);
        overlayCtx.strokeRect(r.x + 1, r.y + 1, Math.max(1, r.w - 2), Math.max(1, r.h - 2));
      }
    }
  }

  // =========================================================================
  // Minimap
  // =========================================================================
  // Map minimap pixel-Y to data-Y. Row centres are at (nRows - 0.5 - r) *
  // scaleY in minimap space, so the continuous inverse is:
  //   d(my) = nRows - 0.5 - my * nRows / mmCurrentH
  function mmYToDataY(my) {
    const nRows = channelOrder.length;
    if (nRows <= 0) return 0;
    return nRows - 0.5 - (my * nRows) / mmCurrentH;
  }
  // Current viewport rectangle in minimap CSS pixels. At full Y domain this
  // rect naturally spans the whole minimap height.
  function mmViewportRect(w) {
    const nRows = channelOrder.length;
    const vx1 = (xMin / (maxTime + 1)) * w;
    const vx2 = (xMax / (maxTime + 1)) * w;
    if (nRows <= 0) {
      return { x1: vx1, x2: vx2, y1: 0, y2: mmCurrentH };
    }
    const scaleY = mmCurrentH / nRows;
    let vy1 = (nRows - 0.5 - yMax) * scaleY;
    let vy2 = (nRows - 0.5 - yMin) * scaleY;
    vy1 = Math.max(0, Math.min(mmCurrentH, vy1));
    vy2 = Math.max(0, Math.min(mmCurrentH, vy2));
    return { x1: vx1, x2: vx2, y1: vy1, y2: vy2 };
  }

  // Build (or rebuild) the minimap rects layer into an offscreen canvas.
  // Only re-runs when mmCacheDirty is set — filter toggles, reload,
  // resize — so wheel-zoom / pan / drag frames don't pay the O(N) cost.
  function refreshMinimapCache(cssW) {
    if (!mmRectsCache) mmRectsCache = document.createElement("canvas");
    const wPx = Math.round(cssW * dpr);
    const hPx = Math.round(mmCurrentH * dpr);
    if (mmRectsCache.width !== wPx) mmRectsCache.width = wPx;
    if (mmRectsCache.height !== hPx) mmRectsCache.height = hPx;
    const ctx = mmRectsCache.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, mmCurrentH);
    ctx.fillStyle = "#f5f5f5";
    ctx.fillRect(0, 0, cssW, mmCurrentH);
    const nRows = channelOrder.length;
    if (maxTime > 0 && nRows > 0) {
      const scaleX = cssW / (maxTime + 1);
      const scaleY = mmCurrentH / nRows;
      for (let i = 0; i < N; i++) {
        if (!visible[i] || isZeroDuration[i]) continue;
        const row = channelRow[channelIds[i]];
        if (row < 0) continue;
        const mx = starts[i] * scaleX;
        const mw = Math.max(1, (finishes[i] - starts[i]) * scaleX);
        const my = (nRows - 1 - row) * scaleY + 0.1 * scaleY;
        const mh = Math.max(1, 0.8 * scaleY);
        ctx.fillStyle = colorMap[gateIds[i]] || "#666";
        ctx.fillRect(mx, my, mw, mh);
      }
    }
    mmCacheDirty = false;
  }

  function renderMinimap() {
    const w = mmCanvas.clientWidth;
    const h = mmCurrentH;
    // Rebuild the cached rects layer if invalidated or if the width /
    // dpr changed since the last cache render.
    const expectedW = Math.round(w * dpr);
    if (mmCacheDirty || !mmRectsCache || mmRectsCache.width !== expectedW) {
      refreshMinimapCache(w);
    }
    // Blit the cached bitmap (device-pixel to device-pixel, no scaling).
    mmCtx.setTransform(1, 0, 0, 1, 0, 0);
    mmCtx.clearRect(0, 0, mmCanvas.width, mmCanvas.height);
    mmCtx.drawImage(mmRectsCache, 0, 0);
    // Now switch to CSS-pixel drawing for the viewport indicator / zoom box.
    mmCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Viewport indicator (2D). See mmViewportRect() for the mapping used by
    // the interaction handlers.
    if (maxTime > 0) {
      const v = mmViewportRect(w);
      mmCtx.fillStyle = "rgba(80,80,220,0.18)";
      mmCtx.fillRect(v.x1, v.y1, v.x2 - v.x1, v.y2 - v.y1);
      mmCtx.strokeStyle = "rgba(60,60,160,0.9)";
      mmCtx.lineWidth = 1;
      mmCtx.strokeRect(v.x1 + 0.5, v.y1 + 0.5,
                       Math.max(0, v.x2 - v.x1 - 1),
                       Math.max(0, v.y2 - v.y1 - 1));
    }

    // Search-hit markers: mirror the main-canvas orange outlines onto the
    // minimap so the user can find matches in the overview. Batched into
    // one stroke() call for speed on large schedules.
    if (searchQuery && maxTime > 0 && channelOrder.length > 0) {
      const nRowsMm = channelOrder.length;
      const scaleX = w / (maxTime + 1);
      const scaleY = mmCurrentH / nRowsMm;
      mmCtx.strokeStyle = "#FF6600";
      mmCtx.lineWidth = 1;
      mmCtx.beginPath();
      for (let i = 0; i < N; i++) {
        if (!searchHits[i] || !visible[i]) continue;
        const row = channelRow[channelIds[i]];
        if (row < 0) continue;
        const mx = starts[i] * scaleX;
        const mw = Math.max(1, (finishes[i] - starts[i]) * scaleX);
        const my = (nRowsMm - 1 - row) * scaleY + 0.1 * scaleY;
        const mh = Math.max(1, 0.8 * scaleY);
        mmCtx.rect(mx - 0.5, my - 0.5, mw + 1, mh + 1);
      }
      mmCtx.stroke();
      // Cursor hit: heavier outline so the user can locate "next hit".
      if (searchCursor >= 0 && searchCursor < N && searchHits[searchCursor]) {
        const i = searchCursor;
        const row = channelRow[channelIds[i]];
        if (row >= 0) {
          const mx = starts[i] * scaleX;
          const mw = Math.max(1, (finishes[i] - starts[i]) * scaleX);
          const my = (nRowsMm - 1 - row) * scaleY + 0.1 * scaleY;
          const mh = Math.max(1, 0.8 * scaleY);
          mmCtx.strokeStyle = "#CC3300";
          mmCtx.lineWidth = 2;
          mmCtx.strokeRect(mx - 1, my - 1, mw + 2, mh + 2);
        }
      }
    }

    // Hover highlights: mirror the main-plot behaviour on the minimap so
    // users can locate the hovered item(s) in the overview.
    if ((hoveredGate >= 0 || hoveredIdx >= 0) &&
        maxTime > 0 && channelOrder.length > 0) {
      const nRowsMm = channelOrder.length;
      const scaleX = w / (maxTime + 1);
      const scaleY = mmCurrentH / nRowsMm;

      // Gate-group (orange). Batch every instance into a single path and
      // issue one stroke() at the end — orders of magnitude cheaper than
      // one strokeRect per instance on big schedules (10K+ per gate).
      if (hoveredGate >= 0 && gateInstructions[hoveredGate]) {
        const arr = gateInstructions[hoveredGate];
        mmCtx.strokeStyle = "#FF6600";
        mmCtx.lineWidth = 1;
        mmCtx.beginPath();
        for (let k = 0; k < arr.length; k++) {
          const i = arr[k];
          if (!visible[i]) continue;
          const row = channelRow[channelIds[i]];
          if (row < 0) continue;
          const mx = starts[i] * scaleX;
          const mw = Math.max(1, (finishes[i] - starts[i]) * scaleX);
          const my = (nRowsMm - 1 - row) * scaleY + 0.1 * scaleY;
          const mh = Math.max(1, 0.8 * scaleY);
          mmCtx.rect(mx - 0.5, my - 0.5, mw + 1, mh + 1);
        }
        mmCtx.stroke();
      }

      // Individual (dark). One rect on top so the outline wins visually.
      if (hoveredIdx >= 0 && visible[hoveredIdx]) {
        const i = hoveredIdx;
        const row = channelRow[channelIds[i]];
        if (row >= 0) {
          const mx = starts[i] * scaleX;
          const mw = Math.max(1, (finishes[i] - starts[i]) * scaleX);
          const my = (nRowsMm - 1 - row) * scaleY + 0.1 * scaleY;
          const mh = Math.max(1, 0.8 * scaleY);
          mmCtx.strokeStyle = "rgba(0,0,0,0.85)";
          mmCtx.lineWidth = 2;
          mmCtx.strokeRect(mx - 1, my - 1, mw + 2, mh + 2);
        }
      }
    }

    if (mmZoomBox) {
      let x1, x2, y1, y2;
      if (mmZoomBox.mode === "yOnly") {
        x1 = 0; x2 = w;
        const rawY1 = Math.min(mmZoomBox.startMy, mmZoomBox.endMy);
        const rawY2 = Math.max(mmZoomBox.startMy, mmZoomBox.endMy);
        y1 = Math.max(0, rawY1);
        y2 = Math.min(h, rawY2);
      } else if (mmZoomBox.mode === "xy") {
        x1 = Math.min(mmZoomBox.startMx, mmZoomBox.endMx);
        x2 = Math.max(mmZoomBox.startMx, mmZoomBox.endMx);
        const rawY1 = Math.min(mmZoomBox.startMy, mmZoomBox.endMy);
        const rawY2 = Math.max(mmZoomBox.startMy, mmZoomBox.endMy);
        y1 = Math.max(0, rawY1);
        y2 = Math.min(h, rawY2);
      } else {  // xOnly
        x1 = Math.min(mmZoomBox.startMx, mmZoomBox.endMx);
        x2 = Math.max(mmZoomBox.startMx, mmZoomBox.endMx);
        y1 = 0; y2 = h;
      }
      mmCtx.fillStyle = "rgba(80,80,220,0.12)";
      mmCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
      mmCtx.strokeStyle = "rgba(80,80,220,0.75)";
      mmCtx.setLineDash([4, 4]);
      mmCtx.beginPath();
      if (mmZoomBox.mode !== "yOnly") {
        // Vertical edges (X bounds).
        mmCtx.moveTo(x1 + 0.5, y1);
        mmCtx.lineTo(x1 + 0.5, y2);
        mmCtx.moveTo(x2 - 0.5, y1);
        mmCtx.lineTo(x2 - 0.5, y2);
      }
      if (mmZoomBox.mode !== "xOnly") {
        // Horizontal edges (Y bounds).
        mmCtx.moveTo(x1, y1 + 0.5);
        mmCtx.lineTo(x2, y1 + 0.5);
        mmCtx.moveTo(x1, y2 - 0.5);
        mmCtx.lineTo(x2, y2 - 0.5);
      }
      mmCtx.stroke();
      mmCtx.setLineDash([]);
    }
    mmCtx.strokeStyle = "#666";
    mmCtx.lineWidth = 1;
    mmCtx.strokeRect(0.5, 0.5, w - 1, h - 1);
  }

  // =========================================================================
  // Frame loop
  // =========================================================================
  function markDirty() {
    if (dirty) return;
    dirty = true;
    requestAnimationFrame(renderFrame);
  }
  // FPS accounting: keep a rolling window of the last N frame timestamps
  // and compute smoothed FPS = (N-1) / (last - first). Drawn on the overlay
  // canvas by renderOverlay(). Dirty-flag renderer, so the display freezes
  // when idle — that's fine, it captures the interactive rate.
  const FPS_WINDOW = 30;
  let fpsTimes = [];
  let smoothedFps = 0;
  function renderFrame() {
    if (!dirty) return;
    dirty = false;
    const now = performance.now();
    fpsTimes.push(now);
    if (fpsTimes.length > FPS_WINDOW) fpsTimes.shift();
    if (fpsTimes.length >= 2) {
      const span = fpsTimes[fpsTimes.length - 1] - fpsTimes[0];
      smoothedFps = span > 0 ? ((fpsTimes.length - 1) * 1000) / span : 0;
    }
    renderGL();
    renderOverlay();
    renderMinimap();
  }

  // =========================================================================
  // Hit testing
  // =========================================================================
  function hitTest(cx, cy) {
    // Narrow candidates to at most 3 rows: the row under the cursor plus
    // the immediate neighbours (barrier rects and diamonds can bleed
    // ±BARRIER_PAD/±ZERO_DUR_HALF into adjacent rows).
    const nRows = channelOrder.length;
    if (nRows === 0) return -1;
    const dy = cyToDataY(cy);
    const centreRow = Math.round(dy);
    const rLo = Math.max(0, centreRow - 1);
    const rHi = Math.min(nRows - 1, centreRow + 1);
    // Pass 1: diamonds first (drawn on top)
    for (let r = rLo; r <= rHi; r++) {
      const cb = channelBuckets[channelOrder[r]];
      if (!cb || cb.idxArr.length === 0) continue;
      const lo = lowerBound(cb.startArr, xMin - 4);
      const hi = upperBound(cb.startArr, xMax + 4);
      for (let k = lo; k < hi; k++) {
        const i = cb.idxArr[k];
        if (!visible[i] || !isZeroDuration[i]) continue;
        const d = getDiamond(i);
        if (Math.abs(cx - d.cx) <= d.halfW + 4 &&
            Math.abs(cy - d.cy) <= d.halfH + 4) {
          return i;
        }
      }
    }
    // Pass 2: rects, reverse-order within each row (later wins on overlap)
    for (let r = rLo; r <= rHi; r++) {
      const cb = channelBuckets[channelOrder[r]];
      if (!cb || cb.idxArr.length === 0) continue;
      const lo = lowerBound(cb.startArr, xMin);
      const hi = upperBound(cb.startArr, xMax);
      for (let k = hi - 1; k >= lo; k--) {
        const i = cb.idxArr[k];
        if (!visible[i] || isZeroDuration[i]) continue;
        if (finishes[i] < xMin) continue;
        const rect = getRect(i);
        if (cx >= rect.x && cx <= rect.x + rect.w &&
            cy >= rect.y && cy <= rect.y + rect.h) {
          return i;
        }
      }
    }
    return -1;
  }

  // =========================================================================
  // Search
  // =========================================================================
  function runSearch(q) {
    searchQuery = (q || "").trim();
    // Reuse buffer: searchHits was sized to N in buildStore. Only reallocate
    // if the length is stale (e.g. after reload with different N).
    if (searchHits.length !== N) searchHits = new Uint8Array(N);
    else searchHits.fill(0);
    if (!searchQuery) { updateSearchStatus(0); markDirty(); return; }
    // Try to compile as a regex; fall back to escaped literal substring on error.
    let re;
    try {
      re = new RegExp(searchQuery, "i");
    } catch (_) {
      re = new RegExp(searchQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
    }
    let n = 0;
    for (let i = 0; i < N; i++) {
      const inst = instructions[instructionIds[i]];
      const gate = gates[gateIds[i]];
      const chan = channels[channelIds[i]];
      if (re.test(inst) || re.test(gate) || re.test(chan)) {
        searchHits[i] = 1;
        n++;
      }
    }
    updateSearchStatus(n);
    markDirty();
  }
  function updateSearchStatus(n) {
    const st = document.getElementById("search-status");
    if (!st) return;
    if (!searchQuery) { st.textContent = ""; return; }
    st.textContent = `${n} hit${n === 1 ? "" : "s"} · Enter \u21B5 to jump`;
  }
  function jumpToNextHit() {
    const start = searchCursor;
    for (let k = 1; k <= N; k++) {
      const idx = (start + k + N) % N;
      if (searchHits[idx] && visible[idx]) {
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

  // =========================================================================
  // Viewport helpers
  // =========================================================================
  function clampX() {
    const span = xMax - xMin;
    if (xMin < 0) { xMin = 0; xMax = Math.max(span, MIN_X_SPAN); }
    if (xMax > maxTime + 1) {
      xMax = maxTime + 1;
      xMin = Math.max(0, xMax - span);
    }
  }
  // Full Y-domain bounds (matches how resetView renders the fit-all case).
  function yDomainMin() { return -0.6; }
  function yDomainMax() {
    return channelOrder.length > 0 ? channelOrder.length - 0.4 : 1.0;
  }
  function clampY() {
    const dMin = yDomainMin();
    const dMax = yDomainMax();
    let span = yMax - yMin;
    // If the requested span exceeds the whole domain, snap to full.
    if (span >= dMax - dMin) {
      yMin = dMin;
      yMax = dMax;
      return;
    }
    if (yMin < dMin) { yMin = dMin; yMax = dMin + span; }
    if (yMax > dMax) { yMax = dMax; yMin = dMax - span; }
  }
  function resetView() {
    xMin = 0;
    xMax = maxTime + 1;
    // Default view is the full-height overview: all rows visible, squished
    // if there are many. The user zooms in (Shift+Wheel Y, 2D drag box, or
    // 2D minimap drag) to see detail at natural row height.
    yMin = yDomainMin();
    yMax = yDomainMax();
  }
  function resetAll() {
    if (mergeActive) {
      mergeActive = false;
      reload(rawCsv);
      return;
    }
    gateVisible.fill(1);
    branchVisible.fill(1);
    chanVisible.fill(1);
    filterBarrier = false;
    soloGate = -1;
    searchQuery = "";
    if (searchHits) searchHits.fill(0);
    applyInitialOptions();
    recomputeVisible();
    resetView();
    sidebarRefs = null;
    buildSidebar();
    markDirty();
  }

  // =========================================================================
  // Interactions
  // =========================================================================
  function onWheel(e) {
    e.preventDefault();
    const rect = glCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    // Some browsers (Chrome, Firefox) route wheel input onto deltaX when
    // Shift is held, so fall back to deltaX to preserve zoom direction.
    const rawDelta = e.deltaY || e.deltaX;
    const factor = rawDelta > 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
    if (shiftHeld) {
      const anchor = cyToDataY(cy);
      const nyMin = anchor + (yMin - anchor) * factor;
      const nyMax = anchor + (yMax - anchor) * factor;
      if (nyMax - nyMin >= MIN_Y_SPAN) {
        yMin = nyMin; yMax = nyMax;
        clampY();
      }
    } else {
      const anchor = cxToDataX(cx);
      const nMin = anchor + (xMin - anchor) * factor;
      const nMax = anchor + (xMax - anchor) * factor;
      if (nMax - nMin >= MIN_X_SPAN) {
        xMin = nMin; xMax = nMax;
        clampX();
      }
    }
    markDirty();
  }

  function onMouseDown(e) {
    if (e.button !== 0) return;
    const rect = glCanvas.getBoundingClientRect();
    dragStartX = e.clientX - rect.left;
    dragStartY = e.clientY - rect.top;
    dragXMin0 = xMin;
    dragXMax0 = xMax;
    dragYMin0 = yMin;
    dragYMax0 = yMax;
    dragMoved = false;
    if (shiftHeld) {
      dragMode = "pan";
      glCanvas.style.cursor = "grabbing";
    } else {
      dragMode = "zoom";
      zoomBox = {
        startCx: dragStartX, endCx: dragStartX,
        startCy: dragStartY, endCy: dragStartY,
        mode: "xOnly",
      };
      glCanvas.style.cursor = "col-resize";
    }
  }
  function onMouseMove(e) {
    const rect = glCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    if (dragMode === "pan") {
      const dx = cx - dragStartX;
      const dy = cy - dragStartY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) dragMoved = true;
      // X pan
      const dxData = -dx * (xMax - xMin) / plotW();
      xMin = dragXMin0 + dxData;
      xMax = dragXMax0 + dxData;
      clampX();
      // Y pan (mirror). dataYtoCy inverts Y, so dragging DOWN (dy > 0)
      // should reveal HIGHER row indices, which sit at LOWER data-Y.
      // yMin/yMax must therefore DECREASE when dy > 0 → sign is +.
      // clampY() makes this a natural no-op when already at full domain.
      const dyData = +dy * (yMax - yMin) / plotH();
      yMin = dragYMin0 + dyData;
      yMax = dragYMax0 + dyData;
      clampY();
      markDirty();
    } else if (dragMode === "zoom") {
      const dx = cx - dragStartX;
      const dy = cy - dragStartY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) dragMoved = true;
      zoomBox.endCx = cx;
      zoomBox.endCy = cy;
      // Gesture: triangular cones from the click position. A drag whose
      // direction is within a cone stays in the corresponding single-axis
      // mode; drags in the diagonal region become "xy". Once "xy", it
      // latches — a 2-D selection never downgrades back to 1-D.
      if (zoomBox.mode !== "xy") {
        const ax = Math.abs(dx), ay = Math.abs(dy);
        if (ax * ax + ay * ay >= GESTURE_MIN_PX * GESTURE_MIN_PX) {
          let newMode;
          if (ay <= ax * CONE_SLOPE)      newMode = "xOnly";
          else if (ax <= ay * CONE_SLOPE) newMode = "yOnly";
          else                            newMode = "xy";
          if (newMode !== zoomBox.mode) {
            zoomBox.mode = newMode;
            glCanvas.style.cursor =
              newMode === "xy"    ? "crosshair" :
              newMode === "yOnly" ? "row-resize" :
                                    "col-resize";
          }
        }
      }
      markDirty();
    } else {
      // Hover — only hit-test when cursor is inside the canvas
      const withinCanvas = cx >= 0 && cy >= 0 &&
          cx <= glCanvas.clientWidth && cy <= glCanvas.clientHeight;
      if (!withinCanvas) {
        if (hoveredIdx !== -1) { hoveredIdx = -1; markDirty(); }
        hideTooltip();
      } else {
        const idx = hitTest(cx, cy);
        if (idx !== hoveredIdx) { hoveredIdx = idx; markDirty(); }
        if (idx >= 0) showTooltip(idx, e.clientX, e.clientY);
        else hideTooltip();
      }
    }
  }
  function onMouseUp(e) {
    if (dragMode === "zoom" && zoomBox && dragMoved) {
      // Apply X for xOnly / xy; apply Y for yOnly / xy.
      if (zoomBox.mode !== "yOnly") {
        const x1 = Math.min(zoomBox.startCx, zoomBox.endCx);
        const x2 = Math.max(zoomBox.startCx, zoomBox.endCx);
        const nMin = cxToDataX(x1);
        const nMax = cxToDataX(x2);
        if (nMax - nMin >= MIN_X_SPAN) {
          xMin = nMin;
          xMax = nMax;
          clampX();
        }
      }
      if (zoomBox.mode !== "xOnly") {
        const y1 = Math.min(zoomBox.startCy, zoomBox.endCy);
        const y2 = Math.max(zoomBox.startCy, zoomBox.endCy);
        // Canvas Y is inverted → top pixel is highest data-Y.
        const nyMax = cyToDataY(y1);
        const nyMin = cyToDataY(y2);
        if (nyMax - nyMin >= MIN_Y_SPAN) {
          yMin = nyMin;
          yMax = nyMax;
          clampY();
        }
      }
    }
    dragMode = "none";
    zoomBox = null;
    glCanvas.style.cursor = shiftHeld ? "grab" : "crosshair";
    markDirty();
  }
  function onDblClick() {
    resetView();
    markDirty();
  }
  function onMouseLeave() {
    hideTooltip();
    hoveredIdx = -1;
    markDirty();
  }

  function showTooltip(i, mx, my) {
    if (!tooltip) return;
    const dur = finishes[i] - starts[i];
    tooltip.innerHTML =
      `<b>${escapeHtml(instructions[instructionIds[i]])}</b><br>` +
      `Pulse: ${escapeHtml(pulseNames[pulseNameIds[i]])}<br>` +
      `Start: ${starts[i]}<br>` +
      `Finish: ${finishes[i]}<br>` +
      `Duration: ${dur}`;
    tooltip.style.display = "block";
    let left = mx + 14;
    let top = my + 14;
    const tw = tooltip.offsetWidth;
    if (left + tw > window.innerWidth) left = mx - tw - 8;
    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
  }
  function hideTooltip() {
    if (tooltip) tooltip.style.display = "none";
  }
  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Viewport-edge hit test. Returns {left, right, top, bottom} booleans
  // for each edge whose grip zone contains (mx, my). An edge only counts
  // when the cursor is also within the rect's extent along the orthogonal
  // axis (± MM_EDGE_PX), so approaching a corner region doesn't false-
  // positive on both edges from far away.
  function mmEdgeHit(mx, my, v) {
    const nearL = Math.abs(mx - v.x1) <= MM_EDGE_PX;
    const nearR = Math.abs(mx - v.x2) <= MM_EDGE_PX;
    const nearT = Math.abs(my - v.y1) <= MM_EDGE_PX;
    const nearB = Math.abs(my - v.y2) <= MM_EDGE_PX;
    const withinY = my >= v.y1 - MM_EDGE_PX && my <= v.y2 + MM_EDGE_PX;
    const withinX = mx >= v.x1 - MM_EDGE_PX && mx <= v.x2 + MM_EDGE_PX;
    return {
      left:   nearL && withinY,
      right:  nearR && withinY,
      top:    nearT && withinX,
      bottom: nearB && withinX,
    };
  }

  function mmEdgeCursor(edges) {
    const hx = edges.left || edges.right;
    const hy = edges.top  || edges.bottom;
    if (hx && hy) {
      // top-left or bottom-right → nwse; top-right or bottom-left → nesw.
      if ((edges.left && edges.top) || (edges.right && edges.bottom)) {
        return "nwse-resize";
      }
      return "nesw-resize";
    }
    if (hx) return "ew-resize";
    if (hy) return "ns-resize";
    return null;
  }

  // Minimap interactions (2D — viewport rectangle can be smaller in X and Y).
  function onMmDown(e) {
    if (e.button !== 0) return;
    const rect = mmCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    mmDragStartX = mx;
    mmDragStartY = my;
    mmXMin0 = xMin;
    mmXMax0 = xMax;
    mmYMin0 = yMin;
    mmYMax0 = yMax;
    mmDragMoved = false;
    const w = mmCanvas.clientWidth;
    const v = mmViewportRect(w);
    const edges = mmEdgeHit(mx, my, v);
    const edgeCursor = mmEdgeCursor(edges);
    if (edgeCursor) {
      mmDragMode = "resize";
      mmResizeEdges = edges;
      mmCanvas.style.cursor = edgeCursor;
    } else if (mx >= v.x1 && mx <= v.x2 && my >= v.y1 && my <= v.y2) {
      mmDragMode = "pan";
      mmCanvas.style.cursor = "grabbing";
    } else {
      mmDragMode = "zoom";
      mmZoomBox = {
        startMx: mx, endMx: mx, startMy: my, endMy: my,
        mode: "xOnly",
      };
      mmCanvas.style.cursor = "col-resize";
    }
  }
  function onMmMove(e) {
    const rect = mmCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const w = mmCanvas.clientWidth;
    if (mmDragMode === "none") {
      // Cursor swap: edge-resize on any edge, grab inside viewport rect,
      // crosshair elsewhere.
      if (maxTime > 0) {
        const v = mmViewportRect(w);
        const edgeCursor = mmEdgeCursor(mmEdgeHit(mx, my, v));
        if (edgeCursor) {
          mmCanvas.style.cursor = edgeCursor;
        } else {
          const inside = (mx >= v.x1 && mx <= v.x2 && my >= v.y1 && my <= v.y2);
          mmCanvas.style.cursor = inside ? "grab" : "crosshair";
        }
      }
      return;
    }
    if (Math.abs(mx - mmDragStartX) > 3 || Math.abs(my - mmDragStartY) > 3) {
      mmDragMoved = true;
    }
    if (mmDragMode === "resize") {
      // Move the pinned edges to the current cursor position, anchoring
      // against the frozen opposite bounds (snapshots taken on mousedown).
      // Enforce MIN_X_SPAN / MIN_Y_SPAN so an edge can't cross its mate,
      // then clamp into the legal domain.
      if (mmResizeEdges.left) {
        const dataX = (mx / w) * (maxTime + 1);
        xMin = Math.min(dataX, mmXMax0 - MIN_X_SPAN);
        xMax = mmXMax0;
      } else if (mmResizeEdges.right) {
        const dataX = (mx / w) * (maxTime + 1);
        xMax = Math.max(dataX, mmXMin0 + MIN_X_SPAN);
        xMin = mmXMin0;
      }
      if (mmResizeEdges.top) {
        const dataY = mmYToDataY(my);
        yMax = Math.max(dataY, mmYMin0 + MIN_Y_SPAN);
        yMin = mmYMin0;
      } else if (mmResizeEdges.bottom) {
        const dataY = mmYToDataY(my);
        yMin = Math.min(dataY, mmYMax0 - MIN_Y_SPAN);
        yMax = mmYMax0;
      }
      clampX();
      clampY();
      markDirty();
      return;
    }
    if (mmDragMode === "pan") {
      // X pan
      const dPx = mx - mmDragStartX;
      const dDataX = dPx * (maxTime + 1) / w;
      xMin = mmXMin0 + dDataX;
      xMax = mmXMax0 + dDataX;
      clampX();
      // Y pan. Higher my → lower data-Y row, so dragging DOWN (dPy > 0)
      // should DECREASE yMin/yMax. clampY() is a natural no-op at full
      // domain (default view).
      const nRows = channelOrder.length;
      const dPy = my - mmDragStartY;
      const dDataY = -dPy * nRows / mmCurrentH;
      yMin = mmYMin0 + dDataY;
      yMax = mmYMax0 + dDataY;
      clampY();
      markDirty();
    } else {
      mmZoomBox.endMx = mx;
      mmZoomBox.endMy = my;
      // Mirror the plot-area three-mode gesture on the minimap.
      const dx = mx - mmDragStartX;
      const dy = my - mmDragStartY;
      const THR = 24;
      if (mmZoomBox.mode === "xOnly") {
        if (Math.abs(dy) > THR) {
          if (Math.abs(dx) < THR) {
            mmZoomBox.mode = "yOnly";
            mmCanvas.style.cursor = "row-resize";
          } else {
            mmZoomBox.mode = "xy";
            mmCanvas.style.cursor = "crosshair";
          }
        }
      } else if (mmZoomBox.mode === "yOnly") {
        if (Math.abs(dx) > THR) {
          mmZoomBox.mode = "xy";
          mmCanvas.style.cursor = "crosshair";
        }
      }
      markDirty();
    }
  }
  function onMmUp(e) {
    if (mmDragMode === "zoom" && mmZoomBox && mmDragMoved) {
      const w = mmCanvas.clientWidth;
      // Minimap: single-axis drags reset the *other* axis to full domain,
      // matching the "overview navigation" mental model. xy applies both.
      if (mmZoomBox.mode === "yOnly") {
        xMin = 0;
        xMax = maxTime + 1;
      } else {
        const x1 = Math.min(mmZoomBox.startMx, mmZoomBox.endMx);
        const x2 = Math.max(mmZoomBox.startMx, mmZoomBox.endMx);
        const nMin = (x1 / w) * (maxTime + 1);
        const nMax = (x2 / w) * (maxTime + 1);
        if (nMax - nMin >= MIN_X_SPAN) {
          xMin = nMin; xMax = nMax;
          clampX();
        }
      }
      if (mmZoomBox.mode === "xOnly") {
        yMin = yDomainMin();
        yMax = yDomainMax();
      } else {
        // Convert the box's pixel-Y extents into data-Y. Small my → large
        // data-Y, so the top pixel maps to nyMax.
        const py1 = Math.min(mmZoomBox.startMy, mmZoomBox.endMy);
        const py2 = Math.max(mmZoomBox.startMy, mmZoomBox.endMy);
        const nyMax = mmYToDataY(py1);
        const nyMin = mmYToDataY(py2);
        if (nyMax - nyMin >= MIN_Y_SPAN) {
          yMin = nyMin; yMax = nyMax;
          clampY();
        }
      }
    } else if (mmDragMode === "resize") {
      // Bounds were updated live during move; nothing to commit here.
      // Also: a click on an edge that never moved is intentionally a
      // no-op (skip the click-to-recenter fall-through below).
    } else if (mmDragMode !== "none" && !mmDragMoved) {
      // Click to recenter (both axes)
      const rect = mmCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const w = mmCanvas.clientWidth;
      const centreX = (mx / w) * (maxTime + 1);
      const spanX = xMax - xMin;
      xMin = Math.max(0, centreX - spanX / 2);
      xMax = xMin + spanX;
      clampX();
      const centreY = mmYToDataY(my);
      const spanY = yMax - yMin;
      yMin = centreY - spanY / 2;
      yMax = yMin + spanY;
      clampY();
    }
    mmDragMode = "none";
    mmZoomBox = null;
    mmResizeEdges = null;
    // Set the release cursor from the final hover position so an edge
    // that was just released doesn't flash back to crosshair while the
    // user is still hovering it.
    {
      const rect = mmCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const w = mmCanvas.clientWidth;
      let cursor = "crosshair";
      if (maxTime > 0) {
        const v = mmViewportRect(w);
        const edgeCursor = mmEdgeCursor(mmEdgeHit(mx, my, v));
        if (edgeCursor) {
          cursor = edgeCursor;
        } else if (mx >= v.x1 && mx <= v.x2 && my >= v.y1 && my <= v.y2) {
          cursor = "grab";
        }
      }
      mmCanvas.style.cursor = cursor;
    }
    markDirty();
  }
  function onMmWheel(e) {
    e.preventDefault();
    // Some browsers (Chrome, Firefox) route wheel input onto deltaX when
    // Shift is held, so fall back to deltaX to preserve zoom direction.
    const rawDelta = e.deltaY || e.deltaX;
    const factor = rawDelta > 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
    if (e.shiftKey) {
      // Shift+wheel = Y zoom around the minimap-cursor Y (in data-space).
      const rect = mmCanvas.getBoundingClientRect();
      const my = e.clientY - rect.top;
      const anchor = mmYToDataY(my);
      const nyMin = anchor + (yMin - anchor) * factor;
      const nyMax = anchor + (yMax - anchor) * factor;
      if (nyMax - nyMin >= MIN_Y_SPAN) {
        yMin = nyMin; yMax = nyMax;
        clampY();
      }
    } else {
      const centre = (xMin + xMax) / 2;
      const nMin = centre + (xMin - centre) * factor;
      const nMax = centre + (xMax - centre) * factor;
      if (nMax - nMin >= MIN_X_SPAN) {
        xMin = nMin; xMax = nMax;
        clampX();
      }
    }
    markDirty();
  }

  function onKeyDown(e) {
    if (e.key === "Shift") {
      shiftHeld = true;
      if (dragMode === "none") glCanvas.style.cursor = "grab";
    }
  }
  function onKeyUp(e) {
    if (e.key === "Shift") {
      shiftHeld = false;
      if (dragMode === "none") glCanvas.style.cursor = "crosshair";
    }
  }

  // Minimap resize handle (drag the divider between minimap and plot).
  function onMmResizeDown(e) {
    if (e.button !== 0) return;
    mmResizeDragging = true;
    mmResizeStartY = e.clientY;
    mmResizeStartH = mmCurrentH;
    const handle = document.getElementById("mm-resize-handle");
    if (handle) handle.classList.add("dragging");
    document.body.style.cursor = "row-resize";
    e.preventDefault();
  }
  function onMmResizeMove(e) {
    if (!mmResizeDragging) return;
    const dy = e.clientY - mmResizeStartY;
    const maxH = Math.min(MM_H_MAX, Math.floor(window.innerHeight * 0.5));
    mmCurrentH = Math.max(MM_H_MIN, Math.min(maxH, mmResizeStartH + dy));
    mmCacheDirty = true;
    resizeCanvas();
  }
  function onMmResizeUp() {
    if (!mmResizeDragging) return;
    mmResizeDragging = false;
    const handle = document.getElementById("mm-resize-handle");
    if (handle) handle.classList.remove("dragging");
    document.body.style.cursor = "";
  }

  // =========================================================================
  // Sidebar UI (DOM)
  // =========================================================================
  function el(tag, props, ...children) {
    const n = document.createElement(tag);
    if (props) for (const k in props) {
      if (k === "className") n.className = props[k];
      else if (k === "onClick") n.addEventListener("click", props[k]);
      else if (k === "onInput") n.addEventListener("input", props[k]);
      else if (k === "onChange") n.addEventListener("change", props[k]);
      else if (k === "onKeyDown") n.addEventListener("keydown", props[k]);
      else if (k === "html") n.innerHTML = props[k];
      else n.setAttribute(k, props[k]);
    }
    for (const c of children) {
      if (c == null) continue;
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return n;
  }

  function makeCollapsibleSection(key, titleText, bodyBuilder, extraHeaderNodes) {
    const initiallyOpen = expandedSections.has(key);
    const body = el("div", { className: "sb-section-body" + (initiallyOpen ? " expanded" : "") });
    const chevron = el("span", { className: "sb-chevron" }, initiallyOpen ? "\u25BC" : "\u25B6");
    const header = el(
      "div",
      { className: "sb-section-header" },
      chevron,
      el("span", { className: "sb-section-title" }, titleText),
      ...(extraHeaderNodes || [])
    );
    header.addEventListener("click", (e) => {
      // Ignore clicks on child controls (like the master checkbox)
      if (e.target !== header && e.target !== chevron && !e.target.classList.contains("sb-section-title")) return;
      if (body.classList.contains("expanded")) {
        body.classList.remove("expanded");
        expandedSections.delete(key);
        chevron.textContent = "\u25B6";
      } else {
        body.classList.add("expanded");
        expandedSections.add(key);
        chevron.textContent = "\u25BC";
      }
    });
    bodyBuilder(body);
    const section = el("div", { className: "sb-section" }, header, body);
    return section;
  }

  function updateMasterCheck(cb, nOn, nTotal) {
    if (nOn === nTotal) { cb.checked = true; cb.indeterminate = false; }
    else if (nOn === 0) { cb.checked = false; cb.indeterminate = false; }
    else { cb.checked = false; cb.indeterminate = true; }
  }

  // ---------------------------------------------------------------------------
  // Sidebar in-place refresh helpers (called by click handlers instead of
  // rebuilding the entire DOM).
  // ---------------------------------------------------------------------------

  function refreshGateRow(gi) {
    if (!sidebarRefs) return;
    const ref = sidebarRefs.gateRows[gi];
    if (!ref) return;
    const isOn = !!gateVisible[gi];
    ref.label.style.textDecoration = isOn ? "" : "line-through";
    ref.label.style.color = isOn ? "" : "#999";
    ref.swatch.style.opacity = isOn ? "" : "0.35";
    ref.row.classList.toggle("solo", soloGate === gi);
  }

  function refreshAllGateRows() {
    if (!sidebarRefs) return;
    for (let gi = 0; gi < gates.length; gi++) refreshGateRow(gi);
  }

  function refreshInstMaster() {
    if (!sidebarRefs || !sidebarRefs.instMaster) return;
    const nOn = gateVisible.reduce((a, b) => a + b, 0);
    updateMasterCheck(sidebarRefs.instMaster, nOn, gates.length);
  }

  function refreshChanRow(c) {
    if (!sidebarRefs) return;
    const ref = sidebarRefs.chanRows[c];
    if (!ref) return;
    ref.cb.checked = !!chanVisible[c];
  }

  function refreshGroupMaster(key) {
    if (!sidebarRefs || !sidebarRefs.groupMasters) return;
    const cb = sidebarRefs.groupMasters[key];
    const members = sidebarRefs.groupMembers[key];
    if (!cb || !members) return;
    const nOn = members.reduce((s, c) => s + chanVisible[c], 0);
    updateMasterCheck(cb, nOn, members.length);
  }

  // Build the sidebar DOM exactly once per reload. Stores live references in
  // sidebarRefs so click handlers can update individual nodes without touching
  // the rest of the sidebar.
  function buildSidebar() {
    if (!sidebarBuiltOnce) {
      expandedSections.add("instructions");
      expandedSections.add("source");
      sidebarBuiltOnce = true;
    }
    sidebar.innerHTML = "";

    // Initialise sidebarRefs fresh for this load.
    sidebarRefs = {
      gateRows: [],
      chanRows: new Array(channels.length),
      branchRows: [],
      groupMasters: {},
      groupMembers: {},
      instMaster: null,
    };

    // ---------- Instructions (gate legend) ----------
    const nGatesOn = gateVisible.reduce((a, b) => a + b, 0);
    const masterCb = el("input", { type: "checkbox", className: "sb-master-check" });
    updateMasterCheck(masterCb, nGatesOn, gates.length);
    sidebarRefs.instMaster = masterCb;
    masterCb.addEventListener("click", (ev) => ev.stopPropagation());
    masterCb.addEventListener("change", () => {
      const on = masterCb.checked ? 1 : 0;
      gateVisible.fill(on);
      soloGate = -1;
      updateVisibleAll();
      if (gl) uploadGateVisTex();
      refreshAllGateRows();
      markDirty();
    });

    const instSection = makeCollapsibleSection("instructions", "Instructions", (body) => {
      for (let gi = 0; gi < gates.length; gi++) {
        const g = gates[gi];
        const swatch = el("span", { className: "swatch" });
        swatch.style.background = colorMap[gi] || "#666";
        const label = el("span", null, g);
        if (!gateVisible[gi]) {
          label.style.textDecoration = "line-through";
          label.style.color = "#999";
          swatch.style.opacity = "0.35";
        }
        const cnt = el("span", null, "  " + (gateInstructions[gi] ? gateInstructions[gi].length : 0));
        cnt.style.color = "#888";
        cnt.style.marginLeft = "auto";
        cnt.style.fontSize = "10px";
        const row = el(
          "div",
          { className: "sb-row sb-clickable" + (soloGate === gi ? " solo" : "") },
          swatch, label, cnt
        );
        sidebarRefs.gateRows[gi] = { row, swatch, label };
        row.addEventListener("click", (ev) => {
          if (ev.altKey) {
            if (soloGate === gi) {
              soloGate = -1;
              gateVisible.fill(1);
            } else {
              soloGate = gi;
              gateVisible.fill(0);
              gateVisible[gi] = 1;
            }
            updateVisibleAll();
          } else {
            soloGate = -1;
            gateVisible[gi] = gateVisible[gi] ? 0 : 1;
            updateVisibleForGate(gi);
          }
          if (gl) uploadGateVisTex();
          refreshAllGateRows();
          refreshInstMaster();
          markDirty();
        });
        row.addEventListener("mouseenter", () => { hoveredGate = gi; markDirty(); });
        row.addEventListener("mouseleave", () => { hoveredGate = -1; markDirty(); });
        body.appendChild(row);
      }
    }, [masterCb]);
    sidebar.appendChild(instSection);

    // ---------- Channel sections ----------
    const groups = { qubit: [], readout: [], broadcast: [], other: [] };
    for (let c = 0; c < channels.length; c++) groups[classifyChannel(channels[c])].push(c);
    const grpTitles = {
      qubit: "Qubit channels", readout: "Readout channels",
      broadcast: "Broadcast channels", other: "Other channels",
    };
    for (const key of ["qubit", "readout", "broadcast", "other"]) {
      const members = groups[key];
      if (!members.length) continue;
      sidebarRefs.groupMembers[key] = members;
      const nOn = members.reduce((s, c) => s + chanVisible[c], 0);
      const cb = el("input", { type: "checkbox", className: "sb-master-check" });
      updateMasterCheck(cb, nOn, members.length);
      sidebarRefs.groupMasters[key] = cb;
      cb.addEventListener("click", (e) => e.stopPropagation());
      cb.addEventListener("change", () => {
        const on = cb.checked ? 1 : 0;
        for (const c of members) chanVisible[c] = on;
        for (const c of members) updateVisibleForChannel(c);
        const prevNRows = channelOrder.length;
        computeChannelOrder();
        if (gl) { uploadChannelRowTex(); uploadChanVisTex(); }
        if (channelOrder.length !== prevNRows && container) resizeCanvas();
        for (const c of members) refreshChanRow(c);
        const nNow = members.reduce((s, c) => s + chanVisible[c], 0);
        updateMasterCheck(cb, nNow, members.length);
        markDirty();
      });
      const section = makeCollapsibleSection(key, grpTitles[key], (body) => {
        for (const c of members) {
          const cb2 = el("input", { type: "checkbox" });
          cb2.checked = !!chanVisible[c];
          sidebarRefs.chanRows[c] = { cb: cb2 };
          cb2.addEventListener("change", () => {
            chanVisible[c] = cb2.checked ? 1 : 0;
            updateVisibleForChannel(c);
            const prevNRows = channelOrder.length;
            computeChannelOrder();
            if (gl) { uploadChannelRowTex(); uploadChanVisTex(); }
            if (channelOrder.length !== prevNRows && container) resizeCanvas();
            refreshGroupMaster(key);
            markDirty();
          });
          const rowEl = el("label", { className: "sb-row sb-check" }, cb2, el("span", null, channels[c]));
          body.appendChild(rowEl);
        }
      }, [cb]);
      sidebar.appendChild(section);
    }

    // ---------- Branches ----------
    const brSec = el("div", { className: "sb-section" },
      el("div", { className: "sb-title" }, "Branches"));
    for (let b = 0; b < 3; b++) {
      const cb = el("input", { type: "checkbox" });
      cb.checked = !!branchVisible[b];
      sidebarRefs.branchRows[b] = { cb };
      cb.addEventListener("change", () => {
        branchVisible[b] = cb.checked ? 1 : 0;
        updateVisibleForBranch(b);
        if (gl) uploadBranchVisTex();
        markDirty();
      });
      brSec.appendChild(el("label", { className: "sb-row sb-check" }, cb, el("span", null, BRANCHES[b])));
    }
    sidebar.appendChild(brSec);

    // ---------- Time ----------
    const timeSec = el("div", { className: "sb-section" },
      el("div", { className: "sb-title" }, "Time"));
    const dtLabel = el("label", { className: "sb-row-dt" }, "dt (ns): ");
    const dtInput = el("input", {
      type: "number",
      step: "any",
      min: "0",
      placeholder: "cycles",
      className: "search-input",
    });
    if (dtNs > 0) dtInput.value = String(dtNs);
    dtInput.addEventListener("input", () => {
      const v = parseFloat(dtInput.value);
      dtNs = isFinite(v) && v > 0 ? v : 0;
      markDirty();
    });
    dtLabel.appendChild(dtInput);
    timeSec.appendChild(dtLabel);
    sidebar.appendChild(timeSec);

    // ---------- Search ----------
    const searchSec = el("div", { className: "sb-section" },
      el("div", { className: "sb-title" }, "Search"));
    const searchInput = el("input", { type: "text", className: "search-input", placeholder: "regex · gate / channel / instruction" });
    searchInput.value = searchQuery;
    searchInput.addEventListener("input", () => runSearch(searchInput.value));
    searchInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); jumpToNextHit(); }
    });
    const status = el("div", { id: "search-status", className: "search-status" }, "");
    searchSec.appendChild(searchInput);
    searchSec.appendChild(status);
    sidebar.appendChild(searchSec);

    // ---------- Source editor (textarea populated once; survives toggles) ----------
    const srcSection = makeCollapsibleSection("source", "Source", (body) => {
      const ta = el("textarea", { className: "source-editor", spellcheck: "false" });
      ta.value = rawCsv;
      const errDiv = el("div", { className: "source-error" });
      const btnRow = el("div", { className: "source-btn-row" });
      const applyBtn = el("button", { className: "source-btn source-btn-apply" }, "Apply");
      let flashTimer = null;
      applyBtn.addEventListener("click", () => {
        try {
          parseCsv(ta.value, { strict: true });
          errDiv.textContent = "";
          reload(ta.value);
          const flag = el("span", { className: "source-applied-flag" }, "Applied \u2713");
          btnRow.appendChild(flag);
          if (flashTimer) clearTimeout(flashTimer);
          flashTimer = setTimeout(() => { if (flag.parentNode) flag.parentNode.removeChild(flag); }, 1200);
        } catch (e) {
          errDiv.textContent = String(e.message || e);
        }
      });
      btnRow.appendChild(applyBtn);
      body.appendChild(ta);
      body.appendChild(btnRow);
      body.appendChild(errDiv);
    });
    sidebar.appendChild(srcSection);

    // ---------- Shortcuts ----------
    const shortcutsSec = makeCollapsibleSection("help", "Shortcuts", (body) => {
      const list = el("dl", { className: "shortcuts-list" });
      const pairs = [
        ["Wheel", "Zoom X"],
        ["Shift+Wheel", "Zoom Y"],
        ["Drag", "Zoom to region"],
        ["Shift+Drag", "Pan"],
        ["Dbl-click", "Reset view"],
        ["Alt+click gate", "Solo gate"],
        ["Enter", "Jump to next hit"],
      ];
      for (const [k, v] of pairs) {
        list.appendChild(el("dt", null, k));
        list.appendChild(el("dd", null, v));
      }
      body.appendChild(list);
    });
    sidebar.appendChild(shortcutsSec);

    // ---------- Reset buttons ----------
    const resetZoom = el("button", { className: "reset-btn" }, "Reset zoom");
    resetZoom.addEventListener("click", () => { resetView(); markDirty(); });
    const resetAllBtn = el("button", { className: "reset-btn" }, "Reset");
    resetAllBtn.addEventListener("click", () => resetAll());
    sidebar.appendChild(resetZoom);
    sidebar.appendChild(resetAllBtn);
  }

  // =========================================================================
  // Resize
  // =========================================================================
  function resizeCanvas() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(300, container.clientWidth);
    const nRows = channelOrder.length;
    // Canvas fills available container height, but stops at the ideal size
    // for natural-height rows. Squishing kicks in when nRows exceeds what
    // fits at natural row height, which IS the default full-height overview.
    const availH = Math.max(200, container.clientHeight);
    const idealH = PAD_TOP + PAD_BOTTOM + Math.max(1, nRows) * ROW_HEIGHT_PX;
    const ch = Math.max(200, Math.min(availH, idealH));
    // Filter toggles can change nRows while the user is zoomed in; keep the
    // current Y viewport but clamp it to the new domain.
    clampY();

    glCanvas.style.width = cw + "px";
    glCanvas.style.height = ch + "px";
    overlayCanvas.style.width = cw + "px";
    overlayCanvas.style.height = ch + "px";
    glCanvas.width = Math.round(cw * dpr);
    glCanvas.height = Math.round(ch * dpr);
    overlayCanvas.width = Math.round(cw * dpr);
    overlayCanvas.height = Math.round(ch * dpr);

    const mmParent = mmCanvas.parentElement;
    const mmw = Math.max(300, mmParent ? mmParent.clientWidth - 16 : cw);
    mmCanvas.style.width = mmw + "px";
    mmCanvas.style.height = mmCurrentH + "px";
    mmCanvas.width = Math.round(mmw * dpr);
    mmCanvas.height = Math.round(mmCurrentH * dpr);

    if (gl) gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    // Minimap width / DPR may have changed; force a cache rebuild.
    mmCacheDirty = true;
    markDirty();
  }

  // =========================================================================
  // Reload pipeline
  // =========================================================================
  function reload(csv) {
    rawCsv = csv;
    let rows = parseCsv(csv);
    if (mergeActive) rows = mergeInstructions(rows);
    buildStore(rows);
    applyInitialOptions();
    computeChannelOrder();
    updateVisibleAll();
    if (gl) {
      rebuildInstanceBuffers();  // stable VBO, built once per load
      buildLookupTextures();     // allocate + upload all small textures
    }
    resetView();
    sidebarRefs = null;
    buildSidebar();
    resizeCanvas();
    markDirty();
  }

  // =========================================================================
  // Init
  // =========================================================================
  function init() {
    glCanvas = document.getElementById("gl-canvas");
    overlayCanvas = document.getElementById("overlay-canvas");
    overlayCtx = overlayCanvas.getContext("2d");
    mmCanvas = document.getElementById("mm-canvas");
    mmCtx = mmCanvas.getContext("2d");
    tooltip = document.getElementById("tooltip");
    sidebar = document.getElementById("sidebar");
    container = document.getElementById("canvas-container");
    if (!glCanvas || !overlayCanvas || !mmCanvas || !sidebar || !container) return;

    if (!initGL()) return;

    glCanvas.addEventListener("wheel", onWheel, { passive: false });
    glCanvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    glCanvas.addEventListener("dblclick", onDblClick);
    glCanvas.addEventListener("mouseleave", onMouseLeave);
    mmCanvas.addEventListener("mousedown", onMmDown);
    window.addEventListener("mousemove", onMmMove);
    window.addEventListener("mouseup", onMmUp);
    mmCanvas.addEventListener("wheel", onMmWheel, { passive: false });
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);

    const mmResize = document.getElementById("mm-resize-handle");
    if (mmResize) {
      mmResize.addEventListener("mousedown", onMmResizeDown);
      window.addEventListener("mousemove", onMmResizeMove);
      window.addEventListener("mouseup", onMmResizeUp);
    }

    let resizeQueued = false;
    const onResize = () => {
      if (resizeQueued) return;
      resizeQueued = true;
      requestAnimationFrame(() => { resizeQueued = false; resizeCanvas(); });
    };
    window.addEventListener("resize", onResize);
    if (window.ResizeObserver) {
      new ResizeObserver(onResize).observe(container);
    }

    // Pull data from Python bridge
    const inlineCsv = window.__CIRCUIT_SCHEDULE_TIMING__;
    const inlineOpts = window.__CIRCUIT_SCHEDULE_OPTIONS__;
    if (inlineOpts && typeof inlineOpts === "object") {
      opts = Object.assign(opts, inlineOpts);
    }
    if (typeof inlineCsv === "string" && inlineCsv.trim()) {
      reload(inlineCsv);
    } else {
      // Standalone use: show paste/drop area
      setupPasteArea();
      resizeCanvas();
      markDirty();
    }

    window.__WEBGL_VIEWER_READY__ = true;
  }

  function setupPasteArea() {
    const pa = document.getElementById("paste-area");
    if (!pa) return;
    pa.style.display = "block";
    const ta = pa.querySelector("textarea");
    const commit = (csv) => {
      pa.style.display = "none";
      reload(csv);
    };
    if (ta) {
      ta.addEventListener("paste", (e) => {
        const text = (e.clipboardData || window.clipboardData).getData("text");
        if (text && text.length > 30) {
          e.preventDefault();
          commit(text);
        }
      });
      ta.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
          e.preventDefault();
          if (ta.value.trim()) commit(ta.value);
        }
      });
      ta.addEventListener("blur", () => {
        if (ta.value.trim()) commit(ta.value);
      });
    }
    pa.addEventListener("dragover", (e) => { e.preventDefault(); pa.classList.add("dragover"); });
    pa.addEventListener("dragleave", () => pa.classList.remove("dragover"));
    pa.addEventListener("drop", (e) => {
      e.preventDefault();
      pa.classList.remove("dragover");
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => commit(String(reader.result));
      reader.readAsText(f);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
