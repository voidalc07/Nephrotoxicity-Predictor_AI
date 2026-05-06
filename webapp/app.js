// ---------------------------------------------------------------------------
// Frontend State Container
// The dashboard keeps a single explicit state object so saved benchmark search,
// live routed prediction, analytics rendering, and explanatory modal content
// stay synchronised without introducing a heavier SPA framework.
// ---------------------------------------------------------------------------
const state = {
  overview: null,
  activePage: "predictor",
  activeModel: null,
  screeningMode: "saved",
  matches: [],
  liveResult: null,
  activeMatchIndex: 0,
  charts: {},
  searchStatus: "idle",
  lastAnalyzedQuery: "",
  lastResolvedQuery: "",
  feedbackStatus: "",
};

// ---------------------------------------------------------------------------
// DOM Registry
// Elements are captured once at startup so rendering logic can focus on the
// scientific payloads returned by the backend rather than repeated queries.
// ---------------------------------------------------------------------------
const elements = {
  navItems: document.querySelectorAll(".menu-item"),
  pages: document.querySelectorAll(".page-content"),
  pageTitle: document.getElementById("page-title"),
  pageSubtitle: document.getElementById("page-subtitle"),
  engineSelect: document.getElementById("engine-select"),
  screeningModeToggle: document.getElementById("screening-mode-toggle"),
  screeningModeHelp: document.getElementById("screening-mode-help"),
  sampleLibraryLabel: document.getElementById("sample-library-label"),
  searchForm: document.getElementById("search-form"),
  queryInput: document.getElementById("molecule-query"),
  exampleChips: document.getElementById("example-chips"),
  heroLabel: document.getElementById("hero-label"),
  heroScore: document.getElementById("hero-score"),
  heroDetail: document.getElementById("hero-detail"),
  confidenceBar: document.getElementById("confidence-bar"),
  moleculeCard: document.getElementById("molecule-card"),
  matchList: document.getElementById("match-list"),
  matchCount: document.getElementById("match-count"),
  predictionTableWrapper: document.getElementById("prediction-table-wrapper"),
  statsTableWrapper: document.getElementById("stats-table-wrapper"),
  heatmap: document.getElementById("tanimoto-heatmap"),
  topPairs: document.getElementById("top-similarity-pairs"),
  syncTrigger: document.getElementById("sync-trigger"),
  modal: document.getElementById("detail-modal"),
  modalContent: document.getElementById("modal-content"),
  closeModal: document.getElementById("close-modal"),
};

const PAGE_META = {
  predictor: {
    title: "Screening Terminal",
    subtitle: "Validated molecular screening against the integrated main model set.",
  },
  analytics: {
    title: "Deep Analytics",
    subtitle: "Real chemistry similarity and performance comparisons from the evaluation dataset.",
  },
  knowledge: {
    title: "Intelligence Core",
    subtitle: "Why these model families matter, how they complement each other, and why the project matters beyond the screen.",
  },
  heritage: {
    title: "Research Heritage",
    subtitle: "The experimental lineage behind the current predictor, from early baselines to full-coverage ensembles.",
  },
};

// Small formatting helpers keep numerical evidence consistent across the
// screening terminal and analytics views.
function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function formatMetric(value, digits = 4) {
  return isFiniteNumber(value) ? value.toFixed(digits) : "—";
}

function formatPct(value) {
  return isFiniteNumber(value) ? `${Math.round(value * 100)}%` : "—";
}

function formatLabel(label) {
  if (label === 1) return "Toxic";
  if (label === 0) return "Safe";
  return "Unknown";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function activeMatch() {
  // Saved-mode active molecule from the archived external benchmark outputs.
  return state.matches[state.activeMatchIndex] ?? null;
}

function activeLiveResult() {
  // Live-mode routed analysis dossier returned by `/api/predict-live`.
  return state.liveResult;
}

function setPage(pageId) {
  // -------------------------------------------------------------------------
  // Page Routing
  // Predictor, analytics, intelligence, and heritage are separated because
  // each page serves a different methodological purpose in the project.
  // -------------------------------------------------------------------------
  state.activePage = pageId;
  elements.navItems.forEach((item) => item.classList.toggle("active", item.dataset.page === pageId));
  elements.pages.forEach((page) => page.classList.toggle("active", page.id === `page-${pageId}`));
  const meta = PAGE_META[pageId] || { title: pageId, subtitle: "" };
  elements.pageTitle.textContent = meta.title;
  elements.pageSubtitle.textContent = meta.subtitle;
  if (pageId === "analytics") {
    renderAnalytics();
  }
}

function setPendingInput(query) {
  // Clear any stale result state as soon as the query changes.
  if (typeof query === "string") {
    elements.queryInput.value = query;
  }
  state.searchStatus = elements.queryInput.value.trim() ? "ready" : "idle";
  state.lastAnalyzedQuery = "";
  state.lastResolvedQuery = "";
  state.matches = [];
  state.liveResult = null;
  state.activeMatchIndex = 0;
  state.feedbackStatus = "";
  renderResults();
  renderHero();
}

function renderModeUI() {
  const live = state.screeningMode === "live";
  // The toggle updates labels and inputs so the terminal behaves differently in saved and live screening modes.
  if (elements.screeningModeToggle) {
    elements.screeningModeToggle.querySelectorAll("[data-mode]").forEach((button) => {
      button.classList.toggle("active", button.dataset.mode === state.screeningMode);
    });
  }
  if (elements.screeningModeHelp) {
    elements.screeningModeHelp.textContent = live
      ? "Run live nephrotoxicity analysis on any SMILES string with route-aware consensus, neighbours, scaffold context, and engine availability."
      : "Search saved predictions from the integrated evaluation outputs in this single project.";
  }
  if (elements.sampleLibraryLabel) {
    elements.sampleLibraryLabel.textContent = live ? "Quick Access Starter Queries:" : "Quick Access Verified Samples:";
  }
  if (elements.pageSubtitle && state.activePage === "predictor") {
    elements.pageSubtitle.textContent = live
      ? "Validated live molecular screening with route-aware consensus and chemistry-native explanation layers."
      : PAGE_META.predictor.subtitle;
  }
  elements.queryInput.placeholder = live
    ? "Enter any SMILES for live analysis (e.g. [NH3][Pt]([NH3])(Cl)Cl)..."
    : "Enter SMILES or Medicine Name (e.g. Gentamicin, Cisplatin)...";
  if (elements.engineSelect) {
    elements.engineSelect.disabled = live;
  }
}

function setScreeningMode(mode) {
  state.screeningMode = mode === "live" ? "live" : "saved";
  state.matches = [];
  state.liveResult = null;
  state.activeMatchIndex = 0;
  state.searchStatus = elements.queryInput.value.trim() ? "ready" : "idle";
  state.lastAnalyzedQuery = "";
  state.lastResolvedQuery = "";
  state.feedbackStatus = "";
  renderModeUI();
  renderExampleChips();
  renderResults();
  renderHero();
}

function renderExampleChips() {
  // Demonstration chips expose both medicine aliases and benchmark SMILES.
  const examples = state.overview?.example_queries ?? [];
  const medicines = state.overview?.medicine_names ?? [];
  elements.exampleChips.innerHTML = [
    ...medicines.map((medicine) => `<button type="button" class="chip med-chip" data-query="${escapeHtml(medicine)}">${escapeHtml(medicine)}</button>`),
    ...examples.map((example) => `<button type="button" class="chip" data-query="${escapeHtml(example)}">${escapeHtml(example)}</button>`),
  ].join("");

  elements.exampleChips.querySelectorAll("[data-query]").forEach((chip) => {
    chip.addEventListener("click", () => setPendingInput(chip.dataset.query ?? ""));
  });
}

function renderStatsTable() {
  // The benchmark table foregrounds external transfer while still surfacing the
  // internal AUROC needed to reason about generalisation gap.
  const models = state.overview?.models ?? [];
  elements.statsTableWrapper.innerHTML = `
    <table class="premium-table">
      <thead>
        <tr>
          <th>Engine</th>
          <th>Model Type</th>
          <th>Ext AUROC</th>
          <th>Ext F1</th>
          <th>Int AUROC</th>
          <th>Gap</th>
        </tr>
      </thead>
      <tbody>
        ${models
          .map(
            (model) => `
              <tr>
                <td class="primary-cell">${escapeHtml(model.display_name)}</td>
                <td>${escapeHtml(model.variant_display ?? model.variant ?? "Main Model")}</td>
                <td><strong>${formatMetric(model.external.roc_auc)}</strong></td>
                <td>${formatMetric(model.external.f1)}</td>
                <td>${formatMetric(model.internal.roc_auc)}</td>
                <td>${formatMetric(model.gap_auc)}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function destroyChart(key) {
  // Chart.js instances must be torn down before replacement to avoid duplicate
  // canvases and stale legends after refreshes.
  if (state.charts[key]) {
    state.charts[key].destroy();
    state.charts[key] = null;
  }
}

function renderPerformanceChart() {
  // AUROC and F1 are plotted together because ranking quality and thresholded
  // classification quality capture different operational properties.
  const models = state.overview?.models ?? [];
  const ctx = document.getElementById("aurocChart");
  if (!ctx) return;
  destroyChart("performance");
  state.charts.performance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: models.map((model) => model.display_name),
      datasets: [
        {
          label: "External AUROC",
          data: models.map((model) => model.external.roc_auc),
          backgroundColor: "#10b981",
          borderRadius: 10,
        },
        {
          label: "External F1",
          data: models.map((model) => model.external.f1),
          backgroundColor: "rgba(16, 185, 129, 0.32)",
          borderRadius: 10,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { family: "Outfit" } },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          grid: { color: "#1e293b" },
          ticks: { color: "#64748b" },
        },
        x: {
          grid: { display: false },
          ticks: { color: "#64748b" },
        },
      },
    },
  });
}

function renderDatasetBalanceChart() {
  // Class balance helps interpret why some metrics are more informative than
  // raw accuracy on the external evaluation set.
  const balance = state.overview?.analytics?.dataset_balance;
  const ctx = document.getElementById("balanceChart");
  if (!ctx || !balance) return;
  destroyChart("balance");
  state.charts.balance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: balance.labels,
      datasets: [
        {
          data: balance.counts,
          backgroundColor: ["#10b981", "#ef4444"],
          borderColor: ["#111622", "#111622"],
          borderWidth: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8", font: { family: "Outfit" } },
        },
      },
    },
  });
}

function renderGapChart() {
  // Generalisation gap operationalises the dissertation concern that strong
  // internal validation may not transfer to unseen chemistry.
  const gaps = state.overview?.analytics?.generalization_gap ?? [];
  const ctx = document.getElementById("gapChart");
  if (!ctx || gaps.length === 0) return;
  destroyChart("gap");
  state.charts.gap = new Chart(ctx, {
    type: "bar",
    data: {
      labels: gaps.map((row) => row.display_name),
      datasets: [
        {
          label: "Internal - External AUROC",
          data: gaps.map((row) => row.gap_auc),
          backgroundColor: gaps.map((row) => (row.gap_auc > 0.08 ? "#f59e0b" : "#10b981")),
          borderRadius: 10,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          beginAtZero: true,
          grid: { color: "#1e293b" },
          ticks: { color: "#64748b" },
        },
        y: {
          grid: { display: false },
          ticks: { color: "#64748b" },
        },
      },
    },
  });
}

function heatmapColor(value) {
  const clamped = Math.max(0, Math.min(1, value));
  const red = Math.round(20 + (16 * clamped));
  const green = Math.round(24 + (161 * clamped));
  const blue = Math.round(35 + (88 * clamped));
  return `rgb(${red}, ${green}, ${blue})`;
}

function renderHeatmap() {
  // -------------------------------------------------------------------------
  // Similarity Heatmap Rendering
  // The heatmap visualises pairwise Morgan-fingerprint Tanimoto similarity
  // across a balanced subset of external compounds so the chemical structure of
  // the benchmark space remains visible alongside model metrics.
  // -------------------------------------------------------------------------
  if (!elements.heatmap) return;
  const heatmap = state.overview?.analytics?.tanimoto_heatmap;
  const topPairs = state.overview?.analytics?.tanimoto_heatmap?.top_pairs ?? [];

  if (!heatmap || !heatmap.matrix?.length) {
    elements.heatmap.innerHTML = `<p class="muted">Heatmap data is not available yet.</p>`;
    if (elements.topPairs) {
      elements.topPairs.innerHTML = "";
    }
    return;
  }

  const labels = heatmap.labels;
  let html = `<div class="heatmap-grid" style="grid-template-columns: 110px repeat(${labels.length}, minmax(46px, 1fr));">`;
  html += `<div class="heatmap-corner"></div>`;
  labels.forEach((label) => {
    html += `<div class="heatmap-axis x" title="${escapeHtml(label.smiles)}">${escapeHtml(label.short_label)}</div>`;
  });

  labels.forEach((rowLabel, rowIndex) => {
    html += `<div class="heatmap-axis y" title="${escapeHtml(rowLabel.smiles)}">${escapeHtml(rowLabel.short_label)}</div>`;
    heatmap.matrix[rowIndex].forEach((value, colIndex) => {
      const labelText = `${rowLabel.short_label} vs ${labels[colIndex].short_label}: ${value.toFixed(3)}`;
      html += `
        <div
          class="heatmap-cell ${rowIndex === colIndex ? "diagonal" : ""}"
          title="${escapeHtml(labelText)}"
          style="background:${heatmapColor(value)}"
        >
          <span>${value.toFixed(2)}</span>
        </div>
      `;
    });
  });

  html += `</div>`;
  html += `
    <div class="heatmap-legend">
      <span>Low Similarity</span>
      <div class="heatmap-swatch"></div>
      <span>High Similarity</span>
    </div>
  `;
  elements.heatmap.innerHTML = html;

  if (elements.topPairs) {
    elements.topPairs.innerHTML = `
      <div class="pair-grid">
        ${topPairs
          .map(
            (pair) => `
              <div class="pair-card">
                <div class="pair-score">${pair.similarity.toFixed(3)}</div>
                <div class="pair-label">${escapeHtml(pair.left)} / ${escapeHtml(pair.right)}</div>
                <p>${escapeHtml(pair.left_smiles)}</p>
                <p>${escapeHtml(pair.right_smiles)}</p>
              </div>
            `,
          )
          .join("")}
      </div>
    `;
  }
}

function renderAnalytics() {
  // Central analytics redraw used after overview refreshes and page switches.
  renderStatsTable();
  renderPerformanceChart();
  renderDatasetBalanceChart();
  renderGapChart();
  renderHeatmap();
}

function renderHero() {
  // -------------------------------------------------------------------------
  // High-Salience Decision Card
  // The hero card surfaces either the archived saved-model score or the live
  // routed consensus probability, depending on the active screening mode.
  // -------------------------------------------------------------------------
  if (state.screeningMode === "live") {
    const result = activeLiveResult();
    if (!result) {
      if (state.searchStatus === "ready") {
        elements.heroLabel.textContent = "Ready for live analysis";
        elements.heroScore.textContent = "--";
        elements.confidenceBar.style.width = "0%";
        elements.confidenceBar.style.backgroundColor = "#10b981";
        elements.heroDetail.textContent = "Input captured. Click Analyze to generate a live routed consensus dossier.";
        return;
      }
      elements.heroLabel.textContent = "Standby";
      elements.heroScore.textContent = "--";
      elements.confidenceBar.style.width = "0%";
      elements.confidenceBar.style.backgroundColor = "#10b981";
      elements.heroDetail.textContent = "Awaiting a SMILES query for live analysis.";
      return;
    }

    // Live mode renders the routed consensus dossier returned by /api/predict-live.
    const score = result.consensus?.consensus_probability ?? result.domain?.max_tanimoto ?? 0;
    const label = result.consensus?.consensus_label;
    const route = result.consensus?.route ?? "route";
    const toxic = label === 1;
    const color = route === "route_c" ? "#ef4444" : route === "route_b" ? "#f59e0b" : toxic ? "#ef4444" : "#10b981";
    elements.heroLabel.innerHTML = `<span class="${toxic ? "label-toxic" : "label-safe"}">${toxic ? "TOXIC" : "SAFE"}</span> <small class="muted">/ ${escapeHtml(route)}</small>`;
    elements.heroScore.textContent = formatPct(score);
    elements.confidenceBar.style.width = `${(score ?? 0) * 100}%`;
    elements.confidenceBar.style.backgroundColor = color;
    elements.heroDetail.textContent = result.consensus?.message ?? "Live analysis returned a routed consensus result.";
    return;
  }

  const match = activeMatch();
  if (!match) {
    if (state.searchStatus === "ready") {
      elements.heroLabel.textContent = "Ready to analyze";
      elements.heroScore.textContent = "--";
      elements.confidenceBar.style.width = "0%";
      elements.confidenceBar.style.backgroundColor = "#10b981";
      elements.heroDetail.textContent = "Input captured. Click Analyze to search the saved prediction outputs.";
      return;
    }
    if (state.searchStatus === "no_results") {
      elements.heroLabel.textContent = "No saved match";
      elements.heroScore.textContent = "--";
      elements.confidenceBar.style.width = "0%";
      elements.confidenceBar.style.backgroundColor = "#f59e0b";
      elements.heroDetail.textContent = `No evaluated molecule matched "${state.lastResolvedQuery || state.lastAnalyzedQuery}".`;
      return;
    }
    elements.heroLabel.textContent = "Standby";
    elements.heroScore.textContent = "--";
    elements.confidenceBar.style.width = "0%";
    elements.confidenceBar.style.backgroundColor = "#10b981";
    elements.heroDetail.textContent = "Awaiting molecular or medicinal input.";
    return;
  }

  const prediction =
    match.model_predictions.find((item) => item.model_name === state.activeModel) ?? {
      display_name: "Consensus",
      predicted_label: match.consensus_label,
      predicted_score: match.consensus_score,
    };
  // Saved mode replays the archived prediction outputs so the terminal stays identical to the evaluation view.
  const toxic = prediction.predicted_label === 1;
  elements.heroLabel.innerHTML = `<span class="${toxic ? "label-toxic" : "label-safe"}">${toxic ? "TOXIC" : "SAFE"}</span> <small class="muted">/ ${escapeHtml(prediction.display_name)}</small>`;
  elements.heroScore.textContent = formatPct(prediction.predicted_score);
  elements.confidenceBar.style.width = `${(prediction.predicted_score ?? 0) * 100}%`;
  elements.confidenceBar.style.backgroundColor = toxic ? "#ef4444" : "#10b981";
  elements.heroDetail.textContent = `Screening result for the selected molecule using ${prediction.display_name}.`;
}

function renderResults() {
  // -------------------------------------------------------------------------
  // Mode-Dependent Result Surfaces
  // Saved mode exposes archived per-engine predictions from the external test
  // set, while live mode exposes the explanation stack produced by the
  // applicability-domain and consensus pipeline.
  // -------------------------------------------------------------------------
  if (state.screeningMode === "live") {
    const result = activeLiveResult();
    elements.matchCount.textContent = result ? String((result.explanations?.nearest_neighbours ?? []).length) : "0";

    if (!result) {
      const message =
        state.searchStatus === "ready"
          ? "Input is ready. Click Analyze to run live analysis."
          : "No analyzed result yet.";
      elements.matchList.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
      elements.moleculeCard.innerHTML = `<div class="empty-structure"><p>${escapeHtml(message)}</p></div>`;
      elements.predictionTableWrapper.innerHTML = `<div class="empty-state">Live engine probabilities and explanation layers will appear here after you click Analyze.</div>`;
      return;
    }

    const neighbours = result.explanations?.nearest_neighbours ?? [];
    const alerts = result.explanations?.structural_alerts ?? [];
    const scaffold = result.explanations?.scaffold_context ?? {};

    elements.matchList.innerHTML = `
      <div class="small-muted">Nearest training neighbours</div>
      ${neighbours
        .map(
          (item) => `
            <button class="match-button active">
              <div class="match-top"><strong>${formatMetric(item.similarity, 3)}</strong> <span>#${item.rank}</span></div>
              <div class="match-smiles">${escapeHtml(item.canonical_smiles)}</div>
            </button>
          `,
        )
        .join("")}
      <div class="feedback-actions">
        <button type="button" class="mode-chip" data-feedback-label="1">Confirm toxic</button>
        <button type="button" class="mode-chip" data-feedback-label="0">Confirm non-toxic</button>
      </div>
      <p class="small-muted">${escapeHtml(state.feedbackStatus || "Feedback is optional and records a confirmed label locally.")}</p>
    `;

    elements.moleculeCard.innerHTML = `
      <div class="molecule-profile">
        <div class="smiles-box">${escapeHtml(result.canonical_smiles)}</div>
        <div class="profile-meta">
          <div class="p-stat">
            <label>Applicability</label>
            <strong>${escapeHtml(result.consensus?.badge ?? "Unknown")}</strong>
          </div>
          <div class="p-stat">
            <label>Max Tanimoto</label>
            <strong style="color:var(--accent-primary)">${formatMetric(result.domain?.max_tanimoto)}</strong>
          </div>
          <div class="p-stat">
            <label>Scaffold Status</label>
            <strong>${escapeHtml(scaffold.status ?? "Unknown")}</strong>
          </div>
          <div class="p-stat">
            <label>Alerts</label>
            <strong>${alerts.length > 0 ? escapeHtml(alerts.map((item) => item.name).join(", ")) : "None"}</strong>
          </div>
        </div>
      </div>
    `;

    const engineRows = Object.values(result.engine_predictions ?? {});
  elements.predictionTableWrapper.innerHTML = `
      <table class="compact-table">
        <thead>
          <tr>
            <th>Engine</th>
            <th>Status</th>
            <th>Label</th>
            <th>Prob</th>
          </tr>
        </thead>
        <tbody>
          ${engineRows
            .map(
              (prediction) => `
                <tr>
                  <td>${escapeHtml(prediction.display_name)}</td>
                  <td>${escapeHtml(prediction.available ? "Available" : prediction.reason_unavailable || "Unavailable")}</td>
                  <td class="${prediction.predicted_label === 1 ? "label-toxic" : prediction.predicted_label === 0 ? "label-safe" : ""}">${prediction.available ? escapeHtml(formatLabel(prediction.predicted_label)) : "—"}</td>
                  <td class="mono">${prediction.available ? formatMetric(prediction.predicted_score) : "—"}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    `;
    return;
  }

  elements.matchCount.textContent = String(state.matches.length);

  if (state.matches.length === 0) {
    let message = "No analyzed result yet.";
    if (state.searchStatus === "ready") {
      message = "Input is ready. Click Analyze to query the saved predictions.";
    } else if (state.searchStatus === "no_results") {
      message = `No saved evaluated molecule matched "${state.lastResolvedQuery || state.lastAnalyzedQuery}".`;
    }

    elements.matchList.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
    elements.moleculeCard.innerHTML = `
      <div class="empty-structure">
        <p>${escapeHtml(message)}</p>
      </div>
    `;
    elements.predictionTableWrapper.innerHTML = `
      <div class="empty-state">Per-model predictions will appear here after you click Analyze.</div>
    `;
    return;
  }

  elements.matchList.innerHTML = state.matches
    .map(
      (match, index) => `
        <button class="match-button ${index === state.activeMatchIndex ? "active" : ""}" data-match-index="${index}">
          <div class="match-top"><strong>${formatPct(match.consensus_score)} Conf.</strong> <span>#${index + 1}</span></div>
          <div class="match-smiles">${escapeHtml(match.canonical_smiles)}</div>
        </button>
      `,
    )
    .join("");

  elements.matchList.querySelectorAll("[data-match-index]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeMatchIndex = Number(button.dataset.matchIndex ?? 0);
      renderResults();
      renderHero();
    });
  });

  const match = activeMatch();
  if (!match) {
    return;
  }

  elements.moleculeCard.innerHTML = `
    <div class="molecule-profile">
      <div class="smiles-box">${escapeHtml(match.canonical_smiles)}</div>
      <div class="profile-meta">
        <div class="p-stat">
          <label>Experimental Truth</label>
          <strong>${match.true_label === 1 ? '<span class="label-toxic">Toxic</span>' : match.true_label === 0 ? '<span class="label-safe">Safe</span>' : "Unknown"}</strong>
        </div>
        <div class="p-stat">
          <label>System Consensus</label>
          <strong style="color:var(--accent-primary)">${formatPct(match.consensus_score)}</strong>
        </div>
      </div>
    </div>
  `;

  elements.predictionTableWrapper.innerHTML = `
    <table class="compact-table">
      <thead>
        <tr>
          <th>Engine</th>
          <th>Model Type</th>
          <th>Label</th>
          <th>Prob</th>
        </tr>
      </thead>
      <tbody>
        ${match.model_predictions
          .map(
            (prediction) => `
              <tr>
                <td>${escapeHtml(prediction.display_name)}</td>
                <td>${escapeHtml(prediction.variant_display ?? prediction.variant ?? "Main Model")}</td>
                <td class="${prediction.predicted_label === 1 ? "label-toxic" : "label-safe"}">${prediction.predicted_label === 1 ? "Toxic" : "Safe"}</td>
                <td class="mono">${formatMetric(prediction.predicted_score)}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

async function submitFeedback(label) {
  // Live confirmations are recorded server-side so the project can support a
  // lightweight human-in-the-loop retraining pathway.
  const result = activeLiveResult();
  if (!result) return;

  state.feedbackStatus = "Recording confirmation...";
  renderResults();
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        smiles: result.canonical_smiles,
        label: Number(label),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Feedback could not be recorded.");
    }
    state.feedbackStatus = `Confirmed ${Number(label) === 1 ? "toxic" : "non-toxic"} for ${payload.canonical_smiles}.`;
  } catch (error) {
    state.feedbackStatus = error instanceof Error ? error.message : "Feedback could not be recorded.";
  }
  renderResults();
}

async function runSearch(query) {
  // -------------------------------------------------------------------------
  // Unified Submission Handler
  // The same terminal input can query archived dissertation evidence or run
  // live inference. Branching here keeps the rest of the UI state-driven.
  // -------------------------------------------------------------------------
  const trimmed = query.trim();
  if (!trimmed) {
    setPendingInput("");
    return;
  }

  try {
    if (state.screeningMode === "live") {
      // Live mode sends the query to the predictive API and renders the routed consensus dossier returned by the backend.
      const response = await fetch("/api/predict-live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: trimmed }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Live analysis failed.");
      }
      state.matches = [];
      state.liveResult = data;
      state.activeMatchIndex = 0;
      state.lastAnalyzedQuery = trimmed;
      state.lastResolvedQuery = data.canonical_smiles || trimmed;
      state.searchStatus = "results";
      state.feedbackStatus = "";
      renderResults();
      renderHero();
      return;
    }

    const response = await fetch(`/api/search?q=${encodeURIComponent(trimmed)}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Saved search failed.");
    }
    state.matches = data.matches || [];
    state.liveResult = null;
    state.activeMatchIndex = 0;
    state.lastAnalyzedQuery = trimmed;
    state.lastResolvedQuery = data.resolved_query || trimmed;
    state.searchStatus = state.matches.length > 0 ? "results" : "no_results";
  } catch (error) {
    state.matches = [];
    state.liveResult = null;
    state.activeMatchIndex = 0;
    state.lastAnalyzedQuery = trimmed;
    state.lastResolvedQuery = trimmed;
    state.searchStatus = "no_results";
    state.feedbackStatus = error instanceof Error ? error.message : "";
  }

  renderResults();
  renderHero();
}

async function refreshResults() {
  // Trigger the backend orchestration pathway and then pull a fresh overview so
  // newly generated benchmark artefacts appear without restarting the server.
  if (!elements.syncTrigger) return;
  elements.syncTrigger.disabled = true;
  elements.syncTrigger.textContent = "Resyncing...";
  try {
    await fetch("/api/run-main", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skip_data_prep: true, force_rerun: false }),
    });
    const response = await fetch("/api/overview");
    state.overview = await response.json();
    renderOverview();
  } finally {
    elements.syncTrigger.disabled = false;
    elements.syncTrigger.textContent = "Resync Data";
  }
}

const KNOWLEDGE_CORE = {
  // Curated narrative fragments for the Intelligence Core modal cards.
  ensemble: `
    <h2>Meta-Ensemble Decision Layer</h2>
    <p>The project's most competitive systems are the <strong>Full Coverage Ensemble</strong> and the <strong>Descriptor + Fingerprint Ensemble</strong>. These are not single learners: they combine multiple predictive viewpoints so the final decision is less dependent on one representation.</p>
    <p>Across this project, KNN contributes neighborhood awareness, LightGBM captures non-linear feature splits, and logistic meta-learning helps arbitrate between component signals. This is important for nephrotoxicity because harmful compounds do not always cluster into one simple rule family.</p>
    <p>In practical terms, this layer represents the project's move from isolated experiments to a coordinated decision system. It is the clearest example of how the project became more mature, more robust, and more explainable.</p>
  `,
  chemberta: `
    <h2>Chemical Language Models</h2>
    <p>The <strong>ChemBERTa Hybrid</strong> line brings transformer-based reasoning into the project. Instead of only using handcrafted descriptors or bit vectors, it reads SMILES strings as chemical language and learns higher-order context from token sequences.</p>
    <p>This matters because toxicity is often influenced by combinations of motifs, ordering, and local context that fixed fingerprints may only partially encode. The transformer branch gives the project a more modern representation-learning perspective.</p>
    <p>It also connects the work to a future-facing pipeline where molecular screening can combine cheminformatics features with foundation-model style embeddings.</p>
  `,
  autoencoder: `
    <h2>Latent Representation Models</h2>
    <p>The <strong>Autoencoder Model</strong> explores unsupervised compression. It learns a latent fingerprint space where the model can retain the strongest chemical structure signals while reducing noise and redundancy.</p>
    <p>This branch is important because not all predictive strength comes from direct supervision. Sometimes a model learns cleaner structure by first reconstructing the data and then classifying in a compressed space.</p>
    <p>Within the project, the unsupervised family also helped compare PCA-style reduction with neural latent reduction, which made the evolution of the feature-learning story much easier to explain.</p>
  `,
  similarity: `
    <h2>Similarity and Retrieval Logic</h2>
    <p>The <strong>Similarity Model</strong> keeps a core scientific principle alive: molecules with related structure often share related behavior. Tanimoto similarity, scaffold relationships, and fingerprint overlap remain valuable signals in toxicity work.</p>
    <p>This project did not abandon similarity methods when newer models were added. Instead, it treated similarity as a stable reference frame and then layered more expressive models on top of it.</p>
    <p>That is one reason the dashboard now includes a real Tanimoto heatmap. It shows that chemistry space itself is part of the explanation, not just the prediction score.</p>
  `,
  evolution: `
    <h2>Model Constellation</h2>
    <p>This project investigated a wide experimental field: <strong>SOTA baselines</strong>, <strong>noteworthy fixed/final models</strong>, <strong>modular LightGBM</strong>, <strong>HistGradientBoosting</strong>, <strong>virtual GIN</strong>, <strong>NODE-style models</strong>, <strong>stacking ensembles</strong>, <strong>project-plus GIN</strong>, <strong>project-plus GIN hybrid</strong>, and <strong>project-plus ChemBERTa</strong>.</p>
    <p>That combination history matters because it shows the final predictor was not chosen arbitrarily. It emerged from comparison, iteration, pruning, and consolidation. The current five engines are the outcome of a broader research search process.</p>
    <p>In a more futuristic sense, this page presents the project as an intelligence constellation rather than one static model: multiple reasoning styles were tested, retained, archived, and recombined until the system became clearer and stronger.</p>
  `,
  impact: `
    <h2>Human and Clinical Relevance</h2>
    <p>Nephrotoxicity is not just a modeling problem. It is tied to kidney safety, failed compounds, patient risk, and the cost of discovering too late that a molecule is harmful. Earlier warning can influence which candidates move forward and which are redesigned.</p>
    <p>That is why this predictor matters. Even as a project prototype, it demonstrates how data-driven screening could support medicinal chemistry, toxicology triage, and safer decision-making earlier in the pipeline.</p>
    <p>The futuristic value of the work is that it points toward a workflow where model families, chemical similarity analysis, and structured evaluation outputs can work together as a decision support layer, not just a one-off classifier.</p>
  `,
};

function showDetail(key) {
  // Modal presentation of conceptual model-family explanations.
  const html = KNOWLEDGE_CORE[key];
  if (!html) return;
  elements.modalContent.innerHTML = html;
  elements.modal.style.display = "flex";
}

function renderOverview() {
  // Hydrate the whole dashboard from the consolidated overview payload returned
  // by the backend evidence aggregator.
  const models = state.overview?.models ?? [];
  elements.engineSelect.innerHTML = models
    .map((model) => `<option value="${model.model_name}">${escapeHtml(model.display_name)}</option>`)
    .join("");

  if (!state.activeModel && models.length > 0) {
    state.activeModel = state.overview.best_model?.model_name ?? models[0].model_name;
  }
  elements.engineSelect.value = state.activeModel ?? "";

  renderModeUI();
  renderExampleChips();
  renderAnalytics();
  renderResults();
  renderHero();
}

function attachEvents() {
  // Event wiring remains centralised so the dashboard stays maintainable as a
  // research interface rather than an opaque frontend bundle.
  elements.navItems.forEach((item) => item.addEventListener("click", () => setPage(item.dataset.page)));
  elements.searchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch(elements.queryInput.value);
  });
  elements.queryInput.addEventListener("input", () => {
    if (elements.queryInput.value.trim() !== state.lastAnalyzedQuery) {
      setPendingInput(elements.queryInput.value);
    }
  });
  elements.engineSelect.addEventListener("change", (event) => {
    state.activeModel = event.target.value;
    renderHero();
  });
  if (elements.screeningModeToggle) {
    elements.screeningModeToggle.querySelectorAll("[data-mode]").forEach((button) => {
      button.addEventListener("click", () => setScreeningMode(button.dataset.mode || "saved"));
    });
  }
  if (elements.syncTrigger) {
    elements.syncTrigger.addEventListener("click", refreshResults);
  }
  elements.closeModal.addEventListener("click", () => {
    elements.modal.style.display = "none";
  });
  document.querySelectorAll(".doc-card").forEach((card) => {
    card.addEventListener("click", () => showDetail(card.dataset.model));
  });
  document.addEventListener("click", (event) => {
    const target = event.target instanceof HTMLElement ? event.target.closest("[data-feedback-label]") : null;
    if (!target) return;
    submitFeedback(target.dataset.feedbackLabel);
  });
}

async function init() {
  // Boot sequence: attach handlers, fetch overview data, and render the first
  // consistent dashboard state.
  attachEvents();
  const response = await fetch("/api/overview");
  state.overview = await response.json();
  renderOverview();
}

init();
