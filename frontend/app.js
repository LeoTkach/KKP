const state = {
  model: null,
  source: null,
  imageFile: null,
  imageDataUrl: null,
};

const $ = (id) => document.getElementById(id);

const DATASET_LABEL = "Parveshiiii/AI-vs-Real";

const EXPECTED_LABELS = {
  real: "Реальне фото",
  ai_generated: "AI-генерація",
};

function sourceDisplay(source) {
  return {
    dataset: source.dataset_label || DATASET_LABEL,
    expected:
      source.expected_label || EXPECTED_LABELS[source.expected_class] || "—",
  };
}

const modelSelect = $("model-select");
const preview = $("preview");
const uploadZone = $("upload-zone");
const fileInput = $("file-input");
const resultsEl = $("results");
const imageView = $("image-view");
const compareView = $("compare-view");
const compareSliderWrap = $("compare-slider-wrap");
const compareSliderThumb = $("compare-slider-thumb");
const compareOverlay = $("compare-overlay");
const compareHandle = $("compare-handle");
const compareBase = $("compare-base");
const compareHeat = $("compare-heat");
const compareMedia = $("compare-media");
const compareStage = $("compare-stage");

let comparePct = 50;

function setComparePosition(pct) {
  comparePct = pct;
  const value = `${pct}%`;
  compareOverlay.style.width = value;
  compareHandle.style.left = value;
  compareSliderThumb.style.left = value;
  compareSliderWrap.setAttribute("aria-valuenow", String(Math.round(pct)));
}

function setCompareFromSliderPointer(clientX) {
  const rect = compareSliderWrap.getBoundingClientRect();
  if (!rect.width) return;
  const pct = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
  setComparePosition(pct);
}

compareSliderWrap.addEventListener("pointerdown", (event) => {
  event.stopPropagation();
  compareSliderWrap.setPointerCapture(event.pointerId);
  setCompareFromSliderPointer(event.clientX);
});

compareSliderWrap.addEventListener("pointermove", (event) => {
  if (compareSliderWrap.hasPointerCapture(event.pointerId)) {
    setCompareFromSliderPointer(event.clientX);
  }
});

compareSliderWrap.addEventListener("pointerup", (event) => {
  compareSliderWrap.releasePointerCapture(event.pointerId);
});

compareSliderWrap.addEventListener("keydown", (event) => {
  let next = comparePct;
  if (event.key === "ArrowLeft" || event.key === "ArrowDown") next -= 5;
  else if (event.key === "ArrowRight" || event.key === "ArrowUp") next += 5;
  else if (event.key === "Home") next = 0;
  else if (event.key === "End") next = 100;
  else return;
  event.preventDefault();
  setComparePosition(Math.min(100, Math.max(0, next)));
});

function setCompareFromPointer(clientX) {
  const rect = compareMedia.getBoundingClientRect();
  if (!rect.width) return;
  const pct = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
  setComparePosition(pct);
}

compareStage.addEventListener("pointerdown", (event) => {
  compareStage.setPointerCapture(event.pointerId);
  compareStage.classList.add("compare-stage--dragging");
  setCompareFromPointer(event.clientX);
});

compareStage.addEventListener("pointermove", (event) => {
  if (compareStage.hasPointerCapture(event.pointerId)) {
    setCompareFromPointer(event.clientX);
  }
});

compareStage.addEventListener("pointerup", (event) => {
  compareStage.releasePointerCapture(event.pointerId);
  compareStage.classList.remove("compare-stage--dragging");
});

compareStage.addEventListener("pointercancel", () => {
  compareStage.classList.remove("compare-stage--dragging");
});

function setImageFromDataUrl(dataUrl) {
  state.imageDataUrl = dataUrl;
  state.imageFile = null;
  preview.src = dataUrl;
  preview.classList.remove("hidden");
}

function clearImage() {
  state.imageFile = null;
  state.imageDataUrl = null;
  preview.src = "";
  preview.classList.add("hidden");
  fileInput.value = "";
}

function renderModelSelect(models, defaultKey) {
  modelSelect.innerHTML = "";
  models.forEach(({ key, label }) => {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "radio";
    button.textContent = label;
    button.dataset.key = key;
    button.setAttribute("aria-checked", key === defaultKey ? "true" : "false");
    button.addEventListener("click", () => selectModel(key));
    modelSelect.appendChild(button);
  });
  state.model = defaultKey;
}

function selectModel(key) {
  state.model = key;
  modelSelect.querySelectorAll("button").forEach((btn) => {
    btn.setAttribute("aria-checked", btn.dataset.key === key ? "true" : "false");
  });
  hideHeatmap();
  if (state.imageDataUrl || state.imageFile) {
    if (state.source) {
      renderPending(state.source);
    } else {
      renderAwaiting();
    }
  }
}

function renderEmpty() {
  resultsEl.className = "results results--empty";
  resultsEl.innerHTML = `
    <div class="results-placeholder">
      <p class="results-title">Результат з’явиться тут</p>
      <p class="results-text">Завантажте зображення та натисніть «Аналізувати».</p>
    </div>
  `;
}

function renderAwaiting() {
  resultsEl.className = "results results--hint";
  resultsEl.innerHTML = `
    <div class="results-hint">
      <p class="results-title">Зображення завантажено</p>
      <p class="results-text">Натисніть «Аналізувати», щоб отримати прогноз моделі.</p>
    </div>
  `;
}

function datasetBlockHtml(source, { showMatch = false } = {}) {
  const { dataset, expected } = sourceDisplay(source);
  let matchPart = "";
  if (showMatch && source.match !== undefined) {
    const matchClass = source.match ? "match-ok" : "match-bad";
    matchPart = ` · <span class="${matchClass}">${source.match_label}</span>`;
  }
  return `
    <div class="dataset-info">
      <p class="dataset-summary">
        Датасет: ${dataset} · очікується <strong>${expected}</strong>${matchPart}
      </p>
    </div>
  `;
}

function renderPending(source) {
  resultsEl.className = "results";
  resultsEl.innerHTML = `
    ${datasetBlockHtml(source)}
    <p class="pending-cta">Натисніть «Аналізувати», щоб побачити прогноз моделі та Grad-CAM.</p>
  `;
}

function renderError(message) {
  resultsEl.className = "results results--error";
  resultsEl.innerHTML = `
    <div class="results-placeholder">
      <p class="error-text">${message}</p>
    </div>
  `;
}

function renderAnalysis(data) {
  const toneClass = data.label === "real" ? "analysis--real" : "analysis--ai";
  const probRows = data.confidences
    .map(
      (row) => `
        <div class="prob-row">
          <div class="prob-row-header"><span>${row.label}</span><span>${row.pct.toFixed(1)}%</span></div>
          <div class="prob-track">
            <div class="prob-fill prob-fill--${row.key === "real" ? "real" : "ai"}" style="width:${row.pct}%"></div>
          </div>
        </div>
      `,
    )
    .join("");

  let sourceHtml = "";
  if (data.source) {
    sourceHtml = datasetBlockHtml(data.source, { showMatch: true });
  }

  resultsEl.className = "results";
  resultsEl.innerHTML = `
    <div class="analysis ${toneClass}">
      <div class="verdict-row">
        <h3 class="verdict-title">${data.title}</h3>
        <span class="verdict-pct">${data.confidence_pct.toFixed(1)}%</span>
      </div>
      <div class="prob-list">${probRows}</div>
      ${sourceHtml}
    </div>
  `;

  if (data.heatmap) {
    showHeatmap(data.heatmap);
  } else {
    hideHeatmap();
  }
}

function syncCompareLayout() {
  const stageWidth = compareStage.clientWidth;
  const stageHeight = compareStage.clientHeight;
  if (!stageWidth || !stageHeight) return;

  compareMedia.style.maxWidth = `${stageWidth}px`;
  compareMedia.style.maxHeight = `${stageHeight}px`;

  const width = compareBase.offsetWidth;
  const height = compareBase.offsetHeight;
  if (!width || !height) return;
  compareHeat.style.width = `${width}px`;
  compareHeat.style.height = `${height}px`;
}

let compareLayoutObserver = null;

function showHeatmap(heatmap) {
  compareBase.src = heatmap.original;
  compareHeat.src = heatmap.overlay;
  setComparePosition(50);
  imageView.classList.add("hidden");
  compareView.classList.remove("hidden");

  const onCompareImageReady = () => {
    syncCompareLayout();
    setComparePosition(comparePct);
  };
  compareBase.onload = onCompareImageReady;
  compareHeat.onload = onCompareImageReady;
  requestAnimationFrame(onCompareImageReady);

  if (compareLayoutObserver) compareLayoutObserver.disconnect();
  compareLayoutObserver = new ResizeObserver(onCompareImageReady);
  compareLayoutObserver.observe(compareStage);
}

function hideHeatmap() {
  imageView.classList.remove("hidden");
  compareView.classList.add("hidden");
  compareBase.src = "";
  compareHeat.src = "";
  compareMedia.style.maxWidth = "";
  compareMedia.style.maxHeight = "";
  compareHeat.style.width = "";
  compareHeat.style.height = "";
  if (compareLayoutObserver) {
    compareLayoutObserver.disconnect();
    compareLayoutObserver = null;
  }
}

async function loadModels() {
  const response = await fetch("/api/models");
  if (!response.ok) throw new Error("Не вдалося завантажити моделі");
  const data = await response.json();
  renderModelSelect(data.models, data.default);
}

async function analyze() {
  if (!state.imageFile && !state.imageDataUrl) {
    renderError("Спочатку завантажте зображення.");
    return;
  }

  const form = new FormData();
  form.append("model", state.model);

  if (state.imageFile) {
    form.append("image", state.imageFile);
  } else {
    const blob = await (await fetch(state.imageDataUrl)).blob();
    form.append("image", blob, "upload.jpg");
  }

  if (state.source) {
    form.append("source", JSON.stringify(state.source));
  }

  $("analyze-btn").disabled = true;
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) {
      renderError(data.detail || "Помилка аналізу");
      hideHeatmap();
      return;
    }
    renderAnalysis(data);
  } catch {
    renderError("Помилка мережі. Спробуйте ще раз.");
    hideHeatmap();
  } finally {
    $("analyze-btn").disabled = false;
  }
}

async function pickRandom() {
  $("random-btn").disabled = true;
  try {
    const response = await fetch("/api/random");
    const data = await response.json();
    if (!response.ok) {
      renderError(data.detail || "Датасет не знайдено");
      return;
    }
    state.source = {
      split: data.source.split,
      label_dir: data.source.label_dir,
      expected_class: data.source.expected_class,
      relative_path: data.source.relative_path,
      dataset_label: data.source.dataset_label,
      expected_label: data.source.expected_label,
    };
    setImageFromDataUrl(data.image);
    renderPending(data.source);
    hideHeatmap();
  } catch {
    renderError("Помилка мережі.");
  } finally {
    $("random-btn").disabled = false;
  }
}

function clearAll() {
  state.source = null;
  clearImage();
  renderEmpty();
  hideHeatmap();
}

uploadZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  state.source = null;
  state.imageFile = file;
  state.imageDataUrl = null;
  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
  renderAwaiting();
  hideHeatmap();
});

$("analyze-btn").addEventListener("click", analyze);
$("random-btn").addEventListener("click", pickRandom);
$("clear-btn").addEventListener("click", clearAll);

loadModels().catch(() => renderError("Не вдалося завантажити моделі."));
