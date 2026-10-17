"use strict";

(() => {
  const $ = id => document.getElementById(id);
  const MAX_BYTES = 64 * 1024 * 1024;
  const BUDGET_MS = 300000;
  const state = { file: null, url: null, busy: false, job: null, started: 0, timer: null,
    retryAt: 0, retryTimer: null, demo: false, modelReady: null, duration: null, rules: [], rulesBusy: false };
  const create = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  };
  const finite = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
  function timecode(seconds, precise = true) {
    if (!finite(seconds)) return "—";
    const units = Math.round(seconds * (precise ? 10 : 1));
    const minutes = Math.floor(units / (precise ? 600 : 60));
    const remainder = units % (precise ? 600 : 60);
    return `${String(minutes).padStart(2, "0")}:${precise ? (remainder / 10).toFixed(1).padStart(4, "0") : String(remainder).padStart(2, "0")}`;
  }
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const motionPreference = matchMedia("(prefers-reduced-motion: reduce)");
  let backgroundMoving = !motionPreference.matches;
  let motionChosen = false;
  function updateBackgroundMotion() {
    $("backgroundOrbits").dataset.motion = backgroundMoving ? "running" : "paused";
    $("motionToggle").hidden = false;
    $("motionToggle").textContent = backgroundMoving ? "Пауза фона" : "Оживить фон";
    $("motionToggle").setAttribute("aria-pressed", String(backgroundMoving));
    $("motionToggle").setAttribute("aria-label", backgroundMoving ? "Остановить движение фоновых орбит" : "Включить движение фоновых орбит");
  }
  $("motionToggle").addEventListener("click", () => {
    backgroundMoving = !backgroundMoving;
    motionChosen = true;
    updateBackgroundMotion();
  });
  motionPreference.addEventListener?.("change", event => {
    if (!motionChosen) { backgroundMoving = !event.matches; updateBackgroundMotion(); }
  });
  updateBackgroundMotion();
  function scrollTo(element) {
    element.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
  }
  function clearErrors() {
    for (const id of ["errorBox", "keyError", "fileError"]) $(id).hidden = true;
    for (const id of ["apiKey", "videoFile"]) $(id).removeAttribute("aria-invalid");
  }
  function showError(message, field) {
    $("errorText").textContent = message;
    $("errorBox").hidden = false;
    if (field) {
      const input = $(field === "key" ? "apiKey" : "videoFile");
      const hint = $(field === "key" ? "keyError" : "fileError");
      hint.textContent = message;
      hint.hidden = false;
      input.setAttribute("aria-invalid", "true");
      input.focus();
    } else $("errorBox").focus();
    $("resumeButton").hidden = !state.job;
  }
  function controls() {
    const cooling = Date.now() < state.retryAt;
    const blocked = state.busy || cooling || Boolean(state.job);
    $("analyzeButton").disabled = blocked;
    $("apiKey").disabled = state.busy;
    $("videoFile").disabled = state.busy || Boolean(state.job);
    document.querySelectorAll("[data-example]").forEach(button => button.disabled = blocked);
    $("resumeButton").disabled = state.busy || cooling;
    $("resetButton").disabled = state.busy || cooling;
    $("resetButton").hidden = !state.file && !state.job;
    $("loadRulesButton").disabled = state.rulesBusy;
    $("review").classList.toggle("is-busy", state.busy);
    $("review").setAttribute("aria-busy", String(state.busy));
  }
  function cooldown(value) {
    let seconds = Number(value);
    if (!Number.isFinite(seconds)) seconds = (Date.parse(value) - Date.now()) / 1000;
    if (!Number.isFinite(seconds) || seconds <= 0) seconds = 5;
    state.retryAt = Date.now() + Math.ceil(seconds * 1000);
    clearInterval(state.retryTimer);
    const update = () => {
      const remaining = Math.max(0, Math.ceil((state.retryAt - Date.now()) / 1000));
      $("analyzeButton").firstElementChild.textContent = remaining ? `Сервер занят · повтор через ${remaining} с` : "Анализировать видео";
      if (!remaining) clearInterval(state.retryTimer);
      controls();
    };
    update();
    state.retryTimer = setInterval(update, 1000);
  }
  class ApiError extends Error {
    constructor(message, status = 0, field = null) { super(message); this.status = status; this.field = field; }
  }
  async function request(path, options = {}, timeout = 20000) {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), Math.max(1, timeout));
    try {
      const response = await fetch(path, { ...options, cache: "no-store", signal: abort.signal,
        headers: { ...(path.startsWith("/api/") ? { "X-API-Key": $("apiKey").value } : {}), ...options.headers } });
      if (!response.ok) {
        let detail = "";
        try { const data = await response.json(); if (typeof data.detail === "string") detail = data.detail; } catch { /* HTTP code still provides a safe message. */ }
        if (response.status === 429) cooldown(response.headers.get("Retry-After"));
        const messages = {
          400: "Файл пустой или не может быть обработан.", 401: "Неверный ключ доступа. Проверьте API-ключ команды.",
          413: "Файл превышает допустимый размер загрузки.", 415: "Выберите видео в формате MOV или MP4.",
          429: "Сервис уже обрабатывает другое видео. Дождитесь окончания паузы и повторите действие.",
          503: "Модель не готова. Проверьте её настройку на сервере."
        };
        throw new ApiError(messages[response.status] || detail || `Ошибка сервиса: HTTP ${response.status}.`, response.status, response.status === 401 ? "key" : null);
      }
      if (options.asBlob) return await response.blob();
      try { return await response.json(); } catch { throw new ApiError("Сервис вернул некорректный ответ. Результат анализа не показан."); }
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (error.name === "AbortError") throw new ApiError("Время ожидания ответа истекло. Проверьте состояние задания перед новой загрузкой.");
      throw new ApiError("Не удалось подключиться к сервису. Проверьте сеть и запущен ли локальный сервер.");
    } finally { clearTimeout(timer); }
  }
  async function health() {
    try {
      const result = await request("/health", {}, 6000);
      if (result.status !== "ok") throw new Error("Health status is not ok");
      state.demo = result.demo_only === true;
      state.modelReady = result.model_present === true;
      $("serviceStatus").dataset.state = state.modelReady ? "ready" : "error";
      $("serviceStatus").lastElementChild.textContent = !state.modelReady ? "Модель не настроена" : state.demo ? "Сервис доступен · демо" : "Сервис доступен";
      $("demoNotice").hidden = !state.demo;
      $("demoExamples").hidden = !state.demo;
    } catch {
      state.modelReady = null;
      state.demo = false;
      $("serviceStatus").dataset.state = "error";
      $("serviceStatus").lastElementChild.textContent = "Сервис недоступен";
      $("demoExamples").hidden = true;
      $("demoNotice").hidden = true;
    }
  }
  function clearResult() {
    $("resultSection").hidden = true;
    $("eventList").replaceChildren();
    $("eventTimeline").replaceChildren();
    $("warningList").replaceChildren();
    if ($("frameDialog").open) $("frameDialog").close();
    $("expandedFrame").removeAttribute("src");
  }
  function releaseVideo() {
    $("player").pause();
    $("player").removeAttribute("src");
    $("player").load();
    if (state.url) URL.revokeObjectURL(state.url);
    state.url = null;
  }
  function choose(file) {
    if (state.busy || state.job) return;
    clearErrors();
    clearResult();
    releaseVideo();
    state.file = null;
    state.duration = null;
    $("processingPanel").hidden = true;
    $("playbackFallback").hidden = true;
    $("player").hidden = true;
    $("previewEmpty").hidden = false;
    $("fileBadge").hidden = true;
    $("fileName").textContent = "Перетащите видео сюда";
    $("fileDetail").textContent = "или нажмите, чтобы выбрать файл";
    if (!file) { controls(); return false; }
    let issue = "";
    if (!/\.(mov|mp4)$/i.test(file.name)) issue = "Поддерживаются только MOV и MP4.";
    else if (!file.size) issue = "Выбранный файл пустой.";
    else if (file.size > MAX_BYTES) issue = "Видео превышает 64 МБ. Выберите файл меньшего размера.";
    if (issue) { $("videoFile").value = ""; controls(); showError(issue, "file"); return false; }
    state.file = file;
    $("fileName").textContent = file.name;
    $("fileDetail").textContent = `${(file.size / 1024 / 1024).toFixed(1)} МБ · нажмите для замены`;
    $("fileBadge").hidden = false;
    $("previewEmpty").hidden = true;
    state.url = URL.createObjectURL(file);
    $("player").src = state.url;
    $("player").hidden = false;
    controls();
    return true;
  }
  function validateKey() {
    if (!$("apiKey").value.trim()) { showError("Введите ключ доступа команды.", "key"); return false; }
    if (/[\r\n\x00-\x1f\u0100-\uffff]/.test($("apiKey").value)) { showError("Проверьте ключ: используйте выданный командой API-ключ без переносов строки.", "key"); return false; }
    return true;
  }
  const stageTranslations = {
    "Starting local analysis": "Запускаем локальный анализ", "Decoding video locally": "Декодируем видео локально",
    "Analyzing visible actions": "Анализируем видимые действия", "Attaching source frames and reviewed rules": "Прикрепляем исходные кадры и правила"
  };
  function stage(result) {
    const text = typeof result.stage === "string" ? result.stage : "Обработка видео";
    const windowMatch = /^Analyzed window (\d+) of (\d+)$/.exec(text);
    $("stageText").textContent = stageTranslations[text] || (windowMatch ? `Обработано окон: ${windowMatch[1]} из ${windowMatch[2]}` : text);
    if (finite(result.progress) && result.progress <= 100) {
      $("progressBar").value = result.progress;
      $("progressDetail").textContent = `Прогресс сервера: ${result.progress}%. Оставьте страницу открытой.`;
    } else { $("progressBar").removeAttribute("value"); $("progressDetail").textContent = "Сервис обрабатывает запись. Оставьте страницу открытой."; }
  }
  function startTimer() {
    clearInterval(state.timer);
    const update = () => $("elapsedText").textContent = timecode(Math.floor((Date.now() - state.started) / 1000), false);
    update(); state.timer = setInterval(update, 1000);
  }
  function validResult(result) {
    if (!["violation", "no_violation"].includes(result.verdict) || !Array.isArray(result.violations)) return false;
    if ((result.verdict === "violation") !== (result.violations.length > 0)) return false;
    return result.violations.every(event => event && typeof event.type === "string" && finite(event.timecode_sec) &&
      typeof event.description === "string" && Array.isArray(event.evidence_frames) && event.evidence_frames.length >= 1 && event.evidence_frames.length <= 3 &&
      event.evidence_frames.every(frame => finite(frame.timestamp_sec) && typeof frame.image_base64 === "string" && /^[A-Za-z0-9+/\s]+=*$/.test(frame.image_base64) && frame.image_base64.length > 0));
  }
  function sourceLink(url) {
    try {
      const parsed = new URL(url);
      if (parsed.protocol !== "https:" || parsed.username || parsed.password) return null;
      const link = create("a", "Открыть официальный источник ↗", "source-link");
      link.href = parsed.href; link.target = "_blank"; link.rel = "noopener noreferrer";
      return link;
    } catch { return null; }
  }
  function openFrame(frame) {
    $("expandedFrame").src = `data:image/jpeg;base64,${frame.image_base64}`;
    $("expandedFrame").alt = `Исходный кадр на ${timecode(frame.timestamp_sec)}`;
    $("expandedCaption").textContent = `Исходный кадр · ${timecode(frame.timestamp_sec)}`;
    $("frameDialog").showModal();
  }
  function jumpTo(seconds, index) {
    if (state.url && !$("player").error && $("player").readyState >= 1) {
      try { $("player").currentTime = seconds; } catch { /* Frames remain the fallback. */ }
    }
    document.querySelectorAll(".event-card").forEach((card, i) => card.classList.toggle("is-selected", i === index));
    $("eventTimeline").querySelectorAll("button").forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
    const card = $(`event-${index}`); scrollTo(card); card.focus({ preventScroll: true });
  }
  function render(result) {
    if (!validResult(result)) throw new ApiError("Ответ сервиса не соответствует формату результата. Достоверный вердикт не получен.");
    clearResult();
    const found = result.verdict === "violation";
    const meta = result.meta || {};
    $("verdictBadge").textContent = found ? "ТРЕБУЕТ ПРОВЕРКИ ЧЕЛОВЕКОМ" : "АНАЛИЗ ЗАВЕРШЁН";
    $("verdictBadge").dataset.verdict = result.verdict;
    $("resultTitle").textContent = found ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено";
    $("resultNote").textContent = found ? `Событий для проверки: ${result.violations.length}. Сопоставьте каждый вывод с записью и условиями правила.` : "Модель не обнаружила поддерживаемых событий. Это не подтверждает полное соответствие правилам: оцените видимость и ограничения анализа.";
    $("resultDemoNotice").hidden = meta.demo_only !== true;
    const metrics = [];
    if (finite(meta.duration_sec)) metrics.push(`Запись ${timecode(meta.duration_sec)}`);
    if (finite(meta.wall_time_sec)) metrics.push(`Анализ ${meta.wall_time_sec.toFixed(1)} с`);
    $("resultMetrics").textContent = metrics.join(" · ");
    $("jobLabel").textContent = result.job_id || "";
    result.violations.forEach((event, index) => {
      const timelineButton = create("button", `${timecode(event.timecode_sec)} / ${String(index + 1).padStart(2, "0")}`, "timeline-button");
      timelineButton.type = "button"; timelineButton.setAttribute("aria-pressed", "false");
      timelineButton.setAttribute("aria-label", `Событие ${index + 1}, ${timecode(event.timecode_sec)}: ${event.type}`);
      timelineButton.addEventListener("click", () => jumpTo(event.timecode_sec, index));
      $("eventTimeline").append(timelineButton);
      const card = create("article", undefined, "event-card"); card.id = `event-${index}`; card.tabIndex = -1;
      const heading = create("div", undefined, "event-heading");
      const jump = create("button", timecode(event.timecode_sec), "timeline-button"); jump.type = "button";
      jump.setAttribute("aria-label", `Перейти к ${timecode(event.timecode_sec)} в видео`);
      jump.addEventListener("click", () => { jumpTo(event.timecode_sec, index); if (!$("player").hidden) { scrollTo($("review")); $("player").focus(); } });
      heading.append(jump, create("h3", event.type)); card.append(heading, create("p", event.description));
      const frames = create("div", undefined, "frames");
      event.evidence_frames.forEach(frame => {
        const figure = create("figure"); const button = create("button", undefined, "frame-button"); button.type = "button";
        button.setAttribute("aria-label", `Увеличить кадр ${timecode(frame.timestamp_sec)}`);
        const img = create("img"); img.src = `data:image/jpeg;base64,${frame.image_base64}`;
        img.alt = `Кадр-доказательство на ${timecode(frame.timestamp_sec)}`; img.loading = "lazy";
        button.append(img); button.addEventListener("click", () => openFrame(frame));
        figure.append(button, create("figcaption", `${timecode(frame.timestamp_sec)} · исходный кадр`)); frames.append(figure);
      }); card.append(frames);
      const ruleBox = create("div", undefined, "rule-box");
      if (event.rule && typeof event.rule === "object") {
        const rule = event.rule;
        ruleBox.append(create("span", rule.demo_only ? "СИНТЕТИЧЕСКОЕ ТЕСТОВОЕ УСЛОВИЕ" : "ОСНОВАНИЕ ДЛЯ ПРОВЕРКИ", "section-kicker"));
        ruleBox.append(create("strong", [rule.document, rule.section].filter(value => typeof value === "string").join(" · ")));
        if (typeof rule.summary === "string") ruleBox.append(create("p", rule.summary));
        if (typeof rule.caution === "string") ruleBox.append(create("p", rule.caution, "rule-caution"));
        const link = sourceLink(rule.url); if (link) ruleBox.append(link);
      } else ruleBox.append(create("p", "Источник правила отсутствует в ответе сервиса. Требуется уточнение перед оценкой события."));
      card.append(ruleBox); $("eventList").append(card);
    });
    if (Array.isArray(meta.warnings)) meta.warnings.filter(warning => typeof warning === "string").forEach(warning => $("warningList").append(create("p", warning)));
    $("eventTimeline").hidden = !found;
    $("resultSection").hidden = false;
    $("resultTitle").focus({ preventScroll: true }); scrollTo($("resultSection"));
  }
  async function poll() {
    while (state.job) {
      const remaining = BUDGET_MS - (Date.now() - state.started);
      if (remaining <= 0) throw new ApiError("За 5 минут результат не получен. Проверьте статус задания; повторная загрузка автоматически не выполняется.");
      const result = await request(`/api/v1/jobs/${encodeURIComponent(state.job)}`, {}, Math.min(20000, remaining));
      if (result.status === "failed") {
        state.job = null;
        throw new ApiError(`Сервис сообщил об ошибке анализа. ${typeof result.error === "string" ? result.error : "Вердикт не получен."}`);
      }
      if (result.status === "done") {
        render(result); state.job = null; $("stageText").textContent = "Проверка завершена"; $("progressBar").value = 100;
        $("progressDetail").textContent = "Результат доступен ниже. Проверьте доказательства и ограничения.";
        return;
      }
      if (result.status !== "processing") throw new ApiError("Неизвестный статус задания. Результат не показан.");
      stage(result);
      await sleep(Math.min(1000, Math.max(0, BUDGET_MS - (Date.now() - state.started))));
    }
  }
  async function execute(exampleName = null, resume = false) {
    if (state.busy || Date.now() < state.retryAt) return;
    clearErrors();
    if (!validateKey()) return;
    if (!resume && !exampleName && !state.file) return showError("Выберите MOV или MP4 для проверки.", "file");
    if (!resume && state.duration > 180) return showError("Запись длиннее 180 секунд. Выберите более короткое видео.", "file");
    if (!resume && state.job) return showError("Сначала проверьте статус текущего задания.");
    state.busy = true; controls();
    $("processingPanel").hidden = false; $("stageText").textContent = resume ? "Проверяем статус задания" : exampleName ? "Загружаем синтетический пример" : "Передаём видео сервису";
    $("progressBar").removeAttribute("value"); $("progressDetail").textContent = "Оставьте страницу открытой. Загрузка не показывает процент анализа.";
    if (!resume) clearResult();
    // A manual status recheck gets a fresh bounded polling window; it never resubmits the video.
    state.started = Date.now(); startTimer();
    try {
      if (exampleName) {
        const blob = await request(`/api/v1/demo/${exampleName}`, { asBlob: true }, 30000);
        state.busy = false;
        const accepted = choose(new File([blob], `synthetic-${exampleName}.mov`, { type: "video/quicktime" }));
        state.busy = true; controls(); $("processingPanel").hidden = false;
        if (!accepted) return;
      }
      if (!resume) {
        const form = new FormData(); form.append("file", state.file);
        $("stageText").textContent = "Передаём видео сервису";
        const job = await request("/api/v1/analyze", { method: "POST", body: form }, Math.min(60000, BUDGET_MS - (Date.now() - state.started)));
        if (typeof job.job_id !== "string" || !job.job_id || job.status !== "processing") throw new ApiError("Сервис не подтвердил создание задания.");
        state.job = job.job_id;
      }
      await poll();
    } catch (error) {
      if (state.job) $("stageText").textContent = "Получение статуса приостановлено";
      else $("stageText").textContent = "Анализ не завершён";
      $("progressBar").removeAttribute("value");
      $("progressDetail").textContent = state.job ? "Задание может продолжать работу на сервере. Используйте проверку статуса." : "Можно повторить действие после устранения ошибки.";
      showError(error.message, error.field);
    } finally {
      clearInterval(state.timer); state.busy = false; controls();
      if (!$("errorBox").hidden && !$("apiKey").hasAttribute("aria-invalid") && !$("videoFile").hasAttribute("aria-invalid")) $("errorBox").focus();
      if ($("apiKey").hasAttribute("aria-invalid")) $("apiKey").focus();
      else if ($("videoFile").hasAttribute("aria-invalid")) $("videoFile").focus();
      void health();
    }
  }
  async function loadRules() {
    if (state.rulesBusy) return;
    clearErrors();
    if (!validateKey()) return;
    state.rulesBusy = true; controls(); $("sourcesStatus").textContent = "Загружаем каталог из API…";
    try {
      const catalog = await request("/api/v1/rules");
      if (!Array.isArray(catalog.rules)) throw new ApiError("Каталог получен в неизвестном формате.");
      state.rules = catalog.rules;
      $("sourceList").replaceChildren();
      catalog.rules.filter(rule => !rule.demo_only).forEach(rule => {
        const article = create("article", undefined, "source-item");
        article.append(create("span", rule.enabled ? "ВКЛЮЧЕНО В КАТАЛОГЕ" : "ОТКЛЮЧЕНО В КАТАЛОГЕ", "badge"));
        article.append(create("h3", rule.name || "Правило"), create("p", [rule.document, rule.section].filter(Boolean).join(" · "), "source-summary"));
        if (typeof rule.summary === "string") article.append(create("p", rule.summary));
        if (typeof rule.visual_test === "string") article.append(create("p", `Условия наблюдения: ${rule.visual_test}`));
        if (typeof rule.caution === "string") article.append(create("p", rule.caution));
        const link = sourceLink(rule.url); if (link) article.append(link);
        $("sourceList").append(article);
      });
      $("sourcesStatus").textContent = `Каталог сервиса${typeof catalog.version === "string" ? ` · версия ${catalog.version}` : ""}. Включение правила не подтверждает качество распознавания.`;
    } catch (error) {
      $("sourceList").replaceChildren();
      $("sourcesStatus").textContent = error.message;
      if (error.field) showError(error.message, error.field);
    } finally { state.rulesBusy = false; controls(); }
  }
  $("uploadForm").addEventListener("submit", event => { event.preventDefault(); void execute(); });
  $("videoFile").addEventListener("change", event => { if (event.target.files[0]) choose(event.target.files[0]); });
  for (const name of ["dragenter", "dragover"]) $("dropzone").addEventListener(name, event => {
    event.preventDefault(); if (!state.busy && !state.job) $("dropzone").classList.add("drag");
  });
  for (const name of ["dragleave", "drop"]) $("dropzone").addEventListener(name, event => { event.preventDefault(); $("dropzone").classList.remove("drag"); });
  $("dropzone").addEventListener("drop", event => { if (event.dataTransfer.files.length > 1) { showError("Выберите одну запись для проверки.", "file"); return; } choose(event.dataTransfer.files[0]); });
  // Prevent an accidental file drop outside the upload target from navigating away.
  window.addEventListener("dragover", event => event.preventDefault()); window.addEventListener("drop", event => event.preventDefault());
  $("player").addEventListener("loadedmetadata", () => {
    state.duration = $("player").duration;
    if (finite(state.duration)) $("fileDetail").textContent = `${(state.file.size / 1024 / 1024).toFixed(1)} МБ · ${timecode(state.duration)} · заменить файл`;
    if (state.duration > 180 && !state.busy) showError("Запись длиннее 180 секунд. Выберите более короткое видео.", "file");
  });
  $("player").addEventListener("error", () => {
    if (!state.url) return;
    $("player").hidden = true; $("previewEmpty").hidden = true; $("playbackFallback").hidden = false;
  });
  document.querySelectorAll("[data-example]").forEach(button => button.addEventListener("click", () => void execute(button.dataset.example)));
  $("resumeButton").addEventListener("click", () => { if (state.job) void execute(null, true); });
  $("resetButton").addEventListener("click", () => {
    if (state.busy) return;
    const abandoned = Boolean(state.job); state.job = null; $("videoFile").value = ""; choose(null);
    if (abandoned) showError("Получение статуса остановлено на этой странице. Уже принятое задание может продолжать обработку на сервере.");
  });
  $("loadRulesButton").addEventListener("click", () => void loadRules());
  $("apiKey").addEventListener("input", () => { $("keyError").hidden = true; $("apiKey").removeAttribute("aria-invalid"); $("sourceList").replaceChildren(); $("sourcesStatus").textContent = "Загрузите каталог с текущим ключом доступа."; });
  $("closeDialog").addEventListener("click", () => $("frameDialog").close());
  $("frameDialog").addEventListener("click", event => {
    if (event.target !== $("frameDialog")) return;
    const rect = $("frameDialog").getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $("frameDialog").close();
  });
  $("frameDialog").addEventListener("close", () => $("expandedFrame").removeAttribute("src"));
  controls(); void health();
})();
