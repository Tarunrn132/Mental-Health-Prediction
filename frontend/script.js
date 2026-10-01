(() => {
  "use strict";

  // Determine API base URL dynamically
  function resolveApiBase() {
    const origin = window.location.origin;
    if (origin && origin.startsWith("http")) {
      // If hosted together with FastAPI backend
      if (window.location.port === "8000" || window.location.port === "2200") {
        return "";
      }
    }
    // Default local backend port
    return "http://127.0.0.1:8000";
  }

  const API_BASE = resolveApiBase();
  const GAUGE_ARC_LENGTH = Math.PI * 100; // Exact semicircle perimeter for r=100 (314.159...)

  // DOM Elements
  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const resetBtn = document.getElementById("reset-btn");

  const stateIdle = document.getElementById("state-idle");
  const stateLoading = document.getElementById("state-loading");
  const stateResult = document.getElementById("state-result");

  const scoreNumberEl = document.getElementById("score-number");
  const scoreBandEl = document.getElementById("score-band");
  const scoreContextEl = document.getElementById("score-context");
  const gaugeFill = document.getElementById("gauge-fill");

  const viewMetricsBtn = document.getElementById("view-metrics-btn");
  const metricsModal = document.getElementById("metrics-modal");
  const modalCloseBtn = document.getElementById("modal-close-btn");

  // ---------------------------------------------------------
  // 1. Draw Gauge Ticks (0, 2, 4, 6, 8, 10)
  // ---------------------------------------------------------
  function drawTicks() {
    document.querySelectorAll(".gauge-ticks").forEach((g) => {
      g.innerHTML = "";
      const cx = 120, cy = 140, rOuter = 100, rInner = 88;
      for (let i = 0; i <= 10; i += 2) {
        const angle = Math.PI - (i / 10) * Math.PI; // 180deg -> 0deg
        const x1 = cx + rOuter * Math.cos(angle);
        const y1 = cy - rOuter * Math.sin(angle);
        const x2 = cx + rInner * Math.cos(angle);
        const y2 = cy - rInner * Math.sin(angle);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x1.toFixed(1));
        line.setAttribute("y1", y1.toFixed(1));
        line.setAttribute("x2", x2.toFixed(1));
        line.setAttribute("y2", y2.toFixed(1));
        g.appendChild(line);
      }
    });
  }
  drawTicks();

  // ---------------------------------------------------------
  // 2. Segmented Control for Stress Level
  // ---------------------------------------------------------
  const segGroup = document.getElementById("stress_level_group");
  const stressHiddenInput = document.getElementById("stress_level");
  const segButtons = segGroup.querySelectorAll(".seg-btn");

  function selectStress(btn) {
    segButtons.forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-checked", "false");
    });
    btn.classList.add("active");
    btn.setAttribute("aria-checked", "true");
    stressHiddenInput.value = btn.dataset.value;
    clearFieldError(stressHiddenInput);
  }

  segButtons.forEach((btn, index) => {
    btn.addEventListener("click", () => selectStress(btn));

    // Keyboard navigation (Left/Right/Up/Down arrows)
    btn.addEventListener("keydown", (e) => {
      let targetIndex = null;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        targetIndex = (index + 1) % segButtons.length;
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        targetIndex = (index - 1 + segButtons.length) % segButtons.length;
      }
      if (targetIndex !== null) {
        e.preventDefault();
        segButtons[targetIndex].focus();
        selectStress(segButtons[targetIndex]);
      }
    });
  });

  // ---------------------------------------------------------
  // 3. Field-Level Inline Error Helpers
  // ---------------------------------------------------------
  function fieldWrapper(input) {
    if (!input) return null;
    return input.closest(".field");
  }

  function setFieldError(input, message) {
    const wrap = fieldWrapper(input);
    if (!wrap) return;
    wrap.classList.add("field-error");
    const msgEl = wrap.querySelector(".error-msg");
    if (msgEl) {
      msgEl.textContent = message;
    }
  }

  function clearFieldError(input) {
    const wrap = fieldWrapper(input);
    if (!wrap) return;
    wrap.classList.remove("field-error");
    const msgEl = wrap.querySelector(".error-msg");
    if (msgEl) {
      msgEl.textContent = "";
    }
  }

  function clearAllErrors() {
    form.querySelectorAll(".field").forEach((f) => f.classList.remove("field-error"));
    form.querySelectorAll(".error-msg").forEach((m) => (m.textContent = ""));
  }

  // Live-clear errors as user types or changes input
  form.querySelectorAll("input, select").forEach((el) => {
    el.addEventListener("input", () => clearFieldError(el));
    el.addEventListener("change", () => clearFieldError(el));
  });

  // ---------------------------------------------------------
  // 4. Form Data Collection & Validation
  // ---------------------------------------------------------
  function collectPayload() {
    const fd = new FormData(form);
    return {
      age: fd.get("age") === "" ? NaN : parseInt(fd.get("age"), 10),
      gender: fd.get("gender") || "",
      country: (fd.get("country") || "").trim(),
      academic_level: fd.get("academic_level") || "",
      most_used_platform: fd.get("most_used_platform") || "",
      purpose_of_use: fd.get("purpose_of_use") || "",
      avg_daily_usage_hours: fd.get("avg_daily_usage_hours") === "" ? NaN : parseFloat(fd.get("avg_daily_usage_hours")),
      daily_unlocks: fd.get("daily_unlocks") === "" ? NaN : parseInt(fd.get("daily_unlocks"), 10),
      study_hours: fd.get("study_hours") === "" ? NaN : parseFloat(fd.get("study_hours")),
      physical_activity_hours: fd.get("physical_activity_hours") === "" ? NaN : parseFloat(fd.get("physical_activity_hours")),
      sleep_hours_per_night: fd.get("sleep_hours_per_night") === "" ? NaN : parseFloat(fd.get("sleep_hours_per_night")),
      stress_level: fd.get("stress_level") || "",
    };
  }

  function validate(payload) {
    const errors = [];

    // Numeric range validations
    const numericRules = [
      { key: "age", label: "Age", min: 10, max: 100, isInt: true },
      { key: "avg_daily_usage_hours", label: "Screen time", min: 0, max: 24, isInt: false },
      { key: "daily_unlocks", label: "Daily unlocks", min: 0, max: 2000, isInt: true },
      { key: "study_hours", label: "Study hours", min: 0, max: 24, isInt: false },
      { key: "physical_activity_hours", label: "Physical activity", min: 0, max: 24, isInt: false },
      { key: "sleep_hours_per_night", label: "Sleep duration", min: 0, max: 24, isInt: false },
    ];

    numericRules.forEach((rule) => {
      const input = document.getElementById(rule.key);
      const val = payload[rule.key];
      if (val === "" || val === null || Number.isNaN(val)) {
        errors.push([input, `${rule.label} is required.`]);
      } else if (val < rule.min || val > rule.max) {
        errors.push([input, `Must be between ${rule.min} and ${rule.max}.`]);
      }
    });

    // Required categorical selections
    const categoricalKeys = [
      { key: "gender", label: "Gender" },
      { key: "country", label: "Country" },
      { key: "academic_level", label: "Academic level" },
      { key: "most_used_platform", label: "Most-used platform" },
      { key: "purpose_of_use", label: "Primary purpose" },
    ];

    categoricalKeys.forEach((item) => {
      const input = document.getElementById(item.key);
      const val = payload[item.key];
      if (!val || String(val).trim() === "") {
        errors.push([input, `Please select ${item.label.toLowerCase()}.`]);
      }
    });

    // Stress level
    if (!payload.stress_level) {
      errors.push([stressHiddenInput, "Please select your perceived stress level."]);
    }

    return errors;
  }

  // ---------------------------------------------------------
  // 5. State Management (Idle, Loading, Result only)
  // ---------------------------------------------------------
  function showState(name) {
    [stateIdle, stateLoading, stateResult].forEach((el) => {
      if (el) el.hidden = true;
    });
    const stateMap = {
      idle: stateIdle,
      loading: stateLoading,
      result: stateResult,
    };
    if (stateMap[name]) {
      stateMap[name].hidden = false;
    }
  }

  function setSubmitting(isSubmitting) {
    submitBtn.disabled = isSubmitting;
    submitBtn.classList.toggle("loading", isSubmitting);
  }

  // ---------------------------------------------------------
  // 6. Score Interpretation, Dynamic Color & Arc Animation
  // ---------------------------------------------------------
  function getColorForScore(score) {
    const clamped = Math.max(0, Math.min(10, score));
    // Color stops mapping:
    // 0: #EF4444 (Red)
    // 2.5: #F97316 (Red / Orange)
    // 5.0: #F59E0B (Yellow / Light Orange)
    // 6.0: #FACC15 (Yellow)
    // 7.0: #A3E635 (Light Green)
    // 8.0: #84CC16 (Green)
    // 9.0: #22C55E (Green)
    // 10.0: #16A34A (Strong Green)
    const stops = [
      { score: 0.0, rgb: [239, 68, 68] },
      { score: 2.5, rgb: [249, 115, 22] },
      { score: 5.0, rgb: [245, 158, 11] },
      { score: 6.0, rgb: [250, 204, 21] },
      { score: 7.0, rgb: [163, 230, 53] },
      { score: 8.0, rgb: [132, 204, 22] },
      { score: 9.0, rgb: [34, 197, 94] },
      { score: 10.0, rgb: [22, 163, 74] }
    ];
    for (let i = 0; i < stops.length - 1; i++) {
      if (clamped >= stops[i].score && clamped <= stops[i + 1].score) {
        const t = (clamped - stops[i].score) / (stops[i + 1].score - stops[i].score);
        const r = Math.round(stops[i].rgb[0] + t * (stops[i + 1].rgb[0] - stops[i].rgb[0]));
        const g = Math.round(stops[i].rgb[1] + t * (stops[i + 1].rgb[1] - stops[i].rgb[1]));
        const b = Math.round(stops[i].rgb[2] + t * (stops[i + 1].rgb[2] - stops[i].rgb[2]));
        return `rgb(${r}, ${g}, ${b})`;
      }
    }
    return "#16A34A";
  }

  function getScoreCategory(score) {
    if (score < 4.0) {
      return {
        band: "Signal: strained",
        context: "Your reported habits correspond to elevated strain within this model. Small shifts in sleep or screen time can go a long way."
      };
    } else if (score < 7.0) {
      return {
        band: "Signal: balanced",
        context: "Your rhythm looks fairly steady within this model, with some room to recover and reset."
      };
    } else {
      return {
        band: "Signal: strong",
        context: "Your reported habits correspond to a well-supported, resilient wellness pattern within this model."
      };
    }
  }

  function animateScoreNumber(targetScore, durationMs = 1000) {
    const startTime = performance.now();
    scoreNumberEl.textContent = "0.00";

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / durationMs);
      // easeOutCubic
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = targetScore * ease;
      scoreNumberEl.textContent = current.toFixed(2);

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        scoreNumberEl.textContent = targetScore.toFixed(2);
      }
    }
    requestAnimationFrame(update);
  }

  function renderResult(score, categoryBand, contextText) {
    const numericScore = typeof score === "number" ? score : (parseFloat(score) || 0);
    const clampedScore = Math.max(0, Math.min(10, numericScore));
    const fallbackCategory = getScoreCategory(clampedScore);

    const finalBand = categoryBand || fallbackCategory.band;
    const finalContext = contextText || fallbackCategory.context;

    scoreBandEl.textContent = finalBand;
    scoreContextEl.textContent = finalContext;

    // Dynamic color update on needle hub based on score
    const dynamicScoreColor = getColorForScore(clampedScore);
    const needleHub = document.querySelector(".gauge-needle-hub");
    if (needleHub) {
      needleHub.style.fill = dynamicScoreColor;
    }

    // Measure exact SVG arc length from path
    const totalLength = (gaugeFill && typeof gaugeFill.getTotalLength === "function")
      ? gaugeFill.getTotalLength()
      : (Math.PI * 100);

    // Set dasharray to exact total path length
    gaugeFill.style.strokeDasharray = `${totalLength} ${totalLength}`;

    // Reset gauge fill to 0% (fully hidden offset) without animation
    gaugeFill.style.transition = "none";
    gaugeFill.style.strokeDashoffset = String(totalLength);

    // Show result state
    showState("result");

    // Animate smoothly to exact proportion (clampedScore / 10)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        gaugeFill.style.transition = "stroke-dashoffset 1000ms cubic-bezier(0.16, 1, 0.3, 1)";
        const percentage = clampedScore / 10;
        const targetOffset = totalLength * (1 - percentage);
        gaugeFill.style.strokeDashoffset = String(targetOffset);
        animateScoreNumber(numericScore, 1000);
      });
    });
  }

  // ---------------------------------------------------------
  // 7. Parse Backend Validation Errors (422) directly to fields
  // ---------------------------------------------------------
  function applyServerValidationErrors(detail) {
    if (!Array.isArray(detail)) return false;
    let hasMatched = false;
    detail.forEach((err) => {
      const loc = err.loc || [];
      const fieldKey = loc[loc.length - 1];
      const target = fieldKey === "stress_level"
        ? stressHiddenInput
        : document.getElementById(fieldKey);
      if (target) {
        setFieldError(target, err.msg || "Invalid value.");
        hasMatched = true;
      }
    });
    return hasMatched;
  }

  // ---------------------------------------------------------
  // 8. Form Submission Handler
  // ---------------------------------------------------------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearAllErrors();

    const payload = collectPayload();
    const clientErrors = validate(payload);

    // If client validation fails, display inline messages on the fields only.
    // Do NOT alter the prediction panel or show any error card there.
    if (clientErrors.length > 0) {
      clientErrors.forEach(([input, msg]) => {
        if (input) setFieldError(input, msg);
      });
      // Focus first error field
      if (clientErrors[0][0]) {
        clientErrors[0][0].focus();
      }
      return;
    }

    setSubmitting(true);
    showState("loading");

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (response.status === 422) {
        const errorData = await response.json().catch(() => null);
        if (errorData && errorData.detail) {
          applyServerValidationErrors(errorData.detail);
        }
        // Return prediction panel to idle without showing any error card
        showState("idle");
        return;
      }

      if (!response.ok) {
        showState("idle");
        return;
      }

      const data = await response.json();
      const score = data.score !== undefined ? data.score : data.predicted_mental_health_score;

      if (typeof score !== "number" || Number.isNaN(score)) {
        showState("idle");
        return;
      }

      renderResult(score, data.band, data.context);
    } catch (err) {
      // In case of network disconnection, return panel to idle cleanly
      showState("idle");
    } finally {
      setSubmitting(false);
    }
  });

  // ---------------------------------------------------------
  // 9. Reset Button
  // ---------------------------------------------------------
  resetBtn.addEventListener("click", () => {
    if (gaugeFill) {
      const totalLength = (typeof gaugeFill.getTotalLength === "function")
        ? gaugeFill.getTotalLength()
        : GAUGE_ARC_LENGTH;
      gaugeFill.style.transition = "none";
      gaugeFill.style.strokeDashoffset = String(totalLength);
    }
    showState("idle");
    window.scrollTo({ top: form.offsetTop - 30, behavior: "smooth" });
  });

  // ---------------------------------------------------------
  // 10. Metrics Modal Logic
  // ---------------------------------------------------------
  if (viewMetricsBtn && metricsModal) {
    viewMetricsBtn.addEventListener("click", () => {
      metricsModal.showModal();
    });

    if (modalCloseBtn) {
      modalCloseBtn.addEventListener("click", () => {
        metricsModal.close();
      });
    }

    metricsModal.addEventListener("click", (e) => {
      const rect = metricsModal.getBoundingClientRect();
      const isInDialog = (
        rect.top <= e.clientY &&
        e.clientY <= rect.top + rect.height &&
        rect.left <= e.clientX &&
        e.clientX <= rect.left + rect.width
      );
      if (!isInDialog) {
        metricsModal.close();
      }
    });
  }

})();
