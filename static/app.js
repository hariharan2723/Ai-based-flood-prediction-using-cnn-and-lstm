/**
 * AI HydroGuard Web Application JavaScript
 * Fully connected REST API integration with Flask PyTorch Backend
 */

document.addEventListener("DOMContentLoaded", () => {
    // ----------------------------------------------------
    // State & Variables
    // ----------------------------------------------------
    const SEQ_LEN = 12;
    let rainfallChart = null;
    let metricsChart = null;
    let currentPresets = [];

    // Default rainfall defaults
    const defaultRainfall = [8, 10, 12, 15, 18, 20, 24, 28, 32, 38, 45, 50];

    // DOM Elements
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const currentTabTitle = document.getElementById("current-tab-title");
    const currentTabDesc = document.getElementById("current-tab-desc");

    const waterInput = document.getElementById("input-water");
    const waterDisplay = document.getElementById("water-val-display");
    const rainfallContainer = document.getElementById("rainfall-inputs-container");
    const locationInput = document.getElementById("input-location");

    const btnRunPrediction = document.getElementById("btn-run-prediction");
    const btnResetInputs = document.getElementById("btn-reset-inputs");
    const btnQuickRun = document.getElementById("btn-quick-run");
    const presetContainer = document.getElementById("preset-buttons-container");

    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultContent = document.getElementById("result-content");
    const riskPctText = document.getElementById("risk-pct-text");
    const gaugeCircle = document.getElementById("gauge-circle");
    const riskBadge = document.getElementById("risk-badge");
    const resultTitle = document.getElementById("result-title");
    const resultDesc = document.getElementById("result-desc");
    const execTime = document.getElementById("exec-time");
    const barSpatial = document.getElementById("bar-spatial");
    const valSpatial = document.getElementById("val-spatial");
    const barTemporal = document.getElementById("bar-temporal");
    const valTemporal = document.getElementById("val-temporal");
    const recommendationsList = document.getElementById("recommendations-list");
    const satImgPreview = document.getElementById("sat-img-preview");

    const btnRefreshHeatmap = document.getElementById("btn-refresh-heatmap");
    const heatmapGridContainer = document.getElementById("heatmap-grid-container");
    const zoneDetailsContent = document.getElementById("zone-details-content");

    const historyTableBody = document.getElementById("history-table-body");
    const btnClearHistory = document.getElementById("btn-clear-history");

    // Tab Header Descriptions Map
    const tabHeaders = {
        "tab-overview": { title: "Early Flood Warning System Overview", desc: "Hybrid Deep Learning Model Architecture & System Features" },
        "tab-prediction": { title: "Real-Time Flood Risk Prediction Engine", desc: "Input rainfall sequences & satellite data to execute PyTorch neural inference" },
        "tab-heatmap": { title: "Regional Flood Risk Heatmap Matrix", desc: "Geographic spatial risk grid visualization across monitoring zones" },
        "tab-analytics": { title: "CNN-LSTM Model Performance Analytics", desc: "Validation metrics, confusion performance & network parameter specs" },
        "tab-history": { title: "Prediction Session Audit Logs", desc: "Historical query records and classification logs" }
    };

    // ----------------------------------------------------
    // Initialization
    // ----------------------------------------------------
    initUI();
    checkBackendHealth();
    loadPresets();
    initRainfallChart();
    fetchHeatmap();
    fetchAnalytics();
    fetchHistory();

    // ----------------------------------------------------
    // Tab Navigation Logic
    // ----------------------------------------------------
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");

            navButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(pane => pane.classList.remove("active"));

            btn.classList.add("active");
            const activePane = document.getElementById(targetTab);
            if (activePane) activePane.classList.add("active");

            if (tabHeaders[targetTab]) {
                currentTabTitle.textContent = tabHeaders[targetTab].title;
                currentTabDesc.textContent = tabHeaders[targetTab].desc;
            }
        });
    });

    // ----------------------------------------------------
    // UI Renderers & Setup
    // ----------------------------------------------------
    function initUI() {
        // Render 12 Rainfall Input Boxes
        rainfallContainer.innerHTML = "";
        for (let i = 0; i < SEQ_LEN; i++) {
            const box = document.createElement("div");
            box.className = "rain-input-box";
            box.innerHTML = `
                <span>T${i + 1} (hr)</span>
                <input type="number" min="0" max="300" step="1" class="form-control rain-val" data-index="${i}" value="${defaultRainfall[i]}">
            `;
            rainfallContainer.appendChild(box);
        }

        // Attach Event Listeners to Inputs
        document.querySelectorAll(".rain-val").forEach(input => {
            input.addEventListener("input", updateChartFromInputs);
        });

        waterInput.addEventListener("input", (e) => {
            waterDisplay.textContent = `${e.target.value}%`;
        });

        btnResetInputs.addEventListener("click", resetInputs);
        btnRunPrediction.addEventListener("click", runPrediction);
        btnQuickRun.addEventListener("click", () => {
            document.querySelector('[data-tab="tab-prediction"]').click();
            runPrediction();
        });

        btnRefreshHeatmap.addEventListener("click", fetchHeatmap);
        btnClearHistory.addEventListener("click", clearHistory);
    }

    function resetInputs() {
        const inputs = document.querySelectorAll(".rain-val");
        inputs.forEach((inp, idx) => {
            inp.value = defaultRainfall[idx];
        });
        waterInput.value = 65;
        waterDisplay.textContent = "65%";
        updateChartFromInputs();
    }

    // ----------------------------------------------------
    // Chart.js Setup & Updates
    // ----------------------------------------------------
    function initRainfallChart() {
        const ctx = document.getElementById("rainfallChart").getContext("2d");
        rainfallChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: Array.from({ length: 12 }, (_, i) => `T${i + 1}h`),
                datasets: [{
                    label: "Rainfall Observation (mm)",
                    data: defaultRainfall,
                    borderColor: "#00f2fe",
                    backgroundColor: "rgba(0, 242, 254, 0.15)",
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: "#ffffff",
                    pointBorderColor: "#00f2fe",
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: "index",
                        intersect: false,
                        callbacks: {
                            label: (ctx) => `${ctx.raw} mm rain`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#8a99ad", font: { size: 11 } }
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#8a99ad", font: { size: 11 } }
                    }
                }
            }
        });
    }

    function updateChartFromInputs() {
        const inputs = document.querySelectorAll(".rain-val");
        const vals = Array.from(inputs).map(inp => parseFloat(inp.value) || 0);
        if (rainfallChart) {
            rainfallChart.data.datasets[0].data = vals;
            rainfallChart.update();
        }
    }

    function setRainfallValues(arr) {
        const inputs = document.querySelectorAll(".rain-val");
        inputs.forEach((inp, idx) => {
            if (arr[idx] !== undefined) {
                inp.value = arr[idx];
            }
        });
        updateChartFromInputs();
    }

    // ----------------------------------------------------
    // Backend API Calls
    // ----------------------------------------------------

    async function checkBackendHealth() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            const badge = document.getElementById("server-status-badge");
            if (data.status === "online") {
                badge.innerHTML = `<i class="fa-solid fa-circle-check"></i> PyTorch Backend Online`;
            }
        } catch (e) {
            const badge = document.getElementById("server-status-badge");
            badge.className = "server-badge badge-danger";
            badge.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> Server Disconnected`;
        }
    }

    async function loadPresets() {
        try {
            const res = await fetch("/api/presets");
            currentPresets = await res.json();
            presetContainer.innerHTML = "";

            currentPresets.forEach((p, idx) => {
                const btn = document.createElement("button");
                btn.className = `btn-preset ${idx === 0 ? 'active' : ''}`;
                btn.innerHTML = p.title;
                btn.addEventListener("click", () => {
                    document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");

                    setRainfallValues(p.rainfall);
                    waterInput.value = p.water_body_pct;
                    waterDisplay.textContent = `${p.water_body_pct}%`;
                    runPrediction();
                });
                presetContainer.appendChild(btn);
            });
        } catch (e) {
            console.error("Failed to load presets:", e);
        }
    }

    async function runPrediction() {
        // Collect Inputs
        const inputs = document.querySelectorAll(".rain-val");
        const rainfallSeq = Array.from(inputs).map(inp => parseFloat(inp.value) || 0);
        const waterPct = parseFloat(waterInput.value) || 0;
        const locationName = locationInput.value || "District Zone A-1";

        // Loading State
        btnRunPrediction.disabled = true;
        btnRunPrediction.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running CNN-LSTM Neural Network...`;

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    rainfall: rainfallSeq,
                    water_body_pct: waterPct,
                    location: locationName
                })
            });

            const data = await response.json();

            // Render Results
            resultPlaceholder.classList.add("hidden");
            resultContent.classList.remove("hidden");

            riskPctText.textContent = `${data.risk_pct}%`;
            execTime.textContent = `${data.execution_time_ms} ms`;

            // Update Safety Pill & Badges
            const safetyPill = document.getElementById("safety-status-pill");
            const safetyIcon = document.getElementById("safety-icon");
            const safetyText = document.getElementById("safety-text");

            const isSafe = data.safety_status === "SAFE";
            if (isSafe) {
                safetyPill.className = "safety-pill safe";
                safetyIcon.className = "fa-solid fa-shield-check";
                safetyText.textContent = "STATUS: SAFE";
                resultTitle.textContent = "✅ SAFE ENVIRONMENTAL CONDITION";
                resultDesc.textContent = `Low flood probability (${data.risk_pct}%). All environmental inputs are currently within safe operational thresholds.`;
            } else {
                safetyPill.className = `safety-pill ${data.badge_class === "danger" ? "unsafe-high" : "unsafe-mod"}`;
                safetyIcon.className = "fa-solid fa-triangle-exclamation";
                safetyText.textContent = "STATUS: UNSAFE";
                if (data.badge_class === "danger") {
                    resultTitle.textContent = "🚨 CRITICAL FLOOD WARNING (UNSAFE)";
                    resultDesc.textContent = `Severe flood risk detected (${data.risk_pct}%) by PyTorch CNN-LSTM spatial & temporal fusion model. Immediate action required.`;
                } else {
                    resultTitle.textContent = "⚠️ ELEVATED FLOOD RISK (UNSAFE)";
                    resultDesc.textContent = `Elevated flood probability (${data.risk_pct}%). Monitoring and advisory protocols initiated.`;
                }
            }

            // Update Gauge Circle Conic Gradient
            const riskDegree = (data.risk_pct / 100) * 360;
            let gaugeColor = "#00e676";
            if (data.badge_class === "danger") gaugeColor = "#ff4b2b";
            else if (data.badge_class === "warning") gaugeColor = "#ffb020";

            gaugeCircle.style.background = `conic-gradient(${gaugeColor} 0deg ${riskDegree}deg, rgba(255, 255, 255, 0.08) ${riskDegree}deg 360deg)`;

            riskBadge.className = `badge ${data.badge_class}`;
            riskBadge.textContent = data.status;

            // Update Slide 10 Layout: Information about Chennai & Environmental Parameters
            const slideText = document.getElementById("slide-prediction-text");
            const infoLocHeader = document.getElementById("info-location-header");
            if (infoLocHeader) infoLocHeader.innerHTML = `<i class="fa-solid fa-city"></i> Information about ${locationName}`;

            if (slideText) {
                if (isSafe) {
                    slideText.textContent = "Safe";
                    slideText.className = "prediction-large-val safe";
                } else {
                    slideText.textContent = "Unsafe";
                    slideText.className = "prediction-large-val unsafe";
                }
            }

            if (data.env_parameters) {
                const ep = data.env_parameters;
                const eTemp = document.getElementById("env-temp");
                const eMax = document.getElementById("env-max-temp");
                const eWind = document.getElementById("env-wind");
                const eCloud = document.getElementById("env-cloud");
                const ePrecip = document.getElementById("env-precip");
                const eHum = document.getElementById("env-humidity");

                if (eTemp) eTemp.textContent = ep.temperature_f;
                if (eMax) eMax.textContent = ep.max_temp_f;
                if (eWind) eWind.textContent = ep.windspeed_mph;
                if (eCloud) eCloud.textContent = ep.cloud_cover_pct;
                if (ePrecip) ePrecip.textContent = ep.precipitation_mm;
                if (eHum) eHum.textContent = ep.humidity_pct;
            }

            // Auto alert dispatch if UNSAFE
            if (!isSafe) {
                triggerAlertDispatch("all", true);
            }

            // Progress Bars
            barSpatial.style.width = `${data.spatial_score}%`;
            valSpatial.textContent = `${data.spatial_score}%`;
            barTemporal.style.width = `${data.temporal_score}%`;
            valTemporal.textContent = `${data.temporal_score}%`;

            // Recommendations
            recommendationsList.innerHTML = "";
            data.recommendations.forEach(rec => {
                const li = document.createElement("li");
                li.textContent = rec;
                recommendationsList.appendChild(li);
            });

            // Satellite Image Preview Update
            satImgPreview.src = data.satellite_image;

            // Refresh History Table
            fetchHistory();

        } catch (e) {
            alert("Error connecting to backend model service: " + e.message);
        } finally {
            btnRunPrediction.disabled = false;
            btnRunPrediction.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Calculate Flood Risk Prediction`;
        }
    }

    async function fetchHeatmap() {
        try {
            const res = await fetch("/api/heatmap");
            const data = await res.json();

            heatmapGridContainer.innerHTML = "";
            data.cells.forEach(row => {
                row.forEach(cell => {
                    const el = document.createElement("div");
                    el.className = "heatmap-cell";
                    el.style.backgroundColor = cell.color;
                    el.innerHTML = `<span>${cell.risk_pct}%</span>`;

                    // Hover/Click details
                    el.addEventListener("mouseenter", () => {
                        const safeClass = cell.safety_status === "SAFE" ? "color: var(--status-safe)" : "color: var(--status-danger)";
                        zoneDetailsContent.innerHTML = `
                            <div style="font-size: 14px; font-weight: 700; color: var(--text-heading); margin-bottom: 6px;">
                                ${cell.zone}
                            </div>
                            <div style="display: flex; flex-direction: column; gap: 4px; font-size: 13px;">
                                <span>Safety Status: <strong style="${safeClass}">${cell.safety_status || (cell.risk_pct < 45 ? 'SAFE' : 'UNSAFE')}</strong></span>
                                <span>Risk Level: <strong style="color:${cell.color}">${cell.risk_pct}% (${cell.category})</strong></span>
                                <span>Latitude: <strong>${cell.lat}° N</strong></span>
                                <span>Longitude: <strong>${cell.lon}° E</strong></span>
                            </div>
                        `;
                    });

                    heatmapGridContainer.appendChild(el);
                });
            });
        } catch (e) {
            console.error("Heatmap fetch error:", e);
        }
    }

    async function fetchAnalytics() {
        try {
            const res = await fetch("/api/analytics");
            const data = await res.json();

            const metricsContainer = document.getElementById("analytics-metrics-container");
            const specsTable = document.getElementById("analytics-specs-table");

            const m = data.performance_metrics;
            metricsContainer.innerHTML = `
                <div class="metric-stat-box">
                    <h5>Classification Accuracy</h5>
                    <p>${(m.accuracy * 100).toFixed(2)}%</p>
                </div>
                <div class="metric-stat-box">
                    <h5>Precision Score</h5>
                    <p>${(m.precision * 100).toFixed(2)}%</p>
                </div>
                <div class="metric-stat-box">
                    <h5>Recall Sensitivity</h5>
                    <p>${(m.recall * 100).toFixed(2)}%</p>
                </div>
                <div class="metric-stat-box">
                    <h5>F1-Score Measure</h5>
                    <p>${(m.f1_score * 100).toFixed(2)}%</p>
                </div>
            `;

            // Specs Table
            specsTable.innerHTML = `
                <tr><td>Spatial Extractor</td><td>${data.model_architecture.spatial_extractor}</td></tr>
                <tr><td>Temporal Extractor</td><td>${data.model_architecture.temporal_extractor}</td></tr>
                <tr><td>Fusion Classifier</td><td>${data.model_architecture.fusion_classifier}</td></tr>
                <tr><td>Total Parameters</td><td>${data.model_architecture.total_parameters.toLocaleString()} trainable weights</td></tr>
                <tr><td>Total Samples</td><td>${data.dataset_info.total_samples} synthetic observations</td></tr>
                <tr><td>Sequence Resolution</td><td>${data.dataset_info.seq_length}</td></tr>
            `;

            // Render Chart
            renderMetricsChart(m);
        } catch (e) {
            console.error("Analytics fetch error:", e);
        }
    }

    function renderMetricsChart(m) {
        const ctx = document.getElementById("metricsChart").getContext("2d");
        if (metricsChart) metricsChart.destroy();

        metricsChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Accuracy", "Precision", "Recall", "F1-Score"],
                datasets: [{
                    label: "Validation Metric (%)",
                    data: [m.accuracy * 100, m.precision * 100, m.recall * 100, m.f1_score * 100],
                    backgroundColor: ["#00f2fe", "#4facfe", "#00e676", "#ffb020"],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        min: 70,
                        max: 100,
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#8a99ad" }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#8a99ad" }
                    }
                }
            }
        });
    }

    async function fetchHistory() {
        try {
            const res = await fetch("/api/history");
            const data = await res.json();

            historyTableBody.innerHTML = "";
            if (!data.history || data.history.length === 0) {
                historyTableBody.innerHTML = `
                    <tr><td colspan="8" class="text-center py-4 text-muted">No prediction logs recorded yet.</td></tr>
                `;
                return;
            }

            data.history.forEach(item => {
                const tr = document.createElement("tr");
                const safeVal = item.safety_status || (item.risk_pct < 45 ? "SAFE" : "UNSAFE");
                const safeBadgeClass = safeVal === "SAFE" ? "badge-success" : "badge-danger";
                tr.innerHTML = `
                    <td><strong>#${item.id}</strong></td>
                    <td style="font-size:12px;">${item.timestamp}</td>
                    <td>${item.location}</td>
                    <td>${item.rainfall_avg} mm</td>
                    <td>${item.water_pct}%</td>
                    <td><strong>${item.risk_pct}%</strong></td>
                    <td><span class="badge ${safeBadgeClass}">${safeVal}</span></td>
                    <td><span class="badge ${item.badge_class}">${item.status}</span></td>
                `;
                historyTableBody.appendChild(tr);
            });
        } catch (e) {
            console.error("History fetch error:", e);
        }
    }

    async function clearHistory() {
        if (!confirm("Are you sure you want to clear all prediction history logs?")) return;
        try {
            await fetch("/api/history", { method: "DELETE" });
            fetchHistory();
        } catch (e) {
            alert("Error clearing history: " + e.message);
        }
    }

    // ----------------------------------------------------
    // Emergency Alert System Dispatcher
    // ----------------------------------------------------
    const btnSendEmail = document.getElementById("btn-send-email");
    const btnSendSms = document.getElementById("btn-send-sms");
    const btnSendWhatsapp = document.getElementById("btn-send-whatsapp");
    const btnSendAllAlerts = document.getElementById("btn-send-all-alerts");

    const emailInput = document.getElementById("alert-email-input");
    const phoneInput = document.getElementById("alert-phone-input");
    const alertStatusBanner = document.getElementById("alert-status-banner");
    const alertStatusMsg = document.getElementById("alert-status-msg");

    const alertQuickLinks = document.getElementById("alert-quick-links");

    if (btnSendEmail) btnSendEmail.addEventListener("click", () => triggerAlertDispatch("email"));
    if (btnSendSms) btnSendSms.addEventListener("click", () => triggerAlertDispatch("sms"));
    if (btnSendWhatsapp) btnSendWhatsapp.addEventListener("click", () => triggerAlertDispatch("whatsapp"));
    if (btnSendAllAlerts) btnSendAllAlerts.addEventListener("click", () => triggerAlertDispatch("all"));

    async function triggerAlertDispatch(alertType = "all", autoTrigger = false) {
        const email = emailInput ? emailInput.value.trim() : "haran6124@gmail.com";
        const phone = phoneInput ? phoneInput.value.trim() : "9952613367";
        const locationName = locationInput ? locationInput.value : "Chennai Region";

        const riskText = riskPctText ? riskPctText.textContent.replace("%", "") : "85";
        const safetyPill = document.getElementById("safety-status-pill");
        const safetyStatus = safetyPill && safetyPill.classList.contains("safe") ? "SAFE" : "UNSAFE";

        try {
            const res = await fetch("/api/send-alert", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    email: email,
                    phone: phone,
                    safety_status: safetyStatus,
                    risk_pct: parseFloat(riskText) || 85.0,
                    location: locationName,
                    alert_type: alertType
                })
            });

            const data = await res.json();
            if (data.success && data.targets) {
                const t = data.targets;

                if (alertStatusBanner && alertStatusMsg) {
                    alertStatusBanner.classList.remove("hidden");
                    let statusText = `Alert ready! Targets: Email (${email}), SMS (${phone}) & WhatsApp (+91 ${phone})`;
                    if (alertType === "email") statusText = `Email alert prepared for ${email}!`;
                    if (alertType === "sms") statusText = `SMS alert prepared for ${phone}!`;
                    if (alertType === "whatsapp") statusText = `WhatsApp warning message generated for +91 ${phone}!`;
                    
                    alertStatusMsg.textContent = statusText;

                    if (alertQuickLinks) {
                        alertQuickLinks.innerHTML = `
                            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px;">
                                <a href="${t.whatsapp_url}" target="_blank" class="btn btn-sm btn-whatsapp">
                                    <i class="fa-brands fa-whatsapp"></i> Open WhatsApp Alert (${phone})
                                </a>
                                <a href="${t.sms_url}" class="btn btn-sm btn-outline">
                                    <i class="fa-solid fa-comment-sms"></i> Open Mobile SMS (${phone})
                                </a>
                                <a href="${t.mailto_url}" class="btn btn-sm btn-outline">
                                    <i class="fa-solid fa-envelope"></i> Open Email Client (${email})
                                </a>
                            </div>
                        `;
                    }
                }

                // If user clicked single action, open target link directly
                if (!autoTrigger) {
                    if (alertType === "whatsapp" && t.whatsapp_url) {
                        window.open(t.whatsapp_url, "_blank");
                    } else if (alertType === "sms" && t.sms_url) {
                        window.location.href = t.sms_url;
                    } else if (alertType === "email" && t.mailto_url) {
                        window.location.href = t.mailto_url;
                    } else if (alertType === "all" && t.whatsapp_url) {
                        window.open(t.whatsapp_url, "_blank");
                    }
                }
            }
        } catch (e) {
            console.error("Alert dispatch error:", e);
        }
    }
});
