"use strict";

const byId = (id) => document.getElementById(id);
const runSelect = byId("run-select");
const selectedRun = document.body.dataset.selectedRun || runSelect.value;

function text(value) {
  if (value === null || value === undefined || value === "") return "UNAVAILABLE";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function badge(value) {
  const span = document.createElement("span");
  span.className = `badge ${text(value).toLowerCase().replaceAll("_", "-")}`;
  span.textContent = text(value);
  return span;
}

function metric(label, value) {
  const node = document.createElement("div");
  node.className = "metric";
  const name = document.createElement("span");
  name.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = text(value);
  node.append(name, strong);
  return node;
}

function renderMetrics(target, values) {
  target.replaceChildren(...Object.entries(values).map(([key, value]) => metric(key.replaceAll("_", " "), value)));
}

function renderDetails(target, values) {
  const rows = Object.entries(values).map(([key, value]) => {
    const row = document.createElement("div");
    row.className = "detail-row";
    const name = document.createElement("span");
    name.className = "detail-key";
    name.textContent = key.replaceAll("_", " ");
    const content = document.createElement("code");
    content.textContent = text(value);
    row.append(name, content);
    return row;
  });
  target.replaceChildren(...rows);
}

function renderTable(target, rows, columns) {
  if (!rows || rows.length === 0) {
    target.textContent = "UNAVAILABLE";
    target.className = "empty-state";
    return;
  }
  const wrapper = document.createElement("div");
  wrapper.className = "table-wrap";
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach(([label]) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    headerRow.append(cell);
  });
  head.append(headerRow);
  const body = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach(([, key, mode]) => {
      const td = document.createElement("td");
      const value = typeof key === "function" ? key(row) : row[key];
      if (mode === "badge") td.append(badge(value)); else td.textContent = text(value);
      tr.append(td);
    });
    body.append(tr);
  });
  table.append(head, body);
  wrapper.append(table);
  target.replaceChildren(wrapper);
}

function chartPath(values, width, height) {
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  return values.map((value, index) => {
    const x = values.length === 1 ? width / 2 : index * width / (values.length - 1);
    const y = height - ((value - min) / range) * height;
    return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
}

function renderChart(equity) {
  const target = byId("equity-chart");
  const curve = equity.curve || [];
  if (!curve.length) { target.textContent = "UNAVAILABLE"; return; }
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", "0 0 1000 220");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Equity and drawdown curves");
  const equityLine = document.createElementNS(namespace, "path");
  equityLine.setAttribute("class", "chart-line");
  equityLine.setAttribute("d", chartPath(curve.map((item) => Number(item.equity)), 1000, 205));
  const drawdown = document.createElementNS(namespace, "path");
  drawdown.setAttribute("class", "chart-drawdown");
  drawdown.setAttribute("d", chartPath((equity.drawdown_curve || []).map((item) => Number(item.drawdown)), 1000, 70));
  svg.append(equityLine, drawdown);
  target.replaceChildren(svg);
}

function renderOverview(data) {
  renderMetrics(byId("overview-cards"), {
    system: data.system_status, mode: data.mode, run: data.run_id,
    equity: data.current_equity, cash: data.cash, exposure: data.gross_exposure,
    net_pnl_known: data.net_pnl_known, max_drawdown: data.max_drawdown,
    risk: data.risk_state, positions: data.position_count,
    strategies: data.active_strategy_count, ml: data.ml_mode,
    data_quality: data.data_quality_status, cost_coverage: data.cost_coverage_status,
  });
  byId("source-dot").className = "status-dot ok";
  byId("source-message").textContent = `${data.integrity} · schema ${data.schema_version}`;
}

function renderPortfolio(data) {
  renderTable(byId("positions"), data.positions, [["Symbol", "symbol"], ["Quantity", "quantity"]]);
  renderTable(byId("targets"), data.targets, [["Symbol", "symbol"], ["Current", "current_weight"], ["Target", "target_weight"], ["Group", "group"]]);
  renderTable(byId("sleeves"), data.sleeves, [["Strategy", "strategy_name"], ["Symbol", "symbol"], ["Weight", "target_weight_contribution"]]);
  renderDetails(byId("portfolio-summary"), {
    engine: data.engine_name, version: data.engine_version, config_hash: data.config_hash,
    opportunities: data.opportunities, decision_counts: data.decision_counts,
    metrics: data.metrics,
  });
}

function renderStrategies(data) {
  const cards = (data.strategies || []).map((item) => {
    const card = document.createElement("article"); card.className = "card";
    const title = document.createElement("h3"); title.textContent = `${item.name} · ${item.version}`;
    const details = document.createElement("p"); details.textContent = `Signals ${item.signal_count} · Enter ${item.enter_signals} · Exit ${item.exit_signals}`;
    const config = document.createElement("code"); config.textContent = text(item.config);
    card.append(title, details, config); return card;
  });
  byId("strategy-grid").replaceChildren(...cards);
}

function renderRegimes(data) {
  renderTable(byId("regime-table"), data.latest_by_symbol, [
    ["Symbol", "symbol"], ["Structure", "structure_regime", "badge"],
    ["Volatility", "volatility_regime", "badge"], ["Bars", "bars_in_current_structure_regime"],
    ["Reasons", (row) => row.reason_codes], ["Timestamp", "timestamp"],
  ]);
}

function renderMl(data) {
  renderDetails(byId("ml-summary"), {
    mode: data.mode, model_id: data.model_id, family: data.model_family,
    version: data.model_version, lifecycle: data.model_status,
    feature_schema: data.base_feature_schema_version,
    ml_feature_schema: data.ml_feature_schema_version,
    threshold: data.threshold, decisions: data.decision_counts,
  });
  renderTable(byId("ml-predictions"), data.recent_predictions, [
    ["Timestamp", "timestamp"], ["Symbol", "symbol"], ["Strategy", "strategy_name"],
    ["Probability", "probability_positive"], ["Model", "model_id"],
  ]);
}

function renderRisk(data) {
  renderMetrics(byId("risk-summary"), {
    engine: data.engine_name, version: data.engine_version, state: data.current_state,
    config_hash: data.config_hash, decisions: data.decision_counts,
    deny_all_fail_safe: data.deny_all_fail_safe,
  });
  renderTable(byId("risk-reasons"), data.top_reasons || [], [["Reason", (row) => row[0]], ["Count", (row) => row[1]]]);
}

function renderData(data) {
  renderTable(byId("data-table"), data.datasets, [
    ["Dataset", "dataset_id"], ["Provider", "provider"], ["Symbol", "symbol"],
    ["Timeframe", "timeframe"], ["Start", "actual_start"], ["End", "actual_end"],
    ["Rows", "row_count"], ["Gaps", "gaps"], ["Duplicates", "duplicates"],
    ["Invalid", "invalid_bars"], ["Quality", "quality_status", "badge"],
    ["Checksum", "checksum_sha256"],
  ]);
}

function renderCostTable(target, values) {
  renderTable(target, Object.entries(values).filter(([name]) => name !== "total_variable_cost").map(([name, item]) => ({name, ...item})), [
    ["Component", "name"], ["Status", "status", "badge"], ["Amount", "amount"], ["Source", "source"],
  ]);
}

function renderCosts(data) {
  byId("cost-coverage").textContent = `${data.coverage_status}: ${(data.warnings || []).join(" ") || "All supplied cost categories are covered."}`;
  renderCostTable(byId("trading-costs"), data.trading);
  renderCostTable(byId("operating-costs"), data.operating);
  renderMetrics(byId("pnl-costs"), {
    gross_pnl: data.gross_pnl, known_trading_costs: data.known_trading_costs,
    estimated_trading_costs: data.estimated_trading_costs,
    operating_costs_known: data.known_operating_costs,
    net_before_operating: data.net_trading_pnl_before_operating,
    operating_costs_total: data.operating_costs_total,
    net_economic_pnl: data.net_economic_pnl,
    net_pnl_known: data.net_pnl_known, net_pnl_estimated: data.net_pnl_estimated,
    tariff_profile: data.tariff_profile_id, tariff_status: data.tariff_status,
  });
}

function renderValidation(data) {
  renderMetrics(byId("validation-summary"), {
    status: data.status, campaign: data.real_data_campaign_status,
    final_oos: data.final_oos, tariff_profile: data.tariff_profile_id,
    tariff_status: data.tariff_status, cost_coverage: data.cost_coverage,
    survivorship_warning: data.survivorship_bias_warning,
  });
  renderTable(byId("validation-criteria"), data.criteria || [], [
    ["Criterion", "name"], ["Status", "status", "badge"],
    ["Observed", "observed"], ["Required", "required"], ["Reason", "reason"],
  ]);
}

function renderRobustness(data) {
  const readiness = data.paper_readiness || {};
  const concentration = data.concentration || {};
  const cost = data.cost_robustness || {};
  renderMetrics(byId("robustness-summary"), {
    status: data.campaign_status || data.status,
    period: data.period_classification,
    baseline_reproduced: data.baseline_reproduced,
    plan_hash: data.plan_hash,
    holdout_status: data.holdout_status || "NOT_RUN",
    paper_readiness: readiness.status || "UNAVAILABLE",
    top_contributor: concentration.top_contributor,
    top1_share: concentration.top1_positive_pnl_share,
    historical_tariff: cost.historical_tariff_status,
    survivorship: data.survivorship_status,
  });
  renderTable(byId("robustness-funnel"), (data.decision_funnel || {}).rows || [], [
    ["Strategy", "strategy_name"], ["Symbol", "symbol"], ["Candidates", "candidate_entries"],
    ["Policy eligible", "activation_eligible"], ["Portfolio", "portfolio_selected"],
    ["Risk approved", "risk_approved"], ["Fills", "filled_entries"], ["Trades", "closed_trades"],
  ]);
  renderDetails(byId("robustness-warnings"), {warnings: (data.warnings || []).join(" · ") || "UNAVAILABLE"});
  renderTable(byId("robustness-temporal"), data.temporal_rows || [], [
    ["Period", "label"], ["Status", "availability", "badge"], ["Net return", "net_return"],
    ["Trades", "closed_trades"], ["Drawdown", "max_drawdown"], ["Costs", "variable_costs"],
  ]);
  const comparisons = [
    ...(data.leave_one_symbol_out || []),
    ...(data.leave_one_strategy_out || []),
    ...(data.single_strategy_runs || []),
  ];
  renderTable(byId("robustness-loso"), comparisons, [
    ["Diagnostic", "diagnostic_type"], ["Excluded", "excluded_item"],
    ["Availability", "availability", "badge"], ["Net return", "net_return"],
    ["Drawdown", "max_drawdown"], ["Trades", "closed_trades"],
  ]);
}

function renderPaperReadiness(data) {
  const review = data.paper_readiness_v3 || data.paper_readiness_v2 || {};
  const completeness = data.economic_completeness || {};
  const tariff = data.broker_tariff || {};
  const operating = data.paper_operating_scenario || data.paper_economics || {};
  const metrics = data.metrics || {};
  const human = data.human_review || {};
  const components = completeness.components || (completeness.component_statuses || []).map((item) => ({component: item[0], status: item[1]}));
  renderMetrics(byId("paper-readiness-summary"), {
    readiness: review.status || "UNAVAILABLE",
    holdout_status: data.holdout_status || "UNAVAILABLE",
    recomputation: data.assessment_status || data.mode || "UNAVAILABLE",
    tariff_compatibility: tariff.status || "UNAVAILABLE",
    economic_completeness: completeness.status || "UNAVAILABLE",
    decision_invariance: (data.decision_invariance || {}).status || "UNAVAILABLE",
    cost_invariance: (data.cost_invariance || {}).status || "UNAVAILABLE",
    affected_sells: metrics.affected_fills,
    section31: metrics.recomputed_section31,
    pnl_delta: metrics.pnl_delta,
    human_review: human.status || review.human_review_status || "UNAVAILABLE",
    unlocks_paper_or_live: review.unlocks_paper_or_live,
  });
  renderTable(byId("evidence-components"), components, [
    ["Component", "component"], ["Status", "status", "badge"],
    ["Compatibility", "compatibility"], ["Original", "amount_in_original_run"],
    ["Missing indicated", "indicated_missing_amount"], ["Reason", "reason"],
  ]);
  renderDetails(byId("paper-operating"), {
    scenario: operating.scenario_id,
    monthly_or_period_low: operating.operating_low || operating.monthly_totals,
    period_central: operating.operating_central,
    period_high: operating.operating_high,
    net_before_operating: operating.net_before_operating,
    net_after_central: operating.net_after_central || operating.net_after_operating_central,
    break_even_monthly_fixed: operating.break_even_fixed_monthly || operating.break_even_monthly_fixed_cost,
  });
  renderTable(byId("section31-fills"), data.affected_fills || [], [
    ["Symbol", "symbol"], ["Timestamp", "timestamp"], ["Notional", "notional"],
    ["Rate/million", "rate_per_million"], ["Section 31", "section31_cost"],
    ["Recomputed total", "recomputed_total_variable_cost"],
  ]);
  renderTable(byId("paper-readiness-criteria"), review.criteria || [], [
    ["Criterion", "name"], ["Status", "status", "badge"],
    ["Observed", "observed"], ["Required", "required"], ["Reason", "reason"],
  ]);
  renderDetails(byId("paper-readiness-warnings"), {
    warnings: (review.warnings || data.warnings || []).join(" · ") || "UNAVAILABLE",
  });
}

function renderHealth(data) {
  const cards = (data.components || []).map((item) => {
    const card = document.createElement("article"); card.className = "health-card";
    const title = document.createElement("h3"); title.textContent = item.name;
    const state = badge(item.status);
    const message = document.createElement("p"); message.textContent = item.message;
    card.append(title, state, message); return card;
  });
  byId("health-grid").replaceChildren(...cards);
}

function renderTrace(trace) {
  const steps = (trace.steps || []).map((item) => {
    const node = document.createElement("article"); node.className = `trace-step ${item.status.toLowerCase()}`;
    const title = document.createElement("h3"); title.textContent = item.stage;
    const state = badge(item.status);
    const id = document.createElement("p"); id.textContent = text(item.entity_id);
    const reasons = document.createElement("p"); reasons.textContent = text(item.reason_codes || item.human_reasons);
    node.append(title, state, id, reasons); return node;
  });
  byId("trace-timeline").replaceChildren(...steps);
}

async function fetchJson(path) {
  const response = await fetch(path, {headers: {"Accept": "application/json"}, cache: "no-store"});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function loadDecisionHistory(runId) {
  const params = new URLSearchParams({run_id: runId});
  [["component", "filter-component"], ["symbol", "filter-symbol"], ["strategy", "filter-strategy"], ["status", "filter-status"], ["reason", "filter-reason"]].forEach(([name, id]) => {
    if (byId(id).value) params.set(name, byId(id).value);
  });
  const data = await fetchJson(`/api/v1/decisions?${params}`);
  renderTable(byId("decision-table"), data.decisions, [
    ["Time", "timestamp"], ["Component", "component"], ["Symbol", "symbol"],
    ["Strategy", "strategy"], ["Status", "status", "badge"], ["Reasons", "reasons"],
  ]);
}

async function loadRun(runId) {
  if (!runId) return;
  try {
    const params = `run_id=${encodeURIComponent(runId)}`;
    const [overview, equity, portfolio, strategies, regimes, ml, risk, dataQuality, costs, validation, robustness, paperReadiness, health, snapshot] = await Promise.all([
      fetchJson(`/api/v1/overview?${params}`), fetchJson(`/api/v1/equity?${params}`),
      fetchJson(`/api/v1/portfolio?${params}`), fetchJson(`/api/v1/strategies?${params}`),
      fetchJson(`/api/v1/regimes?${params}`), fetchJson(`/api/v1/ml?${params}`),
      fetchJson(`/api/v1/risk?${params}`), fetchJson(`/api/v1/data-quality?${params}`),
      fetchJson(`/api/v1/costs?${params}`), fetchJson(`/api/v1/validation?${params}`), fetchJson(`/api/v1/robustness?${params}`), fetchJson(`/api/v1/paper-readiness?${params}`), fetchJson(`/api/v1/health?${params}`),
      fetchJson(`/api/v1/snapshot?${params}`),
    ]);
    renderOverview(overview); renderChart(equity); renderDetails(byId("equity-metrics"), equity.metrics || {});
    renderPortfolio(portfolio); renderStrategies(strategies); renderRegimes(regimes); renderMl(ml);
    renderRisk(risk); renderData(dataQuality); renderCosts(costs); renderValidation(validation); renderRobustness(robustness); renderPaperReadiness(paperReadiness); renderHealth(health);
    const traceSelect = byId("trace-select");
    const traces = snapshot.decision_traces || [];
    traceSelect.replaceChildren();
    traces.forEach((trace) => {
      const option = document.createElement("option"); option.value = trace.trace_id;
      option.textContent = `${trace.symbol} · ${trace.strategy_name || "UNAVAILABLE"} · ${trace.trace_id}`;
      traceSelect.append(option);
    });
    renderTrace(traces[0] || {steps: []});
    await loadDecisionHistory(runId);
    byId("last-refresh").textContent = `Observed ${new Date().toISOString()}`;
  } catch (error) {
    byId("source-dot").className = "status-dot error";
    byId("source-message").textContent = error.message;
  }
}

async function loadBrokerInfrastructure() {
  try {
    const data = await fetchJson("/api/v1/broker/sessions");
    renderMetrics(byId("broker-guard-summary"), {
      mode: "READ_ONLY",
      paper_execution_armed: data.paper_execution_armed,
      live_hard_locked: data.live_hard_locked,
      local_sessions: (data.sessions || []).length,
    });
    renderTable(byId("broker-sessions"), data.sessions || [], [
      ["Session", "session_id"], ["Mode", "mode", "badge"],
      ["Account", "account_masked"], ["Integrity", "integrity", "badge"],
      ["Execution armed", "paper_execution_armed"],
    ]);
  } catch (error) {
    renderMetrics(byId("broker-guard-summary"), {
      status: "UNAVAILABLE", paper_execution_armed: false, live_hard_locked: true,
    });
  }
}

runSelect.addEventListener("change", () => {
  const target = runSelect.value ? `?run_id=${encodeURIComponent(runSelect.value)}` : window.location.pathname;
  window.location.assign(target);
});
byId("trace-select").addEventListener("change", async () => {
  if (!selectedRun || !byId("trace-select").value) return;
  renderTrace(await fetchJson(`/api/v1/decision-trace?run_id=${encodeURIComponent(selectedRun)}&trace_id=${encodeURIComponent(byId("trace-select").value)}`));
});
["filter-component", "filter-symbol", "filter-strategy", "filter-status", "filter-reason"].forEach((id) => {
  byId(id).addEventListener("change", () => selectedRun && loadDecisionHistory(selectedRun));
});

if (selectedRun) {
  loadRun(selectedRun);
  window.setInterval(() => loadRun(selectedRun), 5000);
}
loadBrokerInfrastructure();
window.setInterval(loadBrokerInfrastructure, 5000);
