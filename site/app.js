const PATCHES = ["16.05", "16.06", "16.07"];
const state = {
  patch: "16.07",
  data: null,
  selected: new Set(),
  search: "",
  result: null,
  initialized: false,
};

const el = (id) => document.getElementById(id);
const pct = (value, digits = 2) => value == null || Number.isNaN(value) ? "n/a" : `${(value * 100).toFixed(digits)}%`;
const number = (value) => value == null ? "n/a" : Number(value).toLocaleString();
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
}[char]));

function iconPath(champion) {
  const record = state.data?.champions.find((item) => item.name === champion);
  return record?.icon ? `assets/champion_icons/${encodeURIComponent(record.icon)}` : null;
}

function initials(name) {
  return name.split(/\s+/).slice(0, 2).map((part) => part[0]).join("").toUpperCase();
}

function iconHtml(champion, className = "champion-icon") {
  const path = iconPath(champion);
  return path
    ? `<img class="${className}" src="${path}" alt="">`
    : `<span class="${className} placeholder-icon">${escapeHtml(initials(champion))}</span>`;
}

function championRecord(name) {
  return state.data.champions.find((champion) => champion.name === name) || {};
}

async function loadPatch(patch) {
  setStatus("Loading patch data...", "busy");
  try {
    const response = await fetch(`data/${patch}.json`);
    if (!response.ok) throw new Error(`Patch data returned ${response.status}`);
    const previous = state.selected;
    state.data = await response.json();
    state.patch = patch;
    const available = new Set(state.data.champions.map((champion) => champion.name));
    state.selected = state.initialized
      ? new Set([...previous].filter((champion) => available.has(champion)))
      : new Set(available);
    state.initialized = true;
    state.result = null;
    renderPicker();
    resetResults();
    setStatus("Ready");
  } catch (error) {
    setRunStatus(`Could not load patch data: ${error.message}`, true);
    setStatus("Data error");
  }
}

function renderPicker() {
  const champions = state.data?.champions || [];
  const selected = [...state.selected].sort((a, b) => a.localeCompare(b));
  el("candidate-count").textContent = `${selected.length} / ${champions.length}`;

  el("selected-champions").innerHTML = selected.length
    ? selected.map((champion) => `
      <div class="selected-chip">
        ${iconHtml(champion)}
        <span>${escapeHtml(champion)}</span>
        <button class="chip-remove" type="button" data-remove="${escapeHtml(champion)}" aria-label="Remove ${escapeHtml(champion)}">×</button>
      </div>`).join("")
    : `<span class="selected-empty">No champions selected.</span>`;

  const query = state.search.trim().toLowerCase();
  const available = champions.filter((champion) =>
    !state.selected.has(champion.name) && champion.name.toLowerCase().includes(query)
  );
  el("available-champions").innerHTML = available.length
    ? available.map((champion) => `
      <div class="available-row">
        ${iconHtml(champion.name)}
        <span class="available-name">${escapeHtml(champion.name)}</span>
        <span class="available-stat">${pct(champion.pickrate)}</span>
        <button class="add-button" type="button" data-add="${escapeHtml(champion.name)}" aria-label="Add ${escapeHtml(champion.name)}">+</button>
      </div>`).join("")
    : `<div class="selected-empty">No matching champions available.</div>`;
}

function resetResults() {
  el("results").hidden = true;
  el("empty-state").hidden = false;
  el("search-badge").textContent = "Ready";
  el("search-badge").className = "status-badge";
}

function setRunStatus(message, isError = false) {
  el("run-status").textContent = message;
  el("run-status").className = `run-status${isError ? " error" : ""}`;
}

function setStatus(message, type = "") {
  el("search-badge").textContent = message;
  el("search-badge").className = `status-badge${type ? ` ${type}` : ""}`;
}

function buildLookups() {
  const matchup = new Map();
  const matchupGames = new Map();
  for (const row of state.data.matchups) {
    const key = `${row.champion}\u0000${row.enemy}`;
    matchup.set(key, row.winrate);
    matchupGames.set(key, row.games);
  }
  const frequencies = state.data.frequencies.filter((row) => Number.isFinite(row.frequency));
  return { matchup, matchupGames, frequencies };
}

function scoreChampion(champion, lookups) {
  let weighted = 0;
  let mass = 0;
  for (const enemy of lookups.frequencies) {
    if (champion === enemy.enemy) continue;
    const value = lookups.matchup.get(`${champion}\u0000${enemy.enemy}`);
    if (value == null) continue;
    weighted += enemy.frequency * value;
    mass += enemy.frequency;
  }
  return mass > 0 ? weighted / mass : null;
}

function scorePool(pool, lookups) {
  let weighted = 0;
  let mass = 0;
  for (const enemy of lookups.frequencies) {
    const values = pool
      .filter((champion) => champion !== enemy.enemy)
      .map((champion) => lookups.matchup.get(`${champion}\u0000${enemy.enemy}`))
      .filter((value) => value != null);
    if (!values.length) continue;
    weighted += enemy.frequency * Math.max(...values);
    mass += enemy.frequency;
  }
  return mass > 0 ? weighted / mass : null;
}

function combinations(items, size) {
  const result = [];
  const current = [];
  function visit(start) {
    if (current.length === size) {
      result.push([...current]);
      return;
    }
    for (let index = start; index <= items.length - (size - current.length); index += 1) {
      current.push(items[index]);
      visit(index + 1);
      current.pop();
    }
  }
  visit(0);
  return result;
}

function chooseCount(n, k) {
  if (k < 0 || k > n) return 0;
  const short = Math.min(k, n - k);
  let result = 1;
  for (let i = 1; i <= short; i += 1) result = result * (n - short + i) / i;
  return Math.round(result);
}

function topPoolsExact(candidates, poolSize, lookups, topN = 5) {
  const best = [];
  for (const pool of combinations(candidates, poolSize)) {
    const score = scorePool(pool, lookups);
    if (score == null) continue;
    best.push({ pool, score, label: pool.join(", ") });
  }
  return best.sort((a, b) => b.score - a.score || a.label.localeCompare(b.label)).slice(0, topN);
}

function topPoolsBeam(candidates, poolSize, lookups, topN = 5) {
  const beamWidth = Math.max(1000, topN * 10);
  let beam = [[]];
  for (let depth = 0; depth < poolSize; depth += 1) {
    const expanded = [];
    for (const partial of beam) {
      const start = partial.length ? candidates.indexOf(partial.at(-1)) + 1 : 0;
      for (let index = start; index < candidates.length; index += 1) {
        const pool = [...partial, candidates[index]];
        const score = scorePool(pool, lookups);
        if (score != null) expanded.push({ pool, score, label: pool.join(", ") });
      }
    }
    expanded.sort((a, b) => b.score - a.score || a.label.localeCompare(b.label));
    beam = expanded.slice(0, beamWidth).map((item) => item.pool);
  }
  return beam
    .map((pool) => ({ pool, score: scorePool(pool, lookups), label: pool.join(", ") }))
    .sort((a, b) => b.score - a.score || a.label.localeCompare(b.label))
    .slice(0, topN);
}

function buildResult(candidates, poolSize) {
  const lookups = buildLookups();
  const blindScores = candidates
    .map((champion) => ({ champion, score: scoreChampion(champion, lookups) }))
    .filter((row) => row.score != null)
    .sort((a, b) => b.score - a.score || a.champion.localeCompare(b.champion));
  const count = chooseCount(candidates.length, poolSize);
  const topPools = count <= 250000
    ? topPoolsExact(candidates, poolSize, lookups)
    : topPoolsBeam(candidates, poolSize, lookups);
  if (!topPools.length || !blindScores.length) throw new Error("No scorable pools could be generated.");

  const bestPool = topPools[0].pool;
  const counterRows = lookups.frequencies.map((enemy) => {
    const values = bestPool.map((champion) => ({
      champion,
      value: champion === enemy.enemy ? null : lookups.matchup.get(`${champion}\u0000${enemy.enemy}`) ?? null,
    }));
    const usable = values.filter((item) => item.value != null);
    const max = usable.length ? Math.max(...usable.map((item) => item.value)) : null;
    return { ...enemy, values, best: new Set(usable.filter((item) => Math.abs(item.value - max) < 1e-12).map((item) => item.champion)) };
  });

  const responsibility = new Map(bestPool.map((champion) => [champion, 0]));
  for (const row of counterRows) {
    if (!row.best.size) continue;
    const winner = bestPool.find((champion) => row.best.has(champion));
    responsibility.set(winner, responsibility.get(winner) + row.frequency);
  }
  const coveredMass = [...responsibility.values()].reduce((sum, value) => sum + value, 0);

  const skipped = [];
  let removedMass = 0;
  for (const row of counterRows) {
    for (const item of row.values) {
      if (item.champion === row.enemy) {
        skipped.push({ pool: item.champion, enemy: row.enemy, reason: "self matchup", frequency: row.frequency });
      } else if (item.value == null) {
        skipped.push({ pool: item.champion, enemy: row.enemy, reason: "missing matchup value", frequency: row.frequency });
      }
    }
    if (!row.values.some((item) => item.value != null)) removedMass += row.frequency;
  }

  return {
    lookups,
    candidates,
    poolSize,
    blindScores,
    topPools,
    bestPool,
    bestScore: topPools[0].score,
    bestBlind: blindScores[0],
    counterRows,
    responsibility: [...responsibility].map(([champion, mass]) => ({
      champion,
      mass,
      share: coveredMass ? mass / coveredMass : 0,
    })).sort((a, b) => b.share - a.share || a.champion.localeCompare(b.champion)),
    exclusions: { skipped, removedMass },
    searchMethod: count <= 250000 ? `Exact search, ${count.toLocaleString()} pools` : `Beam search, ${count.toLocaleString()} possible pools`,
  };
}

function colorForWinrate(value) {
  if (value == null) return "#20252c";
  const distance = Math.min(Math.abs(value - 0.5) / 0.08, 1);
  if (value < 0.5) {
    const light = 31 + (1 - distance) * 6;
    const saturation = 28 + distance * 14;
    return `hsl(4 ${saturation}% ${light}%)`;
  }
  const light = 29 + (1 - distance) * 8;
  const saturation = 26 + distance * 20;
  return `hsl(137 ${saturation}% ${light}%)`;
}

function renderResult() {
  const result = state.result;
  el("empty-state").hidden = true;
  el("results").hidden = false;
  el("best-score").textContent = pct(result.bestScore);
  el("best-blind").textContent = result.bestBlind.champion;
  el("blind-score").textContent = pct(result.bestBlind.score);
  el("result-patch").textContent = state.patch;
  el("result-candidates").textContent = result.candidates.length;
  setStatus(result.searchMethod, "success");

  el("best-pool").innerHTML = result.bestPool.map((champion) => `
    <div class="pool-item">
      ${iconHtml(champion)}
      <div><strong>${escapeHtml(champion)}</strong><span>${pct(scoreChampion(champion, result.lookups))} blind score</span></div>
    </div>`).join("");

  el("top-pools-body").innerHTML = result.topPools.map((pool, index) => {
    const delta = pool.score - result.bestScore;
    const deltaClass = delta < -1e-12 ? "delta-negative" : delta > 1e-12 ? "delta-positive" : "";
    const display = Math.abs(delta) < 0.0000005 ? "0.00%" : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)}%`;
    return `<tr><td>${index + 1}</td><td>${escapeHtml(pool.label)}</td><td class="numeric">${pct(pool.score)}</td><td class="numeric ${deltaClass}">${display}</td></tr>`;
  }).join("");

  renderExclusions();
  renderMatrix();
  renderResponsibility();
  renderDiagnosticSelect();
  renderDiagnostics();
}

function renderExclusions() {
  const data = state.result.exclusions;
  el("exclusion-section").hidden = !data.skipped.length;
  if (!data.skipped.length) return;
  const selfCount = data.skipped.filter((row) => row.reason === "self matchup").length;
  const missingCount = data.skipped.filter((row) => row.reason === "missing matchup value").length;
  el("exclusion-details").innerHTML = `
    <div class="exclusion-summary">
      <div><span>Total skipped rows</span><strong>${data.skipped.length}</strong></div>
      <div><span>Self matchups</span><strong>${selfCount}</strong></div>
      <div><span>Missing values</span><strong>${missingCount}</strong></div>
      <div><span>Missing frequencies</span><strong>0</strong></div>
      <div><span>Frequency mass removed</span><strong>${pct(data.removedMass)}</strong></div>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>Pool champion</th><th>Enemy champion</th><th>Reason</th><th>Original frequency</th></tr></thead>
      <tbody>${data.skipped.map((row) => `<tr><td>${escapeHtml(row.pool)}</td><td>${escapeHtml(row.enemy)}</td><td>${escapeHtml(row.reason)}</td><td>${pct(row.frequency)}</td></tr>`).join("")}</tbody>
    </table></div>`;
}

function matrixTemplateColumns() {
  return `190px repeat(${state.result.bestPool.length}, 132px)`;
}

function renderMatrix() {
  const result = state.result;
  const columns = matrixTemplateColumns();
  const header = `
    <div class="matrix-header" style="grid-template-columns:${columns}">
      <div class="matrix-header-cell matrix-corner">Enemy champion</div>
      ${result.bestPool.map((champion) => `<div class="matrix-header-cell">${iconHtml(champion)}<span>${escapeHtml(champion)}</span></div>`).join("")}
    </div>`;
  const body = result.counterRows.map((row) => `
    <div class="matrix-row" style="grid-template-columns:${columns}">
      <div class="enemy-label">${iconHtml(row.enemy)}<span>${escapeHtml(row.enemy)}</span><small>${pct(row.frequency)}</small></div>
      ${row.values.map((item) => {
        const best = row.best.has(item.champion);
        const title = `${row.enemy} vs ${item.champion}: ${pct(item.value)}${best ? ", best option" : ""}`;
        return `<div class="matchup-cell${best ? " best" : ""}${item.value == null ? " missing" : ""}" style="background:${colorForWinrate(item.value)}" title="${escapeHtml(title)}">${item.value == null ? "—" : pct(item.value, 1)}</div>`;
      }).join("")}
    </div>`).join("");
  el("counterpick-matrix").innerHTML = `<div class="matrix-horizontal-scroll">${header}<div class="matrix-body">${body}</div></div>`;
}

function renderResponsibility() {
  el("responsibility-chart").innerHTML = state.result.responsibility.map((row) => `
    <div class="bar-row">
      <div class="bar-label">${iconHtml(row.champion)}<span>${escapeHtml(row.champion)}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${row.share * 100}%"></div></div>
      <div class="bar-value">${pct(row.share, 1)}</div>
    </div>`).join("");
}

function renderDiagnosticSelect() {
  const select = el("diagnostic-champion");
  const existing = select.value;
  select.innerHTML = state.result.candidates
    .map((champion) => `<option${champion === state.result.bestBlind.champion ? " selected" : ""}>${escapeHtml(champion)}</option>`)
    .join("");
  if (state.result.candidates.includes(existing)) select.value = existing;
}

function diagnosticData(champion) {
  const record = championRecord(champion);
  const blind = state.result.blindScores.find((row) => row.champion === champion)?.score ?? null;
  const contributions = state.result.lookups.frequencies.map((enemy) => {
    const winrate = champion === enemy.enemy ? null : state.result.lookups.matchup.get(`${champion}\u0000${enemy.enemy}`) ?? null;
    return {
      enemy: enemy.enemy,
      frequency: enemy.frequency,
      winrate,
      contribution: winrate == null ? null : enemy.frequency * winrate,
      games: state.result.lookups.matchupGames.get(`${champion}\u0000${enemy.enemy}`) ?? null,
    };
  }).filter((row) => row.contribution != null).sort((a, b) => b.contribution - a.contribution);
  return { record, blind, contributions };
}

function renderDiagnostics() {
  if (!state.result) return;
  const champion = el("diagnostic-champion").value || state.result.bestBlind.champion;
  const data = diagnosticData(champion);
  const metrics = [
    ["Blind score", pct(data.blind)],
    ["Patch winrate", pct(data.record.winrate)],
    ["Pickrate", pct(data.record.pickrate)],
    ["Banrate", pct(data.record.banrate)],
    ["Total games", number(data.record.total_games)],
    ["Scored matchups", data.contributions.length],
    ["Worst 10 mean", worstMean(data.contributions)],
    ["Weighted CVaR 10", weightedCvar(data.contributions)],
  ];
  el("diagnostic-metrics").innerHTML = metrics.map(([label, value]) => `
    <div class="diagnostic-metric"><span>${label}</span><strong>${value}</strong></div>`).join("");

  const top = data.contributions.slice(0, 10);
  const max = Math.max(...top.map((row) => row.contribution), 0);
  el("contribution-chart").innerHTML = top.map((row) => `
    <div class="bar-row">
      <div class="bar-label">${iconHtml(row.enemy)}<span>${escapeHtml(row.enemy)}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${max ? row.contribution / max * 100 : 0}%"></div></div>
      <div class="bar-value">${(row.contribution * 100).toFixed(2)}</div>
    </div>`).join("");
  renderProfile(champion, data.contributions);
}

function worstMean(contributions) {
  const values = contributions.map((row) => row.winrate).sort((a, b) => a - b).slice(0, 10);
  return values.length ? pct(values.reduce((sum, value) => sum + value, 0) / values.length) : "n/a";
}

function weightedCvar(contributions) {
  const rows = [...contributions].sort((a, b) => a.winrate - b.winrate);
  const target = rows.reduce((sum, row) => sum + row.frequency, 0) * 0.1;
  let mass = 0;
  let weighted = 0;
  for (const row of rows) {
    const used = Math.min(row.frequency, target - mass);
    if (used <= 0) break;
    weighted += used * row.winrate;
    mass += used;
  }
  return mass ? pct(weighted / mass) : "n/a";
}

function renderProfile(champion, contributions) {
  const advanced = el("advanced-toggle").checked;
  el("advanced-analysis").hidden = !advanced;
  if (!advanced) return;
  const averages = new Map();
  for (const enemy of state.result.lookups.frequencies) {
    const values = state.result.candidates
      .filter((candidate) => candidate !== enemy.enemy)
      .map((candidate) => state.result.lookups.matchup.get(`${candidate}\u0000${enemy.enemy}`))
      .filter((value) => value != null);
    averages.set(enemy.enemy, values.length ? enemy.frequency * values.reduce((a, b) => a + b, 0) / values.length : null);
  }
  const rows = contributions.slice(0, 20).map((row) => {
    const baseline = averages.get(row.enemy);
    const delta = baseline == null ? null : row.contribution - baseline;
    return `<tr><td>${escapeHtml(row.enemy)}</td><td class="numeric">${(row.contribution * 100).toFixed(3)}</td><td class="numeric">${baseline == null ? "n/a" : (baseline * 100).toFixed(3)}</td><td class="numeric ${delta >= 0 ? "profile-positive" : "profile-negative"}">${delta == null ? "n/a" : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(3)}`}</td></tr>`;
  }).join("");
  el("profile-chart").innerHTML = `<div class="table-wrap"><table class="profile-table"><thead><tr><th>Enemy</th><th>Champion contribution</th><th>Candidate baseline</th><th>Delta</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

async function runOptimizer() {
  const candidates = [...state.selected].sort((a, b) => a.localeCompare(b));
  const poolSize = Number(el("pool-size").value);
  if (!candidates.length) return setRunStatus("Select at least one candidate champion.", true);
  if (!Number.isInteger(poolSize) || poolSize < 1) return setRunStatus("Pool size must be a positive integer.", true);
  if (poolSize > candidates.length) return setRunStatus(`Pool size ${poolSize} is larger than the ${candidates.length} selected candidates.`, true);

  const button = el("run-optimizer");
  button.disabled = true;
  button.textContent = "Optimizing...";
  setRunStatus("");
  setStatus("Optimizing...", "busy");
  await new Promise((resolve) => setTimeout(resolve, 20));
  try {
    state.result = buildResult(candidates, poolSize);
    renderResult();
  } catch (error) {
    setRunStatus(`Optimizer could not run: ${error.message}`, true);
    setStatus("Error");
  } finally {
    button.disabled = false;
    button.textContent = "Run optimizer";
  }
}

el("patch-select").addEventListener("change", (event) => loadPatch(event.target.value));
el("include-all").addEventListener("click", () => {
  state.selected = new Set(state.data.champions.map((champion) => champion.name));
  renderPicker();
});
el("exclude-all").addEventListener("click", () => {
  state.selected.clear();
  renderPicker();
});
el("champion-search").addEventListener("input", (event) => {
  state.search = event.target.value;
  renderPicker();
});
el("selected-champions").addEventListener("click", (event) => {
  const button = event.target.closest("[data-remove]");
  if (!button) return;
  state.selected.delete(button.dataset.remove);
  renderPicker();
});
el("available-champions").addEventListener("click", (event) => {
  const button = event.target.closest("[data-add]");
  if (!button) return;
  state.selected.add(button.dataset.add);
  renderPicker();
});
el("run-optimizer").addEventListener("click", runOptimizer);
el("diagnostic-champion").addEventListener("change", renderDiagnostics);
el("advanced-toggle").addEventListener("change", renderDiagnostics);
el("exclusion-toggle").addEventListener("click", () => {
  const details = el("exclusion-details");
  const opening = details.hidden;
  details.hidden = !opening;
  el("exclusion-toggle").textContent = opening ? "Hide details" : "Show details";
  el("exclusion-toggle").setAttribute("aria-expanded", String(opening));
});

loadPatch(PATCHES.at(-1));
