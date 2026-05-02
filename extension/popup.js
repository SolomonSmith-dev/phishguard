async function activeTabId() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab?.id;
}

function fmtPct(p) {
  return `${(p * 100).toFixed(1)}%`;
}

function makeRow(name, valueText) {
  const row = document.createElement("div");
  row.className = "row";
  const left = document.createElement("span");
  left.textContent = name;
  const right = document.createElement("span");
  right.textContent = valueText;
  row.append(left, right);
  return row;
}

(async () => {
  const tabId = await activeTabId();
  const cached = await chrome.runtime.sendMessage({ type: "get-verdict", tabId });
  const urlEl = document.getElementById("url");
  const verdictEl = document.getElementById("verdict");
  const modEl = document.getElementById("modalities");

  if (!cached) {
    urlEl.textContent = "(no verdict yet)";
    verdictEl.textContent = "-";
    return;
  }

  urlEl.textContent = cached.url;
  const v = cached.verdict;
  const isPhish = v.is_phish || v.p_phish >= (v.threshold ?? 0.5);
  verdictEl.textContent = isPhish
    ? `Phishing likely (${fmtPct(v.p_phish)})`
    : `Looks safe (${fmtPct(v.p_phish)})`;
  verdictEl.className = "verdict " + (isPhish ? "phish" : "ok");

  for (const [name, m] of Object.entries(v.modalities || {})) {
    if (!m.available) continue;
    const lat = typeof m.latency_ms === "number" ? `${m.latency_ms.toFixed(0)}ms` : "-";
    modEl.append(makeRow(String(name), `${fmtPct(m.p)} - ${lat}`));
  }
})();
