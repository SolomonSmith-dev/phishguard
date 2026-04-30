// PhishGuard background service worker.
//
// Strategy:
//   1. On every committed top-frame navigation, send the URL to the local
//      PhishGuard API and cache the verdict per tab.
//   2. Surface a banner via the popup, NOT inline page injection -- the goal
//      is to alert without breaking sites that legitimately contain phishing
//      keywords (banking pages, security blogs).
//   3. Optionally pull `document.documentElement.outerHTML` for the URL+HTML
//      multi-modal path. Toggleable in popup; off by default for privacy.

const API_BASE = "http://localhost:8000";
const VERDICT_CACHE = new Map(); // tabId -> verdict
const PHISH_THRESHOLD = 0.5;

async function predict(url, html = null) {
  const body = html ? { url, html } : { url };
  const resp = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`api ${resp.status}`);
  return resp.json();
}

chrome.webNavigation.onCommitted.addListener(async (details) => {
  if (details.frameId !== 0) return; // top frame only
  const url = details.url || "";
  if (!/^https?:/i.test(url)) return;

  try {
    const verdict = await predict(url);
    VERDICT_CACHE.set(details.tabId, { url, verdict, at: Date.now() });
    const isPhish = verdict.is_phish || verdict.p_phish >= PHISH_THRESHOLD;
    chrome.action.setBadgeText({
      tabId: details.tabId,
      text: isPhish ? "!" : "ok",
    });
    chrome.action.setBadgeBackgroundColor({
      tabId: details.tabId,
      color: isPhish ? "#cc0000" : "#2e7d32",
    });
  } catch (e) {
    chrome.action.setBadgeText({ tabId: details.tabId, text: "?" });
    chrome.action.setBadgeBackgroundColor({ tabId: details.tabId, color: "#888" });
    console.warn("phishguard predict failed", e);
  }
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === "get-verdict") {
    const tabId = msg.tabId ?? sender.tab?.id;
    sendResponse(VERDICT_CACHE.get(tabId) || null);
    return true;
  }
  return false;
});

chrome.tabs.onRemoved.addListener((tabId) => VERDICT_CACHE.delete(tabId));
