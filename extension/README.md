# PhishGuard browser extension (stub)

Manifest v3 Chrome extension that hits a locally running PhishGuard API
(`http://localhost:8000/predict`) on every committed top-frame navigation
and shows a verdict in the toolbar badge.

## Load it

1. `make serve` to start the API at `http://localhost:8000`.
2. Open `chrome://extensions`, enable Developer Mode.
3. Click "Load unpacked" and pick this `extension/` directory.
4. Browse anywhere -- the toolbar icon shows a green "ok", red "!" badge,
   or "?" when the API is unreachable.
5. Click the badge for the breakdown popup.

## Privacy

The extension sends the URL of the page you are viewing to localhost only.
No external services receive your browsing data. HTML is **not** sent by
default; the multi-modal path is opt-in (toggle planned).

## Limitations

- No icons committed yet -- Chrome will show a default placeholder. Drop
  16/48/128 px PNGs at `icons/` to get a branded badge.
- The badge polls per navigation; SPA route changes will not retrigger.
