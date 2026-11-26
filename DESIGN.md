# Design contract: live demo page

File: `src/phishguard/serving/static/demo.html`, served at `GET /`.

## What it is

One decision: is this URL phishing, and should I trust the number? The page
scores one URL against the served model and puts the measured evidence next to
the score. Adjectives: precise, evidentiary, calm, candid, dense.

Direction (chosen by Solomon, 2026-11-16): evidence page, on the Tufte CSS and
Stripe Press skeleton. Score on top; the reasoning and the measured numbers
below as a reading column with a sidenote column.

## The default it refuses

A centered SaaS hero with a big input, a green check or red X, and three icon
cards of "features". No hero metrics strip, no unexplained confidence, no
hidden threshold.

## First viewport

Wordmark, one-line description, URL field with the single accent action, four
example links (including a top-50 site the model gets wrong), then the result:
`p(phish)` in Plex Mono, the verdict word at the served threshold with the
Tranco flag rate at that threshold, and a 0 to 1 scale marking this URL, the
threshold and the Tranco top-site median.

## Tokens

A3sthetic family project. House paper and ink tokens; IBM Plex Sans body;
IBM Plex Mono for every number and URL; display face Jaro (Agyei Archer, with
Hurka and Velimirovic). One UI accent: Ndebele blue (`#1F5FA8` light,
`#7FB0EA` dark). Kola red only for the "Flagged as phishing" state, always
with the word. Light and dark from tokens, system setting plus a toggle.

## Signature

- Gee's Bend quilts: the evidence block is an asymmetric 7:4 split with an
  uneven gutter, and the result is framed as a "housetop" of nested
  rectangles with deliberately uneven offsets.
- Bogolan mudcloth: section rules are uneven dot-and-dash runs at hairline
  contrast, the hand-painted border line.

Both are named in code comments at the CSS that draws them.

## Content rules

Every number on the page comes from `reports/url_onnx_2026-11-16.md` (served
path) or `LIMITATIONS.md`. If the model is retrained or re-exported, update
the numbers and `TRANCO_MEDIAN` / `FLAG_RATE` in the script block in the same
change.
