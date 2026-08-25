## Context (give Kiro this first)

This is `app.py`, a single-file Streamlit app (`Universal-Data-Analysis-Dashboard-main/`). It has already been through two UI passes: a "premium gold/glass-morphism" look (v1, in `STYLE_GUIDE.md`) and a "dark enterprise blue" look (v2, in `PROJECT_MEMORY.md`, section 4). **Do not repeat either of these.** Also avoid the three most common AI-generated dashboard looks: (1) warm cream background + serif headline + terracotta accent, (2) near-black background + single acid-green/vermilion accent, (3) broadsheet/newspaper hairline-rule layout. This is a data-analysis instrument, not a landing page or a SaaS marketing site — the design should look like it was built by people who use it every day, not people trying to sell it.

Constraints to respect:
- Pure Streamlit + custom CSS injected via `st.markdown(..., unsafe_allow_html=True)`. No new frontend framework, no Docker.
- Must keep `use_container_width=True` on all charts/tables (Streamlit 1.62 compat — do not reintroduce `use_column_width`).
- No `st.balloons()` or confetti on load.
- Six existing tabs must stay: Overview, Visualizations, Insights, Custom Analysis, Advanced Analytics, Data Profiling. Sidebar stays reserved for data-source controls only.
- Responsive down to a laptop-width viewport at minimum; visible keyboard focus states; respect `prefers-reduced-motion`.

## Design direction — "Instrument Panel"

Ground the identity in what this tool actually is: a measurement instrument for data, not a corporate dashboard. Borrow from lab/scientific instrumentation and oscilloscope readouts rather than generic "enterprise SaaS blue" or "premium gold."

**Color tokens** (define these as CSS variables at the top of the injected stylesheet):
- `--bg`: `#0B0E14` (near-black ink, slightly blue, not pure black)
- `--surface`: `#12161F` (cards, tab bar, expanders)
- `--surface-raised`: `#171C27` (hovered/active cards)
- `--border`: `#232A38`
- `--accent-signal`: `#5EEAD4` (phosphor cyan-teal — primary accent: active tab, primary chart series, focus rings)
- `--accent-warn`: `#FDBA74` (warm amber — used sparingly for anomalies/outliers/warnings only, never decoratively)
- `--text-primary`: `#F1F5F9`
- `--text-muted`: `#8B95A7`
- Keep Streamlit's default green/red for success/error alerts.

**Typography:**
- Display/headings: a clean geometric grotesque (e.g. `Space Grotesk` or `General Sans`), weight 600–700, tight letter-spacing.
- Body: `Inter` or system-ui, weight 400–500.
- All numeric values (metric cards, stat tables, axis labels, KPI figures): a monospace face (`IBM Plex Mono` or `JetBrains Mono`). This is the key differentiator from v1/v2 — numbers should read like instrument readouts, not marketing stats. Load fonts via `st.markdown` `<link>` to Google Fonts or bundle locally if offline use matters.

**Signature element:** a thin, full-width "live readout" strip pinned under the header — a single-row sparkline/waveform rendered in `--accent-signal` showing a live pulse of the loaded dataset (e.g. row count ticking, a rolling mean of a numeric column, or a simple "data heartbeat" animation while a file is processing). This is the one place motion and boldness are allowed; everything else stays quiet and disciplined. If a true live signal isn't practical, a static waveform-style divider using the same visual language is an acceptable fallback — but the motion version is preferred.

**Layout:**
- Header: small monospace "STATUS" line (e.g. `DATASET LOADED · 4,213 ROWS · 12 COLS · LAST SYNC 0.4s`) above the main title, styled like a instrument status bar, not a page title.
- Metric row: 3–4 compact cards, flat `--surface` background, 1px `--border`, no gradients or glow — value in monospace `--accent-signal`, label in uppercase `--text-muted` at 11px with wide letter-spacing.
- Tabs: underline-style (not pill/rounded), active tab gets a 2px `--accent-signal` underline and brighter text — closer to an oscilloscope channel selector than a SaaS nav.
- Charts: transparent background, gridlines in `--border` at low opacity, primary series in `--accent-signal`, secondary series in a muted violet (`#A5B4FC`) for contrast, anomalies/outliers highlighted in `--accent-warn`.
- Welcome/empty screen: no hero image or feature-card grid. Instead show the same "instrument idle" status bar (e.g. `NO DATASET LOADED · AWAITING INPUT`) plus a short, plain-language explanation of what to drop in the sidebar. Treat emptiness as an instruction, not decoration.

## What to change in `app.py`

1. Replace the entire injected CSS block (the one currently defining `--surface: #1e293b` / blue accent) with the token system above.
2. Update all Plotly `layout` calls (histogram, scatter, box, heatmap, line, area, 3D scatter) to use the new chart color tokens and transparent backgrounds.
3. Rebuild the metric cards and tab styling per the layout spec above.
4. Rebuild the welcome/empty-state screen — remove the 2-column feature-card grid from v2, replace with the idle-instrument treatment.
5. Add the live-readout strip component under the header.
6. Keep all functional logic (data loading, caching, ML analytics, MySQL connection) completely untouched — this is a visual/CSS/layout pass only, not a logic refactor.
7. After implementing, take a screenshot (or describe what you'd expect to see) for Overview, Visualizations, and the empty/welcome state, and self-check against the constraints above before calling it done.

Update `STYLE_GUIDE.md` and the "UI/UX" section of `PROJECT_MEMORY.md` to reflect this new v3 design system once implemented.