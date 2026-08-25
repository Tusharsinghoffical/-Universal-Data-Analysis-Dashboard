# Style Guide — Universal Data Analysis Dashboard
## v3 · Instrument Panel · 2026-08-25

> This document is the authoritative style reference. v1 (gold/glass-morphism) and v2 (dark enterprise blue) are superseded. Do not reintroduce patterns from either.

---

## Design Philosophy

The dashboard is a **measurement instrument for data**, not a SaaS marketing page or corporate portal. Every visual decision should make the data easier to read, not harder. Borrow from lab instrumentation and oscilloscope readouts. Keep motion and boldness reserved for signal — everything else stays quiet and disciplined.

**Avoid:**
- Warm cream + serif headline + terracotta accent (AI cliché #1)
- Near-black + single acid-green/vermilion accent (AI cliché #2)
- Broadsheet/newspaper hairline-rule layouts (AI cliché #3)
- Gradients, glow effects, glass-morphism, shimmer animations
- `st.balloons()` or any celebratory animation on data load

---

## Color Tokens

Defined as CSS custom properties at the top of the injected stylesheet:

```css
:root {
    --bg:               #0B0E14;   /* page background — near-black ink, slightly blue */
    --surface:          #12161F;   /* cards, tab bar, expanders, sidebar */
    --surface-raised:   #171C27;   /* hovered/focused surfaces */
    --border:           #232A38;   /* all component borders, chart gridlines */
    --accent-signal:    #5EEAD4;   /* PRIMARY accent — phosphor cyan-teal */
    --accent-warn:      #FDBA74;   /* warm amber — anomalies, warnings, idle state ONLY */
    --accent-secondary: #A5B4FC;   /* muted violet — secondary chart series */
    --text-primary:     #F1F5F9;   /* headings, active labels */
    --text-muted:       #8B95A7;   /* body text, axis labels, metadata */
}
```

### Usage Rules
| Token | Use | Never use for |
|-------|-----|---------------|
| `--accent-signal` | Active tab underline, metric values, primary chart series, focus rings, column tags, button border/hover fill, status dot (loaded) | Decorative backgrounds, large fills |
| `--accent-warn` | Anomaly/outlier points, idle status dot, warning messages | General decoration, primary actions |
| `--accent-secondary` | Secondary chart series, moving averages | Backgrounds, text |
| `--border` | All 1px borders, chart gridlines | Text |

---

## Typography

| Role | Font | Weight | Size | Notes |
|------|------|--------|------|-------|
| Display / headings | Space Grotesk | 600–700 | 1.75rem | Loaded via Google Fonts |
| Body / labels | Inter | 400–500 | 0.875rem | Fallback: system-ui |
| **Numbers / readouts** | **IBM Plex Mono** | **400–600** | varies | **All numeric values, KPIs, axis labels, status bar, metric cards, buttons** |
| Status / meta | IBM Plex Mono | 400–500 | 0.68–0.72rem | Uppercase, letter-spacing 0.1–0.14em |

Fonts loaded via `st.markdown` `<link>` tag (Google Fonts CDN). Monospace for numbers is the key v3 differentiator — values must read like instrument readouts.

---

## Components

### Status Bar
- Single line above the main title
- IBM Plex Mono, 0.72rem, uppercase, `--text-muted`
- Blinking 6px dot: `--accent-signal` when dataset loaded, `--accent-warn` when idle
- Format: `DATASET LOADED · 4,213 ROWS · 12 COLS · 4 NUMERIC`
- Dot blink animation respects `prefers-reduced-motion`

### Live Readout Strip
- Full-width `--surface` bar, 36px tall, pinned under the title
- `--border` top and bottom, no border-radius
- Left: key stats (ROWS / COLS / NUMERIC) in IBM Plex Mono, value in `--accent-signal`
- Centre: SVG `<polyline>` waveform drawn in `--accent-signal`, `stroke-dasharray` animation on load
- Right: `SYS READY` in `--accent-signal`
- Animation respects `prefers-reduced-motion`

### Metric Cards
- `--surface` background, 1px `--border`, 4px border-radius (sharp, not rounded)
- Hover: `border-color` transitions to `--accent-signal` only — no glow, no lift
- Label: top, IBM Plex Mono 0.68rem, uppercase, `--text-muted`, letter-spacing 0.12em
- Value: bottom, IBM Plex Mono 2rem, weight 600, `--accent-signal`
- Height: 110px, flex column with `justify-content: space-between`

### Tabs
- Underline style — NO pill, NO rounded background, NO filled active tab
- Tab list: transparent background, bottom `1px --border` only
- Inactive: `--text-muted`, transparent background
- Hover: `--text-primary`, `--border` bottom underline
- **Active: `--text-primary` weight 600, `2px --accent-signal` bottom underline**
- Keyboard focus: `outline: 2px solid --accent-signal`

### Buttons
- Transparent background, `1px --accent-signal` border, `--accent-signal` text
- IBM Plex Mono font, uppercase, 0.8rem, letter-spacing 0.06em
- Hover: fill with `--accent-signal`, text becomes `--bg`
- No box-shadow, no gradient, no shimmer

### Column Type Tags
- Transparent background, `1px --accent-signal` border, `--accent-signal` text
- IBM Plex Mono 0.75rem, inline-block, 2px border-radius
- Used inside expanders for auto-detected column lists

### Expanders
- `--surface` background, `1px --border`, 3px border-radius
- Header: `--text-muted`, hover: `--surface-raised` + `--text-primary`
- Content: `--surface` background

### Form Inputs (text, select, textarea)
- `--surface` background, `1px --border`, 4px border-radius
- Focus: `border-color --accent-signal`, `box-shadow: 0 0 0 2px rgba(94,234,212,0.15)`

---

## Charts (Plotly)

All charts use `fig.update_layout()` with these values:

```python
fig.update_layout(
    plot_bgcolor='rgba(11,14,20,0)',
    paper_bgcolor='rgba(11,14,20,0)',
    font=dict(color='#8B95A7', family='IBM Plex Mono, monospace', size=11),
    xaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='#232A38', gridwidth=1, showgrid=True, zeroline=False),
)
```

| Series role | Color |
|-------------|-------|
| Primary series | `#5EEAD4` (`--accent-signal`) |
| Secondary series | `#A5B4FC` (`--accent-secondary`) |
| Anomalies / outliers | `#FDBA74` (`--accent-warn`) |
| Normal points (anomaly chart) | `#5EEAD4` |
| Color scale (continuous) | `['#12161F', '#5EEAD4']` |

---

## Empty / Idle State

When no dataset is loaded:
- **No hero image**, no feature-card grid, no `st.info()` box
- Full-width `--surface` panel with `1px --border`, 4px radius
- Amber `--accent-warn` status line: `NO DATASET LOADED · AWAITING INPUT`
- Plain `idle-title` (Space Grotesk 1.4rem) + short body paragraph
- Format tags row: monospace uppercase pills in `--border` style
- Static SVG waveform at bottom at 25% opacity in `--accent-signal` — visual language consistency with the loaded readout strip

---

## Layout

- Max content width: 1280px
- Page background: `--bg` (#0B0E14)
- Sidebar: `--surface` background, `1px --border` right edge
- Header zone: status bar → title → readout strip → metric row → expander → tab panel
- `use_container_width=True` on all charts and tables (Streamlit 1.62 compat)

---

## Motion

- Status dot blink: `opacity` keyframe, 2s ease-in-out, infinite
- Waveform draw: `stroke-dashoffset` from 800→0, 3s ease forwards
- All animations wrapped in `@media (prefers-reduced-motion: reduce)` override that disables them
- No other animations anywhere in the UI
