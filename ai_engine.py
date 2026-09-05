import json
import re
import os
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config import AI_CONFIG

# Instrument Panel Plotly Theme Constants
INSTRUMENT_THEME = {
    'bg': 'rgba(11,14,20,0)',
    'paper': 'rgba(11,14,20,0)',
    'font_family': 'IBM Plex Mono, Consolas, monospace',
    'font_color': '#8B95A7',
    'grid_color': '#232A38',
    'accent_signal': '#5EEAD4',
    'accent_warn': '#FDBA74',
    'accent_secondary': '#A5B4FC'
}

def style_plotly_fig(fig, title=None):
    """Applies the v3 Instrument Panel aesthetic to any Plotly figure"""
    fig.update_layout(
        title=dict(
            text=title or fig.layout.title.text or "",
            font=dict(family='Space Grotesk, sans-serif', size=14, color='#F1F5F9')
        ),
        plot_bgcolor=INSTRUMENT_THEME['bg'],
        paper_bgcolor=INSTRUMENT_THEME['paper'],
        font=dict(
            color=INSTRUMENT_THEME['font_color'],
            family=INSTRUMENT_THEME['font_family'],
            size=11
        ),
        xaxis=dict(
            gridcolor=INSTRUMENT_THEME['grid_color'],
            gridwidth=1,
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor=INSTRUMENT_THEME['grid_color'],
            gridwidth=1,
            showgrid=True,
            zeroline=False
        ),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def sanitize_ai_markdown(text):
    """
    Prevents Streamlit KaTeX math mode corruption from currency dollar signs (e.g. $79,859.94)
    and ensures complete, uncorrupted markdown formatting.
    """
    if not text:
        return ""
    # Streamlit parses unescaped $...$ as KaTeX mathematical formula which breaks currency amounts.
    # Escaping dollar signs with \$ renders clean currency symbols without triggering math mode.
    return re.sub(r'(?<!\\)\$', r'\\$', text)


# ── Gemini API Integration ───────────────────────────────────────────────────

def call_gemini_api(prompt, system_instruction=None, api_key=None, model=None, temperature=None, max_tokens=None):
    """
    Direct, robust call to Google Gemini REST API.
    Supports gemini-3.7-flash, gemini-flash-lite-latest, gemini-3.5-flash-lite with automatic fallback across models.
    """
    key = api_key or os.getenv('GEMINI_API_KEY') or AI_CONFIG.get('api_key', '')
    if not key or not key.strip():
        return None, "NO_KEY"

    model_name = model or AI_CONFIG.get('model', 'gemini-3.7-flash')
    temp = temperature if temperature is not None else AI_CONFIG.get('temperature', 0.2)
    tokens = max_tokens or AI_CONFIG.get('max_output_tokens', 8192)

    models_to_try = [model_name]
    for alt in ['gemini-3.7-flash', 'gemini-flash-lite-latest', 'gemini-3.5-flash-lite', 'gemini-3.1-flash-lite', 'gemini-3.6-flash']:
        if alt not in models_to_try:
            models_to_try.append(alt)

    headers = {'Content-Type': 'application/json'}
    last_error = "Unknown error"

    for mod in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key.strip()}"
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": tokens
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('candidates', [])
                if candidates:
                    parts = candidates[0].get('content', {}).get('parts', [])
                    if parts:
                        text = parts[0].get('text', '')
                        return text, None
                return None, "Empty response from Gemini model"
            elif resp.status_code == 429:
                last_error = f"Model {mod} rate-limited or quota exceeded (429), switching to next model..."
                continue
            elif resp.status_code == 404:
                last_error = f"Model {mod} not found (404), trying fallback..."
                continue
            elif resp.status_code in [400, 403]:
                err_json = resp.json().get('error', {})
                err_msg = err_json.get('message', resp.text)
                if "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
                    return None, f"Gemini API Error ({resp.status_code}): {err_msg}"
                last_error = f"Gemini API Error ({resp.status_code}): {err_msg}"
                continue
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:150]}"
        except requests.exceptions.Timeout:
            last_error = f"Request to Gemini API ({mod}) timed out (25s)"
            continue
        except Exception as e:
            last_error = f"Network or parsing error: {str(e)}"
            continue

    return None, last_error


def test_gemini_connection(api_key, model='gemini-3.7-flash'):
    """Tests if a Gemini API key is active and functional."""
    prompt = "Ping test. Respond with only the word: ONLINE"
    res, err = call_gemini_api(prompt, api_key=api_key, model=model, max_tokens=150)
    if res and "ONLINE" in res.upper():
        return True, f"Online · Gemini connected ({model})"
    if res:
        return True, f"Online · {res.strip()}"
    return False, err


# ── Dataset Digest Generator ─────────────────────────────────────────────────

def build_dataset_digest(df):
    """
    Builds a high-density, context-efficient summary of the dataset
    including schema, quantiles, missing rates, categorical distributions, and correlations.
    Automatically samples large datasets (>50k rows) for instantaneous execution.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Use fast representative sample for statistical profiling on large datasets (e.g. 2.7M rows)
    calc_df = df.sample(min(25000, total_rows), random_state=42) if total_rows > 50000 else df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns.tolist()
    
    # Missing values
    null_counts = df.isnull().sum()
    null_summary = {}
    for col, count in null_counts.items():
        if count > 0:
            null_summary[col] = f"{count:,} ({count/total_rows*100:.1f}%)"

    # Numeric summary
    num_stats = {}
    for col in numeric_cols[:15]:  # top 15 numeric
        series = calc_df[col].dropna()
        if len(series) > 0:
            num_stats[col] = {
                'mean': round(float(series.mean()), 2),
                'median': round(float(series.median()), 2),
                'std': round(float(series.std()), 2) if len(series) > 1 else 0.0,
                'min': round(float(series.min()), 2),
                'max': round(float(series.max()), 2),
                'skewness': round(float(series.skew()), 2) if len(series) > 2 else 0.0
            }

    # Categorical summary
    cat_stats = {}
    for col in cat_cols[:10]:  # top 10 categorical
        vc = calc_df[col].value_counts()
        top_vals = vc.head(4).to_dict()
        cat_stats[col] = {
            'unique_count': int(calc_df[col].nunique()),
            'top_frequencies': {str(k): int(v) for k, v in top_vals.items()}
        }

    # Key correlations
    top_corrs = []
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = calc_df[numeric_cols[:12]].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    if not np.isnan(val) and abs(val) >= 0.35:
                        top_corrs.append({
                            'pair': f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}",
                            'r': round(float(val), 3)
                        })
            top_corrs = sorted(top_corrs, key=lambda x: abs(x['r']), reverse=True)[:8]
        except Exception:
            pass

    # Sample rows
    sample_preview = df.head(3).to_dict(orient='records')
    # Clean datetime/nan in sample preview
    clean_sample = []
    for row in sample_preview:
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v):
                clean_row[k] = None
            elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
                clean_row[k] = str(v)
            else:
                clean_row[k] = v
        clean_sample.append(clean_row)

    digest = {
        'total_rows': total_rows,
        'total_columns': total_cols,
        'column_types': {
            'numeric': numeric_cols,
            'categorical': cat_cols,
            'datetime': [str(c) for c in date_cols]
        },
        'missing_values': null_summary,
        'numeric_metrics': num_stats,
        'categorical_breakdown': cat_stats,
        'significant_correlations': top_corrs,
        'sample_rows': clean_sample
    }
    return digest


def dataset_digest_to_markdown(digest):
    """Formats the dataset digest into a prompt-friendly markdown block."""
    md = [
        f"### Dataset Schema & Profile",
        f"- **Records**: {digest['total_rows']:,} rows | **Columns**: {digest['total_columns']}",
        f"- **Numeric Columns ({len(digest['column_types']['numeric'])})**: {', '.join(digest['column_types']['numeric']) or 'None'}",
        f"- **Categorical Columns ({len(digest['column_types']['categorical'])})**: {', '.join(digest['column_types']['categorical']) or 'None'}",
        f"- **Date Columns**: {', '.join(digest['column_types']['datetime']) or 'None'}",
        "",
        "#### Numeric Distributions & Stats:"
    ]
    for col, s in digest['numeric_metrics'].items():
        md.append(f"  - **{col}**: mean={s['mean']}, median={s['median']}, std={s['std']}, min={s['min']}, max={s['max']}, skew={s['skewness']}")

    if digest['categorical_breakdown']:
        md.append("\n#### Categorical Breakdown:")
        for col, c in digest['categorical_breakdown'].items():
            top_str = ", ".join([f"'{k}': {v}" for k, v in c['top_frequencies'].items()])
            md.append(f"  - **{col}** ({c['unique_count']} unique): {top_str}")

    if digest['significant_correlations']:
        md.append("\n#### Notable Correlations:")
        for corr in digest['significant_correlations']:
            md.append(f"  - {corr['pair']}: r = {corr['r']}")

    if digest['missing_values']:
        md.append(f"\n#### Missing Data: {', '.join([f'{k}: {v}' for k, v in digest['missing_values'].items()])}")
    else:
        md.append("\n#### Missing Data: 0% missing values (Complete dataset)")

    return "\n".join(md)


# ── Intelligent Heuristic Analytics Engine (Offline Fallback) ────────────────

def smart_heuristic_qa(df, question):
    """
    Intelligent heuristic fallback when no LLM API key is present or when offline.
    Parses intent (aggregations, rankings, comparisons, stats, counts) and computes exact answers.
    """
    q = question.lower().strip()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Row count / shape questions
    if any(phrase in q for phrase in ['how many rows', 'how many records', 'dataset size', 'total rows', 'number of employees', 'how big']):
        return (
            f"**Dataset Dimensions:**\n\n"
            f"- **Total Records**: {len(df):,}\n"
            f"- **Total Columns**: {len(df.columns)}\n"
            f"- **Numeric Columns**: {len(num_cols)} ({', '.join(num_cols)})\n"
            f"- **Categorical Columns**: {len(cat_cols)} ({', '.join(cat_cols)})"
        )

    # 2. Missing data / Quality
    if any(phrase in q for phrase in ['missing', 'null', 'nan', 'data quality', 'empty values']):
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if len(nulls) == 0:
            return "✅ **No Missing Values**: The dataset is 100% complete across all columns with zero null/NaN records."
        else:
            lines = ["⚠️ **Missing Values Summary:**\n"]
            for col, cnt in nulls.items():
                pct = (cnt / len(df)) * 100
                lines.append(f"- **{col}**: `{cnt:,}` missing (`{pct:.1f}%`)")
            return "\n".join(lines)

    # 3. Correlation / Relationship
    if any(phrase in q for phrase in ['correlation', 'relationship', 'related', 'correlate', 'covariate']):
        if len(num_cols) < 2:
            return "Correlation analysis requires at least two numeric columns."
        corr = df[num_cols].corr()
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                val = corr.iloc[i, j]
                if not np.isnan(val):
                    pairs.append((num_cols[i], num_cols[j], val))
        pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
        lines = ["**Correlation Analysis Findings:**\n"]
        for c1, c2, r in pairs[:5]:
            strength = "Very Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.4 else "Weak"
            direction = "Positive" if r > 0 else "Negative"
            lines.append(f"- **{c1}** ↔ **{c2}**: `r = {r:.3f}` ({strength} {direction})")
        return "\n".join(lines)

    # 4. Find matched column in question
    target_num = None
    for col in num_cols:
        if col.lower() in q:
            target_num = col
            break
            
    target_cat = None
    for col in cat_cols:
        if col.lower() in q:
            target_cat = col
            break

    # If asking for average/mean
    if any(word in q for word in ['average', 'mean', 'typical']):
        if target_num and target_cat:
            grp = df.groupby(target_cat)[target_num].mean().sort_values(ascending=False).reset_index()
            top = grp.iloc[0]
            bot = grp.iloc[-1]
            return (
                f"**Average {target_num} by {target_cat}:**\n\n"
                f"- **Highest**: **{top[target_cat]}** with an average of `{top[target_num]:,.2f}`\n"
                f"- **Lowest**: **{bot[target_cat]}** with an average of `{bot[target_num]:,.2f}`\n"
                f"- **Overall Dataset Average**: `{df[target_num].mean():,.2f}`\n\n"
                f"```text\n{grp.to_string(index=False)}\n```"
            )
        elif target_num:
            return (
                f"**Summary for {target_num}:**\n\n"
                f"- **Mean (Average)**: `{df[target_num].mean():,.2f}`\n"
                f"- **Median**: `{df[target_num].median():,.2f}`\n"
                f"- **Standard Deviation**: `{df[target_num].std():,.2f}`\n"
                f"- **Min / Max**: `{df[target_num].min():,.2f}` to `{df[target_num].max():,.2f}`"
            )

    # If asking for highest / top / max
    if any(word in q for word in ['highest', 'top', 'maximum', 'max', 'most', 'best']):
        if target_num and target_cat:
            grp = df.groupby(target_cat)[target_num].mean().sort_values(ascending=False).reset_index()
            top = grp.iloc[0]
            return (
                f"🏆 **Top {target_cat} by {target_num}:**\n\n"
                f"The highest average **{target_num}** is in **{top[target_cat]}** "
                f"at `{top[target_num]:,.2f}` (vs. overall mean `{df[target_num].mean():,.2f}`)."
            )
        elif target_num:
            top_records = df.sort_values(by=target_num, ascending=False).head(3)
            return (
                f"**Maximum {target_num} Recorded:** `{df[target_num].max():,.2f}`\n\n"
                f"Top records:\n```text\n{top_records[[c for c in [target_cat, target_num] if c]].to_string(index=False)}\n```"
            )
        elif target_cat:
            vc = df[target_cat].value_counts()
            return f"**Most Frequent {target_cat}:** **{vc.index[0]}** with `{vc.iloc[0]:,}` occurrences ({vc.iloc[0]/len(df)*100:.1f}%)."

    # If asking for lowest / minimum / worst
    if any(word in q for word in ['lowest', 'bottom', 'minimum', 'min', 'least', 'worst']):
        if target_num and target_cat:
            grp = df.groupby(target_cat)[target_num].mean().sort_values().reset_index()
            bot = grp.iloc[0]
            return (
                f"📉 **Lowest {target_cat} by {target_num}:**\n\n"
                f"The lowest average **{target_num}** is in **{bot[target_cat]}** "
                f"at `{bot[target_num]:,.2f}` (vs. overall mean `{df[target_num].mean():,.2f}`)."
            )
        elif target_num:
            return f"**Minimum {target_num} Recorded:** `{df[target_num].min():,.2f}`."

    # If asking about departments or categories
    if target_cat:
        vc = df[target_cat].value_counts().reset_index()
        vc.columns = [target_cat, 'Count']
        vc['Percentage'] = (vc['Count'] / len(df) * 100).round(1).astype(str) + '%'
        return (
            f"**Distribution of {target_cat} ({df[target_cat].nunique()} distinct values):**\n\n"
            f"```text\n{vc.head(10).to_string(index=False)}\n```"
        )

    # General overview fallback
    summary_parts = [
        f"**Automated Analysis for:** *\"{question}\"*",
        f"- Dataset has **{len(df):,} records** across **{len(df.columns)} columns**."
    ]
    if num_cols:
        summary_parts.append(f"- Numeric indicators: {', '.join(num_cols[:4])}")
    if cat_cols:
        summary_parts.append(f"- Categorical dimensions: {', '.join(cat_cols[:4])}")
    summary_parts.append(
        "\n💡 *Tip: For open-ended natural language synthesis, add your Google Gemini API key in the sidebar.*"
    )
    return "\n".join(summary_parts)


# ── Chat with your Data ──────────────────────────────────────────────────────

def ask_dataset_ai(df, question, chat_history=None, api_key=None, model=None):
    """
    Primary interface for conversational data Q&A.
    Uses Gemini API when key is available, falls back to Smart Heuristic Engine.
    """
    key = api_key or os.getenv('GEMINI_API_KEY') or AI_CONFIG.get('api_key', '')
    
    if key and key.strip():
        digest = build_dataset_digest(df)
        digest_md = dataset_digest_to_markdown(digest)
        
        # Dynamically compute exact stats if specific columns are mentioned
        q_lower = question.lower()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        matched_nums = [c for c in num_cols if c.lower() in q_lower]
        matched_cats = [c for c in cat_cols if c.lower() in q_lower]

        computed_insights = []
        if matched_cats and matched_nums:
            c_col = matched_cats[0]
            n_col = matched_nums[0]
            try:
                grp = df.groupby(c_col)[n_col].agg(['mean', 'median', 'min', 'max', 'count']).round(2).reset_index()
                computed_insights.append(f"#### Computed Group Aggregation: `{n_col}` by `{c_col}`:\n```text\n{grp.to_string(index=False)}\n```")
            except Exception:
                pass
        elif matched_nums:
            for n_col in matched_nums[:2]:
                try:
                    desc = df[n_col].describe().round(2).to_dict()
                    desc_str = ", ".join([f"{k}={v}" for k, v in desc.items()])
                    computed_insights.append(f"#### Computed Statistics for `{n_col}`: {desc_str}")
                except Exception:
                    pass
        elif matched_cats:
            for c_col in matched_cats[:2]:
                try:
                    vc = df[c_col].value_counts().head(10).to_dict()
                    vc_str = ", ".join([f"'{k}': {v}" for k, v in vc.items()])
                    computed_insights.append(f"#### Computed Frequencies for `{c_col}`: {vc_str}")
                except Exception:
                    pass

        insights_block = "\n\n" + "\n\n".join(computed_insights) if computed_insights else ""

        system_instruction = (
            "You are a Principal Data Analyst and Business Intelligence Expert operating an enterprise data analysis dashboard. "
            "You are provided with a structured summary, statistics digest, and precomputed mathematical aggregations of the user's active tabular dataset. "
            "Answer the user's question directly, accurately, and concisely. "
            "Cite exact numbers, percentages, rankings, and column names from the provided digest and computed tables. "
            "Use clean Markdown formatting, bullet points, and small markdown tables where appropriate. "
            "Never invent facts not supported by the dataset. Keep answers actionable and professional."
        )
        
        prompt = f"""{digest_md}{insights_block}

---
User Question: {question}

Please provide a clear, factual, data-driven answer based directly on the dataset profile and computed insights above.
"""
        response_text, err = call_gemini_api(
            prompt,
            system_instruction=system_instruction,
            api_key=key,
            model=model,
            max_tokens=4096
        )
        if response_text:
            return sanitize_ai_markdown(response_text), "gemini"
        # If API failed, fallback to heuristic with note
        heuristic_res = smart_heuristic_qa(df, question)
        return sanitize_ai_markdown(f"{heuristic_res}\n\n*(Note: Gemini API returned '{err}'. Served via Intelligent Heuristic Engine)*"), "heuristic"
    else:
        return sanitize_ai_markdown(smart_heuristic_qa(df, question)), "heuristic"


# ── Executive Briefing Generator ─────────────────────────────────────────────

def generate_executive_summary(df, api_key=None, model=None):
    """
    Generates a structured Executive Briefing (Synopsis, Patterns, Risks, Recommendations).
    """
    digest = build_dataset_digest(df)
    key = api_key or os.getenv('GEMINI_API_KEY') or AI_CONFIG.get('api_key', '')

    if key and key.strip():
        digest_md = dataset_digest_to_markdown(digest)
        system_instruction = (
            "You are an executive data advisor creating a formal C-suite data briefing. "
            "Write in a crisp, high-signal, executive tone. "
            "Structure your response strictly into 4 sections with Markdown headings: "
            "1. 📊 Executive Synopsis (Overview & scope) "
            "2. 🔍 Key Discoveries & Dominant Trends (Quantified findings) "
            "3. ⚠️ Anomalies & Operational Risk Factors (Quality, skew, or outlier flags) "
            "4. 🎯 Strategic Actionable Recommendations (Concrete next steps) "
            "Use bullet points and bold key numbers. Complete every section and never cut off in the middle."
        )
        prompt = f"""Dataset Profile:
{digest_md}

Generate the comprehensive Executive Briefing for this dataset.
"""
        res, err = call_gemini_api(prompt, system_instruction=system_instruction, api_key=key, model=model, max_tokens=8192)
        if res:
            return sanitize_ai_markdown(res), "gemini"

    # Heuristic Executive Briefing
    rows = len(df)
    cols = len(df.columns)
    calc_df = df.sample(min(25000, rows), random_state=42) if rows > 50000 else df
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Analyze nulls
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    
    # Analyze correlations
    top_corrs = digest.get('significant_correlations', [])
    corr_bullets = []
    for c in top_corrs[:3]:
        corr_bullets.append(f"- **{c['pair']}**: Notable correlation coefficient of `r = {c['r']}`.")
    if not corr_bullets:
        corr_bullets.append("- Variables exhibit low linear collinearity, indicating diverse independent dimensions.")

    # High variance / skew (sampled for speed)
    skews = []
    for col in num_cols[:10]:
        sk = float(calc_df[col].skew()) if len(calc_df[col].dropna()) > 2 else 0.0
        if abs(sk) > 1.0:
            direction = "right (positive)" if sk > 0 else "left (negative)"
            skews.append(f"Column `{col}` displays strong {direction} skewness ({sk:.2f}).")

    mem_mb = df.memory_usage(index=True, deep=False).sum() / (1024 * 1024)

    report = f"""### 📊 Executive Synopsis
- **Data Perimeter**: The active instrument is monitoring **{rows:,} records** across **{cols} parameters** ({len(num_cols)} numerical, {len(cat_cols)} categorical).
- **Dataset Health**: Memory consumption is approximately `{mem_mb:.2f} MB`. Data integrity is {'optimal with zero missing entries' if len(null_cols) == 0 else f'compromised by {len(null_cols)} columns with missing entries'}.
- **Analytical Scope**: Suitable for enterprise descriptive diagnostics, KPI benchmarking, and predictive driver attribution.

### 🔍 Key Discoveries & Dominant Trends
- **Primary Dimensional Drivers**:
{chr(10).join(corr_bullets)}
"""

    if cat_cols:
        top_cat = cat_cols[0]
        vc = calc_df[top_cat].value_counts()
        if len(vc) > 0:
            pct = (vc.iloc[0] / len(calc_df)) * 100
            report += f"- **Categorical Distribution**: In `{top_cat}`, the largest segment is **{vc.index[0]}** accounting for **{int(vc.iloc[0] * (rows / len(calc_df))):,} estimated records ({pct:.1f}%)** across {calc_df[top_cat].nunique()} sampled segments.\n"

    if num_cols:
        top_num = num_cols[0]
        mean_val = float(calc_df[top_num].mean())
        med_val = float(calc_df[top_num].median())
        std_val = float(calc_df[top_num].std()) if len(calc_df) > 1 else 0.0
        report += f"- **Baseline Metric Performance**: `{top_num}` maintains an average of **{mean_val:,.2f}** (median: **{med_val:,.2f}**, std: **{std_val:,.2f}**).\n"

    # Duplicates estimation
    dups_count = len(calc_df) - len(calc_df.drop_duplicates())
    dup_str = f"Estimated ~{int(dups_count * (rows / len(calc_df))):,} duplicate rows." if dups_count > 0 else "No duplicate rows detected across the sampled records."

    report += f"""
### ⚠️ Anomalies & Operational Risk Factors
- **Data Quality Status**: {f"Zero missing cells detected." if len(null_cols) == 0 else f"{len(null_cols)} columns contain null records ({', '.join(null_cols.index[:3])})."}
- **Distribution Irregularities**: {'; '.join(skews[:2]) if skews else 'Distributions fall within standard variance bounds.'}
- **Duplicate Records**: {dup_str}

### 🎯 Strategic Actionable Recommendations
1. **Target Disparity Optimization**: Investigate groups performing below average to implement targeted retention or performance interventions.
2. **Feature Refinement**: Address asymmetric skewness in primary variables before running linear regression or clustering.
3. **Continuous Monitoring**: Track key correlation pairs over quarterly intervals to catch trend shifts early.
"""
    return sanitize_ai_markdown(report), "heuristic"


# ── Smart Chart Generator & NL to Visualizations ────────────────────────────

def recommend_smart_charts(df):
    """
    Analyzes dataset schema and returns 3-5 optimal visualization recommendations.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns.tolist()
    
    recommendations = []
    
    # 1. Date + Numeric Trend
    if date_cols and num_cols:
        recommendations.append({
            'id': 'ts_trend',
            'title': f'Chronological Trend: {num_cols[0]} over {date_cols[0]}',
            'type': 'line',
            'x': date_cols[0],
            'y': num_cols[0],
            'description': f'Tracks fluctuations and temporal trajectory of {num_cols[0]}.'
        })

    # 2. Categorical + Numeric Distribution / Bar
    if cat_cols and num_cols:
        recommendations.append({
            'id': 'cat_num_box',
            'title': f'Distribution of {num_cols[0]} by {cat_cols[0]}',
            'type': 'box',
            'x': cat_cols[0],
            'y': num_cols[0],
            'description': f'Compares quartile spread and identifies outliers across {cat_cols[0]}.'
        })
        recommendations.append({
            'id': 'cat_num_bar',
            'title': f'Average {num_cols[0]} by {cat_cols[0]}',
            'type': 'bar',
            'x': cat_cols[0],
            'y': num_cols[0],
            'agg': 'mean',
            'description': f'Highlights performance rankings across {cat_cols[0]}.'
        })

    # 3. Two Numeric Columns Correlation Scatter
    if len(num_cols) >= 2:
        recommendations.append({
            'id': 'num_scatter',
            'title': f'Co-variance Scatter: {num_cols[1]} vs {num_cols[0]}',
            'type': 'scatter',
            'x': num_cols[0],
            'y': num_cols[1],
            'color': cat_cols[0] if cat_cols else None,
            'description': f'Maps individual record relationships between {num_cols[0]} and {num_cols[1]}.'
        })

    # 4. Single Numeric Histogram
    if num_cols:
        target = num_cols[-1] if len(num_cols) > 2 else num_cols[0]
        recommendations.append({
            'id': 'hist_dist',
            'title': f'Frequency Histogram: {target}',
            'type': 'histogram',
            'x': target,
            'description': f'Visualizes density distribution, modality, and skewness of {target}.'
        })

    # 5. Categorical Breakdown Pie / Donut
    if cat_cols:
        recommendations.append({
            'id': 'cat_share',
            'title': f'Composition Breakdown: {cat_cols[0]}',
            'type': 'pie',
            'names': cat_cols[0],
            'description': f'Proportional share of each category in {cat_cols[0]}.'
        })

    return recommendations[:4]


def render_smart_chart(df, chart_spec):
    """
    Renders a Plotly figure from a chart recommendation specification.
    """
    chart_type = chart_spec.get('type')
    title = chart_spec.get('title', 'Smart Visualization')
    
    if chart_type == 'line':
        fig = px.line(df, x=chart_spec['x'], y=chart_spec['y'], color_discrete_sequence=[INSTRUMENT_THEME['accent_signal']])
    elif chart_type == 'box':
        fig = px.box(df, x=chart_spec['x'], y=chart_spec['y'], color_discrete_sequence=[INSTRUMENT_THEME['accent_signal']])
    elif chart_type == 'bar':
        agg = chart_spec.get('agg', 'mean')
        grouped = df.groupby(chart_spec['x'])[chart_spec['y']].agg(agg).reset_index()
        fig = px.bar(grouped, x=chart_spec['x'], y=chart_spec['y'], color_discrete_sequence=[INSTRUMENT_THEME['accent_signal']])
    elif chart_type == 'scatter':
        color_col = chart_spec.get('color')
        if color_col and color_col in df.columns:
            fig = px.scatter(df, x=chart_spec['x'], y=chart_spec['y'], color=color_col, color_discrete_sequence=[INSTRUMENT_THEME['accent_signal'], INSTRUMENT_THEME['accent_secondary']])
        else:
            fig = px.scatter(df, x=chart_spec['x'], y=chart_spec['y'], color_discrete_sequence=[INSTRUMENT_THEME['accent_signal']])
    elif chart_type == 'histogram':
        fig = px.histogram(df, x=chart_spec['x'], nbins=25, color_discrete_sequence=[INSTRUMENT_THEME['accent_signal']])
    elif chart_type == 'pie':
        vc = df[chart_spec['names']].value_counts().reset_index()
        vc.columns = [chart_spec['names'], 'Count']
        fig = px.pie(vc, names=chart_spec['names'], values='Count', color_discrete_sequence=px.colors.qualitative.Pastel)
    else:
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1])

    return style_plotly_fig(fig, title=title)


def generate_chart_from_nl(df, prompt, api_key=None, model=None):
    """
    Parses a natural language chart request into an interactive Plotly figure.
    """
    p = prompt.lower()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns.tolist()

    # Match columns
    found_num = [c for c in num_cols if c.lower() in p]
    found_cat = [c for c in cat_cols if c.lower() in p]
    found_date = [c for c in date_cols if c.lower() in p]

    # Chart type intent
    if 'scatter' in p:
        chart_type = 'scatter'
    elif any(k in p for k in ['box', 'spread', 'quartile']):
        chart_type = 'box'
    elif any(k in p for k in ['hist', 'histogram', 'distribution', 'density']):
        chart_type = 'histogram'
    elif any(k in p for k in ['pie', 'donut', 'proportion', 'share']):
        chart_type = 'pie'
    elif any(k in p for k in ['line', 'trend', 'over time', 'trajectory']):
        chart_type = 'line'
    else:
        chart_type = 'bar'

    spec = {'type': chart_type, 'title': f"AI Chart: {prompt.title()}"}

    if chart_type == 'line':
        spec['x'] = found_date[0] if found_date else (found_cat[0] if found_cat else df.columns[0])
        spec['y'] = found_num[0] if found_num else num_cols[0]
    elif chart_type == 'box':
        spec['x'] = found_cat[0] if found_cat else (cat_cols[0] if cat_cols else None)
        spec['y'] = found_num[0] if found_num else num_cols[0]
    elif chart_type == 'scatter':
        spec['x'] = found_num[0] if len(found_num) > 0 else num_cols[0]
        spec['y'] = found_num[1] if len(found_num) > 1 else (num_cols[1] if len(num_cols) > 1 else num_cols[0])
        spec['color'] = found_cat[0] if found_cat else None
    elif chart_type == 'histogram':
        spec['x'] = found_num[0] if found_num else (num_cols[0] if num_cols else df.columns[0])
    elif chart_type == 'pie':
        spec['names'] = found_cat[0] if found_cat else (cat_cols[0] if cat_cols else df.columns[0])
    else:  # bar
        spec['x'] = found_cat[0] if found_cat else (cat_cols[0] if cat_cols else df.columns[0])
        spec['y'] = found_num[0] if found_num else (num_cols[0] if num_cols else None)
        spec['agg'] = 'mean'

    try:
        fig = render_smart_chart(df, spec)
        return fig, f"Rendered {chart_type.upper()} visualization: {spec.get('title')}"
    except Exception as e:
        # Fallback to simple bar
        fallback_fig = px.histogram(df, x=num_cols[0] if num_cols else df.columns[0])
        return style_plotly_fig(fallback_fig, title="Fallback Distribution"), f"Rendered best-fit chart (Note: {str(e)})"


# ── AI Data Quality & Cleaning Advisor ──────────────────────────────────────

def analyze_data_quality_advisor(df):
    """
    Performs comprehensive diagnostic audit on dataset quality
    and produces actionable cleaning prescriptions.
    """
    total_rows = len(df)
    diagnostics = []
    
    # 1. Duplicates
    dup_count = len(df) - len(df.drop_duplicates())
    if dup_count > 0:
        diagnostics.append({
            'category': 'Duplicates',
            'severity': 'HIGH' if dup_count / total_rows > 0.05 else 'MEDIUM',
            'issue': f"{dup_count:,} duplicate rows detected ({dup_count/total_rows*100:.1f}%)",
            'action_key': 'drop_duplicates',
            'recommendation': "Drop duplicate entries to avoid skewing model weights and aggregations."
        })

    # 2. Missing values
    null_counts = df.isnull().sum()
    for col, cnt in null_counts.items():
        if cnt > 0:
            pct = cnt / total_rows * 100
            is_num = col in df.select_dtypes(include=[np.number]).columns
            action = 'drop_high_null_cols' if pct > 40 else ('impute_numeric' if is_num else 'impute_categorical')
            diagnostics.append({
                'category': 'Missing Data',
                'severity': 'HIGH' if pct > 20 else 'LOW',
                'issue': f"Column `{col}` has {cnt:,} missing entries ({pct:.1f}%)",
                'action_key': action,
                'col': col,
                'recommendation': f"Impute with {'median' if is_num else 'mode'} or drop records."
            })

    # 3. Outliers (IQR Method)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        series = df[col].dropna()
        if len(series) > 10:
            q25, q75 = np.percentile(series, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                outliers = series[(series < lower) | (series > upper)]
                if len(outliers) > 0 and len(outliers) / total_rows > 0.02:
                    diagnostics.append({
                        'category': 'Outliers',
                        'severity': 'MEDIUM',
                        'issue': f"Column `{col}` has {len(outliers):,} outliers ({len(outliers)/total_rows*100:.1f}%)",
                        'action_key': 'cap_outliers',
                        'col': col,
                        'recommendation': f"Cap outliers at IQR bounds [{lower:.1f}, {upper:.1f}] to stabilize variance."
                    })

    return diagnostics


def apply_cleaning_action(df, action_key, col=None):
    """
    Executes a data cleaning operation safely on a copy of the dataframe.
    """
    df_clean = df.copy()
    msg = ""

    if action_key == 'drop_duplicates':
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        after = len(df_clean)
        msg = f"Removed {before - after:,} duplicate rows."

    elif action_key == 'impute_numeric':
        if col and col in df_clean.columns:
            median_val = df_clean[col].median()
            cnt = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna(median_val)
            msg = f"Imputed {cnt:,} missing entries in `{col}` with median ({median_val:.2f})."
        else:
            # Impute all numeric
            num_cols = df_clean.select_dtypes(include=[np.number]).columns
            for c in num_cols:
                df_clean[c] = df_clean[c].fillna(df_clean[c].median())
            msg = "Imputed missing values across all numeric columns with respective medians."

    elif action_key == 'impute_categorical':
        if col and col in df_clean.columns:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            cnt = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna(mode_val)
            msg = f"Imputed {cnt:,} missing entries in `{col}` with mode ('{mode_val}')."
        else:
            cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
            for c in cat_cols:
                mode_val = df_clean[c].mode()[0] if not df_clean[c].mode().empty else 'Unknown'
                df_clean[c] = df_clean[c].fillna(mode_val)
            msg = "Imputed missing values across categorical columns with respective modes."

    elif action_key == 'cap_outliers':
        if col and col in df_clean.columns:
            series = df_clean[col].dropna()
            q25, q75 = np.percentile(series, [25, 75])
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            df_clean[col] = df_clean[col].clip(lower, upper)
            msg = f"Capped outliers in `{col}` to range [{lower:.2f}, {upper:.2f}]."

    elif action_key == 'drop_high_null_cols':
        thresh = len(df_clean) * 0.5
        before_cols = len(df_clean.columns)
        df_clean = df_clean.dropna(axis=1, thresh=thresh)
        dropped = before_cols - len(df_clean.columns)
        msg = f"Dropped {dropped} column(s) with over 50% missing values."

    return df_clean, msg


# ── Metric Driver & Predictive Influence Analysis ───────────────────────────

def analyze_metric_drivers(df, target_column):
    """
    Discovers which numeric and categorical features have the strongest
    statistical influence or correlation with a selected target variable.
    """
    if target_column not in df.columns:
        return None, "Target column not found."

    results = {
        'target': target_column,
        'numeric_correlations': [],
        'categorical_drivers': [],
        'summary_explanation': ""
    }

    is_numeric = target_column in df.select_dtypes(include=[np.number]).columns
    if not is_numeric:
        return None, "Target driver analysis requires a numeric target column."

    # 1. Numeric Correlations
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    for col in num_cols:
        series_clean = df[[target_column, col]].dropna()
        if len(series_clean) > 2:
            r = series_clean[target_column].corr(series_clean[col])
            if not np.isnan(r):
                results['numeric_correlations'].append({
                    'feature': col,
                    'correlation': round(float(r), 3),
                    'abs_correlation': round(abs(float(r)), 3),
                    'direction': 'Positive' if r > 0 else 'Negative',
                    'strength': 'Strong' if abs(r) > 0.6 else ('Moderate' if abs(r) > 0.35 else 'Weak')
                })
    results['numeric_correlations'] = sorted(
        results['numeric_correlations'],
        key=lambda x: x['abs_correlation'],
        reverse=True
    )

    # 2. Categorical Variance Drivers (Group Means Spread)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if 2 <= df[col].nunique() <= 25:
            grp = df.groupby(col)[target_column].agg(['mean', 'count']).dropna()
            if len(grp) >= 2:
                spread = grp['mean'].max() - grp['mean'].min()
                results['categorical_drivers'].append({
                    'feature': col,
                    'spread': round(float(spread), 2),
                    'top_segment': grp['mean'].idxmax(),
                    'top_mean': round(float(grp['mean'].max()), 2),
                    'bottom_segment': grp['mean'].idxmin(),
                    'bottom_mean': round(float(grp['mean'].min()), 2),
                    'categories_count': len(grp)
                })
    results['categorical_drivers'] = sorted(
        results['categorical_drivers'],
        key=lambda x: x['spread'],
        reverse=True
    )

    # Human-readable summary
    summary_lines = [f"**Driver Attribution for `{target_column}`:**\n"]
    if results['numeric_correlations']:
        top_num = results['numeric_correlations'][0]
        summary_lines.append(
            f"- **Primary Numeric Correlate**: `{top_num['feature']}` exhibits a "
            f"**{top_num['strength']} {top_num['direction']}** correlation (`r = {top_num['correlation']}`)."
        )
    if results['categorical_drivers']:
        top_cat = results['categorical_drivers'][0]
        summary_lines.append(
            f"- **Top Categorical Variance**: `{top_cat['feature']}` creates a spread of "
            f"`{top_cat['spread']}` between highest segment (**{top_cat['top_segment']}**: `{top_cat['top_mean']}`) "
            f"and lowest segment (**{top_cat['bottom_segment']}**: `{top_cat['bottom_mean']}`)."
        )

    results['summary_explanation'] = "\n".join(summary_lines)
    return results, None

def between_range(val, low, high):
    return low <= val <= high
