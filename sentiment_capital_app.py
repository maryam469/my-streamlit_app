"""
MULTI-MODEL SENTIMENT & CAPITAL INTEL SYSTEM  -  Streamlit app

Run locally:   streamlit run sentiment_capital_app.py
Streamlit Cloud: Main file path = sentiment_capital_app.py
"""
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model_engine as M

st.set_page_config(page_title="Multi-Model Sentiment & Capital Intel System",
                   page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ style
BG, ROW, TRACK, CYAN, BLUE = "#0f172a", "#1e293b", "#334155", "#38bdf8", "#3b82f6"

st.markdown(f"""
<style>
.stApp, header[data-testid="stHeader"] {{ background:{BG}; }}
.block-container {{ padding-top:2.5rem; max-width:1600px; }}
h1.title {{ color:#fff; font-size:2.6rem; font-weight:800; margin:0 0 .6rem 0; }}
h3.sec {{ color:#fff; font-size:1.55rem; font-weight:700; margin:.9rem 0 .5rem 0;
          letter-spacing:.3px; }}
.lbl {{ color:#fff; font-weight:700; font-size:1.05rem; margin:.9rem 0 .4rem 0; }}
/* light selectbox like the screenshot */
div[data-baseweb="select"] > div {{ background:#f1f5f9 !important; border-radius:8px; }}
div[data-baseweb="select"] * {{ color:#0f172a !important; }}
[data-testid="stSelectbox"] [role="group"] {{ background:#f1f5f9 !important; border-radius:8px; }}
[data-testid="stSelectbox"] input, [data-testid="stSelectbox"] button svg {{
    color:#0f172a !important; -webkit-text-fill-color:#0f172a; fill:#0f172a; }}
/* dark text areas */
.stTextArea textarea {{ background:{ROW}; color:#e5e7eb; font-size:1rem; }}
.stTextArea textarea:disabled {{ color:#94a3b8; -webkit-text-fill-color:#94a3b8; }}
/* blue full-width button */
div.stButton > button {{ background:{BLUE}; color:#fff; border:none; border-radius:8px;
                         height:3.2rem; }}
div.stButton > button p {{ font-size:1.05rem; letter-spacing:.3px; }}
div.stButton > button:hover {{ background:#2563eb; color:#fff; }}
.verdict {{ color:#fff; font-size:3.2rem; font-weight:800; margin:.4rem 0 1rem 0; }}
.conf {{ color:#fff; font-size:1.05rem; margin-bottom:.4rem; }}
.cbar {{ background:#f1f5f9; height:9px; border-radius:6px; overflow:hidden; }}
.cbar > div {{ background:{BLUE}; height:100%; }}
hr.sep {{ border:none; border-top:1px solid #1e293b; margin:3rem 0 2rem 0; }}
.caprow {{ display:flex; align-items:center; background:{ROW}; border-radius:6px;
           padding:13px 18px; margin-bottom:5px; }}
.caprow .n {{ color:#fff; font-weight:700; font-size:1.1rem; width:25%; }}
.caprow .t {{ flex:1; background:{TRACK}; height:16px; border-radius:8px; margin:0 3%; overflow:hidden; }}
.caprow .t > div {{ background:{CYAN}; height:100%; border-radius:8px; min-width:12px; }}
.caprow .p {{ color:{CYAN}; font-weight:700; font-size:1.1rem; width:80px; text-align:right; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ engines (cached)
@st.cache_resource(show_spinner="Loading models…")
def get_engine(name: str, padding: str):
    if name == M.ENGINE_CLASSIC:
        return M.ClassicEngine()
    lstm = M.LSTMEngine(padding=padding)
    if name == M.ENGINE_LSTM:
        return lstm
    try:
        classic = M.ClassicEngine()
    except Exception:
        classic = None
    return M.FederatedEngine(lstm, classic)


def gauge(value: float | None) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0 if value is None else round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 72, "color": "#fff"}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e5e7eb",
                     "tickfont": {"color": "#e5e7eb", "size": 13}, "dtick": 20},
            "bar": {"color": CYAN, "thickness": 0.6},
            "bgcolor": ROW, "borderwidth": 0,
        },
    ))
    fig.update_layout(height=340, margin=dict(l=50, r=50, t=30, b=30),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
    return fig


def capital_rows(probs: dict[str, float]) -> str:
    return "".join(
        f'<div class="caprow"><span class="n">{html.escape(n)}</span>'
        f'<div class="t"><div style="width:{p*100:.2f}%"></div></div>'
        f'<span class="p">{p*100:.2f}%</span></div>'
        for n, p in probs.items())


def capital_chart(probs: dict[str, float]) -> go.Figure:
    names = list(probs)[::-1]
    vals = [probs[n] * 100 for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=CYAN,
                           hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig.update_layout(
        height=60 + 55 * len(names), margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=ROW, bargap=0.2,
        font=dict(color="#94a3b8", size=13),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False),
        yaxis=dict(gridcolor="#334155", showgrid=True),
    )
    return fig


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_n = st.slider("Capitals to show", 3, 16, 7)
    padding = st.radio("LSTM padding (must match training)", ["post", "pre"], horizontal=True,
                       help="Use the same value as pad_sequences(..., padding=...) in your training code.")
    st.caption(f"Model folder: `{M.MODEL_DIR}`")

# ------------------------------------------------------------------ layout
st.markdown('<h1 class="title">🧠 MULTI-MODEL SENTIMENT &amp; CAPITAL INTEL SYSTEM</h1>',
            unsafe_allow_html=True)
left, _, right = st.columns([1, 0.12, 2.1])
res: M.Prediction | None = st.session_state.get("res")

with left:
    st.markdown('<h3 class="sec">SELECT AI ENGINE MODEL</h3>', unsafe_allow_html=True)
    engine = st.selectbox("engine", M.available_engines(), label_visibility="collapsed")

    st.markdown('<h3 class="sec">INPUT RAW TEXT CONTEXT</h3>', unsafe_allow_html=True)
    text = st.text_area("text", height=175, label_visibility="collapsed",
                        value="i doubt your theory would hold up he said feeling a bit superior")

    if st.button("EXECUTE INFERENCE ENGINE", width="stretch"):
        miss = M.missing(engine)
        if miss:
            st.error("Missing model files:\n\n" + "\n".join(f"- `{m}`" for m in miss))
        elif not text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                with st.spinner("Running inference…"):
                    res = get_engine(engine, padding).predict(text)
                st.session_state["res"] = res
            except Exception as e:
                st.error(f"Prediction failed: {type(e).__name__}: {e}")
                st.info("Open **Model diagnostics** at the bottom of the page to see what each file contains.")

    st.markdown('<div class="lbl">PARSED ENGINE OUTPUT TEXT</div>', unsafe_allow_html=True)
    st.text_area("parsed", value=res.parsed_text if res else "", height=120,
                 disabled=True, label_visibility="collapsed")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.markdown('<h3 class="sec">ENGINE SENTIMENT PROJECTION</h3>', unsafe_allow_html=True)
    if res:
        c = res.confidence * 100
        st.markdown(
            f'<div class="verdict">{html.escape(res.sentiment.upper())}</div>'
            f'<div class="conf">Confidence Matrix: {c:.2f}%</div>'
            f'<div class="cbar"><div style="width:{c:.2f}%"></div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict" style="color:#475569">—</div>'
                    '<div class="conf">Confidence Matrix: —</div><div class="cbar"></div>',
                    unsafe_allow_html=True)

with right:
    st.plotly_chart(gauge(res.confidence if res else None), config={"displayModeBar": False})
    st.markdown(f'<h3 class="sec">TOP {top_n} PREDICTED CAPITALS</h3>', unsafe_allow_html=True)
    if res:
        top = dict(list(res.capital_probs.items())[:top_n])
        st.markdown(capital_rows(top), unsafe_allow_html=True)
        st.plotly_chart(capital_chart(top), config={"displayModeBar": False})
    else:
        st.info("Press **EXECUTE INFERENCE ENGINE** to see the capital probabilities.")

# ------------------------------------------------------------------ diagnostics
with st.expander("🔧 Model diagnostics (which files were found and what they contain)"):
    st.dataframe(pd.DataFrame(M.diagnostics()), hide_index=True)
    try:
        _, info = M.make_cleaner()
        st.caption(f"Text cleaner in use: {info}")
    except Exception as e:
        st.caption(f"Text cleaner error: {e}")
    if res:
        st.caption(f"Last prediction engine: {res.engine}")
