# ============================================================
# 🧠 SENTIMENT & CAPITAL INTELLIGENCE - STREAMLIT APP
#
# Run:   .\.venv\Scripts\python.exe -m streamlit run sentiment_capital_app.py
#
# Engines: LR + SVM Ensemble, Logistic Regression, Linear SVC,
#          Federated (FedAvg SGD), Deep Learning LSTM
# Tabs:    Predict | Batch CSV | Model Performance | About
#
# Saare sklearn models text_preprocessing.clean_text se train hue hain:
# emojis remove NAHI hote, wo prediction mein features ki tarah use hote hain.
# ============================================================
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from text_preprocessing import clean_text

st.set_page_config(page_title="Sentiment & Capital Intelligence", page_icon="🧠", layout="wide")

ENGINES = ["LR + SVM Ensemble", "Logistic Regression", "Linear SVC",
           "Federated (FedAvg SGD)", "Deep Learning LSTM"]
EXAMPLES = [
    "😊 My team is doing amazing work 🚀",
    "😞 My business is losing money 📉",
    "Financial growth is not good",
    "🎓 Years of practical work built my confidence!",
    "🙏 My faith gives me peace and strength every day",
]
POS_COLOR, NEG_COLOR, ACCENT = "#22c55e", "#ef4444", "#38bdf8"


# ============================================================
# 📂 MODEL LOADING (cached - sirf ek baar load hote hain)
# ============================================================
def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading LR / SVM / Federated models...")
def load_sklearn_models():
    return {
        "lr_sent": _load_pickle("sentiment_lr_model.pkl"),
        "lr_cap": _load_pickle("capital_lr_model.pkl"),
        "svm_sent": _load_pickle("sentiment_svm_model.pkl"),
        "svm_cap": _load_pickle("capital_svm_model.pkl"),
        "fed_sent": _load_pickle("federated_sentiment_model.pkl"),
        "fed_cap": _load_pickle("federated_capital_model.pkl"),
    }


@st.cache_resource(show_spinner="Loading LSTM models (TensorFlow)...")
def load_lstm_models():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow.keras.models import load_model

    sent = load_model("sentiment_lstm_model(1)_fixed.keras", compile=False, safe_mode=False)
    cap = load_model("capital_lstm_model(1)_fixed.keras", compile=False, safe_mode=False)
    params = _load_pickle("lstm_params.pkl") if os.path.exists("lstm_params.pkl") else {}
    return {
        "sent": sent,
        "cap": cap,
        "tokenizer": _load_pickle("lstm_tokenizer.pkl"),
        "sent_enc": _load_pickle("sentiment_encoder.pkl"),
        "cap_enc": _load_pickle("capital_encoder.pkl"),
        "max_len": sent.input_shape[1],
        # Naya (emoji-aware) LSTM shared cleaner ke saath train hua hai; purana ASCII cleaner ke saath
        "emoji_aware": isinstance(params, dict) and params.get("cleaner") == "text_preprocessing.clean_text",
    }


# ============================================================
# 🧮 PROBABILITY HELPERS
# ============================================================
def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _svm_proba(model, cleaned_list):
    """LinearSVC ke decision scores -> pseudo-probabilities (binary: sigmoid, multiclass: softmax)."""
    d = np.atleast_2d(model.decision_function(cleaned_list))
    if d.shape[0] == 1 and len(cleaned_list) > 1:  # binary returns 1-D
        d = d.T
    if d.shape[1] == 1:
        p = 1 / (1 + np.exp(-d[:, 0]))
        return np.column_stack([1 - p, p])
    return np.apply_along_axis(_softmax, 1, d)


def _legacy_lstm_clean(text):
    """Purane (emoji ke bina train hue) LSTM ka cleaner."""
    text = str(text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)
    text = re.sub(r"\bnever\s+(\w+)", r"never_\1", text)
    text = re.sub(r"\bno\s+(\w+)", r"no_\1", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s_]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# 🧠 PREDICTION (batch - list of texts in, probabilities out)
# ============================================================
def predict_proba(engine, texts):
    """Returns (cleaned_texts, sent_probs[n,2], sent_classes, cap_probs[n,k], cap_classes)."""
    if engine == "Deep Learning LSTM":
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        m = load_lstm_models()
        cleaned = [clean_text(t) if m["emoji_aware"] else _legacy_lstm_clean(t) for t in texts]
        seq = pad_sequences(m["tokenizer"].texts_to_sequences(cleaned), maxlen=m["max_len"],
                            padding="post", truncating="post")
        sp = m["sent"].predict(seq, batch_size=512, verbose=0)
        cp = m["cap"].predict(seq, batch_size=512, verbose=0)
        return cleaned, sp, m["sent_enc"].classes_, cp, m["cap_enc"].classes_

    m = load_sklearn_models()
    cleaned = [clean_text(t) for t in texts]

    if engine == "Federated (FedAvg SGD)":
        sp = m["fed_sent"]["model"].predict_proba(m["fed_sent"]["vectorizer"].transform(cleaned))
        cp = m["fed_cap"]["model"].predict_proba(m["fed_cap"]["vectorizer"].transform(cleaned))
        return cleaned, sp, m["fed_sent"]["encoder"].classes_, cp, m["fed_cap"]["encoder"].classes_

    lr_sp = m["lr_sent"]["model"].predict_proba(cleaned)
    lr_cp = m["lr_cap"]["model"].predict_proba(cleaned)
    svm_sp = _svm_proba(m["svm_sent"]["model"], cleaned)
    svm_cp = _svm_proba(m["svm_cap"]["model"], cleaned)
    if engine == "Logistic Regression":
        sp, cp = lr_sp, lr_cp
    elif engine == "Linear SVC":
        sp, cp = svm_sp, svm_cp
    else:
        sp, cp = (lr_sp + svm_sp) / 2, (lr_cp + svm_cp) / 2
    return cleaned, sp, m["lr_sent"]["encoder"].classes_, cp, m["lr_cap"]["encoder"].classes_


def predict_one(engine, text):
    cleaned, sp, s_cls, cp, c_cls = predict_proba(engine, [text])
    s_idx = int(np.argmax(sp[0]))
    capitals = sorted(zip(c_cls, cp[0]), key=lambda x: x[1], reverse=True)
    return {
        "cleaned": cleaned[0],
        "sentiment": s_cls[s_idx],
        "sentiment_conf": float(sp[0][s_idx]),
        "sentiment_probs": dict(zip(s_cls, map(float, sp[0]))),
        "capitals": [(c, float(p)) for c, p in capitals],
    }


# ============================================================
# 📊 CHARTS
# ============================================================
def sentiment_gauge(result):
    pos = result["sentiment_probs"].get("Positive", 0.0) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pos,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Positive probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": POS_COLOR if pos >= 50 else NEG_COLOR},
            "steps": [{"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                      {"range": [50, 100], "color": "rgba(34,197,94,0.15)"}],
            "threshold": {"line": {"color": "#94a3b8", "width": 2}, "value": 50},
        },
    ))
    fig.update_layout(height=230, margin=dict(t=40, b=10, l=30, r=30))
    return fig


def capital_bar(capitals, top_n=7):
    top = capitals[:top_n][::-1]
    fig = go.Figure(go.Bar(
        x=[p * 100 for _, p in top], y=[c for c, _ in top], orientation="h",
        marker_color=ACCENT, text=[f"{p * 100:.1f}%" for _, p in top], textposition="outside",
    ))
    fig.update_layout(height=330, margin=dict(t=10, b=10, l=10, r=40),
                      xaxis=dict(title="Probability (%)", range=[0, 110]))
    return fig


# ============================================================
# 🖥️ LAYOUT
# ============================================================
st.title("🧠 Sentiment & Capital Intelligence")
st.caption("Sentence (emojis ke saath) likhein: model Sentiment (Positive / Negative) "
           "aur 16 types mein se Capital predict karega.")

with st.sidebar:
    st.header("⚙️ Settings")
    engine = st.selectbox("AI engine", ENGINES, index=0)
    if engine == "Deep Learning LSTM":
        st.info("LSTM pehli dafa load hone mein kuch second lagata hai (TensorFlow).")
    st.divider()
    st.markdown("**Emojis 😊 📉 🚀** remove nahi hote. LR, SVM, Federated aur naya LSTM "
                "unhe prediction mein use karte hain.")

tab_predict, tab_batch, tab_perf, tab_about = st.tabs(
    ["🔮 Predict", "📂 Batch CSV", "📊 Model Performance", "ℹ️ About"])

# ---------------------------- PREDICT ----------------------------
with tab_predict:
    if "text_input" not in st.session_state:
        st.session_state.text_input = EXAMPLES[0]

    st.markdown("**Examples:**")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex[:28] + ("…" if len(ex) > 28 else ""), key=f"ex_{ex}", use_container_width=True):
            st.session_state.text_input = ex

    text = st.text_area("Sentence", key="text_input", height=110,
                        placeholder="e.g. 😊 My team is doing amazing work 🚀")
    run = st.button("🚀 Predict", type="primary")

    if run:
        if not text.strip():
            st.warning("Pehle koi sentence likhein.")
        else:
            try:
                res = predict_one(engine, text)
            except FileNotFoundError as e:
                st.error(f"Model file nahi mili: `{e.filename}`. Saari .pkl / .keras files project folder mein honi chahiye.")
                st.stop()
            except Exception as e:  # e.g. TensorFlow load issue
                st.error(f"{engine} se prediction nahi ho saki: {e}")
                st.stop()

            st.markdown("##### 🧹 Cleaned text (jo model ne dekha)")
            st.code(res["cleaned"] or "(empty)", language=None)

            left, right = st.columns([1, 1.4], gap="large")
            with left:
                color = POS_COLOR if res["sentiment"].lower() == "positive" else NEG_COLOR
                icon = "😊" if res["sentiment"].lower() == "positive" else "😞"
                st.markdown(f"### Sentiment: <span style='color:{color}'>{icon} {res['sentiment'].upper()}</span>",
                            unsafe_allow_html=True)
                st.metric("Confidence", f"{res['sentiment_conf'] * 100:.2f}%")
                st.plotly_chart(sentiment_gauge(res), use_container_width=True)
            with right:
                top_cap, top_p = res["capitals"][0]
                st.markdown(f"### Capital: <span style='color:{ACCENT}'>{top_cap}</span>", unsafe_allow_html=True)
                st.metric("Confidence", f"{top_p * 100:.2f}%")
                st.plotly_chart(capital_bar(res["capitals"]), use_container_width=True)

            with st.expander("🔍 Compare all engines on this sentence"):
                rows = []
                for eng in ENGINES:
                    try:
                        r = predict_one(eng, text)
                        rows.append({"Engine": eng, "Sentiment": r["sentiment"],
                                     "Sentiment conf.": f"{r['sentiment_conf'] * 100:.1f}%",
                                     "Capital": r["capitals"][0][0],
                                     "Capital conf.": f"{r['capitals'][0][1] * 100:.1f}%"})
                    except Exception as e:
                        rows.append({"Engine": eng, "Sentiment": f"error: {e}"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---------------------------- BATCH CSV ----------------------------
with tab_batch:
    st.markdown(f"CSV upload karein, text column chunein, aur **{engine}** se saari rows predict karein.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            bdf = pd.read_csv(up, low_memory=False)
        except UnicodeDecodeError:
            up.seek(0)
            bdf = pd.read_csv(up, low_memory=False, encoding="latin-1")
        st.write(f"Rows: **{len(bdf):,}**")
        text_cols = [c for c in bdf.columns if bdf[c].dtype == object] or list(bdf.columns)
        default = text_cols.index("Capitals Text Generation") if "Capitals Text Generation" in text_cols else 0
        col = st.selectbox("Text column", text_cols, index=default)
        max_rows = st.number_input("Max rows to predict", 1, len(bdf), min(len(bdf), 5000), step=1000)
        if st.button("▶️ Run batch prediction", type="primary"):
            sub = bdf.head(int(max_rows)).copy()
            texts = sub[col].fillna("").astype(str).tolist()
            out_cleaned, out_sent, out_sconf, out_cap, out_cconf = [], [], [], [], []
            bar = st.progress(0.0, text="Predicting...")
            chunk = 2000
            for i in range(0, len(texts), chunk):
                cleaned, sp, s_cls, cp, c_cls = predict_proba(engine, texts[i:i + chunk])
                out_cleaned += cleaned
                out_sent += list(s_cls[np.argmax(sp, axis=1)])
                out_sconf += list(np.max(sp, axis=1))
                out_cap += list(c_cls[np.argmax(cp, axis=1)])
                out_cconf += list(np.max(cp, axis=1))
                bar.progress(min(1.0, (i + chunk) / len(texts)), text=f"Predicting... {min(i + chunk, len(texts)):,}/{len(texts):,}")
            bar.empty()
            sub["Cleaned Text"] = out_cleaned
            sub["Predicted Sentiment"] = out_sent
            sub["Sentiment Confidence"] = np.round(out_sconf, 4)
            sub["Predicted Capital (model)"] = out_cap
            sub["Capital Confidence"] = np.round(out_cconf, 4)

            c1, c2 = st.columns(2)
            with c1:
                vc = sub["Predicted Sentiment"].value_counts()
                fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.5,
                                       marker_colors=[POS_COLOR if l == "Positive" else NEG_COLOR for l in vc.index]))
                fig.update_layout(height=300, margin=dict(t=30, b=10), title="Sentiment distribution")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                vc = sub["Predicted Capital (model)"].value_counts().head(10)[::-1]
                fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h", marker_color=ACCENT))
                fig.update_layout(height=300, margin=dict(t=30, b=10), title="Top predicted capitals")
                st.plotly_chart(fig, use_container_width=True)

            if "Sentiment Label" in sub.columns:
                acc = (sub["Sentiment Label"].astype(str).str.strip().str.capitalize() == sub["Predicted Sentiment"]).mean()
                st.metric("Sentiment agreement with 'Sentiment Label' column", f"{acc * 100:.2f}%")

            st.dataframe(sub, use_container_width=True, height=350)
            st.download_button("⬇️ Download predictions CSV", sub.to_csv(index=False).encode("utf-8-sig"),
                               file_name="predictions.csv", mime="text/csv")

# ---------------------------- MODEL PERFORMANCE ----------------------------
with tab_perf:
    st.markdown("Held-out test set (20%, duplicate texts removed before splitting) par results.")
    rows = []
    if os.path.exists("retrain_metrics_report.json"):
        rep = json.load(open("retrain_metrics_report.json", encoding="utf-8"))
        for key, name in [("lr_sentiment", "Logistic Regression"), ("svm_sentiment", "Linear SVC")]:
            if key in rep:
                cap_key = key.replace("sentiment", "capital")
                rows.append({"Model": name,
                             "Sentiment accuracy": rep[key]["accuracy"],
                             "Sentiment F1 (macro)": rep[key]["f1_macro"],
                             "Capital accuracy": rep.get(cap_key, {}).get("accuracy"),
                             "Capital F1 (macro)": rep.get(cap_key, {}).get("f1_macro")})
    if os.path.exists("federated_metrics_report.json"):
        rep = json.load(open("federated_metrics_report.json", encoding="utf-8"))
        rows.append({"Model": "Federated (FedAvg SGD)",
                     "Sentiment accuracy": rep["sentiment"]["accuracy"],
                     "Sentiment F1 (macro)": rep["sentiment"]["f1_macro"],
                     "Capital accuracy": rep["capital"]["accuracy"],
                     "Capital F1 (macro)": rep["capital"]["f1_macro"]})
    if os.path.exists("lstm_metrics_report.json"):
        rep = json.load(open("lstm_metrics_report.json", encoding="utf-8"))
        rows.append({"Model": "Deep Learning LSTM",
                     "Sentiment accuracy": rep.get("sentiment_accuracy"),
                     "Capital accuracy": rep.get("capital_accuracy")})

    if rows:
        perf = pd.DataFrame(rows)
        st.dataframe(perf.style.format({c: "{:.2%}" for c in perf.columns if c != "Model"}, na_rep="—"),
                     hide_index=True, use_container_width=True)
        fig = go.Figure()
        for metric, color in [("Sentiment accuracy", POS_COLOR), ("Capital accuracy", ACCENT)]:
            fig.add_bar(name=metric, x=perf["Model"], y=perf[metric] * 100, marker_color=color,
                        text=[f"{v * 100:.1f}%" if pd.notna(v) else "" for v in perf[metric]],
                        textposition="outside")
        fig.update_layout(barmode="group", height=380, yaxis=dict(title="Accuracy (%)", range=[0, 110]),
                          margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Abhi koi metrics report nahi mili. Pehle training scripts chalayein.")

    images = [("sentiment_confusion_matrix.png", "Sentiment - LR"),
              ("svm_sentiment_confusion.png", "Sentiment - Linear SVC"),
              ("capital_confusion_matrix.png", "Capital - LR")]
    images = [(p, c) for p, c in images if os.path.exists(p)]
    if images:
        st.markdown("#### Confusion matrices")
        for col, (path, cap) in zip(st.columns(len(images)), images):
            col.image(path, caption=cap, use_container_width=True)

# ---------------------------- ABOUT ----------------------------
with tab_about:
    st.markdown("""
#### Project
Social media text se **Sentiment** (Positive / Negative) aur **Capital** type
(Emotional, Social, Economic, Religious, ... 16 classes) predict karna.

#### Engines
| Engine | Kya hai |
|---|---|
| **LR + SVM Ensemble** | Logistic Regression aur Linear SVC ki probabilities ka average |
| **Logistic Regression** | TF-IDF (1-2 grams, 50k features) + LR |
| **Linear SVC** | TF-IDF + Linear Support Vector Classifier |
| **Federated (FedAvg SGD)** | 5 clients, 10 rounds federated averaging, SGD (log-loss) |
| **Deep Learning LSTM** | Embedding + 2x Bidirectional LSTM |

#### Preprocessing (`text_preprocessing.py`)
1. Dataset ke kharab encoded emojis repair (`Ã°Å¸Å½â€œ` → 🎓)
2. Lowercase, URLs / HTML remove
3. **Emojis rakhe jaate hain** aur alag tokens bante hain
4. Negation handling: `not good` → `not_good` (punctuation / "but" tak)

#### Re-training
```
.\\.venv\\Scripts\\python.exe retrain_lr_svm_fixed.py   # LR + Linear SVC
.\\.venv\\Scripts\\python.exe train_federated_sgd.py    # Federated SGD
.\\.venv\\Scripts\\python.exe train_lstm.py             # LSTM (CPU par ~2-3 ghante)
```
""")
