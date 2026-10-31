"""
model_engine.py - loads YOUR trained files and runs predictions.

Expected files (same folder as this file, or set MODEL_DIR):
  text_cleaner.pkl, tfidf.pkl,
  sentiment_lr_model.pkl, capital_lr_model.pkl, sentiment_encoder.pkl, capital_encoder.pkl,
  lstm_tokenizer.pkl, lstm_params.pkl,
  best_sentiment_lstm.h5 (or sentiment_lstm_model.h5), best_capital_lstm.h5,
  sentiment_lstm_encoder.pkl, capital_lstm_encoder.pkl,
  federated_sentiment_model.pkl (optional)

Everything is loaded defensively: the loader inspects each object and adapts
(predict_proba vs decision_function, LabelEncoder vs dict vs list encoders,
softmax vs sigmoid LSTM outputs, ...). The "Model diagnostics" panel in the
app shows what was found, so you can see exactly what each file contains.
"""
from __future__ import annotations

import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", APP_DIR))
# text_cleaner.pkl may reference classes/functions defined in text_preprocessing.py
for p in (str(APP_DIR), str(MODEL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

ENGINE_CLASSIC = "LR + SVM Ensemble"
ENGINE_LSTM = "Deep Learning LSTM"
ENGINE_FED = "Federated LSTM (FedAvg)"

SENT_LSTM_CANDIDATES = ["best_sentiment_lstm.h5", "sentiment_lstm_model.h5"]
ALL_FILES = [
    "text_cleaner.pkl", "tfidf.pkl",
    "sentiment_lr_model.pkl", "capital_lr_model.pkl",
    "sentiment_encoder.pkl", "capital_encoder.pkl",
    "lstm_tokenizer.pkl", "lstm_params.pkl",
    *SENT_LSTM_CANDIDATES, "best_capital_lstm.h5",
    "sentiment_lstm_encoder.pkl", "capital_lstm_encoder.pkl",
    "federated_sentiment_model.pkl",
]


# ------------------------------------------------------------------ helpers
def path_of(name: str) -> Path:
    return MODEL_DIR / name


def exists(name: str) -> bool:
    return path_of(name).exists()


def load_pickle(name: str):
    """joblib.load reads both joblib and plain-pickle files."""
    p = path_of(name)
    try:
        import joblib
        return joblib.load(p)
    except Exception:
        with open(p, "rb") as f:
            return pickle.load(f)


def describe(obj) -> str:
    t = type(obj)
    s = f"{t.__module__}.{t.__name__}"
    if isinstance(obj, dict):
        s += " {" + ", ".join(f"{k}: {type(v).__name__}" for k, v in list(obj.items())[:8]) + "}"
    elif isinstance(obj, (list, tuple)):
        s += f" len={len(obj)}"
    elif hasattr(obj, "classes_"):
        s += f" classes={list(getattr(obj, 'classes_'))[:6]}"
    return s


def _softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def labels_from(classes, encoder=None) -> list[str]:
    """Turn model class ids into readable names using whatever encoder format exists."""
    classes = list(classes)
    if classes and all(isinstance(c, str) for c in classes):
        return classes
    if encoder is not None:
        try:
            if hasattr(encoder, "inverse_transform"):
                return [str(x) for x in encoder.inverse_transform(np.array(classes).astype(int))]
        except Exception:
            pass
        if isinstance(encoder, dict):
            # {label: id} or {id: label}
            if all(isinstance(v, (int, np.integer)) for v in encoder.values()):
                inv = {int(v): str(k) for k, v in encoder.items()}
            else:
                inv = {int(k): str(v) for k, v in encoder.items()}
            return [inv.get(int(c), str(c)) for c in classes]
        if isinstance(encoder, (list, tuple, np.ndarray)):
            return [str(encoder[int(c)]) for c in classes]
    return [str(c) for c in classes]


def encoder_names(encoder, n: int) -> list[str]:
    """Names for output index 0..n-1 of a neural network."""
    if encoder is None:
        return [str(i) for i in range(n)]
    if hasattr(encoder, "classes_"):
        return [str(c) for c in encoder.classes_][:n]
    return labels_from(range(n), encoder)


def sklearn_proba(model, X):
    """(probabilities, classes) for any sklearn classifier / pipeline."""
    classes = getattr(model, "classes_", None)
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X))[0], classes
        except Exception:
            pass  # e.g. hard-voting ensemble
    if hasattr(model, "decision_function"):
        d = np.atleast_1d(np.asarray(model.decision_function(X))[0])
        if d.size == 1:                       # binary
            p = 1 / (1 + np.exp(-d[0]))
            return np.array([1 - p, p]), classes
        return _softmax(d), classes
    pred = model.predict(X)[0]                # last resort: one-hot of prediction
    classes = classes if classes is not None else [pred]
    return np.array([1.0 if c == pred else 0.0 for c in classes]), classes


# ------------------------------------------------------------------ dict-wrapped models
MODEL_KEYS_HINT = ("model", "clf", "classifier", "lr", "svm", "ensemble", "estimator")


class DictModel:
    """Wraps a pickled dict such as {"lr": LR, "svm": SVM, "vectorizer": tfidf,
    "label_encoder": le}. All estimators inside are combined by averaging their
    probabilities (soft voting). Exposes predict_proba / classes_ like sklearn."""

    def __init__(self, d: dict, fname: str):
        self.fname, self.keys = fname, list(d.keys())
        self.models, self.vectorizer, self.encoder, classes = [], None, None, None
        for k, v in d.items():
            key = str(k).lower()
            items = list(v) if isinstance(v, (list, tuple)) else [v]
            for x in items:
                if hasattr(x, "predict"):
                    self.models.append((str(k), x))
                elif hasattr(x, "inverse_transform") and hasattr(x, "classes_") and not hasattr(x, "vocabulary_"):
                    self.encoder = x
                elif hasattr(x, "transform") and ("vocabulary_" in dir(x) or "vec" in key or "tfidf" in key):
                    self.vectorizer = x
                elif isinstance(v, dict) and any(t in key for t in MODEL_KEYS_HINT):
                    inner = DictModel(v, f"{fname}[{k}]")       # nested dict of models
                    self.models += inner.models
                    self.vectorizer = self.vectorizer or inner.vectorizer
                    self.encoder = self.encoder or inner.encoder
            if key in ("classes", "classes_", "labels", "label_names", "class_names", "target_names"):
                classes = list(v)
        if not self.models:
            raise ValueError(f"{fname} is a dict with keys {self.keys} but none of them is a trained "
                             f"model (nothing with .predict). Re-save the model object itself.")
        self.classes_ = classes if classes is not None else getattr(self.models[0][1], "classes_", None)

    @property
    def names(self) -> str:
        return " + ".join(f"{k}:{type(m).__name__}" for k, m in self.models)

    def predict_proba(self, X):
        total = None
        for _, m in self.models:
            p, cls = sklearn_proba(m, X)
            cls = list(cls) if cls is not None else list(range(len(p)))
            if self.classes_ is not None and len(self.classes_) == len(p):
                ref = list(self.classes_)
                if all(c in ref for c in cls):                  # align class order
                    aligned = np.zeros(len(ref))
                    for c, v in zip(cls, p):
                        aligned[ref.index(c)] = v
                    p = aligned
            total = p if total is None else total + p
        out = total / len(self.models)
        return out.reshape(1, -1)

    def predict(self, X):
        return np.array([np.asarray(self.classes_)[int(self.predict_proba(X).argmax())]])


def as_model(obj, fname: str):
    """Return something with predict/predict_proba, unwrapping dicts if needed."""
    if isinstance(obj, dict):
        return DictModel(obj, fname)
    if not hasattr(obj, "predict"):
        raise ValueError(f"{fname} contains {describe(obj)}, which is not a trained model.")
    return obj


def as_vectorizer(obj, fname: str):
    if isinstance(obj, dict):
        for v in obj.values():
            if hasattr(v, "transform"):
                return v
        raise ValueError(f"{fname} is a dict with keys {list(obj)} but holds no vectorizer.")
    return obj


# ------------------------------------------------------------------ text cleaning
def _builtin_clean(text: str) -> str:
    """Same steps as the notebook: strip specials -> tokenize -> drop stopwords."""
    text = re.sub(r"<.*?>", "", str(text))
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            sw = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            sw = set(stopwords.words("english"))
        try:
            toks = nltk.word_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            toks = nltk.word_tokenize(text)
    except Exception:
        sw, toks = set(), text.split()
    return " ".join(w for w in toks if w.lower() not in sw)


def make_cleaner():
    """Returns (clean_fn, description)."""
    # 1) text_cleaner.pkl
    if exists("text_cleaner.pkl"):
        try:
            obj = load_pickle("text_cleaner.pkl")
            for attr in ("clean", "clean_text", "preprocess", "preprocess_text", "transform_text"):
                if hasattr(obj, attr) and callable(getattr(obj, attr)):
                    fn = getattr(obj, attr)
                    return (lambda t, fn=fn: str(fn(t))), f"text_cleaner.pkl -> {describe(obj)}.{attr}()"
            if hasattr(obj, "transform"):
                def fn(t, obj=obj):
                    out = obj.transform([t])
                    return str(out[0]) if not isinstance(out, str) else out
                return fn, f"text_cleaner.pkl -> {describe(obj)}.transform()"
            if callable(obj):
                return (lambda t, obj=obj: str(obj(t))), f"text_cleaner.pkl -> {describe(obj)}()"
        except Exception as e:
            err = f"text_cleaner.pkl failed to load ({type(e).__name__}: {e})"
        else:
            err = "text_cleaner.pkl has no clean/transform method"
    else:
        err = "text_cleaner.pkl not found"
    # 2) a function inside text_preprocessing.py
    try:
        import text_preprocessing as tp
        for name in ("clean_text", "preprocess_text", "preprocess", "clean", "transform_text"):
            fn = getattr(tp, name, None)
            if callable(fn):
                return (lambda t, fn=fn: str(fn(t))), f"text_preprocessing.{name}()  [{err}]"
    except Exception:
        pass
    # 3) built-in notebook preprocessing
    return _builtin_clean, f"built-in cleaner (notebook steps)  [{err}]"


# ------------------------------------------------------------------ result
@dataclass
class Prediction:
    engine: str
    parsed_text: str
    sentiment: str
    confidence: float
    sentiment_probs: dict[str, float]
    capital: str
    capital_probs: dict[str, float] = field(default_factory=dict)   # sorted high -> low


def _pack(engine, parsed, s_names, s_p, c_names, c_p) -> Prediction:
    s_p, c_p = np.asarray(s_p, float), np.asarray(c_p, float)
    si, ci = int(s_p.argmax()), int(c_p.argmax())
    order = np.argsort(c_p)[::-1]
    return Prediction(
        engine=engine, parsed_text=parsed,
        sentiment=str(s_names[si]), confidence=float(s_p[si]),
        sentiment_probs={str(n): float(p) for n, p in zip(s_names, s_p)},
        capital=str(c_names[ci]),
        capital_probs={str(c_names[i]): float(c_p[i]) for i in order},
    )


# ------------------------------------------------------------------ engines
class ClassicEngine:
    """tfidf.pkl + sentiment_lr_model.pkl + capital_lr_model.pkl (+ encoders)."""

    def __init__(self):
        self.clean, self.cleaner_info = make_cleaner()
        self.tfidf = as_vectorizer(load_pickle("tfidf.pkl"), "tfidf.pkl")
        self.m_sent = as_model(load_pickle("sentiment_lr_model.pkl"), "sentiment_lr_model.pkl")
        self.m_cap = as_model(load_pickle("capital_lr_model.pkl"), "capital_lr_model.pkl")
        self.e_sent = load_pickle("sentiment_encoder.pkl") if exists("sentiment_encoder.pkl") else None
        self.e_cap = load_pickle("capital_encoder.pkl") if exists("capital_encoder.pkl") else None
        # encoders stored inside a dict model are used if the separate file is missing
        self.e_sent = self.e_sent if self.e_sent is not None else getattr(self.m_sent, "encoder", None)
        self.e_cap = self.e_cap if self.e_cap is not None else getattr(self.m_cap, "encoder", None)

    def _features(self, model, parsed):
        if hasattr(model, "steps"):                       # Pipeline with its own vectorizer
            return [parsed]
        vec = getattr(model, "vectorizer", None) or self.tfidf
        return vec.transform([parsed])

    def predict(self, text: str) -> Prediction:
        parsed = self.clean(text)
        ps, cs = sklearn_proba(self.m_sent, self._features(self.m_sent, parsed))
        pc, cc = sklearn_proba(self.m_cap, self._features(self.m_cap, parsed))
        return _pack(ENGINE_CLASSIC, parsed,
                     labels_from(cs if cs is not None else range(len(ps)), self.e_sent), ps,
                     labels_from(cc if cc is not None else range(len(pc)), self.e_cap), pc)


class LSTMEngine:
    """lstm_tokenizer.pkl + lstm_params.pkl + sentiment/capital .h5 (+ LSTM encoders)."""

    def __init__(self, padding: str | None = None):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        from tensorflow.keras.utils import pad_sequences

        self._tf, self._pad = tf, pad_sequences
        self.clean, self.cleaner_info = make_cleaner()
        self.tok = load_pickle("lstm_tokenizer.pkl")
        self.params = load_pickle("lstm_params.pkl") if exists("lstm_params.pkl") else {}
        if not isinstance(self.params, dict):
            self.params = {"value": self.params}
        self.sent_file = next(n for n in SENT_LSTM_CANDIDATES if exists(n))
        self.m_sent = tf.keras.models.load_model(path_of(self.sent_file), compile=False)
        self.m_cap = tf.keras.models.load_model(path_of("best_capital_lstm.h5"), compile=False)
        self.e_sent = load_pickle("sentiment_lstm_encoder.pkl") if exists("sentiment_lstm_encoder.pkl") \
            else (load_pickle("sentiment_encoder.pkl") if exists("sentiment_encoder.pkl") else None)
        self.e_cap = load_pickle("capital_lstm_encoder.pkl") if exists("capital_lstm_encoder.pkl") \
            else (load_pickle("capital_encoder.pkl") if exists("capital_encoder.pkl") else None)

        p = {k.lower(): v for k, v in self.params.items()}
        self.padding = padding or p.get("padding", "post")
        self.truncating = p.get("truncating", self.padding)

    def _maxlen(self, model) -> int:
        n = model.input_shape[1] if isinstance(model.input_shape, tuple) else None
        if n:
            return int(n)
        p = {k.lower(): v for k, v in self.params.items()}
        for k in ("max_len", "maxlen", "max_length", "max_sequence_length", "sequence_length"):
            if k in p:
                return int(p[k])
        return 100

    def _seq(self, parsed, model):
        if hasattr(self.tok, "texts_to_sequences"):
            seq = self.tok.texts_to_sequences([parsed])
        else:                                    # dict word_index
            wi = getattr(self.tok, "word_index", self.tok)
            seq = [[wi.get(w, 1) for w in parsed.lower().split()]]
        return self._pad(seq, maxlen=self._maxlen(model), padding=self.padding, truncating=self.truncating)

    @staticmethod
    def _probs(out):
        out = np.asarray(out[0] if isinstance(out, (list, tuple)) else out)[0]
        out = np.atleast_1d(out)
        if out.size == 1:                        # sigmoid binary output
            return np.array([1 - out[0], out[0]])
        return out

    def predict(self, text: str, sent_model=None, engine_name=ENGINE_LSTM) -> Prediction:
        parsed = self.clean(text)
        m_sent = sent_model or self.m_sent
        ps = self._probs(m_sent.predict(self._seq(parsed, m_sent), verbose=0))
        pc = self._probs(self.m_cap.predict(self._seq(parsed, self.m_cap), verbose=0))
        return _pack(engine_name, parsed,
                     encoder_names(self.e_sent, len(ps)), ps,
                     encoder_names(self.e_cap, len(pc)), pc)


class FederatedEngine:
    """federated_sentiment_model.pkl = FedAvg global model.
    Supported formats: list of numpy weight arrays (Keras get_weights / Flower),
    a dict holding such a list, a pickled Keras model, or an sklearn classifier."""

    def __init__(self, lstm: LSTMEngine, classic: ClassicEngine | None):
        self.lstm, self.classic = lstm, classic
        obj = load_pickle("federated_sentiment_model.pkl")
        self.info = describe(obj)
        if isinstance(obj, dict):
            for k in ("weights", "parameters", "global_weights", "params", "model_weights"):
                if k in obj:
                    obj = obj[k]
                    break
        self.kind, self.model = None, None
        if isinstance(obj, (list, tuple)) and obj and all(hasattr(a, "shape") for a in obj):
            m = lstm._tf.keras.models.clone_model(lstm.m_sent)
            m.set_weights([np.asarray(a) for a in obj])
            self.kind, self.model = "keras-weights", m
        elif hasattr(obj, "predict") and hasattr(obj, "layers"):
            self.kind, self.model = "keras-model", obj
        elif isinstance(obj, dict) and classic is not None:
            self.kind, self.model = "sklearn", DictModel(obj, "federated_sentiment_model.pkl")
        elif hasattr(obj, "predict") and classic is not None:
            self.kind, self.model = "sklearn", obj
        else:
            raise ValueError(f"Unrecognised federated model format: {self.info}")

    def predict(self, text: str) -> Prediction:
        if self.kind in ("keras-weights", "keras-model"):
            return self.lstm.predict(text, sent_model=self.model, engine_name=ENGINE_FED)
        c = self.classic
        parsed = c.clean(text)
        ps, cs = sklearn_proba(self.model, c._features(self.model, parsed))
        pc, cc = sklearn_proba(c.m_cap, c._features(c.m_cap, parsed))
        return _pack(ENGINE_FED, parsed, labels_from(cs if cs is not None else range(len(ps)), c.e_sent), ps,
                     labels_from(cc if cc is not None else range(len(pc)), c.e_cap), pc)


# ------------------------------------------------------------------ availability
def required_files(engine: str) -> list[str]:
    if engine == ENGINE_CLASSIC:
        return ["tfidf.pkl", "sentiment_lr_model.pkl", "capital_lr_model.pkl"]
    if engine == ENGINE_LSTM:
        return ["lstm_tokenizer.pkl", "best_capital_lstm.h5"]
    return ["federated_sentiment_model.pkl", "lstm_tokenizer.pkl", "best_capital_lstm.h5"]


def missing(engine: str) -> list[str]:
    miss = [f for f in required_files(engine) if not exists(f)]
    if engine in (ENGINE_LSTM, ENGINE_FED) and not any(exists(n) for n in SENT_LSTM_CANDIDATES):
        miss.append(" or ".join(SENT_LSTM_CANDIDATES))
    return miss


def available_engines() -> list[str]:
    engines = [ENGINE_CLASSIC, ENGINE_LSTM]
    if exists("federated_sentiment_model.pkl"):
        engines.append(ENGINE_FED)
    return engines


def diagnostics() -> list[dict]:
    rows = []
    for n in ALL_FILES:
        p = path_of(n)
        row = {"file": n, "found": p.exists(),
               "size_MB": round(p.stat().st_size / 1e6, 2) if p.exists() else None, "content": ""}
        if p.exists() and n.endswith(".pkl"):
            try:
                row["content"] = describe(load_pickle(n))
            except Exception as e:
                row["content"] = f"LOAD ERROR: {type(e).__name__}: {e}"[:160]
        elif p.exists():
            row["content"] = "Keras model (.h5)"
        rows.append(row)
    return rows
