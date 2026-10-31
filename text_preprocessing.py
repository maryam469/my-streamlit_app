# ============================================================
# SHARED TEXT PREPROCESSING (LR + Linear SVC)
#
# Training aur prediction DONO isi function ko use karte hain.
# Pehle training ek cleaner use karti thi aur prediction doosra
# (NOT_ prefixes ke saath), is liye model ko prediction ke waqt
# aise words milte the jo usne training mein kabhi dekhe hi nahi.
#
# EMOJIS remove NAHI hote - har emoji ek alag token/feature banta hai
# (😊, 😞, 🚀 ...). Is ke liye TfidfVectorizer mein TOKEN_PATTERN dena zaroori hai.
# ============================================================
import re
import unicodedata

# Default sklearn pattern (2+ letter words) + koi bhi single emoji/symbol.
# Cleaner baaki saari punctuation hata deta hai, is liye [^\w\s] sirf emojis pakadta hai.
TOKEN_PATTERN = r"(?u)\b\w\w+\b|[^\w\s]"

NEGATION_WORDS = {
    "not", "no", "never", "non", "nor", "cannot", "cant", "couldnt", "wouldnt",
    "shouldnt", "mightnt", "mustnt", "dont", "doesnt", "isnt", "arent", "hasnt",
    "havent", "hadnt", "didnt", "wasnt", "werent", "wont", "aint", "neither",
    "nowhere", "nothing", "nobody", "none", "unable", "without", "hardly",
    "barely", "scarcely",
}

# Ye words / punctuation negation ka scope khatam kar dete hain
# e.g. "not good, but great" -> not_good but great
SCOPE_BREAKERS = {"but", "however", "although", "though", "yet", "still"}
_PUNCT_BREAK = re.compile(r"[.,!?;:]")

# "didn t", "don t" (dataset mein apostrophe hata hua hai) -> "didnt", "dont"
_SPLIT_CONTRACTION = re.compile(r"\b(\w+)n\s+t\b")
# "can not" -> "cannot"
_CAN_NOT = re.compile(r"\bcan\s+not\b")
_MOJIBAKE_HINT = re.compile("[ÃÂâð][\u0080-ÿŒ-™]")

# Emoji ke saath aane wale invisible/modifier characters (variation selector,
# zero-width joiner, skin tones) - inhe hata kar base emoji rakhte hain
_EMOJI_JUNK = {"️", "︎", "‍", "⃣"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def _mojibake_bytes(text):
    out = bytearray()
    for ch in text:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) < 256:
                out.append(ord(ch))
            else:
                return None
    return bytes(out)


def fix_mojibake(text):
    """Dataset ke kharab emojis wapas asli banata hai: 'Ã°Å¸â€™Â¸' -> '💸'.

    CSV mein text UTF-8 -> cp1252 -> UTF-8 mein do baar kharab hua hai,
    is liye repair tab tak chalta hai jab tak text badalna band na ho.
    """
    for _ in range(3):
        if not _MOJIBAKE_HINT.search(text):
            break
        raw = _mojibake_bytes(text)
        if raw is None:
            break
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            break
    return text


def _is_emoji(ch):
    return unicodedata.category(ch) == "So"


is_emoji = _is_emoji


def separate_emojis(text):
    """Mojibake repair + har emoji ke gird space; skin tone / variation selector hatao.
    (LSTM / federated cleaners ke liye, jo apni baaki cleaning khud karte hain.)"""
    out = []
    for ch in fix_mojibake(str(text)).translate(_QUOTES):
        if ch in _EMOJI_JUNK:
            continue
        out.append(f" {ch} " if _is_emoji(ch) else ch)
    return "".join(out)


def strip_emojis(text):
    """Model input ke liye: sirf un models ke liye jo emojis par train nahi hue."""
    return " ".join(w for w in str(text).split() if not (len(w) == 1 and _is_emoji(w)))


def _normalize(text):
    text = fix_mojibake(str(text)).translate(_QUOTES).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("'", "")
    text = _SPLIT_CONTRACTION.sub(r"\1nt", text)
    text = _CAN_NOT.sub("cannot", text)

    chars = []
    for ch in text:
        if ch in _EMOJI_JUNK:
            continue
        if _is_emoji(ch):
            chars.append(f" {ch} ")          # har emoji alag token
        elif ch in ".,!?;:":
            chars.append(f" {ch} ")          # negation scope breaker
        elif ch.isascii() and (ch.isalnum() or ch.isspace()):
            chars.append(ch)
        elif ch.isalpha():
            # é -> e, ü -> u
            base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
            chars.append(base or " ")
        else:
            chars.append(" ")                # baaki symbols / accents hatao
    return "".join(chars).split()


def clean_text_with_count(text):
    """Returns (cleaned_text, negation_count).

    Negation word ke baad wale words ko 'not_' prefix milta hai jab tak
    punctuation ya 'but' jaisa word na aa jaye. Do negations (even) aapas
    mein cancel ho jati hain: "not not good" -> "good".
    Emojis waise hi rehte hain (unka apna sentiment hota hai).
    """
    if text is None or (isinstance(text, float) and text != text):  # NaN
        return "", 0

    out = []
    total_negations = 0
    negate = False
    for word in _normalize(text):
        if _PUNCT_BREAK.fullmatch(word) or word in SCOPE_BREAKERS:
            negate = False
            if word in SCOPE_BREAKERS:
                out.append(word)
            continue
        if word in NEGATION_WORDS:
            negate = not negate
            total_negations += 1
            continue
        if len(word) == 1 and _is_emoji(word):
            out.append(word)
            continue
        out.append(f"not_{word}" if negate else word)
    return " ".join(out), total_negations


def clean_text(text):
    return clean_text_with_count(text)[0]
