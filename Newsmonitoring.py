import re
import os
import json
import calendar
from datetime import date, timedelta, datetime
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd


import requests
from dateutil import parser as dtparser


# NORMALISE

def normalize_title_key(t: str) -> str:
    t = t.lower()
    t = re.sub(r"['’]", "", t)
    t = re.sub(r"[^a-z0-9à-öø-ÿ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t



# STOPWORDS NLTK
#
try:
    _ = stopwords.words("italian")
except LookupError:
    nltk.download("stopwords")

try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOP_IT = set(stopwords.words("italian"))
STOP_EN = set(stopwords.words("english"))


# SPACY (EN + IT) optional
#
try:
    import spacy
except Exception as e:
    spacy = None
    print("[WARN] spaCy not installed/available:", e)

nlp_en = None
nlp_it = None

if spacy is not None:
    try:
        nlp_en = spacy.load("en_core_web_sm")
        print("[INFO] spaCy English model loaded.")
    except Exception as e:
        nlp_en = None
        print("[WARN] spaCy English model NOT loaded. Install with: python -m spacy download en_core_web_sm", e)

    try:
        nlp_it = spacy.load("it_core_news_sm")
        print("[INFO] spaCy Italian model loaded.")
    except Exception as e:
        nlp_it = None
        print("[WARN] spaCy Italian model NOT loaded. Install with: python -m spacy download it_core_news_sm", e)



# RSS FEEDS
COUNTRY_FEEDS = {
    "italy": [
        "https://www.repubblica.it/rss/homepage/rss2.0.xml",
        "https://www.rainews.it/rss/esteri",
        "https://www.ilsole24ore.com/rss/italia.xml",
        "https://www.ilsole24ore.com/rss/mondo.xml",
        "https://www.rainews.it/rss/politica",
        "https://www.rainews.it/rss/ultimora",
        "https://www.agi.it/politica/rss",
        "https://www.agi.it/estero/rss",
        "https://www.adnkronos.com/RSS_Politica.xml",
        "https://www.adnkronos.com/RSS_Esteri.xml",
    ],
    "uk": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://feeds.skynews.com/feeds/rss/home.xml",
        "https://www.theguardian.com/uk-news/rss",
        "https://www.theguardian.com/world/rss",
        "https://www.independent.co.uk/news/uk/rss",
        "https://www.ft.com/rss/home/international",
        "https://feeds.bbci.co.uk/news/world/rss.xml?edition=uk",
        "https://feeds.skynews.com/feeds/rss/world.xml",
        "https://feeds.skynews.com/feeds/rss/uk.xml",
    ],
}

COUNTRY_TZ = {
    "italy": ZoneInfo("Europe/Rome"),
    "uk": ZoneInfo("Europe/London"),
}

# Anchor everything to one day
ANCHOR_TZ = ZoneInfo("Europe/Rome")



# FETCH + DATE PARSING

def fetch_feed(url: str):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        return feedparser.parse(r.content)
    except Exception as e:
        print(f"[WARN] requests fetch failed for {url}: {e}. Falling back to feedparser.parse(url).")
        return feedparser.parse(url)


def parse_entry_datetime(entry, fallback_tz: ZoneInfo):

    for k in ("published", "updated"):
        s = entry.get(k)
        if s:
            try:
                dt = dtparser.parse(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=fallback_tz)
                return dt
            except Exception:
                pass

    dt_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if dt_struct:
        try:
            naive = datetime(*dt_struct[:6])
            return naive.replace(tzinfo=fallback_tz)
        except Exception:
            return None

    return None


def fetch_titles(feeds, feed_tz: ZoneInfo, anchor_tz: ZoneInfo):

    #Fetch titles whose date == "today" in anchor_tz.


    today_anchor = datetime.now(anchor_tz).date()
    titles: list[str] = []

    for url in feeds:
        feed = fetch_feed(url)

        if getattr(feed, "bozo", 0):
            ex = getattr(feed, "bozo_exception", None)
            print(f"[WARN] Feed parse issue for {url}: {ex}")

        items: list[tuple[datetime, str]] = []

        for entry in getattr(feed, "entries", []):
            title = (entry.get("title", "") or "").strip()
            if not title:
                continue

            dt = parse_entry_datetime(entry, fallback_tz=feed_tz)
            if not dt:
                continue

            dt_anchor = dt.astimezone(anchor_tz)
            if dt_anchor.date() == today_anchor:
                items.append((dt_anchor, title))


        seen = set()
        deduped: list[tuple[datetime, str]] = []
        for dt, title in items:
            k = normalize_title_key(title)
            if k in seen:
                continue
            seen.add(k)
            deduped.append((dt, title))
        items = deduped


        for _, title in items:
            titles.append(title)

    return titles


# STOPWORDS CUSTOM

COMMON_STOPWORDS = {
    # Italiani generici
    "oggi", "ieri", "due", "tre", "solo", "ancora",
    "dopo", "prima", "contro", "secondo",
    "di", "per", "del", "della", "delle", "degli", "dei",
    "il", "la", "lo", "i", "gli",
    "un", "una", "uno", "nel", "nella", "nelle", "negli",
    "e", "ed", "le", "è", "ecco",
    "poi", "senza", "mai", "anni",
    "nuova", "tutto", "cosa", "casa", "nuovo", "nuova", "nuove", "nuovi", "ora",
    "repubblica", "ansa", "rai", "rainews", "notizie", "italia", "italiani",

    # Italian possessives
    "mio", "mia", "miei", "mie",
    "tuo", "tua", "tuoi", "tue",
    "suo", "sua", "suoi", "sue",
    "nostro", "nostra", "nostri", "nostre",
    "vostro", "vostra", "vostri", "vostre",
    "loro",

    # English determiners/possessives
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",

    # English generici
    "today", "yesterday", "two", "three", "only", "again",
    "after", "before", "against", "according",
    "new",

    # Rumore
    "img", "https", "con", "alt", "video", "foto", "corriere", "images2",
}

EXTRA_STOPWORDS_UK = {
    "uk", "britain", "english", "england",
    "say", "says", "said",
    "best", "year", "years", "week", "weeks",
    "another", "many", "much", "big", "small",
    "newest", "latest", "live", "update", "updates", "breaking",
}

EXTRA_STOPWORDS_NEWS_UK = {
    "exclusive", "coverage", "report", "reports",
    "story", "storys", "analysis", "review", "reviews", "recap",
    "day", "days", "month", "months", "season", "seasons", "era", "eras",
    "huge", "major", "massive", "super", "extra", "ultra",
    "extreme", "incredible", "unbelievable", "remarkable", "dramatic",
    "shocking", "surprising", "unexpected", "outrageous", "disgusting",
    "better", "worst", "upper", "lower", "high", "higher", "highest",
    "low", "lowest", "level", "levels", "rank", "ranking", "rankings",
    "few", "several", "various", "numerous", "multiple", "others",
    "other", "else", "different", "difference", "differences",
    "behind", "front", "back", "ahead", "close", "near", "nearby",
    "across", "along", "around", "inside", "outside", "under", "over",
    "through", "between", "beyond", "beneath", "beside", "below", "above",
    "up", "down","british",
    "start", "starts", "started", "finish", "finishes", "ending", "ends",
    "continue", "continues", "continued", "complete", "completed",
    "move", "moves", "moving", "go", "goes", "going", "gone",
    "come", "comes", "coming", "return", "returns", "returned",
    "enter", "enters", "leave", "leaves", "left",
    "claim", "claims", "claimed",
    "think", "thinks", "thought",
    "believe", "believes", "believed",
    "reveal", "reveals", "revealed", "revealings",
    "show", "shows", "showed", "seen", "view", "views", "viewed",
    "person", "people", "individual", "group", "groups", "team", "teams",
    "member", "members", "official", "officials", "staff", "worker",
    "workers", "crowd", "fans", "as",
    "lot", "lots", "amount", "amounts", "number", "numbers",
    "figure", "figures", "percent", "percentage", "percentages",
    "portion", "portions",
    "possible", "possibly", "probable", "probably", "likely", "unlikely",
    "expected", "unexpected", "maybe", "perhaps", "might", "could",
    "would", "should", "can", "cannot",
    "issue", "issues", "problem", "problems", "matter", "matters",
    "case", "cases", "part", "parts", "aspect", "aspects", "factor",
    "factors", "situation", "situations",
    "current", "past", "present", "future", "recent", "recently",
    "early", "earlier", "late", "later", "long", "longer", "short",
    "shorter", "shortly",
    "good", "great", "bad", "poor", "nice", "awful", "terrible",
    "horrible", "serious", "important", "crucial", "critical", "significant", "key",
    "star", "stars", "celebrity", "celebrities", "public", "social", "publicly",
    "brand", "brands", "company", "companies", "business", "market",
    "markets", "stock", "stocks", "project", "projects",
    "however", "though", "although", "despite", "meanwhile", "instead",
    "therefore", "moreover", "furthermore", "besides", "nonetheless",
    "otherwise",
    "make", "makes", "made", "doing", "do", "does", "done", "take",
    "takes", "took", "give", "gives", "gave", "work", "works", "worked",
    "end", "world", "home", "still", "first", "man", "woman",
}

STOPWORDS_BY_COUNTRY = {
    "italy": STOP_IT | COMMON_STOPWORDS,
    "uk": STOP_EN | COMMON_STOPWORDS | EXTRA_STOPWORDS_UK | EXTRA_STOPWORDS_NEWS_UK,
}


#  TOKEN + TOPIC (PHRASE) EXTRACTION

def tokenize_term_pairs(text, stopwords_set, nlp=None, only_nouns_propn=True):
    out = []
    if nlp is not None:
        doc = nlp(text)
        allowed_pos = {"NOUN", "PROPN"} if only_nouns_propn else {"NOUN", "PROPN", "ADJ"}

        for t in doc:
            if t.is_space or t.is_punct:
                continue
            if t.pos_ not in allowed_pos:
                continue

            lemma = (t.lemma_ or t.text).strip()
            surface = (t.text or "").strip()
            if not lemma:
                continue

            key = lemma.lower()
            if len(key) < 3:
                continue
            if not re.search(r"[a-zà-öø-ÿ]", key, flags=re.IGNORECASE):
                continue
            if key in stopwords_set:
                continue

            out.append((key, surface if surface else lemma))
        return out


    tokens_raw = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)
    for w in tokens_raw:
        key = w.lower()
        if len(key) < 3:
            continue
        if key in stopwords_set:
            continue
        out.append((key, w))
    return out


def _clean_phrase(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^[\"'’\(\[\{]+|[\"'’\)\]\}]+$", "", s).strip()
    return s


def _words_alpha(s: str):
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", s)


def _is_stopword_phrase(phrase_key: str, stopwords_set: set[str]) -> bool:
    parts = _words_alpha(phrase_key.lower())
    return bool(parts) and all(p in stopwords_set for p in parts)


def _phrase_key_from_surface(surface: str) -> str:
    s = surface.lower()
    s = re.sub(r"['’]", "", s)
    s = re.sub(r"[^a-z0-9à-öø-ÿ\s]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s



EN_LEADING_WORDS = [
    "the", "a", "an",
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
]

EN_HEDGE_PREFIXES = [
    "alleged", "reportedly", "suspected", "possible", "potential",
    "apparent", "so called", "so-called",
]

_EN_LEADING_RE = re.compile(
    r"^(?:(?:%s)\s+)+" % "|".join(re.escape(w) for w in EN_LEADING_WORDS),
    flags=re.IGNORECASE
)
_EN_HEDGE_RE = re.compile(
    r"^(?:(?:%s)\s+)+" % "|".join(re.escape(w) for w in sorted(EN_HEDGE_PREFIXES, key=len, reverse=True)),
    flags=re.IGNORECASE
)


def strip_en_leading_function_words(phrase: str) -> str:
    s = phrase.strip()
    s = _EN_HEDGE_RE.sub("", s).strip()
    s = _EN_LEADING_RE.sub("", s).strip()
    return s



ITALIAN_LEADING_CLITICS = [
    "l'", "l’", "un'", "un’",
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una",
    "del", "dello", "della", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "sul", "sullo", "sulla", "sui", "sugli", "sulle",
    "col", "coi",
    "da", "di", "a", "in", "su", "con", "per", "tra", "fra",
]

ITALIAN_POSSESSIVES = [
    "mio", "mia", "miei", "mie",
    "tuo", "tua", "tuoi", "tue",
    "suo", "sua", "suoi", "sue",
    "nostro", "nostra", "nostri", "nostre",
    "vostro", "vostra", "vostri", "vostre",
    "loro",
]

_IT_LEADING_RE = re.compile(
    r"^(?:(?:%s)\s+)+" % "|".join(re.escape(x.replace("’", "'")) for x in ITALIAN_LEADING_CLITICS),
    flags=re.IGNORECASE
)
_IT_LEADING_APOS_RE = re.compile(r"^(?:l['’]|un['’])", flags=re.IGNORECASE)
_IT_POSSESSIVE_RE = re.compile(
    r"^(?:(?:%s)\s+)+" % "|".join(re.escape(x) for x in ITALIAN_POSSESSIVES),
    flags=re.IGNORECASE
)


def strip_it_leading_function_words(phrase: str) -> str:
    s = phrase.strip()
    s = s.replace("’", "'")
    s = _IT_LEADING_APOS_RE.sub("", s).strip()
    s = _IT_LEADING_RE.sub("", s).strip()
    s = _IT_POSSESSIVE_RE.sub("", s).strip()
    return s


# NEW: phrase quality scoring

def phrase_tokens_info(span, stopwords_set: set[str]):


    words = []
    content_words = []
    has_propn = False

    for t in span:
        if t.is_space or t.is_punct:
            continue
        if not t.text:
            continue

        w = t.text.lower()
        if not re.search(r"[a-zà-öø-ÿ]", w, flags=re.IGNORECASE):
            continue

        w = w.replace("’", "'")
        w = re.sub(r"^[\"'’\(\[\{]+|[\"'’\)\]\}]+$", "", w).strip()
        if not w:
            continue

        words.append(w)
        if t.pos_ == "PROPN":
            has_propn = True
        if len(w) >= 3 and w not in stopwords_set:
            content_words.append(w)

    return words, content_words, has_propn


def chunk_overlaps_entity(chunk, doc, allowed_ent_labels: set[str]) -> bool:

    for ent in doc.ents:
        if ent.label_ in allowed_ent_labels and chunk.start < ent.end and ent.start < chunk.end:
            return True
    return False


def extract_phrase_pairs_from_doc(
    doc,
    stopwords_set,
    allowed_ent_labels=None,
    use_noun_chunks=True,
    lang="en",
    max_chunk_words_by_lang=None,
):

    if allowed_ent_labels is None:
        allowed_ent_labels = {"PERSON", "ORG", "GPE", "LOC", "NORP"}

    if max_chunk_words_by_lang is None:

        max_chunk_words_by_lang = {"en": 6, "it": 6}

    max_words = max_chunk_words_by_lang.get(lang, 6)

    out = []

    def clean_disp(disp: str) -> str:
        disp = _clean_phrase(disp)
        if lang == "it":
            disp = strip_it_leading_function_words(disp)
        else:
            disp = strip_en_leading_function_words(disp)
        disp = _clean_phrase(disp)
        return disp


    for ent in doc.ents:
        if ent.label_ not in allowed_ent_labels:
            continue

        disp = clean_disp(ent.text)
        if len(disp) < 3:
            continue

        key = _phrase_key_from_surface(disp)
        if len(key) < 3:
            continue
        if _is_stopword_phrase(key, stopwords_set):
            continue

        out.append((key, disp))


    if use_noun_chunks and hasattr(doc, "noun_chunks"):
        for chunk in doc.noun_chunks:
            disp = clean_disp(chunk.text)

            if " " not in disp:
                continue
            if len(disp) < 5:
                continue

            key = _phrase_key_from_surface(disp)
            if len(key) < 5 or " " not in key:
                continue
            if _is_stopword_phrase(key, stopwords_set):
                continue

            words, content_words, has_propn = phrase_tokens_info(chunk, stopwords_set)
            n_words = len(words)

            if n_words < 2 or n_words > max_words:
                continue

            overlaps_ent = chunk_overlaps_entity(chunk, doc, allowed_ent_labels)


            if not (has_propn or overlaps_ent):
                if len(content_words) < 2:
                    continue

            out.append((key, disp))

    return out


# COUNTING (DF) + TOPICS

def get_document_frequency_counts_topics(
    titles: list[str],
    stopwords_set,
    nlp=None,
    only_nouns_propn=True,
    include_phrases=True,
    phrase_min_df=2,
    suppress_unigrams_in_phrases=True,
    lang="en",
    global_suppress_unigrams_inside_phrases=True,
):

    df_uni = Counter()
    df_phrase = Counter()
    display_votes = defaultdict(Counter)

    if nlp is None:

        for title in titles:
            token_pairs = tokenize_term_pairs(title, stopwords_set, nlp=None, only_nouns_propn=only_nouns_propn)
            keys_in_title = set()
            for k, disp in token_pairs:
                display_votes[k][disp] += 1
                keys_in_title.add(k)
            for k in keys_in_title:
                df_uni[k] += 1

        display_map = {k: c.most_common(1)[0][0] for k, c in display_votes.items()}
        final_counts = dict(df_uni)
        return final_counts, display_map

    docs = list(nlp.pipe(titles))

    for title, doc in zip(titles, docs):
        token_pairs = tokenize_term_pairs(
            title,
            stopwords_set=stopwords_set,
            nlp=nlp,
            only_nouns_propn=only_nouns_propn,
        ) or []

        phrase_pairs = []
        if include_phrases:
            phrase_pairs = extract_phrase_pairs_from_doc(
                doc,
                stopwords_set=stopwords_set,
                lang=lang,
                use_noun_chunks=True,
                # tweak if you want different caps:
                max_chunk_words_by_lang={"en": 6, "it": 7},
            ) or []

        for k, disp in token_pairs:
            if disp:
                display_votes[k][disp] += 1
        for k, disp in phrase_pairs:
            if disp:
                display_votes[k][disp] += 1

        unigram_keys = {k for k, _ in token_pairs}
        phrase_keys = {k for k, _ in phrase_pairs}

        for pk in phrase_keys:
            df_phrase[pk] += 1

        if suppress_unigrams_in_phrases and phrase_keys:
            phrase_words = set()
            for pk in phrase_keys:
                phrase_words.update(pk.split())
            unigram_keys = {k for k in unigram_keys if k not in phrase_words}

        for uk in unigram_keys:
            df_uni[uk] += 1

    if phrase_min_df and phrase_min_df > 1:
        df_phrase = Counter({k: v for k, v in df_phrase.items() if v >= phrase_min_df})

    # Global suppression of unigrams that are inside any kept phrase
    if global_suppress_unigrams_inside_phrases and df_phrase:
        phrase_words_global = set()
        for pk in df_phrase.keys():
            phrase_words_global.update(pk.split())
        for w in list(df_uni.keys()):
            if w in phrase_words_global:
                del df_uni[w]

    display_map = {}
    for k, c in display_votes.items():
        best = sorted(c.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))[0][0]
        display_map[k] = best

    final_counts = dict(df_uni)
    final_counts.update(df_phrase)
    return final_counts, display_map


#  WORDCLOUD

def make_wordcloud_from_term_counts(term_counts, save_path=None):
    wc = WordCloud(
        width=1600,
        height=800,
        background_color="black",
        colormap="rainbow",
        contour_width=3,
        contour_color="white",
        collocations=False,
        max_words=350,
        prefer_horizontal=0.8,
        relative_scaling=1.0,
        random_state=42,
    ).generate_from_frequencies(term_counts)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.close()


# FILE SYSTEM

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)


def get_counts_filename(day: date, country: str) -> str:
    return str(DATA_DIR / f"{day.isoformat()}_{country}_terms.json")


def save_counts_for_day(day: date, country: str, counts: dict):
    filename = get_counts_filename(day, country)
    fullpath = os.path.abspath(filename)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Saved term/topic counts for {country} to {fullpath}")


def load_counts_for_day(day: date, country: str):
    filename = get_counts_filename(day, country)
    if not os.path.exists(filename):
        return None
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def get_wordcloud_filename_for_day(day: date, country: str) -> Path:
    return DATA_DIR / f"{day.isoformat()}_{country}_wordcloud.png"


#  TABLES: NEW / RISING / FALLING

def get_new_words_table(today_counts: dict, yesterday_counts: dict, top_n=10):
    new_terms = set(today_counts.keys()) - set(yesterday_counts.keys())
    rows = [{"Term": t, "Current day frequency": today_counts[t]} for t in new_terms]
    if not rows:
        return pd.DataFrame(columns=["Term", "Current day frequency"])
    return pd.DataFrame(rows).sort_values("Current day frequency", ascending=False).head(top_n)


def get_rising_words_table(today_counts: dict, yesterday_counts: dict, top_n=10):
    rows = []
    for t in set(today_counts.keys()) & set(yesterday_counts.keys()):
        d = today_counts[t] - yesterday_counts.get(t, 0)
        if d > 0:
            rows.append({
                "Term": t,
                "Difference": d,
                "Current day frequency": today_counts[t],
                "Previous day frequency": yesterday_counts.get(t, 0),
            })
    if not rows:
        return pd.DataFrame(columns=["Term", "Difference", "Current day frequency", "Previous day frequency"])
    return pd.DataFrame(rows).sort_values("Difference", ascending=False).head(top_n)


def get_falling_words_table(today_counts: dict, yesterday_counts: dict, top_n=10):
    rows = []
    for t in set(today_counts.keys()) & set(yesterday_counts.keys()):
        d = today_counts[t] - yesterday_counts.get(t, 0)
        if d < 0:
            rows.append({
                "Term": t,
                "Difference": d,
                "Current day frequency": today_counts[t],
                "Previous day frequency": yesterday_counts.get(t, 0),
            })
    if not rows:
        return pd.DataFrame(columns=["Term", "Difference", "Current day frequency", "Previous day frequency"])
    return pd.DataFrame(rows).sort_values("Difference", ascending=True).head(top_n)


def compare_days(today_counts: dict, yesterday_counts: dict, top_n=10):
    df_new = get_new_words_table(today_counts, yesterday_counts, top_n=top_n)
    df_rising = get_rising_words_table(today_counts, yesterday_counts, top_n=top_n)
    df_falling = get_falling_words_table(today_counts, yesterday_counts, top_n=top_n)
    return df_new, df_rising, df_falling


def get_top_terms_table(counts: dict, top_n=10, kind="all"):
    """
    kind:
      - 'unigram': single tokens (no space)
      - 'phrase': multiword topics (has space)
      - 'all'
    """
    if not counts:
        return pd.DataFrame(columns=["Term", "Frequency"])

    items = counts.items()
    if kind == "unigram":
        items = [(t, f) for t, f in items if " " not in t]
    elif kind == "phrase":
        items = [(t, f) for t, f in items if " " in t]

    rows = [{"Term": t, "Frequency": f} for t, f in Counter(dict(items)).most_common(top_n)]
    return pd.DataFrame(rows)


#  WEEK VIEW DASHBOARD (index.html)

def list_available_days(data_dir: Path, country: str):
    suffix = f"_{country}_terms.json"
    days = []
    for p in data_dir.glob(f"*{suffix}"):
        try:
            day_str = p.name.split("_")[0]
            days.append(date.fromisoformat(day_str))
        except Exception:
            continue
    return sorted(set(days))


def build_per_country_results_for_day(day: date, countries: list[str]):
    per_country_results = {}
    for country in countries:
        today_counts = load_counts_for_day(day, country)
        if today_counts is None:
            per_country_results[country] = None
            continue

        top_uni_df = get_top_terms_table(today_counts, top_n=10, kind="unigram")
        top_phrase_df = get_top_terms_table(today_counts, top_n=10, kind="phrase")

        prev_counts = load_counts_for_day(day - timedelta(days=1), country)
        new_df = rising_df = falling_df = None
        if prev_counts is not None:
            new_df, rising_df, falling_df = compare_days(today_counts, prev_counts, top_n=10)

        wc_path = get_wordcloud_filename_for_day(day, country)
        wc_rel = os.path.join("data", wc_path.name)

        per_country_results[country] = {
            "top_uni_df": top_uni_df,
            "top_phrase_df": top_phrase_df,
            "new_df": new_df,
            "rising_df": rising_df,
            "falling_df": falling_df,
            "wc_rel_path": wc_rel,
        }
    return per_country_results


def save_week_dashboard_html(last_day: date, countries: list[str], days_back: int = 7, out_filename: str = "index.html"):
    available_days = set()
    for c in countries:
        available_days.update(list_available_days(DATA_DIR, c))
    days = [d for d in sorted(available_days, reverse=True) if d <= last_day][-days_back:]

    if not days:
        print("[WARN] No saved days found in data/. Cannot build dashboard.")
        return

    generated_str = datetime.now(ANCHOR_TZ).strftime("%d %b %Y, %H:%M %Z")

    def df_or_message(df, msg):
        if df is None or df.empty:
            return f"<p class='empty-msg'>{msg}</p>"
        return df.to_html(index=False, classes=["data-table"])

    day_buttons_html = "".join(
        f'<button class="country-btn day-btn" data-day="{d.isoformat()}">{d.strftime("%d %b %Y")}</button>'
        for d in days
    )

    def country_label(c: str) -> str:
        if c == "italy":
            return "🇮🇹 Italy"
        if c == "uk":
            return "🇬🇧 UK"
        return c.capitalize()

    country_buttons_html = "".join(
        f'<button class="country-btn country-only-btn" data-country="{c}">{country_label(c)}</button>'
        for c in countries
    )

    panels_html = []
    for d in days:
        per_country = build_per_country_results_for_day(d, countries)
        for country in countries:
            data = per_country.get(country)
            flag = "🇮🇹" if country == "italy" else ("🇬🇧" if country == "uk" else "🏳️")

            if data is None:
                panel = f"""
                <div class="country-dashboard panel" id="panel-{d.isoformat()}-{country}" data-day="{d.isoformat()}" data-country="{country}" style="display:none;">
                    <div class="header-row">
                        <h2>{flag} {"UK" if country == "uk" else country.capitalize()} — {d.isoformat()}</h2>
                        <span class="date-pill">Updated: {generated_str}</span>
                    </div>
                    <div class="card">
                        <h3>No data</h3>
                        <p class='empty-msg'>No saved data found for this day/country combination.</p>
                    </div>
                </div>
                """
                panels_html.append(panel)
                continue

            html_top_uni = df_or_message(data["top_uni_df"], "No unigrams found for this day.")
            html_top_phrase = df_or_message(data["top_phrase_df"], "No phrases found for this day.")
            html_new = df_or_message(data["new_df"], "No new terms compared to previous day.")
            html_rising = df_or_message(data["rising_df"], "No significantly increasing terms.")
            html_falling = df_or_message(data["falling_df"], "No significantly decreasing terms.")

            panel = f"""
            <div class="country-dashboard panel" id="panel-{d.isoformat()}-{country}" data-day="{d.isoformat()}" data-country="{country}" style="display:none;">
                <div class="header-row">
                    <h2>{flag} {"UK" if country == "uk" else country.capitalize()} — {d.strftime("%d %b %Y")}</h2>
                    <span class="date-pill">Updated: {generated_str}</span>
                </div>

                <div class="content-grid">
                    <div class="card">
                        <h3>Wordcloud</h3>
                        <img src="{data["wc_rel_path"]}" alt="Wordcloud {country} {d.isoformat()}" class="wordcloud">
                    </div>

                    <div class="card tables-card">
                        <h3>Top 10 unigrams</h3>
                        {html_top_uni}

                        <h3>Top 10 phrases</h3>
                        {html_top_phrase}

                        <h3>Top 10 new terms (vs previous day)</h3>
                        {html_new}
                        <h3>Top 10 increasing terms (vs previous day)</h3>
                        {html_rising}
                        <h3>Top 10 decreasing terms (vs previous day)</h3>
                        {html_falling}
                    </div>
                </div>
            </div>
            """
            panels_html.append(panel)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>European News Dashboard - {generated_str}</title>
    <style>
        :root {{
            --bg-main: #05070a;
            --bg-panel: #11141a;
            --bg-card: #151924;
            --accent: #4fd1c5;
            --accent-soft: rgba(79, 209, 197, 0.12);
            --text-main: #f5f5f5;
            --text-muted: #a0aec0;
            --border-subtle: #2d3748;
            --shadow-soft: 0 18px 40px rgba(0, 0, 0, 0.6);
            --font-mono: "SF Mono", "JetBrains Mono", Menlo, monospace;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0;
            padding: 32px;
            background-color: var(--bg-main);
            color: var(--text-main);
            display: flex;
            gap: 24px;
        }}
        .sidebar {{
            width: 240px;
            padding: 20px 18px;
            background: radial-gradient(circle at top, #1a202c 0, #05070a 55%);
            border-radius: 16px;
            border: 1px solid var(--border-subtle);
            box-shadow: var(--shadow-soft);
            position: sticky;
            top: 24px;
            height: fit-content;
        }}
        .sidebar h2 {{
            color: #edf2f7;
            margin: 0 0 12px;
            font-size: 1.1rem;
        }}
        .sidebar p {{
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 10px;
        }}
        .country-btn {{
            display: block;
            width: 100%;
            margin-bottom: 8px;
            padding: 8px 10px;
            background: transparent;
            color: var(--text-main);
            border-radius: 999px;
            border: 1px solid var(--border-subtle);
            cursor: pointer;
            text-align: left;
            font-size: 0.95rem;
            transition: background 0.15s ease, border-color 0.15s ease, transform 0.05s ease;
        }}
        .country-btn:hover {{
            background: rgba(255, 255, 255, 0.05);
            transform: translateY(-1px);
        }}
        .country-btn.active {{
            background: var(--accent-soft);
            border-color: var(--accent);
        }}
        .container {{
            flex: 1;
            max-width: 1100px;
            background: radial-gradient(circle at top left, #1a202c 0, #05070a 55%);
            padding: 24px 28px;
            border-radius: 18px;
            border: 1px solid var(--border-subtle);
            box-shadow: var(--shadow-soft);
        }}
        h1 {{ margin: 0 0 6px; font-size: 1.4rem; }}
        .subtitle {{ margin: 0 0 4px; font-size: 0.9rem; color: var(--text-muted); }}
        .updated-info {{ margin: 0 0 18px; font-size: 0.8rem; color: var(--text-muted); }}
        .header-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
        }}
        .header-row h2 {{ margin: 0; font-size: 1.2rem; }}
        .date-pill {{
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid var(--border-subtle);
            font-size: 0.78rem;
            color: var(--text-muted);
        }}
        @media (max-width: 960px) {{
            body {{ flex-direction: column; }}
            .sidebar {{
                position: static;
                width: 100%;
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 8px;
            }}
            .sidebar h2 {{ flex: 1 0 100%; }}
            .container {{ width: 100%; }}
        }}
        .card {{
            background: var(--bg-card);
            border-radius: 14px;
            border: 1px solid #202637;
            padding: 14px 16px;
            margin-bottom: 14px;
        }}
        .card h3 {{ margin-top: 0; margin-bottom: 10px; font-size: 1rem; }}
        .wordcloud {{
            width: 100%;
            height: auto;
            border-radius: 10px;
            border: 1px solid #2d3748;
            display: block;
        }}
        .tables-card h3 {{ margin-top: 12px; font-size: 0.98rem; }}
        .tables-card h3:first-of-type {{ margin-top: 0; }}
        .empty-msg {{ font-size: 0.85rem; color: var(--text-muted); margin: 4px 0 12px; }}
        table.data-table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 6px;
            margin-bottom: 10px;
            font-size: 0.85rem;
        }}
        table.data-table th,
        table.data-table td {{
            border: 1px solid #2d3748;
            padding: 6px 8px;
            text-align: left;
        }}
        table.data-table th {{
            background-color: #111827;
            color: #e2e8f0;
            font-weight: 500;
            font-size: 0.82rem;
        }}
        table.data-table tr:nth-child(even) {{ background-color: #0b1220; }}
        table.data-table tr:nth-child(odd)  {{ background-color: #050b18; }}
        table.data-table td:nth-child(2),
        table.data-table td:nth-child(3),
        table.data-table td:nth-child(4) {{
            font-family: "SF Mono", "JetBrains Mono", Menlo, monospace;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="sidebar">
  <h2>Select Country</h2>
  <p>Pick a country:</p>
  {country_buttons_html}

  <h2 style="margin-top:16px;">Select Date</h2>
  <p>Pick a date (up to the last 14 days):</p>
  {day_buttons_html}
</div>
    <div class="container">
        <h1>European News Dashboard</h1>
        <p class="subtitle">Daily view</p>
        <p class="updated-info">Last update: {generated_str}</p>
        {''.join(panels_html)}
    </div>

    <script>
        const panels = document.querySelectorAll('.panel');
        const dayButtons = document.querySelectorAll('.day-btn');
        const countryButtons = document.querySelectorAll('.country-only-btn');

        let selectedDay = null;
        let selectedCountry = null;

        function showPanel(day, country) {{
            panels.forEach(p => {{
                const ok = (p.getAttribute('data-day') === day) && (p.getAttribute('data-country') === country);
                p.style.display = ok ? 'block' : 'none';
            }});

            dayButtons.forEach(b => {{
                b.classList.toggle('active', b.getAttribute('data-day') === day);
            }});

            countryButtons.forEach(b => {{
                b.classList.toggle('active', b.getAttribute('data-country') === country);
            }});
        }}

        dayButtons.forEach(btn => {{
            btn.addEventListener('click', () => {{
                selectedDay = btn.getAttribute('data-day');
                if (selectedCountry === null && countryButtons.length > 0) {{
                    selectedCountry = countryButtons[0].getAttribute('data-country');
                }}
                showPanel(selectedDay, selectedCountry);
            }});
        }});

        countryButtons.forEach(btn => {{
            btn.addEventListener('click', () => {{
                selectedCountry = btn.getAttribute('data-country');
                if (selectedDay === null && dayButtons.length > 0) {{
                    selectedDay = dayButtons[dayButtons.length - 1].getAttribute('data-day');
                }}
                showPanel(selectedDay, selectedCountry);
            }});
        }});

        window.addEventListener('DOMContentLoaded', () => {{
            if (dayButtons.length > 0) {{
                selectedDay = dayButtons[dayButtons.length - 1].getAttribute('data-day'); // newest
            }}
            if (countryButtons.length > 0) {{
                selectedCountry = countryButtons[0].getAttribute('data-country');
            }}
            if (selectedDay && selectedCountry) {{
                showPanel(selectedDay, selectedCountry);
            }}
        }});
    </script>
</body>
</html>
"""
    out_path = BASE_DIR / out_filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print("[INFO] Saved dashboard to", out_path)


#  MAIN

if __name__ == "__main__":
    today_anchor = datetime.now(ANCHOR_TZ).date()
    print("[DEBUG] Current working directory:", os.getcwd())
    print("[DEBUG] Anchor day (Europe/Rome):", today_anchor.isoformat())

    MIN_PHRASE_BY_COUNTRY = {"italy": 2, "uk": 2}

    for country, feeds in COUNTRY_FEEDS.items():
        print(f"\n=== Processing {country} ===")

        feed_tz = COUNTRY_TZ.get(country, ANCHOR_TZ)
        titles = fetch_titles(feeds, feed_tz=feed_tz, anchor_tz=ANCHOR_TZ)
        titles = [t.strip() for t in titles if t and t.strip()]

        stopwords_set = STOPWORDS_BY_COUNTRY.get(country, STOP_IT | STOP_EN | COMMON_STOPWORDS)

        nlp = nlp_it if country == "italy" else (nlp_en if country == "uk" else None)

        only_nouns_propn = (country == "uk")
        phrase_min_df = MIN_PHRASE_BY_COUNTRY.get(country, 2)

        today_counts, display_map = get_document_frequency_counts_topics(
            titles=titles,
            stopwords_set=stopwords_set,
            nlp=nlp,
            only_nouns_propn=only_nouns_propn,
            include_phrases=True,
            phrase_min_df=phrase_min_df,
            suppress_unigrams_in_phrases=True,
            lang=("it" if country == "italy" else "en"),
            global_suppress_unigrams_inside_phrases=True,
        )

        save_counts_for_day(today_anchor, country, today_counts)

        counts_for_wc = {display_map.get(k, k): v for k, v in today_counts.items()}
        wc_path_today = get_wordcloud_filename_for_day(today_anchor, country)
        make_wordcloud_from_term_counts(counts_for_wc, save_path=str(wc_path_today))

    save_week_dashboard_html(
        today_anchor,
        countries=list(COUNTRY_FEEDS.keys()),
        days_back=14,
        out_filename="index.html",
    )
