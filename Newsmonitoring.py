import re
import os
import json
import time
from datetime import date, timedelta, datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path

import feedparser
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

import matplotlib
matplotlib.use("Agg")  # safe non-GUI backend for CI / servers
import matplotlib.pyplot as plt

import pandas as pd

# ------------------------
# 0. STOPWORDS NLTK
# ------------------------
try:
    _ = stopwords.words("italian")
except LookupError:
    nltk.download("stopwords")

try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# ------------------------
# 0-bis. SPACY (per l'inglese)
# ------------------------
try:
    import spacy
    nlp_en = spacy.load("en_core_web_sm")
    print("[INFO] spaCy English model loaded.")
except Exception as e:
    nlp_en = None
    print("[WARN] spaCy English model NOT loaded, fallback to regex tokenization for UK.", e)

# ------------------------
# 1. RSS FEEDS PER NAZIONE
# ------------------------
COUNTRY_FEEDS = {
    "italy": [
        "https://www.repubblica.it/rss/homepage/rss2.0.xml",
        "https://www.rainews.it/rss/esteri",
        "https://www.servizitelevideo.rai.it/televideo/pub/rss101.xml",
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
        "https://www.theguardian.com/politics/rss",
        "https://www.theguardian.com/world/rss",
        "https://www.telegraph.co.uk/rss.xml",
        "http://www.independent.co.uk/rss",
        "https://www.ft.com/rss/home/international",
        "https://feeds.bbci.co.uk/news/world/rss.xml?edition=uk",
        "https://www.ft.com/world?format=rss",
        "https://feeds.skynews.com/feeds/rss/world.xml",
    ],
}


# ------------------------
# 1-bis. FETCH ARTICLES ONLY FROM TODAY
# ------------------------
def fetch_articles(feeds):
    """
    Scarica i titoli dai feed RSS e li concatena in un'unica stringa.
    Tiene solo gli articoli pubblicati OGGI (per data, in UTC).
    """
    today_date = datetime.now(timezone.utc).date()
    texts = []

    for url in feeds:
        print("[RSS]", url)
        feed = feedparser.parse(url)

        for entry in feed.entries:  # puoi usare [:20] se vuoi limitare
            title = entry.get("title", "")

            # usa solo i campi parsed (struct_time)
            dt_struct = entry.get("published_parsed") or entry.get("updated_parsed")
            if not dt_struct:
                # nessuna data parsabile -> salta
                continue

            dt = datetime.fromtimestamp(time.mktime(dt_struct), tz=timezone.utc)
            pub_date = dt.date()

            if pub_date == today_date:
                #print(pub_date, " ", title)
                texts.append(title)

    print("______________________________")
    return " ".join(texts)


# ------------------------
# 2. STOPWORDS CUSTOM (IT + EN, per paese)
# ------------------------
STOP_IT = set(stopwords.words("italian"))
STOP_EN = set(stopwords.words("english"))

# Stopwords comuni (rumore, nomi giornali, ecc.)
COMMON_STOPWORDS = {
    # Italiani generici
    "oggi", "ieri", "due", "tre", "solo", "ancora",
    "dopo", "prima", "contro", "secondo",
    "di", "per", "del", "della", "delle", "degli", "dei",
    "il", "la", "lo", "i", "gli",
    "un", "una", "uno", "nel", "nella", "nelle", "negli",
    "e", "ed", "le", "è", "ecco",
    "poi", "senza", "mai", "anni",
    "nuova", "tutto", "cosa", "casa",
    "repubblica", "ansa", "rai", "rainews", "notizie", "italia",
    # Inglesi generici
    "today", "yesterday", "two", "three", "only", "again",
    "after", "before", "against", "according",
    "new",
    # Rumore generico
    "img", "https", "con", "alt", "video", "foto", "corriere", "images2",
}

# Extra-stopwords manuali per l’inglese / UK (politica + news)
EXTRA_STOPWORDS_UK = {
    "uk", "britain", "english", "england",
    "say", "says", "said",
    "best", "year", "years", "week", "weeks",
    "another", "many", "much", "big", "small",
    "newest", "latest", "live", "update", "updates", "breaking",
}

# Lista più corposa pensata per news / tabloid
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
    "up", "down",
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
    "workers", "crowd", "fans",
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

# ------------------------
# 3. TOKENIZZAZIONE E FILTRO
# ------------------------
def tokenize_and_filter(text, stopwords_set, use_spacy_nouns=False, nlp=None):
    """
    Se use_spacy_nouns=True e nlp non è None:
        - usa spaCy
        - tiene solo NOUN e PROPN
    Altrimenti:
        - tokenizzazione regex standard
    In entrambi i casi applica stopwords_set e lunghezza >= 3.
    """
    tokens_raw = []

    if use_spacy_nouns and nlp is not None:
        doc = nlp(text)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue
            tokens_raw.append(token.text)
    else:
        # fallback semplice con regex
        tokens_raw = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)

    filtered = []
    for w in tokens_raw:
        wl = w.lower()
        if len(wl) < 3:
            continue
        if wl in stopwords_set:
            continue
        filtered.append(w)
    return filtered


# ------------------------
# 4. WORDCLOUD (UNIGRAMS + BIGRAMS)
# ------------------------
def make_wordcloud_from_tokens(tokens, stopwords_set, min_bigram_freq=2, save_path=None):
    # 1) unigrams: conteggio case-insensitive
    counts_lower = Counter()
    forms_by_lower = defaultdict(Counter)

    for w in tokens:
        wl = w.lower()
        counts_lower[wl] += 1
        forms_by_lower[wl][w] += 1

    # 2) bigrams espliciti (usiamo le forme originali)
    bigram_counts = Counter()
    for w1, w2 in zip(tokens, tokens[1:]):
        wl1, wl2 = w1.lower(), w2.lower()
        if wl1 in stopwords_set or wl2 in stopwords_set:
            continue
        bigram = f"{w1} {w2}"
        bigram_counts[bigram] += 1

    # 3) dizionario per la wordcloud
    freq_for_wc = {}

    # unigrams
    for wl, count in counts_lower.items():
        best_form, _ = forms_by_lower[wl].most_common(1)[0]
        freq_for_wc[best_form] = count

    # bigrams
    for bigram, count in bigram_counts.items():
        if count >= min_bigram_freq:
            freq_for_wc[bigram] = count

    # 4) WordCloud
    wc = WordCloud(
        width=1600,
        height=800,
        background_color="black",
        colormap="rainbow",
        contour_width=3,
        contour_color="white",
        collocations=False,
        max_words=500,
        prefer_horizontal=0.5,
        relative_scaling=1.0,
        random_state=42,
    ).generate_from_frequencies(freq_for_wc)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.close()  # importante in CI per non accumulare figure


# ------------------------
# 5. ANALISI TOP PAROLE
# ------------------------
def show_top_words(tokens, x=10):
    counts = Counter(w.lower() for w in tokens)
    print(f"\nTop {x} parole (senza collocations):\n" + "-" * 30)
    for parola, freq in counts.most_common(x):
        print(f"{parola:25s} {freq}")


def get_word_counts(tokens):
    """
    Ritorna un dict: parola (lowercase) -> frequenza
    da usare per il salvataggio giornaliero.
    """
    counts = Counter(w.lower() for w in tokens)
    return dict(counts)


# ------------------------
# 6. GESTIONE FILE GIORNALIERI
# ------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)


def get_counts_filename(day: date, country: str) -> str:
    return str(DATA_DIR / f"{day.isoformat()}_{country}_words.json")


def save_counts_for_day(day: date, country: str, counts: dict):
    filename = get_counts_filename(day, country)
    fullpath = os.path.abspath(filename)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Saved counts for {country} to {fullpath}")


def load_counts_for_day(day: date, country: str):
    filename = get_counts_filename(day, country)
    if not os.path.exists(filename):
        print(f"[INFO] No data for {day.isoformat()} ({country}) ({filename} not found)")
        return None
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------
# 6-bis. WORDCLOUD FILENAME (per giorno)
# ------------------------
def get_wordcloud_filename_for_day(day: date, country: str) -> Path:
    """
    Path per il wordcloud di uno specifico giorno e paese.
    Esempio: data/2025-12-21_italy_wordcloud.png
    """
    return DATA_DIR / f"{day.isoformat()}_{country}_wordcloud.png"


# ------------------------
# 7. TABELLE: NUOVE PAROLE / UP / DOWN
# ------------------------
def get_new_words_table(today_counts: dict, yesterday_counts: dict, top_n=10):
    today_words = set(today_counts.keys())
    yest_words = set(yesterday_counts.keys())

    new_words = today_words - yest_words

    rows = []
    for w in new_words:
        rows.append({"Word": w, "Today frequency": today_counts[w]})

    if not rows:
        return pd.DataFrame(columns=["Word", "Today frequency"])

    df = pd.DataFrame(rows)
    df = df.sort_values("Today frequency", ascending=False).head(top_n)
    return df


def get_rising_words_table(today_counts: dict, yesterday_counts: dict, top_n=10):
    today_words = set(today_counts.keys())
    yest_words = set(yesterday_counts.keys())

    rows = []
    for w in today_words & yest_words:
        today_f = today_counts[w]
        yest_f = yesterday_counts.get(w, 0)
        delta = today_f - yest_f
        if delta > 0:
            rows.append({
                "Word": w,
                "Difference": delta,
                "Today frequency": today_f,
                "Yesterday frequency": yest_f,
            })

    if not rows:
        return pd.DataFrame(columns=["Word", "Difference", "Today frequency", "Yesterday frequency"])

    df = pd.DataFrame(rows)
    df = df.sort_values("Difference", ascending=False).head(top_n)
    return df


def get_falling_words_table(today_counts: dict, yesterday_counts: dict, top_n=10):
    today_words = set(today_counts.keys())
    yest_words = set(yesterday_counts.keys())

    rows = []
    for w in today_words & yest_words:
        today_f = today_counts[w]
        yest_f = yesterday_counts.get(w, 0)
        delta = today_f - yest_f
        if delta < 0:  # calo
            rows.append({
                "Word": w,
                "Difference": delta,
                "Today frequency": today_f,
                "Yesterday frequency": yest_f,
            })

    if not rows:
        return pd.DataFrame(columns=["Word", "Difference", "Today frequency", "Yesterday frequency"])

    df = pd.DataFrame(rows)
    # ordina per calo maggiore (delta più negativo)
    df = df.sort_values("Difference", ascending=True).head(top_n)
    return df


def compare_days(today_counts: dict, yesterday_counts: dict, top_n=10):
    """
    Stampa un riassunto a console e ritorna
    (df_new, df_rising, df_falling).
    """
    df_new = get_new_words_table(today_counts, yesterday_counts, top_n=top_n)
    df_rising = get_rising_words_table(today_counts, yesterday_counts, top_n=top_n)
    df_falling = get_falling_words_table(today_counts, yesterday_counts, top_n=top_n)

    print("\n=== TOP NUOVE PAROLE (oggi ma non ieri) ===")
    if df_new.empty:
        print("Nessuna nuova parola.")
    else:
        for _, row in df_new.iterrows():
            print(f"{row['Word']:25s} {int(row['Today frequency'])}")

    print("\n=== TOP PAROLE IN CRESCITA (oggi >> ieri) ===")
    if df_rising.empty:
        print("Nessuna parola in crescita.")
    else:
        for _, row in df_rising.iterrows():
            print(
                f"{row['Word']:25s} +{int(row['Difference']):3d} "
                f"(yesterday: {int(row['Yesterday frequency'])}, today: {int(row['Today frequency'])})"
            )

    print("\n=== TOP PAROLE IN CALO (oggi << ieri) ===")
    if df_falling.empty:
        print("Nessuna parola in calo.")
    else:
        for _, row in df_falling.iterrows():
            print(
                f"{row['Word']:25s} {int(row['Difference']):3d} "
                f"(yesterday: {int(row['Yesterday frequency'])}, today: {int(row['Today frequency'])})"
            )

    return df_new, df_rising, df_falling


# ------------------------
# 8. DASHBOARD HTML (senza slider)
# ------------------------
def save_dashboard_html_multi(today: date, per_country_results: dict):
    """
    per_country_results: dict
      country -> {
        "new_df": DataFrame or None,
        "rising_df": DataFrame or None,
        "falling_df": DataFrame or None,
        "wc_rel_path": str (path all'immagine di oggi)
      }
    """
    generated_at = datetime.now()
    generated_str = generated_at.strftime("%d %b %Y, %H:%M")

    def df_or_message(df, msg):
        if df is None or df.empty:
            return f"<p class='empty-msg'>{msg}</p>"
        else:
            html = df.to_html(index=False, classes=["data-table"])
            return html

    country_sections_html = []
    for country, data in per_country_results.items():
        new_df = data["new_df"]
        rising_df = data["rising_df"]
        falling_df = data["falling_df"]
        wc_rel_path = data["wc_rel_path"]

        html_new = df_or_message(new_df, "No new words compared to yesterday.")
        html_rising = df_or_message(rising_df, "No significantly increasing words.")
        html_falling = df_or_message(falling_df, "No significantly decreasing words.")

        flag = "🇮🇹" if country == "italy" else ("🇬🇧" if country == "uk" else "🏳️")

        section = f"""
        <div class="country-dashboard" id="country-{country}" style="display:none;">
            <div class="header-row">
                <h2>{flag} {country.capitalize()}</h2>
                <span class="date-pill">Updated: {generated_str}</span>
            </div>

            <div class="content-grid">
                <div class="card">
                    <h3>Today's wordcloud</h3>
                    <img src="{wc_rel_path}" alt="Wordcloud {country}" class="wordcloud">
                </div>

                <div class="card tables-card">
                    <h3>Top 10 new words (today vs yesterday)</h3>
                    {html_new}
                    <h3>Top 10 increasing words (today ≫ yesterday)</h3>
                    {html_rising}
                    <h3>Top 10 decreasing words (today ≪ yesterday)</h3>
                    {html_falling}
                </div>
            </div>
        </div>
        """
        country_sections_html.append(section)

    def country_label(c: str) -> str:
        if c == "italy":
            return "🇮🇹 Italy"
        if c == "uk":
            return "🇬🇧 UK"
        return c.capitalize()

    buttons_html = "".join(
        f'<button class="country-btn" data-country="{c}">{country_label(c)}</button>'
        for c in per_country_results.keys()
    )

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

        * {{
            box-sizing: border-box;
        }}

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
            margin-bottom: 16px;
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

        h1 {{
            margin: 0 0 6px;
            font-size: 1.4rem;
        }}

        .subtitle {{
            margin: 0 0 4px;
            font-size: 0.9rem;
            color: var(--text-muted);
        }}

        .updated-info {{
            margin: 0 0 18px;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .country-dashboard {{
            margin-top: 8px;
        }}

        .header-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
        }}

        .header-row h2 {{
            margin: 0;
            font-size: 1.2rem;
        }}

        .date-pill {{
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid var(--border-subtle);
            font-size: 0.78rem;
            color: var(--text-muted);
        }}

        .content-grid {{
            display: block;
        }}

        @media (max-width: 960px) {{
            body {{
                flex-direction: column;
            }}
            .sidebar {{
                position: static;
                width: 100%;
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 8px;
            }}
            .sidebar h2 {{
                flex: 1 0 100%;
            }}
            .container {{
                width: 100%;
            }}
        }}

        .card {{
            background: var(--bg-card);
            border-radius: 14px;
            border: 1px solid #202637;
            padding: 14px 16px;
            margin-bottom: 14px;
        }}

        .card h3 {{
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 1rem;
        }}

        .wordcloud {{
            width: 100%;
            height: auto;
            border-radius: 10px;
            border: 1px solid #2d3748;
            display: block;
        }}

        .tables-card h3 {{
            margin-top: 12px;
            font-size: 0.98rem;
        }}

        .tables-card h3:first-of-type {{
            margin-top: 0;
        }}

        .empty-msg {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin: 4px 0 12px;
        }}

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

        table.data-table tr:nth-child(even) {{
            background-color: #0b1220;
        }}

        table.data-table tr:nth-child(odd) {{
            background-color: #050b18;
        }}

        table.data-table td:nth-child(2),
        table.data-table td:nth-child(3),
        table.data-table td:nth-child(4) {{
            font-family: var(--font-mono);
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Select a Country</h2>
        <p>Click a country to explore its top topics, emerging words, and fading stories.</p>
        {buttons_html}
    </div>
    <div class="container">
        <h1>European News Dashboard</h1>
        <p class="subtitle">Daily RSS headlines analysis – key words and trends compared to yesterday.</p>
        <p class="updated-info">Last update: {generated_str}</p>
        {''.join(country_sections_html)}
    </div>

    <script>
        const dashboards = document.querySelectorAll('.country-dashboard');
        const buttons = document.querySelectorAll('.country-btn');

        function showCountry(country) {{
            dashboards.forEach(d => {{
                if (d.id === 'country-' + country) {{
                    d.style.display = 'block';
                }} else {{
                    d.style.display = 'none';
                }}
            }});
            buttons.forEach(b => {{
                if (b.getAttribute('data-country') === country) {{
                    b.classList.add('active');
                }} else {{
                    b.classList.remove('active');
                }}
            }});
        }}

        buttons.forEach(btn => {{
            btn.addEventListener('click', () => {{
                const country = btn.getAttribute('data-country');
                showCountry(country);
            }});
        }});

        window.addEventListener('DOMContentLoaded', () => {{
            if (buttons.length > 0) {{
                const firstCountry = buttons[0].getAttribute('data-country');
                showCountry(firstCountry);
            }}
        }});
    </script>
</body>
</html>
"""
    out_path = BASE_DIR / "dashboard" / "index.html"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print("[INFO] Saved multi-country dashboard (no slider) to", out_path)


# ------------------------
# 9. MAIN
# ------------------------
if __name__ == "__main__":
    today = date.today()
    yesterday = today - timedelta(days=1)

    print("[DEBUG] Current working directory:", os.getcwd())

    per_country_results = {}

    for country, feeds in COUNTRY_FEEDS.items():
        print(f"\n=== Processing {country} ===")

        # 1. Fetch RSS e tokenizzazione
        raw_text = fetch_articles(feeds)
        stopwords_set = STOPWORDS_BY_COUNTRY.get(
            country, STOP_IT | STOP_EN | COMMON_STOPWORDS
        )

        # Per UK usiamo spaCy per tenere solo NOUN/PROPN
        use_spacy = (country == "uk" and nlp_en is not None)
        tokens = tokenize_and_filter(
            raw_text,
            stopwords_set,
            use_spacy_nouns=use_spacy,
            nlp=nlp_en if use_spacy else None,
        )

        # 2. Analisi base del giorno
        show_top_words(tokens, x=15)

        # 3. Frequenze e salvataggio
        today_counts = get_word_counts(tokens)
        save_counts_for_day(today, country, today_counts)

        # 4. Confronto con ieri
        yesterday_counts = load_counts_for_day(yesterday, country)
        new_df = rising_df = falling_df = None
        if yesterday_counts is not None:
            new_df, rising_df, falling_df = compare_days(
                today_counts, yesterday_counts, top_n=10
            )
        else:
            print("\nNo data for yesterday, cannot compare yet for", country)

        # 5. Wordcloud del giorno (file con data nel nome)
        wc_path_today = get_wordcloud_filename_for_day(today, country)
        make_wordcloud_from_tokens(tokens, stopwords_set, save_path=str(wc_path_today))
        wc_rel_path_today = os.path.join("..", "data", wc_path_today.name)

        per_country_results[country] = {
            "new_df": new_df,
            "rising_df": rising_df,
            "falling_df": falling_df,
            "wc_rel_path": wc_rel_path_today,
        }

    # 7. Dashboard HTML
    save_dashboard_html_multi(today, per_country_results)
