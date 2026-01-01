import re
import os
import json
import calendar
from dataclasses import dataclass
from datetime import datetime, timezone, date
from collections import defaultdict, Counter
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import nltk
from nltk.corpus import stopwords


# 1 STOPWORDS (NLTK)
# Carichiamo le stopwords con Natural Language TOOLKIT. Si usa qquesta libreria perché la lista di stopword è più stabile e collaudata rispetto a SpaCy

def _ensure_stopwords(lang: str):
    try:
        _ = stopwords.words(lang)  # vediamo se sono già caricate
    except LookupError:
        nltk.download("stopwords")  # se dà errore facciamo download


_ensure_stopwords("italian")  # funzione che garantisce che ci siano le stopwords
_ensure_stopwords("english")

STOP_IT = set(stopwords.words("italian"))  # creaiamo i set per le due lingue
STOP_EN = set(stopwords.words("english"))

# aggiungiamo altre stopwords (più piccolo dell'altro dashboard visto che è meno importante qui)
COMMON_STOPWORDS = {
    # IT
    "oggi", "ieri", "solo", "ancora", "dopo", "prima", "contro", "secondo",
    "di", "per", "del", "della", "delle", "degli", "dei",
    "il", "la", "lo", "i", "gli", "un", "una", "uno",
    "nel", "nella", "nelle", "negli", "e", "ed", "è", "poi", "senza", "mai",
    "repubblica", "ansa", "rai", "rainews", "notizie", "italia", "italiani",
    # EN
    "today", "yesterday", "only", "again", "after", "before", "against", "according",
    "new",
    # noise
    "img", "https", "video", "foto", "images",
}

EXTRA_STOPWORDS_UK = {
    "uk", "britain", "english", "england",
    "say", "says", "said",
    "live", "update", "updates", "breaking",
}

STOPWORDS_BY_COUNTRY = {
    "italy": STOP_IT | COMMON_STOPWORDS,
    "uk": STOP_EN | COMMON_STOPWORDS | EXTRA_STOPWORDS_UK,
}

# 2 FEEDS
# Selezioniamo i feed (come quelli dell'altro codice). Buona varietà e equilibrio
COUNTRY_FEEDS = {
    "italy": [
        "https://www.repubblica.it/rss/homepage/rss2.0.xml",
        "https://www.rainews.it/rss/esteri",
        # "https://www.servizitelevideo.rai.it/televideo/pub/rss101.xml", (questo fa troppi feed e sbilancia, alcuni non rilevanit)
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
        # "https://www.independent.co.uk/news/world/rss", (troppi feed)
        "https://www.ft.com/rss/home/international",
        "https://feeds.bbci.co.uk/news/world/rss.xml?edition=uk",
        "https://feeds.skynews.com/feeds/rss/world.xml",
        "https://feeds.skynews.com/feeds/rss/uk.xml",
    ],
}


# 3) DATA MODEL credo una classe (ITEM) con @dataclass
# (stesso che fare "class Item: def __init__ etc." ma più immediato e con meno codice) dove immagazzino le news e i loro dati
#
@dataclass(frozen=True)
class Item:
    country: str
    publisher: str
    dt_utc: str
    title: str
    link: str
    feed_url: str


#  4 PUBLISHER NORMALISATION
# facciamo un dizionario per normalizzare le fonti e renderle più leggibili
PUBLISHER_MAP = {
    # BBC
    "feeds.bbci.co.uk": "BBC",
    "bbc.co.uk": "BBC",
    # Guardian
    "theguardian.com": "The Guardian",
    # Sky
    "feeds.skynews.com": "Sky News",
    "news.sky.com": "Sky News",
    # Independent
    "independent.co.uk": "The Independent",
    # FT
    "ft.com": "Financial Times",
    # Italy sources (optional; add more if you like)
    "repubblica.it": "La Repubblica",
    "rainews.it": "RaiNews",
    # "servizitelevideo.rai.it": "Rai Televideo",
    "ilsole24ore.com": "Il Sole 24 Ore",
    "agi.it": "AGI",
    "adnkronos.com": "Adnkronos",
}


def publisher_from_url(url: str) -> str:
    netloc = (urlparse(url).netloc or "").lower().replace("www.",
                                                          "")  # prende il networklocation dell'url oppure ritorna stringa vuota (or se la prima condizione è falsa cioè il netloc non c'è)
    return PUBLISHER_MAP.get(netloc,
                             netloc or "unknown")  # fa il mapping oppure (se non trova nel dizionario) torna netloc or unknown se vuoto


# 5 NORMALISATION + TOKENS dei titoli

def normalize_title_for_dedup(t: str) -> str:
    t = (t or "").lower().strip()  # in minuscolo (se vuoto torna stringa vuota)
    t = re.sub(r"['’]", "", t)  # toglie apostrofi
    t = re.sub(r"[^a-z0-9à-öø-ÿ\s]", " ", t,
               flags=re.IGNORECASE)  # sostituisce con spazio tutto ciò che non è: lettere a–z numeri 0–9 lettere accentate nel range latino à-öø-ÿ spazi (\s) Quindi elimina punteggiatura e simboli: : , . ! ? ( ) — … ecc
    t = re.sub(r"\s+", " ", t).strip()  # compatta spazi multipli in uno solo (se prima hai messo tanti spazi).
    return t


KEEP_SHORT = {"eu", "ue", "us", "uk", "un", "g7", "g8"}


def title_tokens(title: str, stopwords_set: set[str]) -> set[
    str]:  # annotazione che mi dice che mi aspetto la funzione ritorni
    norm = normalize_title_for_dedup(title)
    words = norm.split()
    out = set()
    for w in words:  # filtriamo stringe più corte di tre, quelle nella lista di stopword e i numeri
        if w.isdigit():
            continue
        if len(w) < 3 and w not in KEEP_SHORT:
            continue
        if w in stopwords_set and w not in KEEP_SHORT:
            continue

        out.add(w)
    return out


# 5 FETCH  + DE-DUP
# prende gli item dai feed e poi toglie i duplicati
def fetch_items_for_country(country: str, feeds: list[str], target_day: date) -> list[Item]:
    items = []  # creo una lista che sarà riempita di items
    x = 0
    for feed_url in feeds:
        pub = publisher_from_url(feed_url)  # chiamo la funzione di prima
        feed = feedparser.parse(feed_url)  # prendo il feed

        if getattr(feed, "bozo", 0):  # se il feed è malformato
            ex = getattr(feed, "bozo_exception", None)
            print(f"[WARN] Feed parse issue for {feed_url}: {ex}")

        tmp = []
        for entry in getattr(feed, "entries", []):  # per feed trova gli entries
            title = (entry.get("title", "") or "").strip()
            if not title:
                continue

            dt_struct = entry.get("published_parsed") or entry.get("updated_parsed")
            if not dt_struct:
                continue

            dt = datetime.fromtimestamp(calendar.timegm(dt_struct), tz=timezone.utc)
            if dt.date() != target_day:
                continue

            link = (entry.get("link", "") or "").strip()
            # print(dt, pub,title)
            tmp.append(
                Item(
                    country=country,
                    publisher=pub,
                    dt_utc=dt.isoformat(),
                    title=title,
                    link=link,
                    feed_url=feed_url,
                ))

        # togliamo duplicati se il titolo appare nello stesso feed
        seen_feed = set()
        tmp2: list[Item] = []  # una lista che sara di oggetti
        for it in tmp:
            k = normalize_title_for_dedup(it.title)  # normalizziamo con la funzione fatta prima il titolo
            if k in seen_feed:
                continue  # se il titolo è già presente skippa altimenti aggiunge
            seen_feed.add(k)
            tmp2.append(it)

        items.extend(tmp2)  # aggiunge gli elementi uno per uno. Con append aggiungerebbe la lista come singolo elemento

    # extra de duplicazione: stesso publisher + stesso UTC hour bucket + stesso title normalizzato
    deduped: list[Item] = []
    seen_pub_hour_title = set()
    for it in items:
        dt = datetime.fromisoformat(it.dt_utc)
        hour_bucket = dt.replace(minute=0, second=0, microsecond=0).isoformat()  # facciamo una finestra di un'ora
        norm_title = normalize_title_for_dedup(it.title)
        key = (it.publisher, hour_bucket,
               norm_title)  # salta se il titolo viene dalla stesso publisher, nella stessa ora e con lo stesso titolo
        if key in seen_pub_hour_title:
            continue
        seen_pub_hour_title.add(key)
        deduped.append(it)

    return deduped


# ============================================================
# 6 STORY CLUSTERING (greedy)
# _____________________________________________________________
# funzione che definisce score di Jaccard. Definito da Wiki come:
# The Jaccard index is a statistic used for gauging the similarity and diversity of sample sets. It is defined in general taking the ratio of two sizes (areas or volumes),
# the intersection size divided by the union size, also called intersection over union (IoU).
def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)  # intersezione
    union = len(a | b)  # unione
    return inter / union if union else 0.0


def best_rep_title(items: list[Item], centroid_tokens: set[str], stopset: set[str]) -> str:
    # Sceglie il titolo più 'rappresentativo' del cluster: quello con la Jaccard più alta rispetto al centroide (soft). Tie-break: titolo più recente. (altrimenti prenderebbe solo titolo più recente)
    if not items:
        return ""

    best_title = items[0].title
    best_score = -1.0
    best_dt = ""

    for it in items:
        toks = title_tokens(it.title, stopset)
        score = jaccard(toks, centroid_tokens)
        # tie-break: se stesso score, preferisci il più recente
        if (score > best_score) or (score == best_score and it.dt_utc > best_dt):
            best_score = score
            best_title = it.title
            best_dt = it.dt_utc

    return best_title


# Centroide a "intersezione morbida":
# invece di fare union dei token (che cresce e può rendere il cluster troppo largo),
# teniamo nel centroide solo i token che compaiono in almeno X% dei titoli del cluster.
def recompute_soft_centroid(token_counts: Counter, n_titles: int, min_frac: float) -> set[str]:
    if n_titles <= 0:
        return set()

    # soglia in conteggio: approx ceil(min_frac * n_titles) ma senza import math
    thr = max(1, int((min_frac * n_titles) + (1.0 - 1e-9)))
    return {tok for tok, c in token_counts.items() if c >= thr
            }

    # inseriamo la lista di Items, le stopword e il threshold dello score di jaccard per cui mette nello stesso cluster


def cluster_items_into_stories(items: list[Item], stopwords_by_country: dict[str, set[str]], threshold: float = 0.25,
                               centroid_min_frac: float = 0.6, soft_after: int = 3):
    items_sorted = sorted(items, key=lambda x: x.dt_utc,
                          reverse=True)  # ordiniamo gli item dal più nuovo (questa scelta può essere discussa perché favorisce storie più recenti)

    stories = []  # questa diventerà una lista di dizionari
    for it in items_sorted:
        stopset = stopwords_by_country.get(it.country,
                                           STOP_IT | STOP_EN | COMMON_STOPWORDS)  # carica le stopword per la lingua di Item
        toks = title_tokens(it.title, stopset)  # tokenizza il titolo

        best_idx = None  # indice della storia più simile trovata finora
        best_score = 0.0
        for i, st in enumerate(stories):  # torna indice e dizionario
            if st["country"] != it.country:
                continue

            score = jaccard(toks, st["centroid_tokens"])  # calcolo jaccard
            shared_n = len(toks & st["centroid_tokens"])
            toks_strong = {t for t in toks if
                           len(t) >= 5}  # assumiamo che parole di 5 o più lettere portino più significato
            cent_strong = {t for t in st["centroid_tokens"] if len(t) >= 5}  # centroide di parole più lunghe di 5
            shared_strong_n = len(toks_strong & cent_strong)  # intersezione
            fallback = (shared_n >= 3 and shared_strong_n >= 1)  # almeno 3 parole condivise di cui 1 importante
            if fallback:
                score = max(score, threshold)

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None and best_score >= threshold:  # se il best score è più alto del threshold, aggiungo al cluster
            st = stories[best_idx]
            st["items"].append(it)
            st["publishers"].add(it.publisher)
            st["last_dt"] = max(st["last_dt"], it.dt_utc)
            st["first_dt"] = min(st["first_dt"], it.dt_utc)  # finestra temporare
            # st["centroid_tokens"] = st["centroid_tokens"] | toks  # unione centroid (il centroide cresce) problema?? può attrarre titoli non attinenti o abbassare troppo lo score
            st["n_titles"] += 1
            st["token_counts"].update(toks)

            # aggiorniamo centroide in modo "soft": token che compaiono in almeno X% dei titoli del cluster
            if st["n_titles"] < soft_after:
                st["centroid_tokens"] = set(st["token_counts"].keys())  # union-ish
            else:
                st["centroid_tokens"] = recompute_soft_centroid(
                    st["token_counts"], st["n_titles"], centroid_min_frac)

            # aggiorna titolo rappresentativo (B): titolo con Jaccard più alta rispetto al centroide soft
            st_stopset = stopwords_by_country.get(st["country"], STOP_IT | STOP_EN | COMMON_STOPWORDS)
            st["rep_title"] = best_rep_title(st["items"], st["centroid_tokens"], st_stopset)
        else:  # altrimenti creo nuova storia con nuovo indice
            sid = f"{it.country}-{len(stories) + 1}"
            token_counts = Counter(toks)
            stories.append(
                {
                    "id": sid,
                    "country": it.country,
                    "rep_title": it.title,
                    "centroid_tokens": recompute_soft_centroid(token_counts, 1, centroid_min_frac),
                    "items": [it],
                    "publishers": {it.publisher},
                    "first_dt": it.dt_utc,
                    "last_dt": it.dt_utc,

                    # supporto per centroide soft
                    "n_titles": 1,
                    "token_counts": token_counts,
                }
            )

    # ranking: quante editori riportatano, quante notizie riportate, più recente
    for st in stories:
        st["coverage"] = len(st["publishers"])
        st["n_items"] = len(st["items"])

    stories.sort(key=lambda s: (s["coverage"], s["n_items"], s["last_dt"]), reverse=True)
    return stories


#
# 7) STORY DASHBOARD HTML
#
# 7) STORY DASHBOARD HTML
#
def save_story_dashboard_html(out_path: Path, day: date):
    generated_str = datetime.now().strftime("%d %b %Y, %H:%M")
    day_str = day.isoformat()

    # storyboard stile "index": sidebar a sinistra, controlli solo lì (paese con bandiere + day selector + search)
    # la parte destra carica un file "content-only" stories_YYYY-MM-DD.html dentro un iframe
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Story Dashboard — {day_str}</title>
<style>
:root {{
  --bg:#05070a;
  --panel:#101521;
  --card:#141b2a;
  --text:#f5f5f5;
  --muted:#a0aec0;
  --border:#263145;
  --accent:#4fd1c5;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  background: var(--bg); color: var(--text);
}}
.app {{
  display: flex;
  min-height: 100vh;
}}
.sidebar {{
  width: 320px; max-width: 85vw;
  border-right: 1px solid var(--border);
  background: rgba(16,21,33,0.65);
  padding: 18px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow: auto;
}}
.brand {{
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 6px;
}}
.brand-day {{
  color: var(--muted);
  font-weight: 600;
  margin-left: 6px;
  font-size: .95rem;
}}
.sub {{
  color: var(--muted);
  font-size: .85rem;
  margin-bottom: 16px;
  line-height: 1.35;
}}
label {{
  display: block;
  color: var(--muted);
  font-size: .82rem;
  margin: 14px 0 6px;
}}
select, input {{
  width: 100%;
  background: var(--panel);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 10px 11px;
  border-radius: 12px;
  outline: none;
}}
.hint {{
  color: var(--muted);
  font-size: .78rem;
  margin-top: 12px;
  line-height: 1.35;
}}
.dashlink{{
  display:block;
  margin-top: 110px;
  padding: 10px 11px;
  border-radius: 12px;
  background: var(--card);
  border: 1px solid var(--border);
  color: var(--text);
  text-decoration: none;
  font-weight: 600;
}}
.dashlink-title{{
  font-size: 1.35rem;   
  font-weight: 800;
  line-height: 1.15;
}}

.dashlink:hover{{
  border-color: var(--accent);
  color: var(--accent);
}}
.dashlink .sub{{
  margin: 0 0 4px 0;
  font-size: .8rem;
  color: var(--muted);
  font-weight: 500;
}}


.main {{
  flex: 1;
  padding: 0;
}}
.frame {{
  width: 100%;
  height: 100vh;
  border: 0;
  background: var(--bg);
}}
.notfound {{
  height: 100vh;
  padding: 26px 26px;
}}
.notfound .title {{
  font-size: 1.1rem;
  font-weight: 700;
  margin-bottom: 8px;
}}
.notfound .meta {{
  color: var(--muted);
  font-size: .9rem;
  line-height: 1.4;
}}
code {{ color: var(--accent); }}
</style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">Story Dashboard <span id="brandDay" class="brand-day"></span></div>
      <div class="sub">Clusters of repeated stories across sources</div>

      <label for="countryFilter">Country</label>
      <select id="countryFilter">
        <option value="all">🌍 All</option>
        <option value="italy">🇮🇹 Italy</option>
        <option value="uk">🇬🇧 UK</option>
      </select>

      <label for="dayFilter">Day (up to last 14)</label>
      <select id="dayFilter"></select>

      <label for="q">Search</label>
      <input id="q" type="text" placeholder="Search story titles (e.g., Trump, Meloni)" />

      <div class="hint">
        <br/>
        Updated: {generated_str}<br/>
      </div>
      <a class="dashlink" href="https://andrevitali.github.io/NewsMonitoring/" target="_blank" rel="noopener">
      <div class="sub">Click here for the main dashboard</div>
      <div class="dashlink-title">News Dashboard</div>
      </a>
    </aside>

    <main class="main">
      <div id="notFound" class="notfound" style="display:none;">
        <div class="title">File not found</div>
        <div class="meta" id="notFoundMsg"></div>
      </div>

      <iframe id="contentFrame" class="frame" src=""></iframe>
    </main>
  </div>

<script>
const countryFilter = document.getElementById('countryFilter');
const dayFilter = document.getElementById('dayFilter');
const q = document.getElementById('q');

const frame = document.getElementById('contentFrame');
const notFound = document.getElementById('notFound');
const notFoundMsg = document.getElementById('notFoundMsg');

function fmt(d){{
  const y=d.getFullYear();
  const m=String(d.getMonth()+1).padStart(2,'0');
  const da=String(d.getDate()).padStart(2,'0');
  return `${{y}}-${{m}}-${{da}}`;
}}

function buildLast14() {{
  const today = new Date(); // usa il clock locale per la lista (ok per UI)
  const days = [];
  for (let i=0;i<14;i++) {{
    const d = new Date(today);
    d.setDate(today.getDate()-i);
    days.push(fmt(d));
  }}
  return days;
}}

function setRightNotFound(day) {{
  frame.style.display = "none";
  notFound.style.display = "block";
  notFoundMsg.innerHTML = `Missing: <code>stories_${{day}}.html</code><br/>Generate it by running the script on that day (UTC), or keep daily outputs.`;
}}

function setRightFrame(url) {{
  notFound.style.display = "none";
  frame.style.display = "block";
  frame.src = url;
}}

// carica contenuto e passa i filtri via querystring (country + q) al file content-only
async function loadDay(day) {{
  const country = countryFilter.value;
  const qq = encodeURIComponent((q.value || '').trim());
  const url = `stories_${{day}}.html?country=${{country}}&q=${{qq}}`;

  // verifica esistenza del file:
  // Nota: funziona bene via HTTP (es. python -m http.server). Su file:// fetch può essere bloccato.
  try {{
    const res = await fetch(`stories_${{day}}.html`, {{ method: "HEAD" }});
    if (!res.ok) {{
      setRightNotFound(day);
      return;
    }}
    setRightFrame(url);
  }} catch (e) {{
    // fallback: prova a caricare comunque (se fallisce, almeno non rompiamo la UI)
    setRightFrame(url);
  }}
}}

(function initUI(){{
  const currentDay = "{day_str}";
  const days = buildLast14();

  function prettyLabel(iso) {{
    const [y, m, d] = iso.split('-').map(Number);
    const dt = new Date(Date.UTC(y, m - 1, d));
    return dt.toLocaleDateString('en-GB', {{
      day: '2-digit',
      month: 'short',
      year: 'numeric'
    }});
  }}

  days.forEach(ds => {{
    const opt = document.createElement('option');
    // IMPORTANT: keep ISO for filenames
    opt.value = ds;
    // Nice human label
    opt.textContent = prettyLabel(ds);
    dayFilter.appendChild(opt);
  }});

  dayFilter.value = currentDay;

  const brandDay = document.getElementById('brandDay');
  function updateBrandDay() {{
    brandDay.textContent = `— ${{prettyLabel(dayFilter.value)}}`;
  }}
  updateBrandDay();

  // init load
  loadDay(currentDay);

  dayFilter.addEventListener('change', () => {{
    updateBrandDay();
    loadDay(dayFilter.value);
  }});

  function reloadSameDay() {{
    loadDay(dayFilter.value);
  }}

  countryFilter.addEventListener('change', reloadSameDay);
  q.addEventListener('input', () => {{
    window.clearTimeout(window.__t);
    window.__t = window.setTimeout(reloadSameDay, 120);
  }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print("[INFO] Saved story dashboard to:", out_path)


# 7b) CONTENT-ONLY HTML (una pagina per giorno, solo cards, filtri via querystring)

def save_story_content_html(out_path: Path, day: date, stories: list[dict]):
    day_str = day.isoformat()

    def esc(s: str) -> str:  # Serve perché i titoli RSS possono contenere &, <, ", ecc. Senza escaping, un titolo potrebbe: rompere l’HTML (render sbagliato)
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    cards_html = []  # si crea una card per ogni storia
    for st in stories:
        items_sorted = sorted(st["items"], key=lambda x: x.dt_utc, reverse=True)  # ordina le storie
        sources = ", ".join(sorted(st["publishers"]))
        first_hm = datetime.fromisoformat(st["first_dt"]).strftime("%H:%M")
        last_hm = datetime.fromisoformat(st["last_dt"]).strftime("%H:%M")

        items_rows = []
        for it in items_sorted:
            dt = esc(it.dt_utc.replace("+00:00", "Z"))
            title = esc(it.title)
            link = it.link or ""
            pub = esc(it.publisher)
            if link:
                items_rows.append(
                    f"<li><span class='meta'>[{pub}] {dt}</span> — "
                    f"<a href='{esc(link)}' target='_blank' rel='noopener'>{title}</a></li>"
                )
            else:
                items_rows.append(f"<li><span class='meta'>[{pub}] {dt}</span> — {title}</li>")
        all_titles = " ".join(it.title for it in items_sorted)
        all_text = esc(" ".join(it.title for it in items_sorted))
        cards_html.append(
            f"""
        <div class="story-card" data-country="{esc(st['country'])}" data-search="{all_text}">
            <div class="story-head">
                <div class="story-title">{esc(st['rep_title'])}</div>
                <div class="story-badges">
                    <span class="badge">coverage: {st['coverage']}</span>
                    <span class="badge">items: {st['n_items']}</span>
                    <span class="badge muted">{esc(st['country'])}</span>
                </div>
            </div>
            <div class="story-sub">
                <span class="meta">News sources: {esc(sources)}</span><br/>
                <span class='meta'>Time window: {first_hm} → {last_hm} (UTC)</span>
            </div>

            <details class="story-details">
                <summary>Show news</summary>
                <ul class="items">
                    {''.join(items_rows)}
                </ul>
            </details>
        </div>
        """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Stories — {day_str}</title>
<style>
:root {{
  --bg:#05070a;
  --panel:#101521;
  --card:#141b2a;
  --text:#f5f5f5;
  --muted:#a0aec0;
  --border:#263145;
  --accent:#4fd1c5;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 22px 22px;
  font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  background: var(--bg); color: var(--text);
}}
.meta {{ color: var(--muted); font-size: 0.85rem; }}
.story-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 14px;
  margin-bottom: 12px;
}}
.story-head {{
  display: flex; gap: 14px; justify-content: space-between; align-items: flex-start;
}}
.story-title {{
  font-size: 1.02rem;
  font-weight: 600;
  line-height: 1.25;
}}
.story-badges {{
  display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end;
}}
.badge {{
  border: 1px solid var(--border);
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 0.78rem;
}}
.badge.muted {{ color: var(--muted); }}
.story-sub {{ margin-top: 8px; }}
.story-details {{ margin-top: 10px; }}
.story-details summary {{
  cursor: pointer;
  color: var(--accent);
  font-size: 0.92rem;
}}
ul.items {{ margin: 10px 0 0 0; padding-left: 18px; }}
ul.items li {{ margin-bottom: 6px; line-height: 1.25; }}
a {{ color: var(--text); }}
a:hover {{ color: var(--accent); }}
</style>
</head>
<body>
<div id="stories">
{''.join(cards_html)}
</div>

<script>
// filtri arrivano dal wrapper via querystring: ?country=...&q=...
const params = new URLSearchParams(window.location.search);
const country = params.get('country') || 'all';
const q = (params.get('q') || '').toLowerCase().trim();

const cards = Array.from(document.querySelectorAll('.story-card'));

cards.forEach(card => {{
  const okCountry = (country === 'all') || (card.getAttribute('data-country') === country);
  const hay = (card.getAttribute('data-search') || '').toLowerCase();
  const okSearch = !q || hay.includes(q);
  card.style.display = (okCountry && okSearch) ? 'block' : 'none';
}});
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print("[INFO] Saved story content to:", out_path)


#
# 8 MAIN
#
if __name__ == "__main__":  # Questo è il modo standard in Python per dire: se eseguo questo file come script (python miofile.py) →
    # allora esegui il codice dentro se invece importo questo file come modulo (import miofile) →
    # non eseguire automaticamente questa parte
    try:
        BASE_DIR = Path(__file__).resolve().parent
    except NameError:
        BASE_DIR = Path.cwd()

    target_day = datetime.now(timezone.utc).date()  # oggi in UTC (credo sia = GMT)

    # prendiamo tutti i feed di oggi
    all_items: list[Item] = []
    for country, feeds in COUNTRY_FEEDS.items():
        print(f"\n=== Fetching {country} ({target_day.isoformat()} UTC) ===")
        items = fetch_items_for_country(country, feeds, target_day)
        print(f"[INFO] {country}: {len(items)} items after cleaning")
        all_items.extend(items)  # ricorda diff tra append ed extend

    # cluster (per country)
    stories = cluster_items_into_stories(
        items=all_items,
        stopwords_by_country=STOPWORDS_BY_COUNTRY,
        threshold=0.25,
        centroid_min_frac=0.6,  # tiene token che appaiono in almeno il 50% dei titoli del cluster
        soft_after=7  # il soft dopo 5 notizie aggregate
    )

    # 1) salviamo il contenuto giornaliero (solo cards)
    out_content = BASE_DIR / f"stories_{target_day.isoformat()}.html"
    save_story_content_html(out_content, target_day, stories)

    # 2) salviamo il wrapper (sidebar + iframe + day selector) => SEMPRE lo stesso file
    out_dashboard = BASE_DIR / "story_dashboard.html"
    save_story_dashboard_html(out_dashboard, target_day)
