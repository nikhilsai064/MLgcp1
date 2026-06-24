import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import  r2_score
print('t')

"""
McKesson MMS Scraper v3 — Fast, concurrent, junk-URL filtered
================================================================

Key improvements over v2:
  1. URL pre-filtering: skip /alternatives, /compare, ?src=AL, ?src=PR, etc.
     before making any HTTP request — these pages never have HCPCS codes.
  2. Thread-pool: N_WORKERS concurrent product-page fetches (default 8).
     Category crawl still runs single-threaded (polite), products run in parallel.
  3. Per-worker rate-limit: each worker sleeps MIN_DELAY–MAX_DELAY between
     requests, but N workers effectively divide total wall-clock time by N.
  4. Resume: re-running skips already-processed product URLs automatically.
  5. All previously confirmed real selectors from v2 are kept.
"""

import csv
import itertools
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config  — tune these
# ---------------------------------------------------------------------------

BASE_URL = "https://mms.mckesson.com"
START_URL = f"{BASE_URL}/shop-products"

HEADERS = {
    "User-Agent": "MyCompanyResearchBot/1.0 (+mailto:you@example.com)",
    "Accept-Language": "en-US,en;q=0.9",
}

N_WORKERS = 8            # parallel product-page fetchers; raise to 12-16 if no throttling
MIN_DELAY_SEC = 1.0      # per-worker sleep between requests
MAX_DELAY_SEC = 2.0
MAX_RETRIES = 3
REQUEST_TIMEOUT = 20

MAX_CATEGORY_PAGES = 10_000
CHECKPOINT_EVERY = 50    # save outputs every N products processed

OUTPUT_DIR = Path("mckesson_hcpcs_output")
HTML_DIR = OUTPUT_DIR / "raw_html"
STATE_FILE = OUTPUT_DIR / "crawl_state.json"
OUTPUT_CSV = OUTPUT_DIR / "hcpcs_products.csv"
OUTPUT_JSON = OUTPUT_DIR / "hcpcs_products.json"
DISCOVERED_FILE = OUTPUT_DIR / "discovered_product_urls.json"

PRODUCT_URL_RE = re.compile(r"/product/\d+/")
CATEGORY_URL_RE = re.compile(r"/catalog(/category)?\?node=")
HCPCS_CODE_RE = re.compile(r"\b[A-Z]\d{4}\b")

# URL path segments / query params that indicate non-product or junk pages
# — skip before even making an HTTP request
SKIP_PATH_RE = re.compile(
    r"/alternatives|/compare|/content/|/assets/|/login|"
    r"/cart|/account|/forms|/search|"
    r"\.(pdf|jpg|jpeg|png|gif|css|js|ico|svg|xml|zip)(\?|$)",
    re.IGNORECASE,
)
SKIP_SRC_PARAMS = {"AL", "PR", "RR"}   # ?src= values that appear on alternatives/related cards

# Faceted/filtered category URLs — the node param contains Base64-encoded filter
# state after a '+', creating infinite unique URLs for the same products.
# e.g. catalog?node=133373+eyJrZXki0iJNYW51ZmFjdHVyZXIg...
# We only want plain node IDs like catalog?node=12345 or catalog?node=12345+67890
FACET_TRAP_RE = re.compile(r"node=[^&]*\+[A-Za-z0-9+/=]{20,}")

VARIATION_LABEL_BLACKLIST = {
    "features", "product specifications", "more information",
    "other online resources", "related products", "frequently viewed together",
    "alternatives", "compare", "recently viewed", "customers also bought",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mckesson_hcpcs_scraper_v3")


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def is_junk_url(url: str) -> bool:
    parsed = urlparse(url)
    if SKIP_PATH_RE.search(parsed.path):
        return True
    src = parse_qs(parsed.query).get("src", [None])[0]
    if src and src.upper() in SKIP_SRC_PARAMS:
        return True
    # skip faceted/filtered category URLs — Base64-encoded filter state after '+'
    # e.g. node=133373+eyJrZXki0iJNYW51ZmFjdHVyZXIg... (infinite crawler trap)
    if FACET_TRAP_RE.search(url):
        return True
    # skip URLs where the path after /product/{id}/ contains noise keywords
    if "/product/" in parsed.path:
        tail = parsed.path.split("/product/", 1)[-1].lower()
        if any(kw in tail for kw in ("alternatives", "compare", "reviews")):
            return True
    return False


def clean_url(url: str) -> str:
    """Strip tracking params that create duplicate URLs for the same product."""
    parsed = urlparse(url)
    # keep only essential query params for category pages; strip src/back/etc. for product pages
    if PRODUCT_URL_RE.search(parsed.path):
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if CATEGORY_URL_RE.search(parsed.path):
        qs = parse_qs(parsed.query)
        node = qs.get("node", [""])[0]
        page = qs.get("page", [None])[0]
        q = f"node={node}" + (f"&page={page}" if page else "")
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{q}"
    return url


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------

_robots_cache: dict = {}

def robots_allows(url: str) -> bool:
    if BASE_URL not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(urljoin(BASE_URL, "/robots.txt"))
        try:
            rp.read()
            _robots_cache[BASE_URL] = rp
        except Exception as e:
            log.warning("Could not read robots.txt (%s) — defaulting to NOT crawling.", e)
            _robots_cache[BASE_URL] = None
    rp = _robots_cache[BASE_URL]
    return rp is not None and rp.can_fetch(HEADERS["User-Agent"], url)


# ---------------------------------------------------------------------------
# HTTP (one session per thread)
# ---------------------------------------------------------------------------

import threading
_thread_local = threading.local()

def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return _thread_local.session


def fetch_html(url: str) -> Optional[str]:
    if not robots_allows(url):
        return None
    session = _get_session()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 503):
                wait = 5 * attempt
                log.warning("HTTP %s on %s — backing off %ss", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            log.debug("HTTP %s for %s", resp.status_code, url)
            return None
        except requests.RequestException as e:
            log.warning("Request error (%s/%s) %s: %s", attempt, MAX_RETRIES, url, e)
            time.sleep(2 * attempt)
    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ProductCombinationRecord:
    item_name: str = ""
    manufacturer_number: str = ""
    manufacturer: str = ""
    brand: str = ""
    mckesson_number: str = ""
    hcpcs: str = ""
    country_of_origin: str = ""
    material: str = ""
    size: str = ""
    category_breadcrumb: str = ""
    features: list = field(default_factory=list)
    combination: dict = field(default_factory=dict)
    source_url: str = ""
    saved_html_path: str = ""
    all_specs: dict = field(default_factory=dict)
    scraped_at: str = ""


# ---------------------------------------------------------------------------
# Category crawl (single-threaded, polite)
# ---------------------------------------------------------------------------

def discover_links(soup: BeautifulSoup, current_url: str, pattern: re.Pattern) -> set:
    urls = set()
    for a in soup.find_all("a", href=True):
        raw = a["href"].split("#")[0].strip()
        if not raw:
            continue
        full = urljoin(current_url, raw)
        if urlparse(full).netloc not in ("", urlparse(BASE_URL).netloc):
            continue
        if pattern.search(full) and not is_junk_url(full):
            urls.add(clean_url(full))
    return urls


def find_pagination_urls(soup: BeautifulSoup, current_url: str) -> set:
    urls = set()
    for a in soup.select("a[href]"):
        text = a.get_text(strip=True).lower()
        rel = a.get("rel", [])
        classes = " ".join(a.get("class", [])).lower()
        if text in ("next", "next page", ">", "»") or "next" in rel or "next" in classes:
            u = clean_url(urljoin(current_url, a["href"]))
            if not is_junk_url(u):
                urls.add(u)
    # also try ?page=N pattern: if current URL has ?page=N, try N+1
    parsed = urlparse(current_url)
    qs = parse_qs(parsed.query)
    if "page" in qs:
        try:
            next_page = int(qs["page"][0]) + 1
            qs_next = {k: v for k, v in qs.items()}
            qs_next["page"] = [str(next_page)]
            from urllib.parse import urlencode
            next_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?" + \
                       "&".join(f"{k}={v[0]}" for k, v in qs_next.items())
            urls.add(next_url)
        except ValueError:
            pass
    return urls


def crawl_categories_for_products() -> set:
    seen_pages: set = set()
    to_visit = [START_URL]
    product_urls: set = set()

    while to_visit and len(seen_pages) < MAX_CATEGORY_PAGES:
        url = to_visit.pop(0)
        if url in seen_pages:
            continue
        seen_pages.add(url)

        log.info("Category crawl [%d/%d visited | %d products found]: %s",
                 len(seen_pages), MAX_CATEGORY_PAGES, len(product_urls), url)

        html = fetch_html(url)
        time.sleep(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC))
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")

        new_prods = discover_links(soup, url, PRODUCT_URL_RE)
        product_urls |= new_prods
        if new_prods:
            log.info("  -> +%d product links (total %d)", len(new_prods), len(product_urls))

        for u in (discover_links(soup, url, CATEGORY_URL_RE) | find_pagination_urls(soup, url)):
            if u not in seen_pages and u not in to_visit:
                to_visit.append(u)

        if len(seen_pages) % 50 == 0:
            _save_discovered(product_urls)

    _save_discovered(product_urls)
    log.info("Category crawl done: %d pages, %d product URLs", len(seen_pages), len(product_urls))
    return product_urls


def _save_discovered(urls: set):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DISCOVERED_FILE.write_text(json.dumps(sorted(urls), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Product page parsing
# ---------------------------------------------------------------------------

def parse_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1", class_="prod-title") or soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def parse_breadcrumb(soup: BeautifulSoup) -> str:
    nav = soup.find("nav", attrs={"aria-label": "breadcrumb"})
    if nav:
        parts = [li.get_text(strip=True) for li in nav.find_all("li") if li.get_text(strip=True)]
        return " > ".join(parts)
    return ""


def parse_spec_table(soup: BeautifulSoup) -> dict:
    specs = {}
    container = soup.find("div", id="specifications") or soup
    for table in container.find_all("table"):
        for row in table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td:
                k = th.get_text(strip=True)
                v = td.get_text(strip=True)
                if k:
                    specs[k] = v
    return specs


def parse_features(soup: BeautifulSoup) -> list:
    ul = soup.find("ul", class_="product-features")
    if ul:
        return [li.get_text(strip=True) for li in ul.find_all("li") if li.get_text(strip=True)]
    h = soup.find(lambda t: t.name in ("h2","h3","h4","h6") and t.get_text(strip=True).lower() == "features")
    if h:
        ul = h.find_next("ul")
        if ul:
            return [li.get_text(strip=True) for li in ul.find_all("li") if li.get_text(strip=True)]
    return []


def find_hcpcs(specs: dict) -> str:
    for k, v in specs.items():
        if k.strip().lower() == "hcpcs":
            return v.strip()
    for k, v in specs.items():
        if "hcpcs" in k.lower():
            m = HCPCS_CODE_RE.search(v)
            return m.group(0) if m else v.strip()
    return ""


def extract_known_fields(specs: dict) -> dict:
    def exact(*keys):
        for target in keys:
            for k, v in specs.items():
                if k.strip().lower() == target:
                    return v
        return ""
    return {
        "manufacturer_number": exact("manufacturer #", "manufacturer number", "mfg #"),
        "manufacturer": exact("manufacturer"),
        "brand": exact("brand"),
        "mckesson_number": exact("mckesson #", "mckesson number"),
        "country_of_origin": exact("country of origin"),
        "material": exact("material"),
        "size": exact("size"),
    }


def parse_variation_groups(soup: BeautifulSoup) -> list:
    product_detail = soup.find("div", class_="product-detail")
    if not product_detail:
        return []
    sidebar = product_detail.find("div", class_="product-detail-sidebar")
    specs_div = product_detail.find("div", id="specifications")

    groups = []
    seen_names: set = set()

    for label_tag in product_detail.find_all(["h2","h3","h4","h6","strong","b","label"]):
        if sidebar and label_tag.find_parent("div", class_="product-detail-sidebar"):
            continue
        if specs_div and label_tag.find_parent("div", id="specifications"):
            continue

        label_text = label_tag.get_text(strip=True)
        if not label_text or label_text.lower() in VARIATION_LABEL_BLACKLIST:
            continue
        if label_text in seen_names:
            continue

        group_container = label_tag.find_next_sibling()
        if not group_container:
            continue
        if "slider" in " ".join(group_container.get("class", [])).lower():
            continue

        options = [
            el.get_text(strip=True)
            for el in group_container.find_all(["button","a","span"])
            if el.get_text(strip=True)
            and "slider" not in " ".join(el.get("class", [])).lower()
        ]
        options = list(dict.fromkeys(options))
        if options:
            groups.append({"name": label_text, "options": options})
            seen_names.add(label_text)
    return groups


def all_combinations(groups: list) -> list:
    if not groups:
        return [{}]
    names = [g["name"] for g in groups]
    return [dict(zip(names, combo)) for combo in itertools.product(*[g["options"] for g in groups])]


def safe_filename(url: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", url.strip("/").split("//")[-1])
    return slug[:180] + ".html"


# ---------------------------------------------------------------------------
# Per-product pipeline (called from worker threads)
# ---------------------------------------------------------------------------

def process_product(url: str) -> list:
    if is_junk_url(url):
        return []

    time.sleep(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC))
    html = fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    specs = parse_spec_table(soup)
    hcpcs = find_hcpcs(specs)
    if not hcpcs:
        return []

    log.info("HCPCS '%s' found: %s", hcpcs, url)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    html_path = HTML_DIR / safe_filename(url)
    html_path.write_text(html, encoding="utf-8")

    known = extract_known_fields(specs)
    groups = parse_variation_groups(soup)
    combos = all_combinations(groups)
    now = datetime.now(timezone.utc).isoformat()

    records = []
    for combo in combos:
        records.append(ProductCombinationRecord(
            item_name=parse_title(soup),
            manufacturer_number=known["manufacturer_number"],
            manufacturer=known["manufacturer"],
            brand=known["brand"],
            mckesson_number=known["mckesson_number"],
            hcpcs=hcpcs,
            country_of_origin=known["country_of_origin"],
            material=known["material"],
            size=known["size"],
            category_breadcrumb=parse_breadcrumb(soup),
            features=parse_features(soup),
            combination=combo,
            source_url=url,
            saved_html_path=str(html_path),
            all_specs=specs,
            scraped_at=now,
        ))
    return records


# ---------------------------------------------------------------------------
# Output (thread-safe via lock)
# ---------------------------------------------------------------------------

_output_lock = threading.Lock()

def save_outputs(records: list):
    rows = [asdict(r) for r in records]
    with _output_lock:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        if rows:
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for row in rows:
                    row = dict(row)
                    row["features"] = "; ".join(row["features"])
                    row["combination"] = json.dumps(row["combination"], ensure_ascii=False)
                    row["all_specs"] = json.dumps(row["all_specs"], ensure_ascii=False)
                    writer.writerow(row)
    log.info("Checkpoint: %d records saved.", len(records))


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"processed_product_urls": []}


def save_state(done: set):
    with _output_lock:
        STATE_FILE.write_text(json.dumps({"processed_product_urls": sorted(done)}, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(resume: bool = True):
    if not robots_allows(START_URL):
        log.error("robots.txt disallows crawling. Confirm access before proceeding.")
        return

    # ── 1. Category crawl (or reuse previous run) ─────────────────────────
    if resume and DISCOVERED_FILE.exists():
        log.info("Loading previously discovered product URLs from %s", DISCOVERED_FILE)
        product_urls = set(json.loads(DISCOVERED_FILE.read_text(encoding="utf-8")))
        log.info("Loaded %d product URLs.", len(product_urls))
    else:
        product_urls = crawl_categories_for_products()

    # Pre-filter junk before even spawning workers
    product_urls = {clean_url(u) for u in product_urls if not is_junk_url(u)}
    log.info("After filtering junk URLs: %d clean product URLs.", len(product_urls))

    # ── 2. Resume: skip already-processed ──────────────────────────────────
    state = load_state() if resume else {"processed_product_urls": []}
    already_done: set = set(state["processed_product_urls"])
    remaining = sorted(product_urls - already_done)
    log.info("Remaining to process: %d  (already done: %d)", len(remaining), len(already_done))

    # ── 3. Reload existing results so checkpoints append correctly ─────────
    all_records: list = []
    if resume and OUTPUT_JSON.exists():
        try:
            raw = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
            all_records = [ProductCombinationRecord(**r) for r in raw]
            log.info("Loaded %d existing records from previous run.", len(all_records))
        except Exception as e:
            log.warning("Could not reload previous results: %s", e)

    done_lock = threading.Lock()

    # ── 4. Parallel product scrape ─────────────────────────────────────────
    log.info("Starting %d workers across %d URLs…", N_WORKERS, len(remaining))
    processed_count = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_product, url): url for url in remaining}
        for future in as_completed(futures):
            url = futures[future]
            try:
                records = future.result()
            except Exception as e:
                log.error("Error on %s: %s", url, e)
                records = []

            with done_lock:
                all_records.extend(records)
                already_done.add(url)
                processed_count += 1

                if processed_count % CHECKPOINT_EVERY == 0:
                    hcpcs_count = len({r.source_url for r in all_records})
                    log.info("Progress: %d/%d checked | %d with HCPCS | %d combination rows",
                             processed_count, len(remaining), hcpcs_count, len(all_records))
                    save_outputs(all_records)
                    save_state(already_done)

    save_outputs(all_records)
    save_state(already_done)
    log.info("Done. %d products had HCPCS codes, %d total rows.",
              len({r.source_url for r in all_records}), len(all_records))


if __name__ == "__main__":
    main(resume=True)
