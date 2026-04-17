
# pip install requests beautifulsoup4 tqdm Pillow google-api-python-client transformers torch
"""
Digitisation of Historical South Asian Inscriptions and Manuscripts
Team CDV01 - RV College of Engineering

Multi-source scraper with four phases:
1) Category verification with fallback resolution
2) Smart downloading with filtering, retries, deduplication, parallel downloads
3) Organized storage with clean filenames
4) Final JSON and text reports

Image sources (in priority order):
  1. Wikimedia Commons categories
  2. Wikimedia Commons search
  3. Google Custom Search API  (set GOOGLE_API_KEY + GOOGLE_CSE_ID env vars)

AI accuracy filter:
  CLIP (openai/clip-vit-base-patch32) rejects non-inscription images.
  Requires: pip install transformers torch
  Falls back gracefully if not installed.
"""

import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm


# ---------------------------
# Configuration and constants
# ---------------------------
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "InscriptionScraper/1.0 (CDV01 RVCollegeEngineering; educational research)"
REQUEST_DELAY_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_RETRIES = 3
MAX_HTTP_RETRIES = 5
TARGET_PER_LANGUAGE = 200
MIN_TOTAL_TARGET = 1500
MAX_TOTAL_TARGET = 2000
MAX_SUBCATEGORY_DEPTH = 2
IMAGEINFO_BATCH_SIZE = 50
PARALLEL_DOWNLOAD_WORKERS = 8
CLIP_SCORE_THRESHOLD = 0.22

# API keys — set these as environment variables before running
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID  = os.environ.get("GOOGLE_CSE_ID", "")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
BANNED_KEYWORDS = {
    "map",
    "diagram",
    "drawing",
    "sketch",
    "chart",
    "illustration",
    "logo",
    "icon",
    "modern",
    "painting",
}
BANNED_NOISE_TERMS = {
    "coat of arms",
    "stamp",
    "seal",
    "poster",
    "book cover",
    "banner",
    "flag",
}
RELEVANT_KEYWORDS = {
    "inscription",
    "manuscript",
    "palm",
    "leaf",
    "stone",
    "epigraph",
    "copper",
    "plate",
    "rock",
    "stele",
    "slab",
    "engraving",
    "grantha",
    "brahmi",
    "devanagari",
    "tamil",
    "kannada",
    "telugu",
    "sanskrit",
    "malayalam",
    "bengali",
    "odia",
}
BANNED_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in sorted(BANNED_KEYWORDS)) + r")\b",
    re.IGNORECASE,
)

LANGUAGE_CONFIG = [
    {
        "key": "tamil",
        "display": "Tamil",
        "folder": "01_tamil",
        "primary": ["Category:Tamil_inscriptions"],
        "backup": [
            "Category:Tamil_Brahmi_inscriptions",
            "Category:Palm_leaf_manuscripts_in_Tamil",
        ],
    },
    {
        "key": "kannada",
        "display": "Kannada",
        "folder": "02_kannada",
        "primary": ["Category:Kannada_inscriptions"],
        "backup": [
            "Category:Kannada_palm_leaf_manuscripts",
            "Category:Hoysala_inscriptions",
        ],
    },
    {
        "key": "telugu",
        "display": "Telugu",
        "folder": "03_telugu",
        "primary": ["Category:Telugu_inscriptions"],
        "backup": ["Category:Telugu_palm_leaf_manuscripts"],
    },
    {
        "key": "sanskrit",
        "display": "Sanskrit",
        "folder": "04_sanskrit",
        "primary": ["Category:Sanskrit_inscriptions"],
        "backup": [
            "Category:Sanskrit_palm_leaf_manuscripts",
            "Category:Sanskrit_manuscripts",
        ],
    },
    {
        "key": "malayalam",
        "display": "Malayalam",
        "folder": "05_malayalam",
        "primary": ["Category:Malayalam_inscriptions"],
        "backup": ["Category:Malayalam_palm_leaf_manuscripts"],
    },
    {
        "key": "hindi_devanagari",
        "display": "Hindi (Devanagari)",
        "folder": "06_hindi_devanagari",
        "primary": ["Category:Devanagari_inscriptions"],
        "backup": ["Category:Devanagari_manuscripts"],
    },
    {
        "key": "bengali",
        "display": "Bengali",
        "folder": "07_bengali",
        "primary": ["Category:Bengali_inscriptions"],
        "backup": ["Category:Bengali_manuscripts"],
    },
    {
        "key": "odia",
        "display": "Odia",
        "folder": "08_odia",
        "primary": ["Category:Odia_inscriptions"],
        "backup": ["Category:Odia_palm_leaf_manuscripts"],
    },
    {
        "key": "grantha",
        "display": "Grantha",
        "folder": "09_grantha",
        "primary": ["Category:Grantha_inscriptions"],
        "backup": ["Category:Grantha_script"],
    },
    {
        "key": "brahmi",
        "display": "Brahmi",
        "folder": "10_brahmi",
        "primary": ["Category:Brahmi_inscriptions"],
        "backup": [
            "Category:Brahmi_script",
            "Category:Inscriptions_in_Brahmi_script",
        ],
    },
]

SEARCH_QUERY_MAP = {
    "tamil": [
        "filetype:bitmap tamil inscription",
        "filetype:bitmap tamil palm leaf manuscript",
        "filetype:bitmap tamil epigraphy stone",
    ],
    "kannada": [
        "filetype:bitmap kannada inscription",
        "filetype:bitmap kannada palm leaf manuscript",
        "filetype:bitmap kannada epigraphy stone",
    ],
    "telugu": [
        "filetype:bitmap telugu inscription",
        "filetype:bitmap telugu palm leaf manuscript",
        "filetype:bitmap telugu epigraphy stone",
    ],
    "sanskrit": [
        "filetype:bitmap sanskrit inscription",
        "filetype:bitmap sanskrit manuscript",
        "filetype:bitmap devanagari sanskrit manuscript",
    ],
    "malayalam": [
        "filetype:bitmap malayalam inscription",
        "filetype:bitmap malayalam palm leaf manuscript",
        "filetype:bitmap malayalam epigraphy",
    ],
    "hindi_devanagari": [
        "filetype:bitmap devanagari inscription",
        "filetype:bitmap devanagari manuscript",
        "filetype:bitmap hindi epigraphy inscription",
    ],
    "bengali": [
        "filetype:bitmap bengali inscription",
        "filetype:bitmap bengali manuscript",
        "filetype:bitmap bengali palm leaf",
    ],
    "odia": [
        "filetype:bitmap odia inscription",
        "filetype:bitmap odia palm leaf manuscript",
        "filetype:bitmap odia epigraphy",
    ],
    "grantha": [
        "filetype:bitmap grantha inscription",
        "filetype:bitmap grantha manuscript",
        "filetype:bitmap grantha script epigraphy",
    ],
    "brahmi": [
        "filetype:bitmap brahmi inscription",
        "filetype:bitmap brahmi script inscription",
        "filetype:bitmap ashokan brahmi inscription",
    ],
}

# Internet search queries for Google CSE and Bing Image Search
INTERNET_SEARCH_QUERY_MAP: Dict[str, List[str]] = {
    "tamil": [
        "ancient Tamil stone inscription epigraphy carved text",
        "Tamil palm leaf manuscript historical writing",
        "Tamil Brahmi rock edict inscription ancient",
        "Tamil copper plate inscription ancient script",
    ],
    "kannada": [
        "ancient Kannada stone inscription epigraphy carved",
        "Kannada palm leaf manuscript historical",
        "Hoysala Kannada inscription ancient script",
        "Karnataka rock inscription ancient Kannada",
    ],
    "telugu": [
        "ancient Telugu stone inscription epigraphy",
        "Telugu palm leaf manuscript ancient writing",
        "Telugu rock inscription ancient carved text",
        "Andhra Pradesh ancient Telugu inscription",
    ],
    "sanskrit": [
        "Sanskrit stone inscription ancient epigraphy carved",
        "Sanskrit palm leaf manuscript ancient text",
        "Sanskrit copper plate inscription historical",
        "Sanskrit rock edict ancient Devanagari carved",
    ],
    "malayalam": [
        "ancient Malayalam stone inscription epigraphy",
        "Malayalam palm leaf manuscript ancient writing",
        "Kerala ancient inscription carved rock script",
        "Grantha Malayalam copper plate inscription",
    ],
    "hindi_devanagari": [
        "Devanagari ancient stone inscription epigraphy",
        "Hindi Sanskrit manuscript ancient Devanagari text",
        "ancient Devanagari rock inscription carved India",
        "Nagari script ancient inscription historical",
    ],
    "bengali": [
        "ancient Bengali stone inscription epigraphy carved",
        "Bengali palm leaf manuscript ancient writing",
        "Bengal ancient inscription rock carved script",
        "Bengali copper plate inscription historical",
    ],
    "odia": [
        "ancient Odia stone inscription epigraphy carved",
        "Odia palm leaf manuscript historical ancient",
        "Odisha ancient inscription carved rock script",
        "Kalinga inscription ancient Odia script",
    ],
    "grantha": [
        "Grantha script inscription ancient stone carved",
        "Grantha manuscript ancient writing Tamil Nadu",
        "Pallava Grantha inscription ancient carved",
        "Grantha palm leaf manuscript historical",
    ],
    "brahmi": [
        "Brahmi script ancient inscription stone carved",
        "Ashokan Brahmi rock edict inscription ancient",
        "ancient Brahmi inscription archaeological India",
        "Brahmi script carved stone pillar edict",
    ],
}


# ------------------------------------------
# Internet image search sources (Google/Bing)
# ------------------------------------------
_internet_session: Optional[requests.Session] = None


def _get_internet_session() -> requests.Session:
    global _internet_session
    if _internet_session is None:
        _internet_session = requests.Session()
        _internet_session.headers.update({"User-Agent": USER_AGENT})
    return _internet_session


def iter_google_image_results(query: str) -> Generator[Tuple[str, str], None, None]:
    """Yield (image_url, context_text) from Google Custom Search image API.

    Requires GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.
    Up to 100 results per query (API hard limit).
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return
    try:
        from googleapiclient.discovery import build  # type: ignore
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY, cache_discovery=False)
    except Exception:
        return

    for start in range(1, 100, 10):  # 1,11,21,...,91 → 10 pages × 10 results
        try:
            result = (
                service.cse()
                .list(
                    q=query,
                    cx=GOOGLE_CSE_ID,
                    searchType="image",
                    imgType="photo",
                    num=10,
                    start=start,
                    safe="off",
                )
                .execute()
            )
        except Exception:
            break
        items = result.get("items", [])
        if not items:
            break
        for item in items:
            url = item.get("link", "")
            ctx = f"{item.get('title', '')} {item.get('snippet', '')} {item.get('displayLink', '')}"
            if url:
                yield url, ctx.lower()


def iter_internet_images(lang_key: str) -> Generator[Tuple[str, str], None, None]:
    """Yield (url, context_text) from Google CSE for one language, deduplicated."""
    queries = INTERNET_SEARCH_QUERY_MAP.get(lang_key, [])
    seen_urls: Set[str] = set()
    for query in queries:
        for url, ctx in iter_google_image_results(query):
            if url not in seen_urls:
                seen_urls.add(url)
                yield url, ctx


class WikimediaClient:
    """Small Wikimedia API client with a strict 1.5-second request interval."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.last_request_time = 0.0

    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)

    def get(self, params: Dict, stream: bool = False) -> requests.Response:
        """Execute one HTTP GET with consistent throttling and timeout."""
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_HTTP_RETRIES + 1):
            try:
                self._throttle()
                response = self.session.get(
                    API_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    stream=stream,
                )
                self.last_request_time = time.time()
                if response.status_code in {429, 500, 502, 503, 504}:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        retry_after_seconds = int(retry_after) if retry_after else 0
                    except ValueError:
                        retry_after_seconds = 0
                    wait_s = max((REQUEST_DELAY_SECONDS * attempt), retry_after_seconds, 3)
                    time.sleep(wait_s + random.uniform(0.2, 0.8))
                    last_error = requests.HTTPError(f"retryable_http_{response.status_code}")
                    continue
                response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= MAX_HTTP_RETRIES:
                    break
                time.sleep((REQUEST_DELAY_SECONDS * attempt) + random.uniform(0.2, 0.8))
        raise RuntimeError(f"API request failed after {MAX_HTTP_RETRIES} attempts: {last_error}")

    def get_url(self, url: str, stream: bool = True) -> requests.Response:
        """Execute one direct URL GET with the same throttling policy."""
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_HTTP_RETRIES + 1):
            try:
                self._throttle()
                response = self.session.get(
                    url,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    stream=stream,
                )
                self.last_request_time = time.time()
                if response.status_code in {429, 500, 502, 503, 504}:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        retry_after_seconds = int(retry_after) if retry_after else 0
                    except ValueError:
                        retry_after_seconds = 0
                    wait_s = max((REQUEST_DELAY_SECONDS * attempt), retry_after_seconds, 3)
                    time.sleep(wait_s + random.uniform(0.2, 0.8))
                    last_error = requests.HTTPError(f"retryable_http_{response.status_code}")
                    continue
                response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= MAX_HTTP_RETRIES:
                    break
                time.sleep((REQUEST_DELAY_SECONDS * attempt) + random.uniform(0.2, 0.8))
        raise RuntimeError(f"URL request failed after {MAX_HTTP_RETRIES} attempts: {last_error}")


# ------------------------------------------
# CLIP-based AI image classifier (optional)
# ------------------------------------------
class CLIPClassifier:
    """Uses CLIP to score images against inscription/manuscript text prompts.

    Requires: pip install transformers torch
    If torch/transformers are not installed, instantiation raises ImportError
    and the caller degrades gracefully by setting clip_classifier = None.
    """

    POSITIVE_PROMPTS = [
        "ancient stone inscription carved text epigraphy",
        "palm leaf manuscript ancient writing historical",
        "historical rock edict carved ancient script letters",
        "old copper plate inscription ancient text",
        "archaeological engraved stone ancient language",
    ]
    NEGATIVE_PROMPTS = [
        "modern photograph person portrait selfie",
        "landscape nature scenery tourist photo",
        "painting artwork illustration drawing",
        "building architecture modern structure",
        "food product commercial advertisement",
    ]

    def __init__(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        self._pos_features = self._encode_texts(self.POSITIVE_PROMPTS)
        self._neg_features = self._encode_texts(self.NEGATIVE_PROMPTS)

    def _encode_texts(self, texts: List[str]):
        import torch  # noqa: PLC0415
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def score(self, image_bytes: bytes) -> float:
        """Return max positive similarity minus max negative similarity (-1..1)."""
        import torch  # noqa: PLC0415
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return -1.0
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_features = self.model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        pos_sim = (img_features @ self._pos_features.T).squeeze(0).max().item()
        neg_sim = (img_features @ self._neg_features.T).squeeze(0).max().item()
        return float(pos_sim - neg_sim)

    def is_inscription(self, image_bytes: bytes) -> bool:
        return self.score(image_bytes) >= CLIP_SCORE_THRESHOLD


# --------------------------------------
# Phase 1 - category counting and verify
# --------------------------------------
def iterate_category_file_titles(client: WikimediaClient, category: str) -> Generator[str, None, None]:
    """Yield all file titles (File:...) directly listed in a Wikimedia Commons category."""
    cont = None
    consecutive_failures = 0
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file",
            "cmlimit": "500",
        }
        if cont:
            params["cmcontinue"] = cont

        try:
            data = client.get(params).json()
            consecutive_failures = 0
        except Exception:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
            continue
        members = data.get("query", {}).get("categorymembers", [])
        for item in members:
            title = item.get("title", "")
            if title.startswith("File:"):
                yield title

        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break


def iterate_category_subcategories(client: WikimediaClient, category: str) -> Generator[str, None, None]:
    """Yield subcategory titles (Category:...) for one category."""
    cont = None
    consecutive_failures = 0
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "subcat",
            "cmlimit": "500",
        }
        if cont:
            params["cmcontinue"] = cont

        try:
            data = client.get(params).json()
            consecutive_failures = 0
        except Exception:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
            continue
        members = data.get("query", {}).get("categorymembers", [])
        for item in members:
            title = item.get("title", "")
            if title.startswith("Category:"):
                yield title

        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break


def get_category_tree_file_set(
    client: WikimediaClient,
    category: str,
    max_depth: int = MAX_SUBCATEGORY_DEPTH,
) -> Set[str]:
    """Return file titles from a category tree as a set."""
    return set(iterate_category_tree_file_titles(client, category, max_depth=max_depth))


def iterate_category_tree_file_titles(
    client: WikimediaClient,
    category: str,
    max_depth: int = MAX_SUBCATEGORY_DEPTH,
) -> Generator[str, None, None]:
    """Yield file titles from a category and its subcategories up to max_depth."""
    visited_categories: Set[str] = set()
    queue: List[Tuple[str, int]] = [(category, 0)]

    while queue:
        current_category, depth = queue.pop(0)
        if current_category in visited_categories:
            continue
        visited_categories.add(current_category)

        for file_title in iterate_category_file_titles(client, current_category):
            yield file_title

        if depth < max_depth:
            for subcat in iterate_category_subcategories(client, current_category):
                if subcat not in visited_categories:
                    queue.append((subcat, depth + 1))

    return


def get_category_file_set(client: WikimediaClient, category: str) -> Set[str]:
    """Return all file titles in a category as a set for count and dedup calculations."""
    return get_category_tree_file_set(client, category, max_depth=MAX_SUBCATEGORY_DEPTH)


def get_category_file_count_fast(client: WikimediaClient, category: str) -> int:
    """Return file count for a category using categoryinfo (single API call)."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "categoryinfo",
        "titles": category,
    }
    try:
        data = client.get(params).json()
    except Exception:
        return 0
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return 0
    page = next(iter(pages.values()))
    categoryinfo = page.get("categoryinfo", {}) or {}
    # Wikimedia reports direct file count in this category as "files".
    return int(categoryinfo.get("files", 0) or 0)


def print_count_table(rows: List[Tuple[str, int]]) -> None:
    """Print a clean fixed-width table for category/image counts."""
    title_width = max(len("Category"), *(len(name) for name, _ in rows))
    count_width = max(len("Image Count"), *(len(str(count)) for _, count in rows))

    border = "+-" + "-" * title_width + "-+-" + "-" * count_width + "-+"
    print(border)
    print(f"| {'Category'.ljust(title_width)} | {'Image Count'.rjust(count_width)} |")
    print(border)
    for name, count in rows:
        print(f"| {name.ljust(title_width)} | {str(count).rjust(count_width)} |")
    print(border)


def verify_language_categories(client: WikimediaClient) -> Dict[str, Dict]:
    """
    Verify primary categories, apply fallback categories if needed,
    and return resolved category plans for downloading.
    """
    print("\nPHASE 1 - Category Verification")
    print("Checking Wikimedia Commons category image counts...\n")

    # Print per-category primary counts first (fast metadata-based count).
    primary_rows: List[Tuple[str, int]] = []
    primary_counts_by_key: Dict[str, int] = {}

    for cfg in LANGUAGE_CONFIG:
        primary_category = cfg["primary"][0]
        primary_count = get_category_file_count_fast(client, primary_category)
        primary_counts_by_key[cfg["key"]] = primary_count
        primary_rows.append((primary_category, primary_count))

    print("Primary category counts:")
    print_count_table(primary_rows)

    # Build resolved plan per language.
    resolved: Dict[str, Dict] = {}
    print("\nResolving fallback categories for languages with fewer than 200 primary images...\n")

    final_rows: List[Tuple[str, int]] = []
    for cfg in LANGUAGE_CONFIG:
        key = cfg["key"]
        display = cfg["display"]
        primary_categories = cfg["primary"]
        backup_categories = cfg["backup"]

        used_categories = list(primary_categories)
        combined_count = primary_counts_by_key[key]

        if primary_counts_by_key[key] < TARGET_PER_LANGUAGE:
            for backup_cat in backup_categories:
                backup_count = get_category_file_count_fast(client, backup_cat)
                combined_count += backup_count
                used_categories.append(backup_cat)

        final_count = combined_count
        resolved[key] = {
            "display": display,
            "folder": cfg["folder"],
            "used_categories": used_categories,
            "final_count": final_count,
        }
        final_rows.append((display, final_count))

    print("Final verified counts per language (after fallback resolution):")
    print_count_table(final_rows)

    print("\nResolved category sources per language:")
    for cfg in LANGUAGE_CONFIG:
        key = cfg["key"]
        entry = resolved[key]
        print(f"- {entry['display']}: {', '.join(entry['used_categories'])}")

    try:
        input("\nPress Enter to proceed with downloading... ")
    except EOFError:
        print("No interactive input detected. Continuing automatically...")
    return resolved


# ------------------------------
# Shared helpers for phase 2/3
# ------------------------------
def is_allowed_extension(title: str) -> bool:
    """Check if Wikimedia file title has an allowed image extension."""
    filename = title.lower().strip()
    for ext in ALLOWED_EXTENSIONS:
        if filename.endswith(ext):
            return True
    return False


def clean_html_text(value: Optional[str]) -> str:
    """Convert HTML-ish metadata text into plain lowercase text for filtering."""
    if not value:
        return ""
    text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_banned_keywords(*texts: str) -> bool:
    """Return True if any banned keyword appears in provided lowercase text blocks."""
    merged = " ".join(texts).lower()
    if BANNED_KEYWORD_PATTERN.search(merged):
        return True
    return any(term in merged for term in BANNED_NOISE_TERMS)


def appears_relevant_inscription_or_manuscript(*texts: str) -> bool:
    """Keep files whose title/description suggests inscription/manuscript content."""
    merged = " ".join(texts).lower()
    return any(keyword in merged for keyword in RELEVANT_KEYWORDS)


def get_image_info_for_title(client: WikimediaClient, file_title: str) -> Optional[Dict]:
    """Fetch width/height/url/description metadata for a single Wikimedia Commons file title."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": file_title,
        "iiprop": "url|size|extmetadata",
    }

    try:
        data = client.get(params).json()
    except Exception:
        return None
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    imageinfo = page.get("imageinfo", [])
    if not imageinfo:
        return None

    info = imageinfo[0]
    ext = info.get("extmetadata", {}) or {}
    description = clean_html_text((ext.get("ImageDescription") or {}).get("value", ""))

    return {
        "title": file_title,
        "url": info.get("url"),
        "width": int(info.get("width", 0) or 0),
        "height": int(info.get("height", 0) or 0),
        "description": description,
    }


def get_image_info_batch(client: WikimediaClient, titles: List[str]) -> Dict[str, Dict]:
    """Fetch imageinfo metadata for up to 50 file titles in a single API call.

    Returns a dict keyed by file title. Titles with no usable imageinfo are omitted.
    """
    if not titles:
        return {}
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": "|".join(titles),
        "iiprop": "url|size|extmetadata",
    }
    try:
        data = client.get(params).json()
    except Exception:
        return {}

    result: Dict[str, Dict] = {}
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        title = page.get("title", "")
        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            continue
        info = imageinfo[0]
        ext = info.get("extmetadata", {}) or {}
        description = clean_html_text((ext.get("ImageDescription") or {}).get("value", ""))
        result[title] = {
            "title": title,
            "url": info.get("url"),
            "width": int(info.get("width", 0) or 0),
            "height": int(info.get("height", 0) or 0),
            "description": description,
        }
    return result


def iter_titles_from_categories(client: WikimediaClient, categories: List[str]) -> Generator[str, None, None]:
    """Yield de-duplicated file titles across multiple categories in sequence."""
    seen: Set[str] = set()
    for category in categories:
        for title in iterate_category_tree_file_titles(client, category, max_depth=MAX_SUBCATEGORY_DEPTH):
            if title not in seen:
                seen.add(title)
                yield title


def iterate_titles_from_search_query(client: WikimediaClient, query: str) -> Generator[str, None, None]:
    """Yield file titles from Wikimedia search for File namespace using one query."""
    offset = 0
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srnamespace": "6",
            "srlimit": "50",
            "sroffset": str(offset),
        }
        try:
            data = client.get(params).json()
        except Exception:
            break
        results = data.get("query", {}).get("search", [])
        if not results:
            break

        for item in results:
            title = item.get("title", "")
            if title.startswith("File:"):
                yield title

        if "continue" not in data:
            break
        offset = int(data.get("continue", {}).get("sroffset", 0) or 0)


def iter_titles_from_search_queries(client: WikimediaClient, queries: List[str]) -> Generator[str, None, None]:
    """Yield de-duplicated file titles from multiple Wikimedia search queries."""
    seen: Set[str] = set()
    for query in queries:
        for title in iterate_titles_from_search_query(client, query):
            if title not in seen:
                seen.add(title)
                yield title


def make_output_paths() -> Path:
    """Create data/raw/inscription_dataset and return its Path object."""
    dataset_root = Path("data") / "raw"
    dataset_root.mkdir(parents=True, exist_ok=True)
    return dataset_root


def compute_md5(content: bytes) -> str:
    """Compute MD5 digest for duplicate image detection."""
    return hashlib.md5(content).hexdigest()


def choose_extension_from_title(title: str) -> str:
    """Pick output extension based on source title; normalize .jpeg to .jpg."""
    suffix = Path(title).suffix.lower()
    if suffix == ".jpeg":
        return ".jpg"
    if suffix in ALLOWED_EXTENSIONS:
        return suffix
    return ".jpg"


def download_image_bytes(client: WikimediaClient, url: str) -> bytes:
    """Download image bytes via WikimediaClient (throttled) and return raw bytes."""
    last_error: Optional[Exception] = None
    for _ in range(MAX_DOWNLOAD_RETRIES):
        try:
            resp = client.get_url(url, stream=True)
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Failed after {MAX_DOWNLOAD_RETRIES} retries: {last_error}")


def download_url_direct(url: str) -> bytes:
    """Download image bytes from any URL using the shared internet session (no throttle)."""
    session = _get_internet_session()
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < MAX_DOWNLOAD_RETRIES:
                time.sleep(attempt * 0.5)
    raise RuntimeError(f"Failed after {MAX_DOWNLOAD_RETRIES} retries: {last_error}")


def validate_image_bytes(content: bytes) -> Tuple[bool, str]:
    """Ensure bytes can be opened as an image and satisfy minimum 300x300 resolution."""
    try:
        with Image.open(BytesIO(content)) as img:
            width, height = img.size
            if width < 300 or height < 300:
                return False, f"resolution_too_small_{width}x{height}"
    except Exception:  # noqa: BLE001
        return False, "invalid_image_data"
    return True, "ok"


def save_image(content: bytes, out_path: Path) -> int:
    """Write image bytes to disk and return written file size in bytes."""
    out_path.write_bytes(content)
    return out_path.stat().st_size


# -------------------------
# Phase 2/3 - downloading
# -------------------------
def _finalize_candidate(
    content: bytes,
    url: str,
    label: str,
    lang_key: str,
    downloaded_count: int,
    target_folder: Path,
    global_hashes: Set[str],
    add_skip,
    clip_classifier: Optional["CLIPClassifier"],
) -> Tuple[bool, int]:
    """Validate, CLIP-check, dedup and save already-downloaded image bytes.

    Returns (True, size_bytes) on success, (False, 0) if rejected.
    ``label`` is used as the filename hint (title or URL).
    """
    ok, reason = validate_image_bytes(content)
    if not ok:
        add_skip(reason, url=url, title=label)
        return False, 0

    if clip_classifier is not None:
        try:
            if not clip_classifier.is_inscription(content):
                add_skip("clip_rejected", url=url, title=label)
                return False, 0
        except Exception:  # noqa: BLE001
            pass  # don't crash the whole run on CLIP errors

    content_md5 = compute_md5(content)
    if content_md5 in global_hashes:
        add_skip("duplicate_md5", url=url, title=label)
        return False, 0
    global_hashes.add(content_md5)

    ext = Path(label).suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".jpg"
    clean_name = f"{lang_key}_{downloaded_count + 1:03d}{ext}"
    out_path = target_folder / clean_name
    size_bytes = save_image(content, out_path)
    return True, size_bytes


def download_language_dataset(
    client: WikimediaClient,
    cfg: Dict,
    resolved_cfg: Dict,
    dataset_root: Path,
    global_hashes: Set[str],
    skipped_log: List[Dict],
    clip_classifier: Optional["CLIPClassifier"] = None,
) -> Dict:
    """Download exactly 200 images for one language from all available sources.

    Sources tried in order:
      1. Wikimedia categories (batched imageinfo, parallel downloads)
      2. Wikimedia search queries (batched imageinfo, parallel downloads)
      3. Google CSE + Bing Image Search (parallel downloads, if API keys set)
    """
    lang_key = cfg["key"]
    lang_display = cfg["display"]
    folder_name = resolved_cfg["folder"]
    categories = resolved_cfg["used_categories"]
    wiki_search_queries = SEARCH_QUERY_MAP.get(lang_key, [])

    target_folder = dataset_root / folder_name
    target_folder.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    saved_bytes = 0
    skip_reason_counts: Dict[str, int] = {}
    processed_titles: Set[str] = set()
    processed_urls: Set[str] = set()

    def add_skip(reason: str, url: str = "", title: str = "") -> None:
        nonlocal skipped
        skipped += 1
        skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
        skipped_log.append({"language": lang_display, "url": url, "title": title, "reason": reason})

    progress = tqdm(total=TARGET_PER_LANGUAGE, desc=f"{lang_display}", unit="img", leave=True)

    def _update_progress() -> None:
        progress.set_postfix(downloaded=downloaded, skipped=skipped, refresh=False)

    def run_wikimedia_source(title_iter: Generator, from_search: bool) -> None:
        """Drain a Wikimedia title iterator: batch imageinfo → parallel download."""
        nonlocal downloaded, saved_bytes
        buffer: List[str] = []

        def flush_buffer() -> None:
            nonlocal downloaded, saved_bytes
            if not buffer:
                return
            batch_info = get_image_info_batch(client, buffer)

            # Pre-filter: keyword checks (free, no I/O) to build download list
            candidates: List[Tuple[str, str]] = []  # (file_title, url)
            for ft in buffer:
                if downloaded + len(candidates) >= TARGET_PER_LANGUAGE:
                    break
                info = batch_info.get(ft)
                if not info or not info.get("url"):
                    add_skip("missing_image_info", title=ft)
                    continue
                fname = ft.replace("File:", "", 1)
                title_text = fname.lower()
                desc_text = info.get("description", "")
                if contains_banned_keywords(title_text, desc_text):
                    add_skip("banned_keyword", url=info["url"], title=ft)
                    continue
                if from_search and not appears_relevant_inscription_or_manuscript(title_text, desc_text):
                    add_skip("not_relevant_to_inscription_or_manuscript", url=info["url"], title=ft)
                    continue
                if info["width"] < 300 or info["height"] < 300:
                    add_skip(f"resolution_too_small_{info['width']}x{info['height']}", url=info["url"], title=ft)
                    continue
                candidates.append((ft, info["url"]))

            if not candidates:
                buffer.clear()
                return

            # Parallel download of candidates
            future_to_meta: Dict = {}
            with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_WORKERS) as pool:
                for ft, url in candidates:
                    future = pool.submit(download_image_bytes, client, url)
                    future_to_meta[future] = (ft, url)

                for future in as_completed(future_to_meta):
                    if downloaded >= TARGET_PER_LANGUAGE:
                        future.cancel()
                        continue
                    ft, url = future_to_meta[future]
                    try:
                        content = future.result()
                    except Exception as exc:  # noqa: BLE001
                        add_skip(f"download_failed_{str(exc)[:80]}", url=url, title=ft)
                        continue
                    saved, size = _finalize_candidate(
                        content, url, ft, lang_key, downloaded,
                        target_folder, global_hashes, add_skip, clip_classifier,
                    )
                    if saved:
                        saved_bytes += size
                        downloaded += 1
                        progress.update(1)
                    _update_progress()

            buffer.clear()

        for file_title in title_iter:
            if downloaded >= TARGET_PER_LANGUAGE:
                break
            if file_title in processed_titles:
                continue
            processed_titles.add(file_title)
            fname = file_title.replace("File:", "", 1)
            if not is_allowed_extension(fname):
                add_skip("disallowed_extension", title=file_title)
                continue
            buffer.append(file_title)
            if len(buffer) >= IMAGEINFO_BATCH_SIZE:
                flush_buffer()

        flush_buffer()

    def run_internet_source() -> None:
        """Download from Google CSE in parallel batches."""
        nonlocal downloaded, saved_bytes
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            return

        batch_size = PARALLEL_DOWNLOAD_WORKERS * 4
        batch: List[Tuple[str, str]] = []  # (url, context_text)

        def flush_internet_batch() -> None:
            nonlocal downloaded, saved_bytes
            if not batch:
                return

            future_to_meta: Dict = {}
            with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_WORKERS) as pool:
                for url, ctx in batch:
                    future = pool.submit(download_url_direct, url)
                    future_to_meta[future] = (url, ctx)

                for future in as_completed(future_to_meta):
                    if downloaded >= TARGET_PER_LANGUAGE:
                        future.cancel()
                        continue
                    url, ctx = future_to_meta[future]
                    try:
                        content = future.result()
                    except Exception as exc:  # noqa: BLE001
                        add_skip(f"download_failed_{str(exc)[:80]}", url=url)
                        continue
                    saved, size = _finalize_candidate(
                        content, url, url, lang_key, downloaded,
                        target_folder, global_hashes, add_skip, clip_classifier,
                    )
                    if saved:
                        saved_bytes += size
                        downloaded += 1
                        progress.update(1)
                    _update_progress()

            batch.clear()

        for url, ctx in iter_internet_images(lang_key):
            if downloaded >= TARGET_PER_LANGUAGE:
                break
            if url in processed_urls:
                continue
            processed_urls.add(url)

            # Extension check from URL
            url_ext = Path(url.split("?")[0]).suffix.lower()
            if url_ext and url_ext not in ALLOWED_EXTENSIONS:
                add_skip("disallowed_extension", url=url)
                continue

            # Keyword filter on context text (free)
            if contains_banned_keywords(ctx):
                add_skip("banned_keyword", url=url)
                continue
            if not appears_relevant_inscription_or_manuscript(ctx):
                add_skip("not_relevant_to_inscription_or_manuscript", url=url)
                continue

            batch.append((url, ctx))
            if len(batch) >= batch_size:
                flush_internet_batch()

        flush_internet_batch()

    # --- Source 1: Wikimedia categories ---
    run_wikimedia_source(iter_titles_from_categories(client, categories), from_search=False)

    # --- Source 2: Wikimedia search queries ---
    if downloaded < TARGET_PER_LANGUAGE and wiki_search_queries:
        print(f"\nExpanding with Wikimedia search for {lang_display}...")
        run_wikimedia_source(iter_titles_from_search_queries(client, wiki_search_queries), from_search=True)

    # --- Source 3: Internet (Google CSE) ---
    if downloaded < TARGET_PER_LANGUAGE and GOOGLE_API_KEY and GOOGLE_CSE_ID:
        print(f"\nExpanding with Google Custom Search for {lang_display}...")
        run_internet_source()

    progress.close()

    if downloaded < TARGET_PER_LANGUAGE:
        add_skip(f"insufficient_valid_images_downloaded_{downloaded}_of_{TARGET_PER_LANGUAGE}")

    if skip_reason_counts:
        top_reasons = sorted(skip_reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        summary = ", ".join(f"{reason}:{count}" for reason, count in top_reasons)
        print(f"Top skip reasons for {lang_display}: {summary}")

    return {
        "language": lang_display,
        "key": lang_key,
        "downloaded": downloaded,
        "skipped": skipped,
        "saved_bytes": saved_bytes,
        "folder": str(target_folder),
    }


# -------------------------
# Phase 4 - final reporting
# -------------------------
def write_summary_json(
    dataset_root: Path,
    language_stats: List[Dict],
    skipped_log: List[Dict],
) -> Path:
    """Write machine-readable summary report required for project submission."""
    total_downloaded = sum(item["downloaded"] for item in language_stats)
    total_bytes = sum(item["saved_bytes"] for item in language_stats)

    per_language = {
        item["language"]: {
            "downloaded": item["downloaded"],
            "skipped": item["skipped"],
            "folder": item["folder"],
        }
        for item in language_stats
    }

    summary = {
        "project": "Digitisation of Historical South Asian Inscriptions and Manuscripts",
        "team": "CDV01",
        "institution": "RV College of Engineering",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_images_downloaded": total_downloaded,
        "count_per_language": per_language,
        "skipped_urls_with_reason": skipped_log,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
    }

    summary_path = dataset_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def write_readme_txt(
    dataset_root: Path,
    language_stats: List[Dict],
    skipped_log: List[Dict],
) -> Path:
    """Write human-readable report mirroring summary.json content."""
    total_downloaded = sum(item["downloaded"] for item in language_stats)
    total_bytes = sum(item["saved_bytes"] for item in language_stats)
    timestamp = datetime.now(timezone.utc).isoformat()

    lines: List[str] = []
    lines.append("Digitisation of Historical South Asian Inscriptions and Manuscripts")
    lines.append("Team CDV01 - RV College of Engineering")
    lines.append("=" * 72)
    lines.append(f"Generated (UTC): {timestamp}")
    lines.append("")
    lines.append(f"Total images downloaded: {total_downloaded}")
    lines.append(f"Total size (MB): {round(total_bytes / (1024 * 1024), 2)}")
    lines.append("")
    lines.append("Per-language results:")
    for item in language_stats:
        lines.append(
            f"- {item['language']}: downloaded={item['downloaded']}, skipped={item['skipped']}, folder={item['folder']}"
        )

    lines.append("")
    lines.append("Skipped URLs with reason:")
    if skipped_log:
        for rec in skipped_log:
            lines.append(
                f"- language={rec.get('language','')}, title={rec.get('title','')}, "
                f"url={rec.get('url','')}, reason={rec.get('reason','')}"
            )
    else:
        lines.append("- None")

    readme_path = dataset_root / "README.txt"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


# -------------------------
# Main execution entrypoint
# -------------------------
def main() -> None:
    """Run the full 4-phase scraping workflow with strict project requirements."""
    print("Digitisation of Historical South Asian Inscriptions and Manuscripts")
    print("Team CDV01 - RV College of Engineering\n")

    # Print internet source status
    print("--- Configuration ---")
    print(f"Google CSE: {'ENABLED' if GOOGLE_API_KEY and GOOGLE_CSE_ID else 'disabled (set GOOGLE_API_KEY + GOOGLE_CSE_ID)'}")
    print(f"Parallel workers: {PARALLEL_DOWNLOAD_WORKERS}")

    # Try to load CLIP classifier
    clip_classifier: Optional[CLIPClassifier] = None
    print("\nLoading CLIP classifier...", end=" ", flush=True)
    print("DISABLED (temporarily)")
    print()

    dataset_root = make_output_paths()
    client = WikimediaClient()

    # Phase 1: category verification and fallback resolution.
    resolved_plan = verify_language_categories(client)

    # Phase 2 + 3: smart downloading and organized storage.
    print("\nPHASE 2 - Smart Downloading")
    print("PHASE 3 - Organisation")
    print(f"Saving data under: {dataset_root}\n")

    global_hashes: Set[str] = set()
    skipped_log: List[Dict] = []
    language_stats: List[Dict] = []

    for cfg in LANGUAGE_CONFIG:
        key = cfg["key"]
        print(f"\nDownloading for {cfg['display']}...")
        stats = download_language_dataset(
            client=client,
            cfg=cfg,
            resolved_cfg=resolved_plan[key],
            dataset_root=dataset_root,
            global_hashes=global_hashes,
            skipped_log=skipped_log,
            clip_classifier=clip_classifier,
        )
        language_stats.append(stats)
        print(
            f"Completed {cfg['display']}: downloaded={stats['downloaded']}, skipped={stats['skipped']}"
        )

    # Phase 4: generate summary reports.
    print("\nPHASE 4 - Final Report")
    summary_path = write_summary_json(dataset_root, language_stats, skipped_log)
    readme_path = write_readme_txt(dataset_root, language_stats, skipped_log)

    total_downloaded = sum(item["downloaded"] for item in language_stats)
    print("\nAll phases complete.")
    print(f"Total images downloaded: {total_downloaded}")
    if total_downloaded < MIN_TOTAL_TARGET:
        print(
            f"WARNING: Total downloaded ({total_downloaded}) is below minimum target ({MIN_TOTAL_TARGET})."
        )
    elif total_downloaded > MAX_TOTAL_TARGET:
        print(
            f"WARNING: Total downloaded ({total_downloaded}) is above desired maximum ({MAX_TOTAL_TARGET})."
        )
    else:
        print(f"Target range satisfied: {MIN_TOTAL_TARGET} to {MAX_TOTAL_TARGET} images.")
    print(f"Summary file: {summary_path}")
    print(f"README file: {readme_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as err:  # noqa: BLE001
        print(f"\nFatal error: {err}")
        sys.exit(1)
