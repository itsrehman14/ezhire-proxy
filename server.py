from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, Response
import html
import httpx
import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Literal
from html.parser import HTMLParser
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

BASE_URL = "https://ezhire.me/rental"
SUPPORTED_COUNTRY_CODES = ["971", "966", "973"]
SUPPORT_PAGES_DIR = os.getenv("SUPPORT_PAGES_DIR", "webpages")
SUPPORT_FAQ_FILE = os.getenv("SUPPORT_FAQ_FILE", "FAQ.html")
SUPPORT_TERMS_FILE = os.getenv("SUPPORT_TERMS_FILE", "TnC.html")
FAQ_SOURCE_URL = "https://www.ezhire.ae/uae-car-rental-faqs"
TERMS_SOURCE_URL = "https://www.ezhire.ae/terms-and-conditions"
SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "+97144594600")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
SUPPORT_LLM_MODEL = os.getenv("SUPPORT_LLM_MODEL", "anthropic/claude-3.5-haiku")
SUPPORT_LLM_TIMEOUT = float(os.getenv("SUPPORT_LLM_TIMEOUT", "10.0"))

_SUPPORT_CHUNKS: list[dict[str, Any]] = []
_SUPPORT_TERM_IDF: dict[str, float] = {}
_SUPPORT_AVG_DOC_LEN = 0.0
_SUPPORT_INDEX_ERROR: str | None = None
_http_client: httpx.AsyncClient | None = None

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how",
    "i", "if", "in", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our",
    "the", "their", "them", "there", "these", "they", "this", "to", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "with", "you", "your", "can", "will",
    "do", "does", "did", "all", "any", "about", "into", "than", "then", "that", "been",
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=30.0)
    try:
        build_support_index()
    except Exception as exc:
        logger.exception("Support index initialization failed: %s", exc)
    yield
    await _http_client.aclose()
    _http_client = None


app = FastAPI(title="eZhire API", version="1.0.0", lifespan=lifespan)



def split_phone_e164(phone_e164: str) -> tuple[str, str]:
    if not phone_e164.startswith("+"):
        raise HTTPException(status_code=422, detail="phone_e164 must start with '+'")
    digits = phone_e164[1:]
    for code in sorted(SUPPORTED_COUNTRY_CODES, key=len, reverse=True):
        if digits.startswith(code):
            return f"+{code}", digits[len(code):]
    raise HTTPException(status_code=422, detail="Unsupported country code in phone_e164")


def normalize_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to DD/MM/YYYY for the upstream API. Pass through otherwise."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d/%m/%Y")
    except ValueError:
        return date_str



async def proxy_request(method: str, path: str, authorization: str, json_body=None):
    assert _http_client is not None, "HTTP client not initialized"
    url = f"{BASE_URL}{path}"
    headers = {"Authorization": authorization}
    try:
        response = await _http_client.request(method, url, headers=headers, json=json_body)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}")
    content_type = response.headers.get("content-type", "text/plain")
    try:
        return JSONResponse(status_code=response.status_code, content=response.json())
    except ValueError:
        return Response(status_code=response.status_code, content=response.text, media_type=content_type)


async def fetch_upstream_json(path: str, authorization: str):
    assert _http_client is not None, "HTTP client not initialized"
    url = f"{BASE_URL}{path}"
    headers = {"Authorization": authorization}
    try:
        response = await _http_client.request("GET", url, headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}")
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"status": False, "message": response.text or "Upstream error"}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if tag in {"h1", "h2", "h3", "h4", "p", "li", "br"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag in {"h1", "h2", "h3", "h4", "p", "li", "br"}:
            self._chunks.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth > 0:
            return
        cleaned = " ".join(data.split())
        if cleaned:
            self._chunks.append(cleaned)

    def get_lines(self) -> list[str]:
        text = "".join(self._chunks)
        lines = []
        for line in text.splitlines():
            cleaned = " ".join(line.split()).strip()
            if cleaned:
                lines.append(cleaned)
        return lines


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    chunks = []
    if not text:
        return chunks
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def isolate_main_html(raw_html: str) -> str:
    lowered = raw_html.lower()
    main_start = lowered.find("<main")
    if main_start < 0:
        return raw_html
    main_end = lowered.find("</main>", main_start)
    if main_end < 0:
        return raw_html[main_start:]
    return raw_html[main_start : main_end + len("</main>")]


def strip_html(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text, flags=re.S)
    unescaped = html.unescape(without_tags)
    return " ".join(unescaped.split()).strip()


def is_relevant_line(line: str) -> bool:
    lowered = line.lower()
    if len(lowered) < 3:
        return False
    if lowered.startswith("table of contents"):
        return False
    if lowered in {"book now", "download app", "quick links"}:
        return False
    if "all rights reserved" in lowered:
        return False
    if "cookie" in lowered and "policy" in lowered:
        return False
    social_markers = ("facebook", "instagram", "linkedin", "youtube", "twitter", "tiktok")
    if any(lowered.startswith(marker) for marker in social_markers):
        return False
    return True


def extract_faq_entries(main_html: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r'<span[^>]*class="[^"]*pwr-accordion__title[^"]*"[^>]*>(.*?)<i[^>]*>.*?</span>\s*'
        r'<span[^>]*class="[^"]*pwr-rich-text[^"]*pwr-accordion__desc[^"]*"[^>]*>(.*?)</span>',
        flags=re.I | re.S,
    )
    entries = []
    seen_questions = set()
    for question_raw, answer_raw in pattern.findall(main_html):
        question = strip_html(question_raw).rstrip("?").strip()
        answer = strip_html(answer_raw)
        if not question or not answer:
            continue
        question_key = question.lower()
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        text = f"Q: {question}? A: {answer}"
        entries.append(
            {
                "url": FAQ_SOURCE_URL,
                "source": "FAQ",
                "section": question,
                "text": text,
            }
        )
    if entries:
        return entries

    # Fallback parser if accordion markup changes in future exports.
    parser = TextExtractor()
    parser.feed(main_html)
    lines = [line.strip() for line in parser.get_lines() if line.strip()]
    for i, line in enumerate(lines):
        if not line.endswith("?"):
            continue
        if len(line) < 8 or len(line) > 180:
            continue
        question = line.rstrip("?").strip()
        question_key = question.lower()
        if question_key in seen_questions:
            continue
        answers = []
        for j in range(i + 1, len(lines)):
            candidate = lines[j].strip()
            if candidate.endswith("?"):
                break
            if is_relevant_line(candidate):
                answers.append(candidate)
            if len(" ".join(answers)) > 700:
                break
        if not answers:
            continue
        seen_questions.add(question_key)
        entries.append(
            {
                "url": FAQ_SOURCE_URL,
                "source": "FAQ",
                "section": question,
                "text": f"Q: {question}? A: {' '.join(answers)}",
            }
        )
    return entries


def extract_terms_lines(main_html: str) -> list[str]:
    parser = TextExtractor()
    parser.feed(main_html)
    lines = parser.get_lines()
    cleaned_lines: list[str] = []
    for line in lines:
        cleaned = html.unescape(" ".join(line.split())).strip()
        if not is_relevant_line(cleaned):
            continue
        if cleaned_lines and cleaned_lines[-1].lower() == cleaned.lower():
            continue
        cleaned_lines.append(cleaned)
    return cleaned_lines



def load_support_documents() -> list[dict[str, str]]:
    json_path = os.path.join(SUPPORT_PAGES_DIR, "support_docs.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        if docs:
            return docs

    logger.warning("support_docs.json not found; falling back to HTML parsing")

    faq_path = os.path.join(SUPPORT_PAGES_DIR, SUPPORT_FAQ_FILE)
    terms_path = os.path.join(SUPPORT_PAGES_DIR, SUPPORT_TERMS_FILE)
    if not os.path.exists(faq_path):
        raise RuntimeError(f"Missing FAQ support page: {faq_path}")
    if not os.path.exists(terms_path):
        raise RuntimeError(f"Missing Terms support page: {terms_path}")

    with open(faq_path, "r", encoding="utf-8") as handle:
        faq_html = handle.read()
    with open(terms_path, "r", encoding="utf-8") as handle:
        terms_html = handle.read()

    faq_main = isolate_main_html(faq_html)
    terms_main = isolate_main_html(terms_html)

    faq_entries = extract_faq_entries(faq_main)
    if not faq_entries:
        raise RuntimeError("No FAQ entries were extracted from local FAQ.html")

    terms_lines = extract_terms_lines(terms_main)
    if not terms_lines:
        raise RuntimeError("No terms content was extracted from local TnC.html")

    documents = list(faq_entries)
    terms_text = "\n".join(terms_lines)
    for chunk in chunk_text(terms_text, chunk_size=1200, overlap=180):
        documents.append(
            {
                "url": TERMS_SOURCE_URL,
                "source": "TnC",
                "section": "Terms and Conditions",
                "text": chunk,
            }
        )
    return documents


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())
    return [token for token in tokens if len(token) > 1 and token not in _STOP_WORDS]


def build_support_index() -> None:
    global _SUPPORT_CHUNKS, _SUPPORT_TERM_IDF, _SUPPORT_AVG_DOC_LEN, _SUPPORT_INDEX_ERROR
    documents = load_support_documents()
    chunks: list[dict[str, Any]] = []
    doc_frequency = Counter()
    total_doc_len = 0

    for doc in documents:
        tokens = tokenize(doc["text"])
        if not tokens:
            continue
        term_freq = Counter(tokens)
        doc_len = sum(term_freq.values())
        total_doc_len += doc_len
        for term in term_freq:
            doc_frequency[term] += 1
        chunks.append(
            {
                "url": doc["url"],
                "source": doc["source"],
                "section": doc["section"],
                "text": doc["text"],
                "term_freq": dict(term_freq),
                "doc_len": doc_len,
            }
        )

    if not chunks:
        raise RuntimeError("No support content available for indexing")

    num_docs = len(chunks)
    idf = {}
    for term, freq in doc_frequency.items():
        idf[term] = math.log(1.0 + ((num_docs - freq + 0.5) / (freq + 0.5)))

    _SUPPORT_CHUNKS = chunks
    _SUPPORT_TERM_IDF = idf
    _SUPPORT_AVG_DOC_LEN = total_doc_len / num_docs
    _SUPPORT_INDEX_ERROR = None


def retrieve_support_chunks(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    if not _SUPPORT_CHUNKS:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    avg_doc_len = _SUPPORT_AVG_DOC_LEN if _SUPPORT_AVG_DOC_LEN > 0 else 1.0
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, int]] = []
    query_counter = Counter(query_tokens)
    query_lower = query.lower()

    for index, chunk in enumerate(_SUPPORT_CHUNKS):
        term_freq: dict[str, int] = chunk["term_freq"]
        doc_len = chunk["doc_len"]
        score = 0.0
        for term, query_freq in query_counter.items():
            freq = term_freq.get(term, 0)
            if freq <= 0:
                continue
            idf = _SUPPORT_TERM_IDF.get(term, 0.0)
            denominator = freq + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            score += idf * ((freq * (k1 + 1.0)) / denominator) * (1.0 + (0.1 * (query_freq - 1)))

        if score <= 0.0:
            continue
        if query_lower in chunk["text"].lower():
            score += 1.0

        overlap = len(set(query_tokens).intersection(term_freq.keys()))
        score += overlap * 0.05
        scored.append((score, index))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [_SUPPORT_CHUNKS[index] for _, index in scored[:top_k]]


def select_relevant_sentences(text: str, query: str, limit: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return text
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return " ".join(sentences[:limit])

    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_tokens = set(tokenize(sentence))
        overlap = len(query_tokens.intersection(sentence_tokens))
        scored.append((overlap, -index, sentence))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in scored[:limit] if item[0] > 0]
    if not selected:
        selected = sentences[:limit]
    return " ".join(selected)


def build_support_answer(query: str, chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        text = chunk["text"]
        if text.startswith("Q: ") and " A: " in text:
            answer_text = text.split(" A: ", 1)[1].strip()
            snippet = select_relevant_sentences(answer_text, query, limit=2).strip()
        else:
            snippet = select_relevant_sentences(text, query, limit=2).strip()
        if snippet:
            parts.append(snippet)
        if len(" ".join(parts)) > 400:
            break

    if not parts:
        return f"We couldn't find a direct answer. For help, call {SUPPORT_PHONE}."

    answer = " ".join(parts).strip()
    return f"{answer} For help, call {SUPPORT_PHONE}."


def build_llm_prompt(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    system_msg = (
        "You are a customer support assistant for eZhire, a car rental company in the UAE. "
        "Answer in 2-3 concise sentences using ONLY the provided sources. "
        "Do not repeat information. Do not invent facts. "
        f"End with the support phone number: {SUPPORT_PHONE}."
    )
    source_lines = []
    for i, chunk in enumerate(chunks, 1):
        label = chunk.get("source", "Unknown")
        section = chunk.get("section", "")
        source_lines.append(f"[Source {i} - {label}: {section}]\n{chunk['text']}")
    user_msg = f"Customer question: {query}\n\nSources:\n" + "\n\n".join(source_lines)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


async def generate_llm_answer(query: str, chunks: list[dict[str, Any]]) -> str | None:
    if not OPENROUTER_API_KEY or _http_client is None:
        return None
    messages = build_llm_prompt(query, chunks)
    try:
        response = await _http_client.post(
            OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": SUPPORT_LLM_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 256,
            },
            timeout=SUPPORT_LLM_TIMEOUT,
        )
        if response.status_code != 200:
            logger.warning("OpenRouter returned status %d: %s", response.status_code, response.text[:200])
            return None
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        if not content:
            return None
        return content
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        logger.warning("LLM call failed: %s", exc)
        return None
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("LLM response parsing failed: %s", exc)
        return None


# --- Request models ---

class PreBookingRequest(BaseModel):
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    phone_e164: str = Field(pattern=r"^\+\d{10,15}$")
    renting_in: int = Field(gt=0)
    car_id: int = Field(gt=0)
    deliver_date: str
    deliver_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    self_pickup_id: int = Field(gt=0)
    return_date: str
    self_return_id: int = Field(gt=0)

    @field_validator("deliver_date", "return_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                datetime.strptime(v, fmt)
                return v
            except ValueError:
                continue
        raise ValueError("date must be in YYYY-MM-DD or DD/MM/YYYY format")

    @model_validator(mode="after")
    def validate_return_after_deliver(self):
        def _parse(s: str) -> datetime:
            for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            raise ValueError(f"unparseable date: {s}")

        deliver = _parse(self.deliver_date)
        ret = _parse(self.return_date)
        if ret < deliver + timedelta(days=1):
            raise ValueError("return_date must be at least one day after deliver_date")
        return self


class CancelBookingRequest(BaseModel):
    booking_id: int = Field(gt=0)
    reason: str = Field(min_length=1)


class QuoteRequest(BaseModel):
    car_id: int = Field(gt=0)
    rental_days: int = Field(gt=0)


class SupportAnswerRequest(BaseModel):
    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty")
        return v


# --- Response models ---

class Vehicle(BaseModel):
    vehicle_id: int
    vehicle_category: str
    make: str
    model: str
    vehicle_image: str
    daily_rate: float
    weekly_rate: float
    monthly_rate: float
    tax_rate: float
    currency: str
    total_estimated_amount_weekly: float
    total_estimated_amount_daily: float
    total_estimated_amount_monthly: float
    daily_allowed_km: float
    weekly_allowed_km: float
    monthly_allowed_km: float
    extra_km_price: float
    transmission: str
    number_of_seats: int
    air_conditioning: int
    number_of_doors: int
    luggage_capacity: int
    car_name: str
    vehicle_class: str | None = None


class RentalCarsResponse(BaseModel):
    status: bool
    message: str
    data: list[Vehicle]


class PickupLocation(BaseModel):
    id: int
    name: str
    address: str
    latitude: str
    longitude: str
    phone_no: str | None = None
    friday_open_time: str | None = None
    sat_open_time: str | None = None
    sun_open_time: str | None = None
    mon_open_time: str | None = None
    tues_open_time: str | None = None
    wed_open_time: str | None = None
    thur_open_time: str | None = None
    friday_close_time: str | None = None
    sat_close_time: str | None = None
    sun_close_time: str | None = None
    mon_close_time: str | None = None
    tues_close_time: str | None = None
    wed_close_time: str | None = None
    thur_close_time: str | None = None
    city_id: int | None = None
    city: str | None = None
    country: str | None = None
    email: str | None = None


class PickupLocationsResponse(BaseModel):
    status: bool
    message: str
    data: list[PickupLocation]


class PreBookingData(BaseModel):
    status: bool
    booking_id: int | None = None
    amount: float | None = None
    hashable_url: str | None = None
    message: str | None = None


class CancelBookingResponse(BaseModel):
    status: bool
    message: str


class QuoteData(BaseModel):
    car_id: int
    rental_days: int
    rate_type: Literal["daily", "weekly", "monthly"]
    rate: float
    tax_rate: float
    base_amount: float
    tax_amount: float
    total_amount: float
    currency: str | None


class QuoteResponse(BaseModel):
    status: bool
    message: str
    data: QuoteData


class SupportSource(BaseModel):
    url: str
    section: str


class SupportData(BaseModel):
    answer: str
    sources: list[SupportSource]


class SupportResponse(BaseModel):
    status: bool
    message: str
    data: SupportData


class WelcomeData(BaseModel):
    greeting: str
    vehicles: list[Vehicle]
    pickup_locations: list[PickupLocation]
    support_phone: str


class WelcomeResponse(BaseModel):
    status: bool
    message: str
    data: WelcomeData


# --- Endpoints ---

@app.get("/welcome/", response_model=WelcomeResponse)
async def welcome(authorization: str = Header(...)):
    """Aggregated dashboard for the welcome agent: cars + locations in one call."""
    cars_status, cars_data = await fetch_upstream_json("/get_rental_cars/", authorization)
    locs_status, locs_data = await fetch_upstream_json("/get_pickup_locations/", authorization)

    if cars_status != 200 or not isinstance(cars_data, dict):
        raise HTTPException(status_code=502, detail="Failed to fetch rental cars from upstream")
    if locs_status != 200 or not isinstance(locs_data, dict):
        raise HTTPException(status_code=502, detail="Failed to fetch pickup locations from upstream")

    return {
        "status": True,
        "message": "Welcome to eZhire",
        "data": {
            "greeting": (
                "Welcome to eZhire! Here are the latest rental cars available "
                "across the UAE, along with our pickup and return locations."
            ),
            "vehicles": cars_data.get("data", []),
            "pickup_locations": locs_data.get("data", []),
            "support_phone": SUPPORT_PHONE,
        },
    }


@app.get("/get_rental_cars/", response_model=RentalCarsResponse)
async def get_rental_cars(authorization: str = Header(...)):
    return await proxy_request("GET", "/get_rental_cars/", authorization)


@app.get("/get_pickup_locations/", response_model=PickupLocationsResponse)
async def get_pickup_locations(
    authorization: str = Header(...),
    city: str | None = Query(default=None),
    country: str | None = Query(default=None),
):
    if city and not country:
        raise HTTPException(status_code=422, detail="country is required when filtering by city")
    response = await proxy_request("GET", "/get_pickup_locations/", authorization)
    if isinstance(response, JSONResponse) and isinstance(response.body, (bytes, bytearray)):
        try:
            data = json.loads(response.body)
        except Exception:
            return response
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            filtered = []
            available = {}
            for item in data["data"]:
                if not isinstance(item, dict):
                    continue
                item_city = item.get("city", "")
                item_country = item.get("country", "")
                if item_city and item_country:
                    available.setdefault(item_country.lower(), set()).add(item_city.lower())
                if country and item_country.lower() != country.lower():
                    continue
                if city and item_city.lower() != city.lower():
                    continue
                filtered.append(item)
            if country and city:
                if city.lower() not in available.get(country.lower(), set()):
                    raise HTTPException(
                        status_code=422,
                        detail="city does not belong to the provided country",
                    )
            data["data"] = filtered
            return JSONResponse(status_code=response.status_code, content=data)
    return response


@app.post("/generate_pre_booking/", response_model=PreBookingData)
async def generate_pre_booking(body: PreBookingRequest, authorization: str = Header(...)):
    country_code, _ = split_phone_e164(body.phone_e164)
    payload = body.model_dump()
    payload["country_code"] = country_code
    payload["mobile"] = body.phone_e164
    payload.pop("phone_e164", None)
    payload["deliver_date"] = normalize_date(payload["deliver_date"])
    payload["return_date"] = normalize_date(payload["return_date"])
    return await proxy_request("POST", "/generate_pre_booking/", authorization, json_body=payload)


@app.post("/cancel_pre_booking/", response_model=CancelBookingResponse)
async def cancel_pre_booking(body: CancelBookingRequest, authorization: str = Header(...)):
    return await proxy_request("POST", "/cancel_pre_booking/", authorization, json_body=body.model_dump())


@app.post("/quote/", response_model=QuoteResponse)
async def quote_rental(body: QuoteRequest, authorization: str = Header(...)):
    status, data = await fetch_upstream_json("/get_rental_cars/", authorization)
    if status != 200 or not isinstance(data, dict):
        return JSONResponse(status_code=status, content=data or {"status": False, "message": "Upstream error"})
    cars = data.get("data", [])
    selected = None
    for item in cars:
        if isinstance(item, dict) and item.get("vehicle_id") == body.car_id:
            selected = item
            break
    if not selected:
        raise HTTPException(status_code=404, detail="car_id not found")
    tax_rate = float(selected.get("tax_rate", 0))
    if body.rental_days >= 30:
        rate_type = "monthly"
        rate = float(selected.get("monthly_rate", 0))
    elif body.rental_days >= 7:
        rate_type = "weekly"
        rate = float(selected.get("weekly_rate", 0))
    else:
        rate_type = "daily"
        rate = float(selected.get("daily_rate", 0))
    base_amount = rate * body.rental_days
    tax_amount = base_amount * (tax_rate / 100.0)
    total_amount = base_amount + tax_amount
    return {
        "status": True,
        "message": "Success",
        "data": {
            "car_id": body.car_id,
            "rental_days": body.rental_days,
            "rate_type": rate_type,
            "rate": rate,
            "tax_rate": tax_rate,
            "base_amount": round(base_amount, 2),
            "tax_amount": round(tax_amount, 2),
            "total_amount": round(total_amount, 2),
            "currency": selected.get("currency"),
        },
    }


@app.post("/support_answer/", response_model=SupportResponse)
async def support_answer(body: SupportAnswerRequest):
    global _SUPPORT_INDEX_ERROR
    query = body.query
    if not _SUPPORT_CHUNKS:
        try:
            build_support_index()
        except Exception as exc:
            _SUPPORT_INDEX_ERROR = str(exc)
            raise HTTPException(status_code=503, detail=f"Support index is not ready: {_SUPPORT_INDEX_ERROR}")
    chunks = retrieve_support_chunks(query, top_k=5)
    if not chunks:
        return {
            "status": True,
            "message": "No relevant answer found.",
            "data": {
                "answer": f"We couldn't find a direct answer. For help, call {SUPPORT_PHONE}.",
                "sources": [],
            },
        }
    sources = []
    seen_sources = set()
    for chunk in chunks:
        url = chunk["url"]
        section = chunk["section"]
        source_key = (url, section)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        sources.append({"url": url, "section": section})
    answer = await generate_llm_answer(query, chunks)
    if answer is None:
        answer = build_support_answer(query, chunks)
    return {
        "status": True,
        "message": "Success",
        "data": {
            "answer": answer,
            "sources": sources,
        },
    }
