import json

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import server


@pytest.fixture(scope="module", autouse=True)
def _build_local_support_index():
    # Build once from local webpages/ so support tests never depend on network APIs.
    server.build_support_index()


def _json_body(response: JSONResponse) -> dict:
    return json.loads(response.body)


# --- get_pickup_locations ---


@pytest.mark.asyncio
async def test_get_pickup_locations_requires_country_if_city_provided():
    with pytest.raises(HTTPException) as exc_info:
        await server.get_pickup_locations("Token test", city="Dubai", country=None)
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "country is required when filtering by city"


@pytest.mark.asyncio
async def test_get_pickup_locations_filters_by_country_and_city(monkeypatch: pytest.MonkeyPatch):
    payload = {
        "status": True,
        "data": [
            {"city": "Dubai", "country": "UAE"},
            {"city": "Abu Dhabi", "country": "UAE"},
            {"city": "Riyadh", "country": "KSA"},
        ],
    }

    async def fake_proxy(method, path, authorization, json_body=None):
        return JSONResponse(status_code=200, content=payload)

    monkeypatch.setattr(server, "proxy_request", fake_proxy)
    response = await server.get_pickup_locations("Token test", city="Dubai", country="UAE")
    body = _json_body(response)

    assert response.status_code == 200
    assert len(body["data"]) == 1
    assert body["data"][0]["city"] == "Dubai"
    assert body["data"][0]["country"] == "UAE"


@pytest.mark.asyncio
async def test_get_pickup_locations_rejects_city_country_mismatch(monkeypatch: pytest.MonkeyPatch):
    payload = {
        "status": True,
        "data": [
            {"city": "Dubai", "country": "UAE"},
            {"city": "Riyadh", "country": "KSA"},
        ],
    }

    async def fake_proxy(method, path, authorization, json_body=None):
        return JSONResponse(status_code=200, content=payload)

    monkeypatch.setattr(server, "proxy_request", fake_proxy)
    with pytest.raises(HTTPException) as exc_info:
        await server.get_pickup_locations("Token test", city="Riyadh", country="UAE")
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "city does not belong to the provided country"


# --- generate_pre_booking ---


@pytest.mark.asyncio
async def test_generate_pre_booking_splits_e164_and_removes_original_field(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    async def fake_proxy(method, path, authorization, json_body=None):
        captured["json_body"] = json_body
        return JSONResponse(status_code=200, content={"status": True})

    monkeypatch.setattr(server, "proxy_request", fake_proxy)
    body = server.PreBookingRequest(
        first_name="John",
        last_name="Doe",
        email="john@example.com",
        phone_e164="+971501234567",
        renting_in=1,
        car_id=1162,
        deliver_date="2026-02-12",
        deliver_time="10:00",
        self_pickup_id=10,
        return_date="2026-02-14",
        self_return_id=11,
    )
    response = await server.generate_pre_booking(body, "Token test")

    assert response.status_code == 200
    outbound = captured["json_body"]
    assert outbound["country_code"] == "+971"
    assert outbound["mobile"] == "+971501234567"
    assert "phone_e164" not in outbound
    assert outbound["deliver_date"] == "12/02/2026"
    assert outbound["return_date"] == "14/02/2026"


def test_generate_pre_booking_rejects_invalid_phone():
    with pytest.raises(ValidationError, match="phone_e164"):
        server.PreBookingRequest(
            first_name="John",
            last_name="Doe",
            email="john@example.com",
            phone_e164="0501234567",
            renting_in=1,
            car_id=1162,
            deliver_date="2026-02-12",
            deliver_time="10:00",
            self_pickup_id=10,
            return_date="2026-02-14",
            self_return_id=11,
        )


def test_generate_pre_booking_rejects_return_before_deliver():
    with pytest.raises(Exception, match="return_date must be at least one day after deliver_date"):
        server.PreBookingRequest(
            first_name="John",
            last_name="Doe",
            email="john@example.com",
            phone_e164="+971501234567",
            renting_in=1,
            car_id=1162,
            deliver_date="2026-02-14",
            deliver_time="10:00",
            self_pickup_id=10,
            return_date="2026-02-14",
            self_return_id=11,
        )


def test_generate_pre_booking_rejects_return_same_day_ddmmyyyy():
    with pytest.raises(Exception, match="return_date must be at least one day after deliver_date"):
        server.PreBookingRequest(
            first_name="John",
            last_name="Doe",
            email="john@example.com",
            phone_e164="+971501234567",
            renting_in=1,
            car_id=1162,
            deliver_date="14/02/2026",
            deliver_time="10:00",
            self_pickup_id=10,
            return_date="14/02/2026",
            self_return_id=11,
        )


@pytest.mark.asyncio
async def test_proxy_returns_upstream_error(monkeypatch: pytest.MonkeyPatch):
    async def fake_proxy(method, path, authorization, json_body=None):
        return JSONResponse(status_code=400, content={"status": False, "message": "Invalid date format. Please use DD/MM/YYYY."})

    monkeypatch.setattr(server, "proxy_request", fake_proxy)
    body = server.PreBookingRequest(
        first_name="John",
        last_name="Doe",
        email="john@example.com",
        phone_e164="+971501234567",
        renting_in=1,
        car_id=1162,
        deliver_date="2026-02-12",
        deliver_time="10:00",
        self_pickup_id=10,
        return_date="2026-02-14",
        self_return_id=11,
    )
    response = await server.generate_pre_booking(body, "Token test")
    parsed = _json_body(response)

    assert response.status_code == 400
    assert parsed["status"] is False
    assert "Invalid date format" in parsed["message"]


# --- cancel_pre_booking ---


@pytest.mark.asyncio
async def test_cancel_pre_booking_passthrough(monkeypatch: pytest.MonkeyPatch):
    async def fake_proxy(method, path, authorization, json_body=None):
        return JSONResponse(status_code=200, content={"status": True, "message": "cancelled"})

    monkeypatch.setattr(server, "proxy_request", fake_proxy)
    body = server.CancelBookingRequest(booking_id=123, reason="Changed plans")
    response = await server.cancel_pre_booking(body, "Token test")
    parsed = _json_body(response)

    assert response.status_code == 200
    assert parsed["message"] == "cancelled"


# --- quote ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rental_days, expected_rate_type, expected_total",
    [
        (2, "daily", 210.0),
        (10, "weekly", 945.0),
        (30, "monthly", 2205.0),
    ],
)
async def test_quote_rate_selection_and_total(
    monkeypatch: pytest.MonkeyPatch, rental_days: int, expected_rate_type: str, expected_total: float
):
    cars = {
        "status": True,
        "data": [
            {
                "vehicle_id": 1162,
                "tax_rate": 5,
                "daily_rate": 100,
                "weekly_rate": 90,
                "monthly_rate": 70,
                "currency": "AED",
            }
        ],
    }

    async def fake_fetch(path, authorization):
        return 200, cars

    monkeypatch.setattr(server, "fetch_upstream_json", fake_fetch)
    response = await server.quote_rental(server.QuoteRequest(car_id=1162, rental_days=rental_days), "Token test")

    assert response["status"] is True
    assert response["data"]["rate_type"] == expected_rate_type
    assert response["data"]["total_amount"] == expected_total


def test_quote_rejects_non_positive_rental_days():
    with pytest.raises(ValidationError, match="rental_days"):
        server.QuoteRequest(car_id=1162, rental_days=0)


@pytest.mark.asyncio
async def test_quote_rejects_missing_car(monkeypatch: pytest.MonkeyPatch):
    async def fake_fetch(path, authorization):
        return 200, {"status": True, "data": []}

    monkeypatch.setattr(server, "fetch_upstream_json", fake_fetch)
    with pytest.raises(HTTPException) as exc_info:
        await server.quote_rental(server.QuoteRequest(car_id=999, rental_days=2), "Token test")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "car_id not found"


# --- support_answer ---


def test_support_answer_requires_non_empty_query():
    with pytest.raises(ValidationError, match="query"):
        server.SupportAnswerRequest(query="   ")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expected_keywords",
    [
        ("Do you require a deposit to rent a car?", {"deposit"}),
        ("What documents do UAE residents need to rent?", {"license", "emirates", "passport"}),
        ("How do I extend my rental?", {"extend"}),
        ("Will I pay a fee if I cancel?", {"cancel"}),
        ("Is delivery available at night?", {"24/7", "delivery", "night"}),
        ("What is full insurance?", {"insurance"}),
        ("What payment modes are supported?", {"payment", "card"}),
    ],
)
async def test_support_answer_reasonable_questions(
    monkeypatch: pytest.MonkeyPatch, query: str, expected_keywords: set[str]
):
    async def fake_llm(q, c):
        return None

    monkeypatch.setattr(server, "generate_llm_answer", fake_llm)
    response = await server.support_answer(server.SupportAnswerRequest(query=query))
    answer = response["data"]["answer"].lower()
    sources = response["data"]["sources"]

    assert response["status"] is True
    assert len(sources) >= 1
    assert server.SUPPORT_PHONE in response["data"]["answer"]
    assert any(keyword in answer for keyword in expected_keywords)


@pytest.mark.asyncio
async def test_support_answer_for_irrelevant_query_returns_safe_fallback(monkeypatch: pytest.MonkeyPatch):
    async def fake_llm(q, c):
        return None

    monkeypatch.setattr(server, "generate_llm_answer", fake_llm)
    response = await server.support_answer(server.SupportAnswerRequest(query="zxqv plmokn qwertyui asdfghj"))
    assert response["status"] is True
    assert response["message"] == "No relevant answer found."
    assert response["data"]["sources"] == []
    assert "For help, call" in response["data"]["answer"]


@pytest.mark.asyncio
async def test_support_answer_falls_back_when_llm_fails(monkeypatch: pytest.MonkeyPatch):
    async def fake_llm(q, c):
        return None

    monkeypatch.setattr(server, "generate_llm_answer", fake_llm)
    response = await server.support_answer(server.SupportAnswerRequest(query="Do you require a deposit?"))
    assert response["status"] is True
    assert server.SUPPORT_PHONE in response["data"]["answer"]
    assert "deposit" in response["data"]["answer"].lower()


@pytest.mark.asyncio
async def test_support_answer_uses_llm_when_available(monkeypatch: pytest.MonkeyPatch):
    expected = "Yes, a deposit is required. For help, call +97144594600."

    async def fake_llm(q, c):
        return expected

    monkeypatch.setattr(server, "generate_llm_answer", fake_llm)
    response = await server.support_answer(server.SupportAnswerRequest(query="Do you require a deposit?"))
    assert response["status"] is True
    assert response["data"]["answer"] == expected
