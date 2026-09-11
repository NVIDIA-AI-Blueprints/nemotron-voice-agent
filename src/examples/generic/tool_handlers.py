# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool handlers for the generic_assistant pipeline.

Each handler is registered with ``NvidiaLLMService.register_function`` and
receives a :class:`pipecat.services.llm_service.FunctionCallParams`.  Handlers
deliver their result via ``params.result_callback``.

Live data sources (when keys are configured):
  - Currency conversion:  https://api.frankfurter.app  (no key required)
  - Stock prices:         Finnhub (``FINNHUB_API_KEY``, optional) → Yahoo Finance fallback
  - Weather:              WeatherAPI (``WEATHERAPI_KEY``, optional) → static mock fallback

All handlers are tolerant to network failures and fall back to static mock
data so the pipeline keeps responding (and the default demo works) even
when external APIs are down or the optional keys are unset.
"""

import os
import random
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

# ---------------------------------------------------------------------------
# Currency conversion
# ---------------------------------------------------------------------------

# Static fallback rates (USD-based) — mirrors the client-side table.
_STATIC_RATES: dict[str, float] = {
    # Major
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 149.5,
    "CHF": 0.89,
    # Americas
    "CAD": 1.36,
    "AUD": 1.53,
    "MXN": 17.15,
    "BRL": 4.97,
    "ARS": 878.0,
    "CLP": 948.0,
    "COP": 3900.0,
    "PEN": 3.72,
    "UYU": 38.5,
    # Asia-Pacific
    "CNY": 7.24,
    "INR": 83.12,
    "KRW": 1325.0,
    "SGD": 1.34,
    "HKD": 7.82,
    "TWD": 31.8,
    "THB": 35.1,
    "MYR": 4.72,
    "IDR": 15700.0,
    "PHP": 56.5,
    "VND": 24500.0,
    "PKR": 278.0,
    "BDT": 110.0,
    "LKR": 305.0,
    "NPR": 133.0,
    # Europe
    "SEK": 10.42,
    "NOK": 10.55,
    "DKK": 6.88,
    "PLN": 4.02,
    "CZK": 22.8,
    "HUF": 355.0,
    "RON": 4.57,
    "BGN": 1.80,
    "HRK": 6.93,
    "RSD": 107.5,
    "TRY": 32.1,
    "RUB": 91.5,
    "UAH": 37.2,
    # Middle East & Africa
    "AED": 3.67,
    "SAR": 3.75,
    "QAR": 3.64,
    "KWD": 0.307,
    "BHD": 0.376,
    "OMR": 0.385,
    "ILS": 3.71,
    "EGP": 30.9,
    "ZAR": 18.6,
    "NGN": 1480.0,
    "KES": 129.0,
    "GHS": 12.5,
    "MAD": 9.98,
    "TND": 3.12,
    "ETB": 56.5,
    # Other
    "NZD": 1.63,
    "XAU": 0.000508,
    "XAG": 0.0426,
}

_HTTP_TIMEOUT = httpx.Timeout(connect=3.0, read=4.0, write=3.0, pool=2.0)


async def handle_convert_currency(params: FunctionCallParams) -> None:
    """Convert an amount between two currencies using live ECB rates with a static fallback."""
    args = params.arguments or {}
    try:
        amount = float(args.get("amount", 0) or 0)
    except (TypeError, ValueError):
        amount = 0.0
    from_currency = str(args.get("from_currency", "USD") or "USD").upper()
    to_currency = str(args.get("to_currency", "USD") or "USD").upper()

    # Try live rates from frankfurter.app (ECB data, no API key required).
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.get(
                "https://api.frankfurter.app/latest",
                params={"from": from_currency, "to": to_currency},
            )
        if response.status_code == 200:
            data = response.json()
            rate = data.get("rates", {}).get(to_currency)
            if isinstance(rate, (int, float)):
                converted = round(amount * rate, 2)
                await params.result_callback(
                    {
                        "converted_amount": converted,
                        "from_currency": from_currency,
                        "to_currency": to_currency,
                        "exchange_rate": rate,
                        "date": data.get("date"),
                        "source": "live",
                    }
                )
                return
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug(f"convert_currency live lookup failed: {exc}")

    # Static fallback.
    from_rate = _STATIC_RATES.get(from_currency, 1.0)
    to_rate = _STATIC_RATES.get(to_currency, 1.0)
    converted = round((amount / from_rate) * to_rate, 2)
    await params.result_callback(
        {
            "converted_amount": converted,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "exchange_rate": round(to_rate / from_rate, 6),
            "source": "static_fallback",
        }
    )


# ---------------------------------------------------------------------------
# BMI
# ---------------------------------------------------------------------------


async def handle_calculate_bmi(params: FunctionCallParams) -> None:
    """Calculate BMI given weight in kg and height in meters."""
    args = params.arguments or {}
    try:
        weight = float(args.get("weight_kg", 0) or 0)
        height = float(args.get("height_m", 1) or 1)
    except (TypeError, ValueError):
        await params.result_callback({"error": "weight_kg and height_m must be numeric"})
        return

    if weight < 0:
        await params.result_callback({"error": "weight_kg must be non-negative"})
        return
    if height <= 0:
        await params.result_callback({"error": "height_m must be greater than zero"})
        return

    bmi = round(weight / (height * height), 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25.0:
        category = "Normal weight"
    elif bmi < 30.0:
        category = "Overweight"
    else:
        category = "Obese"

    await params.result_callback({"bmi": bmi, "category": category, "weight_kg": weight, "height_m": height})


# ---------------------------------------------------------------------------
# Current date / time
# ---------------------------------------------------------------------------


async def handle_get_current_date_time(params: FunctionCallParams) -> None:
    """Return current date and time, optionally for a requested IANA timezone."""
    args = params.arguments or {}
    requested_tz = str(args.get("timezone", "") or "").strip()

    if requested_tz:
        try:
            tz = ZoneInfo(requested_tz)
        except (ZoneInfoNotFoundError, ValueError):
            await params.result_callback(
                {
                    "error": (
                        f"Unknown timezone {requested_tz!r}; expected an IANA name "
                        "(e.g. UTC, America/New_York, Asia/Kolkata)"
                    )
                }
            )
            return
        now = datetime.now(tz)
    else:
        now = datetime.now().astimezone()

    await params.result_callback(
        {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "iso8601": now.isoformat(timespec="seconds"),
            "day_of_week": now.strftime("%A"),
            "timezone": str(now.tzinfo) if now.tzinfo else "local",
        }
    )


# ---------------------------------------------------------------------------
# Stock prices
# ---------------------------------------------------------------------------

_COMPANY_SYMBOLS: list[dict] = [
    {"keywords": ["apple"], "symbol": "AAPL", "name": "Apple Inc.", "price": 213.49},
    {"keywords": ["microsoft"], "symbol": "MSFT", "name": "Microsoft Corp.", "price": 415.32},
    {"keywords": ["google", "alphabet"], "symbol": "GOOGL", "name": "Alphabet Inc.", "price": 175.84},
    {"keywords": ["amazon"], "symbol": "AMZN", "name": "Amazon.com Inc.", "price": 196.21},
    {"keywords": ["tesla"], "symbol": "TSLA", "name": "Tesla Inc.", "price": 177.58},
    {"keywords": ["nvidia"], "symbol": "NVDA", "name": "NVIDIA Corp.", "price": 875.40},
    {"keywords": ["meta", "facebook"], "symbol": "META", "name": "Meta Platforms Inc.", "price": 527.15},
    {"keywords": ["netflix"], "symbol": "NFLX", "name": "Netflix Inc.", "price": 648.30},
    {"keywords": ["amd", "advanced micro"], "symbol": "AMD", "name": "Advanced Micro Devices", "price": 162.45},
    {"keywords": ["intel"], "symbol": "INTC", "name": "Intel Corp.", "price": 30.12},
    {"keywords": ["samsung"], "symbol": "005930.KS", "name": "Samsung Electronics", "price": 71500.0},
    {"keywords": ["tsmc", "taiwan semiconductor"], "symbol": "TSM", "name": "TSMC", "price": 145.30},
    {"keywords": ["broadcom"], "symbol": "AVGO", "name": "Broadcom Inc.", "price": 1320.0},
    {"keywords": ["qualcomm"], "symbol": "QCOM", "name": "Qualcomm Inc.", "price": 168.50},
    {"keywords": ["arm"], "symbol": "ARM", "name": "Arm Holdings", "price": 128.40},
    {"keywords": ["salesforce"], "symbol": "CRM", "name": "Salesforce Inc.", "price": 274.60},
    {"keywords": ["oracle"], "symbol": "ORCL", "name": "Oracle Corp.", "price": 128.90},
    {"keywords": ["ibm"], "symbol": "IBM", "name": "IBM Corp.", "price": 189.20},
    {"keywords": ["uber"], "symbol": "UBER", "name": "Uber Technologies", "price": 72.30},
    {"keywords": ["airbnb"], "symbol": "ABNB", "name": "Airbnb Inc.", "price": 152.80},
    {"keywords": ["spotify"], "symbol": "SPOT", "name": "Spotify Technology", "price": 318.50},
    {"keywords": ["jp morgan", "jpmorgan"], "symbol": "JPM", "name": "JPMorgan Chase", "price": 198.40},
    {"keywords": ["goldman sachs", "goldman"], "symbol": "GS", "name": "Goldman Sachs", "price": 478.20},
    {"keywords": ["visa"], "symbol": "V", "name": "Visa Inc.", "price": 274.90},
    {"keywords": ["mastercard"], "symbol": "MA", "name": "Mastercard Inc.", "price": 468.30},
    {"keywords": ["johnson", "j&j"], "symbol": "JNJ", "name": "Johnson & Johnson", "price": 147.60},
    {"keywords": ["pfizer"], "symbol": "PFE", "name": "Pfizer Inc.", "price": 27.80},
    {"keywords": ["exxon", "exxonmobil"], "symbol": "XOM", "name": "ExxonMobil Corp.", "price": 112.40},
    {"keywords": ["walmart"], "symbol": "WMT", "name": "Walmart Inc.", "price": 68.50},
    {"keywords": ["disney"], "symbol": "DIS", "name": "The Walt Disney Co.", "price": 111.30},
    {"keywords": ["boeing"], "symbol": "BA", "name": "Boeing Co.", "price": 172.60},
    {"keywords": ["ford"], "symbol": "F", "name": "Ford Motor Co.", "price": 12.40},
    {"keywords": ["general motors", "gm"], "symbol": "GM", "name": "General Motors Co.", "price": 46.80},
    {"keywords": ["coca cola", "coca-cola"], "symbol": "KO", "name": "The Coca-Cola Co.", "price": 62.30},
    {"keywords": ["pepsi", "pepsico"], "symbol": "PEP", "name": "PepsiCo Inc.", "price": 168.90},
]


def _resolve_company_symbol(query: str) -> dict | None:
    q = query.lower().strip()
    if not q:
        return None
    for entry in _COMPANY_SYMBOLS:
        if any(k in q or q in k for k in entry["keywords"]):
            return entry
    return None


async def handle_get_stock_price(params: FunctionCallParams) -> None:
    """Fetch the current stock price using Finnhub → Yahoo Finance, with a mock fallback."""
    args = params.arguments or {}
    company_name = str(args.get("company_name", "") or "").strip()
    if not company_name:
        await params.result_callback({"error": "company_name is required"})
        return

    resolved = _resolve_company_symbol(company_name)
    if resolved is None:
        await params.result_callback({"error": f"Could not find a known stock for company '{company_name}'"})
        return

    symbol = resolved["symbol"]
    name = resolved["name"]
    mock_price = resolved["price"]

    # 1) Finnhub
    finnhub_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if finnhub_key:
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                response = await client.get(
                    "https://finnhub.io/api/v1/quote",
                    params={"symbol": symbol, "token": finnhub_key},
                    headers={"Accept": "application/json"},
                )
            if response.status_code == 200:
                data = response.json()
                if data.get("c"):
                    await params.result_callback(
                        {
                            "company": name,
                            "symbol": symbol,
                            "price": data.get("c"),
                            "currency": "USD",
                            "previous_close": data.get("pc"),
                            "day_high": data.get("h"),
                            "day_low": data.get("l"),
                            "change": data.get("d"),
                            "change_percent": data.get("dp"),
                            "source": "live (finnhub)",
                        }
                    )
                    return
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug(f"finnhub lookup failed for {symbol}: {exc}")

    # 2) Yahoo Finance
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}",
                params={"interval": "1d", "range": "1d"},
                headers={"Accept": "application/json"},
            )
        if response.status_code == 200:
            data = response.json()
            result_list = data.get("chart", {}).get("result") or []
            meta = result_list[0].get("meta") if result_list else None
            if meta and meta.get("regularMarketPrice") is not None:
                await params.result_callback(
                    {
                        "company": name,
                        "symbol": symbol,
                        "price": meta.get("regularMarketPrice"),
                        "currency": meta.get("currency", "USD"),
                        "previous_close": meta.get("previousClose"),
                        "day_high": meta.get("regularMarketDayHigh"),
                        "day_low": meta.get("regularMarketDayLow"),
                        "exchange": meta.get("exchangeName"),
                        "source": "live (yahoo)",
                    }
                )
                return
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug(f"yahoo lookup failed for {symbol}: {exc}")

    # 3) Mock fallback
    await params.result_callback(
        {
            "company": name,
            "symbol": symbol,
            "price": mock_price,
            "currency": "USD",
            "source": "mock",
            "note": "Live price unavailable - returning static mock data",
        }
    )


# ---------------------------------------------------------------------------
# Random number
# ---------------------------------------------------------------------------


async def handle_generate_random_number(params: FunctionCallParams) -> None:
    """Return a uniformly random integer in ``[min, max]`` (defaults 1..100)."""
    args = params.arguments or {}
    try:
        low = int(args.get("min", 1) or 1)
        high = int(args.get("max", 100) or 100)
    except (TypeError, ValueError):
        await params.result_callback({"error": "min and max must be integers"})
        return

    if low > high:
        await params.result_callback({"error": "min must be less than or equal to max"})
        return

    await params.result_callback({"result": random.randint(low, high), "min": low, "max": high})


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

# Static mock used when WEATHERAPI_KEY is unset, so the demo works out of the
# box. Unknown cities fall back to ``_WEATHER_MOCK_DEFAULT``.
_WEATHER_MOCK: dict[str, dict] = {
    "london": {"condition": "Light rain", "temp_c": 12.0, "humidity": 78, "wind_kph": 14.0, "wind_dir": "SW"},
    "new york": {"condition": "Partly cloudy", "temp_c": 18.0, "humidity": 60, "wind_kph": 12.0, "wind_dir": "NW"},
    "tokyo": {"condition": "Clear", "temp_c": 22.0, "humidity": 55, "wind_kph": 8.0, "wind_dir": "E"},
    "san francisco": {"condition": "Foggy", "temp_c": 15.0, "humidity": 80, "wind_kph": 18.0, "wind_dir": "W"},
    "mumbai": {"condition": "Humid and hot", "temp_c": 32.0, "humidity": 85, "wind_kph": 10.0, "wind_dir": "SW"},
    "paris": {"condition": "Overcast", "temp_c": 14.0, "humidity": 70, "wind_kph": 11.0, "wind_dir": "W"},
    "sydney": {"condition": "Sunny", "temp_c": 24.0, "humidity": 58, "wind_kph": 16.0, "wind_dir": "NE"},
    "bangalore": {
        "condition": "Pleasant with showers",
        "temp_c": 26.0,
        "humidity": 72,
        "wind_kph": 9.0,
        "wind_dir": "S",
    },
}
_WEATHER_MOCK_DEFAULT = {
    "condition": "Mild and clear",
    "temp_c": 20.0,
    "humidity": 65,
    "wind_kph": 10.0,
    "wind_dir": "W",
}


def _mock_weather(city: str, use_fahrenheit: bool) -> dict:
    """Return a deterministic mock weather payload for ``city``."""
    entry = _WEATHER_MOCK.get(city.lower(), _WEATHER_MOCK_DEFAULT)
    temp_c = entry["temp_c"]
    temp = f"{round(temp_c * 9 / 5 + 32, 1)}°F" if use_fahrenheit else f"{temp_c}°C"
    return {
        "city": city,
        "condition": entry["condition"],
        "temperature": temp,
        "humidity": f"{entry['humidity']}%",
        "wind": f"{entry['wind_kph']} kph {entry['wind_dir']}",
        "source": "mock",
        "note": "Live weather unavailable - returning static mock data (set WEATHERAPI_KEY for live data)",
    }


async def handle_get_weather(params: FunctionCallParams) -> None:
    """Fetch current weather for a city via WeatherAPI, falling back to mock data."""
    args = params.arguments or {}
    city = str(args.get("city", "") or "").strip()
    if not city:
        await params.result_callback({"error": "city is required"})
        return

    use_fahrenheit = str(args.get("units", "") or "").lower().startswith("f")
    api_key = os.getenv("WEATHERAPI_KEY", "").strip()
    if not api_key:
        await params.result_callback(_mock_weather(city, use_fahrenheit))
        return

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.get(
                "https://api.weatherapi.com/v1/current.json",
                params={"key": api_key, "q": city, "aqi": "no"},
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        logger.debug(f"get_weather live lookup failed: {exc}")
        await params.result_callback(_mock_weather(city, use_fahrenheit))
        return

    if response.status_code != 200:
        logger.debug(f"get_weather live lookup non-200 for {city!r}: {response.status_code}")
        await params.result_callback(_mock_weather(city, use_fahrenheit))
        return

    data = response.json()
    loc = data.get("location", {})
    cur = data.get("current", {})
    temp = f"{cur.get('temp_f')}°F" if use_fahrenheit else f"{cur.get('temp_c')}°C"
    feels = f"{cur.get('feelslike_f')}°F" if use_fahrenheit else f"{cur.get('feelslike_c')}°C"
    await params.result_callback(
        {
            "city": loc.get("name"),
            "region": loc.get("region"),
            "country": loc.get("country"),
            "local_time": loc.get("localtime"),
            "condition": cur.get("condition", {}).get("text"),
            "temperature": temp,
            "feels_like": feels,
            "humidity": f"{cur.get('humidity')}%",
            "wind": f"{cur.get('wind_kph')} kph {cur.get('wind_dir')}",
            "visibility": f"{cur.get('vis_km')} km",
            "uv_index": cur.get("uv"),
            "source": "live (weatherapi)",
        }
    )


# ---------------------------------------------------------------------------
# News headlines (dummy)
# ---------------------------------------------------------------------------

_DUMMY_HEADLINES = [
    "Global markets rally as inflation data comes in lower than expected",
    "Scientists announce breakthrough in renewable energy storage",
    "World leaders gather for climate summit in Geneva",
    "Tech giant unveils next-generation AI assistant",
    "Major earthquake strikes Pacific region; tsunami warnings issued",
]


async def handle_get_news_headlines(params: FunctionCallParams) -> None:
    """Return three dummy news headlines."""
    args = params.arguments or {}
    result: dict = {"headlines": _DUMMY_HEADLINES[:3]}
    if args.get("country"):
        result["country"] = args["country"]
    if args.get("category"):
        result["category"] = args["category"]
    result["note"] = "dummy result - not live news"
    await params.result_callback(result)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "convert_currency": handle_convert_currency,
    "calculate_bmi": handle_calculate_bmi,
    "get_current_date_time": handle_get_current_date_time,
    "get_stock_price": handle_get_stock_price,
    "generate_random_number": handle_generate_random_number,
    "get_weather": handle_get_weather,
    "get_news_headlines": handle_get_news_headlines,
}
