# Booking Server — Dummy Database Reference

This is the SQLite-backed booking backend the airline voice agent talks
to. It's seeded with fixture PNRs and a synthesised flight catalog so
you can hold realistic conversations with the agent without running a
real reservation system. This page lists **what's actually in the
database** so you can craft queries the agent can answer.

If you ask the agent about a PNR that doesn't exist or a city that was
never seeded, the agent will (correctly) tell you it can't find the
record — so use this page to stay inside the supported envelope.

## What the agent can do

The voice agent surfaces five caller-facing flows, all backed by the
HTTP routes in `server.py`:

| Intent | What the caller can ask for |
| --- | --- |
| **Booking lookup** | Look up a PNR, read back passenger / flight / status / seat / meal |
| **Flight status** | Check status (scheduled / delayed / cancelled / diverted / misconnect) and delay minutes for any flight number |
| **Rebook** | Move an active PNR to a different flight on the same or different route. Optionally change seat / meal |
| **Cancel** | Cancel an active PNR under the appropriate fare / IRROPS policy |
| **New booking** | Create a fresh PNR on a scheduled flight, with optional seat / meal |
| **Standby** | List a PNR on an earlier flight without giving up the original |
| **Route discovery** | Ask "where can I fly from Chicago?" or "from what cities can I fly to Miami?" — works with city names or IATA codes |
| **Activity log** | Read back the audit trail (book → rebook → cancel events) for a PNR |

## Sample PNRs you can quote to the agent

The fresh demo DB (`EVAL_MODE` not set) ships with 11 fixture bookings.
Each one is tuned to exercise a specific scenario class — quote the PNR
or read the passenger name to the agent and it will pull the record.

| PNR | Passenger | Flight | Route | Departure | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `ABC123` | Jane Doe | AA123 | JFK → LAX | 2026-05-01 08:00 | scheduled | Happy-path lookup / voluntary rebook |
| `DEF456` | John Smith | AA456 | ORD → SFO | 2026-04-25 14:30 | **delayed 240m** | Long-delay refund eligibility |
| `GHI789` | Maria Garcia | AA789 | ATL → MIA | 2026-04-22 19:00 | **cancelled_weather** | Weather IRROPS — caller wants rebook |
| `JKL234` | Ahmed Khan | AA106 | JFK → LHR | 2026-05-03 19:30 | delayed 90m | International, business cabin |
| `MNO567` | Priya Patel | AA234 | BOS → ORD | 2026-04-26 06:45 | **misconnect** | Downline rebooking |
| `PQR890` | Carlos Rodriguez | AA612 | LAX → SEA | 2026-05-02 15:00 | scheduled | Basic economy — restrictive change rules |
| `STU345` | Linda Williams | AA881 | ORD → MIA | 2026-04-27 10:15 | **cancelled_airline** | Airline-caused cancel → airline refund |
| `VWX678` | Robert Chen | AA445 | DFW → ATL | 2026-04-28 13:30 | delayed 45m | Short delay, nonrefundable |
| `YZA901` | Sarah Thompson | AA1299 | DEN → LAX | 2026-04-29 09:00 | **diversion** | Diversion handling |
| `BCD234` | Michael O'Brien | AA189 | SFO → JFK | 2026-04-30 22:15 | scheduled | Red-eye, business / platinum |
| `EFG567` | Emma Davis | AA732 | MIA → BOS | 2026-05-04 07:30 | delayed 120m | Basic economy + delay |

Cancelled PNRs (`GHI789`, `STU345`) are returned read-only — the agent
will refuse to rebook or re-cancel them.

## Flight catalog

All flights live in `seed_data/flights.jsonl` (one JSON record per
line), totalling ~6,690 entries:

- **~40 curated** flights aligned with the PNR fixtures above so every
  demo scenario has at least one viable rebook target.
- **~6,650 schedules** built on real OpenFlights route topology between
  36 US hub airports across **2026-04-22 → 2026-05-10** (19 days).

### Cities covered

The expanded catalog includes these 36 US airports. The voice agent
accepts either the IATA code or the city name — say "fly me from
Chicago to Miami" or "from ORD to MIA," whichever feels natural.

| Region | Airports (IATA — City) |
| --- | --- |
| Northeast | JFK — New York (JFK), LGA — New York (LaGuardia), EWR — Newark, BOS — Boston, PHL — Philadelphia, DCA — Washington (Reagan), IAD — Washington (Dulles), BWI — Baltimore |
| Southeast / FL | ATL — Atlanta, CLT — Charlotte, MIA — Miami, FLL — Fort Lauderdale, MCO — Orlando, TPA — Tampa, RDU — Raleigh-Durham, BNA — Nashville |
| Midwest | ORD — Chicago (O'Hare), MDW — Chicago (Midway), MSP — Minneapolis-St. Paul, DTW — Detroit, STL — St. Louis, MCI — Kansas City |
| South / Central | DFW — Dallas-Fort Worth, IAH — Houston, AUS — Austin, MSY — New Orleans |
| Mountain / West | DEN — Denver, SLC — Salt Lake City, PHX — Phoenix, LAS — Las Vegas |
| West Coast | LAX — Los Angeles, SFO — San Francisco, SEA — Seattle, SAN — San Diego, PDX — Portland, SJC — San Jose |

Plus `LHR` from the curated set (so JFK ↔ LHR international flows work
out of the box for PNR `JKL234`).

### Flight number ranges

| Range | Source | Notes |
| --- | --- | --- |
| `AA100`–`AA1305` | Curated entries in `flights.jsonl` + PNR fixtures (`pnrs.jsonl`) | Stable demo identifiers (e.g. `AA123` is Jane Doe's flight) |
| `AA2000`–`AA2349` | Generated OpenFlights catalog | 350 distinct numbers, each anchored to one (route, daily-slot) pair |

A given flight number recurs across multiple days (same daily slot), so
when the caller asks the agent to rebook onto `AA2100` it picks the
earliest occurrence unless a `departure` is supplied.

### Cabin mix

The catalog targets ≈78 % economy / 14 % business / 8 % premium_economy
to match the agent's fare/policy heuristics.

## What voice queries to try

The queries below are the ones we've verified end-to-end against the
seeded data. Phrasing is illustrative — the fast LLM tolerates
paraphrase, but the **entities** (PNR, flight number, city names) must
line up with what's actually in the database.

Multi-step flows (rebook, cancel, new booking, standby, IRROPS,
activity log) have known correctness issues in the current pipeline
and are not included here yet — see
`conversation_and_flow_analysis.md` at the repo root for the open
failure modes.

### Booking lookup
Reference the PNR, not the passenger name — the agent resolves
lookups by the 6-character booking reference.

- "I want you to pull up the booking `ABC123`."
- "Read me PNR `JKL234`."

### Flight status
- "Is flight `AA456` on time?" → expects "delayed 240 minutes."
- "Status of `AA789`?" → expects "cancelled due to weather."
- "What's the status of `AA1299`?" → expects "diverted."

### Route discovery
Ask by city name — the agent resolves spoken cities to the right
airports automatically.

- "Where can I go from Chicago?"
- "How do I get to San Francisco from Seattle?"

## Date and time conventions

- All `departure` / `arrival` columns are **naive ISO-8601 local time at
  the origin airport** (no timezone suffix). The agent reads these back
  as wall-clock times — it does not perform timezone math.
- The seeded date window is **2026-04-22 through 2026-05-10** for the
  generated catalog. Curated alternatives spread across the same range.
- Asking about dates outside that window will get a "no flights found"
  reply — that's the database, not a bug in the agent.

## What's *not* in the database

- Real-time pricing (the agent quotes a flat per-cabin rate from
  `BookingAPI.price_for`, with a 2× international surcharge).
- Real seat maps (any seat string is accepted on input. `ancillaries.seat`
  is stored verbatim).
- Frequent-flyer balances (only the `elite_tier` enum is stored).
- Connections / multi-segment itineraries (each PNR is a single
  flight).
- Time zones (departures are naive ISO-8601 — see above).

## Resetting / reseeding the database

```bash
# Demo mode — wipe and reseed
rm -f data/bookings.db
PYTHONPATH=src uv run python3 -m examples.frontend_backend_agent.airline.database.server
# → seeds 6,701 flights + 11 PNRs on first boot
#   (6,690 from flights.jsonl + 11 PNR-fixture flights inserted as separate rows)
```

The flight catalog in `seed_data/flights.jsonl` and the PNR fixtures
in `seed_data/pnrs.jsonl` are committed as-is. The OpenFlights
schedules were generated once and baked in. If you need to expand the
hub set or extend the date range, edit `flights.jsonl` directly (one
JSON record per line) — `seed.py` will pick the new rows up on the
next reseed.

## HTTP API quick reference

The agent talks to this server over HTTP. You can poke the same routes
directly with `curl` for debugging:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/health` | Liveness check |
| `GET`  | `/pnrs/{pnr}` | Lookup booking |
| `GET`  | `/pnrs/{pnr}/activity` | Activity log |
| `GET`  | `/pnrs/{pnr}/ancillaries-diff?new_flight_number=...` | Preview rebook seat / bag / meal carry-over |
| `GET`  | `/flights?origin=...&destination=...` | List alternatives on a route |
| `GET`  | `/flights/{flight_number}/status` | Operational status |
| `GET`  | `/flights/{flight_number}/pnrs` | Active PNRs on a flight |
| `GET`  | `/routes?origin=...` *or* `?destination=...` | Reachable destinations / inbound origins |
| `GET`  | `/price?origin=...&destination=...&cabin=...` | Fare quote |
| `POST` | `/pnrs` | Create booking |
| `POST` | `/pnrs/{pnr}/rebook` | Commit rebook |
| `POST` | `/pnrs/{pnr}/cancel` | Cancel |
| `POST` | `/pnrs/{pnr}/standby` | Standby listing |

Server defaults to `http://localhost:8001`. Override the SQLite path
with `BOOKING_DB_PATH=...`.
