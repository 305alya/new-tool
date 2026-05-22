import itertools
import requests
import numpy as np
import pandas as pd
import streamlit as st

API_BASE = "https://api.odds-api.io/v3"


# -----------------------------
# ODDS MATH
# -----------------------------

def american_to_decimal(odds):
    odds = float(odds)
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def decimal_to_american(decimal_odds):
    decimal_odds = float(decimal_odds)
    if decimal_odds >= 2:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def decimal_to_prob(decimal_odds):
    return 1 / float(decimal_odds)


def american_to_prob(odds):
    return 1 / american_to_decimal(odds)


def ev_percent(fair_prob, book_decimal):
    return (fair_prob * book_decimal - 1) * 100


def no_vig_two_way(p1, p2):
    total = p1 + p2
    if total == 0:
        return None, None
    return p1 / total, p2 / total


def parlay_decimal(legs):
    result = 1
    for leg in legs:
        result *= leg["retail_decimal"]
    return result


def parlay_prob(legs):
    result = 1
    for leg in legs:
        result *= leg["fair_prob"]
    return result


# -----------------------------
# API HELPERS
# -----------------------------

def get_api_key():
    return "e8e5c1f63bc6d0f6f6f52b6d4a8f3c9f8f90d6a6aac16bcf2297ad0ce4c1e808"


def api_get(path, params=None):
    api_key = get_api_key()

    if not api_key:
        st.error("Missing API key. Add ODDS_API_KEY in Streamlit Secrets.")
        st.stop()

    params = params or {}
    params["apiKey"] = api_key

    url = f"{API_BASE}{path}"

    response = requests.get(url, params=params, timeout=30)

    if response.status_code == 401:
        st.error("401 Unauthorized. Your API key is not accepted by Odds-API.io.")
        st.stop()

    if response.status_code == 404:
        st.error(f"404 Not Found. Endpoint issue: {response.url}")
        st.stop()

    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def fetch_leagues():
    data = api_get("/leagues")

    if isinstance(data, dict):
        for key in ["data", "leagues", "results"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    if isinstance(data, list):
        return data

    return []


@st.cache_data(ttl=180)
def fetch_events(sport):
    data = api_get("/events", {"sport": sport})

    if isinstance(data, dict):
        for key in ["data", "events", "results"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    if isinstance(data, list):
        return data

    return []


@st.cache_data(ttl=120)
def fetch_event_odds(event_id):
    possible_paths = [
        f"/odds/{event_id}",
        "/odds",
    ]

    for path in possible_paths:
        try:
            if path == "/odds":
                data = api_get(path, {"eventId": event_id})
            else:
                data = api_get(path)
            return data
        except Exception:
            continue

    return None


# -----------------------------
# NORMALIZATION
# -----------------------------

def find_value(obj, possible_keys):
    if not isinstance(obj, dict):
        return None

    for key in possible_keys:
        if key in obj:
            return obj[key]

    return None


def normalize_leagues(leagues):
    rows = []

    for item in leagues:
        if not isinstance(item, dict):
            continue

        sport = find_value(item, ["sport", "key", "slug", "id", "league"])
        title = find_value(item, ["title", "name", "label", "league_name"])
        event_count = find_value(item, ["event_count", "events", "live_event_count", "count"])

        if sport:
            rows.append({
                "sport": sport,
                "title": title or sport,
                "event_count": event_count,
            })

    return pd.DataFrame(rows)


def normalize_events(events):
    rows = []

    for event in events:
        if not isinstance(event, dict):
            continue

        event_id = find_value(event, ["id", "event_id", "eventId"])
        home = find_value(event, ["home", "home_team", "homeTeam"])
        away = find_value(event, ["away", "away_team", "awayTeam"])
        start_time = find_value(event, ["startTime", "start_time", "commence_time", "date"])

        if event_id:
            rows.append({
                "event_id": event_id,
                "home_team": home,
                "away_team": away,
                "matchup": f"{away} @ {home}" if home and away else str(event_id),
                "start_time": start_time,
            })

    return pd.DataFrame(rows)


def extract_odds_rows(data, event_info):
    rows = []

    if data is None:
        return rows

    possible_wrappers = ["data", "odds", "bookmakers", "markets", "results"]

    def unwrap(x):
        if isinstance(x, dict):
            for key in possible_wrappers:
                if key in x and isinstance(x[key], list):
                    return x[key]
        return x

    data = unwrap(data)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return rows

    def walk(obj, current_book=None, current_market=None):
        if isinstance(obj, dict):
            book = find_value(obj, ["bookmaker", "book", "sportsbook", "site", "name"]) or current_book
            market = find_value(obj, ["market", "market_name", "marketKey", "type"]) or current_market

            selection = find_value(obj, ["selection", "outcome", "name", "team", "label"])
            price = find_value(obj, ["price", "odds", "decimal", "decimalOdds", "americanOdds"])
            point = find_value(obj, ["point", "line", "handicap", "total"])

            if selection is not None and price is not None and book is not None:
                try:
                    price_float = float(price)

                    if price_float > 20 or price_float < -20:
                        decimal_price = american_to_decimal(price_float)
                        american_price = price_float
                    else:
                        decimal_price = price_float
                        american_price = decimal_to_american(decimal_price)

                    rows.append({
                        "event_id": event_info["event_id"],
                        "matchup": event_info["matchup"],
                        "start_time": event_info["start_time"],
                        "book": str(book).lower().replace(" ", "_"),
                        "book_raw": str(book),
                        "market": str(market or "unknown").lower(),
                        "selection": str(selection),
                        "point": point,
                        "american_odds": american_price,
                        "decimal_odds": decimal_price,
                        "implied_prob": decimal_to_prob(decimal_price),
                    })
                except Exception:
                    pass

            for value in obj.values():
                walk(value, book, market)

        elif isinstance(obj, list):
            for item in obj:
                walk(item, current_book, current_market)

    walk(data)
    return rows


def load_all_odds(events_df, max_events):
    all_rows = []

    selected_events = events_df.head(max_events)

    progress = st.progress(0)

    for i, event in selected_events.iterrows():
        event_info = event.to_dict()
        odds_data = fetch_event_odds(event_info["event_id"])
        rows = extract_odds_rows(odds_data, event_info)
        all_rows.extend(rows)
        progress.progress((i + 1) / len(selected_events))

    return pd.DataFrame(all_rows)


# -----------------------------
# VALUE / FAIR ODDS
# -----------------------------

def calculate_edges(df, retail_book, sharp_book):
    if df.empty:
        return pd.DataFrame()

    retail_book = retail_book.lower().replace(" ", "_")
    sharp_book = sharp_book.lower().replace(" ", "_")

    retail = df[df["book"].str.contains(retail_book, na=False)].copy()
    sharp = df[df["book"].str.contains(sharp_book, na=False)].copy()

    if retail.empty or sharp.empty:
        return pd.DataFrame()

    results = []

    group_cols = ["event_id", "market", "point"]

    for _, sharp_group in sharp.groupby(group_cols, dropna=False):
        if len(sharp_group) != 2:
            continue

        sharp_group = sharp_group.reset_index(drop=True)

        p1 = sharp_group.loc[0, "implied_prob"]
        p2 = sharp_group.loc[1, "implied_prob"]

        fair1, fair2 = no_vig_two_way(p1, p2)

        fair_map = {
            str(sharp_group.loc[0, "selection"]).lower(): fair1,
            str(sharp_group.loc[1, "selection"]).lower(): fair2,
        }

        same_retail = retail[
            (retail["event_id"] == sharp_group.loc[0, "event_id"]) &
            (retail["market"] == sharp_group.loc[0, "market"]) &
            (retail["point"].astype(str) == str(sharp_group.loc[0, "point"]))
        ]

        for _, r in same_retail.iterrows():
            selection_key = str(r["selection"]).lower()

            if selection_key not in fair_map:
                continue

            fair_prob = fair_map[selection_key]
            retail_decimal = r["decimal_odds"]

            results.append({
                "event_id": r["event_id"],
                "matchup": r["matchup"],
                "start_time": r["start_time"],
                "market": r["market"],
                "selection": r["selection"],
                "point": r["point"],
                "retail_book": r["book_raw"],
                "sharp_book": sharp_group.loc[0, "book_raw"],
                "retail_american": r["american_odds"],
                "retail_decimal": retail_decimal,
                "fair_prob": fair_prob,
                "fair_prob_pct": fair_prob * 100,
                "fair_american": decimal_to_american(1 / fair_prob),
                "ev_percent": ev_percent(fair_prob, retail_decimal),
            })

    return pd.DataFrame(results).sort_values("ev_percent", ascending=False)


# -----------------------------
# CORRELATION SLIPS
# -----------------------------

def correlation_score(legs):
    score = 0

    event_ids = [leg["event_id"] for leg in legs]
    markets = [str(leg["market"]).lower() for leg in legs]
    selections = [str(leg["selection"]).lower() for leg in legs]

    if len(set(event_ids)) == 1:
        score += 5
    elif len(set(event_ids)) < len(event_ids):
        score += 2

    if any("total" in m or "over" in m for m in markets):
        if any("over" in s for s in selections):
            score += 1

    if any("spread" in m or "handicap" in m for m in markets):
        score += 1

    if len(set(event_ids)) == len(event_ids):
        score -= 2

    return score


def build_slips(edges, slip_size, mode, max_slips):
    if edges.empty:
        return pd.DataFrame()

    if mode == "Highest EV":
        pool = edges[edges["ev_percent"] > 0].sort_values("ev_percent", ascending=False).head(30)
    else:
        pool = edges.sort_values("fair_prob", ascending=False).head(30)

    records = pool.to_dict("records")
    slips = []

    for combo in itertools.combinations(records, slip_size):
        keys = [(x["event_id"], x["market"], x["selection"], str(x["point"])) for x in combo]

        if len(keys) != len(set(keys)):
            continue

        base_prob = parlay_prob(combo)
        odds_decimal = parlay_decimal(combo)
        corr = correlation_score(combo)

        adjusted_prob = base_prob * (1 + max(corr, 0) * 0.03)
        adjusted_prob = min(adjusted_prob, 0.95)

        slips.append({
            "legs": combo,
            "base_probability_pct": base_prob * 100,
            "adjusted_probability_pct": adjusted_prob * 100,
            "parlay_decimal": odds_decimal,
            "parlay_american": decimal_to_american(odds_decimal),
            "correlation_score": corr,
            "average_ev_pct": np.mean([x["ev_percent"] for x in combo]),
            "parlay_ev_pct": ev_percent(adjusted_prob, odds_decimal),
        })

    out = pd.DataFrame(slips)

    if out.empty:
        return out

    if mode == "Highest EV":
        return out.sort_values(["parlay_ev_pct", "correlation_score"], ascending=False).head(max_slips)

    return out.sort_values(["adjusted_probability_pct", "correlation_score"], ascending=False).head(max_slips)


def format_leg(leg):
    point = "" if pd.isna(leg["point"]) or leg["point"] is None else f" {leg['point']}"
    return (
        f"{leg['matchup']} | {leg['market']} | {leg['selection']}{point} | "
        f"{leg['retail_book']} {leg['retail_american']} | "
        f"Fair {leg['fair_american']} | EV {leg['ev_percent']:.2f}%"
    )


# -----------------------------
# STREAMLIT APP
# -----------------------------

st.set_page_config(page_title="Sharp Odds + Correlation Slip Builder", layout="wide")

st.title("Sharp Odds + Correlation Slip Builder")
st.caption("Odds-API.io version: FanDuel vs Pinnacle-style sharp fair odds, no-vig pricing, EV, and correlation slips.")

with st.sidebar:
    st.header("Settings")

    leagues_raw = fetch_leagues()
    leagues_df = normalize_leagues(leagues_raw)

    if leagues_df.empty:
        st.error("Could not load leagues from Odds-API.io.")
        st.stop()

    leagues_df = leagues_df.sort_values("title")

    league_options = {
        f"{row['title']} — {row['sport']}": row["sport"]
        for _, row in leagues_df.iterrows()
    }

    selected_label = st.selectbox("Sport / League", list(league_options.keys()))
    selected_sport = league_options[selected_label]

    retail_book = st.text_input("Retail book", value="fanduel")
    sharp_book = st.text_input("Sharp book", value="pinnacle")

    max_events = st.slider("Events to scan", 1, 25, 8)
    slip_size = st.slider("Slip size", 2, 6, 4)
    max_slips = st.slider("Number of slips", 3, 20, 8)

    min_ev = st.number_input("Minimum single-leg EV %", value=-100.0, step=0.5)
    min_prob = st.number_input("Minimum fair probability %", value=0.0, step=1.0)

    scan = st.button("Scan Odds")

if not scan:
    st.info("Choose a league, then click Scan Odds.")
    st.stop()

events_raw = fetch_events(selected_sport)
events_df = normalize_events(events_raw)

if events_df.empty:
    st.warning("No events found for this league right now.")
    st.stop()

st.subheader("Events Found")
st.dataframe(events_df.head(max_events), use_container_width=True)

odds_df = load_all_odds(events_df, max_events)

if odds_df.empty:
    st.warning("No odds found. Your plan may not include this league/bookmaker, or the event has no markets yet.")
    st.stop()

edges_df = calculate_edges(odds_df, retail_book, sharp_book)

if edges_df.empty:
    st.warning(
        "No matched FanDuel vs Pinnacle-style edges found. "
        "Try changing the book names to match what appears in Raw Odds."
    )

    st.subheader("Available Books Found")
    st.write(sorted(odds_df["book_raw"].dropna().unique()))

    st.subheader("Raw Odds")
    st.dataframe(odds_df, use_container_width=True)
    st.stop()

filtered = edges_df[
    (edges_df["ev_percent"] >= min_ev) &
    (edges_df["fair_prob_pct"] >= min_prob)
].copy()

tab1, tab2, tab3, tab4 = st.tabs([
    "Best Single Bets",
    "Highest Probability Slips",
    "Highest EV Slips",
    "Raw Odds"
])

with tab1:
    st.subheader("Best Single Bets")

    cols = [
        "matchup",
        "market",
        "selection",
        "point",
        "retail_book",
        "sharp_book",
        "retail_american",
        "fair_american",
        "fair_prob_pct",
        "ev_percent",
    ]

    st.dataframe(filtered[cols], use_container_width=True)

with tab2:
    st.subheader("Highest Probability Slips")

    slips = build_slips(filtered, slip_size, "Highest Probability", max_slips)

    if slips.empty:
        st.info("No slips found.")
    else:
        for _, row in slips.iterrows():
            with st.expander(
                f"Probability {row['adjusted_probability_pct']:.2f}% | "
                f"Odds {row['parlay_american']} | Corr {row['correlation_score']}"
            ):
                st.write(f"Base probability: {row['base_probability_pct']:.2f}%")
                st.write(f"Adjusted probability: {row['adjusted_probability_pct']:.2f}%")
                st.write(f"Parlay EV: {row['parlay_ev_pct']:.2f}%")

                for leg in row["legs"]:
                    st.write("- " + format_leg(leg))

with tab3:
    st.subheader("Highest EV Slips")

    slips = build_slips(filtered, slip_size, "Highest EV", max_slips)

    if slips.empty:
        st.info("No positive-EV slips found.")
    else:
        for _, row in slips.iterrows():
            with st.expander(
                f"Parlay EV {row['parlay_ev_pct']:.2f}% | "
                f"Odds {row['parlay_american']} | Corr {row['correlation_score']}"
            ):
                st.write(f"Base probability: {row['base_probability_pct']:.2f}%")
                st.write(f"Adjusted probability: {row['adjusted_probability_pct']:.2f}%")
                st.write(f"Average leg EV: {row['average_ev_pct']:.2f}%")

                for leg in row["legs"]:
                    st.write("- " + format_leg(leg))

with tab4:
    st.subheader("Raw Odds")
    st.dataframe(odds_df, use_container_width=True)

    st.subheader("Books Detected")
    st.write(sorted(odds_df["book_raw"].dropna().unique()))
