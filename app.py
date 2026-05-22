import itertools
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

API_BASE = "https://api.the-odds-api.com/v4"


# -----------------------------
# Odds Math
# -----------------------------

def american_to_decimal(odds):
    odds = float(odds)
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def american_to_implied_prob(odds):
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def decimal_to_american(decimal_odds):
    decimal_odds = float(decimal_odds)
    if decimal_odds >= 2:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def no_vig_probs(prob_a, prob_b):
    total = prob_a + prob_b
    if total == 0:
        return None, None
    return prob_a / total, prob_b / total


def ev_percent(fair_prob, sportsbook_decimal):
    """
    EV = probability * payout - loss probability
    Decimal odds include stake.
    """
    return (fair_prob * sportsbook_decimal - 1) * 100


def parlay_decimal_odds(decimal_odds_list):
    product = 1
    for d in decimal_odds_list:
        product *= d
    return product


def parlay_probability(prob_list):
    """
    Assumes independence. Correlation adjustment is added separately.
    """
    product = 1
    for p in prob_list:
        product *= p
    return product


# -----------------------------
# API
# -----------------------------

@st.cache_data(ttl=120)
def get_sports(api_key):
    url = f"{API_BASE}/sports"
    params = {"apiKey": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=120)
def get_odds(api_key, sport_key, markets, bookmakers, odds_format="american"):
    url = f"{API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us,eu",
        "markets": ",".join(markets),
        "bookmakers": ",".join(bookmakers),
        "oddsFormat": odds_format,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# -----------------------------
# Parsing
# -----------------------------

def normalize_odds(raw_events):
    rows = []

    for event in raw_events:
        event_id = event.get("id")
        home = event.get("home_team")
        away = event.get("away_team")
        commence_time = event.get("commence_time")

        for book in event.get("bookmakers", []):
            book_key = book.get("key")
            book_title = book.get("title")

            for market in book.get("markets", []):
                market_key = market.get("key")
                last_update = market.get("last_update")

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    price = outcome.get("price")
                    point = outcome.get("point", None)

                    if price is None:
                        continue

                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home,
                        "away_team": away,
                        "matchup": f"{away} @ {home}",
                        "book": book_key,
                        "book_title": book_title,
                        "market": market_key,
                        "selection": name,
                        "point": point,
                        "american_odds": price,
                        "decimal_odds": american_to_decimal(price),
                        "implied_prob": american_to_implied_prob(price),
                        "last_update": last_update,
                    })

    return pd.DataFrame(rows)


def market_identity(row):
    return (
        row["event_id"],
        row["market"],
        row["point"],
    )


def selection_identity(row):
    return (
        row["event_id"],
        row["market"],
        row["selection"],
        row["point"],
    )


# -----------------------------
# Fair Odds Calculation
# -----------------------------

def calculate_fair_odds(df, retail_book="fanduel", sharp_book="pinnacle"):
    if df.empty:
        return pd.DataFrame()

    retail = df[df["book"] == retail_book].copy()
    sharp = df[df["book"] == sharp_book].copy()

    if retail.empty or sharp.empty:
        return pd.DataFrame()

    results = []

    grouped = sharp.groupby(["event_id", "market", "point"], dropna=False)

    for _, group in grouped:
        if len(group) < 2:
            continue

        # Only clean two-way markets for no-vig math
        if len(group) != 2:
            continue

        g = group.reset_index(drop=True)
        p1 = g.loc[0, "implied_prob"]
        p2 = g.loc[1, "implied_prob"]
        fair1, fair2 = no_vig_probs(p1, p2)

        fair_map = {
            selection_identity(g.loc[0]): fair1,
            selection_identity(g.loc[1]): fair2,
        }

        for _, r in retail.iterrows():
            sid = selection_identity(r)
            if sid not in fair_map:
                continue

            fair_prob = fair_map[sid]
            sportsbook_decimal = r["decimal_odds"]
            ev = ev_percent(fair_prob, sportsbook_decimal)

            results.append({
                "matchup": r["matchup"],
                "commence_time": r["commence_time"],
                "event_id": r["event_id"],
                "market": r["market"],
                "selection": r["selection"],
                "point": r["point"],
                "fanduel_american": r["american_odds"],
                "fanduel_decimal": sportsbook_decimal,
                "pinnacle_fair_prob": fair_prob,
                "fair_american": decimal_to_american(1 / fair_prob),
                "ev_percent": ev,
                "implied_prob_fanduel": r["implied_prob"],
            })

    out = pd.DataFrame(results)

    if out.empty:
        return out

    out["pinnacle_fair_prob_pct"] = out["pinnacle_fair_prob"] * 100
    out["implied_prob_fanduel_pct"] = out["implied_prob_fanduel"] * 100

    return out.sort_values("ev_percent", ascending=False)


# -----------------------------
# Correlation Logic
# -----------------------------

def correlation_score(legs):
    """
    Simple starter rules.
    This does NOT guarantee correlation.
    It creates reasonable same-game clusters.
    """

    score = 0

    event_ids = [leg["event_id"] for leg in legs]
    markets = [leg["market"] for leg in legs]
    selections = [str(leg["selection"]).lower() for leg in legs]

    same_game_count = len(event_ids) - len(set(event_ids))
    score += same_game_count * 2

    # Totals + team/player overs often correlate positively
    if "totals" in markets:
        over_count = sum("over" in s for s in selections)
        if over_count >= 1:
            score += 2

    # Spread + moneyline same side can correlate
    if "spreads" in markets and "h2h" in markets:
        score += 2

    # Too many legs from totally different games lowers correlation
    unique_games = len(set(event_ids))
    if unique_games == len(legs):
        score -= 2

    return score


def build_slips(df, category="highest_probability", slip_size=4, max_slips=10):
    if df.empty:
        return pd.DataFrame()

    pool = df.copy()

    if category == "highest_ev":
        pool = pool[pool["ev_percent"] > 0].sort_values("ev_percent", ascending=False)
    else:
        pool = pool.sort_values("pinnacle_fair_prob", ascending=False)

    pool = pool.head(25)

    slips = []

    records = pool.to_dict("records")

    for combo in itertools.combinations(records, slip_size):
        # Avoid exact duplicate market/selection conflicts
        selection_keys = [
            (
                leg["event_id"],
                leg["market"],
                leg["selection"],
                leg["point"],
            )
            for leg in combo
        ]

        if len(selection_keys) != len(set(selection_keys)):
            continue

        probs = [leg["pinnacle_fair_prob"] for leg in combo]
        decimals = [leg["fanduel_decimal"] for leg in combo]
        evs = [leg["ev_percent"] for leg in combo]

        base_prob = parlay_probability(probs)
        payout_decimal = parlay_decimal_odds(decimals)
        corr_score = correlation_score(combo)

        # Conservative correlation boost cap
        adjusted_prob = base_prob * (1 + min(max(corr_score, 0), 6) * 0.04)
        adjusted_prob = min(adjusted_prob, 0.95)

        parlay_ev = ev_percent(adjusted_prob, payout_decimal)

        slips.append({
            "legs": combo,
            "slip_size": slip_size,
            "category": category,
            "base_probability_pct": base_prob * 100,
            "adjusted_probability_pct": adjusted_prob * 100,
            "correlation_score": corr_score,
            "parlay_decimal_odds": payout_decimal,
            "parlay_american_odds": decimal_to_american(payout_decimal),
            "average_leg_ev_pct": np.mean(evs),
            "parlay_ev_pct": parlay_ev,
        })

    slip_df = pd.DataFrame(slips)

    if slip_df.empty:
        return slip_df

    if category == "highest_ev":
        slip_df = slip_df.sort_values(
            ["parlay_ev_pct", "correlation_score"],
            ascending=False
        )
    else:
        slip_df = slip_df.sort_values(
            ["adjusted_probability_pct", "correlation_score"],
            ascending=False
        )

    return slip_df.head(max_slips)


def format_leg(leg):
    point = "" if pd.isna(leg["point"]) else f" {leg['point']}"
    return (
        f"{leg['matchup']} | "
        f"{leg['market']} | "
        f"{leg['selection']}{point} | "
        f"FanDuel {leg['fanduel_american']} | "
        f"Fair {leg['fair_american']} | "
        f"EV {leg['ev_percent']:.2f}%"
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Sharp Odds + Correlation Slip Builder", layout="wide")

st.title("Sharp Odds + Correlation Slip Builder")
st.caption("FanDuel vs Pinnacle-style sharp fair odds, no-vig pricing, EV, and correlation slip generation.")

api_key = st.secrets.get("ODDS_API_KEY", "")

if not api_key:
    st.error("Add your API key to .streamlit/secrets.toml as ODDS_API_KEY.")
    st.stop()

with st.sidebar:
    st.header("Settings")

    try:
        sports = get_sports(api_key)
    except Exception as e:
        st.error(f"Could not load sports: {e}")
        st.stop()

    active_sports = [s for s in sports if s.get("active")]
    sport_map = {f"{s['title']} ({s['key']})": s["key"] for s in active_sports}

    sport_label = st.selectbox(
        "Sport",
        list(sport_map.keys()),
        index=0
    )

    sport_key = sport_map[sport_label]

    markets = st.multiselect(
        "Markets",
        ["h2h", "spreads", "totals"],
        default=["h2h", "spreads", "totals"]
    )

    retail_book = st.text_input("Retail book key", value="fanduel")
    sharp_book = st.text_input("Sharp book key", value="pinnacle")

    slip_size = st.slider("Slip size", 2, 6, 4)

    max_slips = st.slider("Number of slips", 3, 25, 10)

    min_ev = st.number_input("Minimum single-leg EV %", value=-100.0, step=0.5)

    min_prob = st.number_input("Minimum fair probability %", value=0.0, step=1.0)

    refresh = st.button("Refresh Odds")


bookmakers = [retail_book, sharp_book]

try:
    raw = get_odds(api_key, sport_key, markets, bookmakers)
except Exception as e:
    st.error(f"Could not load odds: {e}")
    st.stop()

df = normalize_odds(raw)

if df.empty:
    st.warning("No odds found. Try another sport, market, or bookmaker key.")
    st.stop()

fair_df = calculate_fair_odds(
    df,
    retail_book=retail_book,
    sharp_book=sharp_book
)

if fair_df.empty:
    st.warning(
        "No matched FanDuel/Pinnacle markets found. "
        "Try h2h, spreads, totals, or confirm your bookmaker keys are available."
    )
    st.dataframe(df)
    st.stop()

filtered = fair_df[
    (fair_df["ev_percent"] >= min_ev) &
    (fair_df["pinnacle_fair_prob_pct"] >= min_prob)
].copy()

tab1, tab2, tab3, tab4 = st.tabs([
    "Best Single Bets",
    "Highest Probability Slips",
    "Highest EV Slips",
    "Raw Odds"
])

with tab1:
    st.subheader("Best Single Bets")

    show_cols = [
        "matchup",
        "market",
        "selection",
        "point",
        "fanduel_american",
        "fair_american",
        "pinnacle_fair_prob_pct",
        "implied_prob_fanduel_pct",
        "ev_percent",
    ]

    st.dataframe(
        filtered[show_cols].sort_values("ev_percent", ascending=False),
        use_container_width=True
    )

with tab2:
    st.subheader("Highest Probability Correlation Slips")

    prob_slips = build_slips(
        filtered,
        category="highest_probability",
        slip_size=slip_size,
        max_slips=max_slips
    )

    if prob_slips.empty:
        st.info("No probability slips found with the current filters.")
    else:
        for i, row in prob_slips.iterrows():
            with st.expander(
                f"Slip | Adjusted Probability {row['adjusted_probability_pct']:.2f}% | "
                f"Odds {row['parlay_american_odds']} | Corr {row['correlation_score']}"
            ):
                st.write(f"Base probability: {row['base_probability_pct']:.2f}%")
                st.write(f"Adjusted probability: {row['adjusted_probability_pct']:.2f}%")
                st.write(f"Parlay EV: {row['parlay_ev_pct']:.2f}%")
                st.write(f"Average leg EV: {row['average_leg_ev_pct']:.2f}%")
                st.write("### Legs")
                for leg in row["legs"]:
                    st.write("- " + format_leg(leg))

with tab3:
    st.subheader("Highest EV Correlation Slips")

    ev_slips = build_slips(
        filtered,
        category="highest_ev",
        slip_size=slip_size,
        max_slips=max_slips
    )

    if ev_slips.empty:
        st.info("No EV slips found with the current filters.")
    else:
        for i, row in ev_slips.iterrows():
            with st.expander(
                f"Slip | Parlay EV {row['parlay_ev_pct']:.2f}% | "
                f"Odds {row['parlay_american_odds']} | Corr {row['correlation_score']}"
            ):
                st.write(f"Base probability: {row['base_probability_pct']:.2f}%")
                st.write(f"Adjusted probability: {row['adjusted_probability_pct']:.2f}%")
                st.write(f"Average leg EV: {row['average_leg_ev_pct']:.2f}%")
                st.write("### Legs")
                for leg in row["legs"]:
                    st.write("- " + format_leg(leg))

with tab4:
    st.subheader("Raw Normalized Odds")
    st.dataframe(df, use_container_width=True)
