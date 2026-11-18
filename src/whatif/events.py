"""WhatIf's mapping of recorded Events and page views onto its parties and topics.

The sources themselves (GDELT, Wikipedia page views) live in the data layer. What a record
means for WhatIf is decided here: which party a GDELT record involves, which topic it is about,
when a party was founded, and which Wikipedia articles measure attention to which party or
topic. These are modelling choices that belong to the researcher, so they are the visible
tables below.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

from whatif.data import gdelt, wikipedia
from whatif.data.gdelt import EventCount, EventLog  # noqa: F401  (part of this module's interface)
from whatif.data.polls import TRACKED_PARTIES
from whatif.data.wikipedia import PageViewLog, PageViews  # noqa: F401

# Every Event is filed under one topic. "other" is anything the keywords below don't place.
TOPICS: tuple[str, ...] = ("economy", "migration", "climate", "security", "social", "other")

# A record involves a party when one of these phrases appears, as whole words, in either actor's
# name or in the source article's URL path. Umlauts are written out (ü → ue), and a few are also
# given with the umlaut dropped (ü → u), because GDELT and URLs do both. A record can involve
# several parties; it counts once for each.
PARTY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "CDU/CSU": (
        "cdu", "csu", "christian democrat", "christian democrats", "christian democratic union",
        "merz", "merkel", "laschet", "kramp karrenbauer", "soeder", "soder",
    ),
    "AfD": (
        "afd", "alternative for germany", "alternative fuer deutschland", "alternative fur deutschland",
        "weidel", "chrupalla", "hoecke", "hocke", "gauland", "meuthen",
    ),
    "SPD": (
        "spd", "social democrat", "social democrats", "social democratic party",
        "scholz", "esken", "klingbeil", "nahles",
    ),
    "Greens": ("greens", "green party", "gruene", "gruenen", "grune", "grunen", "habeck", "baerbock"),
    "Left": ("linke", "linken", "left party", "wissler", "reichinnek", "gysi"),
    "BSW": ("bsw", "wagenknecht"),
    "FDP": ("fdp", "free democrat", "free democrats", "free democratic party", "lindner"),
}

# Parties founded after 2017: their records before this day are ignored. Sahra Wagenknecht was a
# Left politician until October 2023, so before then her name would wrongly count for BSW.
PARTY_FOUNDED: dict[str, date] = {"BSW": date(2024, 1, 8)}

# A record's topic comes from its source URL path only (actor names are GDELT's own coding and
# say little about the policy area). The first topic, in TOPICS order, with a matching word wins.
# A word ending in "*" matches any word it starts; the others must match a whole word.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "economy": (
        "econom*", "inflation*", "wirtschaft*", "unemploy*", "arbeitslos*", "recession*", "rezession*",
        "steuer*", "budget*", "haushalt*", "schulden*", "wage", "wages", "lohn", "loehne", "job", "jobs",
        "tax", "taxes", "debt", "price", "prices", "preise",
    ),
    "migration": (
        "migra*", "immigra*", "refugee*", "fluecht*", "asyl*", "deport*", "abschieb*", "border", "borders",
        "grenze", "grenzen",
    ),
    "climate": (
        "climat*", "klima*", "energ*", "heizung*", "heating", "emission*", "coal", "kohle*", "renewabl*",
        "akw", "atomkraft*", "nuclear",
    ),
    "security": (
        "ukrain*", "russia*", "russland*", "militar*", "bundeswehr*", "defen*", "verteidigung*", "nato",
        "terror*", "anschlag*", "attack*", "police*", "polizei*", "crime*", "security", "sicherheit*",
        "war", "krieg*",
    ),
    "social": (
        "pension*", "rente*", "health*", "gesundheit*", "pflege*", "rent", "miete*", "housing", "wohnung*",
        "welfare", "buergergeld*",
    ),
}

# German Wikipedia articles whose daily views measure attention to a party or a topic.
# CDU and CSU are two articles for one party.
ATTENTION_ARTICLES: dict[str, str] = {
    "Christlich_Demokratische_Union_Deutschlands": "CDU/CSU",
    "Christlich-Soziale_Union_in_Bayern": "CDU/CSU",
    "Alternative_für_Deutschland": "AfD",
    "Sozialdemokratische_Partei_Deutschlands": "SPD",
    "Bündnis_90/Die_Grünen": "Greens",
    "Die_Linke": "Left",
    "Bündnis_Sahra_Wagenknecht": "BSW",
    "Freie_Demokratische_Partei": "FDP",
    "Inflation": "economy",
    "Arbeitslosigkeit": "economy",
    "Migration": "migration",
    "Asylrecht_(Deutschland)": "migration",
    "Globale_Erwärmung": "climate",
    "Klimawandel": "climate",
    "Bundeswehr": "security",
    "Kriminalität": "security",
    "Rente": "social",
}
assert set(ATTENTION_ARTICLES.values()) <= set(TRACKED_PARTIES) | set(TOPICS)


def _parties(names: list[str], path: list[str]) -> list[str]:
    text = " " + " ".join(names + path) + " "
    return [p for p, phrases in PARTY_KEYWORDS.items() if any(f" {phrase} " in text for phrase in phrases)]


def _topic(path: list[str]) -> str:
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword.endswith("*"):
                if any(word.startswith(keyword[:-1]) for word in path):
                    return topic
            elif keyword in path:
                return topic
    return "other"


def classify(names: list[str], path: list[str]) -> tuple[list[str], str]:
    """The parties a GDELT record involves (from actor names and URL) and its topic (from the URL)."""
    return _parties(names, path), _topic(path)


GDELT = gdelt.gdelt_adapter(classify, name="gdelt-events", fetch_command="fetch-events")
PAGEVIEWS = wikipedia.pageviews_adapter(ATTENTION_ARTICLES, name="wikipedia-pageviews", fetch_command="fetch-attention")


def collect_gdelt(download: Callable[[str], bytes], start: date, end: date) -> bytes:
    """GDELT for a date range, aggregated by WhatIf's parties and topics."""
    return gdelt.collect(download, start, end, classify)


def collect_pageviews(download: Callable[[str], bytes], start: date, end: date) -> bytes:
    """Page views of WhatIf's articles for a date range."""
    return wikipedia.collect(download, start, end, ATTENTION_ARTICLES)


def event_log(data_dir: Path) -> gdelt.EventLog:
    return gdelt.event_log(GDELT, data_dir)


def page_view_log(data_dir: Path) -> wikipedia.PageViewLog:
    return wikipedia.page_view_log(PAGEVIEWS, data_dir)
