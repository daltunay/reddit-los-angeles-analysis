import json
import os
import re
import typing as tp
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from tenacity import retry, wait_fixed

load_dotenv(".env")

THREAD_URL = "https://www.reddit.com/r/MovingToLosAngeles/comments/1kzwad1/moving_to_la_with_two_salaries_of_150k_working_in/"

NEIGHBORHOOD_ALIASES = {
    "Culver City": ["culver city", "culver"],
    "Santa Monica": ["santa monica", "saMo", "samo"],
    "Palms": ["palms"],
    "Venice": ["venice"],
    "Marina del Rey": ["marina del rey", "marina del ray", "mdr", "marina"],
    "Mar Vista": ["mar vista"],
    "Playa Vista": ["playa vista"],
    "Playa del Rey": ["playa del rey", "playa del ray", "pdr"],
    "Del Rey": ["del rey", "del ray"],
}
NEIGHBORHOODS = list(NEIGHBORHOOD_ALIASES.keys())


class RedditComment(BaseModel):
    text: str
    upvotes: int


@retry(wait=wait_fixed(10))
def fetch_reddit_comments(url: str) -> list[RedditComment]:
    print("Fetching Reddit comments...")
    if not url.endswith(".json"):
        url += ".json"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    comments: list[RedditComment] = []

    def parse_comments(children: list[dict]) -> None:
        for child in children:
            d = child.get("data", {})
            if child.get("kind") == "t1":
                comments.append(
                    RedditComment(
                        text=d.get("body", ""),
                        upvotes=d.get("score", 0),
                    )
                )
                replies = d.get("replies")
                if replies and isinstance(replies, dict):
                    parse_comments(replies["data"]["children"])

    parse_comments(data[1]["data"]["children"])
    _save_json([c.model_dump() for c in comments], "1_comments.json")
    return comments


def extract_neighborhood_mentions(text: str) -> list[str]:
    text_lower = text.lower()
    return [
        n
        for n, aliases in NEIGHBORHOOD_ALIASES.items()
        if any(
            re.search(rf"\b{re.escape(alias.lower())}\b", text_lower)
            for alias in aliases
        )
    ]


def _save_json(data, filename: str) -> None:
    print(f"Saving {filename}")
    path = Path("data") / filename
    path.parent.mkdir(exist_ok=True)
    if isinstance(data, pd.DataFrame):
        data.to_json(path, force_ascii=False, indent=2, orient="records")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def compute_mentions(comments: list[RedditComment]) -> pd.DataFrame:
    print("Computing neighborhood mentions...")
    rows = [
        {
            "text": c.text,
            "upvotes": c.upvotes,
            "mentions": extract_neighborhood_mentions(c.text),
        }
        for c in comments
    ]
    df = pd.DataFrame(rows)
    _save_json(df, "2_mentions.json")
    return df


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing stats...")
    stats = [
        {
            "neighborhood": n,
            "mention_count": int(df["mentions"].apply(lambda m: n in m).sum()),
            "upvote_sum": int(
                df.loc[df["mentions"].apply(lambda m: n in m), "upvotes"].sum()
            ),
        }
        for n in NEIGHBORHOODS
    ]
    stats_df = pd.DataFrame(stats).sort_values("mention_count", ascending=False)
    _save_json(stats_df, "3_stats.json")
    return stats_df


class ProConItem(BaseModel):
    name: str = Field(
        title="Name",
        description="Name of the pro or con",
    )
    severity: tp.Literal["low", "medium", "high"] = Field(
        title="Severity",
        description="Severity of the pro or con",
    )


class NeighborhoodProsCons(BaseModel):
    pros: list[ProConItem] = Field(
        title="Pros",
        description="List of pros for the neighborhood",
        default_factory=list,
    )
    cons: list[ProConItem] = Field(
        title="Cons",
        description="List of cons for the neighborhood",
        default_factory=list,
    )


@retry(wait=wait_fixed(1))
def summarize_neighborhood_pros_cons(
    neighborhood: str, comments: list[RedditComment]
) -> NeighborhoodProsCons:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": "You are an expert assistant for summarizing Reddit discussions about Los Angeles neighborhoods. "
                        f"You will be focusing on the neighborhood '{neighborhood}'. "
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"Given Reddit comments recommending the neighborhood '{neighborhood}' in Los Angeles, "
                            f"summarize the main pros and cons for living there ({neighborhood}) for two young professionals moving from Paris. "
                            "Return your answer as a JSON object with two fields: 'pros' (a list of pros) and 'cons' (a list of cons). "
                            "If there are no pros or cons, return an empty list for that field. "
                            "Each pro or con should be a dictionary with 'name' (a short description) and 'severity' (low, medium, high). "
                            "The 'severity' should reflect how strongly the comments support or oppose the point, and should be based on the number of comments and upvotes. "
                            "Your response should be concise and focused on the most relevant points. "
                            "Only include pros and cons that are mentioned by multiple commenters or have significant upvotes. "
                            "Do not include any personal opinions or unverified information."
                        )
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Comments (each with upvotes):\n"
                            + "\n".join(
                                f"- {c.text.replace(chr(10), ' ')} (upvotes: {c.upvotes})"
                                for c in comments
                            )
                        ),
                    }
                ],
            },
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": NeighborhoodProsCons,
        },
    )
    return NeighborhoodProsCons.model_validate(response.parsed)


def compute_pros_cons(df: pd.DataFrame) -> dict[str, NeighborhoodProsCons]:
    print("Summarizing pros and cons for each neighborhood...")
    results = {}
    for i, n in enumerate(NEIGHBORHOODS, 1):
        print(f"  [{i}/{len(NEIGHBORHOODS)}] Analyzing: {n} ...", end="", flush=True)
        mask = df["mentions"].apply(lambda m: n in m)
        comments_n = [
            RedditComment(text=row["text"], upvotes=row["upvotes"])
            for _, row in df[mask].iterrows()
        ]
        if comments_n:
            results[n] = summarize_neighborhood_pros_cons(n, comments_n).model_dump()
            print("done.")
        else:
            print("skipped (no mentions).")
    _save_json(results, "4_pros_cons.json")
    return results


def compute_final(
    stats: pd.DataFrame, pros_cons: dict[str, NeighborhoodProsCons]
) -> list[dict[str, tp.Any]]:
    final = [
        {
            "neighborhood": row["neighborhood"],
            "mention_count": row["mention_count"],
            "upvote_sum": row["upvote_sum"],
            "pros": pros_cons.get(row["neighborhood"], {}).get("pros", []),
            "cons": pros_cons.get(row["neighborhood"], {}).get("cons", []),
        }
        for _, row in stats.iterrows()
    ]
    _save_json(final, "5_final_analysis.json")
    return final


def main() -> None:
    comments = fetch_reddit_comments(THREAD_URL)
    mentions = compute_mentions(comments)
    stats = compute_stats(mentions)
    pros_cons = compute_pros_cons(mentions)
    final = compute_final(stats, pros_cons)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
