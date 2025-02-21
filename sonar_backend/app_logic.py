# sonar_app/app_logic.py

import re
import logging
import pandas as pd
import numpy as np
from transformers import pipeline
from config import Config

logger = logging.getLogger(__name__)

# Initialize GPT pipeline once
gpt_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=-1)

def parse_zoom_in_commands(query: str) -> dict:
    zoom = {}
    pattern = r"zoom in on\s+(\w+)\s+between\s+([\-\d\.]+)\s+and\s+([\-\d\.]+)"
    match = re.search(pattern, query.lower())
    if match:
        axis = match.group(1)
        lo, hi = float(match.group(2)), float(match.group(3))
        logger.info("Parsed zoom command: axis=%s, range=[%f, %f]", axis, lo, hi)
        zoom[axis] = [lo, hi]
    return zoom

def parse_search_in_commands(query: str, df: pd.DataFrame) -> pd.DataFrame:
    lower = query.lower()
    m_track = re.search(r"search track\s+(\d+)", lower)
    if m_track:
        tid = int(m_track.group(1))
        logger.info("Parsed search command: track %d", tid)
        return df[df["TrackID"] == tid]
    m_species = re.search(r"search species\s+(\w+)", lower)
    if m_species and "PredSpecies" in df.columns:
        species = m_species.group(1)
        logger.info("Parsed search command: species %s", species)
        return df[df["PredSpecies"].str.contains(species, case=False, na=False)]
    return pd.DataFrame()  # Return empty if no match

def generate_filter_expression(query: str, df: pd.DataFrame) -> str:
    prompt = (
        "You are a Python expert. Given a pandas DataFrame 'df' with columns: " +
        ", ".join(df.columns) +
        "\nExamples:\n"
        "1. Query: 'search track 2' -> Expression: df[df['TrackID'] == 2]\n"
        "2. Query: 'search species sturgeon' -> Expression: df[df['PredSpecies'].str.contains('sturgeon', case=False, na=False)]\n"
        f"Now, given the query: '{query}', output only the Python expression."
    )
    response = gpt_generator(prompt, max_new_tokens=50, do_sample=False)
    text = response[0]['generated_text'].strip()
    match = re.search(r"(df\[[^\n]+])", text)
    if match:
        return match.group(1).strip()
    else:
        lines = [line for line in text.splitlines() if line.strip() != ""]
        return lines[-1].strip() if lines else ""

def robust_filter_query(query: str, df: pd.DataFrame) -> pd.DataFrame:
    # Try direct parse
    parsed_df = parse_search_in_commands(query, df)
    if not parsed_df.empty:
        return parsed_df

    # Try zoom
    zoom_cmd = parse_zoom_in_commands(query)
    if zoom_cmd:
        for axis, rng in zoom_cmd.items():
            Config.AI_INSTRUCTIONS[f"zoom_{axis}"] = rng
        logger.info("Zoom instructions updated: %s", zoom_cmd)
        return df  # We just update zoom instructions

    # Fallback to GPT expression
    expr = generate_filter_expression(query, df)
    logger.info("GPT-generated filter expression: %s", expr)
    if not expr.strip():
        logger.warning("No valid expression from GPT; returning empty DataFrame.")
        return pd.DataFrame()

    try:
        safe_globals = {"__builtins__": {}, "df": df, "pd": pd, "np": np}
        filtered_df = eval(expr, safe_globals)
        if not isinstance(filtered_df, pd.DataFrame):
            logger.warning("GPT expression did not return a DataFrame.")
            return pd.DataFrame()
        return filtered_df
    except Exception as e:
        logger.exception("Error evaluating GPT expression: %s", e)
        return pd.DataFrame()
