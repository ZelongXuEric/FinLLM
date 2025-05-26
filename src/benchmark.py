import os
import json
import re
import logging
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

BASE       = Path(os.environ.get("FINLLM_BASE", "/content/drive/MyDrive/FinLLM"))
PROC_DIR   = BASE / "data" / "processed"
RESULT_DIR = BASE / "results"
MODEL_DIRS = {
    "qwen3-8b":            BASE / "models" / "qwen3-8b",
    "llama3-8b-instruct":  BASE / "models" / "Llama-3-8B-Instruct",
}

RESULT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(BASE)
logging.info(f"CWD → {BASE}")

KEY_CAUSAL = "causal_llm_pred"
KEY_ASSOC  = "assoc_llm_pred"
KEY_CAUSE  = "cause_llm_pred"
KEY_EFFECT = "effect_llm_pred"

ANNOTATED_CSV = Path(BASE) / "data" / "interim" / "annotated.csv"
BENCH_JSONL   = PROC_DIR / "benchmark.jsonl"

def convert_csv_to_jsonl(csv_path: str, jsonl_path: str, field_map: dict):
    df = pd.read_csv(csv_path)
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            record = { json_key: row[csv_col] for csv_col, json_key in field_map.items() }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

convert_csv_to_jsonl(
    str(ANNOTATED_CSV),
    str(BENCH_JSONL),
    field_map={
        "event":          "event_text",
        "ground_causal":  "ground_causal",
        "cause":          "ground_cause",
        "effect":         "ground_effect",
    }
)

def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        logging.error(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_json(path, lines=True)

def std_cols(df: pd.DataFrame, ticker, id_col, text_col) -> pd.DataFrame:
    """Rename and standardize columns for Tesla/Apple data."""
    if df.empty:
        return df
    d = df.copy()
    d["ticker"] = ticker
    d.rename(columns={id_col: "event_id", text_col: "event_text"}, inplace=True)

    for c in list(d.columns):
        if c.startswith(f"{ticker}_"):
            d.rename(columns={c: c.replace(f"{ticker}_", "")}, inplace=True)
    return d

def build_prompt(event_text: str, reaction_label: str) -> str:
    payload = json.dumps({
        "event": event_text[:1024],
        "reaction": reaction_label
    }, ensure_ascii=False)
    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"{FEW_SHOT_EXAMPLES_STR}\n\n"
        f"User: {payload}\nAssistant:"
    )

def parse_llm(raw: str) -> dict:
    """Extract JSON keys or fall back to regex."""
    out = {
        KEY_CAUSAL: "ParseErr_Causal",
        KEY_ASSOC:  "ParseErr_Assoc",
        KEY_CAUSE:  "N/A",
        KEY_EFFECT: "N/A",
    }
    raw = (raw or "").strip()
    if not raw:
        return out

    #strip code fences
    raw = re.sub(r"```(?:json)?[\s\S]*?```", "", raw)

    #find {...}
    for m in re.finditer(r"\{[\s\S]*?\}", raw):
        try:
            obj = json.loads(m.group(0))
            # normalize yes/no
            yn = lambda v: "Yes" if str(v).lower() in ("yes","true","y","是") else "No"
            if "causal" in obj:
                out[KEY_CAUSAL] = yn(obj["causal"])
            if "assoc" in obj:
                out[KEY_ASSOC]  = yn(obj["assoc"])
            if "cause" in obj:
                out[KEY_CAUSE]  = str(obj["cause"]).strip()
            if "effect" in obj:
                out[KEY_EFFECT] = str(obj["effect"]).strip()
            return out
        except json.JSONDecodeError:
            continue

    #fallback regex for labels
    def rex(key):
        return re.search(rf'"{key}"\s*:\s*"([^"]+)"', raw, re.I)
    if rex("causal"):
        out[KEY_CAUSAL] = "Yes" if rex("causal").group(1).lower() in ("yes","true","y","是") else "No"
    if rex("assoc"):
        out[KEY_ASSOC]  = "Yes" if rex("assoc").group(1).lower() in ("yes","true","y","是") else "No"
    if rex("cause"):
        out[KEY_CAUSE]  = rex("cause").group(1).strip()
    if rex("effect"):
        out[KEY_EFFECT] = rex("effect").group(1).strip()

    return out

# Prompt & Few-Shot Setup
SYSTEM_PROMPT = """
You are a meticulous financial analyst.
Given an event summary and the next-day move, decide assoc & causal.
Return exactly one JSON: {"assoc":"Yes|No","causal":"Yes|No","cause":"…","effect":"…"}
""".strip()

FEW_SHOT_EXAMPLES_STR = """
User: {"event":"X beats earnings","reaction":"Significant Up"}
Assistant: {"assoc":"Yes","causal":"Yes","cause":"X beats earnings","effect":"Stock up significantly"}

User: {"event":"Routine filing","reaction":"Flat"}
Assistant: {"assoc":"No","causal":"No","cause":"Routine filing","effect":"Stock flat"}
""".strip()

GEN_KW = {
    "max_new_tokens": 64,
    "do_sample": False,
    # pad/eos token will be injected per-model below
}

if __name__ == "__main__":
    # 1. load and concat human labels
    df_t = load_jsonl(PROC_DIR / "tesla_manual.jsonl")
    df_a = load_jsonl(PROC_DIR / "apple_manual.jsonl")
    df_t = std_cols(df_t, "TSLA", "news_id", "summary")
    df_a = std_cols(df_a, "AAPL", "filing_id", "item_text")
    EVAL_DF = pd.concat([df_t, df_a], ignore_index=True)
    logging.info(f"Total eval samples: {len(EVAL_DF)}")

    # 2. load models
    loaded = []
    for name, mdir in MODEL_DIRS.items():
        if mdir.exists():
            tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                mdir, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
            mdl.eval()
            loaded.append((name, tok, mdl))
            logging.info(f"Loaded model {name}")
        else:
            logging.error(f"Model dir not found: {mdir}")

    # 3. run each
    for model_name, tokenizer, model in loaded:
        logging.info(f"=== Benchmarking {model_name} ===")
        # inject pad/eos
        GEN_KW["pad_token_id"] = tokenizer.eos_token_id
        GEN_KW["eos_token_id"] = tokenizer.eos_token_id

        recs = []
        for _, row in tqdm(EVAL_DF.iterrows(), total=len(EVAL_DF), desc=model_name):
            prompt = build_prompt(row["event_text"], row.get("cat_D+1","N/A"))
            inp    = tokenizer(prompt, return_tensors="pt").to(model.device)
            outids = model.generate(**inp, **GEN_KW)
            raw    = tokenizer.decode(outids[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            pred   = parse_llm(raw)

            recs.append({
                "event_id":      row["event_id"],
                "ticker":        row["ticker"],
                "ground_causal": row.get("causal_label",""),
                "ground_assoc":  row.get("assoc_label",""),
                KEY_CAUSAL:      pred[KEY_CAUSAL],
                KEY_ASSOC:       pred[KEY_ASSOC],
                "raw_llm":       raw[:300],
            })

        df_res = pd.DataFrame(recs)
        out_path = RESULT_DIR / f"{model_name}_benchmark.csv"
        df_res.to_csv(out_path, index=False)
        logging.info(f"Saved results → {out_path}")
