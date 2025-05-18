from __future__ import annotations
import os, time, json, argparse, logging, requests
from datetime import datetime, timedelta, date
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from .config import get as cfg

PROJECT_DIR  = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DATA_DIR = PROJECT_DIR / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

RAPID_KEY  = cfg("RAPIDAPI_KEY")
RAPID_HOST = "reuters-business-and-financial-news.p.rapidapi.com"
EDGAR_HEADERS = {"User-Agent": cfg("EDGAR_USER_AGENT", "FinLLM/1.0")}
EDGAR_BASE_URL = "https://www.sec.gov"
EDGAR_API_URL  = "https://data.sec.gov/submissions/"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")

# Reuters Fetch
def fetch_reuters(keyword:str, start:str, end:str, per_page:int=20)->list[dict]:
    all_items=[]
    page=0
    while True:
        url=(f"https://{RAPID_HOST}/get-articles-by-keyword-name-date-range/"
             f"{start}/{end}/{keyword}/{page}/{per_page}")
        r=requests.get(url, headers={"x-rapidapi-key":RAPID_KEY,
                                     "x-rapidapi-host":RAPID_HOST}, timeout=15)
        if r.status_code!=200: break
        data=r.json(); arts=data.get("articles") or []
        if not arts: break
        all_items+=arts
        logging.info("Reuters %s page %d ◎ %d", keyword, page, len(arts))
        page+=1
        if page>=data.get("allPages", page): break
        time.sleep(1)
    out=RAW_DATA_DIR/f"reuters_{keyword.lower()}_news.json"
    json.dump(all_items, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    logging.info("Saved %d → %s", len(all_items), out)
    return all_items

# EDGAR Fetch
def latest_filings(cik:str, form:str="8-K", n:int=5)->list[dict]:
    url=f"{EDGAR_API_URL}CIK{cik.zfill(10)}.json"
    recent=pd.DataFrame(requests.get(url,headers=EDGAR_HEADERS).json()
                        ['filings']['recent'])
    recent=recent[recent['form']==form].head(n)
    info=[]
    for _,r in recent.iterrows():
        acc=r['accessionNumber']; no_dash=acc.replace('-','')
        info.append({
            "id":acc,
            "filing_date":r['filingDate'],
            "html_url":f"{EDGAR_BASE_URL}/Archives/edgar/data/{cik}/{no_dash}/{r['primaryDocument']}"
        })
    return info

def fetch_html_text(url:str)->str:
    soup=BeautifulSoup(requests.get(url,headers=EDGAR_HEADERS).text,"html.parser")
    body=soup.find("document") or soup.find("body") or soup
    for tag in body(["script","style","meta","title"]): tag.extract()
    return "\n".join([t for t in body.get_text("\n",strip=True).split("\n") if t.strip()])

def fetch_edgar(cik:str, tag:str, form:str="8-K", n:int=5):
    rec=[]
    for meta in latest_filings(cik,form,n):
        meta["full_filing_text"]=fetch_html_text(meta["html_url"])
        rec.append(meta)
        time.sleep(.5)
    out=RAW_DATA_DIR/f"edgar_{form.lower()}_{cik}_{tag}.json"
    json.dump(rec, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    logging.info("Saved %d → %s", len(rec), out)
    return rec


# FRED Simulation for Complete Process
def save_fake_cpi():
    sim = {'date':[(date(2022,5,1)+timedelta(days=i*30)).strftime('%Y-%m-%d')
                   for i in range(36)],
           'CPIAUCSL':[290+i*0.5+(i%5-2)*0.1 for i in range(36)]}
    out=RAW_DATA_DIR/'fred_cpiaucsl.csv'
    pd.DataFrame(sim).to_csv(out,index=False)
    logging.info("Wrote fake CPI → %s", out)

# CLI
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--keyword", default="Tesla")
    ap.add_argument("--start",   default="2024-01-01")
    ap.add_argument("--end",     default="2025-01-01")
    args=ap.parse_args()

    if not RAPID_KEY:
        raise RuntimeError("RAPIDAPI_KEY is not set, and it is filled in the environment variable or. env")

    fetch_reuters(args.keyword, args.start, args.end)
    fetch_edgar("0000320193", "apple", form="8-K", n=5)
    save_fake_cpi()

if __name__=="__main__":
    main()
