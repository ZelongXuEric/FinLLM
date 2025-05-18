import os
import json
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup

logging.basicConfig(format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

try:
    BASE_DIR = Path(os.environ.get("FINLLM_BASE_PROJECT_PATH", Path(__file__).resolve().parent.parent))
except NameError:
    BASE_DIR = Path(os.environ.get("FINLLM_BASE_PROJECT_PATH", ".")) 

RAW_DIR     = BASE_DIR / 'data' / 'raw'
INTERIM_DIR = BASE_DIR / 'data' / 'interim'
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

logging.info(f"Base project directory: {BASE_DIR}")
logging.info(f"Raw data directory: {RAW_DIR}")
logging.info(f"Interim data directory: {INTERIM_DIR}")

HTML_RE = re.compile(r'<[^>]+>')

def strip_html(txt: str | None) -> str:
    if not isinstance(txt, str):
        return ''
    return HTML_RE.sub('', txt).strip()

CAUSAL_KEYWORDS = [
    'because', 'due to', 'as a result', 'led to', 'cause', 'caused by', 'causes',
    'trigger', 'triggered', 'triggers', 'consequently', 'therefore', 'thus',
    'attributed to', 'impact', 'impacted by', 'impacts', 'following', 'since',
    'affect', 'affected by', 'affects', 'influence', 'influenced by', 'influences',
    'spur', 'spurred by', 'spurs', 'stemmed from', 'derive from', 'responsible for',
    'explain', 'explains', 'explained by', 'reason for', 'rationale for', 'driven by',
    'owe to', 'owing to', 'thanks to', 'on account of', 'resulting from', 'outcome of',
    'prompted', 'effect of', 'consequence of', 'factor in', 'source of'
]

def contains_causal_keyword(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    text_lower = text.lower()
    return any(re.search(r'\b' + re.escape(keyword) + r'\b', text_lower) for keyword in CAUSAL_KEYWORDS)

def further_clean_and_truncate_text(text, max_chars=3000, max_sents=15):
    if not isinstance(text, str) or text == "N/A" or not text.strip():
        return text if isinstance(text, str) else "N/A"

    boilerplate_patterns = [
        re.compile(r"The information contained in this Current Report.*?shall not be deemed.*?filed.*?\.", re.IGNORECASE | re.DOTALL),
        re.compile(r"Pursuant to the requirements of the Securities Exchange Act of 1934.*?\.", re.IGNORECASE | re.DOTALL),
        re.compile(r"SIGNATURES.*$", re.IGNORECASE | re.DOTALL),
        re.compile(r"Exhibit\s*\d+\.\d+.*$", re.IGNORECASE | re.MULTILINE),
    ]
    cleaned_text = text
    for pattern in boilerplate_patterns:
        cleaned_text = pattern.sub("", cleaned_text).strip()

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_text)
    selected_sentences = sentences[:max_sents]
    final_text = " ".join(selected_sentences).strip()

    if len(final_text) > max_chars:
        if '.' in final_text[:max_chars]:
            final_text = final_text[:max_chars].rsplit('.', 1)[0] + "."
        else:
            final_text = final_text[:max_chars].rsplit(' ', 1)[0] + "..."

    return final_text if final_text else "N/A (Cleaned to empty)"

def load_raw_json(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_json(path)
            logging.info(f"Successfully loaded {len(df)} records from {path}")
            return df
        except ValueError as e:
            logging.error(f"Error reading JSON from {path}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {path}: {e}")
            return pd.DataFrame()
    else:
        logging.warning(f"File not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()

def preprocess_reuters(df: pd.DataFrame) -> pd.DataFrame:
    target_cols = ['news_id', 'published_datetime_utc', 'published_date_utc', 'title', 'summary', 'has_causal_keyword']
    if df.empty:
        logging.info("Reuters input DataFrame is empty.")
        return pd.DataFrame(columns=target_cols)

    d = df.copy()
    required_input_cols = ['articlesId', 'publishedAt', 'articlesName', 'articlesDescription']
    if not all(col in d.columns for col in required_input_cols):
        logging.error(f"Reuters DataFrame missing one of required columns: {required_input_cols}")
        return pd.DataFrame(columns=target_cols)

    def parse_pub_datetime(pub_at_val):
        if isinstance(pub_at_val, dict) and 'date' in pub_at_val:
            try:
                dt_str = pub_at_val['date'].split('.')[0].replace('Z', '+00:00')
                return pd.to_datetime(dt_str, utc=True, errors='coerce')
            except Exception: return pd.NaT
        return pd.NaT

    d['published_datetime_utc'] = d['publishedAt'].apply(parse_pub_datetime)
    d.dropna(subset=['published_datetime_utc'], inplace=True)
    d['published_date_utc'] = d['published_datetime_utc'].dt.date
    d['title'] = d['articlesName'].apply(strip_html)

    def build_summary_from_description(desc_val):
        if isinstance(desc_val, list):
            texts = [item.get('content', '') for item in desc_val if isinstance(item, dict) and 'content' in item]
            return strip_html(' '.join(texts))
        return strip_html(str(desc_val))

    d['summary_raw'] = d['articlesDescription'].apply(build_summary_from_description)
    d['summary'] = d['summary_raw'].apply(lambda x: further_clean_and_truncate_text(x, max_chars=1500, max_sents=7))
    
    d.rename(columns={'articlesId': 'news_id'}, inplace=True)
    
    d['has_causal_keyword'] = d['summary'].apply(contains_causal_keyword)
    logging.info(f"Reuters: {d['has_causal_keyword'].sum()} of {len(d)} summaries contain causal keywords.")
    return d[target_cols]

def preprocess_edgar(df: pd.DataFrame, form_type: str) -> pd.DataFrame:
    target_cols = ['filing_id', 'filing_date', 'item_text', 'has_causal_keyword']
    if df.empty:
        logging.info(f"EDGAR {form_type} input DataFrame is empty.")
        return pd.DataFrame(columns=target_cols)

    d = df.copy()
    required_input_cols = ['id', 'filing_date', 'full_filing_text']
    if not all(col in d.columns for col in required_input_cols):
        logging.error(f"EDGAR DataFrame missing one of required columns: {required_input_cols}")
        return pd.DataFrame(columns=target_cols)

    d.rename(columns={'id': 'filing_id'}, inplace=True)
    d['filing_date'] = pd.to_datetime(d['filing_date'], errors='coerce').dt.date
    d.dropna(subset=['filing_date'], inplace=True)

    def extract_relevant_item_from_full_text(full_text_html_content: str, target_form_type: str) -> str:
        if not isinstance(full_text_html_content, str) or not full_text_html_content.strip():
            return "N/A (Empty input)"

        # Parse HTML with BeautifulSoup to extract text more accurately.
        soup = BeautifulSoup(full_text_html_content, 'html.parser')
        
        doc_tag = soup.find('document') or soup.find('DOCUMENT')
        text_content_element = doc_tag if doc_tag else (soup.find('body') or soup)

        if not text_content_element:
            return "N/A (Cannot find document/body tag)"

        # Remove non-content tags such as scripts and styles.
        for tag in text_content_element(["script", "style", "meta", "link", "title", "table"]):
            tag.extract()
        
        full_text = text_content_element.get_text('\n', strip=True)
        full_text = "\n".join([line for line in full_text.split('\n') if line.strip()])


        # 8-K usually focuses on a specific ITEM.
        if target_form_type.upper() == "8-K":
            item_patterns_ordered = [
                r"Item\s+2\.02[\s\S]+?(?=Item\s+\d\.|$)",
                r"Item\s+1\.01[\s\S]+?(?=Item\s+\d\.|$)",
                r"Item\s+5\.02[\s\S]+?(?=Item\s+\d\.|$)",
                r"Item\s+8\.01[\s\S]+?(?=Item\s+\d\.|$)",
                r"Item\s+5\.07[\s\S]+?(?=Item\s+\d\.|$)",
                r"(ITEM\s*\d+\.\d+[\s\S]+?)(?=ITEM\s*\d+\.\d+|$)"
            ]
            extracted_content = ""
            for pattern in item_patterns_ordered:
                match = re.search(pattern, full_text, flags=re.I | re.DOTALL)
                if match:
                    extracted_content = match.group(0)
                    break
            if not extracted_content
                logging.warning(f"No primary ITEM found for 8-K, consider full text or specific handling.")
                extracted_content = full_text 
        elif target_form_type.upper() == "10-K" or target_form_type.upper() == "10-Q":
            # For 10-K/10-Q, "Management's Discussion and Analysis" (MD&A) always lies in Item 7
            # "Risk Factors" always lies in Item 1A
            mda_match = re.search(r"Item\s+7\.?\s*Management's Discussion and Analysis[\s\S]+?(?=Item\s+[78]\.|Item\s+\d{2}\.)", full_text, flags=re.I | re.DOTALL)
            if mda_match:
                extracted_content = mda_match.group(0)
            else:
                # fallback to a portion of the document if MD&A not found
                extracted_content = full_text
        else:
            extracted_content = full_text # Other types of financial reports, the full text for the time being

        return further_clean_and_truncate_text(extracted_content, max_chars=3000, max_sents=15)

    d['item_text'] = d['full_filing_text'].apply(lambda x: extract_relevant_item_from_full_text(x, form_type))
    d['has_causal_keyword'] = d['item_text'].apply(contains_causal_keyword)
    logging.info(f"EDGAR {form_type}: {d['has_causal_keyword'].sum()} of {len(d)} items contain causal keywords.")
    return d[target_cols]


TH_RETURNS = dict(up_big=2.0, up_small=0.5, down_small=-0.5, down_big=-2.0)
def categorize_return(x: float | None) -> str:
    if pd.isna(x) or not isinstance(x, (int, float)): return 'N/A'
    if x >= TH_RETURNS['up_big']:    return 'Significant Up'
    if x >= TH_RETURNS['up_small']:  return 'Small Up'
    if x <= TH_RETURNS['down_big']:  return 'Significant Down'
    if x <= TH_RETURNS['down_small']:return 'Small Down'
    return 'Flat'

def fetch_stock_data(dates_series: pd.Series, ticker_symbol: str) -> pd.DataFrame:
    if dates_series.empty or dates_series.dropna().empty:
        logging.warning(f"No valid dates for {ticker_symbol} stock fetching.")
        return pd.DataFrame()

    valid_dates = pd.to_datetime(dates_series.dropna(), errors='coerce').dt.date
    valid_dates.dropna(inplace=True)
    if valid_dates.empty:
        logging.warning(f"No valid dates after conversion for {ticker_symbol}.")
        return pd.DataFrame()

    # Expand the date range to ensure coverage around the event.
    start_fetch = (valid_dates.min() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_fetch   = (valid_dates.max() + timedelta(days=30)).strftime('%Y-%m-%d')
    logging.info(f"Fetching stock data for {ticker_symbol} from {start_fetch} to {end_fetch}")

    try:
        stock_df = yf.download(ticker_symbol, start=start_fetch, end=end_fetch, progress=False, auto_adjust=False) # auto_adjust=False to get Adj Close
    except Exception as e:
        logging.error(f"Error downloading {ticker_symbol} stock data: {e}")
        return pd.DataFrame()

    if stock_df.empty:
        logging.warning(f"No stock data for {ticker_symbol} in range {start_fetch} to {end_fetch}.")
        return pd.DataFrame()

    stock_df.reset_index(inplace=True)
    stock_df.rename(columns={
        'Date': 'trade_date', 'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
    }, inplace=True)

    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date']).dt.date
    stock_df['adj_close'] = pd.to_numeric(stock_df['adj_close'], errors='coerce')
    stock_df.sort_values(by='trade_date', inplace=True)

    # D0: Today's yield (relative to the previous day's closing price)
    stock_df['ret_pct_D0'] = stock_df['adj_close'].pct_change() * 100
    # D+1: the yield on the first trading day after the event (relative to the closing price on the event day)
    # D+2: the yield on the second trading day after the event (relative to the closing price on the first trading day after the event)
    # Here we simplify it to the holding period yield of 1 day and 2 days after the event date (relative to the event date)
    # Note: It is assumed that the event occurs after the close of trading or before the opening of the next trading day, so we will look at the price changes after     the event date.

    # Calculate the future income relative to the event day (d)
    # D+1 return: (Close_D+1 / Close_D) - 1
    # D+2 return: (Close_D+2 / Close_D) - 1 (cumulative over 2 days)
    # or (close _ d+2/close _ d+1)-1 (marginal for d+2)
    # For simplicity and correspondence with the meanings of D+1 and D+2 in notebook (stock price performance in the next day and the next two days)
    # We calculate the income from the closing of the event day to the closing of the I-th trading day in the future.
    for i in [1, 2]:
        stock_df[f'ret_pct_D+{i}'] = (stock_df['adj_close'].shift(-i) / stock_df['adj_close'] - 1) * 100

    stock_df['cat_D0']  = stock_df['ret_pct_D0'].apply(categorize_return)
    stock_df['cat_D+1'] = stock_df['ret_pct_D+1'].apply(categorize_return)
    stock_df['cat_D+2'] = stock_df['ret_pct_D+2'].apply(categorize_return)

    cols_to_return = ['trade_date', 'adj_close', 
                      'ret_pct_D0', 'cat_D0',
                      'ret_pct_D+1', 'cat_D+1',
                      'ret_pct_D+2', 'cat_D+2']
    return stock_df[cols_to_return]


def align_events_with_stock_prices(events_df: pd.DataFrame, stock_prices_df: pd.DataFrame,
                                   event_date_column: str, stock_ticker_symbol: str) -> pd.DataFrame:
    if events_df.empty:
        logging.warning(f"Events DF for {stock_ticker_symbol} is empty. Skipping alignment.")
        return events_df
    if stock_prices_df.empty:
        logging.warning(f"Stock prices DF for {stock_ticker_symbol} is empty. Merging will result in NaNs for stock data.")
        # Add empty stock columns if they don't exist to maintain schema
        stock_data_cols_expected = ['adj_close', 'ret_pct_D0', 'cat_D0', 'ret_pct_D+1', 'cat_D+1', 'ret_pct_D+2', 'cat_D+2']
        temp_df = events_df.copy()
        for col in stock_data_cols_expected:
            temp_df[f'{stock_ticker_symbol}_{col}'] = np.nan
        return temp_df

    # Ensure date columns are of the same type (date objects)
    events_df[event_date_column] = pd.to_datetime(events_df[event_date_column]).dt.date
    stock_prices_df['trade_date'] = pd.to_datetime(stock_prices_df['trade_date']).dt.date

    # Merge events with stock data for the event date (D0)
    # We need to find the closest NEXT trading day if the event date is not a trading day
    # For simplicity here, we'll use merge_asof which can find nearest, but forward fill is better for "next available"
    # However, a simple left merge on date is often sufficient if events are typically on trading days or we accept NaNs
    merged_df = pd.merge(events_df, stock_prices_df.rename(columns={'trade_date': event_date_column}),
                         on=event_date_column, how='left')

    # Prefix stock columns
    stock_cols_to_prefix = [col for col in stock_prices_df.columns if col != 'trade_date']
    rename_map = {col: f"{stock_ticker_symbol}_{col}" for col in stock_cols_to_prefix if col in merged_df.columns}
    merged_df.rename(columns=rename_map, inplace=True)

    logging.info(f"Aligned {stock_ticker_symbol} events. Shape: {merged_df.shape}")
    return merged_df

def save_df_to_jsonl(df_to_save: pd.DataFrame, output_path: Path):
    if df_to_save.empty:
        logging.warning(f"DataFrame is empty. Skipping save to {output_path}.")
        return
    try:
        with output_path.open('w', encoding='utf-8') as f:
            for record_dict in df_to_save.to_dict(orient='records'):
                for key, value in record_dict.items():
                    if isinstance(value, (datetime, date, pd.Timestamp)):
                        record_dict[key] = value.isoformat()
                    elif pd.isna(value):
                        record_dict[key] = None
                f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
        logging.info(f"Successfully saved {len(df_to_save)} rows to {output_path}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to {output_path}: {e}")

if __name__ == '__main__':
    # Tesla
    tesla_raw_path = RAW_DIR / 'reuters_tesla_news.json'
    tesla_raw_df = load_raw_json(tesla_raw_path)
    tesla_events_df = preprocess_reuters(tesla_raw_df)
    if not tesla_events_df.empty:
        tesla_stock_df = fetch_stock_data(tesla_events_df['published_date_utc'], 'TSLA')
        tesla_aligned_df = align_events_with_stock_prices(tesla_events_df, tesla_stock_df, 'published_date_utc', 'TSLA')
        save_df_to_jsonl(tesla_aligned_df, INTERIM_DIR / 'tesla_news_stock_aligned_with_causal_keywords.jsonl')
    else:
        logging.warning("Skipping Tesla processing as preprocessed events DataFrame is empty.")

    # Apple 8-K
    apple_cik = "0000320193"
    apple_tag = "apple"
    apple_form = "8-K"
    apple_raw_path = RAW_DIR / f"edgar_{apple_form.lower()}_{apple_cik}_{apple_tag}.json"
    apple_raw_df = load_raw_json(apple_raw_path)
    apple_events_df = preprocess_edgar(apple_raw_df, apple_form)
    if not apple_events_df.empty:
        apple_stock_df = fetch_stock_data(apple_events_df['filing_date'], 'AAPL')
        apple_aligned_df = align_events_with_stock_prices(apple_events_df, apple_stock_df, 'filing_date', 'AAPL')
        save_df_to_jsonl(apple_aligned_df, INTERIM_DIR / 'apple_edgar_8k_stock_aligned_with_causal_keywords.jsonl')
    else:
        logging.warning("Skipping Apple 8-K processing as preprocessed events DataFrame is empty.")
        
    logging.info("Data alignment and causal keyword screening complete.")