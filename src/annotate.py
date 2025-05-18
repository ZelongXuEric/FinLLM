import os
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
INTERIM_DIR  = os.path.join(BASE_DIR, '..', 'data', 'interim')
PROCESSED_DIR= os.path.join(BASE_DIR, '..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

def annotate(name, infile, outfile):
    df = pd.read_json(os.path.join(INTERIM_DIR, infile), lines=True)
    # Initialize manual label column
    for col in ('causal_label','assoc_label'):
        df[col] = ''
        
    df.to_csv(os.path.join(PROCESSED_DIR, outfile), index=False)

if __name__ == '__main__':
    annotate(
        'tesla',
        'tesla_news_stock_aligned_keyword_screened.jsonl',
        'tesla_for_pure_manual_annotation.csv'
    )
    annotate(
        'apple',
        'apple_edgar_stock_aligned_keyword_screened.jsonl',
        'apple_for_pure_manual_annotation.csv'
    )
