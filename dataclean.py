import pandas as pd
import numpy as np
import sys

data_path = r"data/training-Obama-Romney-tweets.xlsx"
obama_df = pd.read_excel(data_path, sheetname=0)
romney_df = pd.read_excel(data_path, sheetname=1)

# obama_df.to_csv('data/obama_csv.csv', sep='|', index_label='id')
# romney_df.to_csv('data/romney_csv.csv', sep='|', index_label='id')
#
# obama_df = pd.read_csv('data/obama_csv.csv', encoding='latin1', sep='|')
# romney_df = pd.read_csv('data/romney_csv.csv', encoding='latin1', sep='|')

def clean_text(row):
    row = str(row).encode('ascii', 'ignore').strip()
    return row

obama_df['tweet'] = obama_df['tweet'].apply(clean_text)
romney_df['tweet'] = romney_df['tweet'].apply(clean_text)

print("Writing cleaned op")
obama_df.to_csv('data/obama_csv.csv', sep='\t')
romney_df.to_csv('data/romney_csv.csv', sep='\t')