import pandas as pd
import numpy as np
import re
import sys


def remove_tags(row):
    row = str(row)
    cleanr = re.compile('(</?[a-zA-Z]+>|https?:\/\/[^\s]*|(^|\s)RT(\s|$)|@[^\s]+|\d+)')
    cleantext = re.sub(cleanr, ' ', row)
    cleantext = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)',' ',cleantext)
    cleantext = re.sub('[^\sa-zA-Z]+','',cleantext)
    cleantext = re.sub('\s+',' ',cleantext)
    cleantext = cleantext[1:].strip()
    return cleantext


def clean_text(row):
    row = str(row).encode('ascii', 'ignore').strip()
    row = remove_tags(row)

    return row


# train_data_path = r"data/training-Obama-Romney-tweets.xlsx"
#
# obama_df = pd.read_excel(train_data_path, sheetname=0)
# romney_df = pd.read_excel(train_data_path, sheetname=1)
#
# obama_df['tweet'] = obama_df['tweet'].apply(clean_text)
# romney_df['tweet'] = romney_df['tweet'].apply(clean_text)
#
# print("Writing cleaned training output")
# obama_df.to_csv('data/obama_csv.csv', sep='\t')
# romney_df.to_csv('data/romney_csv.csv', sep='\t')
#
# print("Writing cleaned testing output")

test_data_path = r"data/testing-Obama-Romney-tweets.xlsx"

obama_df = pd.read_excel(test_data_path, sheetname=0)
romney_df = pd.read_excel(test_data_path, sheetname=1)

obama_df['tweet'] = obama_df['tweet'].apply(clean_text)
romney_df['tweet'] = romney_df['tweet'].apply(clean_text)

obama_df.to_csv('data/obama_csv_test.csv', sep='\t')
romney_df.to_csv('data/romney_csv_test.csv', sep='\t')

print("Writing cleaned testing output")