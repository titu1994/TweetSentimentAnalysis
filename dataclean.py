import pandas as pd
import numpy as np

obama_df = pd.read_csv('data/obama_csv.csv', encoding='latin1', sep='|')
romney_df = pd.read_csv('data/romney_csv.csv', encoding='latin1', sep='|')

print(obama_df.head())