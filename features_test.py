"""
Check for presence of previously selected best n.grams in the test dataset and save results in  file
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

selected_features_file = "features_best_50.csv"
dir_benign = "distinct_ngrams_benign_test"
dir_malicious = "distinct_ngrams_malware_test"
filenames_benign = sorted(os.walk(dir_benign))[0][2]
filenames_malicious = sorted(os.walk(dir_malicious))[0][2]
data_benign = pd.DataFrame(filenames_benign, columns=["file"])
data_malicious = pd.DataFrame(filenames_malicious, columns=["file"])

output_file = "features_test.csv"

with open(selected_features_file) as f:
    columns = f.readline().splitlines()[0].split(',')[1:]  # columns are file, class, and then ngrams
data=pd.DataFrame(columns = columns)

ngrams = columns[2:]

for file in tqdm(filenames_benign):
    with open(dir_benign + "/" + file) as f:
        file_ngrams = f.read().splitlines()
    new_row = pd.Series([file, 0] + [1 if ngram in file_ngrams else 0 for ngram in ngrams], index=columns)
    data = data.append(new_row, ignore_index=True)

for file in tqdm(filenames_malicious):
    with open(dir_malicious + "/" + file) as f:
        file_ngrams = f.read().splitlines()
    new_row = pd.Series([file, 1] + [1 if ngram in file_ngrams else 0 for ngram in ngrams], index=columns)
    data = data.append(new_row, ignore_index=True)

data.to_csv(output_file)