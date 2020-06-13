from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import json
import re
import sys


"""
Creates labels for each token for inclusion and exclusion criteria

CSV must have query and cohort columns with each entry in cohort structured as a dictionary with two keys: inclusion and exclusion. Values for thees keys must be lists
    - e.g. {"inclusion": ["example a", "example b"], "exclusion": ["exclusion a"]}

Parameters
----------
document: Path to csv with "query" and "cohort"
output_name: Desired name of output csv


Returns
-------
A csv containing original columns with adidtion of inclusion, exclusion, and label columns.

Label example:
    - query = "undergoing routine antenatal care but don't have adverse effect, caused by correct medicinal substance properly administered"
    - labels = "Neither, Neither, include, include, include, Neither, Neither, Neither, Neither, Neither, exclude, exclude, Neither, exclude, exclude, exclude, exclude, Neither, Neither, include"

Labels are at the level of tokens created by the base BertTokenizer

"""

document = sys.argv[1]
output_name = sys.argv[2]

# Read in text, create inclusion and exclusion columns, and clean
df = pd.read_csv(str(document))
df["cohort"] = df["cohort"].apply(json.loads) 

df["inclusion"] = ["None"]*len(df)
df["exclusion"] = ["None"]*len(df)

cohort = df["cohort"]
for x in range(len(cohort)):
    df["inclusion"][x] = cohort[x]["inclusion"]
    df["exclusion"][x] = cohort[x]["exclusion"]
    

def clean_text(x):
    x = re.sub("-", "", x)
    x = re.sub("\(", " ", x)
    x = re.sub("\)", " ", x)
    return x
    
df["query"] = df["query"].apply(clean_text)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased',do_lower_case=True)

# Creating labels for each token based on exclusion and inclusion criteria
final_labels = []

for index, row in df.iterrows():
    
    tokenized_query = tokenizer.tokenize(row["query"])
    labels = ["Neither"]*len(tokenized_query)
    tokenized_inclusion = [tokenizer.tokenize(x) for x in row["inclusion"]]
    tokenized_exclusion = [tokenizer.tokenize(x) for x in row["exclusion"]]
    
    for criteria in tokenized_inclusion:
        for token in range(int(len(tokenized_query)-len(criteria))+1):
            if tokenized_query[token:token+len(criteria)] == criteria:
                labels[token:token+len(criteria)] = ["include"] * len(criteria)
                
    for criteria in tokenized_exclusion:
        for token in range(int(len(tokenized_query)-len(criteria))+1):
            if tokenized_query[token:token+len(criteria)] == criteria:
                labels[token:token+len(criteria)] = ["exclude"] * len(criteria)
    
    final_labels.append(labels)

df["labels"] = final_labels

df.to_csv("final.csv", index = False)
