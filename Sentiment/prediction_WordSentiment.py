#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 18:17:04 2021

@author: winston
"""

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import os, sys
from pathlib import Path
from scipy.special import softmax
import csv
import urllib.request
import json
from tqdm import tqdm
"""
Paper source:
    "TWEETEVAL: Unified Benchmark and Comparative Evaluation for Tweet Classification"
    link: https://arxiv.org/pdf/2010.12421.pdf
    NOTE: this script needs to run under wav2vec conda env!!
"""

def load_affective_words_dict():
    affective_words_dict = {}
    with open(os.path.join(file_dir, 'senti-words-dict', 'positive-words.txt')) as f:
        for line in f:
            line = line.strip()
            if ';' in line:
                pass
            else:
                affective_words_dict[line]='positive'
    with open(os.path.join(file_dir, 'senti-words-dict', 'negative-words.txt')) as f:
        for line in f:
            line = line.strip()
            if ';' in line:
                pass
            else:
                affective_words_dict[line]='negative' 
    return affective_words_dict

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks: (we only use emotion & sentiment for now)
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary


task='sentiment'  # 'positive', 'neutral', 'negative'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# loading the pretrained model
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)
# tokenizer.save_pretrained(MODEL)


#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[0], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args


args = get_args()
root = args.root
output_location = os.path.join(root, "Outputs")

with open(os.path.join(os.path.join(output_location), "Short_files.json"), "r") as openfile:
    audio_files = json.load(openfile)
    
print("Word sentiment inference: ")
for filename, f_info in tqdm(audio_files.items()):
    if ('text' not in f_info) or f_info['text'] == '':
        continue
    f_text = f_info['text']
    f_text = preprocess(f_text)
    encoded_input = tokenizer(f_text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().tolist()
    scores = softmax(scores)
    f_info['word_sentiment_pred'] = {}
    f_info['word_sentiment_pred']['negative'] = [scores[0],'Not_Used']
    f_info['word_sentiment_pred']['neutral'] = [scores[1],'Not_Used']
    f_info['word_sentiment_pred']['positive'] = [scores[2],'Not_Used']
    
print("Writing json file")
json_object = json.dumps(audio_files, indent=4)
with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
    outfile.write(json_object)  