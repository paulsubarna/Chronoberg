import json, re
import os
import nltk
import pandas as pd 
from nltk.tokenize import sent_tokenize
from nltk.tokenize import sent_tokenize
import itertools 
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
import gensim 
from IPython.display import display, Markdown
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_multiple_whitespaces, strip_numeric
from gensim.parsing.preprocessing import preprocess_string, preprocess_documents

def preprocess_text(texts):
    for i in range(len(texts)):
        text = texts[i]
        text = re.sub(r'[\'\"]', '', text)
        text = re.sub('\.', ' ', text)
        text = re.sub(r'[\x80-\xFF]', '', text)
        text = re.sub(r'\d+','', text) 
        # Reduce all consecutive whitespace to a single whitespace
        text = re.sub(r'\s+', ' ', text)
        texts[i] = text
    return texts

def extract_text_by_year(year: list, data_dict: dict) -> list:
    texts = []

    if len(year) > 1:
        for y in year:
            texts.append(data_dict['text'][int(y-1750)])
    else:
        texts.append(data_dict['text'][year[0]-1750])
    return  texts

def extract_sentence_splits_by_year(year: list, data_dict: dict) -> list:
    sents = []
    texts = extract_text_by_year(year, data_dict)
    for text in texts:
        sents.extend(sent_tokenize(text))

    print(f'Extracted {len(sents)} sentences from the years {year}')
    return sents, texts

def load_data(data_path: str) -> dict:
    data_dict = {'year': [], 'text': []}
    with open(data_path, 'r', encoding='utf-8') as dataset_in:
        for line in tqdm(dataset_in):
            file = json.loads(line)
            text = file['text']
            text = text.replace('\n', ' ')
            # Remove all kinds of quotation marks
            file['text'] = text
            data_dict['text'].append(file['text'])
            data_dict['year'].append(file['year'])

    print('data loaded')

    df = pd.DataFrame(data_dict)
    df.head()
    df = df.sort_values(by=['year'], ascending=True)

    ### Recreate the data_dict based on the sorted dataframe
    data_dict = {'year': [], 'text': []}
    for i in range(len(df)):
        data_dict['year'].append(df['year'].iloc[i])
        data_dict['text'].append(df['text'].iloc[i]) 

    print('data sorted')

    return data_dict