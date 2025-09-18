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
from cade.cade import CADE

from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_multiple_whitespaces, strip_numeric
from gensim.parsing.preprocessing import preprocess_string, preprocess_documents

def main():
    c=0
    nltk.download('punkt_tab')
    data_dict = {'year': [], 'text': []}
    with open('/app/src/ChronoBerg/cade/pg_books_historic.jsonl', 'r', encoding='utf-8') as dataset_in:
        for line in tqdm(dataset_in):
            file = json.loads(line)
            text = file['text']
            text = text.replace('\n', ' ')
            # Remove all kinds of quotation marks
            text = re.sub(r'[\'\"]', '', text)
            text = re.sub('\.', ' ', text)
            text = re.sub(r'[\x80-\xFF]', '', text)
            text = re.sub(r'\d+','', text) 
            # Reduce all consecutive whitespace to a single whitespace
            text = re.sub(r'\s+', ' ', text)
            file['text'] = text
            data_dict['text'].append(file['text'])
            data_dict['year'].append(file['year'])


    print('The length of the dictionary',len(data_dict['text']))
    df = pd.DataFrame(data_dict)
    df.head()
    df = df.sort_values(by=['year'], ascending=True)

    data_dict = {'year': [], 'text': []}
    for i in range(len(df)):
        data_dict['year'].append(df['year'].iloc[i])
        data_dict['text'].append(df['text'].iloc[i])


    total_data = data_dict['text'][:49] + data_dict['text'][50:99] + data_dict['text'][100:149] + data_dict['text'][150:199] + data_dict['text'][200:249] 

    text_t = []
    for text in tqdm(total_data):
        sentence = sent_tokenize(text)
        text_t.append(sentence)
    
    print(len(sentence))

    text_data_one = data_dict['text'][:49]
    #print(len(text_data))
    #text_data[0]
    text_one = []
    for text in tqdm(text_data_one):
        sentence = sent_tokenize(text)
        text_one.append(sentence)
        #print(len(sentence))

    text_data_sec = data_dict['text'][50:99]
    #print(len(text_data))
    #text_data[0]
    text_sec = []
    for text in tqdm(text_data_sec):
        sentence = sent_tokenize(text)
        text_sec.append(sentence)
        #print(len(sentence))

    text_data_trd = data_dict['text'][100:149]
    #print(len(text_data))
    #text_data[0]
    text_trd = []
    for text in tqdm(text_data_trd):
        sentence = sent_tokenize(text)
        text_trd.append(sentence)
        #print(len(sentence))

    text_data_four = data_dict['text'][150:199]
    #print(len(text_data))
    #text_data[0]
    text_four = []
    for text in tqdm(text_data_four):
        sentence = sent_tokenize(text)
        text_four.append(sentence)

    text_data_five = data_dict['text'][200:249] 
    #print(len(text_data))
    #text_data[0]
    text_five = []
    for text in tqdm(text_data_five):
        sentence = sent_tokenize(text)
        text_five.append(sentence)



    text_t = list(itertools.chain(*text_t))
    text_one = list(itertools.chain(*text_one))
    text_sec = list(itertools.chain(*text_sec))
    text_trd = list(itertools.chain(*text_trd))
    text_four = list(itertools.chain(*text_four))
    text_five = list(itertools.chain(*text_five))
    print('One size',len(text_one))
    print('second size',len(text_sec))  
    print('third size',len(text_trd))  
    print('fourth size',len(text_four))
    print('fourth size',len(text_five))



    custom = [lambda x: x.lower(), remove_stopwords, strip_punctuation, strip_numeric]
    for i in tqdm(range(len(text_t))):
        text_t[i] = strip_punctuation(text_t[i])
        text_t[i] = remove_stopwords(text_t[i])
        text_t[i] = text_t[i].lower()


    for i in tqdm(range(len(text_one))):
        text_one[i] = strip_punctuation(text_one[i])
        text_one[i] = remove_stopwords(text_one[i])
        text_one[i] = text_one[i].lower()

    for i in tqdm(range(len(text_sec))):
        text_sec[i] = strip_punctuation(text_sec[i])
        text_sec[i] = remove_stopwords(text_sec[i])
        text_sec[i] = text_sec[i].lower()


    for i in tqdm(range(len(text_trd))):
        text_trd[i] = strip_punctuation(text_trd[i])
        text_trd[i] = remove_stopwords(text_trd[i])
        text_trd[i] = text_trd[i].lower()

    for i in tqdm(range(len(text_four))):
        text_four[i] = strip_punctuation(text_four[i])
        text_four[i] = remove_stopwords(text_four[i])
        text_four[i] = text_four[i].lower()

    for i in tqdm(range(len(text_five))):
        text_five[i] = strip_punctuation(text_five[i])
        text_five[i] = remove_stopwords(text_five[i])
        text_five[i] = text_five[i].lower()

    with open('/app/src/ChronoBerg/cade/text_t.txt', 'a', encoding='utf-8') as file:
        for line in text_t:
            file.write(line + '. ')

    with open('/app/src/ChronoBerg/cade/text_one.txt', 'a', encoding='utf-8') as file:
        for line in text_one:
            file.write(line + '. ')

    with open('/app/src/ChronoBerg/cade/text_sec.txt', 'a', encoding='utf-8') as file:
        for line in text_sec:
            file.write(line + '. ')

    with open('/app/src/ChronoBerg/cade/text_trd.txt', 'a', encoding='utf-8') as file:
        for line in text_trd:
            file.write(line + '. ')
    
    with open('/app/src/ChronoBerg/cade/text_four.txt', 'a', encoding='utf-8') as file:
        for line in text_four:
            file.write(line + '. ')
    with open('/app/src/ChronoBerg/cade/text_five.txt', 'a', encoding='utf-8') as file:
        for line in text_five:
            file.write(line + '. ')

    #os.chdir('/app/src/ChronoBerg/cade/')
    

    aliger = CADE(size= 300 , window= 10, min_count= 1, workers= 6, siter= 10)
    aliger.train_compass('/app/src/ChronoBerg/cade/text_t.txt', overwrite= False, save= True) 

    slice_one = aliger.train_slice('/app/src/ChronoBerg/cade/text_one.txt', save=True)
    slice_two = aliger.train_slice('/app/src/ChronoBerg/cade/text_sec.txt', save=True)
    slice_three = aliger.train_slice('/app/src/ChronoBerg/cade/text_trd.txt', save=True)
    slice_four = aliger.train_slice('/app/src/ChronoBerg/cade/text_four.txt', save=True)
    slice_five = aliger.train_slice('/app/src/ChronoBerg/cade/text_five.txt', save=True)


if __name__ == '__main__':
    main()