import pandas as pd
import gzip
import json

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def get_helpfulness_ratio(hlp_list, pos_smooth=1, neg_smooth=1):
    pos, total = hlp_list
    pos += pos_smooth
    total += pos_smooth + neg_smooth
    return pos/total

def make_dataset(data_path):
    reviews = getDF(data_path)

    engaged_reviews = reviews[sports.helpful.apply(lambda x: x[0]) > 5]
    non_empt = engaged_reviews[engaged_reviews.reviewText.apply(lambda x: len(x) > 10)]

    non_empt.loc[:, 'helpful'] = non_empt.loc[:, 'helpful'].apply(lambda x: get_helpfulness_ratio(x))
    dataset = non_empt[['reviewText', 'helpful']].dropna()
    dataset.to_csv('data/sports_helpfulness.csv', index=False)




data_path = 'data/reviews_Sports_and_Outdoors_5.json.gz'

make_dataset(data_path)