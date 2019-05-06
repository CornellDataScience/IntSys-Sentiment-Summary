import gzip
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

def make_fake(sent_list):
    fake_review_sents = random.sample(sent_list, random.randint(4, min(20, len(sent_list))))
    sents, ratings = zip(*fake_review_sents)
    review = '. '.join(sents) + '.'
    rating = np.average(np.array(ratings))
    return review, rating

def make_product_data(prod, min_inter=3, k_helpful=5, k_unhelpful=5, up_smooth=1, total_smooth=3):
    prod.loc[:,'downs'] = prod.apply(lambda x: x.helpful[1] - x.helpful[0], axis=1)
    prod.loc[:,'ups'] = prod.apply(lambda x: x.helpful[0], axis=1)
    prod.loc[:, 'ratio'] = prod.apply(lambda x: (x.ups + up_smooth) / (x.inters + total_smooth), axis=1)
    most_inter = prod[prod.inters > min_inter]

    revs = len(most_inter)
    if revs < 2:
        raise AttributeError

    n_helpful = min(int(revs *.5), k_helpful)
    n_unhelpful = min(int(revs *.5), k_unhelpful)

    sorted_reviews = most_inter.sort_values(by=['ratio'])
    helpful_data = sorted_reviews[-n_helpful:].loc[:, ['reviewText', 'ratio']]
    unhelpful_data = sorted_reviews[:n_unhelpful].loc[:, ['reviewText', 'ratio']]

    n_prod_data = len(helpful_data) + len(unhelpful_data)
    total_helpfulness = helpful_data.ratio.sum() + unhelpful_data.ratio.sum()
    n_fake = int(n_prod_data/1.5)
    avg_fake_helpfulness = (.5 * (n_fake + n_prod_data) - total_helpfulness) / n_fake

    review_sents = list(map(lambda x: (x[0].split('.'), x[1]),
                        list(prod[['reviewText', 'ratio']].values)))
    sent_list = [(s.strip(), helpful_ratio) for review, helpful_ratio in review_sents
                         for s in review if len(s) > 20]
    fake_reviews = []
    fake_helpfulness = []
    for i in range(n_fake):
        review, rating = make_fake(sent_list)
        fake_reviews.append(review)
        fake_helpfulness.append(rating)

    weights = np.array(fake_helpfulness)
    labels = np.maximum(0,np.random.multivariate_normal(weights * avg_fake_helpfulness, .01*np.eye(len(weights))))

    syn_data = np.array([[rev, score] for rev, score in zip(fake_reviews, labels)])
    all_data = np.concatenate([helpful_data, unhelpful_data, syn_data])
    return all_data
        

def make_dataset(data_path, save_path):
    reviews = getDF(data_path)
    reviews.loc[:,'inters'] = reviews.apply(lambda x: x.helpful[1], axis=1)

    prods = reviews.groupby('asin')
    high_interation_prods = prods.filter(lambda x: x.inters.sum() > 40)
    prods = high_interation_prods.groupby('asin')

    prod_data = []
    for name, group in prods:
        try:
            prod_data.append(make_product_data(group))
        except AttributeError:
            continue
    all_data = np.concatenate(prod_data)
    df = pd.DataFrame(all_data, columns=['reviewText', 'label'])
    df.to_csv(save_path, index=False)


def make_dataset_mk2(data_path, save_path, up_smooth=1, total_smooth=3):
    df = getDF(data_path)
    df['inters'] = df.apply(lambda x: x.helpful[1], axis=1)
    df.loc[:,'downs'] = df.apply(lambda x: x.helpful[1] - x.helpful[0], axis=1)
    df.loc[:,'ups'] = df.apply(lambda x: x.helpful[0], axis=1)
    inter_over2 = df[df.inters > 4]
    inter_over2.loc[:, 'ratio'] = inter_over2.apply(lambda x: (x.ups) / (x.inters), axis=1)

    bad_reviews = inter_over2[inter_over2.ratio < .3]
    n_bad = len(bad_reviews)

    inter_over5 = inter_over2[inter_over2.inters > 8]
    inter_over5.loc[:, 'ratio'] = inter_over2.apply(lambda x: (x.ups + up_smooth) / (x.inters + total_smooth), axis=1)
    h_reviews = inter_over5[inter_over5.ratio > .87]
    n_helpful = len(h_reviews)

    assert n_bad < n_helpful
    n_fake = n_helpful

    pos_revs = df[df.overall >= 4]
    pos_prod_revs = pos_revs.groupby('asin').filter(lambda x: len(x) > 15).groupby('asin')

    sent_dict = {}
    for ix, group in pos_prod_revs:
        sent_dict[ix] = list(map(lambda x: (x.split('.')),list(group.reviewText.values)))

    sentlist_dict = {ix:[s.strip() for r in revs for s in r if len(s) > 30] for ix, revs in sent_dict.items()}

    sample_prods = random.choices(list(sent_dict.keys()), k=n_fake)
    fake_data = []
    for p in sample_prods:
        sent_list = sentlist_dict[p]
        fake_review_sents = random.choices(sent_list, k=random.randint(2, min(24, len(sent_list))))
        if random.random() < .01:
            fake_review_sents = random.choices(fake_review_sents, k=len(fake_review_sents))
        review = '. '.join(fake_review_sents) + '.'
        label = max(0, np.random.normal(.05, .02))
        fake_data.append((review, label))

    syn_data = np.array(fake_data)
    real_unhdata = bad_reviews[['reviewText', 'ratio']].values
    real_hdata = h_reviews[['reviewText', 'ratio']].values

    all_data = np.concatenate([real_unhdata, real_hdata, syn_data])
    df = pd.DataFrame(all_data, columns=['reviewText', 'label'])
    print(len(df))
    train, val = train_test_split(df.dropna(), test_size=.2)
    print(len(train) + len(val))
    train.to_csv(save_path + '_train.csv', index=False)
    val.to_csv(save_path + '_val.csv', index=False)


data_path = 'data/reviews_Books_5.json.gz'
save_path = 'data/books_helpfulness'
make_dataset_mk2(data_path, save_path)
