import numpy as np
import json
import pandas as pd
import os
import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from functools import reduce
import gensim

data_files = ['azn', 'biontech', 'jnj', 'moderna', 'novavax', 'pfizer']


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def list_major_and_related(dir):
    files = os.listdir(dir)
    major_tweets_f = []
    related_tweets_f = []
    for fn in files:
        if 'related' in fn:
            related_tweets_f.append(fn)
        else:
            major_tweets_f.append(fn)
    return major_tweets_f, related_tweets_f


def read_file(n):
    with open(n, 'rb') as f:
        fr = f.readlines()

    def decoder(s):
        try:
            ds = str(s, 'utf_8_sig')
        except UnicodeDecodeError:
            try:
                ds = str(s, 'utf-8')
            except UnicodeDecodeError:
                try:
                    ds = str(s, 'gbk')
                except UnicodeDecodeError:
                    try:
                        ds = str(s, 'GB18030')
                    except UnicodeDecodeError:
                        ds = str(s, 'gb2312', 'ignore')
        return ds

    fr = list(map(lambda x: json.loads(decoder(x)), fr))
    return fr


def process_file(file_dir: str,
                 company_name: str,
                 abandon: list = None) -> pd.DataFrame:
    """
    Processing the related twitters regarding company_name
    :param file_dir: file directory
    :param company_name: target company name
    :param abandon: the columns to abandon
    :return: a pandas dataframe of cleaned data
    """
    if abandon is None:
        abandon = ['conversation_id', 'created_at', 'time', 'timezone', 'user_id',
                   'username', 'name', 'place', 'mentions', 'link', 'quote_url',
                   'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
                   'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src', 'trans_dest']
    op = read_file(file_dir)
    d = pd.DataFrame(op)
    d = d.drop_duplicates(subset=['id'])
    d = d.drop(columns=abandon)
    d = d[d['language'] == 'en'].reset_index(drop=True)
    d = d.drop(columns=['language'])
    # process date
    d['date'] = d['date'].apply(lambda x: x.replace('-', ''))
    # process urls:
    d['linked_url'] = d['urls'].apply(lambda x: 1 if x else 0)
    d = d.drop(columns=['urls'])
    # process photos:
    d['with_photo'] = d['photos'].apply(lambda x: 1 if x else 0)
    d = d.drop(columns=['photos'])
    # process retweet
    d['retweet'] = d['retweet'].apply(lambda x: 1 if x else 0)
    # process video
    d['video'] = d['video'].apply(lambda x: 1 if x else 0)
    # process date:
    d['date'] = d['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    d['date'] = d['date'].apply(
        lambda x: x + datetime.timedelta(days=-1) if x.weekday() == 5 else x + datetime.timedelta(
            days=-2) if x.weekday() == 6 else x)
    d['date'] = d['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y%m%d'))
    # process format:
    d['replies_count'] = d['replies_count'].astype('int64')
    d['retweets_count'] = d['retweets_count'].astype('int64')
    d['likes_count'] = d['likes_count'].astype('int64')
    d['retweet'] = d['retweet'].astype('int64')
    d['video'] = d['video'].astype('int64')
    d['linked_url'] = d['linked_url'].astype('int64')
    d['with_photo'] = d['with_photo'].astype('int64')
    # group by date
    new_df = [dict(zip(sub_df.columns, [sum(sub_df[x])
                                        if sub_df[x].dtype == 'int64'
                                        else sub_df[x].tolist()
                                        for x in sub_df.columns]))
              for _, sub_df in d.groupby('date')]
    new_df = pd.DataFrame(new_df)
    new_df['company_name'] = company_name
    new_df['date'] = new_df['date'].apply(lambda x: x[0])
    return new_df


# process json file for each company
def gather_all(dfs: list = None) -> pd.DataFrame:
    if dfs is None:
        dfs = data_files
    corp_gathered = []
    for f in dfs:
        mtf, rtf = list_major_and_related('./{}'.format(f))
        print('...Processing file {}...{}'.format(f, rtf))
        corp_file = []
        for j in rtf:
            j_d = './{}/'.format(f) + j
            corp_file.append(process_file(file_dir=j_d,
                                          company_name=f))
        corp_file = reduce(lambda x, y: pd.concat([x, y], ignore_index=True), corp_file)
        corp_gathered.append(corp_file)
    corp_gathered = reduce(lambda x, y: pd.concat([x, y], ignore_index=True), corp_gathered)
    corp_gathered = corp_gathered.drop_duplicates(subset=['date', 'company_name'], keep='first')
    return corp_gathered


def cum_sum(target):
    s = sum(target)
    r = list(range(s))
    res = []
    for i in target:
        tmp = r[0:i]
        r = r[i:]
        res.append(tmp)
    return res


class ProcessR(object):
    def __init__(self,
                 text,
                 vs=100,
                 ws=5,
                 mc=2,
                 wks=4):
        self.text = text
        print('...{} Documents Loaded...'.format(len(self.text)))
        self.docs = self.preprocess_docs()
        self.tagged_docs = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(self.docs)]
        self.model = self.d2v(vs, ws, mc, wks)
        self.doc_vs = self.doc_v()
        print('...Language Processing and Language Model Training Finished...')

    def preprocess_docs(self):
        # Including steps: removing punctuations,
        #                  removing duplicated white spaces,
        #                  remove numbers,
        #                  remove stopwords,
        #                  stemming the text.
        return gensim.parsing.preprocessing.preprocess_documents(self.text)

    def d2v(self, vs, ws, mc, wks):
        # reference: https://arxiv.org/pdf/1405.4053v2.pdf <Distributed Representations of Sentences and Documents>
        print('...Training Doc2Vector Model via Gensim.models.Doc2Vec...')
        model = gensim.models.doc2vec.Doc2Vec(self.tagged_docs,
                                              vector_size=vs,
                                              window=ws,
                                              min_count=mc,
                                              workers=wks,
                                              compute_loss=True)
        return model

    def doc_v(self):
        return [self.model[x[1][0]].tolist() for x in self.tagged_docs]

    def process_text(self, s):
        return gensim.parsing.preprocessing.preprocess_string(s)

    def infer(self, dv):
        return self.model.infer_vector(dv)


def generate_finalized_data(pr, ref_d, embedding_size=100):
    res = np.zeros((len(ref_d), embedding_size))
    for idx, key in enumerate(ref_d):
        sub_doc_vs = np.array([pr.doc_vs[x] for x in ref_d[key]])
        res[idx, :] = np.mean(sub_doc_vs, axis=0)
    return res


def df_split(df,
             drop_columns=None,
             test_size=0.2,
             norm=True,
             norm_columns=None,
             norm_approach='min-max',
             predict_time=1,
             window=20):
    df = related_f_m
    if drop_columns is None:
        drop_columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume', 'company_name', 'date',
                        'hashtags', 'cashtags']
    df = df.drop(columns=drop_columns)
    df = df.dropna()
    if norm:
        if norm_columns is None:
            norm_columns = ['replies_count', 'retweets_count',
                            'likes_count', 'retweet',
                            'video', 'linked_url', 'with_photo', 'target_variation']
        if norm_approach == 'min-max':
            s = MinMaxScaler()
        else:
            s = StandardScaler()
        s.fit(df[norm_columns])
        s_f = np.array(s.transform(df[norm_columns]))
    else:
        s_f = np.array(df)
    y = s_f[predict_time:, -1]
    x_remain = np.array(df.drop(columns=norm_columns))[:-predict_time, :]
    x = np.concatenate([x_remain, s_f[:-predict_time, :-1]], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train = [x_train[i:i+window, :].tolist() for i in range(x_train.shape[0] - window + 1)]
    x_test = [x_test[i:i+window, :].tolist() for i in range(x_test.shape[0] - window + 1)]
    y_train = [y_train[i:i+window].tolist() for i in range(y_train.shape[0] - window + 1)]
    y_test = [y_test[i:i+window].tolist() for i in range(y_test.shape[0] - window + 1)]
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    related_corp_tweets = pd.read_csv('data_op.csv', encoding='utf_8_sig')
    major_corp_tweets = pd.read_csv('tweetcompany.csv', encoding='utf_8_sig', index_col=0)
    stock_price = pd.read_csv('stockprice.csv', encoding='utf_8_sig', index_col=0)

    related_corp_tweets['date'] = related_corp_tweets['date'].astype('str')
    major_corp_tweets['date'] = major_corp_tweets['date'].astype('str')
    stock_price['date'] = stock_price['date'].astype('str')
    # mapping date to tweet
    date_text_related = dict(zip(related_corp_tweets.date + '_' + related_corp_tweets.company_name,
                                 cum_sum([len(eval(x)) for x in related_corp_tweets.tweet.tolist()])))
    date_text_major = dict(zip(major_corp_tweets.date + '_' + major_corp_tweets.company_name,
                               cum_sum([len(eval(x)) for x in major_corp_tweets.tweet.tolist()])))

    # format the texts
    text_related = sum([eval(x) for x in related_corp_tweets.tweet.tolist()], [])
    text_major = sum([eval(x) for x in major_corp_tweets.tweet.tolist()], [])

    related_pr = ProcessR(text_related)
    major_pr = ProcessR(text_major)

    related_f = generate_finalized_data(related_pr, date_text_related)
    major_f = generate_finalized_data(major_pr, date_text_major)

    related_f = pd.concat([pd.DataFrame(related_f), related_corp_tweets[['date', 'replies_count', 'retweets_count',
                                                                         'likes_count', 'hashtags', 'cashtags',
                                                                         'retweet',
                                                                         'video', 'linked_url', 'with_photo',
                                                                         'company_name']]], axis=1, ignore_index=False)
    major_f = pd.concat([pd.DataFrame(major_f), major_corp_tweets[['date', 'replies_count', 'retweets_count',
                                                                   'likes_count', 'hashtags', 'cashtags', 'retweet',
                                                                   'video', 'linked_url', 'with_photo',
                                                                   'company_name']]], axis=1, ignore_index=False)
    # merge
    related_f_m = pd.merge(related_f, stock_price, on=['company_name', 'date'], how='right')
    major_f_m = pd.merge(major_f, stock_price, on=['company_name', 'date'], how='right')

    xtr, xte, ytr, yte = df_split(related_f_m)
    save_obj(dict(zip(['x_train', 'x_test', 'y_train', 'y_test'], [xtr, xte, ytr, yte])), 'data_final')
    print('File Saved to local (working directory).')