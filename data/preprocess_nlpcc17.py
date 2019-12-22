from const import dirpath, EOS_token, SOS_token
r'''
该程序专门处理从
'''
from typing import List
import json

import pandas as pd
import pkuseg
from tqdm import tqdm
tqdm.pandas(desc='Progress')


fpath = dirpath + 'raw_data/train_with_summ.txt'
version = '{cut_way}{ignore_paragraph}'.format(
    cut_way='pkuseq', ignore_paragraph='T')
tpath = dirpath + '{}_'.format('nlpcc17')
xy_path = tpath+'XY_{}.json'.format(version)
x2id_path = tpath+'ix2id_{}.json'.format(version)
id2x_path = tpath+'id2ix_{}.json'.format(version)

seg = pkuseg.pkuseg()


def load():
    """
    summarization article index
    """
    data = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
            # break
    df = pd.DataFrame.from_records(data)

    return df


def getdict(docs: List):
    """"""
    x_to_id = {'<EOS>': EOS_token, '<SOS>': SOS_token}
    for d in docs:
        for w in d:
            if w not in x_to_id:
                x_to_id[w] = len(x_to_id)
    id_to_x = {i: x for x, i in x_to_id.items()}
    return x_to_id, id_to_x


def preprocess(df):
    """
    这里分割方式是基于词，使用pkuseg
    """
    # 这里忽略换行符
    df['article'] = df['article'].progress_apply(
        lambda x: x.replace('<paragraph>', '。'))

    df['article'] = df['article'].progress_apply(lambda x: seg.cut(x))
    df['summarization'] = df['summarization'].progress_apply(
        lambda x: seg.cut(x))

    # 词表
    x_to_id, id_to_x = getdict(df['article'].values+df['summarization'].values)

    # 这里在末尾加入 EOS_token
    df['X'] = df['article'].progress_apply(
        lambda X: [x_to_id[x] for x in X]+[EOS_token])
    df['Y'] = df['summarization'].progress_apply(
        lambda X: [x_to_id[x] for x in X]+[EOS_token])
    df['index'] = df.index
    df.drop(['article', 'summarization'], axis=1, inplace=True)

    return df, x_to_id, id_to_x


def dump(df, x_to_id, id_to_x):
    """"""
    # df.to_csv(xy_path, index=False, encoding='utf-8')  # encoding='utf-8-sig'
    df.to_json(xy_path, orient='records')

    with open(x2id_path, 'w', encoding='utf-8') as f:
        json.dump(x_to_id, f)
    with open(id2x_path, 'w', encoding='utf-8') as f:
        json.dump(id_to_x, f)

    #df = pd.read_json(xy_path, orient='records')


def main():
    df = load()
    df, x_to_id, id_to_x = preprocess(df)
    dump(df, x_to_id, id_to_x)


if __name__ == '__main__':
    main()
