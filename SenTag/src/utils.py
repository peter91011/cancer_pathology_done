import re
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np


def remove_tags(text):
    text = text.replace('[CLS]', ' ').replace('[PAD]', ' ').replace('[SEP]', ' ')
    text = re.sub(' +', ' ', text)
    return text.strip()

def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]

        if tag == '[SEP]' or tag == '[CLS]':
            continue

        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def extract_text_from_bert_padding(text):
    m = re.search(r"\[CLS\](.*)\[SEP\]", text)
    text = text[m.start(): m.end()][6:-6]
    return text


def get_entities(seq, id2label, markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)

def dedup_train_test(df):

    df = df.reset_index().drop_duplicates()

    def check_process(note):
        if note.count('<start>') != note.count('<end>'):
            return [note.count('<start>'), note.count('<end>')]
        else:
            return True

    df['check'] = df['note'].map(check_process)
    df = df[(df['pathology_alone'].isin(['Yes', 'No'])) & (df['check'] == True)].reset_index().drop('level_0', axis=1)
    df['note'] = df['note'].map(lambda x: [k for k in re.split('<start>|<end>', x)])

    dfs = []
    for i in range(len(df)):
        if len(df['note'][i]) > 1:
            df_expand = df.iloc[i:i + 1, :].explode('note')
            label1_df = df_expand.iloc[1::2, :].copy().drop_duplicates()
            label0_df = df_expand.iloc[0::2, :].copy().drop_duplicates()
            label1_df['pathology_alone'] = 'Yes'
            label0_df['pathology_alone'] = 'No'
            label1_df['note_check'] = label1_df['note'].map(lambda x: re.sub(r'[\W_]+', '', x))
            label0_df['note_check'] = label0_df['note'].map(lambda x: re.sub(r'[\W_]+', '', x))
            label1_df = label1_df[label1_df['note_check'] != '']
            label0_df = label0_df[label0_df['note_check'] != '']

            dfs.append(label1_df)
            dfs.append(label0_df)
        else:
            dfs.append(df.iloc[i:i + 1, :].explode('note').drop_duplicates())
    newdf = pd.concat(dfs).reset_index()[['id', 'patient_id', 'note', 'pathology_alone']]

    df1 = newdf[newdf['pathology_alone'] == 'Yes'].copy()
    df0 = newdf[newdf['pathology_alone'] == 'No'].copy()
    df0['note'] = df0['note'].map(lambda x: sent_tokenize(x))
    df_out = pd.concat([df0.explode('note').drop_duplicates(), df1], ignore_index=True).sort_values('id')
    df_out['note'] = df_out['note'].map(lambda x: x.strip())

    df_out['label'] = df_out['pathology_alone'].map(lambda x: 1 if x == 'Yes' else 0)

    return df_out

def create_train_test(df):
    df = df.drop_duplicates(subset=['note', 'pathology_alone'])
    df = df[df.groupby(['note'])['label'].transform(max) == df['label']]
    xx = df.groupby('note').filter(lambda g: len(g) > 1).drop_duplicates(subset=['note', 'pathology_alone'],
                                                                         keep="first")
    df = df[~df['note'].isin(xx['note'])]
    df['label'] = df['pathology_alone'].map(lambda x: 1 if x == 'Yes' else 0)
    df1 = df[df['label'] == 1]

    train_test = df
    train, test = train_test_split(train_test, test_size=0.2)

    return train,test