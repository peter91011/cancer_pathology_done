import re
import json
import itertools
import numpy as np
import pandas as pd
import os

def get_notes(files_lst):

    notes_sents_all = dict()
    for fname in files_lst:
        with open(fname, 'r') as fp:
            notes = json.load(fp)

        for note in notes:
            note_id = note['id']
            annotations = note['annotations'][0]['result']
            sents = note['data']['dialogue']
            note_sents = list()

            for sent in sents:
                note_sents.append([sent['author'], 0, sent['text']])

            note_sents.sort(key=lambda x: x[0])
            note_sents = [[sent[1], sent[2]] for sent in note_sents]

            if annotations:
                for res in annotations:
                    start = int(res['value']['start'])
                    end = int(res['value']['end'])
                    label = res['value']['paragraphlabels'][0]
                    if label == "others":
                        continue
                    elif label == "social history":
                        for i in range(start, end + 1):
                            if note_sents[i][0] != 0 and note_sents[i][0] != 2:
                                print("there is a confilt in file: {} and id: {}".format(fname, note['id']))
                                exit()
                            note_sents[i][0] = 2
                    elif label == "family history":
                        for i in range(start, end + 1):
                            if note_sents[i][0] != 0 and note_sents[i][0] != 1:
                                print("there is a confilt in file: {} and id: {}".format(fname, note['id']))
                                exit()
                            note_sents[i][0] = 1
                    else:
                        print("there is a wrong label in file: {}, id: {}. label: {}".format(fname, note['id'], label))
                        exit()
            notes_sents_all[note_id] = note_sents

    return notes_sents_all


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



def create_label_chunks(labels):
    zeros = []
    ones = []
    twos = []
    first = labels[0]
    start_index = 0
    end_index = 0
    for index, i in enumerate(labels):
        if i == first:
            end_index = index + 1
            continue
        if first == 0:
            zeros.append(range(start_index, end_index))
        elif first == 1:
            ones.append(range(start_index, end_index))
        else:
            twos.append(range(start_index, end_index))

        start_index = index
        first = i
        end_index = index + 1

    if first == 0:
        zeros.append(range(start_index, end_index))
    elif first == 1:
        ones.append(range(start_index, end_index))
    else:
        twos.append(range(start_index, end_index))

    return [zeros, ones, twos]


def save_eval_results(input_notes, pred_files_path, file_name):
    """

    Args:
        input_notes: input notes with label in a list: [[0, 'a']. [1, 'ab']. [2, 'cd']]
        pred_files_path: list of json files with full path with the predicted labels
        file_name: file name saved
    Returns:
        full file path
    """
    dfs = []
    for lb in [1, 2]:

        note = []
        chunks = []
        label = []
        model_output = []
        number_of_sentences_in_chunk = []
        recall = []
        precision = []

        for index in range(len(pred_files_path)):

            with open(pred_files_path[index]) as f:
                jsons = json.load(f)
                f.close()
            # remove first two padding 'history'
            output_labels = itertools.chain.from_iterable([l for l in jsons])
            # remove last padding ''
            output_labels = [k for k in output_labels if k[1] != '']

            pred_label = [i[0] for i in output_labels]
            origin_label = [i[0] for i in input_notes[index]]
            origin_groups = create_label_chunks(origin_label)[lb]
            pred_groups = create_label_chunks(pred_label)[lb]

            # calculate all metrics
            for chunk_index, chunk in enumerate(origin_groups):
                related_pred_chunk = [k for k in pred_groups if len(set(chunk) - set(k)) < len(set(chunk))]
                note.append(index)
                chunks.append(chunk_index)
                label.append(str(chunk[0]) + '-' + str(chunk[-1]))

                model_output_list = [[k[0], k[-1]] for k in related_pred_chunk]

                model_output.append(
                    str(model_output_list).replace('],', ';').replace('[', '').replace(']', '').replace(', ', '-'))
                number_of_sentences_in_chunk.append(chunk[-1] - chunk[0] + 1)

                recall.append(1 - (len(set(chunk) - set(itertools.chain.from_iterable(related_pred_chunk)))) / (
                            chunk[-1] - chunk[0] + 1))
                precision.append(1 - len(set(itertools.chain.from_iterable(related_pred_chunk)) - set(chunk)) /
                                 len(set(itertools.chain.from_iterable(related_pred_chunk))) if
                                 len(set(itertools.chain.from_iterable(related_pred_chunk))) != 0 else np.nan)

        chunk_level = pd.DataFrame({'note': note, 'chunks': chunks, 'label': label,
                                    'model_output': model_output,
                                    'number_of_sentences_in_chunk': number_of_sentences_in_chunk,
                                    'recall': recall, 'precision': precision})
        wm = lambda x: np.average(x, weights=chunk_level.loc[x.index, "number_of_sentences_in_chunk"])

        # how many chunks %100 correctly predicted in this note
        chk_lvl = chunk_level.groupby('note')['recall'].apply(lambda x: x[x == 1].count()) / \
                  chunk_level.groupby('note')['recall'].apply(lambda x: x.count()).tolist()
        # how many sentences %100 correctly predicted in this note
        sen_lvl = chunk_level.groupby('note').agg(sentence_lvl=("recall", wm))
        sen_lvl['chunk_lvl'] = chk_lvl
        # how many chunks %100 correctly predicted in this note(except the chunk with only one sent)
        sen_chk_lvl = chunk_level[chunk_level['number_of_sentences_in_chunk'] != 1].groupby('note').agg(
            sentence_chunk_lvl=("recall", wm))
        note_lvl = sen_lvl.join(sen_chk_lvl).reset_index()

        # agg level
        note_level = len([k for k in chunk_level.groupby('note')['recall'].agg('min').tolist() if k == 1]) / \
                     len(chunk_level)
        chunk_lvl = len(chunk_level[chunk_level['recall'] == 1]) / len(chunk_level)
        sentence_level = (round(chunk_level['number_of_sentences_in_chunk'] * chunk_level['recall'])).sum() / \
                         sum(chunk_level['number_of_sentences_in_chunk'])
        chunk_level1 = chunk_level[chunk_level['number_of_sentences_in_chunk'] == 1]
        sentence_chunk_level = (round(chunk_level1['number_of_sentences_in_chunk'] * chunk_level1['recall'])).sum() \
                               / sum(chunk_level1['number_of_sentences_in_chunk'])
        agg_df = pd.DataFrame(
            {'note_level': [note_level], 'chunk_level': [chunk_lvl], 'sentence_level': [sentence_level],
             'sentence_chunk_level': sentence_chunk_level})
        dfs += [chunk_level, note_lvl, agg_df]

        cur_path = os.path.abspath(os.path.dirname(__file__))
        save_path = os.path.join(os.path.join(cur_path, 'outputs'), file_name+'.xlsx')

    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
    st_col = 0
    for inde, x in enumerate(dfs):
        if inde < 3:
            sheet_nm = 'Result_label1'
        else:
            sheet_nm = 'Result_label2'
        x.to_excel(writer, sheet_name=sheet_nm, startrow=1, startcol=st_col, index=False)
        st_col += (len(x.columns) + 1)
        if inde == 2:
            st_col = 0
    writer.close()
