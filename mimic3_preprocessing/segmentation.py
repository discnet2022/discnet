import pandas as pd
from tqdm import tqdm
import re
import numpy as np
from spacy.lang.en import English
import spacy
import pickle
import json


"""
A script for word level, sentence level and discourse level segmentation
"""


class Substitution:
    def __init__(self):
        self.p_space = re.compile(u'\\s+')
        self.p_anonymized = re.compile(u'\[\*\*.*?\*\*\]')

    def __call__(self, text):
        text = re.sub(self.p_space, ' ', text.strip())
        text = re.sub(self.p_anonymized, 'anonymize', text)
        return text


class Paragraph_tokenize:
    def __init__(self, viable_groups_dic):
        self.viable_groups_dic = viable_groups_dic
        # self.pattern = re.compile(".*:\n")
        self.pattern = re.compile("(.*:\n|\n[^a-z]*:)")

    def __call__(self, text):
        paragraphs = []
        spans = []
        _len = 0
        prev_end = 0
        for match in self.pattern.finditer(text):
            span, group = match.span(), match.group()
            group = group.lower().strip()
            spans.append([span, group])

        for i in range(len(spans)):
            span, group = spans[i]
            if not paragraphs:
                para = text[:span[0]]
                _len += len(para)
                paragraphs.append(['head_info', para])
                prev_end = span[0]
            if self.viable_groups_dic.get(group) is not None:
                if i == len(spans) - 1:
                    para = text[prev_end:]
                    _len += len(para)
                    paragraphs.append([group, para])
                else:
                    para = text[prev_end: spans[i + 1][0][0]]
                    _len += len(para)
                    paragraphs.append([group, para])
                    prev_end = spans[i + 1][0][0]
            else:
                if i == len(spans) - 1:
                    para = text[prev_end:]
                    _len += len(para)
                    paragraphs[-1][1] += para
                else:
                    para = text[prev_end: spans[i + 1][0][0]]
                    _len += len(para)
                    paragraphs[-1][1] += para
                    prev_end = spans[i + 1][0][0]

        if _len < len(text):
            paragraphs.append(['tail_info', text[prev_end:]])
        return paragraphs


class PreprocessNotes:
    def __init__(self, groups_dic):
        self.nlp = spacy.load("en_core_web_sm")
        self.paragraph_tokenize = Paragraph_tokenize(groups_dic)
        self.substitute = Substitution()

    def word_tokenize(self, nlp, text):
        return [word.lemma_ for word in nlp(str(text))]

    def sentence_tokenize(self, nlp, text):
        return [i for i in nlp(text).sents if len(str(i).strip()) > 0]

    def rearrange_and_tokenize_sentences(self, nlp, sen_lis):
        return [self.word_tokenize(nlp, sen) for sen in sen_lis]

    def tokenizer_complex(self, para):
        para = self.substitute(para)
        para = self.sentence_tokenize(self.nlp, para)
        return self.rearrange_and_tokenize_sentences(self.nlp, para)

    def __call__(self, text):
        text = text.lower()
        paragraph_lis = self.paragraph_tokenize(text)
        return [[k, self.tokenizer_complex(v)] for k, v in paragraph_lis]


if __name__ == '__main__':
    dtype = {
        "ROW_ID": int,
        "SUBJECT_ID": int,
        "HADM_ID": "O",
        "CATEGORY": str,
        "DESCRIPTION": str,
        "CGID": "O",
        "ISERROR": str,
        "TEXT": str,
    }
    parse_dates = ["CHARTDATE", "CHARTTIME", "STORETIME"]
    df_notes = pd.read_csv('mimic3/NOTEEVENTS.csv',
                           parse_dates=parse_dates,
                           infer_datetime_format=True,
                           dtype=dtype,
                           )
    print('csv has been loaded')
    df_disc = df_notes.loc[df_notes["CATEGORY"] == 'Discharge summary']
    all_text = df_disc['TEXT'].tolist()

    p = re.compile(".*:\n")
    all_groups = []
    for text in tqdm(all_text):
        for m in p.finditer(text):
            all_groups.append(m.group().lower().strip())
    all_groups, counts = np.unique(all_groups, return_counts=True)
    groups_dic = {}
    for g, c in tqdm(zip(all_groups, counts)):
        if c > 5:
            g = g.strip()
            if re.match('^[\d|#]|\(.*\)', g) is None:
                groups_dic[g] = int(c)

    # with open('groups_dic_v3.json', 'w+', encoding='utf-8') as f:
    #     f.write(json.dumps(groups_dic, ensure_ascii=False))

    preprocess = PreprocessNotes(groups_dic)

    dtype = {
        "ROW_ID": int,
        "SUBJECT_ID": int,
        "HADM_ID": int,
        "SEQ_NUM": "O",
        "ICD9_CODE": str,
    }
    df_diag = pd.read_csv('mimic3/DIAGNOSES_ICD.csv', dtype=dtype)
    df_proc = pd.read_csv('mimic3/PROCEDURES_ICD.csv', dtype=dtype)
    df_diag_grouped = df_diag.groupby('HADM_ID')
    df_proc_grouped = df_proc.groupby('HADM_ID')
    labels_dic = {}
    for idx, _df in df_diag_grouped.__iter__():
        labels_dic[idx] = {'diag': _df['ICD9_CODE'].tolist()}
    for idx, _df in df_proc_grouped.__iter__():
        labels_dic[idx].update({'proc': _df['ICD9_CODE'].tolist()})

    df_disc_grouped = df_disc.groupby('HADM_ID')
    all_hadm_id = list(df_disc_grouped.groups.keys())


    data_dic = {}
    num_p = 64
    pool = Pool(num_p)
    for i in tqdm(range(0, len(all_hadm_id), num_p)):
        hadm_ids_lis = all_hadm_id[i: i + num_p]
        # text_lis = pool.map(concat_text, hadm_ids_lis)

        text_lis = []
        for hadm_id in hadm_ids_lis:
            _df = df_disc_grouped.get_group(hadm_id)
            text = '\n\n'.join(_df['TEXT'].tolist())
            text_lis.append(text)

        data_lis = pool.map(preprocess, text_lis)
        for hadm_id, text, data in zip(hadm_ids_lis, text_lis, data_lis):
            data_dic[hadm_id] = {
                'labels': labels_dic[int(hadm_id)],
                'text': text,
                'data': data}
            if len(data_dic[hadm_id]['data']) == 0:
                print('ERROR! ', hadm_id)

    with open('data_v3.pickle', 'wb') as f:
        f.write(pickle.dumps(data_dic))

    with open('data_v3.json', 'w+') as f:
        f.write(json.dumps(data_dic, ensure_ascii=False))
