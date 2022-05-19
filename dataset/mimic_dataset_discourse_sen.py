from torch.utils.data import Dataset
from utils.helper_functions import *
import random
import re


class MimicDataset(Dataset):
    def __init__(self, data,
                 word2idx, # word dict
                 code2idx, # ICD code dict
                 heading2idx, # section heading to idx dict
                 desc_dic, # code description dict
                 all_headings_dic, # headings to their number of occurrences
                 max_seq_len=6000,
                 data_augmentation=False,
                 shuffle_sen=True,
                 shuffle=False,
                 num_len_cut=None,
                 ):
        self.num_len_cut = num_len_cut

        self.data = maybe_load_json(data)
        if num_len_cut is not None and num_len_cut > 1:
            section_len = len(self.data) // self.num_len_cut
            sections = []
            all_hadm_ids = list(self.data.keys())
            all_hadm_ids = sorted(all_hadm_ids, key=lambda x: int(self.data[x]['char_len']))

            for i in range(0, len(all_hadm_ids) + 1, section_len):
                section = all_hadm_ids[i: i+section_len]
                random.shuffle(section)
                sections.append(section)
            random.shuffle(sections)
            all_hadm_ids = []
            for section in sections:
                all_hadm_ids += section
            self.all_hadm_ids = all_hadm_ids
        else:
            self.all_hadm_ids = list(self.data.keys())

        self.data_augmentation = data_augmentation
        self.shuffle_sen = shuffle_sen
        if shuffle:
            self.shuffle_ids()
        self.code2idx = maybe_load_json(code2idx)
        self.idx2code = {v: k for k, v in self.code2idx.items()}
        self.desc_dic = maybe_load_json(desc_dic)
        self.word2idx = maybe_load_json(word2idx)
        self.all_headings_dic = maybe_load_json(all_headings_dic)
        self.heading2idx = maybe_load_json(heading2idx)
        self.max_seq_len = max_seq_len
        # patterns to match numbers
        self.p1 = re.compile("[0-9./:,]+|[^0-9.]+")
        self.p2 = re.compile("[0-9]+")
        self.unk_token = 1
        self.num_token = 2
        self.load_code_desc_text()

    def shuffle_ids(self, ):
        random.shuffle(self.all_hadm_ids)

    def word_tokenize(self, words, word2idx):
        lis = []
        for word in words:
            word = word.lower()
            token = word2idx.get(word)
            if token is not None:
                lis.append(token)
            else:
                for _word in re.findall(self.p1, word):
                    # if current word not contains numbers
                    if len(re.findall(self.p2, _word)) == 0:
                        lis.append(word2idx.get(_word, self.unk_token))
                    # if current words is a number
                    else:
                        lis.append(self.num_token)
        return lis

    def load_code_desc_text(self):
        code_desc_text = []
        for i in range(len(self.idx2code)):
            code = self.idx2code[i]
            words = self.desc_dic.get(code)
            if words is None:
                code_desc_text.append([self.unk_token])
            else:
                code_desc_text.append(self.word_tokenize(words, self.word2idx))
        code_desc_text = [torch.tensor(i) for i in code_desc_text]
        self.code_desc_text = torch.nn.utils.rnn.pad_sequence(code_desc_text, batch_first=True)

    def __getitem__(self, item):
        hadm_id = self.all_hadm_ids[item]
        _text = []
        _segments = []
        _sen_end_pos = []
        _sen_spans = []
        _node_end_pos = []
        _node_spans = []
        _data = self.data[hadm_id]['data']
        node_lis = []
        for node in _data:
            # avoid empty nodes
            if len(node[1]) > 0:
                temp = []
                temp.append(node[0])  # node topic
                # avoid empty sentence
                temp.append([self.word_tokenize(sen, self.word2idx) for sen in node[1] if len(sen) > 0])
                node_lis.append(temp)
        # crop
        while sum([sum([len(sen) for sen in node[1]]) for node in node_lis]) > self.max_seq_len:
            # remove least common topics until total len below limitation
            node_lis = sorted(node_lis, key=lambda node: self.all_headings_dic[node[0]])
            if len(node_lis[0][1]) == 1:
                node_lis = node_lis[1:]
            else:
                node_lis[0][1] = node_lis[0][1][:-1]
        if len(node_lis) == 0:
            return None
        # shuffle sections
        if self.data_augmentation and random.random() < 0.8:
            random.shuffle(node_lis)

        for node in node_lis:
            heading_idx = self.heading2idx.get(node[0], 0)
            # shuffle sentences in a section
            if self.shuffle_sen and self.data_augmentation and random.random() < 0.8:
                random.shuffle(node[1])

            for sen in node[1]:
                if len(sen) > 0:
                    _text += sen
                    _segments += [heading_idx] * len(sen)
                    _sen_end_pos.append(len(_text)-1)
                    _sen_spans.append(len(sen))

            _node_end_pos.append(len(_sen_end_pos)-1)
            _node_spans.append(len(node[1]))

        _text = torch.tensor(_text)
        _segments = torch.tensor(_segments)

        target_codes = self.data[hadm_id]['target_codes']
        labels_bin = np.zeros(len(self.code2idx))
        for target_code in target_codes:
            _idx = self.code2idx.get(target_code)
            # if _idx is not None:
            labels_bin[_idx] = 1

        _sen_start_pos = [0] + [i+1 for i in _sen_end_pos[:-1]]
        _node_start_pos =[0] + [i+1 for i in _node_end_pos[:-1]]



        output = {
            "text": _text,
            "segments": _segments,
            'sen_start_pos': _sen_start_pos,
            'sen_end_pos': _sen_end_pos,
            'node_start_pos': _node_start_pos,
            'node_end_pos': _node_end_pos,
            'sen_spans': _sen_spans,
            'node_spans': _node_spans,
            'labels': torch.tensor(labels_bin),
        }
        return output

    def __len__(self):
        return len(self.all_hadm_ids)

def collate_fn(lis):
    text = [dic['text'] for dic in lis if dic is not None]
    text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
    labels = torch.stack([dic['labels'] for dic in lis if dic is not None])
    segments = [dic['segments'] for dic in lis if dic is not None]
    segments = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True)

    sen_start_pos = [dic['sen_start_pos'] for dic in lis if dic is not None]
    sen_end_pos = [dic['sen_end_pos'] for dic in lis if dic is not None]
    node_start_pos = [dic['node_start_pos'] for dic in lis if dic is not None]
    node_end_pos = [dic['node_end_pos'] for dic in lis if dic is not None]
    sen_spans = [dic['sen_spans'] for dic in lis if dic is not None]
    node_spans = [dic['node_spans'] for dic in lis if dic is not None]

    return {
        "text": text,
        "segments": segments,
        'sen_start_pos': sen_start_pos,
        'sen_end_pos': sen_end_pos,
        'node_start_pos': node_start_pos,
        'node_end_pos': node_end_pos,
        'sen_spans': sen_spans,
        'node_spans': node_spans,
        "labels": labels,
    }