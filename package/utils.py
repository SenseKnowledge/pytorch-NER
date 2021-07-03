# -*- coding: utf-8 -*-
import json


def read_clue_json(path):
    """Read json file in CLUE dataset
    """
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            text = line['text']
            label_entities = line.get('label', None)
            label = ['O'] * len(text)
            if label_entities is not None:
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            assert text[start_index:end_index + 1] == sub_name
                            label[start_index] = 'B-' + key
                            label[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
            lines.append({"text": text, "label": label})
    return lines


def decode_bio_tags(tags):
    """decode entity (type, start, end) from BIO style tags
    """
    chunks = []
    chunk = [-1, -1, -1]
    for i, tag in enumerate(tags):

        if tag.startswith('B-'):
            if chunk[2] != -1:
                chunks.append(chunk)

            chunk = [-1, -1, -1]
            chunk[0] = tag.split('-')[1]
            chunk[1] = i
            chunk[2] = i + 1
            if i == len(tags) - 1:
                chunks.append(chunk)

        elif tag.startswith('I-') and chunk[1] != -1:
            t = tag.split('-')[1]
            if t == chunk[0]:
                chunk[2] = i + 1

            if i == len(tags) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]

    return chunks
