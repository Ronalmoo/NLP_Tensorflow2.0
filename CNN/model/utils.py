import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Mecab
from gluonnlp import Vocab
from typing import Tuple


class PreProcessor:
    def __init__(self, vocab: Vocab, tokenizer: Mecab, pad_idx: int=0, pad_length: int=70) -> None:
        self._vocab = vocab
        self._pad_idx = pad_idx
        self._pad_length = pad_length
        self._tokenizer = tokenizer

    def convert2idx(self, record: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        data, label = tf.io.decode_csv(record, record_defaults=[[''], [0]], field_delim='\t')
        data = [[self._vocab.token_to_idx[token] for token in self._tokenizer.morphs(sen.numpy().decode('utf-8'))] for
                sen in data]
        data = pad_sequences(data, maxlen=self._pad_length, value=self._pad_idx,
                             padding='post', truncating='post')
        data = tf.convert_to_tensor(data, dtype=tf.int32)
        label = tf.reshape(label, (record.get_shape()[0],))
        return data, label

