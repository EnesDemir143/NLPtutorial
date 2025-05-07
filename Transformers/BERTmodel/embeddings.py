import torch
import torch.nn as nn


class BertEmbeding(nn.Module):
    def __init__(self, vocab_size, dim, max_sentences_word, segment_size=2):
        super(BertEmbeding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_sentences_word, dim)
        self.segment_embedding = nn.Embedding(segment_size, dim)
        self.layerNorm = nn.LayerNorm(dim, eps=1e-12)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.shape[1]
        possition_ids = torch.arange(seq_length, dtype=torch.long)
        possition_ids = possition_ids.unsqueeze(0).expand(input_ids.shape[0], seq_length)

        word_embedding = self.token_embedding(input_ids)
        position_embedding = self.position_embedding(possition_ids)
        segment_embedding = self.segment_embedding(token_type_ids) if token_type_ids is not None else 0
        #token_type_ids = [0, 0, 0, 0, 1, 1, 1, 1]   Cümle A'nın token'ları için 0, Cümle B'nin token'ları için 1

        embeddings = word_embedding + position_embedding + segment_embedding
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings