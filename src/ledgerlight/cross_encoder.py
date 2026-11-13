from sentence_transformers import CrossEncoder

from ledgerlight.pairs import torch_pair_scores


class PolicyCrossEncoder(CrossEncoder):
    def predict(self, sentences, batch_size=32, **kwargs):
        return torch_pair_scores(self, sentences, batch_size=batch_size)
