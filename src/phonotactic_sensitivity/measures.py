import torch
import pickle

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform


class EmbeddingSimilarity:
    def __init__(self, distance_metric='cosine'):
        self.distance_metric = distance_metric
        self._measure_dicts = None

    def compute(
        self, step_embeddings, refA_name: str = 'L', refB_name: str = 'R', refA_idx: int = 0, refB_idx: int = -1
    ):
        layers = list(step_embeddings.keys())
        distance_matrices = [squareform(pdist(step_embeddings[l], metric=self.distance_metric)) for l in layers]
        measure_dicts = []
        for l, dist_mat in zip(layers, distance_matrices):
            A_dists = dist_mat[0]
            B_dists = dist_mat[-1]
            step_sims_A = 1 - (A_dists / (A_dists + B_dists))
            step_sims_B = 1 - (B_dists / (A_dists + B_dists))
            N_scores = len(step_sims_A) + len(step_sims_B)
            measure_dict = {
                'measure': ['embedding_similarity'] * N_scores,
                'layer': [l] * N_scores,
                'token': [refA_name] * len(step_sims_A) + [refB_name] * len(step_sims_B),
                'step': list(range(len(step_sims_A))) + list(range(len(step_sims_B))),
                'score': np.concatenate([step_sims_A, step_sims_B]),
            }
            measure_dicts.append(measure_dict)
        self._measure_dicts = measure_dicts
        return self

    def to_df(self):
        assert self._measure_dicts != None, "no computed similarities found; use .compute() before using .to_df()"
        return pd.concat([pd.DataFrame.from_dict(measure_dict) for measure_dict in self._measure_dicts])


class CTCLens:
    def __init__(self, lm_head, tokenizer):
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self._logit_dict = None
        self._token_prob_dicts = None

    def compute_logits(self, step_activations, from_frame: int = 0, to_frame: int = -1):
        transformer_layers = [l for l in step_activations.keys() if not l == 'CNN']
        logit_dict = {}
        for layer in transformer_layers:
            logits = list(self.lm_head(step_activations[layer][:, from_frame:to_frame, :]))
            logit_dict[layer] = logits
        self._logit_dict = logit_dict
        return self

    def to_token_probs(self, tokenA: str = 'L', tokenB: str = 'R', from_frame=0, to_frame=-1):
        assert (
            self._logit_dict != None
        ), "no computed logits found; use .compute_logits() before using .to_token_probs()"

        tokenA_idx = self.tokenizer.vocab[tokenA]
        tokenB_idx = self.tokenizer.vocab[tokenB]

        token_prob_dicts = []
        for l, step_logits in self._logit_dict.items():
            step_probs = [logits.softmax(dim=-1) for logits in step_logits]
            tokenA_probs = [torch.max(probs[from_frame:to_frame, tokenA_idx]).detach().numpy() for probs in step_probs]
            tokenB_probs = [torch.max(probs[from_frame:to_frame, tokenB_idx]).detach().numpy() for probs in step_probs]
            N_scores = len(tokenA_probs) + len(tokenB_probs)
            token_prob_dict = {
                'measure': ['ctc_probability'] * N_scores,
                'layer': [l] * N_scores,
                'token': [tokenA] * len(tokenA_probs) + [tokenB] * len(tokenB_probs),
                'step': list(range(len(tokenA_probs))) + list(range(len(tokenB_probs))),
                'score': np.concatenate([tokenA_probs, tokenB_probs]),
            }
            token_prob_dicts.append(token_prob_dict)
        self._token_prob_dicts = token_prob_dicts
        return self

    def to_transcriptions(self, layer='T12'):
        assert (
            self._logit_dict != None
        ), "no computed logits found; use .compute_logits() before using .to_transcriptions()"
        step_logits = self._logit_dict[layer]
        for step, logits in enumerate(step_logits):
            print('step \t transcription')
            print(step, '\t', self.tokenizer.batch_decode(torch.argmax(logits.unsqueeze(0), dim=-1)))

    def to_df(self):
        assert (
            self._token_prob_dicts != None
        ), "no computed token probabilities found; use .compute_logits().to_token_probs() before using .to_df()"
        return pd.concat([pd.DataFrame.from_dict(token_prob_dict) for token_prob_dict in self._token_prob_dicts])


class ProbeProbability:
    def __init__(self, probes_path):
        self._probes = pickle.load(open(probes_path, 'rb'))

    def compute(self, step_embeddings):
        layers = list(step_embeddings.keys())
        measure_dicts = []
        for l in layers:
            layer_probe = self._probes[l]
            probs = layer_probe.predict_proba(step_embeddings[l])
            L_probs = probs[:, 0]
            R_probs = probs[:, 1]
            N_scores = len(L_probs) + len(R_probs)
            measure_dict = {
                'measure': ['probe_probability'] * N_scores,
                'layer': [l] * N_scores,
                'token': ['L'] * len(L_probs) + ['R'] * len(R_probs),
                'step': list(range(len(L_probs))) + list(range(len(R_probs))),
                'score': np.concatenate([L_probs, R_probs]),
            }
            measure_dicts.append(measure_dict)
        self._measure_dicts = measure_dicts
        return self

    def to_df(self):
        assert self._measure_dicts != None, "no computed probabilities found; use .compute() before using .to_df()"
        return pd.concat([pd.DataFrame.from_dict(measure_dict) for measure_dict in self._measure_dicts])
