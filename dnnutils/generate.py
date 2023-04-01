
import heapq
import random

import numpy as np
import torch
import torch.nn.functional as F

def ensure_length(self, txt, out_len, pad_value=0):
    if len(txt) < out_len:
        txt = list(txt) + [pad_value] * (out_len - len(txt))
    else:
        txt = txt[:out_len]
    return txt

class BeamGenerator:
    def __init__(self, eos_token_id=3, encoder_ids_name='encoder_ids', decoder_ids_name='decoder_ids'):
        self.eos_token_id = eos_token_id
        self.encoder_ids_name = encoder_ids_name
        self.decoder_ids_name = decoder_ids_name

    def __call__(self, model, seed_token_ids, max_len=40, return_hypotheses_n=5, beamsize=5):
        seed_decoder_ids = seed_token_ids[self.decoder_ids_name]
        
        partial_hypotheses = [(0, seed_decoder_ids)]
        final_hypotheses = []

        while len(partial_hypotheses) > 0:
            cur_partial_score, cur_partial_hypothesis = heapq.heappop(partial_hypotheses)

            in_batch = {
                self.encoder_ids_name : seed_token_ids[self.encoder_ids_name].unsqueeze(0), 
                self.decoder_ids_name : cur_partial_hypothesis.unsqueeze(0)
                }
            # in_batch = torch.tensor(cur_partial_hypothesis).unsqueeze(0).to(self.model.device)
            next_tokens_logits = model(in_batch)[0, -1]
            next_tokens_logproba = F.log_softmax(next_tokens_logits, dim=0)
            topk_continuations = next_tokens_logproba.topk(beamsize)

            for token_score, token_idx in zip(topk_continuations.values, topk_continuations.indices):
                token_score = float(token_score)
                token_idx = token_idx.view(1)

                old_denorm_score = cur_partial_score * np.sqrt(len(cur_partial_hypothesis))
                new_score = (old_denorm_score - token_score) / np.sqrt(len(cur_partial_hypothesis) + 1)

                new_hypothesis = torch.cat((cur_partial_hypothesis, token_idx), dim=0)
                new_item = (new_score, new_hypothesis)

                if token_idx == self.eos_token_id or new_hypothesis.shape[0] >= max_len:
                    final_hypotheses.append(new_item)
                else:
                    heapq.heappush(partial_hypotheses, new_item)

            if len(partial_hypotheses) > beamsize:
                partial_hypotheses = heapq.nsmallest(beamsize, partial_hypotheses)
                heapq.heapify(partial_hypotheses)

        final_hypotheses.sort()
        final_hypotheses = final_hypotheses[:return_hypotheses_n]

        return final_hypotheses