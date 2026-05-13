import sys
sys.path.append("../src")
import os
import torch
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sae_encoding import encode_outputs, sae_encoding, encode_output_tokens
from utils import compute_mutual_information_chunked

class RAGLens:

    def __init__(
        self, 
        tokenizer, 
        model, 
        sae, 
        hookpoint, 
        top_k = 1000,
        max_bins = 32, 
        random_state = 0,
        early_stopping_tolerance = 1e-5,
        validation_size = 0.1,
        max_rounds = 1000,
    ):
        
        self.tokenizer = tokenizer
        self.model = model
        self.sae = sae
        self.hookpoint = hookpoint
        self.clf = ExplainableBoostingClassifier(
            interactions = 0, 
            max_bins = max_bins, 
            random_state = random_state, 
            early_stopping_tolerance = early_stopping_tolerance, 
            validation_size = validation_size, 
            max_rounds = max_rounds
        )
        self.top_k = top_k
        self.top_k_indices = None

    def fit(
        self,
        inputs,
        outputs,
        labels,
        save_path=None,
        n_bins = 50,
        chunk_size = 2000
    ):

        assert type(inputs) == type(outputs) == type(labels) == list
        assert len(inputs) == len(outputs) == len(labels)
        
        if save_path and os.path.exists(save_path):
            features = np.load(save_path)
        else:
            print("Encoding training features with SAE...")
            features = encode_outputs(
                inputs, 
                outputs, 
                self.hookpoint, 
                self.tokenizer, 
                self.model, 
                self.sae, 
                agg='max', 
                show_progress=True
            ).numpy()
            if save_path:
                np.save(save_path, features)

        # Mutual Information-based Feature Selection
        print(f"Extracting top {self.top_k} key SAE features...")
        mi_scores = compute_mutual_information_chunked(
            torch.tensor(features), 
            torch.tensor(labels), 
            n_bins = n_bins,
            chunk_size = chunk_size
        ).cpu().numpy()

        sorted_indices = np.argsort(mi_scores)[::-1]
        self.top_k_indices = sorted_indices[:self.top_k]

        # Generalized Additive Model Fitting
        print(f"Fitting an additive model based on key SAE features...")
        self.clf.fit(features[:, self.top_k_indices], labels)

    def predict_proba(self, inputs, outputs):
        
        if type(inputs) == type(outputs) == str:
            inputs = [inputs]
            outputs = [outputs]
        
        assert type(inputs) == type(outputs) == list

        features = encode_outputs(
            inputs, 
            outputs, 
            self.hookpoint, 
            self.tokenizer, 
            self.model, 
            self.sae, 
            agg='max', 
            show_progress=False
        ).numpy()[:, self.top_k_indices]
        logits = self.clf.predict_proba(features)[:, 1]
        return logits
        
    def predict(self, inputs, outputs):

        logits = self.predict_proba(inputs, outputs)
        preds = (logits > 0.5).astype(int)

        return preds

    def explain(self, inputs, outputs, top_n_features=5, token_window=3):
        """For each (input, output) pair, identify the SAE features that push
        the GAM most toward "hallucinated" and locate where they fire in the
        generated answer. Each explanation contains, for every selected feature,
        the GAM contribution, the SAE activation, the index of the
        max-activation token, and the decoded ``±token_window`` span around it.

        Returns a list of dicts (one per example), each with keys:
            ``feature_indices``: list[int] — global SAE feature indices (length ≤ ``top_n_features``).
            ``contributions``: list[float] — GAM additive-term contribution per feature.
            ``activations``: list[float] — pooled feature activation per feature.
            ``token_idxs``: list[int] — global token index (in ``input_ids``) of the max activation.
            ``spans``: list[str] — decoded ``±token_window``-token string around each peak.

        Features are kept only if their GAM contribution is strictly positive
        and the pooled activation is non-zero, mirroring the filter used in the
        paper's mitigation experiments. The returned lists can therefore be
        shorter than ``top_n_features`` (empty if no feature passes the filter).
        """
        if self.top_k_indices is None:
            raise RuntimeError("RAGLens has not been fit yet; call .fit(...) first.")

        if isinstance(inputs, str) and isinstance(outputs, str):
            inputs = [inputs]
            outputs = [outputs]
        assert isinstance(inputs, list) and isinstance(outputs, list)
        assert len(inputs) == len(outputs)

        pooled_features = encode_outputs(
            inputs,
            outputs,
            self.hookpoint,
            self.tokenizer,
            self.model,
            self.sae,
            agg='max',
            show_progress=False,
        ).numpy()[:, self.top_k_indices]

        local_terms = self.clf.eval_terms(pooled_features)

        explanations = []
        for idx, (input_text, output_text) in enumerate(zip(inputs, outputs)):
            ranking = np.argsort(-local_terms[idx])[:top_n_features]
            keep = (local_terms[idx, ranking] > 0) & (pooled_features[idx, ranking] > 0)
            ranking = ranking[keep]

            if len(ranking) == 0:
                explanations.append({
                    'feature_indices': [],
                    'contributions': [],
                    'activations': [],
                    'token_idxs': [],
                    'spans': [],
                })
                continue

            pre_acts, output_start_idx, output_end_idx, encoded_text = encode_output_tokens(
                input_text, output_text, self.hookpoint, self.tokenizer, self.model, self.sae,
            )
            input_ids = encoded_text['input_ids'][0]

            global_indices = self.top_k_indices[ranking]
            token_idxs = []
            spans = []
            for feature_idx in global_indices:
                offset = pre_acts[output_start_idx:output_end_idx, feature_idx].max(dim=0)[1]
                activated_token_idx = int(output_start_idx + offset.item())
                token_idxs.append(activated_token_idx)
                start = max(0, activated_token_idx - token_window)
                end = min(len(input_ids), activated_token_idx + token_window)
                spans.append(self.tokenizer.decode(input_ids[start:end]))

            explanations.append({
                'feature_indices': global_indices.tolist(),
                'contributions': local_terms[idx, ranking].tolist(),
                'activations': pooled_features[idx, ranking].tolist(),
                'token_idxs': token_idxs,
                'spans': spans,
            })

        return explanations