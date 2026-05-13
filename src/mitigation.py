import sys
sys.path.append("../src")
import torch
import numpy as np


_INSTANCE_FEEDBACK = "There are hallucinations in your output. Please revise it."


def _token_feedback(spans):
    return (
        "There are hallucinations in your output, especially on the following spans:\n"
        f"{spans}\n\n"
        "Please revise it."
    )


class RAGLensMitigator:
    """Post-hoc hallucination mitigation driven by a fitted ``RAGLens`` detector.

    Two feedback modes are supported, matching the paper:

    * ``mode='instance'``: an instance-level warning that asks the model to
      revise its output, using only the detector's binary decision.
    * ``mode='token'``: token-level feedback that, in addition to the warning,
      lists short decoded spans around the tokens where the SAE features most
      responsible for the prediction fired.

    Only items flagged by the detector (``preds == 1``) are revised; for
    others the returned slot is ``None`` and the original output should be kept.

    The ``generator`` argument is any callable mapping a list of chat
    conversations (each conversation is a list of ``{"role", "content"}``
    messages) to a list of generated assistant strings. Two ready-made
    factories are provided: :func:`hf_generator` and :func:`vllm_generator`.
    """

    def __init__(self, raglens, generator):
        self.raglens = raglens
        self.generator = generator

    def build_feedback_prompt(self, input_text, output_text, mode='token', explanation=None):
        if mode not in ('instance', 'token'):
            raise ValueError(f"Unknown mitigation mode: {mode!r}; expected 'instance' or 'token'.")

        feedback = _INSTANCE_FEEDBACK
        if mode == 'token':
            spans = explanation.get('spans', []) if explanation else []
            if len(spans) > 0:
                feedback = _token_feedback(spans)

        return [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
            {"role": "user", "content": feedback},
        ]

    def mitigate(self, inputs, outputs, mode='token', preds=None, top_n_features=5, token_window=3):
        """Generate revisions for items the detector flags as hallucinated.

        Returns a list of strings aligned with ``inputs``: revised output for
        flagged items, ``None`` for items the detector marks as faithful.
        """
        assert isinstance(inputs, list) and isinstance(outputs, list)
        assert len(inputs) == len(outputs)

        if preds is None:
            preds = self.raglens.predict(inputs, outputs)
        preds = np.asarray(preds).astype(int)
        assert len(preds) == len(inputs)

        flagged_idx = [i for i, p in enumerate(preds) if p == 1]
        if len(flagged_idx) == 0:
            return [None] * len(inputs)

        flagged_inputs = [inputs[i] for i in flagged_idx]
        flagged_outputs = [outputs[i] for i in flagged_idx]

        if mode == 'token':
            explanations = self.raglens.explain(
                flagged_inputs,
                flagged_outputs,
                top_n_features=top_n_features,
                token_window=token_window,
            )
        else:
            explanations = [None] * len(flagged_idx)

        conversations = [
            self.build_feedback_prompt(inp, out, mode=mode, explanation=exp)
            for inp, out, exp in zip(flagged_inputs, flagged_outputs, explanations)
        ]

        revised = self.generator(conversations)
        assert len(revised) == len(flagged_idx), (
            f"generator returned {len(revised)} outputs for {len(flagged_idx)} prompts"
        )

        revised_aligned = [None] * len(inputs)
        for src_idx, text in zip(flagged_idx, revised):
            revised_aligned[src_idx] = text
        return revised_aligned


def hf_generator(model, tokenizer, max_new_tokens=1024, temperature=0.0):
    """Return a ``generator(conversations) -> list[str]`` backed by HuggingFace
    ``model.generate``. Mitigation requires a chat-tuned tokenizer; the
    revising LLM is typically the chat-tuned variant of the model used for
    SAE encoding (e.g. ``Llama-2-7b-chat-hf`` for the paper's experiments).
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "Mitigation requires a tokenizer with a chat template, but "
            f"{tokenizer.__class__.__name__} has none. Use the chat-tuned "
            "variant of your model (e.g. the *-Instruct or *-chat checkpoint)."
        )
    do_sample = temperature > 0

    def generate(conversations):
        outputs = []
        for conv in conversations:
            chat = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            encoded = tokenizer(chat, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            new_tokens = generated[0, encoded['input_ids'].shape[1]:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outputs

    return generate


def vllm_generator(llm, sampling_params=None):
    """Return a ``generator(conversations) -> list[str]`` backed by a vLLM
    ``LLM`` instance. ``sampling_params`` defaults to greedy decoding with up
    to 1024 new tokens, matching the paper's mitigation experiments.
    """
    from vllm import SamplingParams

    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=1024)

    def generate(conversations):
        outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    return generate
