import tqdm
import torch


def encode_output_tokens(input_text, output_text, hookpoint, tokenizer, model, sae, activation=False):
    """Encode a single (input, output) pair and return per-token SAE pre-activations
    over the generated answer span, together with the answer span boundaries
    and the underlying tokenizer outputs.

    Returns:
        pre_acts: torch.Tensor of shape (T, K), where T is the full sequence length
            (input + output) and K is the SAE dictionary size. The slice
            pre_acts[output_start_idx:output_end_idx] covers the generated answer.
        output_start_idx: int, first token index belonging to ``output_text``.
        output_end_idx: int, one-past-last token index of ``output_text``.
        encoded_text: the tokenizer batch output (contains ``input_ids`` and
            ``offset_mapping``), useful for decoding spans around active tokens.
    """
    if tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': input_text}],
            tokenize=False,
            add_special_tokens=True,
            add_generation_prompt=True,
        )
    text = input_text + output_text
    encoded_text = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False if tokenizer.chat_template else True,
    )
    pre_acts = sae_encoding(
        encoded_text['input_ids'],
        encoded_text['attention_mask'],
        hookpoint,
        model,
        sae,
        activation=activation,
    )
    offsets = encoded_text['offset_mapping'][0]
    output_start_idx = None
    output_end_idx = len(encoded_text['input_ids'][0])
    for i, span in enumerate(offsets):
        if span[0] <= len(input_text) < span[1]:
            output_start_idx = i
            break
    if output_start_idx is None:
        output_start_idx = output_end_idx
    return pre_acts, output_start_idx, output_end_idx, encoded_text


def sae_encoding(input_ids, attention_mask, hookpoint, model, sae, activation=False, topk=False):
    hook_results = []
    def get_hidden_state(module, inputs, outputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = outputs.flatten(0, 1)
        hook_results.append(inputs.flatten(0, 1) if sae.cfg.transcode else outputs)
    with torch.inference_mode():
        handle = model.base_model.get_submodule(hookpoint).register_forward_hook(get_hidden_state)
        _ = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
        )
        hidden_state = hook_results[0]
        handle.remove()
        features = sae.encode(hidden_state.to(sae.device).to(sae.dtype))
    if not topk:        
        if type(sae).__name__ == 'SparseAutoEncoder':
            pre_acts = features
        else:
            if activation:
                pre_acts = torch.zeros_like(features.pre_acts)
                pre_acts = pre_acts.scatter(1, features.top_indices, features.top_acts)
            else:
                pre_acts = features.pre_acts
        return pre_acts
    else:
        if type(sae).__name__ == 'SparseAutoEncoder':
            raise ValueError("topk not supported for SparseAutoEncoder")
        else:
            return features.top_acts, features.top_indices

def encode_outputs(inputs, outputs, hookpoint, tokenizer, model, sae, agg="max", activation=False, show_progress=True, fixed_indices=None):
    assert len(inputs) == len(outputs)
    if show_progress:
        iterator = tqdm.tqdm(range(len(inputs)))
    else:
        iterator = range(len(inputs))
    if agg.startswith("acti_"):
        activation = True
        agg = agg[5:]
    features = []
    for idx in iterator:
        input = inputs[idx]
        output = outputs[idx]
        if tokenizer.chat_template:
            input = tokenizer.apply_chat_template([{'role': 'user', 'content': input}], tokenize=False, add_special_tokens=True, add_generation_prompt=True)
        text = input + output
        encoded_text = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False if tokenizer.chat_template else True)
        pre_acts = sae_encoding(encoded_text['input_ids'], encoded_text['attention_mask'], hookpoint, model, sae, activation=activation)
        offsets = encoded_text['offset_mapping'][0]
        output_start_idx = None
        output_end_idx = len(encoded_text['input_ids'][0])
        for i, span in enumerate(offsets):
            if span[0] <= len(input) < span[1]:
                output_start_idx = i
            if output_start_idx is not None and output_end_idx is not None:
                break
        if agg == "max":
            agg_pre_acts = pre_acts[output_start_idx:output_end_idx].max(dim=0)[0]
        elif agg == "mean":
            agg_pre_acts = pre_acts[output_start_idx:output_end_idx].mean(dim=0)
        elif agg == "sum":
            agg_pre_acts = pre_acts[output_start_idx:output_end_idx].sum(dim=0)
        else:
            raise ValueError(f"Unknown agg method: {agg}")
        features.append(agg_pre_acts.cpu())
    features = torch.stack(features, dim=0)
    return features.half()