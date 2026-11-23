
import torch
import torch.nn.functional as F


def apply_repetition_penalty(logits, input_ids, penalty=1.0):

    if penalty <= 0:
        raise ValueError("penalty must be > 0")

    if penalty == 1.0:
        return logits

    logits = logits.clone()

    # Loop over each sequence in the batch.
    for b in range(input_ids.size(0)):

        # Get the tokens that have already appeared.
        previous_tokens = set(input_ids[b].tolist())

        # Apply the penalty to each previous token.
        for token_id in previous_tokens:

            if logits[b, token_id] < 0:
                logits[b, token_id] *= penalty
            else:
                logits[b, token_id] /= penalty

    return logits


def apply_temperature(logits, temperature=1.0):

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    return logits / temperature


def apply_top_k(logits, top_k=None):

    if top_k is None:
        return logits

    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    top_k = min(top_k, logits.size(-1))

    # Find the K highest logits.
    top_values, top_indices = torch.topk(
        logits,
        top_k,
        dim=-1
    )

    # Create a copy of the logits.
    filtered_logits = logits.clone()

    # Remove every token that is not in the top K.
    for b in range(logits.size(0)):

        for token_id in range(logits.size(-1)):

            if token_id not in top_indices[b]:
                filtered_logits[b, token_id] = float("-inf")

    return filtered_logits


def apply_top_p(logits, top_p=None):

    if top_p is None:
        return logits

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if top_p == 1.0:
        return logits

    filtered_logits = logits.clone()

    # Process each sequence independently.
    for b in range(logits.size(0)):

        # Convert logits into probabilities.
        probs = F.softmax(logits[b], dim=-1)

        # Sort probabilities from highest to lowest.
        sorted_probs, sorted_indices = torch.sort(
            probs,
            descending=True
        )

        cumulative_prob = 0.0

        # Keep tokens until we reach top_p.
        for i in range(len(sorted_probs)):

            token_id = sorted_indices[i].item()
            cumulative_prob += sorted_probs[i].item()

            # Keep the token that reaches or crosses top_p.
            # Remove the remaining tokens.
            if cumulative_prob >= top_p:

                for j in range(i + 1, len(sorted_indices)):
                    remove_id = sorted_indices[j].item()
                    filtered_logits[b, remove_id] = float("-inf")

                break

    return filtered_logits


def sample_next_token(
    logits,
    input_ids=None,
    temperature=1.0,
    top_k=None,
    top_p=None,
    repetition_penalty=1.0,
    do_sample=True
):

    # 1. Repetition penalty.
    if input_ids is not None:
        logits = apply_repetition_penalty(
            logits,
            input_ids,
            repetition_penalty
        )

    # 2. Greedy decoding.
    if not do_sample:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

    # 3. Temperature.
    logits = apply_temperature(logits, temperature)

    # 4. Top-K.
    logits = apply_top_k(logits, top_k)

    # 5. Top-P.
    logits = apply_top_p(logits, top_p)

    # 6. Convert the final logits into probabilities.
    probs = F.softmax(logits, dim=-1)

    # 7. Sample one token.
    next_token = torch.multinomial(
        probs,
        num_samples=1
    )

    return next_token
