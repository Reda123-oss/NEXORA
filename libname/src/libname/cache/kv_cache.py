
import torch


class KVCache:
    """
    Standard growing KV cache.

    Stores:
        key:   [B, n_kv_heads, sequence_length, head_dim]
        value: [B, n_kv_heads, sequence_length, head_dim]
    """

    def __init__(self):
        self.key = None
        self.value = None
        self.seq_length = 0

    def update(self, key, value, start_pos):
        """
        key/value: [B, H_kv, T, D]
        start_pos: absolute position of the first new token

        Returns:
            all_key:   [B, H_kv, total_length, D]
            all_value: [B, H_kv, total_length, D]
            positions: [total_length]
        """

        if key.shape != value.shape:
            raise ValueError("Key and value must have the same shape.")

        if start_pos != self.seq_length:
            raise ValueError(
                f"Expected start_pos={self.seq_length}, got {start_pos}."
            )

        if self.key is None:
            self.key = key
            self.value = value
        else:
            if key.shape[:2] != self.key.shape[:2]:
                raise ValueError("Batch size or KV-head count changed.")

            if key.shape[-1] != self.key.shape[-1]:
                raise ValueError("Head dimension changed.")

            if key.device != self.key.device:
                raise ValueError("Cache and new keys must be on the same device.")

            if key.dtype != self.key.dtype:
                raise ValueError("Cache and new keys must have the same dtype.")

            self.key = torch.cat([self.key, key], dim=2)
            self.value = torch.cat([self.value, value], dim=2)

        self.seq_length += key.shape[2]

        positions = torch.arange(
            self.seq_length,
            device=key.device
        )

        return self.key, self.value, positions

    def get_seq_length(self):
        return self.seq_length

    def reset(self):
        self.key = None
        self.value = None
        self.seq_length = 0
