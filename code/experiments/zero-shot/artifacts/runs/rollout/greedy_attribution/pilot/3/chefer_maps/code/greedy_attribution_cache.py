"""Checkpoint-safe sequential KV with shared immutable prefix chunks.

Each model call owns a new snapshot. Cache updates concatenate its fixed prior
chunks and the current call's K/V without retaining the full concatenation.
Checkpoint recomputation may repeat update, but can neither append twice nor
mutate a later call's frozen chunk references. The tensors remain attached to
the differentiable original prefill.
"""
from transformers.cache_utils import DynamicCache, DynamicLayer
import torch


def _concatenate(chunks):
    return torch.cat(chunks, dim=-2) if chunks else None


class _SnapshotLayer(DynamicLayer):
    is_sliding = False

    def __init__(self, key_chunks=(), value_chunks=()):
        self.base_key_chunks = tuple(key_chunks)
        self.base_value_chunks = tuple(value_chunks)
        self.key_chunks = self.base_key_chunks
        self.value_chunks = self.base_value_chunks
        self.is_initialized = bool(self.key_chunks)
        self.update_count = 0
        if self.is_initialized:
            self.dtype = self.key_chunks[0].dtype
            self.device = self.key_chunks[0].device

    @property
    def keys(self):
        return _concatenate(self.key_chunks)

    @property
    def values(self):
        return _concatenate(self.value_chunks)

    def update(self, key_states, value_states, cache_kwargs=None):
        key_chunks = self.base_key_chunks + (key_states,)
        value_chunks = self.base_value_chunks + (value_states,)
        # Preserve the original forward's output references. A backward
        # recomputation needs its temporary concatenation, but should not keep
        # a second large prefill K/V tensor alive in the cache object.
        if self.update_count == 0:
            self.key_chunks = key_chunks
            self.value_chunks = value_chunks
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True
        self.update_count += 1
        return _concatenate(key_chunks), _concatenate(value_chunks)

    def get_seq_length(self):
        return sum(chunk.shape[-2] for chunk in self.key_chunks)

    def get_mask_sizes(self, cache_position):
        return self.get_seq_length() + cache_position.shape[0], 0

    def reset(self):
        raise RuntimeError('Create a new SnapshotDynamicCache instead of resetting a differentiable snapshot')

    def crop(self, max_length):
        raise RuntimeError('Cropping a differentiable cache snapshot is unsupported')

    def reorder_cache(self, beam_idx):
        raise RuntimeError('Beam reordering is unsupported for attribution snapshots')


class _BaseAccessor:
    def __init__(self, cache, kind):
        self.cache, self.kind = cache, kind

    def __getitem__(self, index):
        return self.cache.base_key(index) if self.kind == 'key' else self.cache.base_value(index)

    def __len__(self):
        return len(self.cache.layers)


class SnapshotDynamicCache(DynamicCache):
    """Drop-in full-attention DynamicCache for one differentiable model call.

    ``SnapshotDynamicCache(previous=last_cache, config=model.config)`` shares
    prior K/V chunk tensors without copying or detaching them. ``base_key(i)``
    and ``base_value(i)`` concatenate only one layer when message diagnostics
    need it. Do not retain those temporary concatenations across forwards.
    """

    def __init__(self, previous=None, config=None):
        super().__init__(config=config)
        if config is None:
            raise ValueError('A model configuration is required for fixed attribution layer coverage')
        if self.offloading or any(layer.is_sliding for layer in self.layers):
            raise ValueError('Only non-offloaded full-attention cache layers are supported')
        if previous is not None and len(previous.layers) != len(self.layers):
            raise ValueError('Previous cache layer count does not match configuration')
        layers = []
        for index in range(len(self.layers)):
            prior = previous.layers[index] if previous is not None else None
            if isinstance(prior, _SnapshotLayer):
                keys, values = prior.key_chunks, prior.value_chunks
            elif prior is not None and prior.is_initialized:
                keys, values = (prior.keys,), (prior.values,)
            else:
                keys, values = (), ()
            layers.append(_SnapshotLayer(keys, values))
        self.layers = layers
        self.base_keys = _BaseAccessor(self, 'key')
        self.base_values = _BaseAccessor(self, 'value')

    def base_key(self, layer_idx):
        return _concatenate(self.layers[layer_idx].base_key_chunks)

    def base_value(self, layer_idx):
        return _concatenate(self.layers[layer_idx].base_value_chunks)

    @property
    def update_counts(self):
        return {index: layer.update_count for index, layer in enumerate(self.layers)}
