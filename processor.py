import random
import numpy as np
import jax
import jax.numpy as jnp
import flax
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right

class DataCollatorForT5MLM(object):

    def __init__(
            self,
            tokenizer,
            max_sentinel_ids,
            denoiser_settings,
            config,
    ):
        self.tokenizer = tokenizer
        self.max_sentinel_ids = max_sentinel_ids
        self.denoiser_settings  = denoiser_settings
        self.decoder_start_token_id = config.decoder_start_token_id

    def __call__(self, batch):
        denoiser_index = random.randint(0,len(self.denoiser_settings)-1)
        denoiser = self.denoiser_settings[denoiser_index]

        denoiser_type = denoiser['denoiser_type']
        noise_density = denoiser['noise_density']
        mean_noise_span_length = denoiser['mean_noise_span_length'] 
        prefix_ids = np.array([[denoiser['prefix_token']]])

        # convert list to dict and tensorize input
        batch_size = len(batch)

        inputs_max_len = 0
        labels_max_len = 0

        input_ids = []
        labels = []
        attention_mask = []

        for i in range(batch_size):
            mask_indices, is_last = self.random_spans_noise_mask(len(batch[i]), 
                                                                 denoiser_type, 
                                                                 noise_density, 
                                                                 mean_noise_span_length)
            labels_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            input_id = self.filter_input_ids(batch[i], input_ids_sentinel)
            label = self.filter_input_ids(batch[i], labels_sentinel)
            if is_last is False:
                label = label[:-1]

            input_id = np.append(input_id, self.tokenizer.eos_token_id) # 추가 부분

            label = np.insert(label, np.where(label> (len(self.tokenizer)-self.max_sentinel_ids))[0], len(self.tokenizer)-1)
            label = np.append(label, self.tokenizer.eos_token_id)
            
            input_ids.append(input_id)
            labels.append(label)

            input_id_len = len(input_id)
            label_len = len(label)

            if input_id_len > inputs_max_len:
                inputs_max_len = input_id_len
            if label_len > labels_max_len:
                labels_max_len = label_len

        inputs_max_len = 512
        labels_max_len = 256 + 1 + 100 + 100

        for i in range(batch_size):
            input_id_len = len(input_ids[i])
            label_len = len(labels[i])

            input_ids[i] = input_ids[i] if input_id_len == inputs_max_len else np.pad(input_ids[i],(0, inputs_max_len-input_id_len))
            labels[i] =  labels[i] if label_len == labels_max_len else np.pad(labels[i],(0, labels_max_len-label_len))

            attention_mask.append([1]*(input_id_len+len(prefix_ids[0])) + [0]*(inputs_max_len-input_id_len))

        batch = {}
        batch['input_ids'] = jnp.int32(np.concatenate((np.asarray(prefix_ids.repeat(batch_size, axis=0)), np.asarray(input_ids)), axis=-1))        
        batch['labels'] = jnp.int32(np.asarray(labels))
        batch["decoder_input_ids"] = shift_tokens_right(batch["labels"], 
                                                        self.tokenizer.pad_token_id, 
                                                        self.decoder_start_token_id)
        batch['attention_mask'] = jnp.int32(attention_mask)

        return batch

    def create_sentinel_ids(self, mask_indices):

        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[0] = mask_indices[0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - self.max_sentinel_ids + sentinel_ids - 1), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full >= 0]

        return input_ids

    def random_spans_noise_mask(self, length, denoiser_type, noise_density, mean_noise_span_length):
        assert length > 1
        orig_length = length

        num_noise_tokens = int(np.round(length * noise_density))
        if num_noise_tokens >= 1:
            num_noise_tokens += 1

        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)

        is_last = True

        if denoiser_type == 'S':
            span_num = np.arange(length) > (length-num_noise_tokens)
        
        else:
            # pick the lengths of the noise spans and the non-noise spans
            def _random_segmentation(num_items, num_segments):
                """Partition a sequence of items randomly into non-empty segments.
                Args:
                    num_items: an integer scalar > 0
                    num_segments: an integer scalar in [1, num_items]
                Returns:
                    a Tensor with shape [num_segments] containing positive integers that add
                    up to num_items
                """
                mask_indices = np.arange(num_items - 1) < (num_segments - 1)
                np.random.shuffle(mask_indices)
                first_in_segment = np.pad(mask_indices, [[1, 0]])
                segment_id = np.cumsum(first_in_segment)
                # count length of sub segments assuming that list is sorted
                _, segment_length = np.unique(segment_id, return_counts=True)
                return segment_length

            num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

            # avoid degeneracy by ensuring positive number of noise spans
            num_noise_spans = max(num_noise_spans, 1)
            num_nonnoise_tokens = length - num_noise_tokens

            noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)

            if random.random() >= 0.5:
                nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

                interleaved_span_lengths = np.reshape(
                    np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
                )
                span_starts = np.cumsum(interleaved_span_lengths)[:-1]

            else:
                is_last = False
                nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans+1)

                interleaved_span_lengths = np.reshape(
                    np.stack([nonnoise_span_lengths[:-1], noise_span_lengths], axis=1), [num_noise_spans * 2]
                )

                span_starts = np.cumsum(interleaved_span_lengths)

            span_start_indicator = np.zeros((length,), dtype=np.int8)
            span_start_indicator[span_starts] = True
            span_num = np.cumsum(span_start_indicator)

        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length], is_last

def custom_data_loader(rng, dataset, batch_size, collate_fn=None, shuffle=False):
    dataset_size = len(dataset)
    steps_per_epoch = dataset_size // batch_size

    if shuffle:
        all_batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        all_batch_idx = jnp.arange(len(dataset))

    all_batch_idx = all_batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    all_batch_idx = all_batch_idx.reshape((steps_per_epoch, batch_size))

    for batch_idx in all_batch_idx:
        batch = dataset[batch_idx]

        if collate_fn is not None:
            batch = collate_fn(batch)
        else:    
            batch = {k: jnp.array(v) for k, v in batch.items()}

        yield batch