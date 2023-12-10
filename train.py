import os
import glob
import time
import copy
import math
import random
import gc
from tqdm import tqdm
import numpy as np

import flax
from flax.training import train_state
from flax import jax_utils, traverse_util
from flax.training.common_utils import get_metrics, onehot, shard

import jax
import jax.numpy as jnp

import optax

from transformers import T5TokenizerFast, FlaxT5ForConditionalGeneration, T5Config

## custom
from utils import read_pickle
from processor import DataCollatorForT5MLM, custom_data_loader

import wandb

run_name="masked_loss_without_drop_out_UL2_5e-3"
project_name="ul2_namu_TPU"

path = '*_texts.pickle'
base_path = 'google/t5-v1_1-base'
config_save_path  = './ul2_config'
tokenizer_path = 'namuwiki_ul2_tokenizer'

seed = 42

epochs = 14
logging_steps = 100

optimizer_type = 'adafactor'
learning_rate = 5e-3
warmup_steps = 2000
weight_decay = 1e-3

per_device_train_batch_size = 16

batch_size = per_device_train_batch_size * jax.device_count()

# initialise a wandb run
wandb.init(
    name=run_name,
    project=project_name,
    )

num_of_hosts = jax.process_count()
current_host_idx = jax.process_index()

denoiser_settings = [
    {'denoiser_type' : 'R',
     'mean_noise_span_length' : 3,
     'noise_density' : 0.15,
     'prefix_token' : 32100},
    {'denoiser_type' : 'R',
     'mean_noise_span_length' : 8,
     'noise_density' : 0.15,
     'prefix_token' : 32100},
    {'denoiser_type' : 'X',
     'mean_noise_span_length' : 3,
     'noise_density' : 0.5,
     'prefix_token' : 32101},
    {'denoiser_type' : 'X',
     'mean_noise_span_length' : 8,
     'noise_density' : 0.5,
     'prefix_token' : 32101},
    {'denoiser_type' : 'X',
     'mean_noise_span_length' : 64,
     'noise_density' : 0.15,
     'prefix_token' : 32101},
    {'denoiser_type' : 'X',
     'mean_noise_span_length' : 64,
     'noise_density' : 0.5,
     'prefix_token' : 32101},
    {'denoiser_type' : 'S',
     'mean_noise_span_length' : -1,
     'noise_density' : 0.25,
     'prefix_token' : 32102},
]

def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    # find out all LayerNorm parameters
    layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
    layer_norm_named_params = set(
        [
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        ]
    )
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)

def accumulate_metrics(metrics):
    metrics = jax.device_get(metrics)
    return {
        k: np.mean([metric[k] for metric in metrics])
        for k in metrics[0]
    }

def main():
    path_list = glob.glob(path)
    dataset = read_pickle(path_list)

    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)

    config = T5Config.from_pretrained(base_path, vocab_size=len(tokenizer))
    config.decoder_start_token_id = 1
    config.dropout_rate = 0
    config.save_pretrained(config_save_path)

    model = FlaxT5ForConditionalGeneration(
                config,
                seed=seed,
                dtype=getattr(jnp, "float32"),
            )

    model.config.dropout_rate = 0.1

    # model = FlaxT5ForConditionalGeneration.from_pretrained(
    #             "/home/caesian/workspace/T5/epoch_6"
    #         )

    DataCollator = DataCollatorForT5MLM(tokenizer, 
                                        max_sentinel_ids=104,
                                        denoiser_settings=denoiser_settings,
                                        config=config)

    num_train_steps = len(dataset) // batch_size * epochs

    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0,
        transition_steps=num_train_steps - warmup_steps,
    )
    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
    )

    # linear_decay_lr_schedule_fn = optax.join_schedules(
    #     schedules=[warmup_fn], boundaries=[warmup_steps]
    # )

    if optimizer_type == 'adafactor':
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=0.9,
            b2=0.999,
            weight_decay=weight_decay,
            mask=decay_mask_fn,
        )

    rng = jax.random.PRNGKey(seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    # Setup train state
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer)
    state = jax_utils.replicate(state)

    # Define gradient update step fn
    @jax.jit
    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params):
            labels = batch.pop("labels")

            logits = state.apply_fn(**batch, 
                                    params=params, 
                                    dropout_rng=dropout_rng, 
                                    train=True)[0]

            # compute loss

            # unmasked_loss
            # loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])).mean()
            # return loss

            # masked_loss
            unmasked_loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
            loss_mask = jnp.float32(labels != 0)

            masked_loss = unmasked_loss * loss_mask

            reduced_masked_loss = jnp.sum(masked_loss) / jnp.sum(loss_mask)

            return reduced_masked_loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )

        return new_state, metrics, new_dropout_rng

    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    train_time = 0
    for epoch in tqdm(range(1, epochs + 1), desc=f"Epoch ...", position=0, leave=True):
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # # Generate an epoch by shuffling sampling indices from the train dataset
        num_train_samples = len(dataset)
        # # Avoid using jax.numpy here in case of TPU training
        # train_samples_idx = np.random.permutation(np.arange(num_train_samples))
        # train_batch_idx = generate_batch_splits(train_samples_idx, batch_size)

        train_dataloader = custom_data_loader(rng, 
                                              dataset, 
                                              batch_size, 
                                              collate_fn=DataCollator,
                                              shuffle=True)

        # Gather the indexes for creating the batch and do a training step
        # for step, batch_idx in enumerate(tqdm(train_batch_idx, desc="Training...", position=1)):

        steps_per_epoch = num_train_samples // batch_size

        with tqdm(total=steps_per_epoch, desc="Training...", leave=False) as progress_bar_train:
            for step, batch in enumerate(tqdm(train_dataloader, desc="Training...", position=1, leave=False)):
                
                local_host_model_inputs = {
                    key: np.split(batch[key], num_of_hosts, axis=0)[current_host_idx]
                    for key, value in batch.items()
                }

                # Model forward
                model_inputs = shard(local_host_model_inputs)
                state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rngs)

                # Save metrics
                train_metric = jax_utils.unreplicate(train_metric)

                if jax.process_index() == 0:
                    metrics = {"train/loss": train_metric['loss'].mean(), 
                                "train/epoch": (step + 1 + (steps_per_epoch * (epoch-1))) / steps_per_epoch,
                                "learning_rate":train_metric['learning_rate'].mean(), 
                                }
                    
                    if step + 1 < steps_per_epoch:
                        wandb.log(metrics)

                progress_bar_train.update(1)

                # if cur_step % training_args.save_steps == 0 and cur_step > 0:
                #     # save checkpoint after each epoch and push checkpoint to the hub
                #     if jax.process_index() == 0:
                #         params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
                #         model.save_pretrained(training_args.output_dir, params=params)
                #         tokenizer.save_pretrained(training_args.output_dir)
                #         if training_args.push_to_hub:
                #             repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
            model.save_pretrained('./epoch_{}'.format(epoch), params=params)
    wandb.finish()
if __name__ == "__main__":
    main()
