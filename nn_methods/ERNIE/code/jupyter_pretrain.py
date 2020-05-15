"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from knowledge_bert.tokenization import BertTokenizer
from knowledge_bert.modeling import BertForPreTraining
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0


def main():
    ## Required parameters
    path_to_ernie = ""
    data_dir = os.path.join(path_to_ernie, "pretrain_data/merge")
    bert_model = os.path.join(path_to_ernie, "ernie_base")
    task_name = "pretrain"
    output_dir = os.path.join(path_to_ernie, "pretrain_out")
    max_seq_length = 256
    do_train = True
    do_eval = False
    do_lower_case = False
    train_batch_size = 4
    eval_batch_size = 8
    learning_rate = 5e-5
    num_train_epochs = 3.0
    warmup_proportion = default = 0.1
    no_cuda = False
    local_rank = -1
    seed = 42
    gradient_accumulation_steps = 1
    fp16 = True
    loss_scale = 0.0

    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            gradient_accumulation_steps))

    train_batch_size = int(train_batch_size / gradient_accumulation_steps)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not do_train and not do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    task_name = task_name.lower()

    vecs = []
    vecs.append([0] * 100)  # CLS
    with open(os.path.join(path_to_ernie, "kg_embed/entity2vec.vec"), 'r') as fin:
        for line in fin:
            # vec = line.strip().split('\t')
            vec = line.strip().split(' ')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed = torch.FloatTensor(vecs)
    embed = torch.nn.Embedding.from_pretrained(embed)
    # embed = torch.nn.Embedding(5041175, 100)

    logger.info("Shape of entity embedding: " + str(embed.weight.size()))
    del vecs

    train_data = None
    num_train_steps = None
    if do_train:
        # TODO
        import indexed_dataset
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
        import iterators
        # train_data = indexed_dataset.IndexedCachedDataset(data_dir)
        train_data = indexed_dataset.IndexedDataset(data_dir, fix_lua_indexing=True)
        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_sampler = BatchSampler(train_sampler, train_batch_size, True)

        def collate_fn(x):
            x = torch.LongTensor([xx for xx in x])

            entity_idx = x[:, 4 * max_seq_length:5 * max_seq_length]
            # Build candidate
            uniq_idx = np.unique(entity_idx.numpy())
            ent_candidate = embed(torch.LongTensor(uniq_idx + 1))
            ent_candidate = ent_candidate.repeat([n_gpu, 1])
            # build entity labels
            d = {}
            dd = []
            for i, idx in enumerate(uniq_idx):
                d[idx] = i
                dd.append(idx)
            ent_size = len(uniq_idx) - 1

            def map(x):
                if x == -1:
                    return -1
                else:
                    rnd = random.uniform(0, 1)
                    if rnd < 0.05:
                        return dd[random.randint(1, ent_size)]
                    elif rnd < 0.2:
                        return -1
                    else:
                        return x

            ent_labels = entity_idx.clone()
            d[-1] = -1
            ent_labels = ent_labels.apply_(lambda x: d[x])

            entity_idx.apply_(map)
            ent_emb = embed(entity_idx + 1)
            mask = entity_idx.clone()
            mask.apply_(lambda x: 0 if x == -1 else 1)
            mask[:, 0] = 1

            return x[:, :max_seq_length], x[:, max_seq_length:2 * max_seq_length], x[:, 2 * max_seq_length:3 * max_seq_length], x[:,3 * max_seq_length:4 * max_seq_length], ent_emb, mask, x[:,6 * max_seq_length:], ent_candidate, ent_labels

        train_iterator = iterators.EpochBatchIterator(train_data, collate_fn, train_sampler)
        num_train_steps = int(
            len(train_data) / train_batch_size / gradient_accumulation_steps * num_train_epochs)

    # Prepare model
    model, missing_keys = BertForPreTraining.from_pretrained(bert_model,
                                                             cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                 local_rank))
    # if fp16:
    #     model.half()
    model.to(device)
    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_linear = ['layer.2.output.dense_ent', 'layer.2.intermediate.dense_1',
                 'bert.encoder.layer.2.intermediate.dense_1_ent', 'layer.2.output.LayerNorm_ent']
    no_linear = [x.replace('2', '11') for x in no_linear]
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nl in n for nl in no_linear)]
    # param_optimizer = [(n, p) for n, p in param_optimizer if not any(nl in n for nl in missing_keys)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'LayerNorm_ent.bias', 'LayerNorm_ent.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if fp16:
        try:
            # from apex.optimizers import FP16_Optimizer
            # from apex.optimizers import FusedAdam
            # from apex.contrib.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
            import apex.amp as amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        # optimizer = FusedAdam(optimizer_grouped_parameters,
        #                       lr=learning_rate,
        #                       bias_correction=False,
        #                       max_grad_norm=1.0)
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O3")
        # if loss_scale == 0:
        #     optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        # else:
        #     optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
        # logger.info(dir(optimizer))
        # op_path = os.path.join(bert_model, "pytorch_op.bin")
        # optimizer.load_state_dict(torch.load(op_path))

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=t_total)

    global_step = 0
    if do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        model.train()
        import datetime
        fout = open(os.path.join(output_dir, "loss.{}".format(datetime.datetime.now())), 'w')
        for _ in trange(int(num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_iterator.next_epoch_itr(), desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, masked_lm_labels, input_ent, ent_mask, next_sentence_label, ent_candidate, ent_labels = batch
                if fp16:
                    loss, original_loss = model(input_ids, segment_ids, input_mask, masked_lm_labels, input_ent.half(),
                                                ent_mask, next_sentence_label, ent_candidate.half(), ent_labels)
                else:
                    loss, original_loss = model(input_ids, segment_ids, input_mask, masked_lm_labels, input_ent,
                                                ent_mask, next_sentence_label, ent_candidate, ent_labels)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    original_loss = original_loss.mean()
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    # optimizer.backward(loss)
                else:
                    loss.backward()

                fout.write("{} {}\n".format(loss.item() * gradient_accumulation_steps, original_loss.item()))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = learning_rate * warmup_linear(global_step / t_total, warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    # if global_step % 1000 == 0:
                    #    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    #    output_model_file = os.path.join(output_dir, "pytorch_model.bin_{}".format(global_step))
                    #    torch.save(model_to_save.state_dict(), output_model_file)
        fout.close()

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    # Save the optimizer
    # output_optimizer_file = os.path.join(output_dir, "pytorch_op.bin")
    # torch.save(optimizer.state_dict(), output_optimizer_file)

    # Load a trained model that you have fine-tuned
    # model_state_dict = torch.load(output_model_file)
    # model = BertForSequenceClassification.from_pretrained(bert_model, state_dict=model_state_dict)
    # model.to(device)

    # if do_eval and (local_rank == -1 or torch.distributed.get_rank() == 0):
    #     eval_examples = processor.get_dev_examples(data_dir)
    #     eval_features = convert_examples_to_features(
    #         eval_examples, label_list, max_seq_length, tokenizer)
    #     logger.info("***** Running evaluation *****")
    #     logger.info("  Num examples = %d", len(eval_examples))
    #     logger.info("  Batch size = %d", eval_batch_size)
    #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    #     # Run prediction for full data
    #     eval_sampler = SequentialSampler(eval_data)
    #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    #     model.eval()
    #     eval_loss, eval_accuracy = 0, 0
    #     nb_eval_steps, nb_eval_examples = 0, 0
    #     for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    #         input_ids = input_ids.to(device)
    #         input_mask = input_mask.to(device)
    #         segment_ids = segment_ids.to(device)
    #         label_ids = label_ids.to(device)

    #         with torch.no_grad():
    #             tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
    #             logits = model(input_ids, segment_ids, input_mask)

    #         logits = logits.detach().cpu().numpy()
    #         label_ids = label_ids.to('cpu').numpy()
    #         tmp_eval_accuracy = accuracy(logits, label_ids)

    #         eval_loss += tmp_eval_loss.mean().item()
    #         eval_accuracy += tmp_eval_accuracy

    #         nb_eval_examples += input_ids.size(0)
    #         nb_eval_steps += 1

    #     eval_loss = eval_loss / nb_eval_steps
    #     eval_accuracy = eval_accuracy / nb_eval_examples

    #     result = {'eval_loss': eval_loss,
    #               'eval_accuracy': eval_accuracy,
    #               'global_step': global_step,
    #               'loss': tr_loss/nb_tr_steps}

    #     output_eval_file = os.path.join(output_dir, "eval_results.txt")
    #     with open(output_eval_file, "w") as writer:
    #         logger.info("***** Eval results *****")
    #         for key in sorted(result.keys()):
    #             logger.info("  %s = %s", key, str(result[key]))
    #             writer.write("%s = %s\n" % (key, str(result[key])))


main()


if __name__ == "__main__":
    main()
