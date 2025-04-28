# Modified from https://opacus.ai/tutorials/building_text_classifier
import os
import pandas as pd
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, OpenAIGPTForSequenceClassification
import torch
import transformers
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
import numpy as np
from tqdm import tqdm
from opacus import PrivacyEngine
# from private_transformers import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import random
import pickle
import sys
from adam_corr import AdamCorr, OrigAdam, MySGDM
import wandb
import configlib
from datasets import load_dataset
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.prv import PRVAccountant
from opacus.accountants.utils import get_noise_multiplier

parser = configlib.add_parser("config")
# parser.add_argument("--opt_model", choices=['adam', 'adamw', 'adam_corr', 'sgd', 'svag'])
parser.add_argument("--opt_model", choices=['adam', 'adamw', 'dp_adamw', 'dp_adamw_bc', 'adam_corr', 'sgdm', 'svag', 'sgd'])
parser.add_argument("--exp_name", default="tmp", type=str)
parser.add_argument("--exp_group", default="tmp", type=str)
parser.add_argument("--eps_root", default=1e-8, type=float)
parser.add_argument("--eps", default=1e-8, type=float)
parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--dp_noise_multiplier", default=0.4, type=float,
                    help="The noise multiple for DP-SGD and derivatives.")
parser.add_argument("--dp_l2_norm_clip", default=0.1, type=float,
                    help="The L2 clipping value for per example gradients for DP-SGD and derivatives.")
parser.add_argument("--batch_size", default=32, type=int,
                    help="The data loader batch size.")
parser.add_argument("--train_from_scratch", default=False, action='store_true')
parser.add_argument("--tmp_err", default=0, type=float)
parser.add_argument("--non_private", default=False, action='store_true')
parser.add_argument("--num_epochs", default=3, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--beta_1", default=0.9, type=float)
parser.add_argument("--beta_2", default=0.999, type=float)
parser.add_argument("--out_dir", default='tmp_dicts/', type=str)
parser.add_argument("--data_name", choices=['qnli', 'mnli', 'qqp', 'sst2'])
# parser.add_argument("--model_name", choices=['bert_base', 'bert_large'])
parser.add_argument("--model_name", choices=['bert_base', 'bert_large'], default='bert_base')
parser.add_argument("--gamma_decay", default=1.0, type=float)
parser.add_argument("--target_epsilon", default=None, type=float)


def main():
    conf = configlib.parse()

    # TODO: change to your own key
    wandb.login(key='db16394d65b7e2b87c7f11d1aedf3c38327c6fca') # Lillian's key
    wandb.init(project='dp-adamw', group=conf.exp_group, name=conf.exp_name)

    # Load data
    if conf.data_name == 'qnli':
        dat_train = load_dataset('glue', 'qnli', split='train')
        dat_val = load_dataset('glue', 'qnli', split='validation')
        LABEL_LIST = [0, 1]
    elif conf.data_name == 'mnli':
        dat_train = load_dataset('glue', 'mnli', split='train')
        dat_val = load_dataset('glue', 'mnli', split='validation_matched')
        LABEL_LIST = [0, 1, 2]
    elif conf.data_name == 'qqp':
        dat_train = load_dataset('glue', 'qqp', split='train')
        dat_val = load_dataset('glue', 'qqp', split='validation')
        LABEL_LIST = [0, 1]
    elif conf.data_name == 'sst2':
        dat_train = load_dataset('glue', 'sst2', split='train')
        dat_val = load_dataset('glue', 'sst2', split='validation')
        LABEL_LIST = [0, 1]
    else:
        raise NotImplementedError
    num_labels = len(LABEL_LIST)

    if conf.debug:
        dat_train = dat_train[:200]
        dat_train = [dict(zip(dat_train, t)) for t in zip(*dat_train.values())]
        dat_val = dat_val[:200]
        dat_val = [dict(zip(dat_val, t)) for t in zip(*dat_val.values())]

    # prepare data
    MAX_SEQ_LENGHT = 128

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-cased",
        do_lower_case=False,
    )

    def _create_examples(df, set_type):
        """ Convert raw dataframe to a list of InputExample. Filter malformed examples
        """
        if conf.data_name == 'sst2':
            sen1 = 'sentence'
            examples = []
            for index, row in enumerate(df):
                if row['label'] not in LABEL_LIST:
                    continue
                if not isinstance(row[sen1], str):
                    continue

                guid = f"{index}-{set_type}"
                examples.append(
                    InputExample(guid=guid, text_a=row[sen1], label=row['label'])
                )
        else:
            if conf.data_name == 'qnli':
                sen1, sen2 = 'question', 'sentence'
            elif conf.data_name == 'mnli':
                sen1, sen2 = 'premise', 'hypothesis'
            elif conf.data_name == 'qqp':
                sen1, sen2 = 'question1', 'question2'
            else:
                raise NotImplementedError

            examples = []
            for index, row in enumerate(df):
                if row['label'] not in LABEL_LIST:
                    continue
                if not isinstance(row[sen1], str) or not isinstance(row[sen2], str):
                    continue

                guid = f"{index}-{set_type}"
                examples.append(
                    InputExample(guid=guid, text_a=row[sen1], text_b=row[sen2], label=row['label'])
                )

        return examples

    def _df_to_features(df, set_type):
        """ Pre-process text. This method will:
        1) tokenize inputs
        2) cut or pad each sequence to MAX_SEQ_LENGHT
        3) convert tokens into ids

        The output will contain:
        `input_ids` - padded token ids sequence
        `attention mask` - mask indicating padded tokens
        `token_type_ids` - mask indicating the split between premise and hypothesis
        `label` - label
        """
        examples = _create_examples(df, set_type)

        # backward compatibility with older transformers versions
        legacy_kwards = {}
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("2.9.0"):
            legacy_kwards = {
                "pad_on_left": False,
                "pad_token": tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                "pad_token_segment_id": 0,
            }

        return glue_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            label_list=LABEL_LIST,
            max_length=MAX_SEQ_LENGHT,
            output_mode="classification",
            **legacy_kwards,
        )

    def _features_to_dataset(features):
        """ Convert features from `_df_to_features` into a single dataset
        """
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long
        )
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )

        return dataset

    # Data processing
    train_features = _df_to_features(dat_train, "train")
    test_features = _df_to_features(dat_val, "test")

    train_dataset = _features_to_dataset(train_features)
    test_dataset = _features_to_dataset(test_features)

    # set seed
    torch.manual_seed(conf.seed)
    random.seed(conf.seed)
    np.random.seed(conf.seed)

    # load model
    if conf.model_name == 'bert_base':
        model_name = "bert-base-cased"
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased",
            config=config,
        )
        trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
        # trainable_layers = [model.bert.encoder.layer[-2], model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
        total_params = 0
        trainable_params = 0
    elif conf.model_name == 'bert_large':
        model_name = "bert-large-cased"
        config = BertConfig.from_pretrained(
            model_name,
            # num_labels=3,
            num_labels=num_labels
        )
        model = BertForSequenceClassification.from_pretrained(
            "bert-large-cased",
            config=config,
        )
        trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
        # trainable_layers = [model.bert.encoder.layer[-2], model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
        total_params = 0
        trainable_params = 0
    else:
        raise NotImplementedError

    BertLayerNorm = torch.nn.LayerNorm
    def _init_weights(module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=1.0)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()

    for layer in trainable_layers:
        if conf.train_from_scratch:
            for n, p in layer.named_modules():
                # print(n)
                _init_weights(p)
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()

    BATCH_SIZE = conf.batch_size

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

    # train
    # Move the model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set the model to train mode (HuggingFace models load in eval mode)
    model = model.train()

    LOGGING_INTERVAL = 10
    DELTA = 1 / len(
        train_dataloader)  # Parameter for privacy accounting. Probability of not achieving privacy guarantees

    def accuracy(preds, labels):
        return (preds == labels).mean()

    # define evaluation cycle
    def evaluate(model):
        model.eval()

        loss_arr = []
        accuracy_arr = []

        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]}

                outputs = model(**inputs)
                loss, logits = outputs[:2]

                preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
                labels = inputs['labels'].detach().cpu().numpy()

                loss_arr.append(loss.item())
                accuracy_arr.append(accuracy(preds, labels))

        model.train()
        return np.mean(loss_arr), np.mean(accuracy_arr)

    if conf.target_epsilon is not None:
        # privacy_accountant = RDPAccountant()
        privacy_accountant = PRVAccountant()
        NOISE_MULTIPLIER = get_noise_multiplier(
            target_epsilon=conf.target_epsilon,
            target_delta=DELTA,
            sample_rate=1 / len(train_dataloader),
            epochs=conf.num_epochs,
            accountant=privacy_accountant.mechanism(),
        )
    else:
        NOISE_MULTIPLIER = conf.dp_noise_multiplier
    MAX_GRAD_NORM = conf.dp_l2_norm_clip

    # Define optimizer
    if conf.opt_model == "adam_corr":
        optimizer = AdamCorr(
            model.parameters(), lr=conf.lr, eps=conf.eps,
            dp_batch_size=BATCH_SIZE,
            dp_noise_multiplier=NOISE_MULTIPLIER,
            dp_l2_norm_clip=MAX_GRAD_NORM,
            eps_root=conf.eps_root,
            betas=(conf.beta_1, conf.beta_2),
            gamma_decay=conf.gamma_decay,
        )
        # scheduler = ExponentialLR(optimizer, gamma=conf.gamma_decay)
    elif conf.opt_model == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, eps=conf.eps)
    elif conf.opt_model == "dp_adamw_bc":
        from adam_corr import DPAdamWBC
        optimizer = DPAdamWBC(
            model.parameters(), lr=conf.lr, eps=conf.eps,
            betas=(conf.beta_1, conf.beta_2),
            weight_decay=1e-2,
            dp_batch_size=BATCH_SIZE,
            dp_noise_multiplier=NOISE_MULTIPLIER,
            dp_l2_norm_clip=MAX_GRAD_NORM,
        )
    elif conf.opt_model == "adam":
        # optimizer = OrigAdam(model.parameters(), lr=conf.lr, eps=conf.eps, tmp_err=conf.tmp_err,
        #                      betas=(conf.beta_1, conf.beta_2),)  # if needed logging
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, eps=conf.eps, betas=(conf.beta_1, conf.beta_2))
        # scheduler = ExponentialLR(optimizer, gamma=conf.gamma_decay)
    elif conf.opt_model == "sgdm":
        # optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.beta_1)
        optimizer = MySGDM(model.parameters(), lr=conf.lr, eps=conf.eps, tmp_err=conf.tmp_err,
                             betas=(conf.beta_1, conf.beta_2),)
    elif conf.opt_model == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr)
    elif conf.opt_model == "svag":
        optimizer = SVAG(model.parameters(), lr=conf.lr)
    else:
        raise NotImplementedError
    if not conf.non_private:
        privacy_engine = PrivacyEngine()
        # model, optimizer, train_dataloader = privacy_engine.make_private(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=train_dataloader,
        #     # target_delta=DELTA,
        #     # target_epsilon=EPSILON,
        #     # epochs=EPOCHS,
        #     max_grad_norm=MAX_GRAD_NORM,
        #     noise_multiplier=NOISE_MULTIPLIER,
        #     poisson_sampling=False
        # )
        model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_delta=DELTA,
            target_epsilon=conf.target_epsilon,
            epochs=conf.num_epochs,
            max_grad_norm=MAX_GRAD_NORM,
            # noise_multiplier=NOISE_MULTIPLIER,
            poisson_sampling=False
        )

    print('Training')
    for epoch in range(1, conf.num_epochs + 1):
        losses = []

        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            actural_batch_size = len(batch[3])
            # print('batch_size: {}'.format(actural_batch_size))

            outputs = model(**inputs)  # output = loss, logits, hidden_states, attentions

            loss = outputs[0]
            loss.backward()
            losses.append(loss.item())

            param_grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    param_grad_norms.append(torch.linalg.norm(param.grad))
            clean_grad_norm = torch.linalg.norm(torch.stack(param_grad_norms))

            if conf.opt_model == 'adam_corr':
                _, logging_stats, hist_dict, summary_stats_dict, dummy_step, cur_gamma = optimizer.step()
                # scheduler.step()
                # cur_lr = scheduler.get_last_lr()[0]
                if hist_dict:
                    if not os.path.exists(conf.out_dir):
                        os.makedirs(conf.out_dir)
                    pickle.dump(hist_dict, open(os.path.join(conf.out_dir, 'hist_step_{}.pkl'.format(dummy_step)), 'wb'))
                if summary_stats_dict:
                    if not os.path.exists(conf.out_dir):
                        os.makedirs(conf.out_dir)
                    pickle.dump(summary_stats_dict, open(os.path.join(conf.out_dir, 'summary_stats_step_{}.pkl'.format(dummy_step)), 'wb'))
            else:
                optimizer.step()
                logging_stats = {}
                cur_gamma = -1
                cur_lr = conf.lr

            if not conf.non_private:
                param_grad_norms = []
                param_clipped_grad_norms = []
                for param in model.parameters():
                    if param.grad is not None:
                        param_grad_norms.append(torch.linalg.norm(param.grad))
                        param_clipped_grad_norms.append(torch.linalg.norm(param.summed_grad))
                private_grad_norm = torch.linalg.norm(torch.stack(param_grad_norms))
                clipped_grad_norm = torch.linalg.norm(torch.stack(param_clipped_grad_norms))
            else:
                private_grad_norm, clipped_grad_norm = torch.nan, torch.nan

            train_log = {
                'epoch': epoch, 'step': step, 'train_loss': np.mean(losses),
                'clean_grad_norm': clean_grad_norm, 'private_grad_norm': private_grad_norm,
                'clipped_grad_norm': clipped_grad_norm,
            }
            train_log.update(logging_stats)
            wandb.log(train_log)

            if step > 0 and step % LOGGING_INTERVAL == 0:
                train_loss = np.mean(losses)

                if not conf.non_private:
                    eps = privacy_engine.get_epsilon(DELTA)
                else:
                    eps = torch.nan

                eval_loss, eval_accuracy = evaluate(model)

                print(
                    f"Epoch: {epoch} | "
                    f"Step: {step} | "
                    f"Train loss: {train_loss:.3f} | "
                    f"Eval loss: {eval_loss:.3f} | "
                    f"Eval accuracy: {eval_accuracy:.3f} | "
                    f"ɛ: {eps:.2f}"
                )

                test_log = {
                    'eval_loss': eval_loss, 'eval_accuracy': eval_accuracy, 'eps': eps,
                    # 'actural_batch_size': actural_batch_size,
                }
                wandb.log(test_log)


if __name__ == "__main__":
    main()
