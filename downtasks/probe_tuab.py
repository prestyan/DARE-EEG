import os
import torch
import time
import json
import datetime
import torch.nn as nn
import pytorch_lightning as pl
from Modules.utils import get_metrics
from Modules.model import DARE_EEG
from Modules.dataset import generate_dataload, generate_dataset_tuab
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from configs import *
from Modules.utils import seed_torch, get_world_size, get_rank, cosine_scheduler, init_distributed_mode
from Modules.utils import TensorboardLogger, save_model, is_main_process, create_ds_config
from Modules.utils import NativeScalerWithGradNormCount as NativeScaler
import math
import numpy as np
import torch.utils.data
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from engine_for_training import train_one_epoch, evaluate
from collections import Counter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == "__main__":
    seed_torch(2025)

    encoder_config = get_base_config(**(BASE_MODELS_CONFIGS[tag]))
    args, ds_init = get_args()
    init_distributed_mode(args)

    if ds_init is not None:
        create_ds_config(args)


    dataset = args.dataset
    if dataset == 'TUAB':
        args.num_classes = 1
    else:
        args.num_classes = 6

    model_config = {
        # ConvHead parameter
        "in_channels": args.in_channels,
        "conv_out_channels": args.conv_out_channels,
        "in_time_length": args.in_time_length,
        "conv_out_time_length": 2000, # not recommend to change
        "conv_layers": 2,
        # Encoder parameter
        "is_eval": args.eval,
        "encoder_path": '../pretrain/logs/checkpoints/DARE-EEG_3_base2@epoch=199-valid_loss=0.5981.ckpt',
        "complete_model_path": './logs/checkpoints/checkpoint-ours_base_TUAB_best_3.pth',
        "models_configs": encoder_config,
        "freeze_encoder": False,
        # MLPClassifier parameter
        "N": 15,
        "embed_num": encoder_config['encoder']['embed_num'],
        "embed_dim": encoder_config['encoder']['embed_dim'],
        "num_classes": args.num_classes,
        "hidden_dims": [256, 128],
        "dropout": args.dropout,
        "pooling_method": 'mean',
    }

    max_epochs = 50
    learning_rate = 5e-4
    # data_path = '../data/downtasks/TUAB/tuh_eeg_abnormal/edf/processed'
    data_path = args.data_path_1
    model_save_name = f"DAREEEG_large_{dataset}"
    device = torch.device(args.device)
    
    model = DARE_EEG(**model_config)
    model.to(device)

    os.makedirs("./logs", exist_ok=True)
    train_dataset, valid_dataset, test_dataset, ch_names, metrics = generate_dataset_tuab(dataset=dataset,
                                                                                          data_path=data_path,)

    num_tasks = get_world_size()
    global_rank = get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    # We recommend not to distribute eval and test dataset
    sampler_val = torch.utils.data.SequentialSampler(valid_dataset)
    sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    if type(test_dataset) == list:
        test_loader = [torch.utils.data.DataLoader(
            test_dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        ) for dataset, sampler in zip(test_dataset, sampler_test)]
    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    steps_per_epoch = math.ceil(len(train_loader))
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * get_world_size()
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % 1)
    print("Number of training examples = %d" % len(train_dataset))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.max_epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.max_epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    all_labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        label = sample[-1]
        all_labels.append(int(label))

    label_counts = Counter(all_labels)
    print("Class counts:", label_counts)

    num_classes = args.num_classes

    if args.num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        class_counts = np.zeros(num_classes, dtype=np.float32)
        for c, cnt in label_counts.items():
            if 0 <= c < num_classes:
                class_counts[c] = cnt

        ##### Label weight smoothing #####
        # IMPORTANT: If you are using the TUAB dataset, we recommend removing this.
        # Simple inverse frequency weighting: weight_c ∝ 1 / count_c
        total = class_counts.sum()
        freq = class_counts / (total + 1e-6)
        freq_mean = freq.mean()

        gamma = 0.5
        raw_weights = (freq_mean / (freq + 1e-6)) ** gamma  # Minority class > 1, Majority class < 1
        class_weights = raw_weights / raw_weights.mean()
        print("Using class weights:", class_weights)
        class_weights = torch.tensor(
            class_weights,
            device=device,
            dtype=next(model.parameters()).dtype                                                                                                                                                                                                                                                                                                                                           
        )
        # criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing, weight=class_weights)
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,           
            label_smoothing=args.smoothing
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.eval:
        balanced_accuracy = []
        accuracy = []
        pr_auc = []
        roc_auc = []
        #cfor data_loader in valid_loader:
        test_stats = evaluate(test_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                                is_binary=(args.num_classes == 1))
        accuracy.append(test_stats['accuracy'])
        balanced_accuracy.append(test_stats['balanced_accuracy'])
        pr_auc.append(test_stats['pr_auc'])
        roc_auc.append(test_stats['roc_auc'])
        print(
            f"#####Test Results#####\n"
            f"Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}, PR AUC: {np.mean(pr_auc)} {np.std(pr_auc)}, ROC AUC: {np.mean(roc_auc)} {np.std(roc_auc)}")
        exit(0)

    print(f"Start training for {args.max_epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    for epoch in range(args.start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer,
            device, epoch, loss_scaler, args.clip_grad, None,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=1,
            ch_names=ch_names, is_binary=args.num_classes == 1
        )

        if valid_loader is not None:
            val_stats = evaluate(valid_loader, model, device, header='Val:', ch_names=ch_names, metrics=metrics,
                                 is_binary=args.num_classes == 1)
            print(f"Accuracy of the network on the {len(valid_dataset)} val EEG: {val_stats['balanced_accuracy']:.2f}%")
            test_stats = evaluate(test_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                                  is_binary=args.num_classes == 1)
            print(f"Accuracy of the network on the {len(test_dataset)} test EEG: {test_stats['balanced_accuracy']:.2f}%")

            if max_accuracy < val_stats["balanced_accuracy"]:
                max_accuracy = val_stats["balanced_accuracy"]
                if args.output_dir and args.save_ckpt:
                    save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=None)
                max_accuracy_test = test_stats["balanced_accuracy"]

            print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="val", step=epoch)
                for key, value in test_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="test", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="test", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="test", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="test", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="test", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    

