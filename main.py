import os
import shutil
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model import get_model
from dataset import load_datasets
from utils import (
    seed_all,
    save_metric,
    save_best_metric,
    save_pickle,
)
from meter import AverageMeter, ProgressMeter, accuracy


def train(
    method,
    train_loader,
    models,
    optimizers,
    epoch,
    device,
):
    loss_meters = []
    top1_meters = []
    top5_meters = []
    inds_updates = [[] for _ in range(len(models))]

    show_logs = []
    for i in range(len(models)):
        loss_meter = AverageMeter(f"Loss{i}", ":.4e")
        top1_meter = AverageMeter(f"Acc{i}@1", ":6.2f")
        top5_meter = AverageMeter(f"Acc{i}@5", ":6.2f")
        loss_meters.append(loss_meter)
        top1_meters.append(top1_meter)
        top5_meters.append(top5_meter)
        show_logs += [loss_meter, top1_meter, top5_meter]
    progress = ProgressMeter(
        len(train_loader), show_logs, prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    for i in range(len(models)):
        models[i].train()

    for i, (images, target, indexes) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.to(device)
            target = target.to(device)

        outputs = []
        for m in range(len(models)):
            output = models[m](images)
            outputs.append(output)

        # calculate loss and selected index
        ind = indexes.cpu().numpy().transpose()
        losses, ind_updates = method.loss(outputs, target, epoch=epoch, ind=ind)

        if None in losses or any(~torch.isfinite(torch.tensor(losses))):
            continue

        # compute gradient and do BP
        for loss, optimizer, model in zip(losses, optimizers, models):
            optimizer.zero_grad()
            loss.backward()
            if hasattr(method, "post_backward_hook"):
                method.post_backward_hook(model)
            optimizer.step()

        # measure accuracy and record loss
        for m in range(len(models)):
            acc1, acc5 = accuracy(outputs[m], target, topk=(1, 5))

            top1_meters[m].update(acc1[0].item(), images.size(0))
            top5_meters[m].update(acc5[0].item(), images.size(0))
            if len(ind_updates[m]) > 0:
                loss_meters[m].update(losses[m].item(), len(ind_updates[m]))
                inds_updates[m] += indexes[ind_updates[m]].numpy().tolist()
            else:
                loss_meters[m].update(losses[m].item(), images.size(0))

        if i % 100 == 0:
            progress.display(i)

    loss_avgs = [loss_meter.avg for loss_meter in loss_meters]
    top1_avgs = [top1_meter.avg for top1_meter in top1_meters]
    top5_avgs = [top5_meter.avg for top5_meter in top1_meters]

    return loss_avgs, top1_avgs, top5_avgs, inds_updates


def validate(val_loader, model, criterion, device, is_test):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    prefix = "Test: " if is_test else "Validation: "
    progress = ProgressMeter(len(val_loader), [losses, top1, top5], prefix=prefix)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.to(device)
                target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            if i % 100 == 0:
                progress.display(i)

    return losses.avg, top1.avg, top5.avg


def run(
    method,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    models,
    optimizers,
    schedulers,
    criterion,
    epochs,
    device,
    writers,
    train_noise_ind,
):
    metrics = [[] for _ in range(len(models))]
    selected_idxs = [[] for _ in range(len(models))]

    best_valid_top1s = [0 for _ in range(len(models))]
    best_epochs = [0 for _ in range(len(models))]

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        if hasattr(method, "pre_epoch_hook"):
            loss, cum_loss, objective, inds_updates, selected_dataloader = method.pre_epoch_hook(train_dataloader, model, device)
        else:
            selected_dataloader = train_dataloader

        train_loss_avgs, train_top1_avgs, train_top5_avgs, inds_updates = train(
            method,
            selected_dataloader,
            models,
            optimizers,
            epoch,
            device,
        )

        epoch_time = time.time() - start_time

        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()

        for i in range(len(models)):
            writers[i].add_scalar("Train/Loss", train_loss_avgs[i], epoch)
            writers[i].add_scalar("Train/Top1", train_top1_avgs[i], epoch)
            writers[i].add_scalar("Train/Top5", train_top5_avgs[i], epoch)

            test_loss, test_top1, test_top5 = validate(
                test_dataloader, models[i], criterion, device, is_test=True
            )

            writers[i].add_scalar("Test/Loss", test_loss, epoch)
            writers[i].add_scalar("Test/Top1", test_top1, epoch)
            writers[i].add_scalar("Test/Top5", test_top5, epoch)

            if valid_dataloader is not None:
                valid_loss, valid_top1, valid_top5 = validate(
                    valid_dataloader, models[i], criterion, device, is_test=False
                )

                writers[i].add_scalar("Valid/Loss", valid_loss, epoch)
                writers[i].add_scalar("Valid/Top1", valid_top1, epoch)
                writers[i].add_scalar("Valid/Top5", valid_top5, epoch)
            else:
                valid_loss = 0
                valid_top1 = 0
                valid_top5 = 0

            metric = {
                "epoch": epoch,
                "train_loss": train_loss_avgs[i],
                "train_top1": train_top1_avgs[i],
                "train_top5": train_top5_avgs[i],
                "valid_loss": valid_loss,
                "valid_top1": valid_top1,
                "valid_top5": valid_top5,
                "test_loss": test_loss,
                "test_top1": test_top1,
                "test_top5": test_top5,
                "epoch_time": epoch_time,
            }
            metrics[i].append(metric)

            if hasattr(method, "post_iter_hook"):
                loss_path = os.path.join(writers[i].log_dir, f"loss_{epoch}.npy")
                np.save(loss_path, loss)

                cum_loss_path = os.path.join(
                    writers[i].log_dir, f"cum_loss_{epoch}.npy"
                )
                np.save(cum_loss_path, cum_loss)

                objective_path = os.path.join(
                    writers[i].log_dir, f"objective_{epoch}.npy"
                )
                np.save(objective_path, objective)

            if len(inds_updates[i]) > 0:
                clean_selected_ind = np.setdiff1d(inds_updates[i], train_noise_ind)
                label_precision = len(clean_selected_ind) / len(inds_updates[i])
                label_recall = len(clean_selected_ind) / (
                    len(train_dataloader.dataset) - len(train_noise_ind)
                )

                writers[i].add_scalar("Select/Label Precision", label_precision, epoch)
                writers[i].add_scalar("Select/Label Recall", label_recall, epoch)
                selected_idxs[i].append(inds_updates[i])

            if best_valid_top1s[i] <= valid_top1:
                best_valid_top1s[i] = valid_top1
                best_epochs[i] = epoch
                save_path = os.path.join(writers[i].log_dir, "best_model.pt")
                torch.save(models[i].state_dict(), save_path)

    return metrics, selected_idxs, best_epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="logs")

    parser.add_argument("--dataset_name", type=str, default="mnist")
    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--noise_type", type=str, default="symmetric")
    parser.add_argument("--noise_ratio", type=float, default=0.8)
    parser.add_argument("--noise_classes", type=list, default=[])
    parser.add_argument('--use_pretrained', action='store_true')

    parser.add_argument("--method_name", type=str, default="ftl")

    # ours
    parser.add_argument("--k_ratio", type=float, default=0.2)
    parser.add_argument("--use_total", type=bool, default=True)

    # precision
    parser.add_argument("--precision", type=float, default=0.2)

    # jocor
    parser.add_argument("--co_lambda", type=float, default=0.9)

    args = parser.parse_args()

    #region general setting
    seed_all(args.seed)
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    device = f"cuda:{args.gpu}"
    #endregion

    #region dataset-specific setting
    if args.dataset_name == "mnist":
        epochs = 30
        batch_size = 128
        model_name = "lenet"
        step_size = 5
        gamma = 0.1
        weight_decay = 1e-4
    elif args.dataset_name in ["cifar10", "deepmind-cifar10"]:
        epochs = 200
        batch_size = 512
        model_name = "resnet26"
        step_size = 40
        gamma = 0.1
        weight_decay = 1e-5
    elif args.dataset_name in ["cifar100", "deepmind-cifar100", "tiny-imagenet"]:
        epochs = 120
        batch_size = 512
        model_name = "preactresnet56"
        step_size = 40
        gamma = 0.1
        weight_decay = 1e-5
    elif args.dataset_name == "clothing1m":
        epochs = 10
        batch_size = 32
        model_name = "resnet50"
        step_size = 5
        gamma = 0.1
        weight_decay = 1e-5
    #endregion

    #region prepare dataset
    train_dataset, valid_dataset, test_dataset, train_noise_ind = load_datasets(
        args.dataset_name,
        args.dataset_path,
        args.train_ratio,
        args.noise_type,
        args.noise_ratio,
        args.noise_classes,
        args.seed,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=16
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=16
    )
    if valid_dataset is not None:
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16
        )
    else:
        valid_dataloader = None

    if args.noise_type not in ["symmetric", "asymmetric"]:
        dataset_log_dir = args.dataset_name
    else:
        if len(args.noise_classes) > 0:
            dataset_log_dir = os.path.join(
                args.dataset_name,
                f"delete({args.noise_classes})",
            )
        else:
            dataset_log_dir = os.path.join(
                args.dataset_name,
                f"{args.noise_type}({args.noise_ratio * 100}%)",
            )

    if args.train_ratio < 1:
        dataset_log_dir += f"-Train({args.train_ratio * 100}%)"
    
    #endregion

    #region prepare methods
    start_time = time.time()
    if args.method_name == "standard":
        from methods.standard import Standard

        method = Standard(criterion)
    elif args.method_name == "f-correction":
        from methods.f_correction import F_correction

        method = F_correction(
            args.dataset_name,
            args.log_dir,
            dataset_log_dir,
            model_name,
            train_dataloader,
            args.seed,
            device,
        )
    elif args.method_name == "decouple":
        from methods.decouple import Decouple

        method = Decouple(criterion)
    elif args.method_name == "co-teaching":
        from methods.co_teaching import Co_teaching

        method = Co_teaching(args.dataset_name, args.noise_ratio, epochs)
    elif args.method_name == "co-teaching+":
        from methods.co_teaching_plus import Co_teaching_plus

        method = Co_teaching_plus(args.dataset_name, args.noise_ratio, epochs)
    elif args.method_name == "jocor":
        from methods.jocor import JoCoR

        method = JoCoR(args.dataset_name, args.noise_ratio, epochs, args.co_lambda)
    elif args.method_name == "cdr":
        clip = 1 - args.noise_ratio
        from methods.cdr import CDR

        method = CDR(criterion, clip)
    elif args.method_name == "tv":
        transition_type = "dirichlet"
        regularization_type = "tv"
        from methods.tv import TV

        method = TV(
            train_dataloader,
            epochs,
            device,
            args.dataset_name,
            transition_type,
            regularization_type,
        )
    elif args.method_name == "class2simi":
        loss_type = "forward"
        from methods.class2simi import Class2Simi

        method = Class2Simi(
            loss_type,
            args.dataset_name,
            args.log_dir,
            dataset_log_dir,
            model_name,
            train_dataloader,
            args.seed,
            device,
        )
    elif args.method_name == "taks":
        from methods.taks import TAkS

        method = TAkS(
            criterion,
            train_dataset,
            batch_size,
            epochs,
            args.k_ratio,
            args.use_total,
        )
    elif args.method_name == "precision":
        from methods.precision import Precision

        method = Precision(
            criterion,
            train_dataset,
            batch_size,
            epochs,
            args.k_ratio,
            args.precision,
            train_noise_ind,
        )
    else:
        raise NameError(f"Invalid method_name: {args.method_name}")

    preprocessing_time = time.time() - start_time
    #endregion

    #region prepare models
    models = []
    for i in range(method.num_models):
        model = get_model(model_name, args.dataset_name, device)
        if args.use_pretrained and args.method_name != "standard":
            root_log_dir = os.path.join(
                args.log_dir,
                dataset_log_dir,
                model_name,
                "Standard",
                str(args.seed),
            )
            standard_path = os.path.join(
                root_log_dir,
                "model0",
                "best_model.pt",
            )
            model.load_state_dict(torch.load(standard_path))
        models.append(model)
    #endregion

    #region prepare optimizers
    optimizers = []
    schedulers = []
    if args.method_name == "jocor":
        parameters = []
        for i in range(method.num_models):
            parameters += list(models[i].parameters())
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        optimizers.append(optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        schedulers.append(scheduler)
    else:
        for i in range(method.num_models):
            optimizer = torch.optim.Adam(models[i].parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizers.append(optimizer)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
            schedulers.append(scheduler)
    #endregion

    #region prepare logger
    if args.method_name != "standard":
        method_dir_name = f"{method.name}_pretrained" if args.use_pretrained else f"{method.name}_scratch"
    else:
        method_dir_name = method.name
    root_log_dir = os.path.join(
        args.log_dir, dataset_log_dir, model_name, method_dir_name, str(args.seed)
    )
    if os.path.exists(root_log_dir):
        for i in range(len(models)):
            metric_path = os.path.join(root_log_dir, f"model{i}", "metric.csv")
            if not os.path.exists(metric_path):
                print(f"Model {i} of {args.method_name} does not exists")
                shutil.rmtree(root_log_dir)
                break
        else:
            exit()

    writers = []
    for i in range(len(models)):
        log_dir = os.path.join(root_log_dir, f"model{i}")
        writer = SummaryWriter(log_dir=log_dir)
        writers.append(writer)
    #endregion

    # run
    metrics, selected_idxs, best_epochs = run(
        method,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        models,
        optimizers,
        schedulers,
        criterion,
        epochs,
        device,
        writers,
        train_noise_ind,
    )
    writer.close()

    # log preprocessing time
    for model_metrics in metrics:
        model_metrics.insert(
            0,
            {
                "epoch": 0,
                "train_loss": 0,
                "train_top1": 0,
                "train_top5": 0,
                "valid_loss": 0,
                "valid_top1": 0,
                "valid_top5": 0,
                "test_loss": 0,
                "test_top1": 0,
                "test_top5": 0,
                "epoch_time": preprocessing_time,
            },
        )

    # save metrics
    for i in range(len(models)):
        metric_path = os.path.join(writers[i].log_dir, "metric.csv")
        save_metric(metrics[i], metric_path)

        best_metric_path = os.path.join(writers[i].log_dir, "best_metric.csv")
        save_best_metric(metrics[i][best_epochs[i]], best_metric_path)

        if len(selected_idxs[i]) > 0:
            selected_idx_path = os.path.join(writers[i].log_dir, "selected_idx.pkl")
            save_pickle(selected_idxs[i], selected_idx_path)
