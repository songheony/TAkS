import os
import shutil
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from player import Player, PrecisionSelector
from model import get_model
from loss import (
    loss_general,
    loss_forward,
    loss_decouple,
    loss_coteaching,
    loss_coteaching_plus,
    loss_jocor,
)
from dataset import load_datasets, selected_loader
from utils import (
    seed_all,
    AverageMeter,
    ProgressMeter,
    accuracy,
    predict,
    NoiseEstimator,
    save_metric,
    save_best_metric,
    save_pickle,
)


def train(
    train_loader, models, optimizers, criterion, epoch, device, method_name, **kwargs,
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
        len(train_loader), show_logs, prefix="Epoch: [{}]".format(epoch),
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
        if method_name in ["ours", "ftl", "greedy", "precision", "itlm"]:
            ind = indexes.cpu().numpy()
            losses, ind_updates = loss_general(outputs, target, criterion)
        elif method_name == "f-correction":
            losses, ind_updates = loss_forward(outputs, target, kwargs["P"])
        elif method_name == "decouple":
            losses, ind_updates = loss_decouple(outputs, target, criterion)
        elif method_name == "co-teaching":
            losses, ind_updates = loss_coteaching(
                outputs, target, kwargs["rate_schedule"][epoch]
            )
        elif method_name == "co-teaching+":
            ind = indexes.cpu().numpy().transpose()
            if epoch < kwargs["init_epoch"]:
                losses, ind_updates = loss_coteaching(
                    outputs, target, kwargs["rate_schedule"][epoch]
                )
            else:
                losses, ind_updates = loss_coteaching_plus(
                    outputs, target, kwargs["rate_schedule"][epoch], ind, epoch * i,
                )
        elif method_name == "jocor":
            losses, ind_updates = loss_jocor(
                outputs, target, kwargs["rate_schedule"][epoch], kwargs["co_lambda"]
            )
        else:
            losses, ind_updates = loss_general(outputs, target, criterion)

        if None in losses or any(~torch.isfinite(torch.tensor(losses))):
            continue

        # compute gradient and do BP
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward()
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
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    models,
    optimizers,
    adjust_learning_rate,
    criterion,
    epochs,
    device,
    writers,
    method_name,
    train_noise_ind,
    **kwargs,
):
    metrics = [[] for _ in range(len(models))]
    selected_idxs = [[] for _ in range(len(models))]

    best_valid_top1s = [0 for _ in range(len(models))]
    best_epochs = [0 for _ in range(len(models))]

    for epoch in range(1, epochs):
        start_time = time.time()

        if method_name in ["ours", "ftl", "greedy", "precision"]:
            indices = np.where(kwargs["player"].w == 1)[0]
            selected_dataloader = selected_loader(train_dataloader, indices)
        elif method_name == "itlm":
            outputs, preds, targets = predict(
                kwargs["fixed_train_dataloader"], models[0], device, softmax=False
            )
            objective = kwargs["loss_fn"](outputs, targets).cpu().numpy()
            indices = np.argpartition(objective, kwargs["k"])[:kwargs["k"]]
            selected_dataloader = selected_loader(train_dataloader, indices)
        else:
            selected_dataloader = train_dataloader

        train_loss_avgs, train_top1_avgs, train_top5_avgs, inds_updates = train(
            selected_dataloader,
            models,
            optimizers,
            criterion,
            epoch,
            device,
            method_name,
            **kwargs,
        )

        if method_name in ["ours", "ftl", "greedy", "precision"]:
            outputs, preds, targets = predict(
                kwargs["fixed_train_dataloader"], models[0], device
            )
            loss, cum_loss, objective = kwargs["player"].update(outputs, preds, targets)
            inds_updates = [indices]
        elif method_name == "itlm":
            inds_updates = [indices] 
        epoch_time = time.time() - start_time

        for optimizer in optimizers:
            adjust_learning_rate(optimizer, epoch)

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

            if method_name in ["ours", "ftl", "greedy", "precision"]:
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
                best_epochs[i] = epoch - 1
                save_path = os.path.join(writers[i].log_dir, "best_model.pt")
                torch.save(models[i].state_dict(), save_path)

    return metrics, selected_idxs, best_epochs


def config_ours(train_dataset, batch_size, epochs, k_ratio, lr_ratio, use_total=True):
    fixed_train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    n_experts = len(train_dataset)
    if 0 < k_ratio <= 1:
        k = int(n_experts * k_ratio)
        player = Player(n_experts, k, epochs, lr_ratio, use_total=use_total)
    else:
        raise ValueError("k_ratio should be less than 1 and greater than 0")
    return player, fixed_train_dataloader


def config_precision(train_dataset, train_noise_ind, k_ratio, precision):
    n_experts = len(train_dataset)
    k = int(n_experts * k_ratio)
    player = PrecisionSelector(n_experts, k, precision, train_noise_ind)
    return player, fixed_train_dataloader


def config_f_correction(
    dataset_name, log_dir, dataset_log_dir, model_name, dataloader, seed, device
):
    if dataset_name == "cifar100":
        filter_outlier = False
    else:
        filter_outlier = True

    if dataset_name == "clothing1m":
        root_log_dir = os.path.join(
            log_dir, dataset_log_dir, model_name, "Standard", str(seed)
        )
    else:
        root_log_dir = os.path.join(
            log_dir, dataset_log_dir, model_name, "Standard(Train-80.0%)", str(seed)
        )

    standard_path = os.path.join(root_log_dir, "model0", "best_model.pt",)
    classifier = get_model(model_name, dataset_name).to(device)
    classifier.load_state_dict(torch.load(standard_path))
    classifier.eval()
    est = NoiseEstimator(
        classifier=classifier, alpha=0.0, filter_outlier=filter_outlier
    )
    est.fit(dataloader, device)
    P_est = torch.tensor(est.predict().copy(), dtype=torch.float).to(device)
    del est
    del classifier

    return P_est


def config_co_teaching(dataset_name, forget_rate, epochs):
    exponent = 1
    if dataset_name in ["mnist", "cifar10", "cifar100", "tiny-imagenet"]:
        num_gradual = 10
    elif dataset_name == "clothing1m":
        num_gradual = 5

    rate_schedule = np.ones(epochs) * forget_rate
    rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)
    return rate_schedule


def config_co_teaching_plus(dataset_name):
    if dataset_name == "mnist":
        init_epoch = 0
    elif dataset_name == "cifar10":
        init_epoch = 20
    elif dataset_name == "cifar100":
        init_epoch = 5
    elif dataset_name == "tiny-imagenet":
        init_epoch = 100
    else:
        init_epoch = 5

    return init_epoch


def config_itlm(train_dataset, batch_size, forget_rate):
    fixed_train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    k = int(len(train_dataset) * (1 - forget_rate))
    loss_fn = nn.CrossEntropyLoss(reduce=False, reduction='none')
    return k, loss_fn, fixed_train_dataloader


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

    parser.add_argument("--method_name", type=str, default="ftl")

    # ours
    parser.add_argument("--k_ratio", type=float, default=0.2)
    parser.add_argument("--lr_ratio", type=float, default=1e-3)

    # precision
    parser.add_argument("--precision", type=float, default=0.2)

    # jocor
    parser.add_argument("--forget_rate", type=float, default=0.2)
    parser.add_argument("--co_lambda", type=float, default=0.9)

    args = parser.parse_args()

    seed_all(args.seed)

    device = f"cuda:{args.gpu}"

    if args.dataset_name in ["mnist", "cifar10", "cifar100", "tiny-imagenet"]:
        epochs = 201
        epoch_decay_start = 80
        batch_size = 128
        learning_rate = 1e-3

        mom1 = 0.9
        mom2 = 0.1
        alpha_plan = [learning_rate] * epochs
        beta1_plan = [mom1] * epochs
        for i in range(epoch_decay_start, epochs):
            alpha_plan[i] = (
                float(epochs - i) / (epochs - epoch_decay_start) * learning_rate
            )
            beta1_plan[i] = mom2

        def adjust_learning_rate(optimizer, epoch):
            for param_group in optimizer.param_groups:
                param_group["lr"] = alpha_plan[epoch]
                param_group["betas"] = (beta1_plan[epoch], 0.999)

    elif args.dataset_name == "clothing1m":
        epochs = 16
        batch_size = 64
        learning_rate = 8e-4

        def adjust_learning_rate(optimizer, epoch):
            for param_group in optimizer.param_groups:
                if epoch < 5:
                    param_group["lr"] = 8e-4
                elif epoch < 10:
                    param_group["lr"] = 5e-4
                elif epoch < 15:
                    param_group["lr"] = 5e-5

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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    if valid_dataset is not None:
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8
        )
    else:
        valid_dataloader = None

    if args.dataset_name == "clothing1m":
        dataset_log_dir = args.dataset_name
    else:
        if len(args.noise_classes) > 0:
            dataset_log_dir = os.path.join(
                args.dataset_name, f"{args.noise_type}-{args.noise_classes}",
            )
        else:
            dataset_log_dir = os.path.join(
                args.dataset_name, f"{args.noise_type}-{args.noise_ratio * 100}%",
            )

    model_name = "jocor_model"
    kwargs = {}
    if args.method_name == "standard":
        if args.train_ratio == 1:
            algorithm_name = "Standard"
        else:
            algorithm_name = f"Standard(Train-{args.train_ratio * 100}%)"
        model = get_model(model_name, args.dataset_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        models = [model]
        optimizers = [optimizer]
    elif args.method_name == "ours":
        algorithm_name = f"Ours(K_ratio-{args.k_ratio*100}%,Lr-{args.lr_ratio})"
        model = get_model(model_name, args.dataset_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        player, fixed_train_dataloader = config_ours(
            train_dataset,
            batch_size,
            epochs,
            args.k_ratio,
            args.lr_ratio,
        )

        models = [model]
        optimizers = [optimizer]
        kwargs = {
            "player": player,
            "fixed_train_dataloader": fixed_train_dataloader,
        }
    elif args.method_name == "ftl":
        algorithm_name = f"FTL(K_ratio-{args.k_ratio*100}%)"
        model = get_model(model_name, args.dataset_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        player, fixed_train_dataloader = config_ours(
            train_dataset,
            batch_size,
            epochs,
            args.k_ratio,
            0,
        )

        models = [model]
        optimizers = [optimizer]
        kwargs = {
            "player": player,
            "fixed_train_dataloader": fixed_train_dataloader,
        }
    elif args.method_name == "greedy":
        algorithm_name = f"Greedy(K_ratio-{args.k_ratio*100}%)"
        model = get_model(model_name, args.dataset_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        player, fixed_train_dataloader = config_ours(
            train_dataset,
            batch_size,
            epochs,
            args.k_ratio,
            0,
            use_total=False,
        )

        models = [model]
        optimizers = [optimizer]
        kwargs = {
            "player": player,
            "fixed_train_dataloader": fixed_train_dataloader,
        }
    elif args.method_name == "precision":
        algorithm_name = (
            f"Precision(K_ratio-{args.k_ratio*100}%,Precision-{args.precision})"
        )
        model = get_model(model_name, args.dataset_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        _, fixed_train_dataloader = config_ours(
            train_dataset, batch_size, epochs, args.k_ratio, 0
        )

        player = config_precision(train_dataset, train_noise_ind, args.k_ratio, args.precision)

        models = [model]
        optimizers = [optimizer]
        kwargs = {
            "player": player,
            "fixed_train_dataloader": fixed_train_dataloader,
        }
    elif args.method_name == "f-correction":
        algorithm_name = "F-correction"
        model = get_model(model_name, args.dataset_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        P_est = config_f_correction(
            args.dataset_name,
            args.log_dir,
            dataset_log_dir,
            model_name,
            train_dataloader,
            args.seed,
            device,
        )

        models = [model]
        optimizers = [optimizer]
        kwargs = {"P": P_est}
    elif args.method_name == "decouple":
        algorithm_name = "Decouple"
        model1 = get_model(model_name, args.dataset_name).to(device)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
        model2 = get_model(model_name, args.dataset_name).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
        models = [model1, model2]
        optimizers = [optimizer1, optimizer2]
    elif args.method_name == "co-teaching":
        algorithm_name = f"Co-teaching(Forget-{args.forget_rate * 100}%)"
        model1 = get_model(model_name, args.dataset_name).to(device)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
        model2 = get_model(model_name, args.dataset_name).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

        rate_schedule = config_co_teaching(args.dataset_name, args.forget_rate, epochs)

        models = [model1, model2]
        optimizers = [optimizer1, optimizer2]
        kwargs = {"rate_schedule": rate_schedule}
    elif args.method_name == "co-teaching+":
        algorithm_name = f"Co-teaching+(Forget-{args.forget_rate * 100}%)"
        model1 = get_model(model_name, args.dataset_name).to(device)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
        model2 = get_model(model_name, args.dataset_name).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

        rate_schedule = config_co_teaching(args.dataset_name, args.forget_rate, epochs)
        init_epoch = config_co_teaching_plus(args.dataset_name)

        models = [model1, model2]
        optimizers = [optimizer1, optimizer2]
        kwargs = {"rate_schedule": rate_schedule, "init_epoch": init_epoch}
    elif args.method_name == "jocor":
        algorithm_name = (
            f"JoCoR(Forget-{args.forget_rate * 100}%,Lambda-{args.co_lambda})"
        )
        model1 = get_model(model_name, args.dataset_name).to(device)
        model2 = get_model(model_name, args.dataset_name).to(device)
        optimizer = torch.optim.Adam(
            list(model1.parameters()) + list(model2.parameters()), lr=learning_rate
        )

        rate_schedule = config_co_teaching(args.dataset_name, args.forget_rate, epochs)

        models = [model1, model2]
        optimizers = [optimizer]
        kwargs = {"rate_schedule": rate_schedule, "co_lambda": args.co_lambda}
    elif args.method_name == "itlm":
        algorithm_name = f"ITLM(Forget_ratio-{args.forget_rate*100}%)"
        model = get_model(model_name, args.dataset_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        k, loss_fn, fixed_train_dataloader = config_itlm(
            train_dataset,
            batch_size,
            args.forget_rate,
        )

        models = [model]
        optimizers = [optimizer]
        kwargs = {
            "k": k,
            "loss_fn": loss_fn,
            "fixed_train_dataloader": fixed_train_dataloader,
        }

    criterion = nn.CrossEntropyLoss()

    root_log_dir = os.path.join(
        args.log_dir, dataset_log_dir, model_name, algorithm_name, str(args.seed)
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

    metrics, selected_idxs, best_epochs = run(
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        models,
        optimizers,
        adjust_learning_rate,
        criterion,
        epochs,
        device,
        writers,
        args.method_name,
        train_noise_ind,
        **kwargs,
    )
    writer.close()

    for i in range(len(models)):
        metric_path = os.path.join(writers[i].log_dir, "metric.csv")
        save_metric(metrics[i], metric_path)

        best_metric_path = os.path.join(writers[i].log_dir, "best_metric.csv")
        save_best_metric(metrics[i][best_epochs[i]], best_metric_path)

        if len(selected_idxs[i]) > 0:
            selected_idx_path = os.path.join(writers[i].log_dir, "selected_idx.pkl")
            save_pickle(selected_idxs[i], selected_idx_path)
