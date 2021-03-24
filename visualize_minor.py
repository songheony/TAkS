import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from cuml import UMAP

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import load_datasets
from model import get_model
from visualize import get_algorithms_name, get_hyperparameter_name


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        v *= 255
        return (v, v, v)
    i = int(h * 6.0)  # XXX assume int() truncates!
    f = (h * 6.0) - i
    p, q, t = (
        int(255 * (v * (1.0 - s))),
        int(255 * (v * (1.0 - s * f))),
        int(255 * (v * (1.0 - s * (1.0 - f)))),
    )
    v *= 255
    i %= 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    if i == 5:
        return (v, p, q)


def draw_wrong_samples(
    dataset, original_labels, algorithm, log_dir, seed, ind_path, save_dir
):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    algorithm_name = get_algorithms_name(log_dir, [algorithm])[0]
    model_dir = log_dir / algorithm_name / str(seed) / "model0"

    idx_path = model_dir / "selected_idx.pkl"
    selected_idxs = pickle.loads(Path(idx_path).read_bytes())

    labels = np.array(original_labels)

    noise_ind = set(np.load(ind_path))
    selected_idx = set(selected_idxs[-1])
    wrong_idx = selected_idx & noise_ind
    transform = transforms.ToPILImage()

    n_class = len(train_dataset.classes)
    conf_matrix = np.zeros((n_class, n_class), dtype=int)

    for idx in wrong_idx:
        image, label, index = train_dataset[idx]
        pil_image = transform(image.squeeze_(0))
        cls_dir = save_dir / str(labels[idx]) / str(label)
        os.makedirs(cls_dir, exist_ok=True)
        pil_image.save(cls_dir / f"{idx}.png")

        conf_matrix[labels[idx], label] += 1

    fig = ff.create_annotated_heatmap(
        conf_matrix,
        x=train_dataset.classes,
        y=train_dataset.classes,
        colorscale="Viridis",
    )
    fig.update_xaxes(title_text="Noisy label")
    fig.update_yaxes(title_text="Clean label", autorange="reversed")
    # fig.write_html(f"{save_dir}/confusion.html")
    fig.write_image(f"{save_dir}/confusion.pdf")


def draw_sampels(
    model, dataset, original_labels, algorithm, log_dir, seed, ind_path, save_dir
):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    algorithm_name = get_algorithms_name(log_dir, [algorithm])[0]
    model_dir = log_dir / algorithm_name / str(seed) / "model0"

    idx_path = model_dir / "selected_idx.pkl"
    selected_idxs = pickle.loads(Path(idx_path).read_bytes())

    pt_path = model_dir / "best_model.pt"
    model.load_state_dict(torch.load(pt_path))

    noise_ind = set(np.load(ind_path))
    selected_idx = set(selected_idxs[-1])

    labels = np.array(original_labels)

    fixed_train_dataloader = DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=8
    )
    model.eval()

    X = []
    with torch.no_grad():
        for i, (images, _, _) in enumerate(fixed_train_dataloader):
            if torch.cuda.is_available():
                images = images.to("cuda:0")
            output = model(images)
            output = output.cpu().numpy()
            X.append(output)

    X = np.concatenate(X, axis=0)

    tsne = UMAP(n_components=2, random_state=0)
    embedding = tsne.fit_transform(X)

    emb1 = embedding[:, 0]
    emb2 = embedding[:, 1]

    targets = ["clean", "noise", "select", "nonsel"]
    multi_fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.05, horizontal_spacing=0.02
    )

    min_x, max_x = [], []
    min_y, max_y = [], []

    for n, target in enumerate(targets):
        fig = go.Figure()
        class_min_x, class_max_x = np.inf, 0
        class_min_y, class_max_y = np.inf, 0
        for i in range(len(dataset.classes)):
            class_ind = set(np.where(labels == i)[0])

            if target == "noise":
                target_ind = class_ind & noise_ind
            elif target == "clean":
                target_ind = class_ind - noise_ind
            elif target == "select":
                target_ind = class_ind & selected_idx
            elif target == "nonsel":
                target_ind = class_ind - selected_idx

            X = emb1[list(target_ind)]
            Y = emb2[list(target_ind)]

            if len(target_ind) > 0:
                class_min_x = min(min(X), class_min_x)
                class_max_x = max(max(X), class_max_x)
                class_min_y = min(min(Y), class_min_y)
                class_max_y = max(max(Y), class_max_y)

            if len(dataset.classes) <= 10:
                scatter = go.Scattergl(x=X, y=Y, mode="markers")
            else:
                color = hsv_to_rgb(360 / len(dataset.classes) * i, 1, 1)
                scatter = go.Scattergl(
                    x=X,
                    y=Y,
                    mode="markers",
                    marker_color=f"rgb({color[0]},{color[1]},{color[2]})",
                )

            fig.add_trace(scatter)

            multi_fig.add_trace(
                scatter, row=n // 2 + 1, col=n % 2 + 1,
            )
        min_x.append(class_min_x)
        max_x.append(class_max_x)
        min_y.append(class_min_y)
        max_y.append(class_max_y)
        fig.update_layout(
            autosize=False,
            width=1200,
            height=800,
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=0,  # top margin
            ),
            showlegend=False,
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        # fig.write_html(f"{save_dir}/umap-{target}.html")
        fig.write_image(f"{save_dir}/umap-{target}.pdf")

    multi_fig.update_layout(
        autosize=False,
        width=3200,
        height=2400,
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=100,  # bottom margin
            t=0,  # top margin
        ),
        font=dict(family="Arial", size=84),
        showlegend=False,
    )
    multi_fig.update_xaxes(title_text="Clean samples", row=1, col=1)
    multi_fig.update_xaxes(title_text="Incorrectly labeled samples", row=1, col=2)
    multi_fig.update_xaxes(title_text="Selected samples", row=2, col=1)
    multi_fig.update_xaxes(title_text="Non-selected samples", row=2, col=2)
    # scaling
    multi_fig.update_xaxes(showticklabels=False, range=[max(min_x), min(max_x)])
    multi_fig.update_yaxes(showticklabels=False, range=[max(min_y), min(max_y)])
    # multi_fig.write_html(f"{save_dir}/umap.html")
    multi_fig.write_image(f"{save_dir}/umap.pdf")


def draw_heatmap(algorithm, n_samples, log_dir, seed, ind_path, save_dir):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    algorithm_name = get_algorithms_name(log_dir, [algorithm])[0]
    model_dir = log_dir / algorithm_name / str(seed) / "model0"

    noise_ind = np.load(ind_path).tolist()
    clean_ind = list(set(range(n_samples)) - set(noise_ind))
    sorted_idx = clean_ind + noise_ind

    targets = ["loss"]
    for target in targets:
        data = [
            np.load(model_dir / f"{target}_{epoch}.npy").reshape(-1, 1)
            for epoch in range(1, 201)
        ]
        z = np.concatenate(data, axis=1)
        colorscale = "Bluered"
        fig = go.Figure(
            data=go.Heatmap(
                z=z[sorted_idx, :], colorscale=colorscale, x=list(range(1, 201))
            )
        )
        fig.update_layout(
            autosize=False,
            width=1200,
            height=900,
            margin=go.layout.Margin(l=80, t=10,),
            xaxis_title="Epoch",
            font=dict(size=36, family="Arial",),
        )
        fig.update_yaxes(showticklabels=False)
        fig.add_shape(
            type="line",
            x0=0,
            y0=len(clean_ind),
            x1=1,
            y1=len(clean_ind),
            line=dict(color="yellow", dash="dot", width=8),
            xref="paper",
            yref="y",
        )
        incorrectly_y = len(clean_ind) + len(noise_ind) / 2
        if len(clean_ind) / len(sorted_idx) > 0.7:
            incorrectly_y -= len(sorted_idx) * 0.01
        fig.add_annotation(
            dict(
                font=dict(size=36, family="Arial",),
                x=-0.085,
                y=incorrectly_y,
                showarrow=False,
                text="Incorrectly<br>labeled",
                textangle=-90,
                xref="paper",
                yref="y",
            )
        )
        fig.add_annotation(
            dict(
                font=dict(size=36, family="Arial",),
                x=-0.06,
                y=len(clean_ind) / 2,
                showarrow=False,
                text="Clean",
                textangle=-90,
                xref="paper",
                yref="y",
            )
        )
        # fig.write_html(f"{save_dir}/{target}.html")
        fig.write_image(f"{save_dir}/{target}.pdf")


def draw_loss(algorithm, n_samples, log_dir, seed, ind_path, save_dir):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    param_names = get_hyperparameter_name(log_dir, algorithm)

    noise_ind = np.load(ind_path).tolist()
    clean_ind = list(set(range(n_samples)) - set(noise_ind))
    sorted_idx = clean_ind + noise_ind

    targets = ["loss", "cum_loss"]
    for target in targets:
        fig = go.Figure()
        vis_names = ["20%", "40%", "60%", "80%", "100%"]
        for param_name, vis_name in zip(sorted(param_names), vis_names):
            model_dir = log_dir / f"{algorithm}({param_name})" / str(seed) / "model0"
            idx_path = model_dir / "selected_idx.pkl"
            selected_idxs = pickle.loads(Path(idx_path).read_bytes())
            x = []
            for i in range(len(selected_idxs)):
                loss = np.load(model_dir / f"{target}_{i + 1}.npy")
                x.append(loss[selected_idxs[i - 1]].sum())
            fig.add_trace(
                go.Scattergl(
                    x=list(range(1, len(x) + 1)), y=x, mode="lines", name=vis_name, line=dict(width=8),
                )
            )

        fig.update_layout(
            autosize=False,
            width=1200,
            height=900,
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=0.75),
            margin=go.layout.Margin(
                l=100,  # left margin
                r=0,  # right margin
                b=40,  # bottom margin
                t=0,  # top margin
            ),
            xaxis_title="Epoch",
            yaxis_title="Total selection risk" if target == "cum_loss" else "Selection risk",
            font=dict(size=30, family="Arial",),
        )
        # fig.write_html(f"{save_dir}/{target}.html")
        fig.write_image(f"{save_dir}/{target}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--algorithm", type=str, default="Ours")
    parser.add_argument("--model_name", type=str, default="jocor_model")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    dataset_name = "mnist"
    n_samples = 60000
    noise_type = "symmetric"
    noise_ratio = 0.5
    log_dir = f"{args.log_dir}/precision/{dataset_name}/{noise_type}-{noise_ratio * 100}%/{args.model_name}"
    ind_path = f"{args.dataset_path}/changed_{dataset_name}_{args.seed}/{noise_type}_{noise_ratio}_ind.npy"
    save_dir = f"results/precision/{dataset_name}/{noise_type}/{noise_ratio}"
    param_names = get_hyperparameter_name(Path(log_dir), "Precision")
    draw_loss("Precision", n_samples, log_dir, args.seed, ind_path, save_dir)

    noise_types = ["symmetric", "asymmetric"]
    datasets_name = ["mnist", "cifar10", "cifar100"]

    for dataset_name in datasets_name:
        if dataset_name == "mnist":
            n_samples = 60000
        else:
            n_samples = 50000
        for noise_type in noise_types:
            if noise_type == "symmetric":
                noise_ratios = [0.2, 0.5, 0.8]
            else:
                noise_ratios = [0.4]
            for noise_ratio in noise_ratios:
                log_dir = f"{args.log_dir}/{dataset_name}/{noise_type}-{noise_ratio * 100}%/{args.model_name}"
                ind_path = f"{args.dataset_path}/changed_{dataset_name}_{args.seed}/{noise_type}_{noise_ratio}_ind.npy"
                save_dir = f"results/minor/{dataset_name}/{noise_type}-{noise_ratio * 100}%/{args.model_name}/{args.algorithm}"

                if args.algorithm == "Ours":
                    draw_heatmap(args.algorithm, n_samples, log_dir, args.seed, ind_path, save_dir)

                (
                    train_dataset,
                    valid_dataset,
                    test_dataset,
                    train_noise_ind,
                    original_labels,
                ) = load_datasets(
                    dataset_name,
                    args.dataset_path,
                    1,
                    noise_type,
                    noise_ratio,
                    [],
                    args.seed,
                    need_original=True,
                )
                model = get_model(args.model_name, dataset_name).to("cuda:0")
                draw_sampels(
                    model,
                    train_dataset,
                    original_labels,
                    args.algorithm,
                    log_dir,
                    args.seed,
                    ind_path,
                    save_dir,
                )
                draw_wrong_samples(
                    train_dataset,
                    original_labels,
                    args.algorithm,
                    log_dir,
                    args.seed,
                    ind_path,
                    save_dir,
                )
