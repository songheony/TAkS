import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px


COLORS = px.colors.qualitative.Plotly


def get_hyperparameter_name(root_dir, algorithm):
    names = []
    for x in root_dir.iterdir():
        if x.is_dir():
            base_name = x.name.split("(")[0]
            if base_name == algorithm:
                param_name = x.name.split("(")[1].split(")")[0]
                names.append(param_name)

    return names


def get_algorithms_name(root_dir, algorithms):
    names = algorithms.copy()
    for x in root_dir.iterdir():
        if x.is_dir():
            base_name = x.name.split("(")[0]
            if base_name in algorithms:
                idx = algorithms.index(base_name)
                names[idx] = x.name

    return names


def write_table(algorithms, accs, times, noise_envs, dataset_name):
    latex = "\\begin{table*}\n"
    latex += "\\centering\n"
    latex += "\\begin{threeparttable}\n"

    write_acc = accs is not None
    write_time = times is not None

    if write_acc and write_time:
        caption = "Average test accuracy over the last 10 epochs and training time"
    elif write_acc:
        caption = "Average test accuracy over the last 10 epochs"
    elif write_time:
        caption = "Average training time"
    latex += f"\\caption{{{caption} on {dataset_name}.\\vspace{{-2mm}}}}\n"

    header = "c"
    if write_acc and write_time:
        for i in range(len(algorithms) * 2 - 2):
            header += "|c"
    else:
        for i in range(len(algorithms)):
            header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    if write_acc and write_time:
        columns = "\\multirow{2}{*}{Noise}"
        for i in range(len(algorithms)):
            if i < 2:
                columns += f" & {algorithms[i]}"
            else:
                split = "c|" if i != len(algorithms) - 1 else "c"
                columns += f" & \\multicolumn{{2}}{{{split}}}{{{algorithms[i]}}}"
    else:
        columns = "Noise"
        for i in range(len(algorithms)):
            columns += f" & {algorithms[i]}"
    latex += f"{columns} \\\\\n"

    if write_acc and write_time:
        small_columns = " "
        for i in range(len(algorithms)):
            if i < 2:
                small_columns += " & Acc"
            else:
                small_columns += " & Acc & Time"
        latex += f"{small_columns} \\\\\n"
    latex += "\\hline\\hline\n"

    acc_values = []
    time_values = []
    for noise_env in noise_envs:
        acc_value = []
        time_value = []
        for algorithm in algorithms:
            if write_acc:
                acc = [x[-10:] for x in accs[noise_env][algorithm]]
                acc_mean = np.nanmean(acc)
                # acc_std = np.std(acc)
                acc_value.append(acc_mean)

            if write_time:
                if algorithm == "F-correction":
                    time_value.append(np.nan)
                else:
                    time = (
                        times[noise_env][algorithm].mean()
                        / times[noise_env]["Standard"].mean()
                    )
                    time_value.append(time)

        acc_values.append(acc_value)
        time_values.append(time_value)

    for i in range(len(noise_envs)):
        line = noise_envs[i].replace("%", "\\%")
        for j in range(len(algorithms)):
            if write_acc:
                acc_mean = acc_values[i][j]
                # if j == np.argmax(acc_values[i]):
                #     line += f" & \\textbf{{{acc_mean:0.2f}}}"
                # else:
                #     line += f" & {acc_mean:0.2f}"
                sorted_idx = np.argsort(acc_values[i])
                if j == sorted_idx[-1]:
                    line += f" & {{\\color{{red}} \\textbf{{{acc_mean:0.2f}}}}}"
                elif j == sorted_idx[-2]:
                    line += f" & {{\\color{{blue}} \\textit{{{acc_mean:0.2f}}}}}"
                else:
                    line += f" & {acc_mean:0.2f}"

            if write_time and j >= 2:
                time = time_values[i][j]
                if j - 1 == np.nanargmin(time_values[i][1:]):
                    line += f" & \\textbf{{{time:0.2f}}}"
                elif np.isnan(time):
                    line += " & -"
                else:
                    line += f" & {time:0.2f}"
        line += " \\\\\n"

        latex += f"{line}"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{threeparttable}\n"
    latex += "\\end{table*}\n"

    return latex


def write_prec_table(algorithms, datasets, dataset_precs, noise_envs):
    latex = "\\begin{table*}\n"
    latex += "\\centering\n"
    latex += "\\begin{threeparttable}\n"

    caption = "Average label precision over the last 10 epochs"
    latex += f"\\caption{{{caption}.\\vspace{{-2mm}}}}\n"

    header = "c"
    for i in range(len(algorithms) * len(datasets)):
        header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    columns = "\\multirow{2}{*}{Noise}"
    for i in range(len(algorithms)):
        split = "c|" if i != len(algorithms) - 1 else "c"
        columns += f" & \\multicolumn{{{len(datasets)}}}{{{split}}}{{{algorithms[i]}}}"
    latex += f"{columns} \\\\\n"

    small_columns = " "
    for i in range(len(algorithms)):
        for dataset in datasets:

            small_columns += f" & {dataset}"
    latex += f"{small_columns} \\\\\n"
    latex += "\\hline\\hline\n"

    prec_values = []
    for noise_env in noise_envs:
        prec_value = []
        for algorithm in algorithms:
            prec_dataset = []
            for dataset in datasets:
                prec = [x[-10:] for x in dataset_precs[dataset][noise_env][algorithm]]
                prec_mean = np.nanmean(prec)
                # prec_std = np.std(prec)
                prec_dataset.append(prec_mean)
            prec_value.append(prec_dataset)

        prec_values.append(prec_value)

    prec_values = np.array(prec_values)
    for i in range(len(noise_envs)):
        line = noise_envs[i].replace("%", "\\%")
        for j in range(len(algorithms)):
            for k in range(len(datasets)):
                prec_mean = prec_values[i][j][k]
                sorted_idx = np.argsort(prec_values[i, :, k])
                if j == sorted_idx[-1]:
                    line += f" & {{\\color{{red}} \\textbf{{{prec_mean:0.2f}}}}}"
                elif j == sorted_idx[-2]:
                    line += f" & {{\\color{{blue}} \\textit{{{prec_mean:0.2f}}}}}"
                else:
                    line += f" & {prec_mean:0.2f}"
        line += " \\\\\n"

        latex += f"{line}"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{threeparttable}\n"
    latex += "\\end{table*}\n"

    return latex


def draw_lines(algorithms, acc):
    fig = go.Figure()

    colors = COLORS[1 : len(algorithms) + 1][::-1]
    for hex_color, algorithm in zip(colors, algorithms):
        if algorithm == "JoCoR":
            continue
        color = plotly.colors.hex_to_rgb(hex_color)
        x = acc[algorithm][0].index.tolist()
        y = np.mean(acc[algorithm], axis=0)

        fig.add_traces(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=algorithm,
                line=dict(color=f"rgb{color}", width=8),
            )
        )

    fig.update_layout(
        autosize=False,
        width=1200,
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        margin=go.layout.Margin(
            l=100,  # left margin
            r=0,  # right margin
            b=40,  # bottom margin
            t=0,  # top margin
        ),
        xaxis_title="Epoch",
        yaxis_title="Test accuracy",
        font=dict(size=30, family="Arial",),
    )

    return fig


def draw_multiple_lines(algorithms, datas, noise_envs, error_band=False):
    fig = make_subplots(
        rows=len(datas),
        cols=len(noise_envs),
        subplot_titles=noise_envs,
        vertical_spacing=0.1,
    )

    for row, data in enumerate(datas):
        for col, noise_env in enumerate(noise_envs):
            if algorithms is None:
                target_algorithms = sorted(data[noise_env].keys())
            else:
                target_algorithms = algorithms

            colors = COLORS[1 : len(target_algorithms) + 1][::-1]
            for hex_color, algorithm in zip(colors, target_algorithms):
                if algorithm not in data[noise_env].keys():
                    continue

                color = plotly.colors.hex_to_rgb(hex_color)

                x = data[noise_env][algorithm][0].index.tolist()
                y = np.mean(data[noise_env][algorithm], axis=0)

                if error_band:
                    y_error = np.std(data[noise_env][algorithm], axis=0)

                    y_upper = y + y_error
                    y_lower = y - y_error

                    fig.add_trace(
                        go.Scatter(
                            x=x + x[::-1],
                            y=y_upper.tolist() + y_lower[::-1].tolist(),
                            fill="tozerox",
                            fillcolor=f"rgba({color[0]},{color[1]},{color[2]},0.4)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                            name=algorithm,
                        ),
                        row=row + 1,
                        col=col + 1,
                    )

                showlegend = (algorithms is None) | (row == 0 and col == 0)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(color=f"rgb{color}"),
                        name=algorithm,
                        legendgroup=algorithm,
                        showlegend=showlegend,
                    ),
                    row=row + 1,
                    col=col + 1,
                )

    fig.update_layout(
        autosize=False,
        width=800,
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=0.9),
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=40,  # bottom margin
            t=0,  # top margin
        ),
        font=dict(family="Arial", size=10),
    )
    fig.add_annotation(
        x=0.5,
        y=0,
        xanchor="center",
        xref="paper",
        yanchor="top",
        yref="paper",
        text="Epoch",
        showarrow=False,
        yshift=-15,
        font=dict(size=12),
    )
    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12
    fig.update_yaxes(title_text="Test accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Label precision (%)", row=2, col=1)

    return fig


def get_accuracys(log_dir, algorithms, vis_names, noise_envs, vis_envs, seeds):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)

    accs = {}
    times = {}
    for vis_env, noise_env in zip(vis_envs, noise_envs):
        env_dir = log_dir / noise_env / "jocor_model"
        algorithm_names = get_algorithms_name(env_dir, algorithms)

        acc = {}
        time = {}
        for vis_name, algorithm_name in zip(vis_names, algorithm_names):
            algorithm_acc = []
            for seed in seeds:
                seed_dir = env_dir / algorithm_name / seed
                for model_dir in seed_dir.iterdir():
                    metric_path = model_dir / "metric.csv"
                    metric = pd.read_csv(metric_path, header=0, index_col=0)
                    algorithm_acc.append(metric["test_top1"])
            acc[vis_name] = algorithm_acc
            time[vis_name] = metric["epoch_time"]

        accs[vis_env] = acc
        times[vis_env] = time

    return accs, times


def calc_precisions(log_dir, ind_paths, algorithms, noise_envs, seeds, precision_dir):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if isinstance(precision_dir, str):
        precision_dir = Path(precision_dir)

    os.makedirs(precision_dir, exist_ok=True)

    precision_dir.mkdir(exist_ok=True)

    for noise_env in noise_envs:
        env_dir = log_dir / noise_env / "jocor_model"
        algorithm_names = get_algorithms_name(env_dir, algorithms)

        for algorithm, algorithm_name in zip(algorithms, algorithm_names):
            for seed in seeds:
                csv_path = (
                    precision_dir / f"{noise_env}_{algorithm}_{seed}_precision.csv"
                )
                if csv_path.exists():
                    continue
                algorithm_prec = []
                train_noise_ind = np.load(ind_paths[noise_env][seed])
                seed_dir = env_dir / algorithm_name / seed
                for model_dir in seed_dir.iterdir():
                    idx_path = model_dir / "selected_idx.pkl"
                    selected_idxs = pickle.loads(idx_path.read_bytes())
                    label_precisions = [np.nan] * 200
                    for i, selected_idx in enumerate(selected_idxs):
                        clean_selected_ind = np.setdiff1d(selected_idx, train_noise_ind)
                        label_precision = len(clean_selected_ind) / len(selected_idx)
                        label_precisions[i] = label_precision
                    algorithm_prec.append(label_precisions)

                df = pd.DataFrame(
                    dict([(k, pd.Series(v)) for k, v in enumerate(algorithm_prec)])
                )
                df.to_csv(csv_path)


def get_precisions(
    log_dir,
    ind_paths,
    algorithms,
    pre_algorithms,
    vis_names,
    noise_envs,
    vis_envs,
    seeds,
    precision_dir,
):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)

    precs = {}
    for vis_env, noise_env in zip(vis_envs, noise_envs):
        env_dir = log_dir / noise_env / "jocor_model"
        algorithm_names = get_algorithms_name(env_dir, algorithms)

        prec = {}
        for vis_name, algorithm, algorithm_name in zip(
            vis_names, algorithms, algorithm_names
        ):
            if algorithm not in pre_algorithms:
                continue

            algorithm_prec = []
            for seed in seeds:
                df = pd.read_csv(
                    f"{precision_dir}/{noise_env}_{algorithm}_{seed}_precision.csv",
                    header=0,
                    index_col=0,
                )
                for column in df.columns:
                    algorithm_prec.append(df[column] * 100)
            prec[vis_name] = algorithm_prec
        precs[vis_env] = prec

    return precs


def draw_curves(
    log_dir,
    ind_paths,
    algorithms,
    pre_algorithms,
    vis_algorithms,
    noise_envs,
    vis_noise_envs,
    seeds,
    precision_dir,
    dataset_name,
    save_dir,
):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    vis_names = [vis_algorithms[algorithm] for algorithm in algorithms]
    vis_envs = [vis_noise_envs[noise_env] for noise_env in noise_envs]

    accs, times = get_accuracys(
        log_dir, algorithms, vis_names, noise_envs, vis_envs, seeds
    )
    precs = get_precisions(
        log_dir,
        ind_paths,
        algorithms,
        pre_algorithms,
        vis_names,
        noise_envs,
        vis_envs,
        seeds,
        precision_dir,
    )

    acc_latex = write_table(vis_names, accs, None, vis_envs, dataset_name)
    txt_file = Path(f"{save_dir}/accuracy.txt")
    txt_file.write_text(acc_latex)

    time_latex = write_table(vis_names, None, times, vis_envs, dataset_name)
    txt_file = Path(f"{save_dir}/time.txt")
    txt_file.write_text(time_latex)

    both_latex = write_table(vis_names, accs, times, vis_envs, dataset_name)
    txt_file = Path(f"{save_dir}/both.txt")
    txt_file.write_text(both_latex)

    fig = draw_multiple_lines(vis_names, [accs, precs], vis_envs, error_band=True)
    fig.write_html(f"{save_dir}/curve.html")
    fig.write_image(f"{save_dir}/curve.pdf")


def draw_acc_tables(
    log_dir,
    algorithms,
    vis_algorithms,
    noise_envs,
    vis_noise_envs,
    seeds,
    datasets,
    save_dir,
):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    vis_names = [vis_algorithms[algorithm] for algorithm in algorithms]
    vis_envs = [vis_noise_envs[noise_env] for noise_env in noise_envs]

    dataset_accs = {}
    for dataset_name in datasets:
        accs, times = get_accuracys(
            log_dir / dataset_name, algorithms, vis_names, noise_envs, vis_envs, seeds
        )
        dataset_accs[dataset_name] = accs

    prec_latex = write_prec_table(algorithms, datasets, dataset_accs, vis_envs)
    txt_file = Path("results/acc_only.txt")
    txt_file.write_text(prec_latex)


def draw_prec_tables(
    log_dir,
    ind_paths,
    algorithms,
    pre_algorithms,
    vis_algorithms,
    noise_envs,
    vis_noise_envs,
    seeds,
    save_dir,
    datasets,
    precision_dirs,
):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    vis_names = [vis_algorithms[algorithm] for algorithm in algorithms]
    vis_envs = [vis_noise_envs[noise_env] for noise_env in noise_envs]

    dataset_precs = {}
    for dataset_name in datasets:
        precs = get_precisions(
            log_dir / dataset_name,
            ind_paths,
            algorithms,
            pre_algorithms,
            vis_names,
            noise_envs,
            vis_envs,
            seeds,
            precision_dirs[dataset_name],
        )
        dataset_precs[dataset_name] = precs

    prec_latex = write_prec_table(pre_algorithms, datasets, dataset_precs, vis_envs)
    txt_file = Path("results/prec.txt")
    txt_file.write_text(prec_latex)


def draw_single(
    log_dir, seed, algorithms, vis_algorithms, save_dir,
):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    vis_names = [vis_algorithms[algorithm] for algorithm in algorithms]

    env_dir = log_dir / "jocor_model"
    algorithm_names = get_algorithms_name(env_dir, algorithms)

    best_acc = {}
    last_acc = {}
    acc = {}
    time = {}
    for vis_name, algorithm_name in zip(vis_names, algorithm_names):
        algorithm_best_acc = []
        algorithm_last_acc = []
        algorithm_acc = []
        seed_dir = env_dir / algorithm_name / str(seed)
        for model_dir in seed_dir.iterdir():
            best_metric_path = model_dir / "best_metric.csv"
            best_metric = pd.read_csv(best_metric_path, header=0, index_col=0)
            algorithm_best_acc.append(best_metric["test_top1"].values[-1])

            metric_path = model_dir / "metric.csv"
            metric = pd.read_csv(metric_path, header=0, index_col=0)
            algorithm_last_acc.append(metric["test_top1"].values[-1])

            algorithm_acc.append(metric["test_top1"])
        best_acc[vis_name] = np.mean(algorithm_best_acc)
        last_acc[vis_name] = np.mean(algorithm_last_acc)
        acc[vis_name] = algorithm_acc
        time[vis_name] = np.mean(metric["epoch_time"])

    fig = draw_lines(vis_names, acc)
    fig.write_html(f"{save_dir}/curve.html")
    fig.write_image(f"{save_dir}/curve.pdf")

    for vis_name in vis_names:
        print(
            f"{vis_name}: Best({best_acc[vis_name]:.2f}), Last({last_acc[vis_name]:.2f}), Time({time[vis_name] / time['Standard']:.2f})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "Standard",
            "F-correction",
            "ITLM",
            "Decouple",
            "Co-teaching",
            "Co-teaching+",
            "JoCoR",
            "Ours",
        ],
    )

    args = parser.parse_args()

    vis_algorithms = {
        "Standard": "Standard",
        "F-correction": "F-correction",
        "ITLM": "ITLM",
        "Decouple": "Decouple",
        "Co-teaching": "Co-teaching",
        "Co-teaching+": "Co-teaching+",
        "JoCoR": "JoCoR",
        "Ours": "Ours",
        "Greedy": "Greedy",
        "FTL": "FTL",
    }
    vis_noise_envs = {
        "symmetric-20.0%": "Symmetric-20%",
        "symmetric-50.0%": "Symmetric-50%",
        "symmetric-80.0%": "Symmetric-80%",
        "asymmetric-40.0%": "Asymmetric-40%",
    }
    vis_dataset_names = {
        "mnist": "MNIST",
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "clothing1m": "Clothing1M",
    }
    noise_envs = [
        "symmetric-20.0%",
        "symmetric-50.0%",
        "symmetric-80.0%",
        "asymmetric-40.0%",
    ]
    dataset_names = ["mnist", "cifar10", "cifar100"]

    # draw accuracy of Clothing1M
    dataset_name = "clothing1m"
    seed = 0
    save_dir = f"results/{dataset_name}"
    draw_single(
        f"{args.log_dir}/{dataset_name}",
        seed,
        args.algorithms,
        vis_algorithms,
        save_dir,
    )

    # draw accuracy of ablation study
    ablation_algorithms = ["Greedy", "FTL", "Ours"]
    save_dir = "results"
    draw_acc_tables(
        args.log_dir,
        ablation_algorithms,
        vis_algorithms,
        noise_envs,
        vis_noise_envs,
        args.seeds,
        dataset_names,
        save_dir,
    )

    # draw accuracy and precision
    pre_algorithms = [
        "ITLM",
        "Decouple",
        "Co-teaching",
        "Co-teaching+",
        "JoCoR",
        "Ours",
    ]
    dataset_inds = {}
    precision_dirs = {}
    for dataset_name in dataset_names:
        precision_dir = f"results/precision/{dataset_name}"
        precision_dirs[dataset_name] = precision_dir
        save_dir = f"results/{dataset_name}"
        ind_paths = {
            "symmetric-20.0%": {
                seed: f"{args.dataset_path}/changed_{dataset_name}_{seed}/symmetric_0.2_ind.npy"
                for seed in args.seeds
            },
            "symmetric-50.0%": {
                seed: f"{args.dataset_path}/changed_{dataset_name}_{seed}/symmetric_0.5_ind.npy"
                for seed in args.seeds
            },
            "symmetric-80.0%": {
                seed: f"{args.dataset_path}/changed_{dataset_name}_{seed}/symmetric_0.8_ind.npy"
                for seed in args.seeds
            },
            "asymmetric-40.0%": {
                seed: f"{args.dataset_path}/changed_{dataset_name}_{seed}/asymmetric_0.4_ind.npy"
                for seed in args.seeds
            },
        }
        dataset_inds[dataset_name] = ind_paths
        calc_precisions(
            f"{args.log_dir}/{dataset_name}",
            ind_paths,
            pre_algorithms,
            noise_envs,
            args.seeds,
            precision_dir,
        )
        draw_curves(
            f"{args.log_dir}/{dataset_name}",
            ind_paths,
            args.algorithms,
            pre_algorithms,
            vis_algorithms,
            noise_envs,
            vis_noise_envs,
            args.seeds,
            precision_dir,
            vis_dataset_names[dataset_name],
            save_dir,
        )

    draw_prec_tables(
        args.log_dir,
        dataset_inds,
        args.algorithms,
        pre_algorithms,
        vis_algorithms,
        noise_envs,
        vis_noise_envs,
        args.seeds,
        "results",
        dataset_names,
        precision_dirs,
    )
