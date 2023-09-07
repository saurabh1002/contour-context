#!/usr/bin/python3

import os
from typing import Set, Tuple

import argh
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table


def main(datadir):
    datadir = os.path.abspath(datadir)
    dataset_name = os.path.basename(datadir)

    gt_closure_indices = np.loadtxt(
        os.path.join(datadir, "loop_closure", "gt_closures.txt"), dtype=int
    )
    gt_closure_overlap_scores = np.loadtxt(os.path.join(datadir, "loop_closure", "gt_overlaps.txt"))
    gt_closure_indices = gt_closure_indices[np.where(gt_closure_overlap_scores > 0.5)[0]]

    gt_closures: Set[Tuple[int]] = set(map(lambda x: tuple(sorted(x)), gt_closure_indices))

    predicted_closures = np.loadtxt(
        os.path.join(datadir, "predicted_closures.txt"),
        dtype=float,
    )

    score_thresholds = np.arange(0.5, 1, 0.01)

    table = Table(box=box.ASCII_DOUBLE_HEAD, title=dataset_name)
    table.add_column("Score Threshold", justify="center", style="white")
    table.add_column("True Positives", justify="center", style="magenta")
    table.add_column("False Positives", justify="center", style="magenta")
    table.add_column("False Negatives", justify="center", style="magenta")
    table.add_column("Precision", justify="left", style="green")
    table.add_column("Recall", justify="left", style="green")
    table.add_column("F1 score", justify="left", style="green")

    scores = predicted_closures[:, 2]
    predicted_closures = np.asarray(predicted_closures[:, :2], int)

    for score_threshold in score_thresholds:
        ids = np.where(scores > score_threshold)[0]
        closures: Set[Tuple[int]] = set(map(lambda x: tuple(sorted(x)), predicted_closures[ids]))

        tp = len(gt_closures.intersection(closures))
        fp = len(closures) - tp
        fn = len(gt_closures) - tp

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = np.nan

        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = np.nan

        try:
            F1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            F1 = np.nan

        table.add_row(
            f"{score_threshold:.4f}",
            f"{tp}",
            f"{fp}",
            f"{fn}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{F1:.4f}",
        )

    console_print = Console()
    console_print.print(table)

    logfile = os.path.join(datadir, "metrics.txt")
    with open(logfile, "wt") as file:
        console_save = Console(file=file, width=100, force_jupyter=False)
        console_save.print(table)


if __name__ == "__main__":
    parser = argh.ArghParser()
    parser.add_commands([main])
    parser.dispatch()
