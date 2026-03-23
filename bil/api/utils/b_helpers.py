"""
Helper functions for accessing data format B.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
import h5py

IGNORE_PARAMS: list[str] = []
IGNORE_TRIALDATA: list[str] = ["startDateNum", "startDateStr", "subject"]

RETYPE: dict[str, Any] = {"isSuccessful": bool}

RENAME: dict[str, str] = {
    "startCounter": "ms_start",
    "endCounter": "ms_end",
    "isSuccessful": "success",
    "p_posTarget_0": "target_loc_x",
    "p_posTarget_1": "target_loc_y",
    "timeFirstTargetAcquire": "time_target_acquire_first",
    "timeLastTargetAcquire": "time_target_acquire_last",
    "timeTargetAcquire": "time_target_acquire",
    "timeTrialEnd": "time_end",
    "timeTargetOn": "time_start",
    "trialLength": "time_duration",
    "trialNum": "number",
}

ORDER: list[str] = [
    "study",
    "number",
    "trial_id",
    "ms_start",
    "ms_end",
    "success",
    "quality",
    "target_loc_x",
    "target_loc_y",
    "target_loc",
    "timestamp",
    "time_start",
    "time_end",
    "time_duration",
    "time_target_acquire",
    "time_target_acquire_first",
    "time_target_acquire_last",
    "pos_target_x",
    "pos_target_y",
]


def angle_helper(row: pd.Series, tol: float = 1e-6) -> str:
    """Map Cartesian target locations to semantic directional labels.

    Args:
        row: A DataFrame row containing target coordinates.
        tol: Tolerance for coordinate comparisons.

    Returns:
        Directional label (e.g., 'center', 'top', 'bottom_left').
    """
    target_x, target_y = row["target_loc_x"], row["target_loc_y"]
    if abs(target_x) < tol and abs(target_y) < tol:
        return "center"
    angle = np.pi / 2 - math.atan2(target_x, target_y)
    if abs(angle) < tol:
        return "right"
    if abs(angle - np.pi * 5 / 4) < tol:
        return "bottom_left"
    if abs(angle - np.pi / 2) < tol:
        return "top"
    if abs(angle - np.pi * 3 / 4) < tol:
        return "top_left"
    if (abs(angle + np.pi / 2) < tol) or (abs(angle - np.pi * 3 / 2) < tol):
        return "bottom"
    if abs(angle + np.pi / 4) < tol:
        return "bottom_right"
    if abs(angle - np.pi / 4) < tol:
        return "top_right"
    if abs(angle - np.pi) < tol:
        return "left"
    return "none"


def df_from_h5(h5_file: h5py.File, run_id: str) -> pd.DataFrame:
    """Parse format B HDF5 head file into a formatted trial DataFrame.

    Args:
        h5_file: The open HDF5 file handle.
        run_id: The study identifier.

    Returns:
        Formatted trial metadata.
    """
    ### generate parameter dataframe ###
    params = [key for key in h5_file.keys() if key.startswith("tp_")]
    params_short = [p[3:] for p in params]
    params_dims = [h5_file[p].shape[2] for p in params]
    data = []
    columns = []
    for param_full, param_short, dim in zip(params, params_short, params_dims):
        if param_short in IGNORE_PARAMS:
            continue
        for idx in range(dim):
            if dim == 1 and idx == 0:
                start_name = f"ps_{param_short}"
                end_name = f"pe_{param_short}"
                param_name = f"p_{param_short}"
            else:
                start_name = f"ps_{param_short}_{idx}"
                end_name = f"pe_{param_short}_{idx}"
                param_name = f"p_{param_short}_{idx}"

            start_val = h5_file[param_full][:, 0, idx]
            end_val = h5_file[param_full][:, 1, idx]
            if h5_file[param_full].dtype.name.startswith("uint"):
                start_val = start_val.astype(int)
                end_val = end_val.astype(int)

            param_val = start_val.copy()
            param_set_idx = start_val == end_val
            param_val[~param_set_idx] = -1
            start_val[param_set_idx] = -1
            end_val[param_set_idx] = -1
            data.extend((param_val, start_val, end_val))
            columns.extend((param_name, start_name, end_name))
    params_df = pd.DataFrame(data, index=columns).T

    ### generate trial dataframe ###
    data_cols = pd.Index(h5_file.keys()).difference(params)
    data_cols = [col for col in data_cols if len(h5_file[col]) == len(params_df)]
    data_cols_dims = [h5_file[col].shape[1] for col in data_cols]
    data = []
    columns = []
    for col, dim in zip(data_cols, data_cols_dims):
        if col in IGNORE_TRIALDATA:
            continue
        if col == "timeTargetAcquire":
            columns.append(col)
            data.append([list(val[: np.argmax(val == 0)]) for val in h5_file[col][:]])
        else:
            for idx in range(dim):
                name = col if dim == 1 and idx == 0 else f"{col}_{idx}"
                data.append(h5_file[col][:, idx])
                columns.append(name)
    data_df = pd.DataFrame(data, index=columns).T

    ### join and reformat dataframe ###
    full_df = pd.concat((data_df, params_df), axis=1)
    full_df["study"] = run_id

    full_df = full_df.astype(RETYPE)
    full_df = full_df.rename(RENAME, axis=1)

    full_df["number"] -= 1

    full_df["trial_id"] = full_df["study"] + "_" + full_df["number"].astype(str)

    full_df["timestamp"] = -1
    full_df["quality"] = 1.0
    full_df["pos_target_x"] = 0
    full_df["pos_target_y"] = 0
    full_df["target_loc"] = full_df.apply(angle_helper, axis=1)

    # Fix timing TIMING
    # startCounter seems to have a constant offset from the true trial start index
    # Trial starts when state switches to 2
    true_trial_starts = []
    idx_list = np.argwhere(h5_file["state"][:].squeeze() == 2)[:, 0]
    prev = None
    for i in idx_list:
        if prev is None or i != prev + 1:
            true_trial_starts.append(i)
        prev = i
    true_trial_starts_arr = np.array(true_trial_starts)
    # Ensure we detected all trials
    assert len(true_trial_starts_arr) == len(full_df)
    # Offset should be constant
    diff = full_df["ms_start"].values - true_trial_starts_arr
    assert len(set(diff)) == 1
    # Correct by this constant offset
    full_df["ms_start"] = full_df["ms_start"] - diff
    full_df["ms_end"] = full_df["ms_end"] - diff

    order = [*ORDER, *full_df.columns.difference(ORDER)]
    full_df = full_df[order]

    return full_df
