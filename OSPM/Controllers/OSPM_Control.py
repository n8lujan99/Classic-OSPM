# OSPM_Control.py
# Control center for OSPM configuration.
# Pure orchestration. No physics. No data logic.

import os
import datetime
import pandas as pd


def ensure_deck(ctrl):
    path = ctrl["CSV_PATH"]
    d = os.path.dirname(path)

    if d:
        os.makedirs(d, exist_ok=True)

    cols = list(ctrl["REQUIRE_COLUMNS"])
    params = list(ctrl["PARAMETER_NAMES"])
    theta0 = list(ctrl["INITIAL_THETA"])

    if os.path.exists(path):
        df = pd.read_csv(path)

        missing = [c for c in cols if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = pd.NA
            df = df[cols]
            df.to_csv(path, index=False)

        return

    row = {k: pd.NA for k in cols}

    for i, name in enumerate(params):
        if i < len(theta0):
            row[name] = theta0[i]

    row["chi2"] = float("inf")
    row["reward"] = pd.NA
    row["status"] = ctrl["FILL_DEFAULT_STATUS"]

    if "proposal_id" in row:
        row["proposal_id"] = 0

    if "refine_passes" in row:
        row["refine_passes"] = 0

    if "chi2_losvd" in row:
        row["chi2_losvd"] = float("inf")

    if "chi2_light" in row:
        row["chi2_light"] = float("inf")

    if "chi2_total" in row:
        row["chi2_total"] = float("inf")

    pd.DataFrame([row], columns=cols).to_csv(path, index=False)


def build_runtime(ctrl):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    wid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    rt = dict(ctrl)
    rt["RUN_ID"] = f"{ts}_pid{pid}" + (f"_w{wid}" if wid else "")
    rt["WORKER_ID"] = wid
    rt["RANDOM_SEED_EFFECTIVE"] = rt.get("RANDOM_SEED", 123456789) + wid

    return rt