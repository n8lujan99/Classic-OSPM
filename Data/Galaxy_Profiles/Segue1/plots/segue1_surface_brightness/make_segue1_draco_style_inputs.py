import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path("Data/Galaxy_Profiles/Segue1")

SB_PROFILE = ROOT / "Segue1_NO09_digitized_profile.csv"
SB_MAPPED = ROOT / "Segue1_SB_mapped_to_LOSVD_bins.csv"
KIN_BINS = ROOT / "Segue1_Simon_kinematic_bins_v2.csv"

OUT_KIN = ROOT / "segue1_simon_kinematic_bins_16.csv"
OUT_SB = ROOT / "segue1_NO09_surface_brightness_on_simon_bins_16.csv"


def interp_log_sigma(r_eval, r_prof, sigma_prof):
    good = (r_prof > 0) & (sigma_prof > 0)
    r_prof = r_prof[good]
    sigma_prof = sigma_prof[good]

    return np.exp(
        np.interp(
            np.log(r_eval),
            np.log(r_prof),
            np.log(sigma_prof),
            left=np.log(sigma_prof[0]),
            right=np.log(sigma_prof[-1]),
        )
    )


def interp_log_err(r_eval, r_prof, err_prof):
    good = (r_prof > 0) & (err_prof > 0)
    r_prof = r_prof[good]
    err_prof = err_prof[good]

    return np.exp(
        np.interp(
            np.log(r_eval),
            np.log(r_prof),
            np.log(err_prof),
            left=np.log(err_prof[0]),
            right=np.log(err_prof[-1]),
        )
    )


def main():
    sb_prof = pd.read_csv(SB_PROFILE)
    sb_map = pd.read_csv(SB_MAPPED)
    kin = pd.read_csv(KIN_BINS)

    # ------------------------------------------------------------------
    # 1. Draco-style kinematic bins
    # ------------------------------------------------------------------
    kin_out = pd.DataFrame()
    kin_out["bin_id"] = np.arange(len(kin), dtype=int)
    kin_out["R_inner_pc"] = kin["rin_pc_edge"].astype(float)
    kin_out["R_outer_pc"] = kin["rout_pc_edge"].astype(float)
    kin_out["R_mid_pc"] = kin["rmid_pc_edge"].astype(float)
    kin_out["N_vlos"] = kin["n_stars"].astype(int)

    kin_out.to_csv(OUT_KIN, index=False)

    # ------------------------------------------------------------------
    # 2. Draco-style surface brightness mapped onto those bins
    # ------------------------------------------------------------------
    r_prof = sb_prof["rmid_pc"].to_numpy(float)
    sig_prof = sb_prof["sigma_pc2"].to_numpy(float)
    err_prof = sb_prof["sigma_err_pc2"].to_numpy(float)

    sb_out = pd.DataFrame()
    sb_out["bin_id"] = np.arange(len(sb_map), dtype=int)
    sb_out["R_inner_pc"] = sb_map["rin_pc"].astype(float)
    sb_out["R_outer_pc"] = sb_map["rout_pc"].astype(float)
    sb_out["R_pc"] = sb_map["rmid_pc"].astype(float)

    # Sigma at the midpoint. The actual aperture light is from the integral.
    sb_out["Sigma"] = interp_log_sigma(
        sb_out["R_pc"].to_numpy(float),
        r_prof,
        sig_prof,
    )
    sb_out["Sigma_err"] = interp_log_err(
        sb_out["R_pc"].to_numpy(float),
        r_prof,
        err_prof,
    )

    sb_out["light_raw"] = sb_map["tracer_weight"].astype(float)
    sb_out["light_frac"] = sb_map["tracer_fraction"].astype(float)

    sb_out["q_axis_ratio"] = 1.0
    sb_out["ellipticity"] = 0.0

    sb_out["area_pc2"] = np.pi * (
        sb_out["R_outer_pc"] ** 2 - sb_out["R_inner_pc"] ** 2
    )

    sb_out["source_surface_brightness_csv"] = str(SB_PROFILE)
    sb_out["source_target_bins_csv"] = str(OUT_KIN)
    sb_out["rebin_method"] = "loglog"
    sb_out["galaxy"] = "Segue1"
    sb_out["source"] = "Niederste-Ostholt_et_al_2009_Fig7_digitized"
    sb_out["preferred_profile"] = "CMD_mask_number_counts"
    sb_out["radius_type"] = "projected_circular_radius"
    sb_out["pc_per_arcmin_assumed"] = 23_000.0 * np.pi / (180.0 * 60.0)
    sb_out["note"] = (
        "Digitized NO09 Fig. 7 stellar number-count profile. "
        "Counts are used as tracer surface density in place of calibrated surface brightness."
    )
    sb_out["N_vlos"] = kin["n_stars"].astype(int).to_numpy()

    sb_out.to_csv(OUT_SB, index=False)

    print("Saved:")
    print(f"  {OUT_KIN}")
    print(f"  {OUT_SB}")
    print()
    print("Kinematic bins:")
    print(kin_out.to_string(index=False))
    print()
    print("SB on bins:")
    print(sb_out.to_string(index=False))


if __name__ == "__main__":
    main()
