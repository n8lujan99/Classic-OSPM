# ========================================================================================================================
# julia_self_check.jl
# Just a quick self check for when DEVELOPING the code and pipeline not for production
# ie not changing code = not using this
# ========================================================================================================================

include(joinpath(@__DIR__, "OSPM_Physics_Spherical.jl"))

const M = OSPMPhysicsSpherical

println("[SELF CHECK] module_loaded=true")

required_symbols = (
    :solve_weights,
    :solve_weights_multinomial,
    :losvd_chi2_state,
    :multinomial_losvd_state,
    :entropy_value,
    :wphase_diagnostics,
    :spear_build_Am,
    :spear_rhs!,
    :spear_delta_w,
    :spear_step_light_losvd,
    :spear_update_expanded,
    :spear_consistency_diagnostic,
    :safe_step_factor,
    :PhaseVolumeState,
    :compute_phase_volumes,
    :collect_equatorial_sos,
    :orbit_family_integrals,
    :load_observables,
    :observed_targets,
)

for name in required_symbols
    isdefined(M, name) || error("Missing expected symbol: $name")
end


M.DEFAULT_LOSVD_FIT_STATISTIC === :legacy_chi2 || error("Expected DEFAULT_LOSVD_FIT_STATISTIC=:legacy_chi2, got $(M.DEFAULT_LOSVD_FIT_STATISTIC)")

A_losvd = [0.6 0.4; 0.4 0.6]
w = [0.5, 0.5]
target = [0.5, 0.5]
sigma = [0.1, 0.1]

state = M.losvd_chi2_state(A_losvd, w, target, sigma, 1, 2)
isfinite(state.chi_total) || error("chi-square state returned nonfinite chi_total")
abs(state.chi_total) <= 1.0e-12 || error("toy chi-square state should be zero, got $(state.chi_total)")

wp = [1.0, 1.0]
entropy = M.entropy_value(w, wp)
isfinite(entropy) || error("entropy_value returned nonfinite result")

wpdiag = M.wphase_diagnostics(wp)
wpdiag.convention === :inverse_phase_volume || error("Unexpected wphase convention: $(wpdiag.convention)")

println("[SELF CHECK] required_symbols=true")
println("[SELF CHECK] default_fit_statistic=", M.DEFAULT_LOSVD_FIT_STATISTIC)
println("[SELF CHECK] toy_chi2=", state.chi_total)
println("[SELF CHECK] entropy=", entropy)
println("[SELF CHECK] PASS")
