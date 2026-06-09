# OSPM/AI_defaults.py
#
# Pass-through defaults layer.
#
# Galaxy configs are now authoritative.  This file intentionally does not
# inject mode, halo, orbit, deck, solver, or AI settings into every galaxy.
#
# load_config.py still merges:
#     cfg = {**AI_DEFAULTS, **mod.CONFIG}
#
# With CONFIG empty, that merge preserves the existing call contract while
# preventing old Segue1-style defaults from overwriting or backfilling the
# active galaxy configuration.

CONFIG = {}