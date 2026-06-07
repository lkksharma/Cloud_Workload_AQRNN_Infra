"""
dp_sweep_run.py
===============
Runs federated_solution_v2 with a specific DP noise_mult passed via CLI.
Saves results to a uniquely named JSON so multiple runs can be plotted together.

Usage:
    python dp_sweep_run.py 0.15
    python dp_sweep_run.py 0.05
"""
import sys
import federated_solution2 as fv2

if len(sys.argv) < 2:
    print("Usage: python dp_sweep_run.py <noise_mult>")
    print("Example: python dp_sweep_run.py 0.15")
    sys.exit(1)

noise_mult = float(sys.argv[1])

# Patch the config before running
fv2.DP_NOISE_MULT = noise_mult

# Scale server_lr inversely with noise: less noise → can afford bigger steps
if noise_mult <= 0.05:
    fv2.SERVER_LR = 0.5
    fv2.DP_CLIP_NORM = 0.7
elif noise_mult <= 0.15:
    fv2.SERVER_LR = 0.45
    fv2.DP_CLIP_NORM = 0.7
else:
    fv2.SERVER_LR = 0.4
    fv2.DP_CLIP_NORM = 0.7

# Override output filenames so they don't clobber each other
tag = f"nm{noise_mult:.2f}".replace(".", "")
fv2.RESULTS_PATH = f"federated_v2_results_{tag}.pkl"

print(f"\n{'='*60}")
print(f"  DP SWEEP: noise_mult={noise_mult}, server_lr={fv2.SERVER_LR}")
print(f"  Output: federated_v2_results_{tag}.json")
print(f"{'='*60}\n")

# Run the simulation
fv2.run_simulation_v2()

# After run completes, rename the JSON too
import os, shutil
if os.path.exists("federated_v2_results.json"):
    shutil.move("federated_v2_results.json", f"federated_v2_results_{tag}.json")
    print(f"\nRenamed results → federated_v2_results_{tag}.json")