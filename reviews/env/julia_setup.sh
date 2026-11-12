# Run inside a job after env/env.sh. One-time online Julia resolve, then offline checks.
set -euo pipefail
if [ ! -f "$JULIA_DEPOT_PATH/.mwr_ready" ]; then
  rm -rf "$VIRTUAL_ENV/julia_env" "$JULIA_DEPOT_PATH"
  mkdir -p "$JULIA_DEPOT_PATH"
fi
# gcc/11 ships libzstd 1.5.0, which breaks Julia's bundled zstd during registry download.
# julia_snippets/*.jl do `using DynamicExpressions, LossFunctions, Statistics`: pin them to the
# versions SymbolicRegression 1.11.3 resolves to, as top-level deps of the venv project.
env -u LD_LIBRARY_PATH PYTHON_JULIAPKG_OFFLINE=no python - <<'PY'
import juliapkg
juliapkg.require_julia("~1.11")
juliapkg.add("DynamicExpressions", "a40a106e-89c9-4ca8-8020-a735e8728b6b", version="=1.10.4")
juliapkg.add("LossFunctions", "30fc2ffe-d236-52d8-8643-a9d8f7c094a7", version="=1.2.2")
juliapkg.add("Statistics", "10745b16-79ce-11e8-11f9-7d13ad32a3b2", version="=1.11.5")
juliapkg.resolve()
print("julia exe:", juliapkg.executable())
PY

echo "=== offline PySR fit (precompile at the depot)"
PYTHON_JULIAPKG_OFFLINE=yes python - <<'PY'
import numpy, pandas
from pysr import PySRRegressor, jl
print("julia", jl.seval("string(VERSION)"), "threads", jl.seval("Threads.nthreads()"))
print("SymbolicRegression", jl.seval("string(pkgversion(SymbolicRegression))"))
import numpy as np
X = np.random.randn(64, 2); y = X[:, 0] * 2 + X[:, 1]
m = PySRRegressor(niterations=2, populations=2, binary_operators=["+", "*"], progress=False,
                  verbosity=0, temp_equation_file=True)
m.fit(X, y)
print("pysr fit ok:", m.get_best().equation)
import glob, os
for f in sorted(glob.glob(os.path.join(os.environ["MWR_SRC"], "conj_refute_env/julia_snippets/*.jl"))):
    jl.include(f)
    print("included", os.path.basename(f))
print("DynamicExpressions", jl.seval("string(pkgversion(DynamicExpressions))"),
      "LossFunctions", jl.seval("string(pkgversion(LossFunctions))"),
      "SymbolicRegression", jl.seval("string(pkgversion(SymbolicRegression))"))
PY

echo "=== offline import order check (pysr -> torch -> agilerl)"
PYTHON_JULIAPKG_OFFLINE=yes python - <<'PY'
import os, subprocess
import numpy, pandas
from pysr import PySRRegressor, jl
print("julia 1+1 =", jl.seval("1+1"))
import torch, agilerl, leanclient, wandb
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "agilerl", agilerl.__version__ if hasattr(agilerl, "__version__") else "?")
libs = subprocess.run(["grep", "-o", "/[^ ]*libstdc++.so.6[^ ]*", f"/proc/{os.getpid()}/maps"],
                      capture_output=True, text=True).stdout.split()
print("libstdc++ in process:", sorted(set(libs)))
print("ENV OK")
PY
touch "$JULIA_DEPOT_PATH/.mwr_ready" "$VIRTUAL_ENV/.mwr_ready"
