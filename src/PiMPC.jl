module PiMPC

using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# Optional CUDA support (weak dependency)
# ---------------------------------------------------------------------------
# CUDA is listed in [weakdeps] so it is never required.  We attempt to load
# it at module initialisation; if unavailable the GPU solver is silently
# disabled and every setup! / solve! call that requests :gpu falls back to
# :cpu with a warning.
# ---------------------------------------------------------------------------

const _CUDA_LOADED = Ref{Bool}(false)

function __init__()
    pkg = Base.identify_package("CUDA")
    if pkg !== nothing
        try
            mod = Base.require(pkg)
            # Bind the loaded module as `PiMPC.CUDA` (used dotted, e.g.
            # `CUDA.zeros`, `CUDA.norm`) and the bare `cu` constructor used
            # throughout `_solve_gpu`.  `Base.require` only loads — it does
            # not create bindings in the caller.
            Core.eval(PiMPC, :(const CUDA = $mod))
            Core.eval(PiMPC, :(const cu   = $(mod).cu))
            _CUDA_LOADED[] = true
        catch
        end
    end
end

# Convenience: true iff CUDA was loaded *and* a functional GPU is present.
_cuda_ok() = _CUDA_LOADED[] && isdefined(PiMPC, :CUDA) && CUDA.functional()

include("problem.jl")
include("solver.jl")

export Model, setup!, solve!, Results

end
