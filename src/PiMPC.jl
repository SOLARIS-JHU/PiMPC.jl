module PiMPC

using LinearAlgebra
using Printf
using CUDA

include("problem.jl")
include("solver.jl")

export Model, setup!, solve!, Results

end
