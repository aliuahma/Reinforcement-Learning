using CSV
using DataFrames
using DataStructures
using LinearAlgebra
using Printf
using Statistics

mutable struct QLearning
    𝒮 # state space (assumes 1:nstates)
    𝒜 # action space (assumes 1:nactions)
    γ # discount
    Q # action value function
    α # learning rate
end

lookahead(model::QLearning, s, a) = model.Q[s,a]

function update!(model::QLearning, s, a, r, s′)
    γ, Q, α = model.γ, model.Q, model.α
    s, s′ = Int(s), Int(s′)
    Q[s,a] += α*(r + γ*maximum(Q[s′,:]) - Q[s,a])
    return model
end

function policy(model::QLearning, s)
    return argmax([lookahead(model, s, a) for a in model.𝒜])
end

function compute(model::QLearning, epochs::Int, infile, outfile)
    data = CSV.read(infile, DataFrame)

    println("Training on data...")
    for epoch in 1:epochs
        for row in eachrow(data)
            s, a, r, sp = row.s, row.a, row.r, row.sp
            update!(model, float(s), Int(a), float(r), float(sp))
        end
    end
    println("Training complete!")

    open(outfile, "w") do f
        for si in model.𝒮
            write(f, @sprintf("%d\n", policy(model, Int(si))))
        end
    end
end

small = QLearning(1:100, 1:4, 0.95, zeros(100,4), 0.1)
medium = QLearning(1:50000, 1:7, 1, zeros(50000,7), 0.1)
large = QLearning(1:302020, 1:9, 0.95, zeros(302020,9), 0.1)

println("small policy")
@time compute(small, 101, "data/small.csv", "policy/small.policy")
println("medium policy")
@time compute(medium, 1000, "data/medium.csv", "policy/medium.policy")
println("large policy")
@time compute(large, 101, "data/large.csv", "policy/large.policy")


"""
julia> include("src/main.jl")
small policy
Training on data...
Training complete!
  6.725500 seconds (121.08 M allocations: 2.110 GiB, 6.37% gc time, 0.94% compilation time)
medium policy
Training on data...
Training complete!
112.866150 seconds (3.20 G allocations: 55.139 GiB, 5.36% gc time)
large policy
Training on data...
Training complete!
 11.038448 seconds (332.05 M allocations: 5.920 GiB, 4.89% gc time)
"""