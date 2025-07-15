using Lux, DiffEqFlux, DifferentialEquations, Random, ComponentArrays, LinearAlgebra
using Optimization, OptimizationOptimJL, OptimizationOptimisers, Statistics

using Plots


Random.seed!(42)

# Generate synthetic data with Lotka-Volterra dynamics
function lotka_volterra!(du, u, p, t)
    α, β, γ, δ = p
    x, y = u
    
    du[1] = α * x - β * x * y      # prey equation
    du[2] = -δ * y + γ * x * y     # predator equation
end

# System parameters and initial conditions
params = [1.5, 1.0, 1.0, 3.0]  # α, β, γ, δ
u0 = [1.0, 1.0]                 # initial prey and predator populations
tspan = (0.0, 10.0)             # time span
dt = 0.1                        # time step

# Solve the true system
prob = ODEProblem(lotka_volterra!, u0, tspan, params)
sol = solve(prob, Tsit5(), saveat=dt)
true_data = Array(sol)
t_data = sol.t

println("Generated $(length(t_data)) data points")

# Create training dataset (first 30% of data)
train_fraction = 0.3
n_train = Int(round(length(t_data) * train_fraction))
t_train = t_data[1:n_train]
data_train = true_data[:, 1:n_train]

println("Training on $(n_train) points, testing on $(length(t_data) - n_train) points")

# Plot the full dataset
p1 = plot(t_data, true_data[1, :], label="Prey (True)", lw=2, color=:blue,
          xlabel="Time", ylabel="Population", title="Lotka-Volterra Dynamics")

          plot!(t_data, true_data[2, :], label="Predator (True)", lw=2, color=:red)
vline!([t_train[end]], label="Training End", color=:black, linestyle=:dash)
display(p1)
savefig(p1, "true_dynamics.png")
# Neural network setup
rng = Random.default_rng()

# Two separate networks for each interaction term
nn1 = Lux.Chain(Lux.Dense(2, 10, tanh), Lux.Dense(10, 1))  # for -βxy term
nn2 = Lux.Chain(Lux.Dense(2, 10, tanh), Lux.Dense(10, 1))  # for +γxy term


p1, st1 = Lux.setup(rng, nn1)
p2, st2 = Lux.setup(rng, nn2)

# Combine parameters
p_combined = ComponentArray(nn1=p1, nn2=p2)

# UDE system - replace interaction terms with neural networks
function ude_system!(du, u, p, t)
    x, y = u
    
    # Neural network predictions for interaction terms
    nn1_out = nn1([x, y], p.nn1, st1)[1][1]
    nn2_out = nn2([x, y], p.nn2, st2)[1][1]
    
    # UDE equations (keeping linear terms, learning interactions)
    du[1] = params[1] * x + nn1_out  # α*x + NN1(x,y) ≈ α*x - β*x*y
    du[2] = -params[4] * y + nn2_out # -δ*y + NN2(x,y) ≈ -δ*y + γ*x*y
end

# Prediction and loss functions
function predict_ude(θ)
    prob_ude = ODEProblem(ude_system!, u0, (0.0, t_train[end]), θ)
    sol_ude = solve(prob_ude, Tsit5(), p=θ, saveat=t_train, sensealg=InterpolatingAdjoint())
    return Array(sol_ude)
end

function loss_function(θ,p)
    pred = predict_ude(θ)
    return sum(abs2, data_train .- pred)
end

# Training callback
losses = Float64[]
function callback(θ, l)
    push!(losses, l)
    if length(losses) % 200 == 0
        println("Iteration $(length(losses)): Loss = $(round(l, digits=6))")
    end
    return false
end



# Stage 1: ADAM optimizer
adam_iters = 5000
optf = Optimization.OptimizationFunction((θ, p) -> loss_function(θ, p), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, p_combined)
res1 = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.01), 
                         callback=callback, maxiters=adam_iters)

println("ADAM completed. Loss: $(round(losses[end], digits=6))")

# Stage 2: BFGS optimizer for fine-tuning
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.BFGS(), callback=callback, maxiters=1000)

println("BFGS completed. Final loss: $(round(losses[end], digits=6))")

# Generate predictions
final_pred = predict_ude(res2.u)

# Plot training results
p2 = plot(t_train, data_train[1, :], label="Prey (Data)", 
          seriestype=:scatter, color=:blue, alpha=0.7)
plot!(t_train, data_train[2, :], label="Predator (Data)", 
      seriestype=:scatter, color=:red, alpha=0.7)
plot!(t_train, final_pred[1, :], label="Prey (UDE)", lw=2, color=:blue)
plot!(t_train, final_pred[2, :], label="Predator (UDE)", lw=2, color=:red)
title!("Training Results")
xlabel!("Time")
ylabel!("Population")
display(p2)
savefig(p2, "training_results.png") 

# Analyze learned interactions
true_interactions = zeros(2, length(t_train))
learned_interactions = zeros(2, length(t_train))

for i in 1:length(t_train)
    x, y = final_pred[1, i], final_pred[2, i]
    
    # True interaction terms
    true_interactions[1, i] = -params[2] * x * y  # -βxy
    true_interactions[2, i] = params[3] * x * y   # +γxy
    
    # Learned interaction terms
    learned_interactions[1, i] = nn1([x, y], res2.u.nn1, st1)[1][1]
    learned_interactions[2, i] = nn2([x, y], res2.u.nn2, st2)[1][1]
end

# Plot interaction comparison
p3 = plot(t_train, true_interactions[1, :], label="True -βxy", lw=2, color=:blue)
plot!(t_train, learned_interactions[1, :], label="Learned -βxy", lw=2, color=:lightblue, linestyle=:dash)
plot!(t_train, true_interactions[2, :], label="True +γxy", lw=2, color=:red)
plot!(t_train, learned_interactions[2, :], label="Learned +γxy", lw=2, color=:pink, linestyle=:dash)
title!("Interaction Terms Comparison")
xlabel!("Time")
ylabel!("Interaction Strength")
display(p3)
savefig(p3, "interaction_comparison.png")

p4 = plot(1:adam_iters, losses[1:adam_iters], label="ADAM", lw=2, color=:blue,
          yscale=:log10, xlabel="Iteration", ylabel="Loss", title="Training Loss")
if length(losses) > adam_iters
    plot!(adam_iters+1:length(losses), losses[adam_iters+1:end], label="BFGS", lw=2, color=:red)
end
display(p4)
savefig(p4, "TrainingLoss.png")