using LinearAlgebra
using JLD2, FileIO
using DelimitedFiles
using Printf
using Plots
using Plots.PlotMeasures
using LaTeXStrings
const σ0 = ComplexF64[1 0; 0 1]
const σ1 = ComplexF64[0 1; 1 0]
const σ2 = ComplexF64[0 -1im; 1im 0]
const σ3 = ComplexF64[1 0; 0 -1]

# Hamiltonian
h0(μ::Float64) = - μ * σ3
h1s(t::Float64, Δ::Float64, φ::Float64) = 0.5 * (- t * σ3 + 1im * Δ * (cos(φ) * σ2 + sin(φ) * σ1))
h1n(t::Float64) = - 0.5 * t * σ3

# bulk solutions for two superconductors
function bulksols(ε::Float64, t::Float64, μ::Float64, Δ::Float64, φ::Float64)
    zs1 = Vector{ComplexF64}(undef, 2)
    zs2 = Vector{ComplexF64}(undef, 2)
    us1 = Matrix{ComplexF64}(undef, 2, 2)
    us2 = Matrix{ComplexF64}(undef, 2, 2)

    if t != Δ
        ws = [(Complex(μ*t) + (-1)^(l-1) * sqrt(Complex(μ^2 * Δ^2 + (t^2 - Δ^2) * (ε^2 - Δ^2)))) / (t^2 - Δ^2) for l in 1:2]
        zs::Vector{ComplexF64} = (
            [- ws[l] + (-1)^(m-1) * sqrt(ws[l]^2 - 1) for l in 1:2, m in 1:2]
            |> (x -> vcat(x...))
        )
        us::Matrix{ComplexF64} = (
            [normalize([Δ * sqrt(ws[l]^2 - 1), (-1)^(m-1) * (μ + ε - t * ws[l])]) for l in 1:2, m in 1:2]
            |> (x -> hcat(x...))
        )
        sortidx = sortperm(zs, by=abs, rev=true)
        zs = zs[sortidx]
        us = us[:, sortidx]; us[2, :] .*= [exp(-0.5im*φ), exp(-0.5im*φ), exp(0.5im*φ), exp(0.5im*φ)]
        zs1, zs2 = zs[1:2], zs[3:4]
        us1, us2 = us[:, 1:2], us[:, 3:4]
    else
        zsext::Vector{ComplexF64} = (
            [(- Complex(μ^2 + t^2 - ε^2) + (-1)^(l-1) * sqrt(Complex((μ^2 + t^2 - ε^2)^2 - 4 * μ^2 * t^2))) / (2 * μ * t) for l in 1:2]
        )
        usext::Matrix{ComplexF64} = (
            [normalize([sqrt(Complex((μ^2 + t^2 - ε^2)^2 - 4 * μ^2 * t^2)), (-1)^(l-1) * ((ε + μ)^2 - t^2)]) for l in 1:2]
            |> (x -> hcat(x...))
        )
        sortidx = sortperm(zsext, by=abs, rev=true)
        zsext = zsext[sortidx]
        usext = usext[:, sortidx]
        zs1, zs2 = [zsext[1], 0.0+0.0im], [zsext[2], 0.0+0.0im]
        us1, us2 = [usext[1, 1] 2^(-0.5); usext[2, 1] 2^(-0.5)], [usext[1, 2] 2^(-0.5); usext[2, 2] -2^(-0.5)]
        us1[2, :] *= exp(-0.5im*φ); us2[2, :] *= exp(0.5im*φ)
    end

    zs1, zs2, us1, us2
end

# bulk solutions for a normal metal
function bulksoln(ε::Float64, t::Float64, μ::Float64)
    wn = [Complex((μ + (-1)^(l-1) * ε) / t) for l in 1:2]
    zn::Vector{ComplexF64} = (
        [- wn[l] + (-1)^(m-1) * sqrt(wn[l]^2 - 1) for l in 1:2, m in 1:2]
        |> (x -> vcat(x...))
    )

    un::Matrix{ComplexF64} = [1.0+0.0im 0.0+0.0im 1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im 0.0+0.0im 1.0+0.0im]
    zn, un
end

# interface (boundary) matrix
function interfacemat(ε::Float64, param::Vector{Float64})
    t, λ, μ, Δ, φ, L = param
    zs1, zs2, us1, us2 = bulksols(ε, t, μ, Δ, φ)
    zn, un = bulksoln(ε, t, μ)

    # construct interface matrix
    imat = zeros(ComplexF64, 8, 8)
    # s1 part
    for l in 1:2
        imat[1:2, l] = ifelse((t == Δ && l == 2), (h0(μ) - ε * σ0), (- zs1[l] * h1s(t, Δ, +φ/2))) * us1[:, l]
        imat[3:4, l] = λ * h1n(t)' * us1[:, l]
    end
    # n part
    for l in 1:4
        imat[1:2, l+2] = zn[l] * λ * h1n(t) * un[:, l]
        imat[3:4, l+2] = - h1n(t)' * un[:, l]
        imat[5:6, l+2] = - zn[l]^(L+1) * h1n(t) * un[:, l]
        imat[7:8, l+2] = zn[l]^L * λ * h1n(t)' * un[:, l]
    end
    # s2 part
    for l in 1:2
        imat[5:6, l+6] = λ * h1n(t) * us2[:, l]
        imat[7:8, l+6] = ifelse((t == Δ && l == 2), (h0(μ) - ε * σ0), (- zs2[l]^(-1) * h1s(t, Δ, -φ/2)')) * us2[:, l]
    end

    # construct Gramian matrix
    gmat = zeros(ComplexF64, 8, 8)
    # s1 part
    for l1 in 1:2, l2 in 1:2
        gmat[l1, l2] = dot(us1[:, l1], us1[:, l2])
        t == Δ || (gmat[l1, l2] /= (1 - (conj(zs1[l1]) * zs1[l2])^(-1)))
    end
    # n part
    for l1 in 1:4, l2 in 1:4
        if l1 % 2 == l2 % 2
            if conj(zn[l1]) * zn[l2] != ComplexF64(1)
                gmat[l1+2, l2+2] = (conj(zn[l1]) * zn[l2] - (conj(zn[l1]) * zn[l2])^(L+1)) / (1 - conj(zn[l1]) * zn[l2])
            else
                gmat[l1+2, l2+2] = L
            end
        end
    end
    # s2 part
    for l1 in 1:2, l2 in 1:2
        gmat[l1+6, l2+6] = dot(us2[:, l1], us2[:, l2])
        t == Δ || (gmat[l1+6, l2+6] /= (1 - conj(zs2[l1]) * zs2[l2]))
    end

    imat, gmat
end

# homotopy method
function homotopy(εinit::Float64, param::Vector{Float64}, dε::Float64=1.0e-10, niter::Int=100)
    ε::Float64 = εinit
    imat, gmat = interfacemat(ε, param)
    absdetI0 = abs(det(imat) / sqrt(det(gmat)))
    for count = 1:niter
        imat, gmat = interfacemat(ε, param)
        imatp, gmatp = interfacemat(ε + dε, param)
        diffabsdetI = (abs(det(imatp) / sqrt(det(gmatp))) - abs(det(imat) / sqrt(det(gmat)))) / dε
        ε -= ifelse(diffabsdetI != 0, absdetI0 / (niter * diffabsdetI), dε * (rand(Float64) - 0.5))
    end
    ε
end

# newton method
function newton(εinit::Float64, param::Vector{Float64}, dε::Float64=1.0e-10, niter::Int=100)
    ε::Float64 = εinit
    conv::Bool = false
    for count = 1:niter
        εold = ε
        imat, gmat = interfacemat(ε, param)
        imatp, gmatp = interfacemat(ε + dε, param)
        (abs(det(gmat)) == ComplexF64(0) || abs(det(gmatp)) == ComplexF64(0)) && break

        absdetI = abs(det(imat) / sqrt(det(gmat)))
        diffabsdetI = (abs(det(imatp) / sqrt(det(gmatp))) - absdetI) / dε
        ε -= ifelse(diffabsdetI != 0, absdetI / diffabsdetI, dε * (rand(Float64) - 0.5))

        if abs(ε - εold) <= max(1.0e-8 * abs(εold), 1.0e-16)
            conv = true
            break
        end
    end
    ε, conv
end

# solve eigenproblem
function solveeigen(param::Vector{Float64})
    # println(param)
    t, λ, μ, Δ, φ, L = param
    εarr = Array{Float64}(undef, 0)
    coeff = Matrix{ComplexF64}(undef, 0, 0)
    coeffarr = Array{Matrix{ComplexF64}}(undef, 0)
    zarr = Array{Vector{ComplexF64}}(undef, 0)
    uarr = Array{Matrix{ComplexF64}}(undef, 0)

    εzerocheck::Bool = false
    nsols::Int = 0
    nduplicate::Int = 0
    # ncount::Int = 0
    while nsols < 8 && nduplicate < 200
        # ncount += 1
        # ncount % 100 == 0 && println(ncount, ", ", nsols, ", ", nduplicate)
        ε = ifelse(εzerocheck, rand([t*λ])*randn(Float64), 0.0)
        εzerocheck = true
        # ε = homotopy(ε, param)
        ε, conv = newton(ε, param)
        conv || continue

        imat = Matrix{ComplexF64}(interfacemat(ε, param)[1])
        try
            coeff = nullspace(imat, atol=1.0e-8)
        catch
            println("catch exception!")
            continue
        end
        if size(coeff, 2) > 0
            if !(any(abs.(εarr .- ε) .<= max(1.0e-6 * abs(ε), 1.0e-16)))
                nsols += size(coeff, 2)
                zs1, zs2, us1, us2 = bulksols(ε, t, μ, Δ, φ)
                zn, un = bulksoln(ε, t, μ)
                push!(εarr, ε)
                push!(coeffarr, coeff)
                push!(zarr, vcat(zs1, zn, zs2))
                push!(uarr, hcat(us1, un, us2))
            else
                nduplicate += 1
            end
        end
    end

    sortidx = sortperm(εarr)
    εarr[sortidx], coeffarr[sortidx], zarr[sortidx], uarr[sortidx]
end

# 2nd-order perturbation theory for off-resonant case
function perturb2ndamp(μbar::Float64, Δbar::Float64, L::Int)
    if any(μbar .≈ [- cos(π*q/(L+1)) for q in 1:L]) || (μbar == 0.0 && L%2 == 1)
        return(Inf)
    end

    zs1::ComplexF64 = (-μbar + sqrt(Complex(μbar^2 + Δbar^2 - 1))) / (1 + Δbar)
    zs2::ComplexF64 = (-μbar - sqrt(Complex(μbar^2 + Δbar^2 - 1))) / (1 + Δbar)
    norm2s::Float64 = 1 / sum(2 * abs2(zs1^j - zs2^j) for j in 1:10000)
    norm2n::Float64 = 2 / (L + 1)
    qsum::Float64 = sum(2 * (-1)^q * (sin(π*q/(L+1)))^2 / ((μbar) + cos(π*q/(L+1))) for q in 1:L)

    norm2s * norm2n * (abs(μbar^2 + Δbar^2 - 1) / (1 + Δbar)^2) * qsum
end

function perturb2ndenemin(param::Vector{Float64})
    t, λ, μ, Δ, φ, L = param
    abs(λ^2 * t * perturb2ndamp(μ/t, Δ/t, Int(L)) * cos(φ/2))
end

# 1st-order perturbation theory for resonant case
function perturb1stamp(q0::Int, Δbar::Float64, L::Int)
    μbar = - cos(π*q0/(L+1))
    zs1::ComplexF64 = (-μbar + sqrt(Complex(μbar^2 + Δbar^2 - 1))) / (1 + Δbar)
    zs2::ComplexF64 = (-μbar - sqrt(Complex(μbar^2 + Δbar^2 - 1))) / (1 + Δbar)
    norms::Float64 = sqrt(1 / sum(2 * abs2(zs1^j - zs2^j) for j in 1:10000))
    normn::Float64 = sqrt(2 / (L + 1))

    2 * norms * normn * (sqrt(abs(Δbar^2 - (sin(π*q0/(L+1)))^2)) / (1 + Δbar)) * sin(π*q0/(L+1))
end

function perturb1stenemin(param::Vector{Float64})
    t, λ, q0, Δ, φ, L = param
    phasepart::Float64 = min(abs(cos((φ+π)/4)), abs(sin((φ+π)/4)))
    abs(λ * t * perturb1stamp(Int(q0), Δ/t, Int(L)) * phasepart)
end


########## main ##########
##### φ dependence #####
function calcφdep(t::Float64, λ::Float64, μ::Float64, Δ::Float64, L::Int)
    # set parameters: t, λ, μ, Δ, φ, L
    Lφ = 128
    φs = [Float64(2π * nφ / Lφ) for nφ in -Lφ:(Lφ-1)]
    params::Array{Float64, 2} = hcat([[t, λ, μ, Δ, φ, L] for φ in φs]...)

    # calculation
    eigsols::Array{Tuple{Array{Float64}, Array{Matrix{ComplexF64}},
        Array{Vector{ComplexF64}}, Array{Matrix{ComplexF64}}}, 2} =
    mapslices(solveeigen, params, dims=[1])

    # data output
    isdir("data") || mkdir("data")
    filename = "data/" * @sprintf("phidep_t%.3f_l%.3f_m%.3f_d%.3f_L%03d.jld2", t, λ, μ, Δ, L)
    datafile = File(format"JLD2", filename)
    save(datafile, "φs", φs, "eigsols", eigsols)

    φs, eigsols
end

function plotφdep(t::Float64, λ::Float64, μ::Float64, Δ::Float64, L::Int, ylim::Float64, title::AbstractString="")
    filename = @sprintf("phidep_t%.3f_l%.3f_m%.3f_d%.3f_L%03d", t, λ, μ, Δ, L)
    isfile("data/"*filename*".jld2") || error("error: file does not exist. Make a calculation first.")
    data = load("data/"*filename*".jld2")
    φs = data["φs"]
    eigsols = data["eigsols"]

    # plot data
    plt = plot(φs, zeros(Float64, length(φs)), linewidth=0.5, linecolor=:black, label=:none)
    for i in 1:length(φs)
        scatter!(plt, fill(φs[i], length(eigsols[1, i][1])), eigsols[1, i][1], label=:none, markercolor=:red, markersize=2, markerstrokecolor=:red, markerstrokewidth=0)
    end
    plot!(plt, xlims=(-2π, 2π), ylims=(-ylim, ylim), size=(400, 300),
        xticks=([-2π:π:2π;], ["−2\\pi", "−\\pi", "0", "\\pi", "2\\pi"]), yformatter=:scientific,
        title=title, titleloc=:left, titlefont=font(16),
        guidefont=font(12), tickfont=font(8),
        xlabel=L"\varphi_{\mathrm{R}} - \varphi_{\mathrm{L}}", ylabel=L"\varepsilon / t"
    )

    isdir("images") || mkdir("images")
    savefig(plt, "images/"*filename*".pdf")
    display(plt)
end

##### μ dependence #####
function calcμdep(t::Float64, λ::Float64, Δ::Float64, φ::Float64, L::Int)
    # set parameters: t, λ, μ, Δ, φ, L
    μs = -0.99:0.01:0.99
    params::Array{Float64, 2} = hcat([[t, λ, μ, Δ, φ, L] for μ in μs]...)

    # calculation
    eigsols::Array{Tuple{Array{Float64}, Array{Matrix{ComplexF64}},
        Array{Vector{ComplexF64}}, Array{Matrix{ComplexF64}}}, 2} =
    mapslices(solveeigen, params, dims=[1])

    # data output
    isdir("data") || mkdir("data")
    filename = @sprintf("data/mudep_t%.3f_l%.3f_d%.3f_p%.3f_L%03d.jld2", t, λ, Δ, φ, L)
    datafile = File(format"JLD2", filename)
    save(datafile, "μs", μs, "eigsols", eigsols)

    μs, eigsols
end

function calcμdepperturb(t::Float64, λ::Float64, Δ::Float64, φ::Float64, L::Int)
    # 1st-order perturbation
    # set parameters: t, λ, q0, Δ, φ, L
    q0s = 1:L
    params1::Array{Float64, 2} = hcat([[t, λ, q0, Δ, φ, L] for q0 in q0s]...)
    # calculation
    eigsols1::Vector{Float64} = vec(mapslices(perturb1stenemin, params1, dims=[1]))

    # 2nd-order perturbation
    # set parameters: t, λ, μ, Δ, φ, L
    μs = -0.99:0.01:0.99
    params2::Array{Float64, 2} = hcat([[t, λ, μ, Δ, φ, L] for μ in μs]...)
    # calculation
    eigsols2::Vector{Float64} = vec(mapslices(perturb2ndenemin, params2, dims=[1]))

    # data output
    isdir("data") || mkdir("data")
    filename = @sprintf("mudep_t%.3f_l%.3f_d%.3f_p%.3f_L%03d.jld2", t, λ, Δ, φ, L)
    datafile1 = File(format"JLD2", "data/p1_" * filename)
    datafile2 = File(format"JLD2", "data/p2_" * filename)
    save(datafile1, "q0s", q0s, "eigsols", eigsols1)
    save(datafile2, "μs", μs, "eigsols", eigsols2)

    q0s, eigsols1, μs, eigsols2
end

function plotμdep(t::Float64, λ::Float64, Δ::Float64, φ::Float64, L::Int, title::AbstractString="", ylims::Tuple{Float64, Float64}=(1.0e-7, 1.0e-2))
    filename = @sprintf("mudep_t%.3f_l%.3f_d%.3f_p%.3f_L%03d", t, λ, Δ, φ, L)
    datafiles = ["data/p1_"*filename*".jld2", "data/p2_"*filename*".jld2", "data/"*filename*".jld2"]
    dataexist = isfile.(datafiles)
    all(dataexist) || error("error: the following file(s) do not exist. Make a calculation first.")
    data = load.(datafiles)
    μs1 = - cos.(π.*data[1]["q0s"]./(L+1))
    eigsols1 = data[1]["eigsols"] ./ t
    μs2 = data[2]["μs"] ./ t
    eigsols2 = data[2]["eigsols"] ./ t
    μse = data[3]["μs"] ./ t
    eigsolse = data[3]["eigsols"]
    mineigens = [minimum(abs.(eigsolse[1, nμ][1]) ./ t) for nμ in 1:length(μse)]

    # plot exact data
    plt = plot(μs1, seriestype=:vline, label=:none, linewidth=0.5, linecolor=:green,
        yaxis=:log10, xlims=(-1, 1), ylims=ylims, size=(600, 350),
        title=title, titleloc=:left, titlefont=font(24), top_margin=4mm,
        legendfont=font(10), guidefont=font(16), tickfont=font(10),
        xlabel=L"\mu / t", ylabel=L"\varepsilon / t"
    )
    plot!(plt, μse, mineigens, label="exact", legend=:bottomright,
        linewidth=2, linecolor=:red, markersize=3, markercolor=:red, markershape=:circle
    )

    # plot the results of perturbation theory
    scatter!(plt, μs1, eigsols1, label="1st perturbation", markersize=5, markercolor=:green, markershape=:utriangle)
    scatter!(plt, μs2, eigsols2, label="2nd perturbation", markersize=3, markercolor=:blue, markershape=:xcross)

    # output
    isdir("images") || mkdir("images")
    savefig(plt, "images/"*filename*".pdf")
    display(plt)
end

##### Δ dependence #####
function calcΔdep(t::Float64, λ::Float64, μ::Float64, φ::Float64, L::Int)
    # set parameters: t, λ, μ, Δ, φ, L
    Δs = 0.1:0.02:1.1
    params::Array{Float64, 2} = hcat([[t, λ, μ, Δ, φ, L] for Δ in Δs]...)

    # calculation
    eigsols::Array{Tuple{Array{Float64}, Array{Matrix{ComplexF64}},
        Array{Vector{ComplexF64}}, Array{Matrix{ComplexF64}}}, 2} =
    mapslices(solveeigen, params, dims=[1])

    # data output
    isdir("data") || mkdir("data")
    filename = "data/" * @sprintf("deltadep_t%.3f_l%.3f_m%.3f_p%.3f_L%03d.jld2", t, λ, μ, φ, L)
    datafile = File(format"JLD2", filename)
    save(datafile, "Δs", Δs, "eigsols", eigsols)

    Δs, eigsols
end

function plotΔdep(t::Float64, λ::Float64, μ::Float64, φ::Float64, L::Int)
    filename = @sprintf("deltadep_t%.3f_l%.3f_m%.3f_p%.3f_L%03d", t, λ, μ, φ, L)
    isfile("data/"*filename*".jld2") || error("error: file does not exist. Make a calculation first.")
    data = load("data/"*filename*".jld2")
    Δs = data["Δs"] ./ t
    eigsols = data["eigsols"]

    # plot data
    mineigens = [minimum(abs.(eigsols[1, nΔ][1]))/t for nΔ in 1:length(Δs)]
    plt = plot(Δs, mineigens, yaxis=:log10, label=:none, xlabel="Δ/t", ylabel="ε/t",
        linewidth=2, linecolor=:blue,
        markersize=3, markercolor=:blue, markershape=:circle
    )
    plot!(plt, [sqrt(1-(μ/t)^2)], seriestype=:vline, label=:none,
        linewidth=0.5, linecolor=:red
    )

    isdir("images") || mkdir("images")
    savefig(plt, "images/"*filename*".pdf")
    display(plt)
end

##### μ-Δ phase diagram #####
# function calcμΔdep(t::Float64, λ::Float64, L::Int)
#     # set parameters: t, λ, μ, Δ, φ, L
#     μs = -0.96:0.04:0.96
#     Δs = 0.04:0.04:1.2
#     params::Array{Float64, 3} = reshape(hcat([[t, λ, μ, Δ, 0.0, L] for μ in μs, Δ in Δs]...), 6, length(μs), length(Δs))

#     # calculation
#     eigsols::Array{Tuple{Array{Float64}, Array{Matrix{ComplexF64}},
#         Array{Vector{ComplexF64}}, Array{Matrix{ComplexF64}}}, 3} =
#     mapslices(solveeigen, params, dims=[1])

#     # data output
#     isdir("data") || mkdir("data")
#     filename = "data/" * @sprintf("mudeltadep_t%.3f_l%.3f_L%03d.jld2", t, λ, L)
#     datafile = File(format"JLD2", filename)
#     save(datafile, "μs", μs, "Δs", Δs, "eigsols", eigsols)

#     μs, Δs, eigsols
# end

# function plotμΔdep(t::Float64, λ::Float64, L::Int)
#     filename = @sprintf("mudeltadep_t%.3f_l%.3f_L%03d", t, λ, L)
#     isfile("data/"*filename*".jld2") || error("error: file does not exist. Make a calculation first.")
#     data = load("data/"*filename*".jld2")
#     μs = data["μs"] ./ t
#     Δs = data["Δs"] ./ t
#     eigsols = data["eigsols"]

#     # plot data
#     mineigens = [minimum(abs.(eigsols[1, nμ, nΔ][1])) for nμ in 1:length(μs), nΔ in 1:length(Δs)]
#     plt = heatmap(μs, Δs, transpose(mineigens), clim=(0.0,0.001), aspect_ratio=:equal, size=(600, 400),
#         xlabel="μ/t", ylabel="Δ/t")

#     isdir("images") || mkdir("images")
#     savefig(plt, "images/"*filename*".pdf")
#     display(plt)
# end

function calcμΔdepperturb2nd(L::Int)
    # set parameters: t, λ, μ, Δ, φ, L
    μbars = -0.999:0.001:0.999
    Δbars = 0.01:0.01:1.5
    params::Array{Float64, 3} = reshape(hcat([[μbar, Δbar, L] for μbar in μbars, Δbar in Δbars]...),
        3, length(μbars), length(Δbars))

    # calculation
    amps::Array{Float64, 2} = mapslices(v -> perturb2ndamp(v[1], v[2], Int(v[3])), params, dims=[1])[1, :, :]

    # data output
    isdir("data") || mkdir("data")
    filename = @sprintf("data/p2_mudeltadep_L%03d.jld2", L)
    datafile = File(format"JLD2", filename)
    save(datafile, "μbars", μbars, "Δbars", Δbars, "amps", amps)

    μbars, Δbars, amps
end

function plotμΔdepperturb2nd(L::Int, title::AbstractString="")
    filename = @sprintf("p2_mudeltadep_L%03d", L)
    isfile("data/"*filename*".jld2") || error("error: file does not exist. Make a calculation first.")
    data = load("data/"*filename*".jld2")
    μbars, Δbars, amps = data["μbars"], data["Δbars"], data["amps"]

    # plot data
    xlims = (-1, 1)
    clims = (-4, 4)
    map = heatmap(μbars, Δbars, transpose(amps), #aspect_ratio=:equal,
        xlims=xlims, ylims=(0, 1.5), clims=clims,
        title=title, titleloc=:left, titlefont=font(24),
        guidefont=font(16), tickfont=font(10),
        ylabel=L"\Delta / t",
        color=:bwr, colorbar=false
    )
    μs1 = [- cos(π*q0/(L+1)) for q0 in 1:L]
    plot!(map, μs1, seriestype=:vline, label=:none, linewidth=2.0, linecolor=:green)
    plot!(map, [0.8], seriestype=:hline, label=:none, linewidth=1.0, linecolor=:black, linestyle=:dash, xformatter=_->"")

    colorbar = scatter([0, 0], [0, 1], zcolor=[0, 1], xlims=(1, 1.1), clims=clims, 
        framestyle=:none, label=:none, color=:bwr, colorbar_title=L"A^{(2)}",
        guidefont=font(16), tickfont=font(10),
    )

    idx = findfirst(isequal(0.8), Δbars)
    # graph = plot(μbars, abs.(amps[:, idx]), xlims=xlims, ylims=(1e-2, 1e0), yaxis=:log10,
    #     linewidth=1, linecolor=:blue, markersize=3, markercolor=:blue, markershape=:xcross,
    #     label=:none, xlabel=L"\mu / t", ylabel=L"|\!A^{(2)}\,|"
    # )
    graph = plot(μbars, amps[:, idx], xlims=xlims, ylims=(-3e0, 3e0),
        linewidth=0, linecolor=:blue,
        markersize=2, markercolor=:blue, markershape=:xcross,
        guidefont=font(16), tickfont=font(10),
        label=:none, xlabel=L"\mu / t", ylabel=L"A^{(2)}"
    )
    plot!(graph, μs1, seriestype=:vline, label=:none, linewidth=1, linecolor=:green)

    whitespace = plot(legend=false, grid=false, foreground_color_subplot=:white)

    layout = @layout(grid(2, 2, widths=[0.95, 0.05], heights=[0.7, 0.3]))
    plt = plot(map, colorbar, graph, whitespace, layout=layout, size=(600, 600))

    isdir("images") || mkdir("images")
    savefig(plt, "images/"*filename*".pdf")
    display(plt)
end
