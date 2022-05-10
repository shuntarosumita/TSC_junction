using LinearAlgebra
using LaTeXStrings
using Printf
import Plots

########## main ##########
##### φ dependence #####
function eigenφdep(t::Float64, εN::Float64, q0::Int, title::AbstractString="", plotlims::Float64=2.2)
    # calculation
    Lφ = 64
    φs = [Float64(2π * nφ / Lφ) for nφ in -Lφ:Lφ]
    eige::Vector{Float64} = [sqrt((εN)^2 + 2 * t^2 * (1 + (-1)^q0 * cos(φ/2))) for φ in φs]
    eigo::Vector{Float64} = [sqrt((εN)^2 + 2 * t^2 * (1 - (-1)^q0 * cos(φ/2))) for φ in φs]
    # gs::Vector{Float64} = vec(minimum(hcat([-eige, -eigo]...), dims=2))

    # plot data
    plt = Plots.plot(φs, zeros(Float64, length(φs)), linewidth=0.5, linecolor=:black, label=:none)
    # Plots.plot!(plt, φs, gs, color=:green, linealpha=0.3, linewidth=8.0, label=:none)
    Plots.plot!(plt, φs, [eige -eige], color=:blue, linewidth=1.5, label=["pFP even" :none])
    Plots.plot!(plt, φs, [eigo -eigo], color=:red, linestyle=:dash, linewidth=1.5, label=["pFP odd" :none])
    Plots.plot!(plt, xlims=(-2π, 2π), ylims=(-plotlims, plotlims), size=(400, 300),
        xticks=([-2π:π:2π;], ["−2\\pi", "−\\pi", "0", "\\pi", "2\\pi"]),
        title=title, titleloc=:left, titlefont=font(16),
        legendfont=font(10), guidefont=font(12), tickfont=font(10),
        xlabel=L"\varphi_{\mathrm{R}} - \varphi_{\mathrm{L}}", ylabel=L"E / \,|\!\!\tilde{t}\,|")

    # data output
    filename = @sprintf("effective_phidep_t%.3f_e%.3f_q%01d", t, εN, q0)
    # isdir("data") || mkdir("data")
    # datafile = File(format"JLD2", "data/"*filename*".jld2")
    # save(datafile, "φs", φs, "eigsols", eigsols)
    isdir("images") || mkdir("images")
    Plots.savefig(plt, "images/"*filename*".pdf")

    display(plt)
end
