
using SparseArrays
using LinearAlgebra
using Clustering
using NearestNeighbors
using Distances
using Metis
using Laplacians
using Arpack
using Statistics
using DelimitedFiles
using StatsBase
using Random



include("Star.jl")
include("h_score3.jl")
include("mx_func.jl")
include("INC3.jl")
include("Filter_fast.jl")

function HyperEF(ar, L)
    

    #m, n = ar.shape
    #colPtr = Int[i+1 for i in PyArray(ar."indptr")]
    #rowVal = Int[i+1 for i in PyArray(ar."indices")]
    #nzVal = Vector{Int64}(PyArray(ar."data"))
    #X = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
    #ar = PyArray(ar)
    #X = nzVal

    ar_new = Any[]
    mx = mx_func(ar)
    idx_mat = Any[]
    println(typeof(ar))
    #if matrix make Vec of Vecs?
    println(typeof(ar[1]))
    println(typeof(ar[1][1]))
    println(typeof(mx))
    println(mx)
    Neff = zeros(Float64, mx)

    @inbounds for loop = 1:L
        W = ones(Float64, length(ar))
        mx = mx_func(ar)
        ## star expansion
        A = Star(ar)
        ## computing the smoothed vectors
        initial = 0
        SmS = 100
        interval = 1
        Nrv = 1
        RedR = 1
        Nsm = Int((SmS - initial) / interval)
        Ntot = Nrv * Nsm
        Qvec = zeros(Float64, 0)
        Eratio = zeros(Float64, length(ar), Ntot)
        SV = zeros(Float64, mx, Ntot)

        for ii = 1:Nrv
            sm = zeros(mx, Nsm)
            Random.seed!(1); randstring()
            rv = (rand(Float64, size(A, 1), 1) .- 0.5).*2
            sm = Filter_fast(rv, SmS, A, mx, initial, interval, Nsm)
            SV[:, (ii-1)*Nsm+1 : ii*Nsm] = sm

        end

        ## Make all the smoothed vectors orthogonal to each other
        QR = qr(SV)
        SV = Matrix(QR.Q)

        ## Computing the ratios using all the smoothed vectors
        for jj = 1:size(SV, 2)
            hscore = h_score3(ar, SV[:, jj])
            Eratio[:, jj] = hscore ./ sum(hscore)

        end #for jj

        ## Approximating the effective resistance of hyperedges by selecting the top ratio
        Evec = sum(Eratio, dims=2) ./ size(SV,2)
        # Adding the effective resistance of super nodes from previous levels
        @inbounds for kk = 1:length(ar)
            nd2 = ar[kk]
            Evec[kk] = Evec[kk] + sum(Neff[nd2])
        end

        ## Normalizing the ERs
        P = Evec ./ maximum(Evec)
        ## Choosing a ratio of all the hyperedges
        Nsample = round(Int, RedR * length(ar))
        PosP = sortperm(P[:,1])
        ## Increasing the weight of the hyperedges with small ERs
        W[PosP[1:Nsample]] = W[PosP[1:Nsample]] .* (1 .+  1 ./ P[PosP[1:Nsample]])
        ## Selecting the hyperedges with higher weights for contraction
        Pos = sortperm(W, rev=true)
        ## Hyperedge contraction
        flag = falses(mx)
        flagE = falses(length(ar))
        val = 1
        idx = zeros(Int, mx)
        Hcard = zeros(Int, 0)
        Neff_new = zeros(Float64, 0)

        @inbounds for ii = 1:Nsample
            nd = ar[Pos[ii]]
            fg = flag[nd]
            fd1 = findall(x->x==0, fg)
            if length(fd1) > 1
                nd = nd[fd1]
                flagE[Pos[ii]] = 1
                idx[nd] .= val
                flag[nd] .= 1
                append!(Hcard, length(ar[ii]))
                val +=1
                ## creating the super node weights
                new_val = Evec[Pos[ii]] + sum(Neff[nd])
                append!(Neff_new, new_val)
            end # endof if
        end #end of for ii

        ## indexing the isolated nodes
        fdz = findall(x-> x==0, idx)
        fdnz = findall(x-> x!=0, idx)
        V = vec(val:val+length(fdz)-1)
        idx[fdz] = V
        ## Adding the weight od isolated nodes
        append!(Neff_new, Neff[fdz])
        push!(idx_mat, idx)
        ## generating the coarse hypergraph
        ar_new = Any[]
        @inbounds for ii = 1:length(ar)
            nd = ar[ii]
            nd_new = unique(idx[nd])
            push!(ar_new, sort(nd_new))
        end #end of for ii
        ## removing the repeated hyperedges
        ar_new = unique(ar_new)
        ### removing hyperedges with cardinality of 1
        HH = INC3(ar_new)
        ss = sum(HH, dims=2)
        fd1 = findall(x->x==1, ss[:,1])
        deleteat!(ar_new, fd1)
        ar = ar_new
        Neff = Neff_new
    end #end for loop

    #mapper
    idx = 1:maximum(idx_mat[end])
    for ii = L:-1:1
        idx = idx[idx_mat[ii]]
    end # for ii
    ar = nothing
    GC.gc()
    return idx
end # end of function
