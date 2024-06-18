function decompositionW2_fast(A, L)

    idx_mat = Any[]

    Neff = zeros(Float64, size(A,1))

    Emat = Any[]

    Nmat = Any[]

    ar_mat = Any[]

    THL = size(A,1) / 10

    @inbounds for loop = 1:L
    #while size(A, 1) > THL

        mx = size(A, 1)
        #println("mx = ", mx)

        fdnz = findnz(triu(A, 1))
        rr = fdnz[1]
        cc = fdnz[2]
        W = fdnz[3]

        ar1 = Any[]

        for ii = 1:length(rr)
            push!(ar1, [rr[ii], cc[ii]])
        end

        push!(ar_mat, ar1)

        MM = length(rr)

        ## computing the smoothed vectors
        initial = 0

        SmS = 300

        interval = 20

        Nrv = 1

        Nsm = floor.(Int, (SmS - initial) / interval)

        Ntot = Nrv * Nsm

        Qvec = zeros(Float64, 0)

        Eratio = zeros(Float64, MM, Ntot)

        SV = zeros(Float64, mx, Ntot)

        for ii = 1:Nrv

            sm = zeros(mx, Nsm)

            Random.seed!(1234); randstring()

            rv = (rand(Float64, size(A, 1), 1) .- 0.5).*2

            sm = Filter_fast2(rv, SmS, A, mx, initial, interval, Nsm)
            #@time sm = Filter_fast3(rv, SmS, rr, cc, W, mx, initial, interval, Ntot)

            SV[:, (ii-1)*Nsm+1 : ii*Nsm] = sm

        end

        ## Make all the smoothed vectors orthogonal to each other
        QR1 = qr(SV)
        SV = Matrix(QR1.Q)

        Qvec = Any[]

        ## Computing the ratios using all the smoothed vectors
        Evec = zeros(Float64, MM)
        for jj = 1:size(SV, 2)
            #include("h_score3_fast.jl")
            E1 = h_scoreW(rr, cc, W, SV[:, jj])
            Evec += E1

        end #for jj


        # Adding the effective resistance of super nodes from previous levels
        @inbounds for kk = 1:length(rr)

            Evec[kk] = Evec[kk] + Neff[rr[kk]] + Neff[cc[kk]]

        end

        push!(Emat, Evec)


        ## Choosing a ratio of the hyperedges for contraction
        Nsample = round(Int, MM)

        Pos = sortperm(Evec[:,1])

        ## low-ER diameter clustering which starts by contracting
        # the hyperedges with low ER diameter
        flag = falses(mx)

        val = 1

        idx = zeros(Int, mx)

        Neff_new = zeros(Float64, 0)

        @inbounds for ii = 1:Nsample

            #nd = ar[Pos[ii]]
            nd = [rr[Pos[ii]], cc[Pos[ii]]]

            fg = flag[nd]

            fd1 = findall(x->x==0, fg)

            if length(fd1) > 1

                nd = nd[fd1]

                idx[nd] .= val

                # flag the discovered nodes
                flag[nd] .= 1

                val +=1

                ## creating the super node weights
                append!(Neff_new, Evec[Pos[ii]] + sum(Neff[nd]))

            end # endof if

        end #end of for ii

        ## indexing the isolated nodes
        fdz = findall(x-> x==0, flag)

        V = vec(val:val+length(fdz)-1)

        idx[fdz] = V
        ## Adding the weight of isolated nodes
        append!(Neff_new, Neff[fdz])

        push!(Nmat, Neff)

        push!(idx_mat, idx)

        ## generating the coarse hypergraph
        rr_new = zeros(Int,0)
        cc_new = zeros(Int,0)

        @inbounds for ii = 1:MM

            append!(rr_new, min(idx[rr[ii]], idx[cc[ii]]))
            append!(cc_new, max(idx[rr[ii]], idx[cc[ii]]))

        end #end of for ii

        R1 = vcat(rr_new, cc_new)
        R2 = vcat(cc_new, rr_new)
        W = vcat(W,W)

        A = sparse(R1, R2, W)
        #println("loop: ", loop)

        Neff = Neff_new

        #append!(Emax, maximum(Evec))

        #println("size A = ", size(A, 1))

    end #end for loop

    push!(Nmat, Neff)

    ## creating ar
    fdnz = findnz(triu(A, 1))
    rr = fdnz[1]
    cc = fdnz[2]

    ar = Any[]

    for ii = 1:length(rr)
        push!(ar, [rr[ii], cc[ii]])
    end




    return ar, idx_mat, ar_mat, Emat
end
