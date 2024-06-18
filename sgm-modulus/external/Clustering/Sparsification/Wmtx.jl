function Wmtx(Inp, L)

    LT = sparse(triu(L)')

    fdz = findnz(LT)

    rr = fdz[1]

    cc = fdz[2]

    vv = fdz[3]

    open(Inp, "w") do io
    #open("T10.mtx", "w") do io
        #println(io, size(L, 1), " ", nnz(LT) - size(LT,1))

        for i =1:length(rr)

            println(io, rr[i], " ", cc[i], " ", round(vv[i], digits = 2))
            #println(io, rr[i], " ", cc[i])

        end #end of i


    end


end
