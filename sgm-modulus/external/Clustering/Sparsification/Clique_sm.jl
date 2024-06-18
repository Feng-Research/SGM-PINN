function Clique_sm(ar)



    mx = mx_func(ar)

    rr = zeros(Int, 0)
    cc = zeros(Int, 0)
    vv = zeros(Float64, 0)
    for i =1:length(ar)
        nd = sort(ar[i])
        for jj=1:length(nd)-1

            for kk = jj+1:length(nd)
                append!(rr, nd[jj])
                append!(cc, nd[kk])
                #append!(vv, 1)

            end #kk

        end #jj
    end

    vv = ones(Float64, length(rr))

    mat1 = sparse(rr,cc,vv, mx, mx)

    fdnz = findnz(mat1)

    rr2 = fdnz[1]

    cc2 = fdnz[2]

    vv2 = ones(Float64, length(rr2))

    mat1 = sparse(rr2,cc2,vv2, mx, mx)

    return mat2 = mat1 + sparse(mat1')

end
