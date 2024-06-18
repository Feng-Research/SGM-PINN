## map an array to 1:mx
function Mapping_fast(ar)

    nd = (hcat(hcat(ar...)...))

    V1 = sort(unique(nd))

    ar_new = Any[]

    for ii = 1:length(ar)

        nd1 = ar[ii]

        fd1 = findall(x->in(x, nd1), V1)

        push!(ar_new, fd1)

    end


    return ar_new, V1

end # function
