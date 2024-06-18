function HyperNodes(ar)

    H = INC3(ar)

    NH1 = Any[]

    rr1 = H.rowval

    cc1 = H.colptr

    for i = 1:size(H, 2)

        st = cc1[i]

        ed = cc1[i+1] - 1

        push!(NH1, rr1[st:ed])

    end

    return NH1

end
