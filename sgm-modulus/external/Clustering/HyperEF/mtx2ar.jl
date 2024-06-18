function mtx2ar(Inp)

    io = open(Inp, "r")
    rr = zeros(Int, 0)
    cc = zeros(Int, 0)
    while !eof(io)
        ln = readline(io)
        sp = split(ln)
        r = parse(Int, sp[1])
        c = parse(Int, sp[2])
        append!(rr, r)
        append!(cc, c)
    end

    ar = Any[]

    for ii = 1:length(rr)

        push!(ar, [rr[ii], cc[ii]])

    end # for ii

    return ar

end #end of function
