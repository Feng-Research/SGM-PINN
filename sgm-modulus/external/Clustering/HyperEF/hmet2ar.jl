function hmet2ar(input)

    io = open(input, "r")

    ar  = Any[]
    while !eof(io)
        rr = zeros(Int, 0)
        ln = readline(io)
        sp = split(ln)

        for kk = 1:length(sp)
            r = parse(Int, sp[kk])
            append!(rr, r)
        end #kk
        push!(ar, rr)

    end

    ar = deleteat!(ar, 1)

    return ar

end #end of function
