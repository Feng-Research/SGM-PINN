function INC3(ar)

    col = zeros(Int, 0)
    row = zeros(Int, 0)



    for iter = 1:length(ar)
        cc = (iter) * ones(Int, length(ar[iter]))
        rr = ar[iter]

        append!(col, cc)
        append!(row, rr)
    end

    row = row

    val = ones(Float64, length(row))

    mat = sparse(col, row, val)

    return mat
end
