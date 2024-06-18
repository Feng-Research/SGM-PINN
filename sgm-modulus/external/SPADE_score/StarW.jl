function StarW(ar, W)

    mx = mx_func(ar)

    sz = length(ar)

    col = zeros(Int32, 0)
    val = zeros(Float32, 0)
    row = zeros(Int32, 0)

    for iter =1:length(ar)
        LN = length(ar[iter])
        cc = (iter+mx) * ones(Int, LN)
        vv = (W[iter]/LN) * ones(Int, LN)

        rr = ar[iter]
        append!(col, cc)

        append!(row, rr)

        append!(val, vv)
    end

    mat = sparse(row, col, val,mx+sz, mx+sz)

    A = mat + mat'

    return A

end
