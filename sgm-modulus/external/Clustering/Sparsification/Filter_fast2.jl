# selecting a few vectors among all the given vectors for the output
function Filter_fast2(rv, SmS, AD, mx, initial, interval, Ntot)

    sz = size(AD, 1)

    V = zeros(mx, Ntot);

    sm_vec = zeros(mx, SmS);

    AD = AD .* 1.0

    AD[diagind(AD, 0)] = AD[diagind(AD, 0)] .+ 0.1

    dg = sum(AD, dims = 1) .^ (-.5)

    I2 = 1:sz

    D = sparse(I2, I2, sparsevec(dg))

    on = ones(Int, length(rv))

    sm_ot = rv - ((dot(rv, on) / dot(on, on)) * on)

    sm = sm_ot ./ norm(sm_ot);

    count = 1

    for loop in 1:SmS

        sm = D * sm

        sm = AD * sm

        sm = D * sm

        if rem(loop, interval) == 0
            sm_ot = sm .- ((dot(rv, on) / dot(on, on)) .* on)
            sm_norm = sm_ot ./ norm(sm_ot);
            V[:, count] = sm
            count +=1
        end


    end # for loop

    return V

end #end of function
