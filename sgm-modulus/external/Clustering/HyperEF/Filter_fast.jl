# selecting a few vectors among all the given vectors for the output
function Filter_fast(rv, k, AD, mx, initial, interval, Ntot)

    #println("+++++inside function")
    #tick()


    sz = size(AD, 1)

    V = zeros(mx, Ntot);

    sm_vec = zeros(mx, k);

    ## new changes

    AD = AD .* 1.0

    AD[diagind(AD, 0)] = AD[diagind(AD, 0)] .+ 0.1

    dg = sum(AD, dims = 1) .^ (-.5)

    I2 = 1:sz

    D = sparse(I2, I2, sparsevec(dg))

    ## end of new changes

    on = ones(Int, length(rv))

    sm_ot = rv - ((dot(rv, on) / dot(on, on)) * on)

    sm = sm_ot ./ norm(sm_ot);

    count = 1

    for loop in 1:k

        sm = D * sm

        sm = AD * sm

        sm = D * sm

        sm_ot = sm - ((dot(sm, on) / dot(on, on)) * on)

        sm_norm = sm_ot ./ norm(sm_ot);

        sm_vec[:, loop] = sm_norm[1:mx]

        #=

        if loop == initial + (count * interval)

            sm_ot = sm - ((dot(sm, on) / dot(on, on)) * on)

            sm_norm = sm_ot ./ norm(sm_ot);

            V[: , count] = sm_norm


            #V[:, count] = sm[1:mx]'

            count += 1

        end # end if
        =#

    end # for loop



    V = sm_vec[:, interval:interval:end]

#=
    on = ones(Int, mx)

    for jj = 1:size(V, 2)
    #for jj = 1:size(V, 2)

        sm = V[:, jj]

        sm_ot = sm - ((dot(sm, on) / dot(on, on)) * on)

        sm_norm = sm_ot ./ norm(sm_ot);

        V[: , jj] = sm_norm;

    end #for jj
=#
    return V



end #end of function
