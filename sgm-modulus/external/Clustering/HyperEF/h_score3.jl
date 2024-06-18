function h_score3(ar, SV)
    score = zeros(eltype(SV), length(ar))
    @inbounds Threads.@threads for i in eachindex(ar)
        nodes = ar[i]
        for j in axes(SV, 2)
            mx, mn = -Inf, +Inf
            for node in nodes
                x = SV[node, j]
                mx = ifelse(x > mx, x, mx)
                mn = ifelse(x < mn, x, mn)
            end
            score[i] += (mx - mn)^2
        end
    end
    return score
end
