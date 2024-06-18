function h_scoreW(rrH, ccH, WH, SV)
    DST = zeros(eltype(SV), length(rrH))
    Qval = 0
    @inbounds for i in eachindex(rrH)
            DST[i] = (SV[rrH[i]] - SV[ccH[i]])^2
            Qval = Qval + (DST[i] * WH[i])
    end
    R = DST ./ Qval

    return R
end
