## map the tree into an array
function Unmapping(Tree, V2, arF)

    fdnz = findnz(triu(Tree,1))

    rr = fdnz[1]
    cc = fdnz[2]

    for ii = 1:length(rr)

        nd = [V2[rr[ii]], V2[cc[ii]]]

        push!(arF, sort(nd))

    end # for ii

    return arF

end # end function
