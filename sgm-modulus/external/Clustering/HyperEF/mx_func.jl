function mx_func(ar)

    mx2 = Int(0)
    aa = Int(0)

    for i =1:length(ar)

    	mx2 = max(aa, maximum(ar[i]))
        aa = mx2

    end
    println(mx2)
    return mx2

end
