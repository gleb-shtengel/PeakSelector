pro padd, a, b, n
    compile_opt idl2

    na = n_elements(a)
    nb = n_elements(b)
    if (na eq 0 || n eq 0) then begin
        n = nb
        a = arg_present(b) ? b : temporary(b)
        return
    endif
    while (na lt n+nb) do begin
        a = [a,a]
        na *= 2
    endwhile

    a[n] = arg_present(b) ? b : temporary(b)
    n += nb
end
