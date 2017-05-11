;
;  file:  compare_parameters.pro
;
;  RTK, 16-Nov-2009
;  Last update, 16-Nov-2009
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;--------------------------------------------------------------
;  compare_parameters
;
pro compare_parameters
    compile_opt idl2, logical_predicate

    restore, 'gauss_fit_50000.sav'
    restore, 'cuda_fit_50000.sav'

    ;  Locate the places where CURVEFIT failed (large values, not-finite)
    i = where(~finite(p_A), c)
    if (c ne 0) then begin
        xy = array_indices(p_A, i)
        r0 = reform(xy[1,*])
    endif

    i = where(abs(p_A) gt 1e4, c)
    if (c ne 0) then begin
        xy = array_indices(p_A, i)
        r1 = reform(xy[1,*])
    endif

    i = where(p_A lt 0, c)
    if (c ne 0) then begin
        xy = array_indices(p_A, i)
        r2 = reform(xy[1,*])
    endif

    r = [1]
    if (n_elements(r0) ne 0) then begin
        r = [r,r0]
    endif
    if (n_elements(r1) ne 0) then begin
        r = [r,r1]
    endif
    if (n_elements(r2) ne 0) then begin
        r = [r,r2]
    endif

    if (n_elements(r) gt 1) then begin
        dims = size(p_A,/DIM)
        m = bytarr(dims[1])
        m[r] = 1
        _ = where(m, COMPLEMENT=keep)
        p_A    = reform(p_A[*,keep])
        u_A    = reform(u_A[*,keep])
        u_Bev  = reform(u_Bev[*,keep])
        p_cuda = reform(p_cuda[*,keep])
        u_cuda = reform(u_cuda[*,keep])
        f_time = f_time[keep]
        print, 'Samples removed due to CURVEFIT failure = ', dims[1] - n_elements(keep)
    endif else begin
        print, 'No samples removed due to CURVEFIT failure'
    endelse

    ;  Calculate the distance between the CURVEFIT parameters and the CUDA parameters
    pdist = sqrt(total((p_A-p_cuda)^2,1))

    ;  Report summary stats for the parameter distances
    print
    print, 'Median parameter dist   = ', median(pdist)
    print, 'Mean parameter distance = ', mean(pdist)
    print, 'Min parameter distance  = ', min(pdist)
    print, 'Max parameter distance  = ', max(pdist)
    print, 'Standard deviation      = ', stddev(pdist)
stop
end

