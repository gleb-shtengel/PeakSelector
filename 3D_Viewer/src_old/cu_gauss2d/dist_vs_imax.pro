;
;  file:  dist_vs_imax.pro
;
;  RTK, 21-Nov-2009
;  Last update, 21-Nov-2009
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;--------------------------------------------------------------
;  dist_vs_imax
;
pro dist_vs_imax
    compile_opt idl2, logical_predicate

    dlm_load, 'cu_gauss2d'

    ;  Repeat for all imax values
    starting = 6
    ending = 200
    distance = dblarr(ending - starting + 1)
    timing = dblarr(ending - starting + 1)
    
    for imax = starting, ending do begin

        ;  Load the Gaussian data and fit with CUDA
        restore, 'gauss50000.sav'
        dims = size(gauss,/DIM)
        params = fltarr(7, dims[2])
        images = uintarr(11,11, dims[2])
        img = uintarr(11,11)
        constraints = float([0,500,100,1500,0,5,0,5,0,10,0,10])
        sss = systime(1)
        p_cuda = cu_gauss2d(gauss, imax, constraints, 0)
        eee = systime(1)
        p_cuda = reform(p_cuda ,13, dims[2], /OVERWRITE)
        p_cuda = p_cuda[0:5,*]

        ;  Determine the median distance between these parameters and the CURVEFIT parameters
        restore, 'gauss_fit_50000.sav'
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
            p_cuda = reform(p_cuda[*,keep])
        endif
        
        ;  Calculate the distance between the CURVEFIT parameters and the CUDA parameters
        pdist = sqrt(total((p_A-p_cuda)^2,1))
        dmedian = median(pdist)

        print, 'imax = ' + strtrim(imax,2) + ', d_median = ' + strtrim(dmedian,2) +  $
               ',  fit time = ' + strtrim(eee-sss,2)
        
        ;  Store the median distance and fit duration
        distance[imax-starting] = dmedian
        timing[imax-starting] = eee-sss
    endfor

    ;  Store the distance and timing results
    save, distance, timing, starting, ending, FILE='dist_vs_imax.sav'
end


