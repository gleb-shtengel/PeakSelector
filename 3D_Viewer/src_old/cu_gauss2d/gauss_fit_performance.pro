;
;  file: gauss_fit_performance.pro
;
;  Use CURVEFIT to fit 2D Gaussians and store the results as
;  a function of the number of images fit.
;
;  RTK, 21-Nov-09
;  Last update:  21-Nov-09
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

pro makeGaussStack, d, N_stack, Gauss		;This makes a stack of small randomized Gaussians typical of the data
seed=11  ;  fixed seed for reproducible results

;offset
A0s=10.
A0=100+A0s*(randomn(seed,N_stack)>0)
;Amplitude
A1s=200.
A1=800+A1s*(randomn(seed,N_stack)>0)
;Sigma X
A2s=0.3
A2=1.2+A2s*(randomn(seed,N_stack)>0)
;Sigma Y
A3s=0.3
A3=1.2+A3s*(randomn(seed,N_stack)>0)
;X0 pos
A4s=1.
A4=A4s*(randomn(seed,N_stack)>0)
;Y0 pos
A5s=1.
A5=A5s*(randomn(seed,N_stack)>0)

xp=((indgen(2*d+1)-5)#replicate(1,N_stack)-replicate(1,2*D+1)#A4)/(replicate(1,2*D+1)#A2)
GaussX=exp(-((xp)^2)/2)
yp=((indgen(2*d+1)-5)#replicate(1,N_stack)-replicate(1,2*D+1)#A5)/(replicate(1,2*D+1)#A3)
GaussY=exp(-((yp)^2)/2)
Gauss=uintarr(2*D+1,2*D+1,N_stack)
for i=0,2*d do begin
	for j=0,2*d do begin
		Gauss[i,j,*]=A0+A1*GaussX[i,*]*GaussY[j,*]
	endfor
endfor
Gauss=fix((Gauss+50*randomn(seed,2*D+1,2*D+1,N_stack))>1,Type=12)		;make unsigned int
;l=2*D+1
;for i=0,N_stack-1 do begin
;	tvscl,rebin(gauss[*,*,i],40*l,40*l,/sample)
;	wait,0.1
;endfor

return
end


PRO	GAUSS2_FUNCT, X, A, F, PDER
    nx = long(x[0])		;Retrieve X and Y vectors
    ny = long(x[1])

    tilt = n_elements(a) eq 7	;TRUE if angle present.
    if tilt then begin
        return	;don't bother with tilted Gaussians
    endif else begin
        xp = (x[2:nx+1]-a[4]) # replicate(1.0/a[2], ny)	;Expand X values
        yp = replicate(1.0/a[3], nx) # (x[nx+2:*]-a[5])	;expand Y values
        s = 0.0 & c = 1.0
    endelse

    n = nx * ny
    u = reform(exp(-0.5 * (xp^2 + yp^2)), n)	;Exp() term, Make it 1D
    F = a[0] + a[1] * u

    if n_params(0) le 3 then return ;need partial?  No.

    PDER = FLTARR(n, n_elements(a))	;YES, make partial array.
    PDER[*,0] = 1.0			;And fill.
    pder[*,1] = u
    u = a[1] * u			;Common term for the rest of the partials
    pder[*,2] = u * xp^2 / a[2]
    pder[*,3] = u * yp^2 / a[3]
    pder[*,4] = u * (c/a[2] * xp + s/a[3] * yp)
    pder[*,5] = u * (-s/a[2] * xp + c/a[3] * yp)
    if tilt then pder[*,6] = -u * xp * yp * (a[2]/a[3]-a[3]/a[2])
END


Function Gauss2dfithh, z, a, x, y, NEGATIVE = neg, TILT=tilt, Sigma = sigma, CHISQ = chisq, FITA = fita, STATUS = status, ITMAX=itmax
    on_error,2                      ;Return to caller if an error occurs
    s = size(z)
    if s[0] ne 2 then $
        message, 'Z must have two dimensions'
    n = n_elements(z)
    nx = s[1]
    ny = s[2]
    np = n_params()
    if np lt 3 then x = findgen(nx)
    if np lt 4 then y = findgen(ny)

    if nx ne n_elements(x) then $
        message,'X array must have size equal to number of columns of Z'
    if ny ne n_elements(y) then $
        message,'Y array must have size equal to number of rows of Z'

    if keyword_set(neg) then q = MIN(SMOOTH(z,3), i) $
        ELSE q = MAX(SMOOTH(z,3), i)	;Dirty peak / valley finder
    i = nx*(ny/2)+nx/2
    ix = i mod nx
    iy = i / nx
    x0 = x[ix]
    y0 = y[iy]

    ;Override internal initial estimate
    A=A[0:5]
    ;a = [	(ax[3] + ay[3])/2., $		;Constant
    ;	sqrt(abs(ax[0] * ay[0])), $	;Exponential factor
    ;	ax[2], ay[2], ax[1], ay[1]]	;Widths and centers

    ;print,'init guess a = ', a
    ;************* print,'1st guess:',string(a,format='(8f10.4)')
    result = curvefit([nx, ny, x, y], reform(z, n, /OVERWRITE), $
            replicate(1.,n), a, sigma, ITMAX = itmax, CHISQ = chisq, $
            function_name = "GAUSS2_FUNCT", FITA = fita, STATUS = status)
    ;print,'1st fit a =',a
    result = curvefit([nx, ny, x, y], reform(z, n, /OVERWRITE), $
            replicate(1.,n), a, sigma, ITMAX = itmax, CHISQ = chisq, $
            function_name = "GAUSS2_FUNCT", FITA = fita, STATUS = status)
    ;print,'final fit a =',a
    z= REFORM(z, nx, ny, /OVERWRITE)	;Restore dimensions
    return, REFORM(result, nx, ny, /OVERWRITE)
end


pro gauss_fit_performance
    compile_opt idl2, logical_predicate

    !EXCEPT = 0

    openw, u, 'gauss_fit_performance.txt', /GET_LUN, /APPEND
    
    argv = command_line_args(COUNT=argc)
    k = long(argv[0])

    x = reform(findgen(11) # replicate(1,11), 11*11)
    y = reform(findgen(11) ## replicate(1,11), 11*11)

    makeGaussStack, 5, k, gauss
    dims = size(gauss, /DIM)

    sss = systime(1)
    for i=0L, dims[2]-1 do begin
        z = reform(1.0*gauss[*,*,i], 11*11)

        ;  Gauss fit to compare the sigmas
        A=[100.,800.,1.2,1.2,5.,5.]
        Region=Gauss[*,*,i]
        d=5
        A[0]=(region[0,0]+region[0,2*d]+region[2*d,0]+region[2*d,2*d])/4.
        A[1]=max(region[d-1:d+1,d-1:d+1]-A[0]) > 1
        himg = gauss2Dfithh(region, A, sigma=hsigmas, ITMAX=100)
    endfor
    eee = systime(1)

    printf, u, 'nimg = ' + string(k, FORMAT='(I9)') + ', fit time = ' + string(eee-sss, FORMAT='(F15.6)')
    free_lun, u
end

