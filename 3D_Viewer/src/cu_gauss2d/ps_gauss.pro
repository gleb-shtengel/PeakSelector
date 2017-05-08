;
; file: ps_gauss.pro
;
;  2D fit to a Gaussian image, 11x11, using PSO.
;
;  RTK, 02-Sep-09
;  Last update:  02-Sep-09
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

function ComputeUncertainties, x, y, p
    compile_opt idl2, logical_predicate

    pp = double(p)
    p0=pp[0] & p1=pp[1] & p2=pp[2] & p3=pp[3] & p4=pp[4] & p5 = pp[5]
    N = n_elements(y)
    Nf = N - 6  ; df
    dchi = dblarr(6)

    arg = -0.5*(((x-p4)/p2)^2+((y-p5)/p3)^2)
    idx = where(arg lt -50.0, count)
    if (count ne 0) then  arg[idx] = -50.0
    idx = where(arg gt 50.0, count)
    if (count ne 0) then  arg[idx] = 50.0

    u = exp(arg)
    u2 = u*u
    g2 = u*(p1*(x-p4)^4/p2^6 - 3.0d*p1*(x-p4)^2/p2^4)
    g3 = u*(p1*(y-p5)^4/p3^6 - 3.0d*p1*(y-p5)^2/p3^4)
    g4 = u*(p1*(x-p4)^2/p2^4 - p1/p2^2)
    g5 = u*(p1*(y-p5)^2/p3^4 - p1/p3^2)

    dchi[0] = N                                                     ;  p0
    dchi[1] = total(u2)                                             ;  p1
    dchi[2] = total((p1*(x-p4)^2/p2^3)^2*u2 - y*g2 + (p0+p1*u)*g2)  ;  p2
    dchi[3] = total((p1*(y-p5)^2/p3^3)^2*u2 - y*g3 + (p0+p1*u)*g3)  ;  p3
    dchi[4] = total((p1*(x-p4)/p2^2)^2*u2 - y*g4 + (p0+p1*u)*g4)    ;  p4
    dchi[5] = total((p1*(y-p5)/p3^2)^2*u2 - y*g5 + (p0+p1*u)*g5)    ;  p5
    dchi *= (2.0d/Nf)
    
    sigmas = sqrt(2.0d/abs(dchi))
    return, sigmas
end

function fg, x, y, p
    compile_opt idl2, logical_predicate

    return, p[0] + p[1]*exp(-0.5*(((x-p[4])/p[2])^2 + ((y-p[5])/p[3])^2))
end


pro ps_gauss, n
    compile_opt idl2, logical_predicate

    restore, 'gauss1000.sav'

    !EXCEPT = 0

    dims = size(gauss, /DIM)
    A_list = fltarr(6,dims[2])
    p_A = fltarr(7,dims[2])
    images = uintarr(11,11,dims[2])

    x = reform(findgen(11) # replicate(1,11), 11*11)
    y = reform(findgen(11) ## replicate(1,11), 11*11)

    window, 0, xs=3*30*11, ys=30*11
    sum = 0.0d

    for i=0L, 99 do begin
        z = reform(1.0*gauss[*,*,i], 11*11)

        particles = 20
        iterations= 10
        ob = obj_new('PSFit', X=x, Y=y, Z=z, /TWO, F='fg', N=6, PARTICLES=particles, PS_IMAX=iterations, PS_W=[0.9,0.9], IMAX=1,  $
                     PS_C1=2.0, PS_C2=2.0, PS_VMAX=100.0, PS_DOMAIN=[[0,500.0d],[100,1500],[0,5],[0,5],[0,10],[0,10]])
        ob->SetProperty, /PS_CONSTRAIN_TO_DOMAIN
        sss = systime(1)
        A_list[*,i] = ob->Fit(ERROR=err)

        ;  Compute the parameter uncertainties (Bevington 11-36)
        p = reform(A_list[*,i])
        sp = ComputeUncertainties(x,y,p)
        
        img = uintarr(11,11)
        img[x,y] = fix(p[0] + p[1]*exp(-0.5*(((x-p[4])/p[2])^2 + ((y-p[5])/p[3])^2)), TYPE=12)
        images[*,*,i] = img
        eee = systime(1)

        ;  Gauss fit to compare the sigmas
        qqq = systime(1)
        A=[100.,800.,1.2,1.2,5.,5.]
        Region=Gauss[*,*,i]
        d=5
        A[0]=(region[0,0]+region[0,2*d]+region[2*d,0]+region[2*d,2*d])/4.		;set A[0] to averaged value of base
        A[1]=max(region[d-1:d+1,d-1:d+1]-A[0]) > 1								;set A[1] to peak amplitude
        himg = gauss2Dfithh(region, A, sigma=hsigmas, ITMAX=100)
        p_A[0:5,i] = A
        www = systime(1)

        tvscl, rebin(gauss[*,*,i],30*11,30*11),0
        tvscl, rebin(img,30*11,30*11),1
        tvscl, rebin(himg,30*11,30*11),2

        print, i+1
        ob->GetProperty, ITERATIONS_DONE=icount
        ob->GetProperty, SIGMAS=sigmas
        obj_destroy, ob

        ddd = sqrt(total((p-A)^2))
        sum += ddd

        print, 'Fit time        : ', eee-sss
        print, 'Curve time      : ', www-qqq
        print
        print, 'Parameters      : ', p
        print, 'curve fit params: ', A
        print, 'distance pA     : ', sqrt(total((p-A)^2))
        print
        print, 'Sigmas          : ', sigmas
        print, 'curve fit sigmas: ', hsigmas
        print, 'calc sigmas     : ', sp
        print
    endfor
    print
    print, 'Mean distance = ', sum/100.0
    print, 'Particles, iterations = ', particles,', ',iterations
    print
    save, A_list, FILE='a_list_ps.sav'
    save, images, FILE='images_ps.sav'
    stop
end
