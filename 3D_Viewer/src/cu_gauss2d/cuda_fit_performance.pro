;
;  file: cuda_fit_performance.pro
;
;  Use CUDA to fit Gaussians and track the performance
;
;  RTK, 23-Nov-09
;  Last update:  23-Nov-09
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


pro cuda_fit_performance
    compile_opt idl2, logical_predicate

    !EXCEPT = 0

    dlm_load, 'cu_gauss2d'

    openw, u, 'cuda_fit_performance.txt', /GET_LUN, /APPEND
    
    argv = command_line_args(COUNT=argc)
    k = long(argv[0])

    x = reform(findgen(11) # replicate(1,11), 11*11)
    y = reform(findgen(11) ## replicate(1,11), 11*11)

    makeGaussStack, 5, k, gauss
    dims = size(gauss, /DIM)
    constraints = float([0,500,100,1500,0,5,0,5,0,10,0,10])

    sss = systime(1)
    p = cu_gauss2d(gauss, 30, constraints, 0)
    eee = systime(1)

    printf, u, 'nimg = ' + string(k, FORMAT='(I9)') + ', fit time = ' + string(eee-sss, FORMAT='(F15.6)')
    free_lun, u
end

