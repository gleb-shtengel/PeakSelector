Function GAUSS_2D_constrained_GS_funct, X, Y, A, fit_model=fit_model, b_coeff=b_coeff, c_coeff=c_coeff
;
; NAME:
;	GAUSS_2D_constrained_GS_funct
; PURPOSE:
;	Evaluate function for Gauss_2D_Constrained_Fit_GS.
; CALLING SEQUENCE:
;	FUNCT,X,Y,A
; INPUTS:
; 	X,Y = 2D values of independent variables (rectilinear pixel coordinates)
;	A = parameters of equation described below.
; OUTPUTS:
;	F = value of function at each X(i,j), Y(i,j).
;	Function is:
;
;---------------- fit_model=0
;		F(x,y,z) = a0 + a1*EXP(-U/2)
;		where: U= (xp/WidX(z))^2 + (yp/WidY(z))^2
;			xp = (x-a2)   and   yp = (y-a3)and
;       	WidX(z) = b0+b1*z+ ... +bN*z^N
;			WidY(z) = c0+c1*z+ ... +cN*z^N
;			z=a4
;   The vector A = [a0, a1, a2, a3, a3, a4]
;
;---------------- fit_model=1
;		F(x,y,z) = a0 + a1*EXP(-U/2)
;		where: U= (xp/WidX(z)^2 + (yp/WidY(z))^2
;			xp = (x-a2)   and   yp = (y-a3)and
;       	WidX(z) = (ellipt(z)*sum(z) + sum(z))/2
;			WidY(z) = (sum(z) - ellipt(z)*sum(z))/2
;			ellipt(z) =  b0+b1*z+ ... +bN*z^N
;			sum(z) = c0+c1*z+ ... +cN*z^N
;			z=a4
;   The vector A = [a0, a1, a2, a3, a3, a4]
;
;---------------- fit_model=2
;		F(x,y,z) = a0 + a1*EXP(-U/2)
;		where: U= (xp/a5*WidX(z))^2 + (yp/a5*WidY(z))^2
;			xp = (x-a2)   and   yp = (y-a3)and
;       	WidX(z) = b0+b1*z+ ... +bN*z^N
;			WidY(z) = c0+c1*z+ ... +cN*z^N
;			z=a4
;   The vector A = [a0, a1, a2, a3, a3, a4, a5]
;
;


; if model is not defined or fit_model=0
	if (~KEYWORD_SET(fit_model)) or (fit_model eq 0) then begin
		xp = (x-a[2])/POLY(a[4],b_coeff)		; Expand X values
		yp = (y-a[3])/POLY(a[4],c_coeff)		; Expand Y values
	endif else begin
	; fit_model = 1
		if fit_model eq 1 then begin
			ellipticity = POLY(a[4],b_coeff)
			sum = POLY(a[4],c_coeff)
			xp = 2.0*(x-a[2])/(ellipticity*sum+sum)
			yp = 2.0*(y-a[3])/(sum-ellipticity*sum)
		endif else begin
			xp = (x-a[2])/POLY(a[4],b_coeff)/a[5]		; Expand X values
			yp = (y-a[3])/POLY(a[4],c_coeff)/a[5]
		endelse
	endelse
	F = a[0] + a[1] * exp(-0.5 * (xp^2 + yp^2))
	return,F
END
;
;-----------------------------------------------------------------
;
Function Gauss_2D_Constrained_MPFit_GS, z, a, b, c, fit_model, NEGATIVE = neg, Sigma = sigma, CHISQ = chisq, FITA = fita, STATUS = status, ITMAX=itmax, YERROR=yerror
;+
; NAME:
;	Gauss_2D_Constrained_Fit_GS
;
; PURPOSE:
; 	Fit a 2 dimensional elliptical gaussian equation to rectilinearly
;	gridded data.
;		Z = F(x,y) where:
; 		F(x,y) = A0 + A1*EXP(-U/2)
;		where: U= ((y-A2)/sigx(A4))^2 + ((x-A3)/sigy(A4))^2
;       and sigx(z) = b0+b1*z+ ... +bN*z^N
;			sigy(z) = c0+c1*z+ ... +cN*z^N
;	   Center is at (h,k).
;
;
;	The coefficients of the function, are returned in a seven
;	element vector:
;	a(0) = A0 = constant term.
;	a(1) = A1 = scale factor.
;	a(2) = h = center X location.
;	a(3) = k = center Y location.
;   a(4) = z = axial position which determines SigX and SigY
;
;
; CATEGORY:
;	curve / data fitting
;
; CALLING SEQUENCE:
;	Result = Gauss_2D_Constrained_Fit_GS(z, a, b, c,)
;
; INPUTS:
;	Z = dependent variable in a 2D array dimensioned (Nx, Ny).  Gridding
;		must be rectilinear.
;	X = optional Nx element vector containing X values of Z.  X(i) = X value
;		for Z(i,j).  If omitted, a regular grid in X is assumed,
;		and the X location of Z(i,j) = i.
;	Y = optional Ny element vector containing Y values of Z.  Y(j) = Y value
;		for Z(i,j).  If omitted, a regular grid in Y is assumed,
;		and the Y location of Z(i,j) = j.
;
; Optional Keyword Parameters:
;	NEGATIVE = if set, implies that the gaussian to be fitted
;		is a valley (such as an absorption line).
;		By default, a peak is fit.
;
;
; OUTPUTS:
;	The fitted function is returned.
; OUTPUT PARAMETERS:
;	A:	The coefficients of the fit.  A is a five element vector as
;		described under PURPOSE.
;
; COMMON BLOCKS:
;	None.
; SIDE EFFECTS:
;	None.
; RESTRICTIONS:
;	Timing:  Approximately 4 seconds for a 128 x 128 array, on a
;		Sun SPARC LX.  Time required is roughly proportional to the
;		number of elements in Z.
;
; PROCEDURE:
;	The peak/valley is found by first smoothing Z and then finding the
;	maximum or minimum respectively.  Then GAUSSFIT is applied to the row
;	and column running through the peak/valley to estimate the parameters
;	of the Gaussian in X and Y.  Finally, CURVEFIT is used to fit the 2D
;	Gaussian to the data.
;
;	Be sure that the 2D array to be fit contains the entire Peak/Valley
;	out to at least 5 to 8 half-widths, or the curve-fitter may not
;	converge.
;
;
on_error,2                      ;Return to caller if an error occurs
s = size(z)
if s[0] ne 2 then $
	message, 'Z must have two dimensions'
n = n_elements(z)
nx = s[1]
ny = s[2]
np = n_params()

x = findgen(nx) # replicate(1.0, ny)			; Expand X values
y = replicate(1.0, nx) # findgen(ny)			; Expand Y values

PARINFO = replicate( {VALUE:0.D, FIXED:0 }, n_elements(a))
PARINFO.VALUE = a
PARINFO.FIXED = 1-fita
if fit_model ne 2 then PARINFO[5].FIXED = 1
argv = {fit_model:fit_model, b_coeff:b, c_coeff:c}
sz = sqrt(z>1)
precision = 5d-5

a = MPFIT2DFUN('GAUSS_2D_constrained_GS_funct', x, y, z, sz, PARINFO=PARINFO, functargs = argv, ftol = precision, /FASTNORM,$
			PERROR=PERROR , BESTNORM=BESTNORM, MAXITER=ITMAX, YFIT=YFIT, STATUS=STATUS, /QUIET)
Sigma = PERROR
CHISQ = BESTNORM
YERROR = YERROR
return, YFIT

end
