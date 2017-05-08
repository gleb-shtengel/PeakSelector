; $Id: gauss2dfit.pro,v 1.12 2004/01/21 15:54:53 scottm Exp $
;
; Copyright (c) 1995-2004, Research Systems, Inc.  All rights reserved.
;	Unauthorized reproduction prohibited.
;

PRO	GAUSS2R_FUNCT, X, A, F, PDER
;+
; NAME:
;	GAUSS2_FUNCT
; PURPOSE:
;	Evaluate function for gauss2fit.
; CALLING SEQUENCE:
;	FUNCT,X,A,F,PDER
; INPUTS:
;	X = values of independent variables, encoded as: [nx, ny, x, y]
;	A = parameters of equation described below.
; OUTPUTS:
;	F = value of function at each X(i,j), Y(i,j).
;	Function is:
;		F(x,y) = A0 + A1*EXP(-U/2)
;		where: U= (yp/A2)^2 + (xp/A2)^2
;
;		xp = (x-A3)   and   yp = (x-A4)
;
; Optional output parameters:
;	PDER = (n_elements(z),5) array containing the
;		partial derivatives.  pder(i,j) = derivative
;		at ith point w/respect to jth parameter.
; PROCEDURE:
;	Evaluate the function and then if requested, eval partials.
;
; MODIFICATION HISTORY:
;	WRITTEN, DMS, RSI, June, 1995.
;	Modified, HFH Sept. 24, 2006
;-

nx = long(x[0])		;Retrieve X and Y vectors
ny = long(x[1])

xp = (x[2:nx+1]-a[3]) # replicate(1.0/a[2], ny)	;Expand X values
yp = replicate(1.0/a[2], nx) # (x[nx+2:*]-a[4])	;expand Y values
s = 0.0 & c = 1.0

n = nx * ny
u = reform(exp(-0.5 * (xp^2 + yp^2)), n)	;Exp() term, Make it 1D
F = a[0] + a[1] * u

if n_params(0) le 3 then return ;need partial?  No.

PDER = FLTARR(n, n_elements(a))	;YES, make partial array.
PDER[*,0] = 1.0			;And fill.
pder[*,1] = u
u = a[1] * u			;Common term for the rest of the partials
pder[*,2] = u * (xp*xp + yp*yp) / a[2]
pder[*,3] = u * (c/a[2] * xp )
pder[*,4] = u * (c/a[2] * yp)

END


Function Gauss2Rdfithh, z, a, x, y, NEGATIVE = neg, TILT=tilt, Sigma = sigma, CHISQ = chisq, FITA = fita, STATUS = status, ITMAX=itmax, YERROR=yerror
;+
; NAME:
;	GAUSS2DFIT
;
; PURPOSE:
; 	Fit a 2 dimensional elliptical gaussian equation to rectilinearly
;	gridded data.
;		Z = F(x,y) where:
; 		F(x,y) = A0 + A1*EXP(-U/2)
;	   And the elliptical function is:
;		U= (x'/a)^2 + (y'/b)^2
;
;	The coefficients of the function, are returned in a seven
;	element vector:
;	a(0) = A0 = constant term.
;	a(1) = A1 = scale factor.
;	a(2) = a = width of gaussian in X and Y direction.
;	a(3) = h = center X location.
;	a(4) = k = center Y location.
;
;
; CATEGORY:
;	curve / data fitting
;
; CALLING SEQUENCE:
;	Result = GAUSS2DFIT(z, a [,x,y])
;
; INPUTS:
;	Z = dependent variable in a 2D array dimensioned (Nx, Ny).  Gridding
;		must be rectilinear.
;
; Optional Keyword Parameters:
;	NEGATIVE = if set, implies that the gaussian to be fitted
;		is a valley (such as an absorption line).
;		By default, a peak is fit.
;
; OUTPUTS:
;	The fitted function is returned.
; OUTPUT PARAMETERS:
;	A:	The coefficients of the fit.  A is a seven element vector as
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
; EXAMPLE:  This example creates a 2D gaussian, adds random noise
;	and then applies GAUSS2DFIT:
;	nx = 128		;Size of array
;	ny = 100
;	;**  Offs Scale X width Y width X cen Y cen  **
;	;**   A0  A1    a       b       h       k    **
;	a = [ 5., 10., nx/6., nx/2., .6*ny]  ;Input function parameters
;	x = findgen(nx) # replicate(1.0, ny)	;Create X and Y arrays
;	y = replicate(1.0, nx) # findgen(ny)
;	u = ((x-a(3))/a(2))^2 + ((y-a(4))/a(2))^2  ;Create ellipse
;	z = a(0) + a(1) * exp(-u/2)		;to gaussian
;	z = z + randomn(seed, nx, ny)		;Add random noise, SD = 1
;	yfit = gauss2Rdfit(z,b)			;Fit the function, no rotation
;	print,'Should be:',string(a,format='(5f10.4)')  ;Report results..
;	print,'Is:      :',string(b(0:4),format='(5f10.4)')
;
; MODIFICATION HISTORY:
;	DMS, RSI, June, 1995.
;-
;
on_error,2                      ;Return to caller if an error occurs
s = size(z)
n = n_elements(z)
nx = s[1]
ny = s[2]

x = findgen(nx)
y = findgen(ny)

q = MAX(SMOOTH(z,3), i)	;Dirty peak finder
i = nx*(ny/2)+nx/2
ix = i mod nx
iy = i / nx
x0 = x[ix]
y0 = y[iy]

;xfit = gaussfit(x, z[*,iy], ax, NTERMS=4) ;Guess at params by taking slices
;yfit = gaussfit(y, z[ix,*], ay, NTERMS=4)

;ax[2]=abs(ax[2])			;in case sigma guess is negative
;ay[2]=abs(ay[2])
; First guess, without XY term...
;Override internal initial estimate
A=A[0:4]
;a = [	(ax[3] + ay[3])/2., $		;Constant
;	sqrt(abs(ax[0] * ay[0])), $	;Exponential factor
;	ax[2], ay[2], ax[1], ay[1]]	;Widths and centers

;  If there's a tilt, add the XY term = 0
;print,'init guess a = ', a
;************* print,'1st guess:',string(a,format='(8f10.4)')
result = curvefit([nx, ny, x, y], reform(z, n, /OVERWRITE), $
		replicate(1.,n), a, sigma, ITMAX = itmax, CHISQ = chisq, $
		function_name = "GAUSS2R_FUNCT", FITA = fita, STATUS = status, ITER=ITER, YERROR=yerror)
;print,'1st fit a =',a
result = curvefit([nx, ny, x, y], reform(z, n, /OVERWRITE), $
		replicate(1.,n), a, sigma, ITMAX = itmax, CHISQ = chisq, $
		function_name = "GAUSS2R_FUNCT", FITA = fita, STATUS = status, ITER=ITER2, YERROR=yerror)
;print,'final fit a =',a
;print,'Iter=',ITER,'   Iter2=',Iter2
z= REFORM(z, nx, ny, /OVERWRITE)	;Restore dimensions
return, REFORM(result, nx, ny, /OVERWRITE)
end
