pro makeGaussStack, d, N_stack, Gauss		;This makes a stack of small randomized Gaussians typical of the data
seed=11

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
;----------------------------------------------------------------------
; $Id: //depot/gsg/HHMI/Phase2/src/cu_gauss2d/gpu_gauss2dfit.pro#5 $
;
; Copyright (c) 1995-2004, Research Systems, Inc.  All rights reserved.
;	Unauthorized reproduction prohibited.
;

PRO	GAUSS2_FUNCT, X, A, F, PDER
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
;		where: U= (yp/A2)^2 + (xp/A3)^2
;
;	  If A has 7 elements a rotation of the ellipse is present and:
;		xp = (x-A4) * cos(A6) - (y-A5) * sin(A6)
;		yp = (x-A4) * sin(A6) + (y-A5) * cos(A6)
;	  If A has 6 elements, A6 (theta) is 0, the major and minor axes
;	  of the ellipse are parallel to the XY axes, and:
;		xp = (x-A4)   and   yp = (x-A5)
;
; Optional output parameters:
;	PDER = (n_elements(z),6 or 7) array containing the
;		partial derivatives.  pder(i,j) = derivative
;		at ith point w/respect to jth parameter.
; PROCEDURE:
;	Evaluate the function and then if requested, eval partials.
;
; MODIFICATION HISTORY:
;	WRITTEN, DMS, RSI, June, 1995.
;-

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
;	The parameters of the ellipse U are:
;	   Axis lengths are 2*a and 2*b, in the unrotated X and Y axes,
;		respectively.
;	   Center is at (h,k).
;	   Rotation of T radians from the X axis, in the CLOCKWISE direction.
;	   The rotated coordinate system is defined as:
;		x' = (x-h) * cos(T) - (y-k) * sin(T)  <rotate by T about (h,k)>
;		y' = (x-h) * sin(T) + (y-k) * cos(T)
;
;	The rotation is optional, and may be forced to 0, making the major/
;	minor axes of the ellipse parallel to the X and Y axes.
;
;	The coefficients of the function, are returned in a seven
;	element vector:
;	a(0) = A0 = constant term.
;	a(1) = A1 = scale factor.
;	a(2) = a = width of gaussian in X direction.
;	a(3) = b = width of gaussian in Y direction.
;	a(4) = h = center X location.
;	a(5) = k = center Y location.
;	a(6) = T = Theta the rotation of the ellipse from the X axis
;		in radians, counterclockwise.
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
;	TILT = if set to  1, allow the orientation of the major/minor axes of
;		the ellipse to be unrestricted.  The default is that
;		the axes of the ellipse must be parallel to the X-Y axes.
;		In this case, A(6) is always returned as 0.
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
;	a = [ 5., 10., nx/6.,  ny/10., nx/2., .6*ny]  ;Input function parameters
;	x = findgen(nx) # replicate(1.0, ny)	;Create X and Y arrays
;	y = replicate(1.0, nx) # findgen(ny)
;	u = ((x-a(4))/a(2))^2 + ((y-a(5))/a(3))^2  ;Create ellipse
;	z = a(0) + a(1) * exp(-u/2)		;to gaussian
;	z = z + randomn(seed, nx, ny)		;Add random noise, SD = 1
;	yfit = gauss2dfit(z,b)			;Fit the function, no rotation
;	print,'Should be:',string(a,format='(6f10.4)')  ;Report results..
;	print,'Is:      :',string(b(0:5),format='(6f10.4)')
;
; MODIFICATION HISTORY:
;	DMS, RSI, June, 1995.
;-
;
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
stop
;print,'final fit a =',a
z= REFORM(z, nx, ny, /OVERWRITE)	;Restore dimensions
return, REFORM(result, nx, ny, /OVERWRITE)
end
;----------------------------------------------------------------------

pro GPU_Gauss2Dfit, NN

D=5		; 2D array is 2*D+1 x 2*D+1 square
N_stack= (n_elements(NN) eq 0) ? 500 : NN[0]

makeGaussStack,d, N_stack, Gauss
save, gauss, FILE='gauss.sav'
ResultStack=fltarr(2*d+1,2*d+1,N_stack)
A_list=fltarr(6,N_stack)
fita = [1,1,1,1,1,1]
;>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Make this as a GPU function that returns the Result and all the keywords in an big array
sss = systime(1)
for i=0L,N_stack-1 do begin
	A=[100.,800.,1.2,1.2,5.,5.]
	Region=Gauss[*,*,i]
	A[0]=(region[0,0]+region[0,2*d]+region[2*d,0]+region[2*d,2*d])/4.		;set A[0] to averaged value of base
	A[1]=max(region[d-1:d+1,d-1:d+1]-A[0]) > 1								;set A[1] to peak amplitude

	ResultStack[*,*,i]=gauss2Dfithh(region,A,sigma=sigma, chisq=chisq, FITA = fita, STATUS = status, ITMAX = 100)	;do 2D fit
	A_list[*,i]=A
endfor
eee = systime(1)
print, 'fit time = ', eee-sss
save, A_list, FILE='a_list.sav'
save, ResultStack, FILE='resultstack.sav'
;>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
if (N_stack lt 5000) then begin
    erase
    l=2*D+1
    for i=0,N_stack-1 do begin
        tvscl,rebin(ResultStack[*,*,i],30*l,30*l,/sample),1
        tvscl,rebin(Gauss[*,*,i],30*l,30*l,/sample),0
    endfor
endif
stop
return
end
