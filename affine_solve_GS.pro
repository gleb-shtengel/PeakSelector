;==================================================

PRO AFFINE_SOLVE_GS, Data, Target, R, c, P, Q, Err_mean, VERBOSE=blab

;==================================================
;    Created: G.Shtengel. Aug 2017
;
; NAME:
;	AFFINE_SOLVE_GS
;
; PURPOSE:
;	Calculate the parameters of a general affine image
;       transformation given a set of points from two images:
;       one of the images is assumed to be the reference image,
;       the other is assumed to be an image translated, rotated,
;       scaled, and possibly sheared relative to the reference image.
;
;
;    uses linear regression to find affine transformation coefficoents such that
;		Data = [X, Y]
;		Target = [X', Y']
;
;		Target = R * Data + C
;
;------------ 2D case----------------------------------
;		| X'|		| R00 R01 Cx |	 |X|
;		| Y'|	= 	| R10 R11 Cy | * |Y|
;		| 1 |		|  0   0  1	 |	 |1|
;
;  		In IDL notation, the output is given by vectors P and Q, such that
;		X' = P00 + P01*X + P10*Y
;   	Y' = Q00 + Q01*X + Q10*Y
;
;	so	P00=Cx   P01=R00  P10=R01, and P11=0
;		Q00=Cy   Q01=R10  Q10=R11, and Q11=0
;-------------------------------------------------------
;
; CALLING SEQUENCE:
;	AFFINE_SOLVE_GS, Data, Target, R, c, P, Q, Err_mean, VERBOSE=blab
;
; INPUTS:
;	Data:    nxM dimensional array of points taken from image1
;               which correspond to the same points in the reference image.
;               M is the number of points.
;
;	Target:   nxM dimensional array of points from the "reference image"
;               which correspond to points in the image.
;
; KEYWORDS:
;       VERBOSE: If set, print the transformation elements to the screen.
;
; OUTPUTS:
;       R - rotation, scaling and shear
;		c - shift
;
;	P and Q are corresponding coefficients that can be later used for transforming the set of points using polywarp

; COMMON BLOCKS:
;	None.
;
; SIDE EFFECTS:
;	None.
;
; RESTRICTIONS:
;	M, the number of matched points in the transformed and reference
;       images should be large (greater than 20), should be taken from
;       widely spaced locations in the image field-of-view, and should
;       be measured to within 1-pixel for greatest accuracy.
;
;       Off-center rotation and translation require a two-stage approach
;       for image registration. i.e. in the first stage, apply the parameters
;       given by this routine to the test image. A second set of points
;       is then selected from the image and the reference image, and
;       a second run of this program should output a final translation
;       to be applied to the test image to bring it in registration with
;       the reference image. This is tested for and the user is alerted.
;
; PROCEDURE:
;	Using least squares estimation, determine the elements
;       of the general affine transformation (rotation and/or scaling
;       and/or translation and/or shearing) of an image onto a reference
;       image.
;
;	See:	Image Processing for Scientific Applications
;               Bernd J"ahne, 	CRC Press, 1997, Chapter 8.
;
;	See: "Affine registration", Moo K.Chung, October 7, 2012
;
ON_ERROR,2

n1 = (SIZE(Data))(1)
n2 = (SIZE(Target))(1)

if n2 ne n1 then begin
    MESSAGE,'Number of points must be the same in both input arrays'
    RETURN
end

ones=dblarr(n1)+1.0
Dp=[[Data],[ones]]
Tp=[[Target],[ones]]
Rc = matrix_multiply(INVERT(matrix_multiply(Dp,Dp,/ATRANSPOSE),/DOUBLE),matrix_multiply(Tp,Dp,/ATRANSPOSE),/BTRANSPOSE)

dim=(size(Rc))[1]-1
R = Rc[0:(dim-1),0:(dim-1)]
c = Rc[dim,0:(dim-1)]
;		P00=Cx   P01=R00  P10=R01, and P11=0
;		Q00=Cy   Q01=R10  Q10=R11, and Q11=0
if dim eq 2 then begin
	P = dblarr(2,2)
	Q=P
	P[0,0] = c[0]
	Q[0,0] = c[1]
	P[0,1] = R[0,0]
	P[1,0] = R[0,1]
	Q[0,1] = R[1,0]
	Q[1,0] = R[1,1]
endif

Err_mean = 0.0

if VERBOSE then begin
	print,'Rotation/scaling matrix: ', R
	print,'Shift matrix: ', c
	print, 'Average Registration Error: ', Err_mean
endif

RETURN

END
;
;--------------------------------------------------------------------
;
