;
; IDL Event Callback Procedures
; AnchorWid_eventcb
;
; Generated on:	03/22/2007 10:19.31
;
;-----------------------------------------------------------------
function CheckForCompleteSet,AnchorPnts
cc=where(AnchorPnts[0,*] ne 0, fid0_cnt)		; number of non-zero fields in one of the columns
cc=where(AnchorPnts[*,*] ne 0, fid_tot_cnt)
if fid_tot_cnt mod fid0_cnt eq 0 then CompleteSet = 1
return, CompleteSet
end
;
;-----------------------------------------------------------------
;
pro Complex_Linear_Regression, Zi, Zo, P,Q, Mag			; GES 04.14.09, linear regression routine to claculate least sqare linear fit in complex space
; Assuming Zi=A*Zo+B
; Then use regression conditions:
;  Sum(Zi)=A*Sum(Zo)+B*N,   where N is the number of elements in Zi (and Zo)
;  Sum(Zi*c.c.(Zo))=A*Sum(Zo*c.c.(Zo))+B*Sum(c.c.(Zo))		, where c.c. is complex conjugate

	b1=total(Zi)
	b2=transpose(conj(Zo)) # Zi
	b=[b1,b2]

	a11=N_ELEMENTS(Zo)
	a12=total(Zo)
	a21=conj(a12)
	a22=transpose(conj(Zo)) # Zo
	a=[[a11,a12],[a21,a22]]

	Compl_Regr = LU_COMPLEX(a,b)		; Compl_Regr[1] - rotation and zoom,  Compl_Regr[0] - shift, real part for X, imaginary for Y

	P=DBLARR(2,2);
	Q=P

	P[0,0] = REAL_PART(Compl_Regr[0])
	P[0,1] = REAL_PART(Compl_Regr[1])
	P[1,0] = -1 * IMAGINARY(Compl_Regr[1])

	Q[0,0] = IMAGINARY(Compl_Regr[0])
	Q[0,1] = -1 * P[1,0]
	Q[1,0] = P[0,1]

	Mag=abs(Compl_Regr[1])
end
;
;-----------------------------------------------------------------
;
PRO AFFINE_SOLVE_GS, Data, Target, R, c, P, Q, Err_mean
;
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
;		| X'|		| R00 R10 Cx |	 |X|
;		| Y'|	= 	| R01 R11 Cy | * |Y|
;		| 1 |		|  0   0  1	 |	 |1|
;
;  		In IDL notation, the output is given by vectors P and Q, such that
;		X' = P[0,0] + P[0,1]*X + P[1,0]*Y
;   	Y' = Q[0,0] + Q[0,1]*X + Q[1,0]*Y
;
;		P00=Cx   P01=R00  P10=R10, and P11=0
;		Q00=Cy   Q01=R01  Q10=R11, and Q11=0
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

if dim eq 2 then begin
	P = dblarr(2,2)
	Q=P
	P[0,0] = c[0]
	Q[0,0] = c[1]
	P[0,1] = R[0,0]
	P[1,0] = R[1,0]
	Q[0,1] = R[0,1]
	Q[1,0] = R[1,1]
endif

Err_mean = 0.0

verbose=0
if VERBOSE then begin
	print,'Rotation/scaling matrix: ', R
	print,'Shift matrix: ', c
	print, 'Average Registration Error: ', Err_mean
endif
RETURN

END
;
;-----------------------------------------------------------------
;
pro AFFINE_Transform_GS, Data, Output, R, c
 PURPOSE:
;	Transform the given set of points according to affine transformation coefficients
;	works in 2D and 3D case
;
;		Output = R * Data + C
;
n1 = (SIZE(Data))[1]
n2 = (SIZE(Data))[2]
ones=dblarr(n1)+1.0
Dp=[[Data],[ones]]
pad=dblarr(n2+1)
pad[n2]=1.0
Rc=[[R,c],[pad]]
Output_padded = matrix_multiply(Dp,Rc)
Output = Output_padded[0:(n1-1),0:(n2-1)]
end
;
;-----------------------------------------------------------------
;
pro DoDataFidtoTargetFid, LabelToTransform,LabelTarget,Event		;Map data with data fiducials onto target fiducials Modify CGroupparam-xy & Record Coeff
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))				; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))				; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)


TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

DatFid=AnchorPnts[(2*(LabelToTransform-1)):(2*LabelToTransform-1),*]
DatZ=ZPnts[(LabelToTransform-1),*]
TargFid=AnchorPnts[(2*(LabelTarget-1)):(2*LabelTarget-1),*]
TargZ=ZPnts[(LabelTarget-1),*]

LTT = max(CGroupParams[LabelSet_ind,*]) < LabelToTransform		; if there is only one label =0, transform it
indecis=where(CGroupParams[LabelSet_ind,*] eq LTT)

anc_ind=where(DatFid[0,*] ne 0)

if (size(indecis))[0] eq 0 then return
FiducialCoeff[LabelToTransform-1].present=1

XDat0=DatFid[0,0]
YDat0=DatFid[1,0]
XTarg0=TargFid[0,0]
YTarg0=TargFid[1,0]

Xo=reform(DatFid[0,anc_ind])
Yo=reform(DatFid[1,anc_ind])
Xi=reform(TargFid[0,anc_ind])
Yi=reform(TargFid[1,anc_ind])

cc=where(AnchorPnts[*,0] ne 0)
cc=where(AnchorPnts[cc[0],*] ne 0, fid0_cnt)		; number of non-zero fields in one of the columns
cc=where(AnchorPnts[*,*] ne 0, fid_tot_cnt)		; total number of non-zero fields

n_fid_lbl=fid_tot_cnt/fid0_cnt					; this ratio should be 4 (two labels) or 6 (3 labels)

if (n_fid_lbl ne 4) and (n_fid_lbl ne 6) then begin	; stop if the above is not true
	print, 'inconsistent fiducial number'
	print, 'number of non-zero fields in the first column=',fid0_cnt
	print, 'total number of non-zero fields=',fid_tot_cnt
	return
endif

; just shift
if  (Transf_Meth eq 0) then begin
	if (fid0_cnt lt 1) then begin
		z=dialog_message('X-Y shift requires at least 1 pair of fiducials, you have only '+string(fid0_cnt))
		return
	endif
	print,'calculating single fiducial shift transformation'
	dX = mean(Xi - Xo)
	dY = mean(Yi - Yo)
	CGroupParams[X_ind,indecis] = (CGroupParams[X_ind,indecis] + dX)>0
	CGroupParams[Y_ind,indecis] = (CGroupParams[Y_ind,indecis] + dY)>0
	CGroupParams[GrX_ind,indecis] = (CGroupParams[GrX_ind,indecis] + dX)>0
	CGroupParams[GrY_ind,indecis] = (CGroupParams[GrY_ind,indecis] + dY)>0
	P=[[-1*dX,0],[1,0]]
	Q=[[-1*dY,1],[0,0]]
endif

; linear regression for complex linear Fit:   Zi=M*Zo+N
if  (Transf_Meth eq 1)  then begin
	if (fid0_cnt lt 3) then begin
		z=dialog_message('linear regression requires at least 3 pars of fiducials, you have only '+string(fid0_cnt))
		return
	endif
	XYo=complex(Xo,Yo)
	XYi=complex(Xi,Yi)
	print,'calculating ',n_elements(Xo),'   fiducial linear regression transformation'
	Zi=XYi
	Zo=XYo
	Complex_Linear_Regression, Zi, Zo, P,Q, Mag
	X=CGroupParams[X_ind,indecis]
	Y=CGroupParams[Y_ind,indecis]
	CGroupParams[X_ind,indecis]= (P[0,0]+P[0,1]*X+P[1,0]*Y)>0
	CGroupParams[Y_ind,indecis]= (Q[0,0]+Q[0,1]*X+Q[1,0]*Y)>0
	X=CGroupParams[GrX_ind,indecis]
	Y=CGroupParams[GrY_ind,indecis]
	CGroupParams[GrX_ind,indecis]= (P[0,0]+P[0,1]*X+P[1,0]*Y)>0
	CGroupParams[GrY_ind,indecis]= (Q[0,0]+Q[0,1]*X+Q[1,0]*Y)>0
	Zi=XYo
	Zo=XYi
	Complex_Linear_Regression, Zi, Zo, P,Q, Mag		; transformation for actual data is inverse of that for the fiducials
endif												; see the difference betweem POLY_2D and POLYWRAP

; POLYWARP
if  (Transf_Meth eq 2) then begin
	if (fid0_cnt lt 3) then begin
		z=dialog_message('POLYWARP requires at least 3 pars of fiducials, you have only '+string(fid0_cnt))
		return
	endif
	print,'calculating ',n_elements(Xo),'   polywarp transformation'
	polywarp,Xi,Yi,Xo,Yo,PW_deg,Kx,Ky				;Xi=sum(kxij#Xo^jYo^i)   Yi=sum(kyij#Xo^jYo^i)
	X=CGroupParams[X_ind,indecis]
	X1=X*0.0
	Y=CGroupParams[Y_ind,indecis]
	Y1=Y*0.0
	for xj=0, PW_deg do begin
		for yj=0, PW_deg do begin
			X1 = X1+Kx[yj,xj]*(X^xj)*(Y^yj)
			Y1 = Y1+Ky[yj,xj]*(X^xj)*(Y^yj)
		endfor
	endfor
	CGroupParams[X_ind,indecis]= X1>0
	CGroupParams[Y_ind,indecis]= Y1>0
	X=CGroupParams[GrX_ind,indecis]
	X1=X*0.0
	Y=CGroupParams[GrY_ind,indecis]
	Y1=Y*0.0
	for xj=0, PW_deg do begin
		for yj=0, PW_deg do begin
			X1 = X1+Kx[yj,xj]*(X^xj)*(Y^yj)
			Y1 = Y1+Ky[yj,xj]*(X^xj)*(Y^yj)
		endfor
	endfor
	CGroupParams[GrX_ind,indecis]= X1>0
	CGroupParams[GrY_ind,indecis]= Y1>0
	polywarp,Xo,Yo,Xi,Yi,PW_deg,FP,FQ				;Xi=sum(kxij#Xo^jYo^i)   Yi=sum(kyij#Xo^jYo^i)
	; transformation for actual data is inverse of that for the fiducials, see the difference betweem POLY_2D and POLYWRAP
	if ((size(FP))[0] eq (size(FiducialCoeff[LabelToTransform-1].P))[0]) and ((size(FP))[1] eq (size(FiducialCoeff[LabelToTransform-1].P))[1])then begin
		P=FP
		Q=FQ
	endif
endif

; Affine 2D
if Transf_Meth eq 3 then begin
	if (fid0_cnt lt 3) then begin
		z=dialog_message('Affine 2D transformation requires at least 3 pars of fiducials, you have only '+string(fid0_cnt))
		return
	endif
	print,'Calculating 2D affine transformation'
	xin = transpose(DatFid[*,anc_ind])
	xpin = transpose(TargFid[*,anc_ind])
	;AFFINE_SOLVE, xin, xpin, P, Q, xb, mx, my, sx, theta, xc, yc, verbose=0
	AFFINE_SOLVE_GS, xin, xpin, R, c, P, Q, Err_mean
	X=CGroupParams[X_ind,indecis]
	Y=CGroupParams[Y_ind,indecis]
	CGroupParams[X_ind,indecis]= (P[0,0]+P[0,1]*X+P[1,0]*Y)>0
	CGroupParams[Y_ind,indecis]= (Q[0,0]+Q[0,1]*X+Q[1,0]*Y)>0
	X=CGroupParams[GrX_ind,indecis]
	Y=CGroupParams[GrY_ind,indecis]
	CGroupParams[GrX_ind,indecis]= (P[0,0]+P[0,1]*X+P[1,0]*Y)>0
	CGroupParams[GrY_ind,indecis]= (Q[0,0]+Q[0,1]*X+Q[1,0]*Y)>0
	;AFFINE_SOLVE, xpin, xin, P, Q, xb, mx, my, sx, theta, xc, yc, verbose=0
	AFFINE_SOLVE_GS, xpin, xin, R, c, P, Q, Err_mean
endif

; Affine 3D
if Transf_Meth eq 4 then begin
	if (fid0_cnt lt 4) then begin
		z=dialog_message('Affine 3D transformation requires at least 4 pars of fiducials, you have only '+string(fid0_cnt))
		return
	endif
	if (total(DatZ) eq 0) then begin
		z=dialog_message('Affine 3D transformation requires Z-fiducuials, none entered')
		return
	endif
	print,'Calculating 3D affine transformation'
	xin = [[transpose(DatFid[*,anc_ind])],[DatZ[anc_ind]]]
	xpin = [[transpose(TargFid[*,anc_ind])],[TargZ[anc_ind]]]
	AFFINE_SOLVE_GS, xin, xpin, R, c, P, Q, Err_mean
	In_arr = transpose([CGroupParams[X_ind,indecis],CGroupParams[Y_ind,indecis],CGroupParams[Z_ind,indecis]])
	AFFINE_Transform_GS, In_arr, Out_arr, R, c
	CGroupParams[X_ind,indecis] = Out_arr[*,0]
	CGroupParams[Y_ind,indecis] = Out_arr[*,1]
	CGroupParams[Z_ind,indecis] = Out_arr[*,2]
	In_arr = transpose([CGroupParams[GrX_ind,indecis],CGroupParams[GrY_ind,indecis],CGroupParams[GrZ_ind,indecis]])
	AFFINE_Transform_GS, In_arr, Out_arr, R, c
	CGroupParams[GrX_ind,indecis] = Out_arr[*,0]
	CGroupParams[GrY_ind,indecis] = Out_arr[*,1]
	CGroupParams[GrZ_ind,indecis] = Out_arr[*,2]

	; placeholder - save 2D transformations
	xin = transpose(DatFid[*,anc_ind])
	xpin = transpose(TargFid[*,anc_ind])
	AFFINE_SOLVE_GS, xpin, xin, R, c, P, Q, Err_mean
endif

FiducialCoeff[LabelToTransform-1].P=P
FiducialCoeff[LabelToTransform-1].Q=Q
print,'P=',FiducialCoeff[LabelToTransform-1].P
print,'Q=',FiducialCoeff[LabelToTransform-1].Q

; if checked (and NOT 3D affine), and Z data exists do Z-alignement
Align_Z_TipTilt_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Align_Z_TipTilt')
Align_Z_TipTilt=widget_info(Align_Z_TipTilt_button_id,/button_set)
Align_Z_Shift_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Align_Z_Shift')
Align_Z_Shift=widget_info(Align_Z_Shift_button_id,/button_set)
Align_Z = (Align_Z_TipTilt or Align_Z_Shift) and (Transf_Meth ne 4)

if (total(DatZ) ne 0) and Align_Z then begin
	Zo=DatZ[anc_ind]
	Zi=TargZ[anc_ind]
	dZi=Zi-Zo
	if (fid0_cnt ge 3) and Align_Z_TipTilt then begin
		A=[[total(Xi*Xi),total(Xi*Yi),total(Xi)],[total(Xi*Yi),total(Yi*Yi),total(Yi)],[total(Xi),total(Yi),n_elements(Xi)]]
		B=[total(Xi*dZi),total(Yi*dZi),total(dZi)]
		LUDC, A, INDEX
		Plane_coeff = LUSOL(A, INDEX, B)

		Zdelta = CGroupParams[X_ind,indecis]*Plane_coeff[0] + CGroupParams[3,indecis]*Plane_coeff[1] + Plane_coeff[2]
		ZGrdelta = CGroupParams[GrX_ind,indecis]*Plane_coeff[0] + CGroupParams[GrY_ind,indecis]*Plane_coeff[1] + Plane_coeff[2]
	endif else begin
		Zdelta = mean(dZi)
		ZGrdelta = Zdelta
	endelse
	if (n_elements(wind_range) gt 0) and (UnwZ_ind ge 0) then begin
		WR=wind_range[LabelToTransform-1]
		CGroupParams[Z_ind,indecis] = (CGroupParams[Z_ind,indecis] + Zdelta + 4.0 * WR) mod WR
		CGroupParams[UnwZ_ind,indecis] = CGroupParams[UnwZ_ind,indecis] + Zdelta
		CGroupParams[GrZ_ind,indecis] = (CGroupParams[GrZ_ind,indecis] + ZGrdelta + 4.0 * WR) mod WR
		CGroupParams[UnwGrZ_ind,indecis] = CGroupParams[UnwGrZ_ind,indecis] + ZGrdelta
	endif else begin
		CGroupParams[Z_ind,indecis] = CGroupParams[Z_ind,indecis] + Zdelta
		CGroupParams[GrZ_ind,indecis] = CGroupParams[GrZ_ind,indecis] + ZGrdelta
	endelse
endif

; reset the image size and transform TotalRaw in the case of a singe label
if LTT le 1 then begin
	x0=xydsz[0] & y0=xydsz[1]
	if  (Transf_Meth eq 1) and (fid0_cnt ge 3) then begin
		X=[0,0,x0,x0]
		X1=X*0.0
		Y=[0,y0,0,y0]
		Y1=Y*0.0
		for xj=0, PW_deg do begin
			for yj=0, PW_deg do begin
				X1 = X1+Kx[yj,xj]*(X^xj)*(Y^yj)
				Y1 = Y1+Ky[yj,xj]*(X^xj)*(Y^yj)
			endfor
		endfor
	endif	else begin
		PQi=[[FiducialCoeff[LabelToTransform-1].P[0,1],FiducialCoeff[LabelToTransform-1].Q[0,1]],$
				[FiducialCoeff[LabelToTransform-1].P[1,0],FiducialCoeff[LabelToTransform-1].Q[1,0]]]
		ABi=[FiducialCoeff[LabelToTransform-1].P[0,0],FiducialCoeff[LabelToTransform-1].Q[0,0]]
		PQ=INVERT(PQi)
		AB=PQ#ABi
		P=[[-AB[0],PQ[0,1]],[PQ[0,0],0]]
		Q=[[-AB[1],PQ[1,1]],[PQ[1,0],0]]
		X=[0,0,x0,x0]
		Y=[0,y0,0,y0]
		X1 = (P[0,0]+P[0,1]*X+P[1,0]*Y)
		Y1 = (Q[0,0]+Q[0,1]*X+Q[1,0]*Y)
	endelse

	xsz = max(X1)
	ysz = max(Y1)
	if Use_XYlimits then begin
		xsz = XYlimits[1,0]
		ysz = XYlimits[1,1]
	endif
	print,'new XY image bounds are: ', X1, Y1, ',  setting window as:', xsz, ysz
	xydsz=[xsz,ysz]

	transf_scl=sqrt(FiducialCoeff[LabelToTransform-1].P[1,0]^2+	FiducialCoeff[LabelToTransform-1].P[0,1]^2)
	if ~LeaveOrigTotalRaw and (long(xsz)*long(ysz) ne 0) then begin
		TotalRawData=poly_2D(temporary(TotalRawData),FiducialCoeff[LabelToTransform-1].P,FiducialCoeff[LabelToTransform-1].Q,1,xsz,ysz,MISSING=0)
		if (n_elements(DIC) gt 10) then DIC=poly_2D(temporary(DIC),FiducialCoeff[LabelToTransform-1].P,FiducialCoeff[LabelToTransform-1].Q,1,xsz,ysz,MISSING=0)
	endif

	ParamLimits[X_ind,0] = XYlimits[0,0]
	ParamLimits[X_ind,1] = xsz
	ParamLimits[X_ind,2] = (ParamLimits[X_ind,0]+ParamLimits[X_ind,1])/2.0
	ParamLimits[X_ind,3] = (ParamLimits[X_ind,1]-ParamLimits[X_ind,0])
	ParamLimits[Y_ind,0] = XYlimits[0,1]
	ParamLimits[Y_ind,1] = ysz
	ParamLimits[Y_ind,2] = (ParamLimits[Y_ind,0]+ParamLimits[Y_ind,1])/2.0
	ParamLimits[Y_ind,3] = (ParamLimits[Y_ind,1]-ParamLimits[Y_ind,0])
	ParamLimits[Z_ind,0] = min(CGroupParams[Z_ind,*])
	ParamLimits[Z_ind,1] = max(CGroupParams[Z_ind,*])
	ParamLimits[Z_ind,2] = (ParamLimits[Z_ind,0]+ParamLimits[Z_ind,1])/2.0
	ParamLimits[Z_ind,3] = (ParamLimits[Z_ind,1]-ParamLimits[Z_ind,0])
	ParamLimits[GrX_ind,*] = ParamLimits[X_ind,*]
	ParamLimits[GrY_ind,*] = ParamLimits[Y_ind,*]
	ParamLimits[GrZ_ind,*] = ParamLimits[Z_ind,*]

; if selected, adjust scales for Gaussian widths and localization sigmas
	if Adj_Scl then begin
		nm_per_pixel=nm_per_pixel*transf_scl
		;scalable parameters
		sc_ind = [Xwid_ind, Ywid_ind, SigNphX_ind, SigNphY_ind, SigX_ind, SigY_ind, GrSigX_ind, GrSigY_ind]

		for isc = 0,(n_elements(sc_ind)-1) do begin
			CGroupParams[sc_ind[isc],*]=CGroupParams[sc_ind[isc],*]/transf_scl
			ParamLimits[sc_ind[isc],*]=ParamLimits[sc_ind[isc],*]/transf_scl
		endfor
		ParamLimits[X_ind,0:2] = [0,xsz,xsz/2.0]
		ParamLimits[GrX_ind,*] = ParamLimits[X_ind,*]
		ParamLimits[Y_ind,0:2] = [0,ysz,ysz/2.0]
		ParamLimits[GrY_ind,*] = ParamLimits[Y_ind,*]
	endif
	wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
	widget_control,wtable,set_value=transpose(ParamLimits[0:(CGrpSize-1),0:3]), use_table_select=[0,0,3,(CGrpSize-1)]
	widget_control, wtable, /editable,/sensitive
endif


; Reset bridge if bridge is loaded
if bridge_exists then begin
	print,'Reloading the Bridge Array'
	CATCH, Error_status
	CGroupParams_bridge = SHMVAR(shmName_data)
	CGroupParams_bridge[X_ind,*] = CGroupParams[X_ind,*]
	CGroupParams_bridge[GrX_ind,*] = CGroupParams[GrX_ind,*]
	CGroupParams_bridge[Y_ind,*] = CGroupParams[Y_ind,*]
	CGroupParams_bridge[GrY_ind,*] = CGroupParams[GrY_ind,*]
	CGroupParams_bridge[Z_ind,*] = CGroupParams[Z_ind,*]
	CGroupParams_bridge[GrZ_ind,*] = CGroupParams[GrZ_ind,*]
	IF Error_status NE 0 THEN BEGIN
		PRINT, 'Bridge Refresh Error: Drift Correction:',!ERROR_STATE.MSG
		bridge_exists = 0
		SHMUnmap, shmName_data
		SHMUnmap, shmName_filter
		for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
		PRINT, 'Starting: Error:',!ERROR_STATE.MSG
		CATCH,/CANCEL
	ENDIF
	CATCH,/CANCEL
	print,'Finished Reloading the Bridge Array'
endif

return
end
;
;-----------------------------------------------------------------
;
pro DoInsertAnchor, Event				;Insert coordinate text into anchor point tables
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
widget_control,event.id,get_value=thevalue
AnchorPnts[event.x,event.y]=thevalue[event.x,event.y]
widget_control,event.id,set_value=(AnchorPnts[0:5,0:(AnchPnts_MaxNum-1)]), use_table_select=[0,0,5,(AnchPnts_MaxNum-1)]
end
;
;-----------------------------------------------------------------
;
pro DoInsertZAnchor, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
widget_control,event.id,get_value=thevalue
ZPnts[event.x,event.y]=thevalue[event.x,event.y]
widget_control,event.id,set_value=(ZPnts[0:2,0:(AnchPnts_MaxNum-1)]), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]
end
;
;-----------------------------------------------------------------
;
; Empty stub procedure used for autoloading.

pro AnchorWid_eventcb
end
;
;-----------------------------------------------------------------
;
pro InitializeFidAnchWid, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

Last_SAV_filename_label_ID = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,Last_SAV_filename_label_ID,get_value=Last_SAV_filename

IDL_pos=strpos(Last_SAV_filename,'_IDL')
if IDL_pos gt 0 then Last_SAV_filename_s = strmid(Last_SAV_filename,0,IDL_pos) else $
Last_SAV_filename_s = strmid(Last_SAV_filename,0,strlen(Last_SAV_filename)-4)
cam_pos=strpos(Last_SAV_filename_s,'cam')

if cam_pos gt 1 then begin
	AnchorFile=strmid(Last_SAV_filename_s,0,cam_pos+3)+'123_F1F2'+strmid(Last_SAV_filename_s,cam_pos+4,strlen(first_file)-cam_pos-4)+'.anc'
endif else begin
	cam_pos3=strpos(Last_SAV_filename_s,'c3')
	if cam_pos3 gt 1 then begin
		AnchorFile=strmid(Last_SAV_filename_s,0,cam_pos3+1)+'123_F1F2'+strmid(Last_SAV_filename_s,cam_pos3+2,strlen(Last_SAV_filename_s)-cam_pos3-2)+'.anc'
	endif else AnchorFile=Last_SAV_filename_s+'_Fid.anc'
endelse

if AnchorFile ne '' then AnchorFile=AddExtension(AnchorFile,'.anc')

AncFileWidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_ANCFilename')
widget_control,AncFileWidID,SET_VALUE = AnchorFile

Populate_FidAnchWid_settings, wWidget

transf_scl=0
end
;
;-----------------------------------------------------------------
;
pro Populate_FidAnchWid_settings, wWidget
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position

if n_elements(AnchorPnts) eq 0 then begin
	AnchorPnts=dblarr(6,AnchPnts_MaxNum)
	ZPnts=dblarr(3,AnchPnts_MaxNum)
endif
XY_Anc_Table_ID=widget_info(wWidget,FIND_BY_UNAME='WID_Anchors_XY_Table')
widget_control,XY_Anc_Table_ID,set_value=(AnchorPnts), use_table_select=[0,0,5,(AnchPnts_MaxNum-1)]
Z_Anc_Table_ID=widget_info(wWidget,FIND_BY_UNAME='WID_Anchors_Z_Table')
widget_control,Z_Anc_Table_ID,set_value=(ZPnts), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]

if n_elements(Transf_Meth) eq 0 then Transf_Meth=0
WidID_Transf_Meth = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_TRANSFORM_METHOD')
widget_control,WidID_Transf_Meth,SET_DROPLIST_SELECT=Transf_Meth	;0 - linear regression 1 - polywarp; 2 - pivot and average transform (in case of three fiducials only)

if n_elements(PW_deg) eq 0 then PW_deg=1
WidID_WID_SLIDER_POLYWARP_Degree = Widget_Info(wWidget, find_by_uname='WID_SLIDER_POLYWARP_Degree')
widget_control,WidID_WID_SLIDER_POLYWARP_Degree,SET_VALUE=PW_deg

if n_elements(Fid_Outl_Sz) eq 0 then Fid_Outl_Sz=0.5
Fid_Outl_Sz_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_FidOutlineSize')
Fid_Outl_Sz_txt=string(Fid_Outl_Sz[0],FORMAT='(F6.2)')
widget_control,Fid_Outl_Sz_ID,SET_VALUE = Fid_Outl_Sz_txt

if n_elements(AutoDisp_Sel_Fids) eq 0 then AutoDisp_Sel_Fids=0
WID_BUTTON_AutoDisp_Sel_Fids_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_AutoDisplay_Selected_Fiducials')
widget_control, WID_BUTTON_AutoDisp_Sel_Fids_ID, set_button = AutoDisp_Sel_Fids

if n_elements(Disp_Fid_IDs) eq 0 then Disp_Fid_IDs=0
Display_Fiducials_IDs_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Display_Fiducial_IDs')
widget_control, Display_Fiducials_IDs_ID, set_button = Disp_Fid_IDs

if n_elements(AutoDet_Params) eq 0 then AutoDet_Params = [20.0,1000.0,1.0]
WID_AutoDet_Params_ID = Widget_Info(wWidget, find_by_uname='WID_AutoDetect_Parameters')
widget_control,WID_AutoDet_Params_ID,set_value=transpose(AutoDet_Params), use_table_select=[0,0,0,2]

if n_elements(AutoMatch_Params) eq 0 then AutoMatch_Params = [20.0,1000.0,10.0]
WID_AutoDetect_Match_Parameters_ID = Widget_Info(wWidget, find_by_uname='WID_AutoDetect_Match_Parameters')
widget_control,WID_AutoDetect_Match_Parameters_ID,set_value=transpose(AutoMatch_Params), use_table_select=[0,0,0,2]

if n_elements(Adj_Scl) eq 0 then Adj_Scl=0
Adj_Scl_button_id=widget_info(wWidget,FIND_BY_UNAME='WID_BUTTON_Adj_Scl')
widget_control, Adj_Scl_button_id, set_button = Adj_Scl

if n_elements(LeaveOrigTotalRaw) eq 0 then LeaveOrigTotalRaw=0
WID_BUTTON_WID_BUTTON_LeaveOrigTotalRaw_button_id=widget_info(wWidget,FIND_BY_UNAME='WID_BUTTON_LeaveOrigTotalRaw')
widget_control, WID_BUTTON_WID_BUTTON_LeaveOrigTotalRaw_button_id, set_button = LeaveOrigTotalRaw

if n_elements(Use_XYlimits) eq 0 then Use_XYlimits=0
WID_BUTTON_XYlimits_button_id=widget_info(wWidget,FIND_BY_UNAME='WID_BUTTON_XYlimits')
widget_control, WID_BUTTON_XYlimits_button_id, set_button = Use_XYlimits

if n_elements(XYlimits) eq 0 then begin
	if n_elements(xydsz) le 0 then XYlimits=[[0,0],[0,0]] else XYlimits=[[0,xydsz[0]],[0,xydsz[1]]]
endif
WID_XYlimits_table_id=widget_info(wWidget,FIND_BY_UNAME='WID_XYlimits_table')
widget_control,WID_XYlimits_table_id,set_value=XYlimits, use_table_select=[0,0,1,1]

end
;
;-----------------------------------------------------------------
;
pro OnButton_AddRedFiducial, Event				; Adds Red Fiducial values to the table (intensity weighted averaged coordinate between all Red label peak on the screen (with filters)
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position

FidSource_WidID = widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Fiducial_Source')
FidSource = widget_info(FidSource_WidID,/DropList_Select)

filter0=filter
filter = filter and (CGroupParams[LabelSet_ind,*] le 1)
indecis=where(filter eq 1)

if FidSource then begin
	xposition=ParamLimits[X_ind,2]
	yposition=ParamLimits[Y_ind,2]
	zposition=ParamLimits[Z_ind,2]
endif else begin
	if (size(indecis))[0] eq 0 then return
	xposition=total(CGroupParams[X_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
	yposition=total(CGroupParams[Y_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
	zposition=(Z_ind ge 0) ? total(CGroupParams[Z_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis]) : 0
endelse
AnchorInd=min(where(AnchorPnts[0,*] eq 0))
if AnchorInd eq -1 then AnchorInd=99
xpr=AnchorPnts[0,((AnchorInd-1)>0)]
ypr=AnchorPnts[1,((AnchorInd-1)>0)]
fdist=sqrt((xpr-xposition)^2+(ypr-yposition)^2)
if fdist gt 0.1 then begin
	AnchorPnts[0:1,AnchorInd]=[xposition,yposition]
	ZPnts[0,AnchorInd]=zposition
	XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
	widget_control,XY_Anc_Table_ID,set_value=([xposition,yposition]), use_table_select=[0,AnchorInd,1,AnchorInd]
	if Z_ind ge 0 then begin
		Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
		widget_control,Z_Anc_Table_ID,set_value=([zposition]), use_table_select=[0,AnchorInd,0,AnchorInd]
	endif
endif
filter=filter0
if AutoDisp_Sel_Fids then begin
	Fid_Outline_xpos = xposition
	Fid_Outline_ypos = yposition
	Fid_Outline_color = 0
	Display_single_fiducial_outline, Event, Fid_Outline_xpos, Fid_Outline_ypos, Fid_Outl_Sz, Fid_Outline_color
endif
end
;
;-----------------------------------------------------------------
;
pro OnButton_AddGreenFiducial, Event					; Adds Green Fiducial values to the table (intensity weighted averaged coordinate between all Green label peak on the screen (with filters)
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position

FidSource_WidID = widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Fiducial_Source')
FidSource = widget_info(FidSource_WidID,/DropList_Select)

filter0=filter
filter = filter and (CGroupParams[LabelSet_ind,*] eq 2)
indecis=where(filter eq 1)

if FidSource then begin
	xposition=ParamLimits[X_ind,2]
	yposition=ParamLimits[Y_ind,2]
	zposition=ParamLimits[Z_ind,2]
endif else begin
	if (size(indecis))[0] eq 0 then return
	xposition=total(CGroupParams[X_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
	yposition=total(CGroupParams[Y_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
	zposition=(Z_ind ge 0) ? total(CGroupParams[Z_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis]) : 0
endelse
AnchorInd=min(where(AnchorPnts[2,*] eq 0))
if AnchorInd eq -1 then AnchorInd=99
xpr=AnchorPnts[2,((AnchorInd-1)>0)]
ypr=AnchorPnts[3,((AnchorInd-1)>0)]
fdist=sqrt((xpr-xposition)^2+(ypr-yposition)^2)
if fdist gt 0.1 then begin
	AnchorPnts[2:3,AnchorInd]=[xposition,yposition]
	ZPnts[1,AnchorInd]=zposition
	XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
	widget_control,XY_Anc_Table_ID,set_value=([xposition,yposition]), use_table_select=[2,AnchorInd,3,AnchorInd]
	if Z_ind ge 0 then begin
		Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
		widget_control,Z_Anc_Table_ID,set_value=([zposition]), use_table_select=[1,AnchorInd,1,AnchorInd]
	endif
endif
filter=filter0
if AutoDisp_Sel_Fids then begin
	Fid_Outline_xpos = xposition
	Fid_Outline_ypos = yposition
	Fid_Outline_color = 1
	Display_single_fiducial_outline, Event, Fid_Outline_xpos, Fid_Outline_ypos, Fid_Outl_Sz, Fid_Outline_color
endif
end
;
;-----------------------------------------------------------------
;
pro OnButton_AddBlueFiducial, Event			; Adds Blue Fiducial values to the table (intensity weighted averaged coordinate between all Blue label peak on the screen (with filters)
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position

FidSource_WidID = widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Fiducial_Source')
FidSource = widget_info(FidSource_WidID,/DropList_Select)

filter0=filter
filter = filter and (CGroupParams[LabelSet_ind,*] eq 3)
indecis=where(filter eq 1)

if FidSource then begin
	xposition=ParamLimits[X_ind,2]
	yposition=ParamLimits[Y_ind,2]
	zposition=ParamLimits[Z_ind,2]
endif else begin
	if (size(indecis))[0] eq 0 then return
	xposition=total(CGroupParams[X_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
	yposition=total(CGroupParams[Y_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
	zposition=(Z_ind ge 0) ? total(CGroupParams[Z_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis]) : 0
endelse
AnchorInd=min(where(AnchorPnts[4,*] eq 0))
if AnchorInd eq -1 then AnchorInd=99
xpr=AnchorPnts[4,((AnchorInd-1)>0)]
ypr=AnchorPnts[5,((AnchorInd-1)>0)]
fdist=sqrt((xpr-xposition)^2+(ypr-yposition)^2)
if fdist gt 0.1 then begin
	AnchorPnts[4:5,AnchorInd]=[xposition,yposition]
	ZPnts[2,AnchorInd]=zposition
	XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
	widget_control,XY_Anc_Table_ID,set_value=([xposition,yposition]), use_table_select=[4,AnchorInd,5,AnchorInd]
	if Z_ind ge 0 then begin
		Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
		widget_control,Z_Anc_Table_ID,set_value=([zposition]), use_table_select=[2,AnchorInd,2,AnchorInd]
	endif
endif
filter=filter0
if AutoDisp_Sel_Fids then begin
	Fid_Outline_xpos = xposition
	Fid_Outline_ypos = yposition
	Fid_Outline_color = 2
	Display_single_fiducial_outline, Event, Fid_Outline_xpos, Fid_Outline_ypos, Fid_Outl_Sz, Fid_Outline_color
endif
end
;
;-----------------------------------------------------------------
;
pro OnButton_SwapRedGreen, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw

Red=AnchorPnts[0:1,*]
RedZ=ZPnts[0,*]
AnchorPnts[0:1,*]=AnchorPnts[2:3,*]
ZPnts[0,*]=ZPnts[1,*]
AnchorPnts[2:3,*]=Red
ZPnts[1,*]=RedZ
XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
widget_control,XY_Anc_Table_ID,set_value=(AnchorPnts), use_table_select=[0,0,5,(AnchPnts_MaxNum-1)]
Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
widget_control,Z_Anc_Table_ID,set_value=(ZPnts), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]
end
;
;-----------------------------------------------------------------
;
pro OnButton_SwapRedBlue, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
Red=AnchorPnts[0:1,*]
RedZ=ZPnts[0,*]
AnchorPnts[0:1,*]=AnchorPnts[4:5,*]
ZPnts[0,*]=ZPnts[2,*]
AnchorPnts[4:5,*]=Red
ZPnts[2,*]=RedZ
XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
widget_control,XY_Anc_Table_ID,set_value=(AnchorPnts), use_table_select=[0,0,5,(AnchPnts_MaxNum-1)]
Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
widget_control,Z_Anc_Table_ID,set_value=(ZPnts), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]
end
;
;-----------------------------------------------------------------
;
pro OnButton_Copy_Red_to_Green, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
AnchorPnts[2:3,*]=AnchorPnts[0:1,*]
ZPnts[1,*]=ZPnts[0,*]
XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
widget_control,XY_Anc_Table_ID,set_value=(AnchorPnts), use_table_select=[0,0,5,(AnchPnts_MaxNum-1)]
Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
widget_control,Z_Anc_Table_ID,set_value=(ZPnts), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]
end
;
;-----------------------------------------------------------------
;
pro ClearFiducials, Event			; Clears the table
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
AnchorPnts=dblarr(6,AnchPnts_MaxNum)
ZPnts=dblarr(3,AnchPnts_MaxNum)
ZeroAverPnts=dblarr(3)
XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
widget_control,XY_Anc_Table_ID,set_value=(AnchorPnts), use_table_select=[0,0,5,(AnchPnts_MaxNum-1)]
Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
widget_control,Z_Anc_Table_ID,set_value=(ZPnts), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]
WID_Anchors_Transf_Error_Test_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Transf_Dist_Test')
widget_control,WID_Anchors_Transf_Error_Test_ID,set_value=(ZPnts), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]
WID_Anchors_Transf_AverageErrors_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Transf_AverageErrors')
widget_control,WID_Anchors_Transf_AverageErrors_ID,set_value=(ZeroAverPnts)
end
;
;-----------------------------------------------------------------
;
pro Do_RGB_Transforms, Event		;Performs the transformations according to selecetions
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

WID_BUTTON_XYlimits_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_XYlimits')
Use_XYlimits=widget_info(WID_BUTTON_XYlimits_button_id,/button_set)

; check which transformations are selected
if AnchorFile ne '' then Save_ANC_File, Event

rtg_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_RTG')		;R=Data   G=Target
rtg=widget_info(rtg_button_id,/button_set)
rtb_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_RTB')		;R=Data   B=Target
rtb=widget_info(rtb_button_id,/button_set)
gtr_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_GTR')		;G=Data   R=Target
gtr=widget_info(gtr_button_id,/button_set)
gtb_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_GTB')		;G=Data   B=Target
gtb=widget_info(gtb_button_id,/button_set)
btr_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_BTR')		;B=Data   R=Target
btr=widget_info(btr_button_id,/button_set)
btg_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_BTG')		;B=Data   G=Target
btg=widget_info(btg_button_id,/button_set)

widget_control,rtg_button_id,set_button=0
widget_control,rtb_button_id,set_button=0
widget_control,gtr_button_id,set_button=0
widget_control,gtb_button_id,set_button=0
widget_control,btr_button_id,set_button=0
widget_control,btg_button_id,set_button=0

if rtg then DoExRedtoGreen, Event
if rtb then DoExRedtoBlue, Event
if gtr then DoExGreentoRed, Event
if gtb then DoExGreentoBlue, Event
if btr then DoExBluetoRed, Event
if btg then DoExBluetoGreen, Event

Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
if Use_XYlimits then Purge_by_Size, Event1
widget_control,event.top,/destroy

end
;
;-----------------------------------------------------------------
pro DoExRedtoGreen, Event				;extrapolate Red data to Green fixed points
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
;R=Data   G=Target
LabelToTransform=1
LabelTarget=2
DoDataFidtoTargetFid, LabelToTransform, LabelTarget, Event
AnchorPnts[0:1,*]=0
end
;
;-----------------------------------------------------------------
;
pro DoExGreentoRed, Event				;extrapolate Green to Red fixed points
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
LabelToTransform=2
LabelTarget=1
DoDataFidtoTargetFid, LabelToTransform, LabelTarget, Event
AnchorPnts[2:3,*]=0
end
;
;-----------------------------------------------------------------
;
pro DoExBluetoRed, Event				;extrapolate Blue to Red fixed points
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
LabelToTransform=3
LabelTarget=1
DoDataFidtoTargetFid, LabelToTransform, LabelTarget, Event
AnchorPnts[4:5,*]=0
end
;
;-----------------------------------------------------------------
;
pro DoExBluetoGreen, Event				;extrapolate Blue to Green fixed points
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
LabelToTransform=3
LabelTarget=2
DoDataFidtoTargetFid, LabelToTransform, LabelTarget, Event
AnchorPnts[4:5,*]=0
end
;
;-----------------------------------------------------------------
;
pro DoExGreentoBlue, Event				;extrapolate Green to Blue fixed points
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
LabelToTransform=2
LabelTarget=3
DoDataFidtoTargetFid, LabelToTransform, LabelTarget, Event
AnchorPnts[2:3,*]=0
end
;
;-----------------------------------------------------------------
;
pro DoExRedtoBlue, Event				;extrapolate Red to Blue fixed points
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
LabelToTransform=1
LabelTarget=3
DoDataFidtoTargetFid, LabelToTransform, LabelTarget, Event
AnchorPnts[0:1,*]=0
end
;
;-----------------------------------------------------------------
;
pro RGB_check_buttons, Event		; checks the selected transformations for consistency and corrects if needed
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
; check which transformations are selected
rtg_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_RTG')		;R=Data   G=Target
rtg=widget_info(rtg_button_id,/button_set)
rtb_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_RTB')		;R=Data   B=Target
rtb=widget_info(rtb_button_id,/button_set)
gtr_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_GTR')		;G=Data   R=Target
gtr=widget_info(gtr_button_id,/button_set)
gtb_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_GTB')		;G=Data   B=Target
gtb=widget_info(gtb_button_id,/button_set)
btr_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_BTR')		;B=Data   R=Target
btr=widget_info(btr_button_id,/button_set)
btg_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_BTG')		;B=Data   G=Target
btg=widget_info(btg_button_id,/button_set)

tr_checked=widget_info(event.id,/UNAME)

if (tr_checked eq 'WID_BUTTON_RTG') and rtg then begin
	widget_control,rtb_button_id,set_button=0
	widget_control,gtr_button_id,set_button=0
endif
if (tr_checked eq 'WID_BUTTON_RTB') and rtb then begin
	widget_control,rtg_button_id,set_button=0
	widget_control,btr_button_id,set_button=0
endif
if (tr_checked eq 'WID_BUTTON_GTR') and gtr then begin
	widget_control,gtb_button_id,set_button=0
	widget_control,rtg_button_id,set_button=0
endif
if (tr_checked eq 'WID_BUTTON_GTB') and gtb then begin
	widget_control,gtr_button_id,set_button=0
	widget_control,btg_button_id,set_button=0
endif
if (tr_checked eq 'WID_BUTTON_BTR') and btr then begin
	widget_control,btg_button_id,set_button=0
	widget_control,rtb_button_id,set_button=0
endif
if (tr_checked eq 'WID_BUTTON_BTG') and btg then begin
	widget_control,btr_button_id,set_button=0
	widget_control,gtb_button_id,set_button=0
endif

end
;
;-----------------------------------------------------------------
;
pro OnPushButton_AlignZ_TipTilt, Event
	Align_Z_Shift_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Align_Z_Shift')
	if Align_Z_Shift_button_id gt 0 then if Event.select then widget_control, Align_Z_Shift_button_id, set_button=0
end
;
;-----------------------------------------------------------------
;
pro OnPushButton_AlignZ_Shift, Event
	Align_Z_TipTilt_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Align_Z_TipTilt')
	if Align_Z_TipTilt_button_id gt 0 then if Event.select then widget_control, Align_Z_TipTilt_button_id, set_button=0
end
;
;-----------------------------------------------------------------
;
pro OnPickANCFile, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
AnchorFile = Dialog_Pickfile(/read,get_path=fpath,filter=['*.anc'],title='Select *.anc file to open')
if AnchorFile ne '' then cd,fpath
AncFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_ANCFilename')
widget_control,AncFileWidID,SET_VALUE = AnchorFile
end
;
;-----------------------------------------------------------------
;
pro Load_ANC_File, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
AncFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_ANCFilename')
widget_control,AncFileWidID,GET_VALUE = AnchorFile
AncFileInfo=file_info(AnchorFile)
if ~AncFileInfo.exists then return
AnchorPnts=dblarr(6,AnchPnts_MaxNum)
AnchorPnts_line=dblarr(6)
ZPnts=dblarr(3,AnchPnts_MaxNum)
ZPnts_line=dblarr(3)
close,5
openr,5,AnchorFile
ip=0
while (not EOF(5)) and (ip lt AnchPnts_MaxNum) do begin
	readf,5,AnchorPnts_line
	AnchorPnts[*,ip] = AnchorPnts_line
	ip+=1
endwhile
close,5

Anc_Z_FileInfo=file_info(AnchorFile+'z')
if Anc_Z_FileInfo.exists then begin
	ip=0
	openr,5,(AnchorFile+'z')
	while (not EOF(5)) and (ip lt AnchPnts_MaxNum)  do begin
		readf,5,ZPnts_line
		ZPnts[*,ip] = ZPnts_line
		ip+=1
	endwhile
	close,5
endif

AncSav=addextension(AnchorFile,'.sav')
AncSav_FileInfo=file_info(AncSav)
if AncSav_FileInfo.exists then restore, filename=AncSav
Populate_FidAnchWid_settings, Event.top

end
;
;-----------------------------------------------------------------
;
pro Save_ANC_File, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
AncFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_ANCFilename')
widget_control,AncFileWidID,GET_VALUE = AnchorFile
if AnchorFile eq '' then return
openw,1,AnchorFile,width=512
printf,1,AnchorPnts
close,1
Align_Z_TipTilt_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Align_Z_TipTilt')
Align_Z_TipTilt=widget_info(Align_Z_TipTilt_button_id,/button_set)
Align_Z_Shift_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Align_Z_Shift')
Align_Z_Shift=widget_info(Align_Z_Shift_button_id,/button_set)
Align_Z = Align_Z_TipTilt or Align_Z_Shift or (Transf_Meth eq 4)
if Align_Z then begin
	openw,1,(AnchorFile+'z'),width=512
	printf,1,ZPnts
	close,1
endif

save, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, $
		Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw, filename=addextension(AnchorFile,'.sav')

end
;
;-----------------------------------------------------------------
;
pro Display_single_fiducial_outline, Event, Fid_Outline_xpos, Fid_Outline_ypos, Fid_Outl_Sz, Fid_Outline_color
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate

dxsz=xydsz[0] & dysz=xydsz[1]
wxsz=1024 & wysz=1024
dxmn = paramlimits[2,0]
dymn = paramlimits[3,0]
dxmx = paramlimits[2,1]
dymx = paramlimits[3,1]
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))

angle_stp=5.0
cir_stps=fix(round(360.0/angle_stp))+1
x_loc_rgb=fltarr(cir_stps)
y_loc_rgb=fltarr(cir_stps)
TVLCT,R0,G0,B0,/GET
TVLCT, [[255], [0], [0]], 0
TVLCT, [[0], [255], [0]], 1
TVLCT, [[0], [0], [255]], 2

x_center = mgw * (Fid_Outline_xpos-dxmn)
y_center = mgw * (Fid_Outline_ypos-dymn)
for th=0,cir_stps-1 do begin
	x_loc_rgb[th] = x_center + Fid_Outl_Sz*(sin(th*angle_stp/180.0*!pi)) * mgw
	y_loc_rgb[th] = y_center + Fid_Outl_Sz*(cos(th*angle_stp/180.0*!pi)) * mgw
endfor
plots,x_loc_rgb,y_loc_rgb,/device,color=Fid_Outline_color
TVLCT,R0,G0,B0
end
;
;-----------------------------------------------------------------
;
pro Display_RGB_fiducials, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
COMMON COLORS, R_orig, G_orig, B_orig, R_curr, G_curr, B_curr

dxsz=xydsz[0] & dysz=xydsz[1]
wxsz=1024 & wysz=1024
dxmn = paramlimits[2,0]
dymn = paramlimits[3,0]
dxmx = paramlimits[2,1]
dymx = paramlimits[3,1]
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))

cc=where(AnchorPnts[*,0] ne 0, fid_col)
num_colors = fid_col/2
angle_stp=5.0
cir_stps=fix(round(360.0/angle_stp))+1
x_loc_rgb=fltarr(cir_stps)
y_loc_rgb=fltarr(cir_stps)
TVLCT,R0,G0,B0,/GET
TVLCT, [[255], [0], [0]], 0
TVLCT, [[0], [255], [0]], 1
TVLCT, [[0], [0], [255]], 2

for i_col=0L,num_colors-1 do begin
	cc=where(AnchorPnts[i_col*2,*] ne 0, fid_cnt_col)
	for i_fid=0,fid_cnt_col-1 do begin
		x_center = mgw * (AnchorPnts[i_col*2,i_fid]-dxmn)
		y_center = mgw * (AnchorPnts[i_col*2+1,i_fid]-dymn)
		for th=0,cir_stps-1 do begin
			x_loc_rgb[th] = x_center + Fid_Outl_Sz*(sin(th*angle_stp/180.0*!pi)) * mgw
			y_loc_rgb[th] = y_center + Fid_Outl_Sz*(cos(th*angle_stp/180.0*!pi)) * mgw
		endfor
		plots,x_loc_rgb,y_loc_rgb,/device,color=i_col
		if Disp_Fid_IDs and (i_col eq 0) then begin
			dx0= (dxmx-dxmn) / 2.0 * mgw - x_center
			dy0= (dymx-dymn) / 2.0 * mgw - y_center
			ddx=(dx0 gt 0) ? dx0/sqrt(dx0*dx0+dy0*dy0) * Fid_Outl_Sz * 1.5 * mgw : dx0/sqrt(dx0*dx0+dy0*dy0) * Fid_Outl_Sz * 1.5 * mgw - 10.0
			ddy=(dy0 gt 0) ? dy0/sqrt(dx0*dx0+dy0*dy0) * Fid_Outl_Sz * 1.5	* mgw : dy0/sqrt(dx0*dx0+dy0*dy0) * Fid_Outl_Sz * 1.5 * mgw - 10.0
			xyouts,x_center+ddx,y_center+ddy,strtrim(i_fid,2), /DEVICE,color=i_col
		endif
	endfor
endfor
TVLCT,R0,G0,B0
end
;
;-----------------------------------------------------------------
;
pro Display_RGB_fiducials_with_overlays, Event
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
OnTotalRawDataButton, Event1
OnPeakOverlayAllCentersButton, Event1
Display_RGB_fiducials, Event
end
;
;-----------------------------------------------------------------
;
pro SetFiducialOutlineSize, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
Fid_Outl_Sz_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_FidOutlineSize')
widget_control,Fid_Outl_Sz_ID,GET_VALUE = Fid_Outl_Sz
end
;
;-----------------------------------------------------------------
;
pro Set_AutoDisp_Sel_Fids, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
AutoDisp_Sel_Fids_ID = Widget_Info(Event.top, find_by_uname='WID_BUTTON_AutoDisplayCompleteFidSet')
AutoDisp_Sel_Fids = widget_info(AutoDisp_Sel_Fids_ID,/BUTTON_SET)
end
;
;-----------------------------------------------------------------
;
pro Set_Disp_Fid_IDs, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
Display_Fiducials_IDs_ID = Widget_Info(Event.top, find_by_uname='WID_BUTTON_Disp_Fid_IDs')
Disp_Fid_IDs = widget_info(Display_Fiducials_IDs_ID,/BUTTON_SET)
end
;
;-----------------------------------------------------------------
;
pro OnButton_AutodetectRedFiducials, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position

ClearFiducials, Event
clip=TotalRawData
totdat=clip
d=AutoDet_Params[2]*5
rad_pix = AutoDet_Params[2]
mxcnt = AnchPnts_MaxNum
threshold = AutoDet_Params[0]
;d = thisfitcond.MaskSize							;d=5.			half size of region of a selected peak
																;setup twinkle structure with all the fitted peak information
thisfitcond_tags =tag_names(thisfitcond)
id=where(thisfitcond_tags eq "GaussSig",cnt0)
if cnt0 then Gauss_sigma=thisfitcond.GaussSig else Gauss_sigma=float(thisfitcond.masksize)/4

WIDGET_CONTROL,/HOURGLASS
FindPeaks, clip, totdat, d, Gauss_sigma, threshold, mxcnt, peakxa, peakya, maxpeakcriteria, criteria		;Create and ordered list of peak candidate coordinates
if mxcnt eq 0 then begin
	z=dialog_message('No fiducials detected: Step1')
	return
endif
filter_peaks = (maxpeakcriteria le AutoDet_Params[1]) and $
	(peakxa ge (ParamLimits[2,0] - rad_pix)) and (peakxa le (ParamLimits[2,1] + rad_pix)) and $
	(peakya ge (ParamLimits[3,0] - rad_pix)) and (peakya le (ParamLimits[3,1] + rad_pix))
peakindecis=where(filter_peaks, n_peaks)
if n_peaks lt 1 then begin
	z=dialog_message('No fiducials detected; Step2')
	return
endif
;print,'threshold of the detected peaks:  ',maxpeakcriteria[peakindecis]
peakxa_filtered = peakxa[peakindecis]
peakya_filtered = peakya[peakindecis]
FilterId = widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Autodetect_Filter')
FilterItem = widget_info(FilterId,/DropList_Select)
Xindex = FilterItem ? GrX_ind : X_ind
Yindex = FilterItem ? GrY_ind : Y_ind
XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
if Z_ind ge 0 then begin
	Zindex = FilterItem ? GrZ_ind : Z_ind
	Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
endif

filter0 = filter and (CGroupParams[LabelSet_ind,*] le 1)
AnchorInd=0

for i=0,(n_peaks-1) do begin
	filter1 = filter0 and (CGroupParams[Xindex,*] ge (peakxa_filtered[i] - rad_pix)) and (CGroupParams[Xindex,*] le (peakxa_filtered[i] + rad_pix)) and $
		(CGroupParams[Yindex,*] ge (peakya_filtered[i] - rad_pix)) and (CGroupParams[Yindex,*] le (peakya_filtered[i] + rad_pix))
	indecis=where(filter1,cnt)
	if (cnt ge 1) then begin
		xposition = total(CGroupParams[Xindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
		yposition = total(CGroupParams[Yindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
		min_dist = (AnchorInd eq 0)? rad_pix*3	:	min(sqrt((AnchorPnts[0,0:(AnchorInd-1)]-xposition)^2+(AnchorPnts[1,0:(AnchorInd-1)]-yposition)^2))
		;print,i,min_dist
		if  min_dist gt rad_pix then begin	; add fiducial to the list if not already there
			AnchorPnts[0,AnchorInd] = xposition
			AnchorPnts[1,AnchorInd] = yposition
			widget_control,XY_Anc_Table_ID,set_value=([xposition,yposition]), use_table_select=[0,AnchorInd,1,AnchorInd]
			if Z_ind ge 0 then begin
				zposition =(Z_ind ge 0) ? total(CGroupParams[Zindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis]) : 0
				ZPnts[0,AnchorInd]=zposition
				widget_control,Z_Anc_Table_ID,set_value=([zposition]), use_table_select=[0,AnchorInd,0,AnchorInd]
			endif
			AnchorInd++
		endif
	endif
endfor
if (AnchorInd eq 0) then z=dialog_message('No fiducials detected')
if AutoDisp_Sel_Fids and (AnchorInd gt 0) then  Display_RGB_fiducials, Event
end
;
;-----------------------------------------------------------------
;
pro OnButton_AutodetectMatchingFiducials, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position

fid=where(AnchorPnts[0,*] gt 0.0,n_peaks)
if n_peaks lt 1 then begin
	z=dialog_message('No red fiducials selected')
	return
endif
l_max=max(CGroupParams[LabelSet_ind,*])
if l_max lt 2 then begin
	z=dialog_message('Only one label, load other labels')
	return
endif
WIDGET_CONTROL,/HOURGLASS
rad_pix = AutoMatch_Params[2]
FilterId = widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Autodetect_Filter')
FilterItem = widget_info(FilterId,/DropList_Select)
Xindex = FilterItem ? GrX_ind : X_ind
Yindex = FilterItem ? GrY_ind : Y_ind
XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
if Z_ind ge 0 then begin
	Zindex = FilterItem ? GrZ_ind : Z_ind
	Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
endif

for lbl_index=2,l_max do begin
	filter0 = filter and (CGroupParams[LabelSet_ind,*] eq lbl_index)
	AnchorInd=0
	for i=0,(n_peaks-1) do begin
		filter1 = filter0 and (CGroupParams[Xindex,*] ge (AnchorPnts[0,i] - rad_pix)) and (CGroupParams[Xindex,*] le (AnchorPnts[0,i] + rad_pix)) and $
			(CGroupParams[Yindex,*] ge (AnchorPnts[1,i] - rad_pix)) and (CGroupParams[Yindex,*] le (AnchorPnts[1,i] + rad_pix)) and $
			(CGroupParams[Amp_ind,*] ge AutoMatch_Params[0]) and (CGroupParams[Amp_ind,*] le AutoMatch_Params[1])
		indecis=where(filter1,cnt)
		;print,'lbl=',lbl_index,'    peak=',i,'      filter_count=',cnt
		if (cnt ge 1) then begin
			xposition = total(CGroupParams[Xindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
			yposition = total(CGroupParams[Yindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
			AnchorPnts[(lbl_index-1)*2,i] = xposition
			AnchorPnts[((lbl_index-1)*2+1),i] = yposition
			widget_control,XY_Anc_Table_ID,set_value=([xposition,yposition]), use_table_select=[(lbl_index-1)*2,i,((lbl_index-1)*2+1),i]
			if Z_ind ge 0 then begin
				zposition =(Z_ind ge 0) ? total(CGroupParams[Zindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis]) : 0
				ZPnts[(lbl_index-1),i]=zposition
				widget_control,Z_Anc_Table_ID,set_value=([zposition]), use_table_select=[(lbl_index-1),i,(lbl_index-1),i]
			endif
			AnchorInd++
		endif
	endfor
endfor
if AutoDisp_Sel_Fids and (AnchorInd gt 0) then  Display_RGB_fiducials, Event
end
;
;-----------------------------------------------------------------
;
pro OnButton_RefindFiducials, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position

fid=where(AnchorPnts[0,*] gt 0.0,n_peaks)
if n_peaks lt 1 then begin
	z=dialog_message('No red fiducials selected')
	return
endif
l_max=max(CGroupParams[LabelSet_ind,*])

WIDGET_CONTROL,/HOURGLASS
rad_pix = AutoMatch_Params[2]
FilterId = widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Autodetect_Filter')
FilterItem = widget_info(FilterId,/DropList_Select)
Xindex = FilterItem ? GrX_ind : X_ind
Yindex = FilterItem ? GrY_ind : Y_ind
XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
if Z_ind ge 0 then begin
	Zindex = FilterItem ? GrZ_ind : Z_ind
	Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
endif
AnchorInd=0
if l_max ge 1 then begin
	for lbl_index=1,l_max do begin
		filter0 = filter and (CGroupParams[LabelSet_ind,*] eq lbl_index)
		for i=0,(n_peaks-1) do begin
			filter1 = filter0 and (sqrt((CGroupParams[Xindex,*] - AnchorPnts[(lbl_index-1)*2,i])^2 + (CGroupParams[Yindex,*] - AnchorPnts[((lbl_index-1)*2+1),i])^2) le rad_pix) and $
				(CGroupParams[Amp_ind,*] ge AutoMatch_Params[0]) and (CGroupParams[Amp_ind,*] le AutoMatch_Params[1])
			indecis=where(filter1,cnt)
			;print,'lbl=',lbl_index,'    peak=',i,'      filter_count=',cnt
			xposition = 0
			yposition = 0
			zposition = 0
			if (cnt ge 1) then begin
				xposition = total(CGroupParams[Xindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
				yposition = total(CGroupParams[Yindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
				if Z_ind ge 0 then zposition = total(CGroupParams[Zindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis]) else zposition = 0
				AnchorInd++
			endif
			AnchorPnts[(lbl_index-1)*2,i] = xposition
			AnchorPnts[((lbl_index-1)*2+1),i] = yposition
			widget_control,XY_Anc_Table_ID,set_value=([xposition,yposition]), use_table_select=[(lbl_index-1)*2,i,((lbl_index-1)*2+1),i]
			if Z_ind ge 0 then begin
					ZPnts[(lbl_index-1),i]=zposition
					widget_control,Z_Anc_Table_ID,set_value=([zposition]), use_table_select=[(lbl_index-1),i,(lbl_index-1),i]
			endif
		endfor
	endfor
endif else begin
	for i=0,(n_peaks-1) do begin
			;print,AnchorPnts[0,i], AnchorPnts[1,i]
			filter1 = filter and (sqrt((CGroupParams[Xindex,*] - AnchorPnts[0,i])^2 + (CGroupParams[Yindex,*] - AnchorPnts[1,i])^2) le rad_pix) and $
				(CGroupParams[Amp_ind,*] ge AutoMatch_Params[0]) and (CGroupParams[Amp_ind,*] le AutoMatch_Params[1])
			indecis=where(filter1,cnt)
			;print,   'peak=',i,'      filter_count=',cnt
			xposition = 0
			yposition = 0
			zposition = 0
			if (cnt ge 1) then begin
				xposition = total(CGroupParams[Xindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
				yposition = total(CGroupParams[Yindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
				if Z_ind ge 0 then zposition = total(CGroupParams[Zindex,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis]) else zposition = 0
				AnchorInd++
			endif
			AnchorPnts[0,i] = xposition
			AnchorPnts[1,i] = yposition
			widget_control,XY_Anc_Table_ID,set_value=([xposition,yposition]), use_table_select=[0,i,1,i]
			;print,[xposition,yposition]
			if Z_ind ge 0 then begin
					ZPnts[0,i]=zposition
					widget_control,Z_Anc_Table_ID,set_value=([zposition]), use_table_select=[0,i,0,i]
			endif
		endfor
endelse
if AutoDisp_Sel_Fids and (AnchorInd gt 0) then  Display_RGB_fiducials, Event
end
;
;-----------------------------------------------------------------
;
pro OnButton_Remove_Unmatched, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position

fid=where(AnchorPnts[0,*] gt 0.0,n_peaks)
if n_peaks lt 1 then begin
	z=dialog_message('No red fiducials selected')
	return
endif
l_max=max(CGroupParams[LabelSet_ind,*])
if l_max lt 2 then begin
	z=dialog_message('Only one label, load other labels')
	return
endif
AnchorPnts_new=dblarr(6,AnchPnts_MaxNum)
Anc_valid_indecis=intarr(AnchPnts_MaxNum)
for i=0,(n_peaks-1) do begin
	Anc_valid_indecis[fid[i]]=1
	for lbl_index=1,l_max do begin
		if (AnchorPnts[(lbl_index-1)*2,fid[i]] lt 0.0001) then  begin
			Anc_valid_indecis[fid[i]]=0
			print,Anc_valid_indecis[0:10]
		endif
	endfor
endfor
print,Anc_valid_indecis[0:10]
matched_indecis=where(Anc_valid_indecis,matched_cnt)
AnchorPnts_new[*,0:(matched_cnt-1)] = AnchorPnts[*,matched_indecis]
AnchorPnts = AnchorPnts_new
XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
widget_control,XY_Anc_Table_ID,set_value=(AnchorPnts), use_table_select=[0,0,5,(AnchPnts_MaxNum-1)]
if Z_ind ge 0 then begin
	ZPnts_new=dblarr(3,AnchPnts_MaxNum)
	ZPnts_new[*,0:(matched_cnt-1)]=ZPnts[*,matched_indecis]
	ZPnts=ZPnts_new
	Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
	widget_control,Z_Anc_Table_ID,set_value=(ZPnts), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]
end
ZeroPnts=dblarr(3,AnchPnts_MaxNum)
WID_Anchors_Transf_Error_Test_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Transf_Dist_Test')
widget_control,WID_Anchors_Transf_Error_Test_ID,set_value=(ZeroPnts), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]
ZeroAverPnts=dblarr(3)
WID_Anchors_Transf_AverageErrors_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Transf_AverageErrors')
widget_control,WID_Anchors_Transf_AverageErrors_ID,set_value=(ZeroAverPnts)
end
;
;-----------------------------------------------------------------
;
pro OnButton_Remove_Bad_Fiducials, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
fid=where(AnchorPnts[0,*] gt 0.0,n_peaks)
if n_peaks lt 1 then begin
	z=dialog_message('No red fiducials selected')
	return
endif
WID_TEXT_FidRemove_Thr_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_FidRemove_Thr')
widget_control,WID_TEXT_FidRemove_Thr_ID,GET_VALUE = Remove_Threshold
R_thr=float(Remove_Threshold[0])
Tr_err_max=2*R_thr
WID_Anchors_Transf_WorstErrors_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Transf_WorstErrors')

while (Tr_err_max gt R_thr) and (n_peaks gt 2) do begin
	TestFiducialTransformation, Event
	widget_control,WID_Anchors_Transf_WorstErrors_ID,get_value=Transform_Error_tot
	Tr_err_max=max(Transform_Error_tot)
	if Tr_err_max gt R_thr then OnButton_RemoveFiducial, Event
	fid=where(AnchorPnts[0,*] gt 0.0,n_peaks)
endwhile

end
;
;-----------------------------------------------------------------
;
pro DoInsert_Autodetect_Param, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
widget_control,event.id,get_value=thevalue
AutoDet_Params[event.y]=thevalue[event.x,event.y]
widget_control,event.id,set_value=transpose(AutoDet_Params), use_table_select=[0,0,0,2]
end
;
;-----------------------------------------------------------------
;
pro DoInsert_Autodetect_Matching_Param, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
widget_control,event.id,get_value=thevalue
AutoMatch_Params[event.y]=thevalue[event.x,event.y]
widget_control,event.id,set_value=transpose(AutoMatch_Params), use_table_select=[0,0,0,2]
end
;
;-----------------------------------------------------------------
;
pro TestFiducialTransformation, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	z=dialog_message(' TestFiducialTransformation:  '+!ERROR_STATE.msg)
	CATCH, /CANCEL
	return
ENDIF

WID_Anchors_Transf_Error_Test_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Transf_Dist_Test')
ZeroPnts=dblarr(3,AnchPnts_MaxNum)
widget_control,WID_Anchors_Transf_Error_Test_ID,set_value=(ZeroPnts), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]
WID_Anchors_Transf_AverageErrors_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Transf_AverageErrors')
ZeroAverPnts=dblarr(3)
widget_control,WID_Anchors_Transf_AverageErrors_ID,set_value=(ZeroAverPnts)

rtg_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_RTG')		;R=Data   G=Target
rtg=widget_info(rtg_button_id,/button_set)
rtb_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_RTB')		;R=Data   B=Target
rtb=widget_info(rtb_button_id,/button_set)
gtr_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_GTR')		;G=Data   R=Target
gtr=widget_info(gtr_button_id,/button_set)
gtb_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_GTB')		;G=Data   B=Target
gtb=widget_info(gtb_button_id,/button_set)
btr_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_BTR')		;B=Data   R=Target
btr=widget_info(btr_button_id,/button_set)
btg_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_BTG')		;B=Data   G=Target
btg=widget_info(btg_button_id,/button_set)

Transform_Error_tot=dblarr(3,AnchPnts_MaxNum)

if rtg then begin
	LabelToTransform=1
	LabelTarget=2
	TestFiducialTransformation_Single, Event, LabelToTransform, LabelTarget, Transform_Error
	cnt=n_elements(Transform_Error)
	widget_control,WID_Anchors_Transf_Error_Test_ID,set_value=transpose(Transform_Error), use_table_select=[0,0,0,(cnt-1)]
	widget_control,WID_Anchors_Transf_AverageErrors_ID,set_value=mean(Transform_Error), use_table_select=[0,0,0,0]
	Transform_Error_tot[0,0:(n_elements(Transform_Error)-1)]=Transform_Error
endif
if rtb then begin
	LabelToTransform=1
	LabelTarget=3
	TestFiducialTransformation_Single, Event, LabelToTransform, LabelTarget, Transform_Error
	cnt=n_elements(Transform_Error)
	widget_control,WID_Anchors_Transf_Error_Test_ID,set_value=transpose(Transform_Error), use_table_select=[2,0,2,(cnt-1)]
	widget_control,WID_Anchors_Transf_AverageErrors_ID,set_value=mean(Transform_Error), use_table_select=[2,0,2,0]
	Transform_Error_tot[2,0:(n_elements(Transform_Error)-1)]=Transform_Error
endif
if gtr then begin
	LabelToTransform=2
	LabelTarget=1
	TestFiducialTransformation_Single, Event, LabelToTransform, LabelTarget, Transform_Error
	cnt=n_elements(Transform_Error)
	widget_control,WID_Anchors_Transf_Error_Test_ID,set_value=transpose(Transform_Error), use_table_select=[0,0,0,(cnt-1)]
	widget_control,WID_Anchors_Transf_AverageErrors_ID,set_value=mean(Transform_Error), use_table_select=[0,0,0,0]
	Transform_Error_tot[0,0:(n_elements(Transform_Error)-1)]=Transform_Error
endif
if gtb then begin
	LabelToTransform=2
	LabelTarget=3
	TestFiducialTransformation_Single, Event, LabelToTransform, LabelTarget, Transform_Error
	cnt=n_elements(Transform_Error)
	widget_control,WID_Anchors_Transf_Error_Test_ID,set_value=transpose(Transform_Error), use_table_select=[1,0,1,(cnt-1)]
	widget_control,WID_Anchors_Transf_AverageErrors_ID,set_value=mean(Transform_Error), use_table_select=[1,0,1,0]
	Transform_Error_tot[1,0:(n_elements(Transform_Error)-1)]=Transform_Error
endif
if btr then begin
	LabelToTransform=3
	LabelTarget=1
	TestFiducialTransformation_Single, Event, LabelToTransform, LabelTarget, Transform_Error
	cnt=n_elements(Transform_Error)
	widget_control,WID_Anchors_Transf_Error_Test_ID,set_value=transpose(Transform_Error), use_table_select=[2,0,2,(cnt-1)]
	widget_control,WID_Anchors_Transf_AverageErrors_ID,set_value=mean(Transform_Error), use_table_select=[2,0,2,0]
	Transform_Error_tot[2,0:(n_elements(Transform_Error)-1)]=Transform_Error
endif
if btg then begin
	LabelToTransform=3
	LabelTarget=2
	TestFiducialTransformation_Single, Event, LabelToTransform, LabelTarget, Transform_Error
	cnt=n_elements(Transform_Error)
	widget_control,WID_Anchors_Transf_Error_Test_ID,set_value=transpose(Transform_Error), use_table_select=[1,0,1,(cnt-1)]
	widget_control,WID_Anchors_Transf_AverageErrors_ID,set_value=mean(Transform_Error), use_table_select=[1,0,1,0]
	Transform_Error_tot[1,0:(n_elements(Transform_Error)-1)]=Transform_Error
endif

Transform_Error_max=max(total(Transform_Error_tot,1),worst_ind)

WID_TEXT_FidRemoveNumber_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_FidRemoveNumber')
worst_ind_txt=string(worst_ind,FORMAT='(F5.0)')
widget_control,WID_TEXT_FidRemoveNumber_ID,SET_VALUE = worst_ind_txt

WID_Anchors_Transf_WorstErrors_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Transf_WorstErrors')
widget_control,WID_Anchors_Transf_WorstErrors_ID,set_value=Transform_Error_tot[*,worst_ind]

end

;
;-----------------------------------------------------------------
;
pro TestFiducialTransformation_Single, Event, LabelToTransform, LabelTarget, Transform_Error
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)

TopID=ids[min(where(names eq 'WID_BASE_AnchorPts'))]

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	z=dialog_message(' TestFiducialTransformation_Single:  '+!ERROR_STATE.msg)
	CATCH, /CANCEL
	return
ENDIF

Align_Z_TipTilt_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Align_Z_TipTilt')
Align_Z_TipTilt=widget_info(Align_Z_TipTilt_button_id,/button_set)
Align_Z_Shift_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Align_Z_Shift')
Align_Z_Shift=widget_info(Align_Z_Shift_button_id,/button_set)
Align_Z = Align_Z_TipTilt or Align_Z_Shift

DatFid=AnchorPnts[(2*(LabelToTransform-1)):(2*LabelToTransform-1),*]
DatZ=ZPnts[(LabelToTransform-1),*]
TargFid=AnchorPnts[(2*(LabelTarget-1)):(2*LabelTarget-1),*]
TargZ=ZPnts[(LabelTarget-1),*]

anc_ind=where(DatFid[0,*] ne 0,anc_cnt)

FiducialCoeff[LabelToTransform-1].present=1

XDat0=DatFid[0,0]
YDat0=DatFid[1,0]
XTarg0=TargFid[0,0]
YTarg0=TargFid[1,0]

cc=where(AnchorPnts[*,0] ne 0)
cc=where(AnchorPnts[cc[0],*] ne 0, fid0_cnt)		; number of non-zero fields in one of the columns
cc=where(AnchorPnts[*,*] ne 0, fid_tot_cnt)		; total number of non-zero fields

n_fid_lbl=fid_tot_cnt/fid0_cnt					; this ratio should be 4 (two labels) or 6 (3 labels)
if (n_fid_lbl ne 4) and (n_fid_lbl ne 6) then begin	; stop if the above is not true
	print, 'inconsisten fiducial number'
	print, 'number of non-zero fields in the first column=',fid0_cnt
	print, 'total number of non-zero fields=',fid_tot_cnt
	return
endif

Xo=reform(DatFid[0,anc_ind])
Yo=reform(DatFid[1,anc_ind])
Xi=reform(TargFid[0,anc_ind])
Yi=reform(TargFid[1,anc_ind])
Transform_Error=dblarr(anc_cnt)
if (n_elements(PW_deg) eq 0) then PW_deg=1

if Transf_Meth ne 4 then begin;   All but 3D affine transformation
	;--------------------------------------------------------
	; X-Y shift only
	if  (Transf_Meth eq 0) then begin
		if (fid0_cnt lt 1) then begin
			z=dialog_message('X-Y shift requires at least 1 pair of fiducials, you have only '+string(fid0_cnt))
				return
		endif
		print,'Testing X-Y shift only'
		dX = mean(Xi - Xo)
		dY = mean(Yi - Yo)
		;P=[[-1*dX,0],[1,0]]
		P=[[dX,0],[1,0]]
		;Q=[[-1*dY,1],[0,0]]
		Q=[[dY,1],[0,0]]
	endif

	;--------------------------------------------------------
	; linear regression for complex linear Fit:   Zi=M*Zo+N
	if  (Transf_Meth eq 1)  then begin
		if (fid0_cnt lt 3) then begin
			z=dialog_message('linear regression requires at least 3 pars of fiducials, you have only '+string(fid0_cnt))
			return
		endif
		print,'Testing linear regression for complex linear Fit:   Zi=M*Zo+N'
		XYo=complex(Xo,Yo)
		XYi=complex(Xi,Yi)
		print,'calculating ',n_elements(Xo),'   fiducial linear regression transformation'
		Zi=XYi
		Zo=XYo
		Complex_Linear_Regression, Zi, Zo, P,Q, Mag
		print,'Magnification= ', Mag
	endif

	;--------------------------------------------------------
	; POLYWARP
	if  (Transf_Meth eq 2) then begin
		if (fid0_cnt lt 3) then begin
			z=dialog_message('POLYWARP requires at least 3 pars of fiducials, you have only '+string(fid0_cnt))
			return
		endif
		print,'Testing POLYWARP transformation'
		print,'PW_deg=',pw_deg
		polywarp,Xi,Yi,Xo,Yo,PW_deg,P,Q				;Xi=sum(kxij#Xo^jYo^i)   Yi=sum(kyij#Xo^jYo^i)
	endif

	;--------------------------------------------------------
	; Affine 2D
	if Transf_Meth eq 3 then begin
		if (fid0_cnt lt 3) then begin
			z=dialog_message('Affine 2D transformation requires at least 3 pars of fiducials, you have only '+string(fid0_cnt))
			return
		endif
		print,'Testing 2D affine transformation'
		xin = transpose(DatFid[*,anc_ind])
		xpin = transpose(TargFid[*,anc_ind])
		AFFINE_SOLVE_GS, xin, xpin, R, c, P, Q, Err_mean
	endif

	;--------------------------------------------------------
	; calculate registration errors now
	print,'P: ',P
	print,'Q: ',Q
	X1=Xo*0.0
	Y1=X1
	pdeg = (size(p))[1]-1
	for xj=0, pdeg do begin
		for yj=0, pdeg do begin
			X1 = X1+P[yj,xj]*(Xo^xj)*(Yo^yj)
			Y1 = Y1+Q[yj,xj]*(Xo^xj)*(Yo^yj)
		endfor
	endfor
    Xerror=X1-Xi
    Yerror=Y1-Yi

	Transform_Error=sqrt(Xerror*Xerror+Yerror*Yerror)*nm_per_pixel
	transf_scl=sqrt(P[1,0]^2+P[0,1]^2)
	Adj_Scl_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Adj_Scl')
	Adj_Scl=widget_info(Adj_Scl_button_id,/button_set)
	if Adj_Scl and (transf_scl gt 0.00001) then Transform_Error=Transform_Error/transf_scl

	if (total(DatZ) ne 0) and (total(TargZ) ne 0) and Align_Z then begin		; if checked, and Z data exists do Z-alignement
		Zo=DatZ[anc_ind]
		Zi=TargZ[anc_ind]
		dZi=Zi-Zo
		if (fid0_cnt ge 3) and Align_Z_TipTilt then begin
			A=[[total(Xi*Xi),total(Xi*Yi),total(Xi)],[total(Xi*Yi),total(Yi*Yi),total(Yi)],[total(Xi),total(Yi),n_elements(Xi)]]
			B=[total(Xi*dZi),total(Yi*dZi),total(dZi)]
			LUDC, A, INDEX
			Plane_coeff = LUSOL(A, INDEX, B)
			Zdelta = Xo*Plane_coeff[0] + Yo*Plane_coeff[1] + Plane_coeff[2]
		endif else begin
			Zdelta = mean(dZi)
		endelse
		if UnwGrZ_ind gt 0 then begin
			WR=wind_range[LabelToTransform-1]
			Z1 = (Zo + Zdelta + 4.0 * WR) mod WR
		endif else Z1 = Zo + Zdelta
		Transform_Error = sqrt(Transform_Error*Transform_Error + (Z1-Zi)*(Z1-Zi))
	endif

;stop
endif else begin; in the case of affine 3D, convert everything to NM first
	;--------------------------------------------------------
	; Affine 3D
	if Transf_Meth eq 4 then begin
		if (fid0_cnt lt 4) then begin
			z=dialog_message('Affine 3D transformation requires at least 4 pars of fiducials, you have only '+string(fid0_cnt))
			return
		endif
		if (total(DatZ) eq 0) then begin
			z=dialog_message('Affine 3D transformation requires Z-fiducuials, none entered')
			return
		endif
		print,'Calculating 3D affine transformation'
		xin = [[transpose(DatFid[*,anc_ind])*nm_per_pixel],[DatZ[anc_ind]]]
		xpin = [[transpose(TargFid[*,anc_ind])*nm_per_pixel],[TargZ[anc_ind]]]
		AFFINE_SOLVE_GS, xin, xpin, R, c, P, Q, Err_mean
		print,'R: ',R
		print,'shift: ',c
		AFFINE_Transform_GS, xin, Out_arr, R, c
		Err = Out_arr - Xpin
		Transform_error = sqrt(total(Err*Err,2))
	endif
endelse

end
;
;-----------------------------------------------------------------
;
pro OnButton_RemoveFiducial, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
WID_TEXT_FidRemoveNumber_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_FidRemoveNumber')
widget_control,WID_TEXT_FidRemoveNumber_ID,GET_VALUE = Remove_Index

AnchorPnts1 = dblarr(6,AnchPnts_MaxNum)
ZPnts1 = dblarr(3,AnchPnts_MaxNum)

if Remove_Index ge 1 then begin
	AnchorPnts1[*,0:(Remove_Index-1)] = AnchorPnts[*,0:(Remove_Index-1)]
	ZPnts1[*,0:(Remove_Index-1)] = ZPnts[*,0:(Remove_Index-1)]
endif
if Remove_Index lt (AnchPnts_MaxNum-1) then begin
	AnchorPnts1[*,(Remove_Index):(AnchPnts_MaxNum-2)] = AnchorPnts[*,(Remove_Index+1):(AnchPnts_MaxNum-1)]
	ZPnts1[*,(Remove_Index):(AnchPnts_MaxNum-2)] = ZPnts[*,(Remove_Index+1):(AnchPnts_MaxNum-1)]
endif
AnchorPnts=AnchorPnts1
ZPnts=ZPnts1

XY_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_XY_Table')
widget_control,XY_Anc_Table_ID,set_value=(AnchorPnts), use_table_select=[0,0,5,(AnchPnts_MaxNum-1)]

Z_Anc_Table_ID=widget_info(event.top,FIND_BY_UNAME='WID_Anchors_Z_Table')
widget_control,Z_Anc_Table_ID,set_value=(ZPnts), use_table_select=[0,0,2,(AnchPnts_MaxNum-1)]

end
;
;-----------------------------------------------------------------
;
pro Set_AutoDisplay_Selected_Fiducials, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
	WID_BUTTON_AutoDisplay_Selected_Fiducials_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_AutoDisplay_Selected_Fiducials')
	AutoDisp_Sel_Fids=widget_info(WID_BUTTON_AutoDisplay_Selected_Fiducials_id,/button_set)
end
;
;-----------------------------------------------------------------
;
pro Set_Display_Fiducial_IDs, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
	Display_Fiducials_IDs_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Display_Fiducial_IDs')
	Disp_Fid_IDs=widget_info(Display_Fiducials_IDs_ID,/button_set)
end
;
;-----------------------------------------------------------------
;
pro Set_AdjustScale, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
	Adj_Scl_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Adj_Scl')
	Adj_Scl=widget_info(Adj_Scl_button_id,/button_set)
end
;
;-----------------------------------------------------------------
;
pro Set_LimitXY, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
	WID_BUTTON_XYlimits_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_XYlimits')
	Use_XYlimits=widget_info(WID_BUTTON_XYlimits_button_id,/button_set)
end
;
;-----------------------------------------------------------------
;
pro Set_LeaveOrigTotRaw, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
	WID_BUTTON_WID_BUTTON_LeaveOrigTotalRaw_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_LeaveOrigTotalRaw')
	LeaveOrigTotalRaw=widget_info(WID_BUTTON_WID_BUTTON_LeaveOrigTotalRaw_button_id,/button_set)
end
;
;-----------------------------------------------------------------
;
pro Set_Transf_Method, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
	WidD_Transf_Meth = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_TRANSFORM_METHOD')
	Transf_Meth = widget_info(WidD_Transf_Meth,/DropList_Select)	;0 - linear regression 1 - polywarp; 2 - pivot and average transform (in case of three fiducials only)
	print,'Set Transf_Meth=',Transf_Meth
end
;
;-----------------------------------------------------------------
;
pro Set_XY_limits, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
	WID_XYlimits_table_id=widget_info(event.top,FIND_BY_UNAME='WID_XYlimits_table')
	widget_control,WID_XYlimits_table_id,get_value=XYlimits
end
;
;-----------------------------------------------------------------
;
pro Set_PW_deg, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
	WID_SLIDER_POLYWARP_Degree_id=widget_info(event.top,FIND_BY_UNAME='WID_SLIDER_POLYWARP_Degree')
	widget_control,WID_SLIDER_POLYWARP_Degree_id,get_value=PW_deg
end
