;+
; NAME:
;	CUBEROOT
; PURPOSE:
;	Real roots of a cubic equation. Complex roots set to -1.0e30.
;	Called by HALFAGAUSS.
; CALLING SEQUENCE:
;	roots = CUBEROOT(cc)
; INPUT:
;	cc = 4 element vector, giving coefficients of cubic polynomial,
;		[c0,c1,c2,c3], where one seeks the roots of
;		c0 + c1*x + c2*x^2 + c3*x^3
; OUTPUT:
;	Function returns a vector of 3 roots. If only one root is real, then
;	that becomes the 1st element.
; EXAMPLE:
;	Find the roots of the equation
;		3.2 + 4.4*x -0.5*x^2 -x^3 = 0
;	IDL> x = cuberoot([3.2,4.4,-0.5,-1])
;
;	will return a 3 element vector with the real roots
;		-1.9228, 2.1846, -0.7618
; REVISION HISTORY:
;	Henry Freudenreich, 1995
;	GES, modified to allow for array of Z elements, 2009
;-
function cuberoot,Z,cc

on_error,2
unreal=-1.0e30

a1=cc[2]/cc[3]
a2=cc[1]/cc[3]
a3=(cc[0]-Z)/cc[3]

q=(a1^2-3.*a2)/9.
r=(2.*a1^3-9.*a1*a2+27.*a3)/54.

disc=r^2-q^3

X1=dblarr(n_elements(z))
X2=X1
X3=X1

neg_ind=where(disc lt 0,neg_cnt)
if neg_cnt ge 1 then begin
;  if r^2 lt q^3 then there are 3 real roots.
   theta=acos(r[neg_ind]/q^1.5)
   X1[neg_ind]=-2.*sqrt(q)*cos(theta/3.)-a1/3.
   X2[neg_ind]=-2.*sqrt(q)*cos((theta+6.28319)/3.)-a1/3.
   X3[neg_ind]=-2.*sqrt(q)*cos((theta-6.28319)/3.)-a1/3.
endif

pos_ind=where(disc ge 0,pos_cnt)
if pos_cnt ge 1 then begin
;  Get the one real root:
   a=-r[pos_ind]/abs(r[pos_ind]) * (abs(r[pos_ind])+sqrt(r[pos_ind]^2-q^3))^.33333
   ;if a eq 0. then b=0. else b=q/a
   zero_ind=where(a eq 0.,zero_cnt)
   if zero_cnt ge 1 then X1[pos_ind[zero_ind]]=a-a1/3.
   nonzero_ind=where(a ne 0.,nonzero_cnt)
   if nonzero_cnt ge 1 then X1[pos_ind[nonzero_ind]]=(a+q/a)-a1/3.
   X2[pos_ind]=unreal
   X3[pos_ind]=unreal
endif

roots=[transpose(reform(x1)),transpose(reform(x2)),transpose(reform(x3))]
return,roots
end

;
; ---------------------------------------------------------------------------
;
FUNCTION wind, x, a						;produces calibration wind data	from functional form
	y= (a[0]+a[1]*cos(a[2] + x*a[3])) * a[4]^2 /(a[4]^2+(x-a[5])^2)
	y1 = a[4]^2 /(a[4]^2+(x-a[5])^2)
	y2 = cos(a[2] + x*a[3]) * a[4]^2 /(a[4]^2+(x-a[5])^2)
	y3 = (-1.)*a[1]*sin(a[2] + x*a[3]) * a[4]^2 /(a[4]^2+(x-a[5])^2)
	y4 = (-1.)*a[1]*x*sin(a[2] + x*a[3]) * a[4]^2 /(a[4]^2+(x-a[5])^2)
	y5 = (a[0]+a[1]*cos(a[2] + x*a[3]))* 2* a[4] * (x-a[5])^2 / (a[4]^2+(x-a[5])^2)^2
	y6 = (a[0]+a[1]*cos(a[2] + x*a[3]))* a[4]^2 * (-2.0) * (a[5]-x) / (a[4]^2+(x-a[5])^2)^2
return, [y,y1,y2,y3,y4,y5,y6]
end
;
FUNCTION ElliptVsZ, X, A
	y=A[0]+A[1]*X+A[2]*X^3
	y1=1
	y2=X
	y3=X^3
	return, [y,y1,y2,y3]
end
;
FUNCTION Z_ell, X, A
	return, (A[0]+A[1]*X+A[2]*X^3)
end
;
FUNCTION Ell_Z, Z, A
	Z=reform(Z)
	coeff=dblarr(4)
	coeff[0:1]=A[0:1]
	coeff[3]=A[2]
	roots=cuberoot(Z,coeff)
	linear_root=(Z-A[0])/A[1]
	if n_elements(Z) eq 1 then trash=min(abs(roots-linear_root),root_ind)	$
	else trash=min(abs(roots-[1.0,1.0,1.0]#linear_root),root_ind,dimension=1)
	return,roots[root_ind]
end
;
;----------------------------------------------------------------------------
;
;
; Empty stub procedure used for autoloading.
;
pro ZoperationsWid_eventcb
end
;
; ---------------------------------------------------------------------------
;
Pro DoCalib,zz,lambda,dd,aa				;fits to
	;	y= (a[0]+a[1]*cos(a[2] + x*a[3])) *(1.-a[4]*(x/200.-a[5])^2)
a=fltarr(6)
aa=fltarr(6,3)
for i=0,2 do begin
	d=dd[*,i]
	a[0] = mean(d)
	a[1] = (max(d)-min(d))/2.0
	mx = where(d eq max(d))
	a[3] = 4.0*!pi/lambda
	a[2] = (10.0 - zz[mx]*2.0/lambda)
	a[2] = 2*!pi * (a[2]-fix(a[2]))
	a[4] = 500.0
	a[5] = 0.000
	plot,zz,d,psym=2
	res=wind(zz,a)
	oplot,zz,res[0,*]

	fita=[0,0,1,1,0,0]
	result=lmfit(zz,d,a,fita=fita,FUNCTION_NAME='wind',/DOUBLE)
	fita=[1,1,1,1,0,0]
	result=lmfit(zz,d,a,fita=fita,FUNCTION_NAME='wind',/DOUBLE)
	fita=[1,1,1,1,1,1]
	result=lmfit(zz,d,a,fita=fita,FUNCTION_NAME='wind',/DOUBLE)

	oplot,zz,result,col=190
	wait,2.0
	print,'        offset          amplitude          phase(deg)	  frequency          envelope slope      offset'
	print,a[0],a[1],a[2]*180/!pi,a[3],a[4],a[5]
	aa[*,i]=a
endfor

for i=0,2 do begin

	a=aa[*,i]
	fita=[1,1,1,0,0,0]
	result=lmfit(zz,d,a,fita=fita,FUNCTION_NAME='wind',/DOUBLE)
	aa[*,i]=a
endfor
aa[3,*]=mean(aa[3,*])
aa[4,*]=mean(aa[4,where((aa[4,*] ge -500000) and (aa[4,*] le 500000))])
aa[5,*]=mean(aa[5,where((aa[5,*] ge -500000) and (aa[5,*] le 500000))])

print,'re-fitting with the same foollowing parameters (for 3 cameras)
print,'frequency        envelope slope      offset'
print,aa[3],aa[4],aa[5]

for i=0,2 do begin
	d=dd[*,i]
	a=aa[*,i]
	fita=[1,1,1,0,0,0]
	result=lmfit(zz,d,a,fita=fita,FUNCTION_NAME='wind',/DOUBLE)
	aa[*,i]=a
endfor
return
end
;
; ---------------------------------------------------------------------------
;
Pro ExtractWindCalib, Event, zz, dd, BKGRND		; perform z-calibration on filtered data set
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w

;WlWidID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_WL')
;wl_id=widget_info(WlWidID,/DROPLIST_SELECT)
;widget_control,WlWidID,GET_VALUE = wl_strings
;lambda_vac=float(strtrim(wl_strings[wl_id]))

WlWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WL')
widget_control,WlWidID,GET_VALUE = wl_txt
lambda_vac=float(wl_txt[0])
lambda=lambda_vac/nd_oil

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

FilterIt

subsetindex=where(filter eq 1,cnt)
print, 'subset has ',cnt,' points'
if cnt le 0 then return
;NFrames=long64(max(CGroupParams[9,*]))
NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[9,*])+1))	: long64(max(CGroupParams[9,*])+1)

subset=CGroupParams[*,subsetindex]
dd=[CGroupParams[27,subsetindex],CGroupParams[28,subsetindex],CGroupParams[29,subsetindex]]			;the three amplitudes
dd=transpose(dd)
ellipticity=CGroupParams[43,subsetindex]
zz=(CGroupParams[9,subsetindex]-NFrames/2.)*nmperframe ;* nd_oil/nd_water    ; rescale here to take into account that calibration is done by varying the pathlengths through oil, and in sample the variable pathlengts through water
DoCalib,zz,lambda,dd,aa
FR=findgen(NFrames)
y0= (aa[0,0]+aa[1,0]*cos(aa[2,0] + zz*aa[3,0]))*aa[4,0]^2/(aa[4,0]^2+(zz-aa[5,0])^2)
y1= (aa[0,1]+aa[1,1]*cos(aa[2,1] + zz*aa[3,1]))*aa[4,1]^2/(aa[4,1]^2+(zz-aa[5,1])^2)
y2= (aa[0,2]+aa[1,2]*cos(aa[2,2] + zz*aa[3,2]))*aa[4,2]^2/(aa[4,2]^2+(zz-aa[5,2])^2)
!p.multi=[0,1,4,0,0]
lbl=['offset','amplitude','phase(deg)','freq']
lbz_unwrp=['offset','slope','sqr']
;wind_range=2*!pi/mean(aa[3,*])
wind_range=lambda_vac/2.3348	; take into accoount factor of 2 (opposing objectives), reafractive index of ~1.37, and Gouy effect -> wind_range=lambda/2/Nr*1.175)
aa[3,*]=2.0*!pi/wind_range		; force the wind range and calibration period to be related on emission wavelength as described above

ReadWindPoint, Event
aa[2,*]=aa[2,*]/!dtor

z_unwrap_coeff=[0,700,700]
fita=[1,1,1]
if total(ellipticity) ne 0 then begin
	zz_fit = lmfit(ellipticity,zz,z_unwrap_coeff,fita=fita,FUNCTION_NAME='ElliptVsZ',/DOUBLE)
	;Analyze_cal_files_create_lookup_templates,5
endif else begin
	z_unwrap_coeff = [0, 0, 0]
	zz_fit = zz
endelse
Update_Table_EllipticityFitCoeff, Event

wset,def_w

if BKGRND eq 'White' then begin
	!p.background=255
	!P.Font=1
	DEVICE, SET_FONT='Helvetica Bold', /TT_FONT
	SET_CHARACTER_SIZE=[50,55]
	col=0
	plot,zz,y0,xtitle='nm',ytitle='Amp Channel 1',yrange=Paramlimits[27,0:1],$
			ystyle=1, col=col,thick=2,charthick=2,charsize=2,xthick=2,ythick=2
	oplot,zz,subset[27,*],psym=6,col=col,symsize=0.16
	for i =0,3 do xyouts,0.8,0.96-0.02*i,aa[i,0],/normal
	for i =0,3 do xyouts,0.9,0.96-0.02*i,lbl[i],/normal

	plot,zz,y1,xtitle='nm',ytitle='Amp Channel 2',yrange=Paramlimits[28,0:1],$
			ystyle=1, col=col,thick=2,charthick=2,charsize=2,xthick=2,ythick=2
	oplot,zz,subset[28,*],psym=6,col=col,symsize=0.16
	for i =0,3 do xyouts,0.8,0.70-0.02*i,aa[i,1],/normal
	for i =0,3 do xyouts,0.9,0.70-0.02*i,lbl[i],/normal

	plot,zz,y2,xtitle='nm',ytitle='Amp Channel 3',yrange=Paramlimits[29,0:1],$
			ystyle=1, col=col,thick=2,charthick=2,charsize=2,xthick=2,ythick=2
	oplot,zz,subset[29,*],psym=6,col=col,symsize=0.16
	for i =0,3 do xyouts,0.8,0.46-0.02*i,aa[i,2],/normal
	for i =0,3 do xyouts,0.9,0.46-0.02*i,lbl[i],/normal

	plot,zz,ellipticity,xtitle='nm',ytitle='Ellipticity',ystyle=1,$
			thick=2,charthick=2,charsize=2,xthick=2,ythick=2,psym=6,col=col,symsize=0.16
	oplot,zz_fit,ellipticity,col=col
	for i =0,2 do xyouts,0.8,0.10-0.02*i,z_unwrap_coeff[i],/normal
	for i =0,2 do xyouts,0.92,0.10-0.02*i,lbz_unwrp[i],/normal
endif
if BKGRND ne 'White' then begin
	plot,zz,y0,xtitle='nm',ytitle='Amp Channel 1',yrange=Paramlimits[27,0:1],ystyle=1
	oplot,zz,subset[27,*],psym=3
	for i =0,3 do xyouts,0.8,0.96-0.02*i,aa[i,0],/normal
	for i =0,3 do xyouts,0.9,0.96-0.02*i,lbl[i],/normal
	plot,zz,y1,xtitle='nm',ytitle='Amp Channel 2',yrange=Paramlimits[28,0:1],ystyle=1
	oplot,zz,subset[28,*],psym=3
	for i =0,3 do xyouts,0.8,0.70-0.02*i,aa[i,1],/normal
	for i =0,3 do xyouts,0.9,0.70-0.02*i,lbl[i],/normal
	plot,zz,y2,xtitle='nm',ytitle='Amp Channel 3',yrange=Paramlimits[29,0:1],ystyle=1
	oplot,zz,subset[29,*],psym=3
	for i =0,3 do xyouts,0.8,0.46-0.02*i,aa[i,2],/normal
	for i =0,3 do xyouts,0.9,0.46-0.02*i,lbl[i],/normal
	plot,zz,ellipticity,xtitle='nm',ytitle='Ellipticity',ystyle=1,psym=6
	oplot,zz_fit,ellipticity
	for i =0,(n_elements(z_unwrap_coeff)-1) do xyouts,0.8,0.10-0.02*i,z_unwrap_coeff[i],/normal
	for i =0,(n_elements(z_unwrap_coeff)-1) do xyouts,0.92,0.10-0.02*i,lbz_unwrp[i],/normal
endif

aa[2,*]=aa[2,*]*!dtor
!p.multi=[0,0,0,0,0]
!p.background=0

wind_range=2*!pi/mean(aa[3,*])
WindRangeID = Widget_Info(event.top, find_by_uname='WID_TEXT_WindPeriod')
wind_range_txt=string(wind_range,FORMAT='(F6.2)')
widget_control,WindRangeID,SET_VALUE = wind_range_txt

return
end
;
;-----------------------------------------------------------------
;
Pro ExtractEllipticityCalib, Event, zz, BKGRND		; perform z-calibration on filtered data set
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common InfoFit, pth, filen, thisfitcond, saved_pks_filenam

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
lbz_unwrp=['offset','slope','sqr']

FilterIt

subsetindex=where(filter eq 1,cnt)
print, 'subset has ',cnt,' points'
if cnt le 0 then return
;NFrames=long64(max(CGroupParams[9,*]))
NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[9,*])+1))	: long64(max(CGroupParams[9,*])+1)
ellipticity=transpose(CGroupParams[43,subsetindex])
;zz=transpose((CGroupParams[9,subsetindex]-NFrames/2.)*nmperframe)
zz=(CGroupParams[9,subsetindex]-NFrames/2.)*nmperframe * nd_oil/nd_water

z_unwrap_coeff= [0,700,700]
fita=[1,1,1]
if total(ellipticity) ne 0 then zz_fit = lmfit(ellipticity,zz,z_unwrap_coeff,fita=fita,FUNCTION_NAME='ElliptVsZ',/DOUBLE) else begin
	z_unwrap_coeff = [0, 0, 0]
	zz_fit = zz
endelse
Update_Table_EllipticityFitCoeff, Event

if BKGRND eq 'White' then begin
	!p.background=255
	!P.Font=1
	DEVICE, SET_FONT='Helvetica Bold', /TT_FONT
	SET_CHARACTER_SIZE=[50,55]
	col=0
	plot,zz,ellipticity,xtitle='nm',ytitle='Ellipticity',ystyle=1,$
			thick=2,charthick=2,charsize=2,xthick=2,ythick=2,psym=6,col=col,symsize=0.16
	oplot,zz_fit,ellipticity,col=col
	for i =0,2 do xyouts,0.8,0.10-0.02*i,z_unwrap_coeff[i],/normal
	for i =0,2 do xyouts,0.92,0.10-0.02*i,lbz_unwrp[i],/normal
endif
if BKGRND ne 'White' then begin
	plot,zz,ellipticity,xtitle='nm',ytitle='Ellipticity',ystyle=1,psym=6
	oplot,zz_fit,ellipticity
	for i =0,(n_elements(z_unwrap_coeff)-1) do xyouts,0.8,0.10-0.02*i,z_unwrap_coeff[i],/normal
	for i =0,(n_elements(z_unwrap_coeff)-1) do xyouts,0.92,0.10-0.02*i,lbz_unwrp[i],/normal
endif

!p.background=0

;Analyze_cal_files_create_lookup_templates,5

return
end
;
;-----------------------------------------------------------------
;
pro OnTestWindPoint, Event		; Tests Z-calibration fit w/o saving the fit parameters
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
BKGRND= 'Black'
ExtractWindCalib, Event, zz, dd, BKGRND
return
end
;
;-----------------------------------------------------------------
pro OnTestEllipOnly, Event
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
BKGRND= 'Black
ExtractEllipticityCalib, Event, zz, BKGRND
end
;
;-----------------------------------------------------------------
;
pro OnWriteCalibWind, Event	; Tests Z-calibration fit and saves the fit parameters into selected WND file.
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
BKGRND= 'Black'
WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WindFilename')
widget_control,WFileWidID,GET_VALUE = wfilename
if wfilename eq '' then return
wfilename_ext=AddExtension(wfilename,'_WND.sav')
if wfilename_ext ne wfilename then begin
	widget_control,WFileWidID,SET_VALUE = wfilename_ext
	wfilename  = wfilename_ext
endif
ExtractWindCalib, Event, zz, dd, BKGRND
save, aa, z_unwrap_coeff, lambda_vac, nd_water, nd_oil, wind_range, cal_lookup_data, cal_lookup_zz, nmperframe, filename=wfilename_ext
end
;
;-----------------------------------------------------------------
;
pro OnPlotEllipticityDataAndFit, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common InfoFit, pth, filen, thisfitcond, saved_pks_filenam

nmperframe = 8.0			; nm per frame. calibration using piezo parameters (2um per 10V, 0.04V  per frame)

lbz_unwrp=['offset','slope','sqr']
subsetindex=where(filter eq 1,cnt)
print, 'subset has ',cnt,' points'
if cnt le 0 then return
;NFrames=long64(max(CGroupParams[9,*]))
NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[9,*])+1))	: long64(max(CGroupParams[9,*])+1)

ellipticity=transpose(CGroupParams[43,subsetindex])
;zz=transpose((CGroupParams[9,subsetindex]-NFrames/2.)*nmperframe)
zz=(CGroupParams[9,subsetindex]-NFrames/2.)*nmperframe * nd_oil/nd_water
zz_fit=ElliptVsZ(ellipticity,z_unwrap_coeff)

plot,zz,ellipticity,xtitle='nm',ytitle='Ellipticity',ystyle=1,psym=6
oplot,zz_fit,ellipticity
for i =0,(n_elements(z_unwrap_coeff)-1) do xyouts,0.8,0.10-0.02*i,z_unwrap_coeff[i],/normal
for i =0,(n_elements(z_unwrap_coeff)-1) do xyouts,0.92,0.10-0.02*i,lbz_unwrp[i],/normal

!p.background=0
end
;
;-----------------------------------------------------------------
;
pro OnSaveEllipAndWind, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
BKGRND= 'Black'
WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WindFilename')
widget_control,WFileWidID,GET_VALUE = wfilename
if wfilename eq '' then return
wfilename_ext=AddExtension(wfilename,'_WND.sav')
if wfilename_ext ne wfilename then widget_control,WFileWidID,SET_VALUE = wfilename_ext
save, aa, z_unwrap_coeff, lambda_vac, nd_water, nd_oil, wind_range, filename=wfilename_ext
end
;
;-----------------------------------------------------------------
pro Edit_Ellipticity_Coeff, Event
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
EllipFitCoeff_WidID = Widget_Info(Event.top, find_by_uname='WID_TABLE_EllipticityFitCoeff')
widget_control,event.id,get_value=thevalue
z_unwrap_coeff[event.x] = thevalue[event.x,event.y]
widget_control,EllipFitCoeff_WidID,set_value=z_unwrap_coeff, use_table_select=[0,0,(n_elements(z_unwrap_coeff)-1),0]
end
;
;-----------------------------------------------------------------
;
pro Edit_Ellipticity_Correction_Slope, Event
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
EllipCorrSlope_WidID = Widget_Info(Event.top, find_by_uname='WID_TABLE_EllipticityCorrectionSlope')
widget_control,event.id,get_value=thevalue
ellipticity_slopes[event.x] = thevalue[event.x,event.y]
widget_control,EllipCorrSlope_WidID,set_value=ellipticity_slopes, use_table_select=[0,0,(n_elements(ellipticity_slopes)-1),0]
end
;
;-----------------------------------------------------------------
;
pro OnWriteCalibASCII, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common InfoFit, pth, filen, thisfitcond, saved_pks_filenam

BKGRND= 'Black'
nmperframe = 8.0			; nm per frame. calibration using piezo parameters (2um per 10V, 0.04V  per frame)

;WlWidID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_WL')
;wl_id=widget_info(WlWidID,/DROPLIST_SELECT)
;widget_control,WlWidID,GET_VALUE = wl_strings
;lambda_vac=float(strtrim(wl_strings[wl_id]))

WlWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WL')
widget_control,WlWidID,GET_VALUE = wl_txt
lambda_vac=float(wl_txt[0])
lambda=lambda_vac/nd_oil

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

FilterIt
subsetindex=where(filter eq 1,cnt)
print, 'subset has ',cnt,' points'
if cnt le 0 then return
;NFrames=long64(max(CGroupParams[9,*]))
NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[9,*])+1))	: long64(max(CGroupParams[9,*])+1)
subset=CGroupParams[*,subsetindex]
ellipticity=CGroupParams[43,subsetindex]
dd=[CGroupParams[27,subsetindex],CGroupParams[28,subsetindex],CGroupParams[29,subsetindex]]			;the three amplitudes
dd=transpose(dd)
;zz=(CGroupParams[9,subsetindex]-NFrames/2.)*nmperframe
zz=(CGroupParams[9,subsetindex]-NFrames/2.)*nmperframe * nd_oil/nd_water
DoCalib,zz,lambda,dd,aa
FR=findgen(NFrames)
y0= (aa[0,0]+aa[1,0]*cos(aa[2,0] + zz*aa[3,0]))*aa[4,0]^2/(aa[4,0]^2+(zz-aa[5,0])^2)
y1= (aa[0,1]+aa[1,1]*cos(aa[2,1] + zz*aa[3,1]))*aa[4,1]^2/(aa[4,1]^2+(zz-aa[5,1])^2)
y2= (aa[0,2]+aa[1,2]*cos(aa[2,2] + zz*aa[3,2]))*aa[4,2]^2/(aa[4,2]^2+(zz-aa[5,2])^2)
!p.multi=[0,1,4,0,0]
lbl=['offset','amplitude','phase(deg)','freq']
lbz_unwrp=['offset','slope','sqr']
wind_range=2*!pi/mean(aa[3,*])
ReadWindPoint, Event
aa[2,*]=aa[2,*]/!dtor

z_unwrap_coeff=[0,700,700]
fita=[1,1,1]
if total(ellipticity) ne 0 then zz_fit = lmfit(ellipticity,zz,z_unwrap_coeff,fita=fita,FUNCTION_NAME='ElliptVsZ',/DOUBLE) else begin
	z_unwrap_coeff = [0, 0, 0]
	zz_fit = zz
endelse
Update_Table_EllipticityFitCoeff, Event

if BKGRND eq 'White' then begin
	!p.background=255
	!P.Font=1
	DEVICE, SET_FONT='Helvetica Bold', /TT_FONT
	SET_CHARACTER_SIZE=[50,55]
	col=0
	plot,zz,y0,xtitle='nm',ytitle='Amp Channel 1',yrange=Paramlimits[27,0:1],$
			ystyle=1, col=col,thick=2,charthick=2,charsize=2,xthick=2,ythick=2
	oplot,zz,subset[27,*],psym=6,col=col,symsize=0.16
	for i =0,3 do xyouts,0.8,0.96-0.02*i,aa[i,0],/normal
	for i =0,3 do xyouts,0.9,0.96-0.02*i,lbl[i],/normal

	plot,zz,y1,xtitle='nm',ytitle='Amp Channel 2',yrange=Paramlimits[28,0:1],$
			ystyle=1, col=col,thick=2,charthick=2,charsize=2,xthick=2,ythick=2
	oplot,zz,subset[28,*],psym=6,col=col,symsize=0.16
	for i =0,3 do xyouts,0.8,0.70-0.02*i,aa[i,1],/normal
	for i =0,3 do xyouts,0.9,0.70-0.02*i,lbl[i],/normal

	plot,zz,y2,xtitle='nm',ytitle='Amp Channel 3',yrange=Paramlimits[29,0:1],$
			ystyle=1, col=col,thick=2,charthick=2,charsize=2,xthick=2,ythick=2
	oplot,zz,subset[29,*],psym=6,col=col,symsize=0.16
	for i =0,3 do xyouts,0.8,0.46-0.02*i,aa[i,2],/normal
	for i =0,3 do xyouts,0.9,0.46-0.02*i,lbl[i],/normal

	plot,zz_fit,ellipticity,xtitle='nm',ytitle='Ellipticity',ystyle=1,$
			col=col,thick=2,charthick=2,charsize=2,xthick=2,ythick=2
	oplot,zz,ellipticity,psym=6,col=col,symsize=0.16
	for i =0,2 do xyouts,0.8,0.10-0.02*i,z_unwrap_coeff[i],/normal
	for i =0,2 do xyouts,0.92,0.10-0.02*i,lbz_unwrp[i],/normal
endif
if BKGRND ne 'White' then begin
	plot,zz,y0,xtitle='nm',ytitle='Amp Channel 1',yrange=Paramlimits[27,0:1]/3,ystyle=1
	oplot,zz,subset[27,*],psym=3
	for i =0,3 do xyouts,0.8,0.96-0.02*i,aa[i,0],/normal
	for i =0,3 do xyouts,0.9,0.96-0.02*i,lbl[i],/normal
	plot,zz,y1,xtitle='nm',ytitle='Amp Channel 2',yrange=Paramlimits[28,0:1]/3,ystyle=1
	oplot,zz,subset[28,*],psym=3
	for i =0,3 do xyouts,0.8,0.70-0.02*i,aa[i,1],/normal
	for i =0,3 do xyouts,0.9,0.70-0.02*i,lbl[i],/normal
	plot,zz,y2,xtitle='nm',ytitle='Amp Channel 3',yrange=Paramlimits[29,0:1]/3,ystyle=1
	oplot,zz,subset[29,*],psym=3
	for i =0,3 do xyouts,0.8,0.46-0.02*i,aa[i,2],/normal
	for i =0,3 do xyouts,0.9,0.46-0.02*i,lbl[i],/normal
	plot,zz_fit,ellipticity,xtitle='nm',ytitle='Ellipticity',ystyle=1
	oplot,zz,ellipticity,psym=6
	for i =0,2 do xyouts,0.8,0.10-0.02*i,z_unwrap_coeff[i],/normal
	for i =0,2 do xyouts,0.92,0.10-0.02*i,lbz_unwrp[i],/normal
endif
aa[2,*]=aa[2,*]*!dtor
!p.multi=[0,0,0,0,0]
!p.background=0

WASCIIFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WindFilename_ASCII')
widget_control,WASCIIFileWidID,GET_VALUE = ASCII_filename
data=[zz,subset[27,*],y0,subset[28,*],y1,subset[29,*],y2]
openw,1,ASCII_filename,width=512
printf,1,'Z-position (nm)	Cam1 Data	Cam1 Fit	Cam2 Data	Cam2 Fit	Cam3 Data	Cam3 Fit'
printf,1,data,FORMAT='(E12.4,"	",E12.4,"	",E12.4,"	",E12.4,"	",E12.4,"	",E12.4,"	",E12.4)'
close,1
end
;
; ---------------------------------------------------------------------------
;
FUNCTION amptriplet, zeta				;for Newton's method find zeta values
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
y0 = aa[0,0]*zeta[0]*(1.0 +aa[1,0]/aa[0,0]*zeta[1]*cos(aa[2,0] + zeta[2]*aa[3,0])) - d[0]
y1 = aa[0,1]*zeta[0]*(1.0 +aa[1,1]/aa[0,1]*zeta[1]*cos(aa[2,1] + zeta[2]*aa[3,1])) - d[1]
y2 = aa[0,2]*zeta[0]*(1.0 +aa[1,2]/aa[0,2]*zeta[1]*cos(aa[2,2] + zeta[2]*aa[3,2])) - d[2]
return,[y0,y1,y2]
end
;
;-----------------------------------------------------------------
;
pro OnExtractZCoord, Event		; Extracts Z-coordinates using the fit parameters from selected WND file.
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

if wfilename eq '' then return

if ((size(aa))[2] eq 0) then begin
	print,'Invalid WND file'
	return
endif
sz=size(CGroupParams)

OnExtractZCoord_Core, CGroupParams, 2, 1	; extract both peaks and groups, display reports

for j=31,41 do ParamLimits[j,0]= min(CGroupParams[j,*])
for j=31,41 do ParamLimits[j,1] = max(CGroupParams[j,*])
for j=31,41 do ParamLimits[j,3]=ParamLimits[j,1] - ParamLimits[j,0]
for j=31,41 do ParamLimits[j,2]=(ParamLimits[j,1] + ParamLimits[j,0])/2.

if CGrpSize ge 49 then begin
	for j=43,48 do begin
		ParamLimits[j,0] = min(CGroupParams[j,*])
		ParamLimits[j,1] = max(CGroupParams[j,*])
		ParamLimits[j,3]=ParamLimits[j,1] - ParamLimits[j,0]
		ParamLimits[j,2]=(ParamLimits[j,1] + ParamLimits[j,0])/2.
	endfor
endif

LimitUnwrapZ

TopIndex = (CGrpSize-1) > 41
topid_ind=where(names eq 'WID_BASE_0_PeakSelector')
TopID=ids[topid_ind[0]]
wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(ParamLimits[0:TopIndex,0:3]), use_table_select=[0,0,3,TopIndex]
widget_control,wtable,set_value=transpose(CGroupParams[0:TopIndex,sz[2]/2]), use_table_select=[4,0,4,TopIndex]
widget_control, wtable, /editable,/sensitive
end
;
;-----------------------------------------------------------------
;
pro OnExtractZCoord_Core, CGroupParams, index, disp	; Extracts Z-coordinates using the fit parameters from selected WND file.
													; index = 0 - perform on Peaks Only
													; index = 1 - perform on Groups Only
													; index = 2 - perform on both
													; disp = 0 - no progress report messages
													; disp = 1 - print progress report messages
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius

sz=size(CGroupParams)
CGrpSize=sz[1]
sz_disp=ceil(sz[2]/50)>1
aa2original=aa[2,*]
av3=total(aa[3,*])/3
wind_range=2*!pi/mean(aa[3,*])

if (index eq 0) or (index eq 2) then begin
	for i = 0l,(sz[2]-1) do begin
		error_count=0
		aa[2,*]=aa2original
		d=CGroupParams[27:29,i]						; Now extract z position and amp, coherence from a peak amp triplet
		zeta0=[0.2,1.0,50.0]						; initial guess
		zeta = zeta0
		catch,error_status
		if error_status NE 0 then begin
			error_count+=1
			zeta0 = zeta
			print,error_count,i,zeta
		endif
		if error_count le 4 then begin
			zeta=NEWTON(zeta0,'amptriplet',/DOUBLE)	; find zeta (amplitude - coherence - z value)
			zeta[2] = zeta[2]+(zeta[1] le 0)*!pi/av3
			zeta[1] = abs(zeta[1])
			zeta[2] = zeta[2] - round(zeta[2]*av3/(2.*!pi))*(2.*!pi/av3)
			zeta[2] = zeta[2] + (zeta[2] lt 0)*2.*!pi/av3
				CGroupParams[34,i]=zeta[2]
			if (total(z_unwrap_coeff[*]) ne 0) and (CGrpSize ge 49) then begin 	; extract unwrapped Z position (from known dependence of ellipticity on Z-position)
				Z_ellip = Z_ell(CGroupParams[43,i],z_unwrap_coeff)
				CGroupParams[44,i]= CGroupParams[34,i] + round((Z_ellip - zeta[2])/wind_range) * wind_range
				CGroupParams[45,i]= CGroupParams[44,i] - Z_ellip
			endif else if CGrpSize ge 49 then CGroupParams[44,i] = CGroupParams[34,i]
			CGroupParams[35,i]=6.*40./sqrt(CGroupParams[6,i])/((zeta[1]>0.05)<2.)					;sigma z = 6 * 40/sqrt(N)/coherence(zeta[1])
			CGroupParams[36,i]=zeta[1]																;coherence
			CGroupParams[31,i]=zeta[0]
			if (i mod sz_disp eq 0) and disp then print,'peak#',i,'   z-data:  ',zeta[0],zeta[1],zeta[2]		;amp norm to fiducial, coher norm to fiducial, zposition (nm)
		endif else begin
			CGroupParams[31,i]= 0.0
			CGroupParams[34:36,i]=0.
			CGroupParams[44:45,i]=0.
			CGroupParams[30,i]=-1.
			print,'peak#',i,'   could not be extracted,    Error Code  ',error_status, '   FitOK=-1 assigned'
		endelse
	endfor
	catch,/cancel
	aa[2,*] = aa2original
endif

indexgroup=(index eq 1) or (index eq 2)
if (mean(CGroupParams[38,*]) ne 0) and indexgroup then begin
	for i = 0l,(sz[2]-1) do begin
		if CGroupParams[24,i] gt 1 then begin
			aa[2,*]=aa2original
			error_count=0
			d=CGroupParams[37:39,i]						; Now extract z position and amp, coherence from a peak amp triplet
			zeta0=[0.2,1.0,100.0]						; initial guess
			zeta = zeta0
			catch,error_status
			if error_status NE 0 then begin
				zeta0 = zeta
				error_count+=1
				print,i,zeta
			endif
			if error_count le 2 then begin
				zeta=NEWTON(zeta0,'amptriplet',/DOUBLE)	; find zeta (amplitude - coherence - z value)
				zeta[2] = zeta[2]+(zeta[1] le 0)*!pi/av3
				zeta[1] = abs(zeta[1])
				zeta[2] = zeta[2] - round(zeta[2]*av3/(2.*!pi))*(2.*!pi/av3)
				zeta[2] = zeta[2] + (zeta[2] lt 0)*2.*!pi/av3
				CGroupParams[40,i]=zeta[2]
				CGroupParams[41,i]=6.*40./sqrt(CGroupParams[23,i])/((zeta[1]>0.05)<2.)			;sigma z = 6 * 40/sqrt(N)/coherence(zeta[1])
				CGroupParams[42,i]=zeta[1]														;coherence
				if (total(z_unwrap_coeff[*]) ne 0) and (CGrpSize ge 49) then begin 	; extract unwrapped Z position (from known dependence of ellipticity on Z-position)
					Z_ellip = Z_ell(CGroupParams[46,i],z_unwrap_coeff)
					CGroupParams[47,i] = CGroupParams[40,i] + round((Z_ellip - zeta[2])/wind_range) * wind_range; * nd_oil/nd_water
					CGroupParams[48,i] = CGroupParams[47,i] - Z_ellip
				endif else if CGrpSize ge 49 then CGroupParams[47,i] = CGroupParams[40,i]
			endif else begin
				CGroupParams[40:42,i]=0.
				CGroupParams[47:48,i]=0.
				CGroupParams[30,i]=-2.
				print,'gr.peak#',i,'   could not be extracted,    Error Code  ',error_status, '   FitOK=-2 assigned'
			endelse

		endif else begin
			CGroupParams[40,i]=CGroupParams[34,i]
			CGroupParams[41,i]=CGroupParams[35,i]
			CGroupParams[42,i]=CGroupParams[36,i]
			if (z_unwrap_coeff[0] ne 0) and (CGrpSize ge 49) then begin
				CGroupParams[47,i] = CGroupParams[44,i]
				CGroupParams[48,i] = CGroupParams[45,i]
			endif
		endelse
		if (i mod sz_disp eq 0) and disp then print,'gr.peak#',i,'   z-data:  ',zeta[0],zeta[1],zeta[2]		;amp norm to fiducial, coher norm to fiducial, zposition (nm)
	endfor
	catch,/cancel
endif
aa[2,*] = aa2original
end
;
;---------------------------------------------------------------------------
;
pro UnwrapZCoord, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

if (total(z_unwrap_coeff[*]) eq 0) then return

Zindex = max(where(RowNames[*] eq 'Z Position'))
UnwZindex = max(where(RowNames[*] eq 'Unwrapped Z'))
GrZindex = max(where(RowNames[*] eq 'Group Z Position'))
UnwGrZindex = max(where(RowNames[*] eq 'Unwrapped Group Z'))
Zeta0_index = max(where(RowNames[*] eq 'Zeta0'))
Z_params=[Zeta0_index,Zindex,UnwZindex,GrZindex,UnwGrZindex]

Add_EllipCorrSlope_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_AddEllipticitySlopeCorrection')
Add_EllipCorrSlope=widget_info(Add_EllipCorrSlope_id,/button_set)

sz=size(CGroupParams)
sz_disp=fix(sz[2]/100)

for i = 0l,(sz[2]-1) do begin
	if (total(z_unwrap_coeff[*]) ne 0) and (CGrpSize ge 49) then begin 	; extract unwrapped Z position (from known dependence of ellipticity on Z-position)
		z_unwrap_coeff_local=z_unwrap_coeff
		if Add_EllipCorrSlope then z_unwrap_coeff_local[0]=z_unwrap_coeff_local[0]	$
				+ (CGroupParams[2,i]-ellipticity_slopes[0])*ellipticity_slopes[1]	$
				+ (CGroupParams[3,i]-ellipticity_slopes[2])*ellipticity_slopes[3]
		Z_ellip = Z_ell(CGroupParams[43,i],z_unwrap_coeff_local)
		CGroupParams[UnwZindex,i]= CGroupParams[Zindex,i] + round((Z_ellip - CGroupParams[Zindex,i])/wind_range) * wind_range
		CGroupParams[45,i]= CGroupParams[UnwZindex,i] - Z_ellip
	endif
	if i mod sz_disp eq 0 then print,'peak#',i,'   z-data:  ',CGroupParams[Zindex,i],CGroupParams[UnwZindex,i]		;amp norm to fiducial, coher norm to fiducial, zposition (nm)

endfor
catch,/cancel

if mean(CGroupParams[38,*]) ne 0 then begin		; if the grouping was perfromed on A1, A2, A3, perform Z-extraction for grpous
	for i = 0l,(sz[2]-1) do begin
		if CGroupParams[24,i] gt 1 then begin
			if (total(z_unwrap_coeff[*]) ne 0) and (CGrpSize ge 49) then begin 	; extract unwrapped Z position (from known dependence of ellipticity on Z-position)
				Grz_unwrap_coeff_local=z_unwrap_coeff
				if Add_EllipCorrSlope then Grz_unwrap_coeff_local[0]=Grz_unwrap_coeff_local[0]	$
					+ (CGroupParams[2,i]-ellipticity_slopes[0])*ellipticity_slopes[1]	$
					+ (CGroupParams[3,i]-ellipticity_slopes[2])*ellipticity_slopes[3]
				Z_ellip = Z_ell(CGroupParams[46,i],Grz_unwrap_coeff_local)
				CGroupParams[UnwGrZindex,i] = CGroupParams[GrZindex,i] + round((Z_ellip - CGroupParams[GrZindex,i])/wind_range) * wind_range
				CGroupParams[48,i] = CGroupParams[UnwGrZindex,i] - Z_ellip
			endif else if CGrpSize ge 49 then CGroupParams[UnwGrZindex,i] = CGroupParams[GrZindex,i]
		endif else begin
			CGroupParams[GrZindex,i]=CGroupParams[Zindex,i]
			CGroupParams[41,i]=CGroupParams[35,i]
			CGroupParams[42,i]=CGroupParams[36,i]
			if (z_unwrap_coeff[0] ne 0) and (CGrpSize ge 49) then begin
				CGroupParams[UnwGrZindex,i] = CGroupParams[UnwZindex,i]
				CGroupParams[48,i] = CGroupParams[45,i]
			endif
		endelse
		if i mod sz_disp eq 0 then print,'gr.peak#',i,'   z-data:  ',CGroupParams[GrZindex,i],CGroupParams[UnwGrZindex,i]		;amp norm to fiducial, coher norm to fiducial, zposition (nm)
	endfor
endif
catch,/cancel

nZpar=n_elements(Z_params)-1
for j=0,nZpar do ParamLimits[Z_params[j],2]=Median(CGroupParams[Z_params[j],*])
ParamLimits[41,2]=(ParamLimits[41,2]>ParamLimits[35,2])
for j=0,nZpar do ParamLimits[Z_params[j],0] = min(CGroupParams[Z_params[j],*])
for j=0,nZpar do ParamLimits[Z_params[j],1] = max(CGroupParams[Z_params[j],*])
for j=0,nZpar do ParamLimits[Z_params[j],3] = ParamLimits[Z_params[j],1] - ParamLimits[Z_params[j],0]
for j=0,nZpar do ParamLimits[Z_params[j],2] = (ParamLimits[Z_params[j],1] + ParamLimits[Z_params[j],0])/2.

LimitUnwrapZ

print,'Z Operations: finished Z-extraction'
TopIndex = (CGrpSize-1) > 41
TopID=ids[where(names eq 'WID_BASE_0_PeakSelector')]
wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(ParamLimits[0:TopIndex,0:3]), use_table_select=[0,0,3,TopIndex]
widget_control,wtable,set_value=transpose(CGroupParams[0:TopIndex,sz[2]/2]), use_table_select=[4,0,4,TopIndex]
widget_control, wtable, /editable,/sensitive

end
;
;---------------------------------------------------------------------------
;
pro OnAddOffsetSlope, Event			; adds constant offset and slope (per frames) to Z-coordinates
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

WidZPhaseOffsetID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_phase_offset')
widget_control,WidZPhaseOffsetID,get_value=phase_offset
WidZPhaseSlopeID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_phase_slope')
widget_control,WidZPhaseSlopeID,get_value=phase_slope
print,'phase offset=',phase_offset,'    phase_slope=',phase_slope

Label_start=min(CGroupParams[26,*])
for i_l=0,(n_elements(wind_range)-1) do begin
	labelindecis=where(CGroupParams[26,*] eq (i_l+Label_start))
	CGroupParams[34,labelindecis]=(Cgroupparams[34,labelindecis] + phase_offset+CGroupParams[9,labelindecis]/1000.0*phase_slope + 4.0 * wind_range[i_l]) mod wind_range[i_l]
	CGroupParams[40,labelindecis]=(Cgroupparams[40,labelindecis] + phase_offset+CGroupParams[9,labelindecis]/1000.0*phase_slope + 4.0 * wind_range[i_l]) mod wind_range[i_l]
endfor
if CGrpSize ge 49 then begin
	CGroupParams[44,*]= Cgroupparams[44,*] + phase_offset + CGroupParams[9,*]/1000.0*phase_slope
	CGroupParams[47,*]= Cgroupparams[47,*] + phase_offset + CGroupParams[9,*]/1000.0*phase_slope
endif

if CGrpSize ge 49 then begin
	for i=44,47,3 do begin
		valid_cgp=WHERE(FINITE(CGroupParams[i,*]),cnt)
		ParamLimits[i,0]=1.1*min(CGroupParams[i,valid_cgp]) < 0.9*min(CGroupParams[i,valid_cgp])
		ParamLimits[i,1]=1.1*max(CGroupParams[i,valid_cgp])
		ParamLimits[i,3]=ParamLimits[i,1] - ParamLimits[i,0]
		ParamLimits[i,2]=(ParamLimits[i,1] + ParamLimits[i,0])/2.
	endfor

	LimitUnwrapZ

	wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
	widget_control,wtable,get_value=ini_table, use_table_select=[0,44,3,47]
	ini_table[0:3,0]=ParamLimits[44,0:3]
	ini_table[0:3,3]=ParamLimits[47,0:3]
	widget_control,wtable,set_value=ini_table, use_table_select=[0,44,3,47]
	widget_control, wtable, /editable,/sensitive
endif

; adjust z_unwrap_coeff[0] for phase offset
z_unwrap_coeff [0] = z_unwrap_coeff [0] + phase_offset
EllipFitCoeff_WidID = Widget_Info(Event.top, find_by_uname='WID_TABLE_EllipticityFitCoeff')
widget_control,EllipFitCoeff_WidID,set_value=z_unwrap_coeff

end
;
;----------------------------------------------------------------------------
;
pro OptimizeSlopeCorrection, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

OptimizationMode_id=widget_info(event.top,FIND_BY_UNAME='WID_DROPLIST_Optimization_Mode')
OptimizationMode = widget_info(OptimizationMode_id,/DROPLIST_SELECT)
print,'Optimization Mode: '+strtrim(OptimizationMode,2)

Gr_Pk = OptimizationMode mod 2
if Gr_Pk then FilterIt else GroupFilterIt
case OptimizationMode of
	0: OptimizeSlopeCorrection_local, Event, 1	; local, Groups
	1: OptimizeSlopeCorrection_local, Event, 0	; local, Peaks
	2: OptimizeSlopeCorrection_Bridge, Event, 1	; Bridge, Groups
	3: OptimizeSlopeCorrection_Bridge, Event, 0	; Bridge, Peaks
endcase

UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))                        ; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))                ; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))                ; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))        ; CGroupParametersGP[48,*] - Group Z Position Error

if bridge_exists then begin
	print,'Reloading the Bridge Array'
	CATCH, Error_status
	CGroupParams_bridge = SHMVAR(shmName_data)
	CGroupParams_bridge[UnwZ_ind,*] = CGroupParams[UnwZ_ind,*]
	CGroupParams_bridge[UnwZErr_ind,*] = CGroupParams[UnwZErr_ind,*]
	CGroupParams_bridge[UnwGrZ_ind,*] = CGroupParams[UnwGrZ_ind,*]
	CGroupParams_bridge[UnwGrZErr_ind,*] = CGroupParams[UnwGrZErr_ind,*]
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

end
;
;----------------------------------------------------------------------------
;
pro OptimizeSlopeCorrection_local, Event, Pk_Gr	; Pk_Gr = 0 - peaks, 1 - groups
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

if (total(z_unwrap_coeff[*]) eq 0) then return

Add_EllipCorrSlope_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_AddEllipticitySlopeCorrection')
widget_control,Add_EllipCorrSlope_id,set_button=1
EllipCorrSlope_WidID = Widget_Info(Event.top, find_by_uname='WID_TABLE_EllipticityCorrectionSlope')
UnwrZ_Err_index = Pk_Gr ? max(where(RowNames[*] eq 'Unwrapped Group Z Error'))	:	max(where(RowNames[*] eq 'Unwrapped Z Error'))

ellipticity_slopes0=ellipticity_slopes
slope_x_center=ellipticity_slopes[1]
slope_y_center=ellipticity_slopes[3]
amp_x=dblarr(5)
slope_x=dblarr(5)
amp_y=dblarr(5)
slope_y=dblarr(5)
offs=dblarr(5)
step=0.2

hist_set = Pk_Gr ? where((filter and (CGroupParams[25,*] eq 1)),cnt)	:	where(filter,cnt)
Zmin=-0.5*wind_range[0]
Zmax=-1.0*Zmin
nbns=50
redrawhist=1
for i=0,4 do begin
	slope_x[i]=slope_x_center+(i-2)*step
	ellipticity_slopes[1]=slope_x[i]
	widget_control,EllipCorrSlope_WidID,set_value=ellipticity_slopes, use_table_select=[0,0,(n_elements(ellipticity_slopes)-1),0]
	UnwrapZCoord, Event
	Estimate_Unwrap_Ghost, UnwrZ_Err_index, hist_set, ZMin, ZMax, nbns, gres, redrawhist,1; display results
	xyouts,0.12,0.86,'X-slope = '+ strtrim(ellipticity_slopes[1],2)+'    Y-slope = '+ strtrim(ellipticity_slopes[3],2), color=100, charsize=1.5,CHARTHICK=1.5,/NORMAL
	amp_x[i]=gres[0]
endfor
xmax=max(amp_x,ind_xmax)
ellipticity_slopes[1]=slope_x[ind_xmax]

for i=0,4 do begin
	slope_y[i]=slope_y_center+(i-2)*step
	ellipticity_slopes[3]=slope_y[i]
	widget_control,EllipCorrSlope_WidID,set_value=ellipticity_slopes, use_table_select=[0,0,(n_elements(ellipticity_slopes)-1),0]
	UnwrapZCoord, Event
	Estimate_Unwrap_Ghost, UnwrZ_Err_index, hist_set, ZMin, ZMax, nbns, gres, redrawhist,1; display results
	xyouts,0.12,0.86,'X-slope = '+ strtrim(ellipticity_slopes[1],2)+'    Y-slope = '+ strtrim(ellipticity_slopes[3],2), color=100, charsize=1.5,CHARTHICK=1.5,/NORMAL
	amp_y[i]=gres[0]
	offs[i]=gres[1]
endfor

ymax=max(amp_y,ind_ymax)
EllipFitCoeff_WidID = Widget_Info(Event.top, find_by_uname='WID_TABLE_EllipticityFitCoeff')
z_unwrap_coeff[0] = z_unwrap_coeff[0] + offs[ind_ymax]
widget_control,EllipFitCoeff_WidID,set_value=z_unwrap_coeff, use_table_select=[0,0,(n_elements(z_unwrap_coeff)-1),0]

xfit_coeffs=poly_fit(slope_x,amp_x,2,yfit=xfit)
yfit_coeffs=poly_fit(slope_y,amp_y,2,yfit=yfit)
x_max=((xfit_coeffs[1]/(-2.0)/xfit_coeffs[2])>min(slope_x))<max(slope_x)
ellipticity_slopes[1]=x_max
y_max=((yfit_coeffs[1]/(-2.0)/yfit_coeffs[2])>min(slope_y))<max(slope_y)
ellipticity_slopes[3]=y_max
x_par=(findgen(101)/50.0-1.0)*2.0*step+slope_x_center
xfit=poly(x_par,xfit_coeffs)
y_par=(findgen(101)/50.0-1.0)*2.0*step+slope_y_center
yfit=poly(y_par,yfit_coeffs)

!p.multi=[0,1,2,0,0]
ymn=min([amp_x,xfit]) & ymx=max([amp_x,xfit])
yrange=[(ymn-0.1*(ymx-ymn)),(ymn-0.1*(ymx-ymn))]
plot,slope_x,amp_x,xtitle='X slope',ytitle='Amplitude',yrange=yrange
oplot,slope_x,amp_x,psym=2
oplot,x_par,xfit,col=100
oplot,[x_max,x_max],[max(xfit),max(xfit)],psym=4,symsize=2.0,col=150
xyouts,0.12,0.96,'Optimal X-slope = '+ strtrim(ellipticity_slopes[1],2)+',   Optimal Y-slope = '+ strtrim(ellipticity_slopes[3],2), color=100, charsize=1.5,CHARTHICK=1.5,/NORMAL

!p.multi=[1,1,2,0,0]
ymn=min([amp_y,yfit]) & ymx=max([amp_y,yfit])
yrange=[(ymn-0.1*(ymx-ymn)),(ymn-0.1*(ymx-ymn))]
plot,slope_y,amp_y,xtitle='Y slope',ytitle='Amplitude',yrange=yrange
oplot,slope_y,amp_y,psym=2
oplot,y_par,yfit,col=100
oplot,[y_max,y_max],[max(yfit),max(yfit)],psym=4,symsize=2.0,col=150

!p.multi=[0,0,0,0,0]
widget_control,EllipCorrSlope_WidID,set_value=ellipticity_slopes, use_table_select=[0,0,(n_elements(ellipticity_slopes)-1),0]

UnwrapZCoord, Event

end
;
;----------------------------------------------------------------------------
;
pro OptimizeSlopeCorrection_Bridge, Event, Pk_Gr	; Pk_Gr = 0 - peaks, 1 - groups
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

if (total(z_unwrap_coeff[*]) eq 0) then return

Add_EllipCorrSlope_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_AddEllipticitySlopeCorrection')
widget_control,Add_EllipCorrSlope_id,set_button=1
EllipCorrSlope_WidID = Widget_Info(Event.top, find_by_uname='WID_TABLE_EllipticityCorrectionSlope')
UnwrZ_Err_index = Pk_Gr ? max(where(RowNames[*] eq 'Unwrapped Group Z Error'))	:	max(where(RowNames[*] eq 'Unwrapped Z Error'))
print,'Pk_Gr='+strtrim(Pk_Gr,2)+',    UnwrZ_Err_index='+strtrim(UnwrZ_Err_index,2)
ellipticity_slopes0=ellipticity_slopes
slope_x_center=ellipticity_slopes[1]
slope_y_center=ellipticity_slopes[3]
amp_x=dblarr(5)
slope_x=dblarr(5)
amp_y=dblarr(5)
slope_y=dblarr(5)
offs=dblarr(5)
step=0.2
nloops=5


Zmin=-0.5*wind_range[0]
Zmax=-1.0*Zmin
nbns=50
redrawhist=1

if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else	idl_pwd=pref_get('IDL_WDE_START_DIR')
cd,current=curr_pwd
FILE_MKDIR,curr_pwd+'/temp'
		;save variables for cluster cpu access

print,'Starting IDL bridge worker routines'
;Starting IDL bridge workers
obridge=obj_new("IDL_IDLBridge", output='')
for i=1, nloops-1 do obridge=[obridge, obj_new("IDL_IDLBridge", output='')]
print,'data_dir:	',curr_pwd
print,'IDL_dir:		',IDL_pwd

shmName='Status_reports'
max_len=150
SHMMAP,shmName,/BYTE, Dimension=nloops*max_len, GET_OS_HANDLE=OS_handle_val1
Reports=SHMVAR(shmName)
shmName_data='iPALM_data'
if Pk_Gr then GP_indecis=where((filter and (CGroupParams[25,*] eq 1)),iPALM_data_cnt) else GP_indecis=where(filter,iPALM_data_cnt)
;iPALM_data_cnt=n_elements(CGroupParams)

SHMMAP,shmName_data,/FLOAT, Dimension=[CGrpSize,iPALM_data_cnt], GET_OS_HANDLE=OS_handle_val2
CGroupParams_bridge=SHMVAR(shmName_data)
CGroupParams_bridge[0,0]=CGroupParams[*,GP_indecis]

save, curr_pwd,idl_pwd, iPALM_data_cnt, CGrpSize, ParamLimits, RowNames, nloops, UnwrZ_Err_index, Pk_Gr, $
	aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius,$
		filename='temp/temp.sav'

for i=0, nloops-1 do begin
	ellipticity_slopes_worker=ellipticity_slopes
	slope_x[i]=slope_x_center+(i-2)*step
	ellipticity_slopes_worker[1]=slope_x[i]
	print,ellipticity_slopes_worker
	obridge[i]->setvar, 'nlps',i
	obridge[i]->setvar, 'data_dir',curr_pwd
	obridge[i]->setvar, 'IDL_dir',IDL_pwd
	obridge[i]->setvar, 'ellipticity_slopes_worker',ellipticity_slopes_worker
	obridge[i]->setvar, 'OS_handle_val1',OS_handle_val1
	obridge[i]->setvar, 'OS_handle_val2',OS_handle_val2
	print,'bridge ',i,'  set variables'
	obridge[i]->execute,'cd, IDL_dir'
	print,'bridge ',i,'  changed directory'
	obridge[i]->execute,"restore,'OptimizeSlopeCorrection_Bridge_Worker.sav'"
	obridge[i]->execute,'OptimizeSlopeCorrection_Bridge_Worker,nlps,data_dir,ellipticity_slopes_worker, gres, OS_handle_val1,OS_handle_val2',/NOWAIT
	print,'bridge ',i,'  started'
endfor

Alldone = 0
while alldone EQ 0 do begin
	wait,1
	Alldone = 1
	for i=0, nloops-1 do begin
		bridge_done=obridge[i]->Status()
		print,'Bridge',i,'  status:',bridge_done,';    ',string(Reports[(i*max_len):((i+1)*max_len-1)])
		Alldone = Alldone * (bridge_done ne 1)
	endfor
endwhile

for i=0, nloops-1 do begin
	gres = obridge[i]->getvar('gres')
	print,gres
	amp_x[i]=gres[0]
endfor

print,amp_x

for nlps=0L,nloops-1 do	obj_destroy, obridge[nlps]

print,'Starting IDL bridge worker routines'
;Starting IDL bridge workers
obridge=obj_new("IDL_IDLBridge", output='')
for i=1, nloops-1 do obridge=[obridge, obj_new("IDL_IDLBridge", output='')]
print,'data_dir:	',curr_pwd
print,'IDL_dir:		',IDL_pwd

CGroupParams_bridge[0,0]=CGroupParams[*,GP_indecis]


xmax=max(amp_x,ind_xmax)
ellipticity_slopes[1]=slope_x[ind_xmax]

for i=0, nloops-1 do begin
	ellipticity_slopes_worker=ellipticity_slopes
	slope_y[i]=slope_y_center+(i-2)*step
	ellipticity_slopes_worker[3]=slope_y[i]
	print,ellipticity_slopes_worker
	obridge[i]->setvar, 'nlps',i
	obridge[i]->setvar, 'data_dir',curr_pwd
	obridge[i]->setvar, 'IDL_dir',IDL_pwd
	obridge[i]->setvar, 'ellipticity_slopes_worker',ellipticity_slopes_worker
	obridge[i]->setvar, 'OS_handle_val1',OS_handle_val1
	obridge[i]->setvar, 'OS_handle_val2',OS_handle_val2
	print,'bridge ',i,'  set variables'
	obridge[i]->execute,'cd, IDL_dir'
	print,'bridge ',i,'  changed directory'
	obridge[i]->execute,"restore,'OptimizeSlopeCorrection_Bridge_Worker.sav'"
	obridge[i]->execute,'OptimizeSlopeCorrection_Bridge_Worker,nlps,data_dir,ellipticity_slopes_worker, gres, OS_handle_val1,OS_handle_val2',/NOWAIT
	print,'bridge ',i,'  started'
endfor

Alldone = 0
while alldone EQ 0 do begin
	wait,1
	Alldone = 1
	for i=0, nloops-1 do begin
		bridge_done=obridge[i]->Status()
		print,'Bridge',i,'  status:',bridge_done,';    ',string(Reports[(i*max_len):((i+1)*max_len-1)])
		Alldone = Alldone * (bridge_done ne 1)
	endfor
endwhile

for i=0, nloops-1 do begin
	gres = obridge[i]->getvar('gres')
	print,gres
	amp_y[i]=gres[0]
	offs[i]=gres[1]
endfor

SHMUnmap, shmName
SHMUnmap, shmName_data
for nlps=0L,nloops-1 do	obj_destroy, obridge[nlps]

file_delete,'temp/temp.sav'
file_delete,'temp'
cd,curr_pwd

ymax=max(amp_y,ind_ymax)
EllipFitCoeff_WidID = Widget_Info(Event.top, find_by_uname='WID_TABLE_EllipticityFitCoeff')
z_unwrap_coeff[0] = z_unwrap_coeff[0] + offs[ind_ymax]
widget_control,EllipFitCoeff_WidID,set_value=z_unwrap_coeff, use_table_select=[0,0,(n_elements(z_unwrap_coeff)-1),0]

xfit_coeffs=poly_fit(slope_x,amp_x,2,yfit=xfit)
yfit_coeffs=poly_fit(slope_y,amp_y,2,yfit=yfit)
x_max=((xfit_coeffs[1]/(-2.0)/xfit_coeffs[2])>min(slope_x))<max(slope_x)
ellipticity_slopes[1]=x_max
y_max=((yfit_coeffs[1]/(-2.0)/yfit_coeffs[2])>min(slope_y))<max(slope_y)
ellipticity_slopes[3]=y_max
x_par=(findgen(101)/50.0-1.0)*2.0*step+slope_x_center
xfit=poly(x_par,xfit_coeffs)
y_par=(findgen(101)/50.0-1.0)*2.0*step+slope_y_center
yfit=poly(y_par,yfit_coeffs)

!p.multi=[0,1,2,0,0]
ymn=min([amp_x,xfit]) & ymx=max([amp_x,xfit])
yrange=[(ymn-0.1*(ymx-ymn)),(ymn-0.1*(ymx-ymn))]
plot,slope_x,amp_x,xtitle='X slope',ytitle='Amplitude',yrange=yrange
oplot,slope_x,amp_x,psym=2
oplot,x_par,xfit,col=100
oplot,[x_max,x_max],[max(xfit),max(xfit)],psym=4,symsize=2.0,col=150
xyouts,0.12,0.96,'Optimal X-slope = '+ strtrim(ellipticity_slopes[1],2)+',   Optimal Y-slope = '+ strtrim(ellipticity_slopes[3],2), color=100, charsize=1.5,CHARTHICK=1.5,/NORMAL

!p.multi=[1,1,2,0,0]
ymn=min([amp_y,yfit]) & ymx=max([amp_y,yfit])
yrange=[(ymn-0.1*(ymx-ymn)),(ymn-0.1*(ymx-ymn))]
plot,slope_y,amp_y,xtitle='Y slope',ytitle='Amplitude',yrange=yrange
oplot,slope_y,amp_y,psym=2
oplot,y_par,yfit,col=100
oplot,[y_max,y_max],[max(yfit),max(yfit)],psym=4,symsize=2.0,col=150

!p.multi=[0,0,0,0,0]
widget_control,EllipCorrSlope_WidID,set_value=ellipticity_slopes, use_table_select=[0,0,(n_elements(ellipticity_slopes)-1),0]

UnwrapZCoord, Event
end
;
;-----------------------------------------------------------------
;
pro OptimizeSlopeCorrection_Bridge_Worker,nlps,data_dir,ellipticity_slopes_worker, gres, OS_handle_val1,OS_handle_val2
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

cd,data_dir
restore,'temp/temp.sav'

debug_fname='temp/debug'+strtrim(nlps,2)+'.txt'
;close,(nlps+3)
;openw,(nlps+3),debug_fname

ellipticity_slopes = ellipticity_slopes_worker
;printf,(nlps+3),'worker started, nloops='+strtrim(nloops,2)
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	rep=' OptimizeSlopeCorrection_Bridge_Worker Error:  '+!ERROR_STATE.msg
	;printf,(nlps+3),rep
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	close,(nlps+3)
	CATCH, /CANCEL
	return
ENDIF
shmName='Status_reports'
max_len=150
SHMMAP,shmName,/BYTE, Dimension=nloops*max_len,OS_Handle=OS_handle_val1
Reports=SHMVAR(shmName)
rep_i=nlps*max_len

shmName_data='iPALM_data'
SHMMAP,shmName_data,/FLOAT, Dimension=[CGrpSize,iPALM_data_cnt],OS_Handle=OS_handle_val2
CGroupParams_bridge=SHMVAR(shmName_data)
CGroupParams=CGroupParams_bridge

if (total(z_unwrap_coeff[*]) eq 0) then return

Zindex = max(where(RowNames[*] eq 'Z Position'))
UnwZindex = max(where(RowNames[*] eq 'Unwrapped Z'))
GrZindex = max(where(RowNames[*] eq 'Group Z Position'))
UnwGrZindex = max(where(RowNames[*] eq 'Unwrapped Group Z'))

sz=size(CGroupParams)
sz_disp=fix(sz[2]/100)
;printf,(nlps+3),'Pk_Gr = '+strtrim(Pk_Gr,2)
;printf,(nlps+3),'ellipticity_slopes = '
;printf,(nlps+3),strtrim(ellipticity_slopes,2)
if Pk_Gr eq 0 then begin
	for i = 0l,(sz[2]-1) do begin
		if (total(z_unwrap_coeff[*]) ne 0) and (CGrpSize ge 49) then begin 	; extract unwrapped Z position (from known dependence of ellipticity on Z-position)
			z_unwrap_coeff_local=z_unwrap_coeff
			z_unwrap_coeff_local[0]=z_unwrap_coeff_local[0]	$
				+ (CGroupParams[2,i]-ellipticity_slopes[0])*ellipticity_slopes[1]	$
				+ (CGroupParams[3,i]-ellipticity_slopes[2])*ellipticity_slopes[3]
			Z_ellip = Z_ell(CGroupParams[43,i],z_unwrap_coeff_local)
			CGroupParams[UnwZindex,i]= CGroupParams[Zindex,i] + round((Z_ellip - CGroupParams[Zindex,i])/wind_range) * wind_range
			CGroupParams[UnwrZ_Err_index,i]= CGroupParams[UnwZindex,i] - Z_ellip
		endif
;	if i mod sz_disp eq 0 then print,'peak#',i,'   z-data:  ',CGroupParams[Zindex,i],CGroupParams[UnwZindex,i]		;amp norm to fiducial, coher norm to fiducial, zposition (nm)
		if i mod sz_disp eq 0 then begin
			rep='Fr.peak: '+strtrim(i,2)+'   z-data:  '+strtrim(CGroupParams[Zindex,i],2)+',   '+strtrim(CGroupParams[UnwZindex,i],2)		;amp norm to fiducial, coher norm to fiducial, zposition (nm)
			;printf,(nlps+3),rep
			if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
			Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
		endif
	endfor
endif

if Pk_Gr eq 1 then begin
	if mean(CGroupParams[38,*]) ne 0 then begin		; if the grouping was perfromed on A1, A2, A3, perform Z-extraction for grpous
		for i = 0l,(sz[2]-1) do begin
			if (total(z_unwrap_coeff[*]) ne 0) and (CGrpSize ge 49) then begin 	; extract unwrapped Z position (from known dependence of ellipticity on Z-position)
					Grz_unwrap_coeff_local=z_unwrap_coeff
					Grz_unwrap_coeff_local[0]=Grz_unwrap_coeff_local[0]	$
					+ (CGroupParams[19,i]-ellipticity_slopes[0])*ellipticity_slopes[1]	$
					+ (CGroupParams[20,i]-ellipticity_slopes[2])*ellipticity_slopes[3]
					Z_ellip = Z_ell(CGroupParams[46,i],Grz_unwrap_coeff_local)
					CGroupParams[UnwGrZindex,i] = CGroupParams[GrZindex,i] + round((Z_ellip - CGroupParams[GrZindex,i])/wind_range) * wind_range
					CGroupParams[UnwrZ_Err_index,i] = CGroupParams[UnwGrZindex,i] - Z_ellip
			endif else if CGrpSize ge 49 then CGroupParams[UnwGrZindex,i] = CGroupParams[GrZindex,i]

			if i mod sz_disp eq 0 then begin
				rep='Gr.peak: '+strtrim(i,2)+'   z-data:  '+strtrim(CGroupParams[GrZindex,i],2)+',   '+strtrim(CGroupParams[UnwGrZindex,i],2)		;amp norm to fiducial, coher norm to fiducial, zposition (nm)
				;printf,(nlps+3),rep
				if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
				Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
			endif
		endfor
	endif
endif
catch,/cancel

ZMin =(-0.5) * wind_range
ZMax = 0.5 * wind_range
nbns=50
hist_set=lindgen(iPALM_data_cnt)
	rep='UnwrZ_Err_index: '+strtrim(UnwrZ_Err_index,2)
	;printf,(nlps+3),rep
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
redrawhist=0
;printf,(nlps+3),'size(hist_set)='
;printf,(nlps+3),strtrim(size(hist_set),2)
;printf,(nlps+3),'ZMin='+strtrim(ZMin,2)
;printf,(nlps+3),'ZMax='+strtrim(ZMax,2)
;printf,(nlps+3),'nbns='+strtrim(nbns,2)
;printf,(nlps+3),'redrawhist='+strtrim(redrawhist,2)
Estimate_Unwrap_Ghost, UnwrZ_Err_index, hist_set, ZMin, ZMax, nbns, gres, redrawhist,0; do not display results
;printf,(nlps+3),'gres='
;printf,(nlps+3),strtrim(gres,2)
;close,(nlps+3)
end
;
;-----------------------------------------------------------------
;
pro OnButtonClose, Event		; closes the menu widget
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnPickWINDFile, Event		; selects the WND file to save the fit parameters (when performing Z-calibration) and to load the fit parameters from (when performing Z-extraction)
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
wfilename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.sav'],title='Select *WND.sav file to open')
if wfilename ne '' then begin
	cd,fpath
	WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WindFilename')
	widget_control,WFileWidID,SET_VALUE = wfilename
	wfile_info=FILE_INFO(wfilename)
		if wfile_info.exists then begin
			wind_range = 0
			z_unwrap_coeff = transpose([0.0,0.0,0.0])
			restore,filename=wfilename
			ReadWindPoint, Event
			Update_Table_EllipticityFitCoeff, Event
		endif

	WlWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WL')
	wl_txt=string(lambda_vac,FORMAT='(F8.2)')
	widget_control,WlWidID,SET_VALUE = wl_txt

endif
end
;
;-----------------------------------------------------------------
;
pro Initialize_Z_operations, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
WidLabel0 = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,WidLabel0,get_value=dfilename
droot = strmid(dfilename,0,strpos(dfilename,sep,/REVERSE_OFFSET,/REVERSE_SEARCH))
ddroot = strmid(droot,0,strpos(droot,sep,/REVERSE_OFFSET,/REVERSE_SEARCH))

Lookup_Unwrap_Display_id=widget_info(wWidget,FIND_BY_UNAME='WID_DROPLIST_LookupUnwrapDisplayType')
widget_control,Lookup_Unwrap_Display_id,SET_DROPLIST_SELECT=(!VERSION.OS_family eq 'unix')? 2 : 0			;Sets the default value to "Local - No Display " for Windows and to "Cluster - No Display" for UNIX

nmperframe = 8.0			; nm per frame. calibration using piezo parameters (2um per 10V, 0.04V  per frame)

WFileWidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_WindFilename')
if dfilename ne '' then begin
	dot_pos=strpos(dfilename,'IDL.sav',/REVERSE_OFFSET,/REVERSE_SEARCH)
	if dot_pos gt 0 then begin
		suggested_windfile=strmid(dfilename,0,dot_pos)+'F1'
		widget_control,WFileWidID,SET_VALUE = suggested_windfile
	endif
endif

wind_range_ini=wind_range
z_unwrap_coeff_ini=z_unwrap_coeff
ellipticity_slopes_ini=ellipticity_slopes
x_center=((size(TotalRawData))[0] ne 0) ? (size(TotalRawData))[1]/2.0 : 0
y_center=((size(TotalRawData))[0] ne 0) ? (size(TotalRawData))[2]/2.0 : 0

if (size(wfilename))[2] gt 0 then begin
	wroot = strmid(wfilename,0,strpos(wfilename,sep,/REVERSE_OFFSET,/REVERSE_SEARCH))
	wwroot = strmid(wroot,0,strpos(wroot,sep,/REVERSE_OFFSET,/REVERSE_SEARCH))
	if (wfilename ne '') and ((wwroot eq ddroot) or (wwroot eq droot)) then begin
		widget_control,WFileWidID,SET_VALUE = wfilename
		wfile_info=FILE_INFO(wfilename)
		if wfile_info.exists then  begin
			z_unwrap_coeff = [0.0,0.0,0.0]
			ellipticity_slopes = [x_center,0.0,y_center,0.0]
			restore,filename=wfilename
		endif
	endif else begin
		z_unwrap_coeff = [0.0,0.0,0.0]
		ellipticity_slopes = [x_center,0.0,y_center,0.0]
	endelse
endif else begin
	z_unwrap_coeff = [0.0,0.0,0.0]
	ellipticity_slopes = [x_center,0.0,y_center,0.0]
endelse

;WlWidID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_WL')
;wl_id=widget_info(WlWidID,/DROPLIST_SELECT)
;wl_setting = ((lambda_vac ge 510) and (lambda_vac le 530)) ? 1 : 0
;widget_control,WlWidID, SET_DROPLIST_SELECT = wl_setting
WlWidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_WL')
wl_txt=string(lambda_vac,FORMAT='(F8.2)')
widget_control,WlWidID,SET_VALUE = wl_txt

FitmethodID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Z_Fit_Method')
widget_control,FitmethodID,SET_DROPLIST_SELECT=1

WSlopeSliderID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Z_phase_slope')
widget_control,WSlopeSliderID,SET_VALUE = 0

WPhaseSliderID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Z_phase_offset')
widget_control,WPhaseSliderID,SET_VALUE = 0

; if the wfilename exists and had already beed read, then check if the wind_range in the memory prior to reading the wfilename
; is different (if wind range was re-sacled). If it is different and not equal to default value (wind_range=220.0), then initialize it to this newely read value.
if wind_range_ini[0] ne 220.0 then begin
	wind_range = wind_range_ini
	z_unwrap_coeff = z_unwrap_coeff_ini
	ellipticity_slopes = ellipticity_slopes_ini
endif
WindRangeID = Widget_Info(wWidget, find_by_uname='WID_TEXT_WindPeriod')
wind_range_txt=string(wind_range[0],FORMAT='(F6.2)')
widget_control,WindRangeID,SET_VALUE = wind_range_txt

WID_TEXT_GuideStar_Radius_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_GuideStar_Radius')
GS_radius_txt=string(GS_radius[0],FORMAT='(F6.2)')
widget_control,WID_TEXT_GuideStar_Radius_ID,SET_VALUE = GS_radius_txt
WID_TEXT_GuideStarAncFilename_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_GuideStarAncFilename')
widget_control,WID_TEXT_GuideStarAncFilename_ID,SET_VALUE = GS_anc_fname

EllipFitCoeff_WidID = Widget_Info(wWidget, find_by_uname='WID_TABLE_EllipticityFitCoeff')
widget_control,EllipFitCoeff_WidID,COLUMN_WIDTH=[150,70,70,70],use_table_select = [ -1, 0, 2, 1 ]
widget_control,EllipFitCoeff_WidID,set_value=z_unwrap_coeff,TABLE_YSIZE=1

EllipCorrSlope_WidID = Widget_Info(wWidget, find_by_uname='WID_TABLE_EllipticityCorrectionSlope')
widget_control,EllipCorrSlope_WidID,COLUMN_WIDTH=[1,90,95,90,95],use_table_select = [ -1, 0, 3, 1 ]
widget_control,EllipCorrSlope_WidID,set_value=ellipticity_slopes,TABLE_YSIZE=1

Zvalue_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Zvalue')
Zvalue_txt='0.0'
widget_control,Zvalue_ID,SET_VALUE = Zvalue_txt
Zsigma_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Zuncertainty')
Zsigma_txt='0.0'
widget_control,Zsigma_ID,SET_VALUE = Zsigma_txt

end
;
;-----------------------------------------------------------------
;
pro WriteGudeStarRadius, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, names, modalList
WID_TEXT_GuideStar_RadiusID = Widget_Info(Event.top, find_by_uname='WID_TEXT_GuideStar_Radius')
widget_control,WID_TEXT_GuideStar_RadiusID,GET_VALUE = GS_radius_txt
BASE_GuideStar_num=min(where(names eq 'WID_BASE_GuideStar'))
if BASE_GuideStar_num ge 0 then begin
	BASE_GuideStarID=ids[BASE_GuideStar_num]
	WID_TEXT_XY_GuideStar_Radius_ID = Widget_Info(BASE_GuideStarID, find_by_uname='WID_TEXT_XY_GuideStar_Radius')
	widget_control,WID_TEXT_XY_GuideStar_Radius_ID,SET_VALUE = GS_radius_txt
endif
GS_radius = float(GS_radius_txt[0])
end
;
;-----------------------------------------------------------------
;
pro OnPickGuideStarAncFile, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, names, modalList
GS_anc_fname = Dialog_Pickfile(/read,get_path=fpath,filter=['*.anc'],title='Select *.anc file to open')
GS_anc_file_info=FILE_INFO(GS_anc_fname)
if (GS_anc_fname ne '') and GS_anc_file_info.exists then begin
	cd,fpath
	WID_TEXT_GuideStarAncFilename_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_GuideStarAncFilename')
	widget_control,WID_TEXT_GuideStarAncFilename_ID,SET_VALUE = GS_anc_fname
	BASE_GuideStar_num=min(where(names eq 'WID_BASE_GuideStar'))
	if BASE_GuideStar_num ge 0 then begin
		BASE_GuideStarID=ids[BASE_GuideStar_num]
		WID_TEXT_XY_GuideStarAncFilename_ID = Widget_Info(BASE_GuideStarID, find_by_uname='WID_TEXT_XY_GuideStarAncFilename')
		widget_control,WID_TEXT_XY_GuideStarAncFilename_ID,SET_VALUE = GS_anc_fname
	endif
endif else GS_anc_fname = ''
end
;
;-----------------------------------------------------------------
;
pro On_Buttonpress_WriteEllipticityGuideStar_Z, Event
WriteEllipticityGuideStar_Z_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_WriteEllipticityGuideStar')
WriteEllipticityGuideStar_Z=widget_info(WriteEllipticityGuideStar_Z_id,/button_set)
WriteEllipticityGuideStar_E_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_WriteEllipticityGuideStar_E')
WriteEllipticityGuideStar_E=widget_info(WriteEllipticityGuideStar_E_id,/button_set)
if WriteEllipticityGuideStar_Z then widget_control,WriteEllipticityGuideStar_E_id,set_button=0
end
;
;-----------------------------------------------------------------
;
pro On_Buttonpress_WriteEllipticityGuideStar_E, Event
WriteEllipticityGuideStar_Z_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_WriteEllipticityGuideStar')
WriteEllipticityGuideStar_Z=widget_info(WriteEllipticityGuideStar_Z_id,/button_set)
WriteEllipticityGuideStar_E_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_WriteEllipticityGuideStar_E')
WriteEllipticityGuideStar_E=widget_info(WriteEllipticityGuideStar_E_id,/button_set)
if WriteEllipticityGuideStar_E then widget_control,WriteEllipticityGuideStar_Z_id,set_button=0
end
;
;-----------------------------------------------------------------
;
pro OnTestZDrift, Event	;Shows fit to guide star w/o changing data: Z-coordinate
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius

use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs')
use_multiple_GS = widget_info(use_multiple_GS_id,/button_set)
WID_TEXT_GuideStarAncFilename_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_GuideStarAncFilename')
widget_control,WID_TEXT_GuideStarAncFilename_ID,GET_VALUE = GS_anc_fname
GS_anc_file_info=FILE_INFO(GS_anc_fname)
use_multiple_GS = use_multiple_GS and GS_anc_file_info.exists
ExtractSubsetZ, Event, zdift, use_multiple_GS
end
;
;-----------------------------------------------------------------
;
pro OnWriteZDrift, Event	;Drift corrects z to constant guide star coordinates
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
UnwZ_ind=min(where(RowNames eq 'Unwrapped Z'))
UnwZ_Err_ind=min(where(RowNames eq 'Unwrapped Z Error'))
UnwGrZ_ind=min(where(RowNames eq 'Unwrapped Group Z'))
UnwGrZ_Err_ind=min(where(RowNames eq 'Unwrapped Group Z Error'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))

WriteEllipticityGuideStar_Z_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_WriteEllipticityGuideStar')
WriteEllipticityGuideStar_Z=widget_info(WriteEllipticityGuideStar_Z_id,/button_set)
WriteEllipticityGuideStar_E_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_WriteEllipticityGuideStar_E')
WriteEllipticityGuideStar_E=widget_info(WriteEllipticityGuideStar_E_id,/button_set)

WriteEllipticityGuideStar = WriteEllipticityGuideStar_Z or WriteEllipticityGuideStar_E

use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs')
use_multiple_GS = widget_info(use_multiple_GS_id,/button_set)
WID_TEXT_GuideStarAncFilename_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_GuideStarAncFilename')
widget_control,WID_TEXT_GuideStarAncFilename_ID,GET_VALUE = GS_anc_fname
GS_anc_file_info=FILE_INFO(GS_anc_fname)
use_multiple_GS = use_multiple_GS and GS_anc_file_info.exists

ExtractSubsetZ, Event, zdrift, use_multiple_GS

CGroupParams[Z_ind,*] = (CGroupParams[Z_ind,*] - zdrift[CGroupParams[9,*]] + 10.0 * wind_range) mod wind_range
CGroupParams[GrZ_ind,*] = (CGroupParams[GrZ_ind,*] - zdrift[CGroupParams[9,*]] + 10.0 * wind_range) mod wind_range
if CGrpSize ge 49 then begin
	CGroupParams[UnwZ_ind,*] = CGroupParams[UnwZ_ind,*] - zdrift[CGroupParams[9,*]]
	CGroupParams[UnwGrZ_ind,*] = CGroupParams[UnwGrZ_ind,*] - zdrift[CGroupParams[9,*]]
	if (total(abs(CGroupParams[Ell_ind,0:(100 < n_elements(zdrift))])) ne 0.0) and WriteEllipticityGuideStar then begin
		if WriteEllipticityGuideStar_E then begin
			NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[9,*])+1))	: long64(max(CGroupParams[9,*])+1)
			FR=findgen(NFrames)
			WidDListDispFitMethodID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z_Fit_Method')
			FitMethod = widget_info(WidDListDispFitMethodID,/DropList_Select)
			subsetindex=where(filter eq 1,cnt)
			subset=CGroupParams[*,subsetindex]
			if FitMethod eq 0 then begin
				WidSldFitOrderID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_Fit')
				widget_control,WidSldFitOrderID,get_value=fitorder
				e_coef=poly_fit(subset[9,*],subset[Ell_ind,*],fitorder,YFIT=fit_to_x)
				E_Fit=poly(FR,e_coef)
			endif else begin
				WidSmWidthID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_Sm_Width')
				widget_control,WidSmWidthID,get_value=SmWidth
				indecis=uniq(subset[9,*])
				E_smooth=smooth(subset[6,indecis]*subset[Ell_ind,indecis],SmWidth,/EDGE_TRUNCATE)/smooth(subset[6,indecis],SmWidth,/EDGE_TRUNCATE)
				E_Fit=interpol(E_smooth,subset[9,indecis],FR)
				frame_low = min(subset[9,indecis])
				ind_low=where(FR[*] lt frame_low,c)
				if c ge 1 then E_Fit[ind_low]=E_fit[frame_low]
				frame_high = max(subset[9,indecis])
				ind_high=where(FR[*] gt frame_high,c)
				if c ge 1 then	E_fit[ind_high]=E_fit[frame_high]
			endelse
			ZE_fit=Z_ell (E_fit, z_unwrap_coeff)
		   	ze_drift=ZE_fit-ZE_fit[0]
		endif else ze_drift=zdrift
		Z_ellipticity_corr = Z_ell (CGroupParams[Ell_ind,*], z_unwrap_coeff) - ze_drift[CGroupParams[9,*]]
		CGroupParams[Ell_ind,*] = Ell_Z (Z_ellipticity_corr, z_unwrap_coeff)
		Zgr_ellipticity_corr = Z_ell (CGroupParams[Gr_Ell_ind,*], z_unwrap_coeff) - ze_drift[CGroupParams[9,*]]
		CGroupParams[Gr_Ell_ind,*] = Ell_Z (Zgr_ellipticity_corr, z_unwrap_coeff)
	endif
endif
GuideStarDrift[0].zdrift = zdrift

ReloadParamlists, Event, [Z_ind, SigZ_ind, GrZ_ind, GrSigZ_ind, Ell_ind, UnwZ_ind, UnwZ_Err_ind, Gr_Ell_ind, UnwGrZ_ind, UnwGrZ_Err_ind]

end
;
;-----------------------------------------------------------------
;
pro ExtractSubsetZ, Event, zdrift, use_multiple_GS	;Pulls out subset of data from param limits and fits z vs frames
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, thisfitcond, saved_pks_filenam
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, XYlimits, Use_XYlimits, LeaveOrigTotalRaw

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

WidDListDispFitMethodID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z_Fit_Method')
FitMethod = widget_info(WidDListDispFitMethodID,/DropList_Select)
!p.multi=[0,1,2,0,0]

GS_anc_file_info=FILE_INFO(GS_anc_fname)
use_multiple_GS = use_multiple_GS and GS_anc_file_info.exists
GSAnchorPnts=dblarr(2,AnchPnts_MaxNum)
GSAnchorPnts_line=dblarr(6)
if use_multiple_GS  then begin
	close,5
	openr,5,GS_anc_fname
	ip=0
	while not EOF(5) do begin
		readf,5,GSAnchorPnts_line
		GSAnchorPnts[0:1,ip] = GSAnchorPnts_line[0:1]
		ip+=1
	endwhile
	close,5
endif else begin
	GSAnchorPnts[0,0]= ParamLimits[2,2]
	GSAnchorPnts[1,0]= ParamLimits[3,2]
endelse

indecis=where((GSAnchorPnts[0,*] gt 0.001),ip_cnt)

ParamLimits0 = ParamLimits
filter0=filter

for jp=0,(ip_cnt-1) do begin
	filter=filter0
	if use_multiple_GS then begin
		ParamLimits[2,2] = GSAnchorPnts[0,jp]
		ParamLimits[2,0] = GSAnchorPnts[0,jp]-GS_radius
		ParamLimits[2,1] = GSAnchorPnts[0,jp]+GS_radius
		ParamLimits[3,2] = GSAnchorPnts[1,jp]
		ParamLimits[3,0] = GSAnchorPnts[1,jp]-GS_radius
		ParamLimits[3,1] = GSAnchorPnts[1,jp]+GS_radius
	endif

	FilterIt

	subsetindex=where(filter eq 1,cnt)
	print,jp,cnt
	print, 'Z- GuideStar subset has ',cnt,' points'
	if cnt gt 0 then begin
		;NFrames=long64(max(CGroupParams[9,*]))
		NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[9,*])+1))	: long64(max(CGroupParams[9,*])+1)
		subset=CGroupParams[*,subsetindex]
		FR=findgen(NFrames)

		if FitMethod eq 0 then begin
			WidSldFitOrderID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_Fit')
			widget_control,WidSldFitOrderID,get_value=fitorder
			zcoef=poly_fit(subset[9,*],subset[34,*],fitorder,YFIT=fit_to_x)
			ZFit=poly(FR,zcoef)
		endif else begin
			WidSmWidthID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_Sm_Width')
			widget_control,WidSmWidthID,get_value=SmWidth
			indecis=uniq(subset[9,*])
			Zsmooth=smooth(subset[6,indecis]*subset[34,indecis],SmWidth,/EDGE_TRUNCATE)/smooth(subset[6,indecis],SmWidth,/EDGE_TRUNCATE)
			Zfit=interpol(Zsmooth,subset[9,indecis],FR)
			frame_low = min(subset[9,indecis])
			ind_low=where(FR[*] lt frame_low,c)
			if c ge 1 then Zfit[ind_low]=Zfit[frame_low]
			frame_high = max(subset[9,indecis])
			ind_high=where(FR[*] gt frame_high,c)
			if c ge 1 then	Zfit[ind_high]=Zfit[frame_high]
		endelse
		zdrift=Zfit-Zfit[0]
		zdrift_mult = (jp eq 0) ? transpose(zdrift) : [zdrift_mult,transpose(zdrift)]

		if jp eq 0 then begin
			!P.NOERASE=0
			plot,FR,ZFit,xtitle='frame',ytitle='Z-DRIFT (nm)',xrange=Paramlimits[9,0:1],xstyle=1,yrange=Paramlimits[34,0:1],ystyle=1
			oplot,subset[9,*],subset[34,*],psym=3
			if FitMethod eq 0 then for i =0,fitorder do xyouts,0.8,0.95-0.02*i,zcoef[i],/normal
			if total(abs(subset[43,0:(100 < n_elements(zdrift))])) ne 0.0 then begin
				Z0_unwrap=Z_Ell(mean(subset[43,0:(100 < n_elements(zdrift))]),z_unwrap_coeff)
				Ellipt_Fit = Ell_Z ((Z0_unwrap+zdrift), z_unwrap_coeff)
				plot,FR,Ellipt_Fit,xtitle='frame',ytitle='X-Y Ellipticity',xrange=Paramlimits[9,0:1],xstyle=1,yrange=Paramlimits[43,0:1],ystyle=1
				oplot,subset[9,*],subset[43,*],psym=3
			endif
			!P.NOERASE=1
		endif else begin
			!p.multi=[0,1,2,0,0]
			plot,FR,ZFit,xtitle='frame',ytitle='Z-DRIFT (nm)',xrange=Paramlimits[9,0:1],xstyle=1,yrange=Paramlimits[34,0:1],ystyle=1
			col=(250-jp*50)>50
			oplot,FR,ZFit,col=col
			oplot,subset[9,*],subset[34,*],psym=3,col=col
			if FitMethod eq 0 then for i =0,fitorder do xyouts,0.8,0.95-0.02*i,zcoef[i],/normal
			if total(abs(subset[43,0:(100 < n_elements(zdrift))])) ne 0.0 then begin
				!p.multi=[1,1,2,0,0]
				Z0_unwrap=Z_Ell(mean(subset[43,0:(100 < n_elements(zdrift))]),z_unwrap_coeff)
				Ellipt_Fit = Ell_Z ((Z0_unwrap+zdrift), z_unwrap_coeff)
				plot,FR,Ellipt_Fit,xtitle='frame',ytitle='X-Y Ellipticity',xrange=Paramlimits[9,0:1],xstyle=1,yrange=Paramlimits[43,0:1],ystyle=1
				oplot,FR,Ellipt_Fit,col=col
				oplot,subset[9,*],subset[43,*],psym=3,col=col
			endif
		endelse
	endif
endfor

!p.multi=[0,0,0,0,0]
!p.background=0
!P.NOERASE=0

if use_multiple_GS then begin
	zdrift=total(zdrift_mult,1)/ip_cnt
	residual=dblarr(ip_cnt)
	for i=0,(ip_cnt-1) do residual[i]=max(zdrift_mult[i,*]-zdrift)-min(zdrift_mult[i,*]-zdrift)
	print,'Z- residual uncertainties (nm):   ',residual
	xyouts,0.12,0.96,'Z- residual uncertainties (nm):   ' + string(residual,FORMAT='(10(F8.2," "))'),color=200, charsize=1.5,CHARTHICK=1.0,/NORMAL
endif else begin
	max_drift=(max(zdrift)-min(zdrift))
	xyouts,0.12,0.96,'Z- drift (nm):   ' + string(max_drift,FORMAT='(F8.2)'),color=250, charsize=1.5,CHARTHICK=1.0,/NORMAL
endelse

ParamLimits = ParamLimits0
filter = filter0
return
end
;
;-----------------------------------------------------------------
;
pro OnRemoveTilt, Event
Remove_XYZ_tilt
;ReloadParamlists, Event, [34,35,40,41,44,45,47,48]
end
;
;-----------------------------------------------------------------
;
pro ReadWindPoint, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
WindRangeID = Widget_Info(Event.top, find_by_uname='WID_TEXT_WindPeriod')
wind_range_txt=string(wind_range,FORMAT='(F8.2)')
widget_control,WindRangeID,SET_VALUE = wind_range_txt
end
;
;-----------------------------------------------------------------
;
pro WriteWindPoint, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
WindRangeID = Widget_Info(Event.top, find_by_uname='WID_TEXT_WindPeriod')
widget_control,WindRangeID,GET_VALUE = wind_range_txt
new_range=float(wind_range_txt[0])
CGroupParams[34:35,*]=CGroupParams[34:35,*]*new_range/wind_range
CGroupParams[40:41,*]=CGroupParams[40:41,*]*new_range/wind_range
if CGrpSize ge 49 then begin
	CGroupParams[44,*]=CGroupParams[44,*]*new_range/wind_range
	CGroupParams[45,*]=CGroupParams[45,*]*new_range/wind_range
	CGroupParams[47,*]=CGroupParams[47,*]*new_range/wind_range
	CGroupParams[48,*]=CGroupParams[48,*]*new_range/wind_range
endif

; rescale z_unwrap_coeff
z_unwrap_coeff = z_unwrap_coeff * new_range / wind_range
EllipFitCoeff_WidID = Widget_Info(Event.top, find_by_uname='WID_TABLE_EllipticityFitCoeff')
widget_control,EllipFitCoeff_WidID,set_value=z_unwrap_coeff

; rescale look-up calibration scale if exists
if ((size(cal_lookup_zz))[2] eq 2) then cal_lookup_zz = cal_lookup_zz * new_range / wind_range

wind_range=new_range
ReloadParamlists, Event, [34,35,40,41,44,45,47,48]
end
;
;-----------------------------------------------------------------
;
pro WriteWindPointWithoutScaling, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
WindRangeID = Widget_Info(Event.top, find_by_uname='WID_TEXT_WindPeriod')
widget_control,WindRangeID,GET_VALUE = wind_range_txt
wind_range = float(wind_range_txt[0])
end
;
;-----------------------------------------------------------------
;
pro Update_Table_EllipticityFitCoeff, Event
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
EllipFitCoeff_WidID = Widget_Info(Event.top, find_by_uname='WID_TABLE_EllipticityFitCoeff')
widget_control,EllipFitCoeff_WidID,set_value=z_unwrap_coeff
end
;
;-----------------------------------------------------------------
;
pro On_Write_Zvalue, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
Zvalue_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Zvalue')
widget_control,Zvalue_ID,GET_VALUE = Zvalue_txt
Zvalue=float(Zvalue_txt[0])
Zsigma_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Zuncertainty')
widget_control,Zsigma_ID,GET_VALUE = Zsigma_txt
Zsigma=float(Zsigma_txt[0])
CGroupParams[34,*]=Zvalue
CGroupParams[40,*]=Zvalue
CGroupParams[35,*]=Zsigma
CGroupParams[41,*]=Zsigma
if CGrpSize ge 49 then begin
	CGroupParams[44,*]=Zvalue
	CGroupParams[47,*]=Zvalue
endif
ReloadParamlists, Event, [34,35,40,41,44,45,47,48]
end
;
;-----------------------------------------------------------------
;
function transform_extract_region,data,peakx,peaky,dim_xy
	if ((size(dim_xy))[2] eq 0) then dim_xy=5	; if half window size parameter is not supplied, make it 5
	dx=peakx-floor(peakx)-0.5
	dy=peaky-floor(peaky)-0.5
	TransformP=[[dx,0],[1,0]]
	TransformQ=[[dy,1],[0,0]]

	xsz=(size(data))[1]
	ysz=(size(data))[2]
	dim_xy2=2*dim_xy+1

	x0=(floor(peakx)-dim_xy+1)>0
	x1=x0+dim_xy2 < (xsz-2)
	x0=(x1-dim_xy2-1)>1
	x1=x0+dim_xy2+1
	y0=(floor(peaky)-dim_xy+1)>0
	y1=y0+dim_xy2 < (ysz-2)
	y0=(y1-dim_xy2-1)>1
	y1=y0+dim_xy2+1

	data_tr=POLY_2D(float(temporary(data[x0-1:x1+1,y0-1:y1+1])),TransformP,TransformQ,1)
	return,data_tr[1:dim_xy2,1:dim_xy2]
end
;
;-----------------------------------------------------------------
;
pro Analyze_cal_files_create_lookup_templates,dim_xy
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster

if ((size(nmperframe))[2] eq 0) then nmperframe=8.0
close,1
Fnames=[MLRawFilenames, RawFilenames[0]]
peakx=mean(CGroupParams[2,where(filter)])
peaky=mean(CGroupParams[3,where(filter)])
dim_xy2=2*dim_xy+1
ReadThisFitCond, (fnames[3]+'.txt'), pth, filen, ini_filename, thisfitcond
NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[9,*])+1))	: long64(max(CGroupParams[9,*])+1)
cal_lookup_zz=(findgen(NFrames)-NFrames/2.) * nmperframe * nd_oil/nd_water
mag=3.0
scl=0.08
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz
cal_lookup_data=fltarr(dim_xy2,dim_xy2,4,NFrames)
data=intarr(xsz,ysz)
orig_window = !D.WINDOW
xws=1024 & yws=1024
window,20,xs=xws,ys=yws
loadct,3

for k=0,3 do begin
	fname=fnames[k]
	ReadThisFitCond, (fname+'.txt'), pth, filen, ini_filename, thisfitcond
	openr,1,(fname+'.dat')
	for i=0,NFrames-1 do begin
		point_lun,1,2ull*xsz*ysz*i
		readu,1,data
   		cal_lookup_data[*,*,k,i]=(transform_extract_region(data,peakx,peaky,dim_xy)-thisfitcond.zerodark) / thisfitcond.cntpere > 0.
   	endfor
	close,1
endfor
;save, cal_lookup_data,cal_lookup_zz, nmperframe, lambda_vac, nd_water, nd_oil, filename=RawFilenames[0]+'_image_cal_data.sav'

scl0=254.0/max(cal_lookup_data[*,*,0:2,*])
scl1=254.0/max(cal_lookup_data[*,*,3,*])
pix_sz = mag*(dim_xy2+1)
npx = floor((xws-2*pix_sz)/pix_sz) < 31
npy = floor(Nframes/npx)
for j=0,npy do begin
	xyouts,5,25+j*5.0*pix_sz,'CCD1',CHARSIZE=1,/device
	xyouts,5,25+(j*5.0+1)*pix_sz,'CCD2',CHARSIZE=1,/device
	xyouts,5,25+(j*5.0+2)*pix_sz,'CCD3',CHARSIZE=1,/device
	xyouts,5,25+(j*5.0+3)*pix_sz,'SUM',CHARSIZE=1,/device
endfor
for k=0,3 do begin
	for i=0,NFrames-1 do begin
		if k eq 3 then scl=scl1 else scl=scl0
		x_pos = (i mod npx)*pix_sz+50
		y_pos =	floor((i-(i mod npx))/npx)*5.0*pix_sz+k*pix_sz+20
		tv,scl*rebin(cal_lookup_data[*,*,k,i],mag*dim_xy2,mag*dim_xy2,/sample),x_pos,y_pos
		if (k eq 0) and (i mod 2 eq 0) then begin
			str_out=string(round(cal_lookup_zz[i]),FORMAT='(I4)')
   			xyouts,x_pos,y_pos-15,str_out,CHARSIZE=0.8,/device
   		endif
   	endfor
endfor

if wfilename eq '' then return
presentimage=tvrd(true=1)
filename=AddExtension(wfilename,'_lookup_images.bmp')
write_bmp,filename,presentimage,/rgb
wset,orig_window
end
;
;-----------------------------------------------------------------
;
pro Select_Best_Lookup_Triplet, Z_int, peak_data, dim_xy, DisplayType, 	Z_lookup, Best_STD, wind_range, cal_lookup_data, cal_lookup_zz

wxsz=1024 & wysz=1024
dim_xy2=2*dim_xy+1
mag=4.0
pix_sz = mag*(dim_xy2+1)
scale_search_range=11			; number of points around the ppeak that are used to calculate scale
dxc=dim_xy2-scale_search_range

	n_possible=ceil((max(cal_lookup_zz)-min(cal_lookup_zz))/wind_range)
	z_possible=Z_int+wind_range*(findgen(2*n_possible)-n_possible)
	indicis=where((z_possible le max(cal_lookup_zz)) and (z_possible ge min(cal_lookup_zz)))
	z_possible=z_possible[indicis[uniq(indicis)]]
	if ((min(z_possible) - min(cal_lookup_zz)) ge (0.8 * wind_range)) then z_possible = [min(cal_lookup_zz),z_possible]
	if ((max(cal_lookup_zz) - max(z_possible)) ge (0.8 * wind_range)) then z_possible = [z_possible,max(cal_lookup_zz)]
	cnt = n_elements(z_possible)
	lookup_triplets=fltarr(dim_xy2,dim_xy2,4,cnt)
	diff_triplets=fltarr(dim_xy2,dim_xy2,4,cnt)
	st_devs=fltarr(3,cnt)

	for jj=0,cnt-1 do begin
		diff_zz=abs(cal_lookup_zz - z_possible[jj])
		trash=min(diff_zz,i_min)
		if (i_min eq 0) or (i_min eq n_elements(cal_lookup_zz)-1) then begin
			lookup_triplets[*,*,*,jj] = cal_lookup_data[*,*,*,i_min]
		endif else begin
			i_min_adj = (diff_zz[i_min-1] le diff_zz[i_min+1])	?	i_min-1	:	i_min+1
			rat=diff_zz[i_min_adj]/(diff_zz[i_min_adj]+diff_zz[i_min])
			lookup_triplets[*,*,*,jj] = cal_lookup_data[*,*,*,i_min]*rat+cal_lookup_data[*,*,*,i_min_adj]*(1.0-rat)
		endelse
		peak_center=peak_data[dxc:(dxc+scale_search_range-1),dxc:(dxc+scale_search_range-1),3]
		lookup_center = lookup_triplets[dxc:(dxc+scale_search_range-1),dxc:(dxc+scale_search_range-1),3,jj]
		scale=total(peak_center * lookup_center) / total((lookup_center)^2)					;  this scale minimizes the sum of (peak_data - scale * lookup_triplets)^2
		diff_triplets[*,*,*,jj]=abs(peak_data[*,*,*]- scale * lookup_triplets[*,*,*,jj])

		if DisplayType then begin
			xaxis=findgen(dim_xy2)-dim_xy2/2.0
			yaxis=findgen(dim_xy2)-dim_xy2/2.0
			tr=max(peak_data[*,*,3],ii_max)
			i_max=ii_max/dim_xy2
			j_max=ii_max-(ii_max/dim_xy2)*dim_xy2
			Peak_X_mag=peak_data[*,i_max,3]
			Lookup_X_mag=scale * lookup_triplets[*,i_max,3,jj]
			Peak_Y_mag=peak_data[j_max,*,3]
			Lookup_y_mag=scale * lookup_triplets[j_max,*,3,jj]
			x_pos = 50+8*pix_sz+40
			y_pos =	jj*4.5*pix_sz+50
			plot,xaxis,Lookup_X_mag,xtitle='X,Y coordinate (pix)',psym=5,ticklen=1,YCHARSIZE=1.2,/device,POSITION=[x_pos,y_pos,x_pos+3*pix_sz,y_pos+3*pix_sz]
			oplot,xaxis,Lookup_X_mag,linestyle=0
			oplot,xaxis,Peak_X_mag,psym=5,color=3000
			oplot,xaxis,Peak_X_mag,linestyle=0,color=3000
			oplot,yaxis,Lookup_Y_mag,psym=6
			oplot,yaxis,Lookup_Y_mag,linestyle=0
			oplot,yaxis,Peak_y_mag,psym=6,THICK=2,color=3000
			oplot,yaxis,Peak_y_mag,linestyle=0,color=3000
		endif

		for k=0,3 do begin
			x_pos = 50
			y_pos =	jj*4.5*pix_sz+k*pix_sz+20
			if DisplayType then begin
				if k eq 3 then scl=0.05 else scl=0.2
				tv,scl*rebin(lookup_triplets[*,*,k,jj],mag*dim_xy2,mag*dim_xy2,/sample),x_pos,y_pos
				tv,scl*rebin(peak_data[*,*,k] / scale,mag*dim_xy2,mag*dim_xy2,/sample),x_pos+pix_sz+10,y_pos
				tv,scl*rebin(diff_triplets[*,*,k,jj],mag*dim_xy2,mag*dim_xy2,/sample),x_pos+2*pix_sz+20,y_pos
			endif
			if k lt 3 then begin

				st_devs[k,jj]=sqrt(total((diff_triplets[*,*,k,jj])^2));/sqrt(total(lookup_triplets[*,*,3,jj]))
				if DisplayType then begin
					str_out=string(st_devs[k,jj],FORMAT='(E12.2)')
					xyouts,x_pos+3*pix_sz+40,y_pos,str_out,CHARSIZE=1.5,/device
				endif
			endif else begin
				if DisplayType then begin
					str_out=string(total(st_devs[*,jj]),FORMAT='(E12.2)')
					xyouts,x_pos+3*pix_sz+40,y_pos,str_out,CHARSIZE=1.5,/device
				endif
			endelse
		endfor
	endfor
	; find best fit between the image triplet and the cal data
	criterion=total(st_devs,1)
	Best_STD=min(criterion,best_fit_ind) / sqrt(total(lookup_triplets[*,*,3,best_fit_ind]))
	Z_lookup = z_possible[best_fit_ind]
	if DisplayType then xyouts,800,best_fit_ind*4.5*pix_sz+50,'Best Fit',CHARSIZE=2.0,/device
end
;
;-----------------------------------------------------------------
;
pro LookupUnwrapZCoord, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
if (total(z_unwrap_coeff[*]) eq 0) then return
wxsz=1024 & wysz=1024
Zindex = max(where(RowNames[*] eq 'Z Position'))
UnwZindex = max(where(RowNames[*] eq 'Unwrapped Z'))
GrZindex = max(where(RowNames[*] eq 'Group Z Position'))
UnwGrZindex = max(where(RowNames[*] eq 'Unwrapped Group Z'))
Z_params=[Zindex,UnwZindex,GrZindex,UnwGrZindex]
Fnames=[MLRawFilenames, RawFilenames[0]]

dim_xy=5
dim_xy2=2*dim_xy+1
mag=4.0
pix_sz = mag*(dim_xy2+1)

close,1
ReadThisFitCond, (RawFilenames[0]+'.txt'), pth, filen, ini_filename, thisfitcond
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz
thisfitconds=replicate(thisfitcond,4)
sz=size(CGroupParams)
sz_disp=fix(sz[2]/100)
indecis=where((CGroupParams[2,*] gt (dim_xy+1)) and (CGroupParams[2,*] lt (xsz-dim_xy-2)) and (CGroupParams[3,*] gt (dim_xy+1)) and (CGroupParams[3,*] lt (ysz-dim_xy-2)),peak_cnt)

peak_data=fltarr(dim_xy2,dim_xy2,4)
data=intarr(xsz,ysz)
npx = floor((wxsz-2*pix_sz)/pix_sz) < 31

Lookup_Unwrap_Display_id=widget_info(event.top,FIND_BY_UNAME='WID_DROPLIST_LookupUnwrapDisplayType')
DispType=widget_info(Lookup_Unwrap_Display_id,/DropList_Select)	;set to 0 - Local-No Display; 1 - Local - with display;  2 - Cluster - No Display
DisplayType = (DispType eq 1)	?	1	:	0

if DispType le 1 then LookupUnwrapZCoord_local, dim_xy, DisplayType

if DispType eq 2 then begin
	LookupUnwrapZCoord_cluster, dim_xy
endif

nZpar=n_elements(Z_params)-1
for j=0,nZpar do ParamLimits[Z_params[j],2]=Median(CGroupParams[Z_params[j],*])
ParamLimits[41,2]=(ParamLimits[41,2]>ParamLimits[35,2])
for j=0,nZpar do ParamLimits[Z_params[j],0]= min(CGroupParams[Z_params[j],*])
for j=0,nZpar do ParamLimits[Z_params[j],1]=(20.*ParamLimits[Z_params[j],2] < max(CGroupParams[Z_params[j],*]))
for j=0,nZpar do ParamLimits[Z_params[j],3]=ParamLimits[Z_params[j],1] - ParamLimits[Z_params[j],0]
for j=0,nZpar do ParamLimits[Z_params[j],2]=(ParamLimits[Z_params[j],1] + ParamLimits[Z_params[j],0])/2.

if CGrpSize ge 49 then begin
	for j=43,48 do begin
		ParamLimits[j,0] = min(CGroupParams[j,*])
		ParamLimits[j,1] = max(CGroupParams[j,*])
		ParamLimits[j,3]=ParamLimits[j,1] - ParamLimits[j,0]
		ParamLimits[j,2]=(ParamLimits[j,1] + ParamLimits[j,0])/2.
	endfor
endif

TopIndex = (CGrpSize-1) > 41
TopID=ids[where(names eq 'WID_BASE_0_PeakSelector')]
wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(ParamLimits[0:TopIndex,0:3]), use_table_select=[0,0,3,TopIndex]
widget_control,wtable,set_value=transpose(CGroupParams[0:TopIndex,sz[2]/2]), use_table_select=[4,0,4,TopIndex]
widget_control, wtable, /editable,/sensitive

if DisplayType then !p.noerase=0
for k=0,3 do close,k+5
end
;
;-----------------------------------------------------------------
;
pro LookupUnwrapZCoord_local, dim_xy, DisplayType
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
if (total(z_unwrap_coeff[*]) eq 0) then return
wxsz=1024 & wysz=1024
Zindex = max(where(RowNames[*] eq 'Z Position'))
UnwZindex = max(where(RowNames[*] eq 'Unwrapped Z'))
GrZindex = max(where(RowNames[*] eq 'Group Z Position'))
UnwGrZindex = max(where(RowNames[*] eq 'Unwrapped Group Z'))
Z_params=[Zindex,UnwZindex,GrZindex,UnwGrZindex]
Fnames=[MLRawFilenames, RawFilenames[0]]

dim_xy2=2*dim_xy+1
mag=4.0
pix_sz = mag*(dim_xy2+1)

close,1
ReadThisFitCond, (RawFilenames[0]+'.txt'), pth, filen, ini_filename, thisfitcond
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz
thisfitconds=replicate(thisfitcond,4)
sz=size(CGroupParams)
sz_disp=fix(sz[2]/100)
indecis=where((CGroupParams[2,*] gt (dim_xy+2)) and $
				(CGroupParams[2,*] lt (xsz-dim_xy-3)) and $
				(CGroupParams[3,*] gt (dim_xy+2)) and $
				(CGroupParams[3,*] lt (ysz-dim_xy-3)),peak_cnt)

peak_data=fltarr(dim_xy2,dim_xy2,4)
data=intarr(xsz,ysz)
npx = floor((wxsz-2*pix_sz)/pix_sz) < 31

for k=0,3 do begin
	close,k+5
	fname=fnames[k]
	ReadThisFitCond,(fname+'.txt'), pth, filen, ini_filename, thisfitcond
	thisfitconds[k]=thisfitcond
	openr,k+5,(fname+'.dat')
	for jj=0,(n_elements(cal_lookup_zz)-1) do begin
	   		perimeter=[cal_lookup_data[0:(dim_xy2-1),0,k,jj],	cal_lookup_data[0:(dim_xy2-1),(dim_xy2-1),k,jj],		$
   			transpose(cal_lookup_data[0,1:(dim_xy2-2),k,jj]),	transpose(cal_lookup_data[(dim_xy2-1),1:(dim_xy2-2),k,jj])]
   			mean_perimeter=mean(perimeter)
   			not_outliers=where(abs((perimeter-mean_perimeter)/mean_perimeter)  le 3)
   			perimeter_offset=mean(perimeter[not_outliers])
   			;perimeter_stdev=stdev(perimeter[not_outliers])
   			cal_lookup_data[*,*,k,jj]=(cal_lookup_data[*,*,k,jj]-perimeter_offset) > 0.0
   	endfor
endfor

for i = 0l,(peak_cnt-1) do begin
	;  form image triplets for the peak
	for k=0,3 do begin
		point_lun,k+5,2ull*xsz*ysz*CGroupParams[9,indecis[i]]
		readu,k+5,data
		peak_data[*,*,k]=(transform_extract_region(data,CGroupParams[2,indecis[i]],CGroupParams[3,indecis[i]],dim_xy)-thisfitconds[k].zerodark) / thisfitconds[k].cntpere
		perimeter=[peak_data[0:(dim_xy2-1),0,k],	peak_data[0:(dim_xy2-1),(dim_xy2-1),k],		$
   				transpose(peak_data[0,1:(dim_xy2-2),k]),	transpose(peak_data[(dim_xy2-1),1:(dim_xy2-2),k])]
		mean_perimeter=mean(perimeter)
		not_outliers=where(abs((perimeter-mean_perimeter)/mean_perimeter)  le 3)
		perimeter_offset=mean(perimeter[not_outliers])
		;perimeter_stdev=stdev(perimeter[not_outliers])
		peak_data[*,*,k]=(peak_data[*,*,k] - perimeter_offset) > 0.0
	endfor
   	if DisplayType then begin
		erase,0
		!p.noerase=1
	endif
		;  pick the possible matching image triplets from the cal data
	Z_int=CGroupParams[Zindex,indecis[i]]
	Select_Best_Lookup_Triplet, Z_int, peak_data, dim_xy, DisplayType, 	Z_lookup, Best_STD, wind_range, cal_lookup_data, cal_lookup_zz
	CGroupParams[UnwZindex,indecis[i]]= CGroupParams[Zindex,indecis[i]] + round((Z_lookup - CGroupParams[Zindex,indecis[i]])/wind_range) * wind_range
	CGroupParams[45,i] = Best_STD
	if (i mod sz_disp) eq 0 then print,'peak#',i,'   z-data:  ',CGroupParams[Zindex,indecis[i]],CGroupParams[UnwZindex,indecis[i]]
endfor
catch,/cancel

if max(CGroupParams[18,*]) gt 0 then begin		; if the grouping was perfromed  perform Z-extraction for grpous
	for i = 0l,(peak_cnt-1) do begin
		if CGroupParams[24,indecis[i]] eq 1 then begin			; the group consists of a single element
			CGroupParams[GrZindex,indecis[i]]=CGroupParams[Zindex,indecis[i]]
			CGroupParams[41,indecis[i]]=CGroupParams[35,indecis[i]]
			CGroupParams[42,indecis[i]]=CGroupParams[36,indecis[i]]
			CGroupParams[UnwGrZindex,indecis[i]] = CGroupParams[UnwZindex,indecis[i]]
			CGroupParams[48,indecis[i]] = CGroupParams[45,indecis[i]]
		endif else begin
			if CGroupParams[25,indecis[i]] eq 1 then begin	; perform group analysis only if we encounter the second group element, then set all group element values to it.
				; form image triplets for the group of peaks (read all peak images in group and sum them up)
				gr_indices=where((CGroupParams[18,*] eq CGroupParams[18,indecis[i]]),gr_cnt)
				for k=0,3 do begin
					for ig=0,gr_cnt-1 do begin
						point_lun,k+5,2ull*xsz*ysz*CGroupParams[9,indecis[i]]
						readu,k+5,data
						if ig eq 0 then begin
							peak_data[*,*,k]=(transform_extract_region(data,CGroupParams[2,indecis[i]],CGroupParams[3,indecis[i]],dim_xy)-thisfitconds[k].zerodark) / thisfitconds[k].cntpere
						endif else begin
							peak_data[*,*,k]= peak_data[*,*,k] + (transform_extract_region(data,CGroupParams[2,indecis[i]],CGroupParams[3,indecis[i]],dim_xy)-thisfitconds[k].zerodark) / thisfitconds[k].cntpere
						endelse
					endfor
					perimeter=[peak_data[0:(dim_xy2-1),0,k],	peak_data[0:(dim_xy2-1),(dim_xy2-1),k],		$
   							transpose(peak_data[0,1:(dim_xy2-2),k]),	transpose(peak_data[(dim_xy2-1),1:(dim_xy2-2),k])]
					mean_perimeter=mean(perimeter)
					not_outliers=where(abs((perimeter-mean_perimeter)/mean_perimeter)  le 3)
					perimeter_offset=mean(perimeter[not_outliers])

					;perimeter_stdev=stdev(perimeter[not_outliers])
					peak_data[*,*,k]=(peak_data[*,*,k] - perimeter_offset) > 0.0
				endfor

   				if DisplayType then begin
					erase,0
					!p.noerase=1
				endif
				;  pick the possible matching image triplets from the cal data
				Z_int=CGroupParams[GrZindex,indecis[i]]
				Select_Best_Lookup_Triplet, Z_int, peak_data, dim_xy, DisplayType, 	Z_lookup, Best_STD, wind_range, cal_lookup_data, cal_lookup_zz
				CGroupParams[UnwGrZindex,gr_indices] = CGroupParams[GrZindex,indecis[i]] + round((Z_lookup - CGroupParams[GrZindex,indecis[i]])/wind_range) * wind_range
				CGroupParams[48,gr_indices] = Best_STD
			endif
		endelse
		if (i mod sz_disp) eq 0 then print,'gr. peak#',i,'   z-data:  ',CGroupParams[GrZindex,indecis[i]],CGroupParams[UnwGrZindex,indecis[i]]
	endfor
endif
catch,/cancel
end
;
;-----------------------------------------------------------------
;
pro LookupUnwrapZCoord_cluster, dim_xy
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster

DisplayType=0

Zindex = max(where(RowNames[*] eq 'Z Position'))
UnwZindex = max(where(RowNames[*] eq 'Unwrapped Z'))
GrZindex = max(where(RowNames[*] eq 'Group Z Position'))
UnwGrZindex = max(where(RowNames[*] eq 'Unwrapped Group Z'))
Z_params=[Zindex,UnwZindex,GrZindex,UnwGrZindex]
Fnames=[MLRawFilenames, RawFilenames[0]]
dim_xy2=2*dim_xy+1

close,1
ReadThisFitCond, (RawFilenames[0]+'.txt'), pth, filen, ini_filename, thisfitcond
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz
thisfitconds=replicate(thisfitcond,4)
CGPsz=size(CGroupParams)
sz_disp=fix(CGPsz[2]/100)
indecis=where((CGroupParams[2,*] gt (dim_xy+1)) and (CGroupParams[2,*] lt (xsz-dim_xy-2)) and (CGroupParams[3,*] gt (dim_xy+1)) and (CGroupParams[3,*] lt (ysz-dim_xy-2)),peak_cnt)
framefirst=min(CGroupParams[9,indecis])
framelast=max(CGroupParams[9,indecis])
increment=2500
nloops=long(ceil((framelast-framefirst+1.0)/increment))

for k=0,3 do begin
	close,k+5
	fname=fnames[k]
	ReadThisFitCond,(fname+'.txt'), pth, filen, ini_filename, thisfitcond
	thisfitconds[k]=thisfitcond
	openr,k+5,(fname+'.dat')
	for jj=0,(n_elements(cal_lookup_zz)-1) do begin
	   		perimeter=[cal_lookup_data[0:(dim_xy2-1),0,k,jj],	cal_lookup_data[0:(dim_xy2-1),(dim_xy2-1),k,jj],		$
   			transpose(cal_lookup_data[0,1:(dim_xy2-2),k,jj]),	transpose(cal_lookup_data[(dim_xy2-1),1:(dim_xy2-2),k,jj])]
   			mean_perimeter=mean(perimeter)
   			not_outliers=where(abs((perimeter-mean_perimeter)/mean_perimeter)  le 3)
   			perimeter_offset=mean(perimeter[not_outliers])
   			;perimeter_stdev=stdev(perimeter[not_outliers])
   			cal_lookup_data[*,*,k,jj]=(cal_lookup_data[*,*,k,jj]-perimeter_offset) > 0.0
   	endfor
endfor

if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else	idl_pwd=pref_get('IDL_WDE_START_DIR')
cd,current=curr_pwd
FILE_MKDIR,curr_pwd+'/temp'
save, curr_pwd, idl_pwd, ini_filename, CGroupParams, CGrpSize, ParamLimits, indecis, Fnames, RowNames, increment, nloops, $
		cal_lookup_data, cal_lookup_zz, wind_range, nmperframe, dim_xy, filename='temp/temp.sav'		;save variables for cluster cpu access
spawn,'sync'
spawn,'sync'
spawn,'sh '+idl_pwd+'/LookupUnwrapZ_runme.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd			;Spawn grouping workers in cluster

for nlps=0L,nloops-1 do begin			;reassemble little pks files from all the workers into on big one
	framestart=	framefirst + (nlps)*increment						;first frame in batch
	framestop=(framefirst + (nlps+1L)*increment-1)<framelast
	test1=file_info(curr_pwd+'/temp/LookupZUnwr_parameters_'+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)+'_cgr.par')
	if ~test1.exists then begin
		print,'Did not found file:',test1.name
		stop
	endif
	restore,filename=curr_pwd+'/temp/LookupZUnwr_parameters_'+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)+'_cgr.par'
	if n_elements(CGroupParamsUnwZ) gt 2 then begin
		current_indecis=where((CGroupParams[2,*] gt (dim_xy+2)) and $
				(CGroupParams[2,*] lt (xsz-dim_xy-3)) and $
				(CGroupParams[3,*] gt (dim_xy+2)) and $
				(CGroupParams[3,*] lt (ysz-dim_xy-3)) and $
				(CGroupParams[9,*] ge framestart) and $
				(CGroupParams[9,*] le framestop),peak_cnt)
		test2=size(CGroupParams[*,current_indecis])
		test3=size(CGroupParamsUnwZ)
		if test2[2] ne test3[2] then begin
			print,' The dimensions of CGroupParams[*,indecis] and CGroupParamsUnwZ do not agree:'
			print,'CGroupParams[*,current_indecis]:', test2
			print,'CGroupParamsUnwZ:', test3
			stop
		endif
		CGroupParams[*,current_indecis]=CGroupParamsUnwZ
	endif
	file_delete,curr_pwd+'/temp/LookupZUnwr_parameters_'+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)+'_cgr.par'
endfor

file_delete,'temp/temp.sav'
file_delete,'temp'
cd,curr_pwd
end
;
;-----------------------------------------------------------------
;
pro LookupUnwrapZ_Worker, nlps,data_dir						;spawn mulitple copies of this programs for cluster
Nlps=long((COMMAND_LINE_ARGS())[0])
data_dir=(COMMAND_LINE_ARGS())[1]
cd,data_dir
print,'restoring:   ',data_dir,'/temp/temp.sav'
restore,'temp/temp.sav'
print,'restore complete'

Zindex = max(where(RowNames[*] eq 'Z Position'))
UnwZindex = max(where(RowNames[*] eq 'Unwrapped Z'))
GrZindex = max(where(RowNames[*] eq 'Group Z Position'))
UnwGrZindex = max(where(RowNames[*] eq 'Unwrapped Group Z'))
Z_params=[Zindex,UnwZindex,GrZindex,UnwGrZindex]
DisplayType=0
dim_xy2=2*dim_xy+1

ReadThisFitCond, (Fnames[3]+'.txt'), pth, filen, ini_filename, thisfitcond

xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz
thisfitconds=replicate(thisfitcond,4)
sz=size(CGroupParams)
sz_disp=fix(sz[2]/100)

peak_data=fltarr(dim_xy2,dim_xy2,4)
data=intarr(xsz,ysz)
framefirst=min(CGroupParams[9,indecis])
framelast=max(CGroupParams[9,indecis])
framestart =	framefirst + nlps*increment						;first frame in batch
framestop =(framefirst + (Nlps+1L)*increment-1L)<framelast
current_indecis=where((CGroupParams[2,*] gt (dim_xy+2)) and $
				(CGroupParams[2,*] lt (xsz-dim_xy-3)) and $
				(CGroupParams[3,*] gt (dim_xy+2)) and $
				(CGroupParams[3,*] lt (ysz-dim_xy-3)) and $
				(CGroupParams[9,*] ge framestart) and $
				(CGroupParams[9,*] le framestop),peak_cnt)
i=0
catch, Error_status
IF Error_status NE 0 THEN BEGIN
	PRINT, 'Error index: ', Error_status
	PRINT, 'Error message: ', !ERROR_STATE.MSG
	print,CGroupParams[2,current_indecis[i]],CGroupParams[3,current_indecis[i]]
	stop
ENDIF
for k=0,3 do begin
	close,k+5
	fname=fnames[k]
	ReadThisFitCond,(fname+'.txt'), pth, filen, ini_filename, thisfitcond
	thisfitconds[k]=thisfitcond
	openr,k+5,(fname+'.dat')
endfor
print,'step 0: ',framestart,framestop,increment,nlps, xsz,ysz,peak_cnt
for i = 0l,(peak_cnt-1) do begin
	;form image triplets for the peak'
	for k=0,3 do begin
		point_lun,k+5,2ull*xsz*ysz*CGroupParams[9,current_indecis[i]]
		readu,k+5,data
		peak_data[*,*,k]=(transform_extract_region(data,CGroupParams[2,current_indecis[i]],CGroupParams[3,current_indecis[i]],dim_xy)-thisfitconds[k].zerodark) / thisfitconds[k].cntpere
		perimeter=[peak_data[0:(dim_xy2-1),0,k],	peak_data[0:(dim_xy2-1),(dim_xy2-1),k],		$
   				transpose(peak_data[0,1:(dim_xy2-2),k]),	transpose(peak_data[(dim_xy2-1),1:(dim_xy2-2),k])]
		mean_perimeter=mean(perimeter)
		not_outliers=where(abs((perimeter-mean_perimeter)/mean_perimeter)  le 3)
		perimeter_offset=mean(perimeter[not_outliers])
		;perimeter_stdev=stdev(perimeter[not_outliers])
		peak_data[*,*,k]=(peak_data[*,*,k] - perimeter_offset) > 0.0
	endfor
    ;  pick the possible matching image triplets from the cal data
	Z_int=CGroupParams[Zindex,current_indecis[i]]
	Select_Best_Lookup_Triplet, Z_int, peak_data, dim_xy, DisplayType, 	Z_lookup, Best_STD, wind_range, cal_lookup_data, cal_lookup_zz
	CGroupParams[UnwZindex,current_indecis[i]]= CGroupParams[Zindex,current_indecis[i]] + round((Z_lookup - CGroupParams[Zindex,current_indecis[i]])/wind_range) * wind_range
	CGroupParams[45,i] = Best_STD
	if (i mod sz_disp) eq 0 then	print,'peak#',i,'   z-data:  ',CGroupParams[Zindex,current_indecis[i]],CGroupParams[UnwZindex,current_indecis[i]]
endfor

if max(CGroupParams[18,*]) gt 0 then begin		; if the grouping was perfromed  perform Z-extraction for grpous
	for i = 0l,(peak_cnt-1) do begin
		if CGroupParams[24,current_indecis[i]] eq 1 then begin			; the group consists of a single element
			CGroupParams[GrZindex,current_indecis[i]]=CGroupParams[Zindex,current_indecis[i]]
			CGroupParams[41,current_indecis[i]]=CGroupParams[35,current_indecis[i]]
			CGroupParams[42,current_indecis[i]]=CGroupParams[36,current_indecis[i]]
			CGroupParams[UnwGrZindex,current_indecis[i]] = CGroupParams[UnwZindex,current_indecis[i]]
			CGroupParams[48,current_indecis[i]] = CGroupParams[45,current_indecis[i]]
		endif else begin
			if CGroupParams[25,current_indecis[i]] eq 1 then begin	; perform group analysis only if we encounter the second group element, then set all group element values to it.
				; form image triplets for the group of peaks (read all peak images in group and sum them up)
				gr_indices=where((CGroupParams[18,*] eq CGroupParams[18,current_indecis[i]]),gr_cnt)
				for k=0,3 do begin
					for ig=0,gr_cnt-1 do begin
						point_lun,k+5,2ull*xsz*ysz*CGroupParams[9,current_indecis[i]]
						readu,k+5,data
						if ig eq 0 then begin
							peak_data[*,*,k]=(transform_extract_region(data,CGroupParams[2,current_indecis[i]],CGroupParams[3,current_indecis[i]],dim_xy)-thisfitconds[k].zerodark) / thisfitconds[k].cntpere
						endif else begin
							peak_data[*,*,k]= peak_data[*,*,k] + (transform_extract_region(data,CGroupParams[2,current_indecis[i]],CGroupParams[3,current_indecis[i]],dim_xy)-thisfitconds[k].zerodark) / thisfitconds[k].cntpere
						endelse
					endfor
					perimeter=[peak_data[0:(dim_xy2-1),0,k],	peak_data[0:(dim_xy2-1),(dim_xy2-1),k],		$
   							transpose(peak_data[0,1:(dim_xy2-2),k]),	transpose(peak_data[(dim_xy2-1),1:(dim_xy2-2),k])]
					mean_perimeter=mean(perimeter)
					not_outliers=where(abs((perimeter-mean_perimeter)/mean_perimeter)  le 3)
					perimeter_offset=mean(perimeter[not_outliers])

					;perimeter_stdev=stdev(perimeter[not_outliers])
					peak_data[*,*,k]=(peak_data[*,*,k] - perimeter_offset) > 0.0
				endfor

				;  pick the possible matching image triplets from the cal data
				Z_int=CGroupParams[GrZindex,current_indecis[i]]
			   	Select_Best_Lookup_Triplet, Z_int, peak_data, dim_xy, DisplayType, 	Z_lookup, Best_STD, wind_range, cal_lookup_data, cal_lookup_zz
				CGroupParams[UnwGrZindex,gr_indices] = CGroupParams[GrZindex,current_indecis[i]] + round((Z_lookup - CGroupParams[GrZindex,current_indecis[i]])/wind_range) * wind_range
				CGroupParams[48,gr_indices] = Best_STD
			endif
		endelse
		if (i mod sz_disp) eq 0 then		print,'gr. peak#',i,'   z-data:  ',CGroupParams[GrZindex,current_indecis[i]],CGroupParams[UnwGrZindex,current_indecis[i]]
	endfor
endif
for k=0,3 do close,k+5
CGroupParamsUnwZ=CGroupParams[*,current_indecis]
save,CGroupParamsUnwZ,filename=data_dir+'/temp/LookupZUnwr_parameters_'+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)+'_cgr.par'
spawn,'sync'
spawn,'sync'
print,'Wrote file '+data_dir+'/temp/LookupZUnwr_parameters_'+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)+'_cgr.par'
return
end
