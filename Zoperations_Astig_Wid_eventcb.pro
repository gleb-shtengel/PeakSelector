;
; Empty stub procedure used for autoloading.
;
pro Zoperations_Astig_Wid_eventcb
end
;
;-----------------------------------------------------------------
;
; thuis funcyion will be used when we throw away non-unique peaks within a single frame for each fiducial
function uniq_first_elements, X
	return, reverse(n_elements(X)-uniq(reverse(X))-1)
end
;
;-----------------------------------------------------------------
;
pro Initialize_Z_operations_Astig, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
WidLabel0 = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,WidLabel0,get_value=dfilename
droot = strmid(dfilename,0,strpos(dfilename,sep,/REVERSE_OFFSET,/REVERSE_SEARCH))
ddroot = strmid(droot,0,strpos(droot,sep,/REVERSE_OFFSET,/REVERSE_SEARCH))

WIDID_TEXT_ZCalStep = Widget_Info(wWidget, find_by_uname='WID_TEXT_ZCalStep_Astig')
nmperframe_txt=string(nmperframe,FORMAT='(F8.2)')
widget_control,WIDID_TEXT_ZCalStep,SET_VALUE = nmperframe_txt

WID_TEXT_Zmin_Astig_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Zmin_Astig')
z_cal_min_txt=string(z_cal_min,FORMAT='(F8.2)')
widget_control,WID_TEXT_Zmin_Astig_ID,SET_VALUE = z_cal_min_txt

WID_TEXT_Zmax_Astig_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Zmax_Astig')
z_cal_max_txt=string(z_cal_max,FORMAT='(F8.2)')
widget_control,WID_TEXT_Zmax_Astig_ID,SET_VALUE = z_cal_max_txt

num_iter = 3
WID_TEXT_ZCal_Astig_num_iter_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_ZCal_Astig_num_iter')
num_iter_txt=string(num_iter,FORMAT='(I2)')
widget_control,WID_TEXT_ZCal_Astig_num_iter_ID,SET_VALUE = num_iter_txt


WFileWid_astig_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_WindFilename_Astig')
suggested_windfile = 'Calibration_Curve_F1_WND.sav'
if dfilename ne '' then begin
	dot_pos0=strpos(dfilename,'IDL.sav',/REVERSE_OFFSET,/REVERSE_SEARCH)
	dot_pos1=strpos(dfilename,'IDL.pks',/REVERSE_OFFSET,/REVERSE_SEARCH)
	dot_pos = dot_pos0 > dot_pos1
	if dot_pos gt 0 then suggested_windfile=strmid(dfilename,0,dot_pos)+'F1_WND.sav'
endif
widget_control,WFileWid_astig_ID,SET_VALUE = suggested_windfile

if (size(wfilename))[2] gt 0 then begin
	wroot = strmid(wfilename,0,strpos(wfilename,sep,/REVERSE_OFFSET,/REVERSE_SEARCH))
	wwroot = strmid(wroot,0,strpos(wroot,sep,/REVERSE_OFFSET,/REVERSE_SEARCH))
	if (wfilename ne '') and ((wwroot eq ddroot) or (wwroot eq droot)) then begin
		widget_control,WFileWid_astig_ID,SET_VALUE = wfilename
		wfile_info=FILE_INFO(wfilename)
		if wfile_info.exists then  begin
			restore,filename=wfilename
		endif
	endif
endif

FitmethodID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Z_Fit_Method')
widget_control,FitmethodID,SET_DROPLIST_SELECT=1

WPhaseSliderID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Z_phase_offset_Astig')
widget_control,WPhaseSliderID,SET_VALUE = 0

WID_TEXT_GuideStar_Radius_Astig_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_GuideStar_Radius_Astig')
GS_radius_txt=string(GS_radius[0],FORMAT='(F6.2)')
widget_control,WID_TEXT_GuideStar_Radius_Astig_ID,SET_VALUE = GS_radius_txt
WID_TEXT_GuideStarAncFilename_Astig_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_GuideStarAncFilename_Astig')
widget_control,WID_TEXT_GuideStarAncFilename_Astig_ID,SET_VALUE = GS_anc_fname

WIDID_TEXT_ZCalStep = Widget_Info(wWidget, find_by_uname='WID_TEXT_ZCalStep_Astig')
nmperframe_txt=string(nmperframe[0],FORMAT='(F6.2)')
widget_control,WIDID_TEXT_ZCalStep,SET_VALUE = nmperframe_txt

end
;
;-----------------------------------------------------------------
;
pro OnButton_Press_use_multiple_GS, Event
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
	use_multiple_GS_DH_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs_DH')
	if use_multiple_GS_DH_id gt 0 then if Event.select then widget_control, use_multiple_GS_DH_id, set_button=0
end
;
;-----------------------------------------------------------------
;
pro OnButton_Press_use_multiple_GS_DH, Event
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
	use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs')
	if use_multiple_GS_id gt 0 then if Event.select then widget_control, use_multiple_GS_id, set_button=0
end
;
;-----------------------------------------------------------------
;
function find_center_frame, subset, fitorder, frame_range
	common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
	Frame_Number = min(where(RowNames eq 'Frame Number'))
	Nph_ind = min(where(RowNames eq '6 N Photons'))                            ; CGroupParametersGP[6,*] - Number of Photons in the Peak
	; step1: find the frame with near max amplitude
	center_fr0 = total(subset[Frame_Number,*] * subset[Nph_ind,*])/total(subset[Nph_ind,*])
	subset_ind0 = where((subset[Frame_Number,*] gt (center_fr0 - frame_range/2)) and (subset[Frame_Number,*] lt (center_fr0 + frame_range/2)))
	min_ell0 = min(abs(subset[Ell_ind,subset_ind0]), center_fr_ind0)
	center_fr1 = subset[Frame_Number,subset_ind0[center_fr_ind0]]
	; find a frame near the previously found center with ellipticity closest to 0
	subset_ind1 = where((subset[Frame_Number,*] gt (center_fr1 - frame_range/2)) and (subset[Frame_Number,*] lt (center_fr1 + frame_range/2)))
	min_ell = min(abs(subset[Ell_ind,subset_ind1]), center_fr_ind1)
	center_fr = subset[Frame_Number,subset_ind1[center_fr_ind1]]
	print,'center frame: step0:', center_fr0,', step 1:', center_fr1, ', step 2:', center_fr
	return, center_fr
end

;
;-----------------------------------------------------------------
;
Pro ExtractEllipticityCalib_Astig, Event		; perform z-calibration on filtered data set
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

WIDID_TEXT_ZCalStep = Widget_Info(Event.top, find_by_uname='WID_TEXT_ZCalStep_Astig')
widget_control,WIDID_TEXT_ZCalStep,GET_VALUE = nmperframe_txt
nmperframe = float(nmperframe_txt[0])

WID_TEXT_Zmin_Astig_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Zmin_Astig')
widget_control,WID_TEXT_Zmin_Astig_ID,GET_VALUE = z_cal_min_txt
z_cal_min = float(z_cal_min_txt[0])

WID_TEXT_Zmax_Astig_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Zmax_Astig')
widget_control,WID_TEXT_Zmax_Astig_ID,GET_VALUE = z_cal_max_txt
z_cal_max = float(z_cal_max_txt[0])

WID_TEXT_ZCal_Astig_num_iter_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_ZCal_Astig_num_iter')
widget_control,WID_TEXT_ZCal_Astig_num_iter_ID,GET_VALUE = num_iter_txt
num_iter = float(num_iter_txt[0])

Off_ind = min(where(RowNames eq 'Offset'))                                ; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))                            ; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))                            ; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))                            ; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))                        ; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))                        ; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))                            ; CGroupParametersGP[6,*] - Number of Photons in the Peak
Chi_ind = min(where(RowNames eq 'ChiSquared'))                            ; CGroupParametersGP[7,*] - Chi Squared
FitOK_ind = min(where(RowNames eq 'FitOK'))                                ; CGroupParametersGP[8,*] - Original FitOK
FrNum_ind = min(where(RowNames eq 'Frame Number'))                        ; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))                    ; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigAmp_ind = min(where(RowNames eq 'Sigma Amplitude'))
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))                ; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))                ; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))                    ; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))                    ; CGroupParametersGP[17,*] - y - sigma

Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size = min(where(RowNames eq '24 Group Size'))

sz=size(CGroupParams)

use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs')
if use_multiple_GS_id gt 0 then use_multiple_GS = widget_info(use_multiple_GS_id,/button_set) else use_multiple_GS=0
use_multiple_GS_DH_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs_DH')
if use_multiple_GS_DH_id gt 0 then use_multiple_GS_DH = widget_info(use_multiple_GS_DH_id,/button_set) else use_multiple_GS_DH=0
use_multiple_GS = use_multiple_GS or use_multiple_GS_DH

WID_TEXT_GuideStarAncFilename_Astig_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_GuideStarAncFilename_Astig')
widget_control,WID_TEXT_GuideStarAncFilename_Astig_ID,GET_VALUE = GS_anc_fname
GS_anc_file_info=FILE_INFO(GS_anc_fname)
use_multiple_GS = use_multiple_GS and GS_anc_file_info.exists

WidSldFitOrderID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Zastig_Fit')
widget_control,WidSldFitOrderID,get_value=fitorder

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
nmperframe_in_medium = nmperframe * z_media_multiplier
print,'nm per frame in medium = ',nmperframe_in_medium

!p.background=0

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
	subsetind0=where(filter eq 1,cnt)
	if jp eq 0 then superset_indices = subsetind0 else superset_indices = [superset_indices, subsetind0]
	yrng = [0.5,3.5]
	if n_elements(zz) eq 0 and cnt gt 0 then begin
		ellipticity = transpose(CGroupParams[Ell_ind,subsetind0])
		Xwid = transpose(CGroupParams[Xwid_ind,subsetind0])
		Ywid = transpose(CGroupParams[Ywid_ind,subsetind0])
		center_fr = find_center_frame(CGroupParams[*,subsetind0], fitorder, 20)
		zz=transpose(CGroupParams[FrNum_ind,subsetind0] - center_fr) * nmperframe_in_medium
		xi= min(zz) - (max(zz)-min(zz))/20.0
		xa= max(zz) + (max(zz)-min(zz))/20.0
    	xrng=[xi,xa]
		!P.NOERASE=0
		!p.multi=[0,1,2,0,0]
		plot, zz, ellipticity, xtitle='Z position (nm)', ytitle='Ellipticity', yrange = [-0.5,0.5], ystyle=1, xrange = xrng, xstyle=1, psym=6, xticklen=1, xgridstyle=1, yticklen=1, ygridstyle=1
		!p.multi=[1,1,2,0,0]
		plot, zz, Xwid, xtitle='Z position (nm)', ytitle='Gaussian Width', yrange = yrng, ystyle=1, xrange = xrng, xstyle=1,  psym=6, xticklen=1, xgridstyle=1, yticklen=1, ygridstyle=1
		oplot, zz, Ywid, psym=5
		oplot, zz, (Xwid+Ywid), psym=4
		!P.NOERASE=1
	endif else begin
		col=(250-jp*50)>50
		ellipticity_add = transpose(CGroupParams[Ell_ind,subsetind0])
		Xwid_add = transpose(CGroupParams[Xwid_ind,subsetind0])
		Ywid_add = transpose(CGroupParams[Ywid_ind,subsetind0])
		ellipticity = [ellipticity, ellipticity_add]
		Xwid = [Xwid, Xwid_add]
		Ywid = [Ywid, Ywid_add]
		center_fr = find_center_frame(CGroupParams[*,subsetind0], fitorder, 20)
		zz_add = transpose(CGroupParams[FrNum_ind,subsetind0] - center_fr) * nmperframe_in_medium
		zz= [zz, zz_add]
		!p.multi=[0,1,2,0,0]
		plot, zz_add, ellipticity_add, yrange = [-0.5,0.5], ystyle=1, xrange = xrng, xstyle=1, psym=6, col=col
		!p.multi=[1,1,2,0,0]
		plot, zz_add, Xwid_add, yrange = yrng, ystyle=1, xrange = xrng, xstyle=1, psym=6, col=col
		oplot, zz_add, Ywid_add, psym=5, col=col
		oplot, zz_add, (Xwid_add+Ywid_add), psym=4, col=col
	endelse

endfor

zlim_min = z_cal_min;
zlim_max = z_cal_max;

zzmin=round(min(zz)) > zlim_min
zzmax=round(max(zz)) < zlim_max
cal_lookup_zz=zzmin+findgen(zzmax-zzmin)

zz_fit_ind0 = where((zz ge zlim_min) and (zz le zlim_max))
; first attempt
ellipt_coeff = poly_fit(zz[zz_fit_ind0],ellipticity[zz_fit_ind0], fitorder, /double)
; now limit the range to between the mean and max of fit ellipticity
ellipt_fit = poly(cal_lookup_zz,ellipt_coeff)
efit_max = max(ellipt_fit,efit_max_ind)
efit_min = min(ellipt_fit,efit_min_ind)
;zlim_min_new = min([cal_lookup_zz[efit_max_ind], cal_lookup_zz[efit_min_ind]])
zlim_min_new = zlim_min
;zlim_max_new = max([cal_lookup_zz[efit_max_ind], cal_lookup_zz[efit_min_ind]])
zlim_max_new = zlim_max

print,'Range of z-calibration:  ', zlim_min_new,zlim_max_new
zz_fit_ind = where((zz ge zlim_min_new) and (zz le zlim_max_new))

b_coeff = poly_fit(zz[zz_fit_ind], Xwid[zz_fit_ind], fitorder, /double)
c_coeff = poly_fit(zz[zz_fit_ind], Ywid[zz_fit_ind], fitorder, /double)
sum_coeff = poly_fit(zz[zz_fit_ind],(Xwid[zz_fit_ind]+Ywid[zz_fit_ind]), fitorder, /double)
aa = [b_coeff, c_coeff, ellipt_coeff, sum_coeff]
print,'aa:'
print,transpose(aa)
cal_lookup_data = poly(cal_lookup_zz,ellipt_coeff)
Xwid_fit = poly(cal_lookup_zz,b_coeff)
Ywid_fit = poly(cal_lookup_zz,c_coeff)
sum_fit = poly(cal_lookup_zz,sum_coeff)

cal_min = min(cal_lookup_data, ind0)
cal_max = max(cal_lookup_data, ind1)
indmin = min([ind0,ind1])+1
indmax = max([ind0,ind1])-1
cal_lookup_data_lim = cal_lookup_data[indmin:indmax]
cal_lookup_zz_lim = cal_lookup_zz[indmin:indmax]

ncoeff = n_elements(ellipt_coeff)-1
deriv_coeff = (ellipt_coeff*findgen(ncoeff+1))[1:ncoeff]
deriv_lookup_data = poly(cal_lookup_zz_lim,deriv_coeff)
	!p.multi=[0,1,2,0,0]
	plot,cal_lookup_zz,cal_lookup_data, yrange = [-0.5,0.5], ystyle=1, xrange = xrng, xstyle=1, col=250, THICK=3
	oplot,cal_lookup_zz_lim,1.0/deriv_lookup_data/1.0e5, col=150, THICK=3
	!p.multi=[1,1,2,0,0]
	;plot, zz, Xwid, yrange = [0,5.0], ystyle=1, xrange = xrng, xstyle=1, psym=6, col=col
	plot, cal_lookup_zz, Xwid_fit, yrange = yrng, ystyle=1, xrange = xrng, xstyle=1, col=250, THICK=3
	oplot, cal_lookup_zz, Ywid_fit, col=200, THICK=3
	oplot, cal_lookup_zz, sum_fit, col=150, THICK=3
	for i =0,(n_elements(ellipt_coeff)-1) do xyouts,0.8,0.90-0.02*i,ellipt_coeff[i],/normal

thisfitcond0=thisfitcond
thisfitcond.SigmaSym = 2

;iterate calibratios if num_iter>1
if num_iter gt 1 then begin
	superset = CGroupParams[*,superset_indices]
	npks_superset = n_elements(superset_indices)
	for i_iter =1, (num_iter-1) do begin
		print,'Iteration #', i_iter,', re-extracting peaks'

		; First step, re-exract the data for the fiducial set
		for ip = 0, (npks_superset-1) do begin
			clip=ReadData(RawFilenames[0],thisfitcond,superset[FrNum_ind,ip],1)
			criteria = clip
			d=thisfitcond.MaskSize
			SigmaSym = thisfitcond.SigmaSym
			Dispxy=[0,0]
			DisplaySet=-1
			peakparams = {twinkle_z, frameindex:0l, peakindex:0l, fitOK:1, peakx:0.0, peaky:0.0, peak_widx:0.0, peak_widy:0.0, A:fltarr(6), sigma:fltarr(6), chisq:0.0, Nphot:0l}
			peakparams.frameindex = superset[FrNum_ind,ip]
			peakparams.peakindex = superset[PkInd_ind,ip]
			peakparams.A=[0.0,1.0,d,d,0.0,1.0]
			fita = [1,1,1,1,1,1]
			peakx = superset[X_ind,ip]
			peaky = superset[Y_ind,ip]
			;print, ip, peakx, peaky,superset[Z_ind,ip], superset[Xwid_ind,ip], superset[Ywid_ind,ip]
			FindnWackaPeak_AstigZ, clip, d, peakparams, fita, result, thisfitcond, aa, DisplaySet, peakx, peaky, criteria, Dispxy, 0		;Find, fit, and remove the peak with biggest criteria
			superset[Off_ind:Amp_ind,ip]=peakparams.A[0:1]>0
			if Chi_ind gt 0 then superset[Chi_ind,ip]=peakparams.ChiSq
			superset[Xwid_ind,ip] = peakparams.peak_widx
			superset[Ywid_ind,ip] = peakparams.peak_widy
			if Z_ind gt 0 then superset[Z_ind,ip]=peakparams.A[4]
			if SigX_ind gt 0 then superset[SigX_ind:SigY_ind,ip]=peakparams.Sigma[2:3]
			if SigZ_ind gt 0 then superset[SigZ_ind,ip]=peakparams.Sigma[4]
			if Ell_ind gt 0 then superset[Ell_ind,ip]=(peakparams.peak_widx-peakparams.peak_widy)/(peakparams.peak_widx+peakparams.peak_widy)
			if thisfitcond.SigmaSym eq 4 then if Par12_ind gt 0 then superset[Par12_ind,ip]=peakparams.A[5]
			superset[Nph_ind,ip]=peakparams.NPhot
			superset[FitOK_ind,ip]=peakparams.FitOK
			;print, ip, peakx, peaky,superset[Z_ind,ip], superset[Xwid_ind,ip], superset[Ywid_ind,ip]
		endfor
		; Second step, now re-extract calibration
		print,'Iteration #', i_iter,', re-extracting calibration coefficients'
		!P.NOERASE=0
		!p.multi=[0,0,0,0,0]
		!p.background=0
		counter = 0
		for jp=0,(ip_cnt-1) do begin
			if use_multiple_GS then begin
				xmin = GSAnchorPnts[0,jp]-GS_radius
				xmax = GSAnchorPnts[0,jp]+GS_radius
				ymin = GSAnchorPnts[1,jp]-GS_radius
				ymax = GSAnchorPnts[1,jp]+GS_radius
			endif else begin
				xmin = 0
				xmax = xydsz[0]
				ymin = 0
				ymax = xydsz[1]
			endelse
			subsetindex=where((superset[X_ind,*] ge xmin) $
					and  (superset[X_ind,*] le xmax) $
					and  (superset[Y_ind,*] ge ymin) $
					and  (superset[Y_ind,*] le ymax),cnt)
 			yrng = [0.5,3.5]
			if (counter eq 0) and (cnt gt 0) then begin
				counter = 1
				ellipticity = transpose(superset[Ell_ind,subsetindex])
				Xwid = transpose(superset[Xwid_ind,subsetindex])
				Ywid = transpose(superset[Ywid_ind,subsetindex])
				center_fr = find_center_frame(superset[*,subsetindex], fitorder, 20)
				zz=transpose(superset[FrNum_ind,subsetindex] - center_fr) * nmperframe_in_medium
				!P.NOERASE=0
				!p.multi=[0,1,2,0,0]
				plot, zz, ellipticity, xtitle='Z position (nm)', ytitle='Ellipticity', yrange = [-0.5,0.5], ystyle=1, xrange = xrng, xstyle=1, psym=6, xticklen=1, xgridstyle=1, yticklen=1, ygridstyle=1
				!p.multi=[1,1,2,0,0]
				plot, zz, Xwid, xtitle='Z position (nm)', ytitle='Gaussian Width', yrange = yrng, ystyle=1, xrange = xrng, xstyle=1,  psym=6, xticklen=1, xgridstyle=1, yticklen=1, ygridstyle=1
				oplot, zz, Ywid, psym=5
				oplot, zz, (Xwid+Ywid), psym=4
				!P.NOERASE=1
			endif else begin
				if cnt gt 0 then begin
					col=(250-jp*50)>50
					ellipticity_add = transpose(superset[Ell_ind,subsetindex])
					Xwid_add = transpose(superset[Xwid_ind,subsetindex])
					Ywid_add = transpose(superset[Ywid_ind,subsetindex])
					ellipticity = [ellipticity, ellipticity_add]
					Xwid = [Xwid, Xwid_add]
					Ywid = [Ywid, Ywid_add]
					center_fr = find_center_frame(superset[*,subsetindex], fitorder, 20)
					zz_add = transpose(superset[FrNum_ind,subsetindex] - center_fr) * nmperframe_in_medium
					zz= [zz, zz_add]
					!p.multi=[0,1,2,0,0]
					plot, zz_add, ellipticity_add, yrange = [-0.5,0.5], ystyle=1, xrange = xrng, xstyle=1, psym=6, col=col
					!p.multi=[1,1,2,0,0]
					plot, zz_add, Xwid_add, yrange = yrng, ystyle=1, xrange = xrng, xstyle=1, psym=6, col=col
					oplot, zz_add, Ywid_add, psym=5, col=col
					oplot, zz_add, (Xwid_add+Ywid_add), psym=4, col=col
				endif
			endelse
		endfor

		zzmin=round(min(zz)) > zlim_min
		zzmax=round(max(zz)) < zlim_max
		cal_lookup_zz=zzmin+findgen(zzmax-zzmin)

		zz_fit_ind0 = where((zz ge zlim_min) and (zz le zlim_max))
		; first attempt
		ellipt_coeff = poly_fit(zz[zz_fit_ind0],ellipticity[zz_fit_ind0], fitorder, /double)
		; now limit the range to between the mean and max of fit ellipticity
		ellipt_fit = poly(cal_lookup_zz,ellipt_coeff)
		efit_max = max(ellipt_fit,efit_max_ind)
		efit_min = min(ellipt_fit,efit_min_ind)
		zlim_min_new = zlim_min
		zlim_max_new = zlim_max

		print,'Range of z-calibration:  ', zlim_min_new,zlim_max_new
		zz_fit_ind = where((zz ge zlim_min_new) and (zz le zlim_max_new))

		b_coeff = poly_fit(zz[zz_fit_ind], Xwid[zz_fit_ind], fitorder, /double)
		c_coeff = poly_fit(zz[zz_fit_ind], Ywid[zz_fit_ind], fitorder, /double)
		sum_coeff = poly_fit(zz[zz_fit_ind],(Xwid[zz_fit_ind]+Ywid[zz_fit_ind]), fitorder, /double)
		aa = [b_coeff, c_coeff, ellipt_coeff, sum_coeff]
		print,'aa:'
		print,transpose(aa)
		cal_lookup_data = poly(cal_lookup_zz,ellipt_coeff)
		Xwid_fit = poly(cal_lookup_zz,b_coeff)
		Ywid_fit = poly(cal_lookup_zz,c_coeff)
		sum_fit = poly(cal_lookup_zz,sum_coeff)

		cal_min = min(cal_lookup_data, ind0)
		cal_max = max(cal_lookup_data, ind1)
		indmin = min([ind0,ind1])+1
		indmax = max([ind0,ind1])-1
		cal_lookup_data_lim = cal_lookup_data[indmin:indmax]
		cal_lookup_zz_lim = cal_lookup_zz[indmin:indmax]

		ncoeff = n_elements(ellipt_coeff)-1
		deriv_coeff = (ellipt_coeff*findgen(ncoeff+1))[1:ncoeff]
		deriv_lookup_data = poly(cal_lookup_zz_lim,deriv_coeff)
		!p.multi=[0,1,2,0,0]
		plot,cal_lookup_zz,cal_lookup_data, yrange = [-0.5,0.5], ystyle=1, xrange = xrng, xstyle=1, col=250, THICK=3
		oplot,cal_lookup_zz_lim,1.0/deriv_lookup_data/1.0e5, col=150, THICK=3
		!p.multi=[1,1,2,0,0]
		plot, cal_lookup_zz, Xwid_fit, yrange = yrng, ystyle=1, xrange = xrng, xstyle=1, col=250, THICK=3
		oplot, cal_lookup_zz, Ywid_fit, col=200, THICK=3
		oplot, cal_lookup_zz, sum_fit, col=150, THICK=3
		for i =0,(n_elements(ellipt_coeff)-1) do xyouts,0.8,0.90-0.02*i,ellipt_coeff[i],/normal
	endfor
endif
thisfitcond=thisfitcond0

!P.NOERASE=0
!p.multi=[0,0,0,0,0]
!p.background=0
return
end
;
;-----------------------------------------------------------------
;
pro OnExtractZCoord_Astig, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

if wfilename eq '' then return

Xwid_ind = min(where(RowNames eq 'X Peak Width'))                        ; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))                        ; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))                            ; CGroupParametersGP[6,*] - Number of Photons in the Peak
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))                    ; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))                    ; CGroupParametersGP[17,*] - y - sigma
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))                ; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))                ; CGroupParametersGP[22,*] - new y - position sigma

Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size = min(where(RowNames eq '24 Group Size'))

if (n_elements(cal_lookup_zz) le 1) or (n_elements(cal_lookup_data) le 1) then begin
	z=dialog_message('Please load a Calibration file')
	return      ; if data not loaded return
endif
ellipt_coeff = aa[2,*]
ncoeff = n_elements(ellipt_coeff)-1
deriv_coeff = (ellipt_coeff*findgen(ncoeff+1))[1:ncoeff]
deriv_lookup_data = poly(cal_lookup_zz,deriv_coeff)

fit_index = VALUE_LOCATE(cal_lookup_data,CGroupParams[Ell_ind,*])
CGroupParams[Z_ind,*] = cal_lookup_zz[fit_index>0]

; below is an estimate of uncertainty of determination of ellipticity
;sigma_ellipticity = 2.0*CGroupParams[Xwid_ind,*]*CGroupParams[Ywid_ind,*]/(CGroupParams[Xwid_ind,*]+CGroupParams[Ywid_ind,*])^2;/sqrt(CGroupParams[Nph_ind,*])
;sigma_ellipticity = 0.5*(1.0-CGroupParams[Ell_ind,*]*CGroupParams[Ell_ind,*])/sqrt(CGroupParams[Nph_ind,*])
sigma_ellipticity = 2.0 * sqrt((CGroupParams[Xwid_ind,*]*CGroupParams[SigY_ind,*])^2+(CGroupParams[Ywid_ind,*]*CGroupParams[SigX_ind,*])^2) / $
(CGroupParams[Xwid_ind,*]+CGroupParams[Ywid_ind,*])^2;/sqrt(CGroupParams[Nph_ind,*])
; propagate uncertainty through ellipticity-> extraction
CGroupParams[SigZ_ind,*] = sigma_ellipticity/abs(deriv_lookup_data[fit_index>0])
;CGroupParams[11,*] = 1/abs(deriv_lookup_data[fit_index>0])

if (mean(CGroupParams[Gr_Size,*]) ne 0) then  begin
	fit_gr_index = VALUE_LOCATE(cal_lookup_data,CGroupParams[Gr_Ell_ind,*])
	CGroupParams[GrZ_ind,*] = cal_lookup_zz[fit_gr_index>0]
	sigma_ellipticity = 2.0 * sqrt((CGroupParams[Xwid_ind,*]*CGroupParams[GrSigY_ind,*])^2+(CGroupParams[Ywid_ind,*]*CGroupParams[GrSigX_ind,*])^2) / $
	(CGroupParams[Xwid_ind,*]+CGroupParams[Ywid_ind,*])^2;/sqrt(CGroupParams[Nph_ind,*])
	; propagate uncertainty through ellipticity-> extraction
	CGroupParams[GrSigZ_ind,*] = sigma_ellipticity/abs(deriv_lookup_data[fit_index>0])
endif

ReloadParamlists, Event, [Z_ind, SigZ_ind, GrZ_ind, GrSigZ_ind]

if bridge_exists then begin
	print,'Reloading the Bridge Array'
	CATCH, Error_status
	CGroupParams_bridge = SHMVAR(shmName_data)
	CGroupParams_bridge[Z_ind,*] = CGroupParams[Z_ind,*]
	CGroupParams_bridge[GrZ_ind,*] = CGroupParams[GrZ_ind,*]
	CGroupParams_bridge[SigZ_ind,*] = CGroupParams[SigZ_ind,*]
	CGroupParams_bridge[GrSigZ_ind,*] = CGroupParams[GrSigZ_ind,*]
	IF Error_status NE 0 THEN BEGIN
		PRINT, 'Bridge Refresh Error: Drift Correction:',!ERROR_STATE.MSG
		PRINT, 'System Error Message:',!ERROR_STATE.SYS_MSG
		bridge_exists = 0
		SHMUnmap, shmName_data
		SHMUnmap, shmName_filter
		for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
		CATCH,/CANCEL
	ENDIF
	CATCH,/CANCEL
	print,'Finished Reloading the Bridge Array'
endif

end
;
;-----------------------------------------------------------------
;
pro OnSaveEllipticityCal_Astig, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
BKGRND= 'Black'
WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WindFilename_Astig')
widget_control,WFileWidID,GET_VALUE = wfilename
if wfilename eq '' then return
wfilename_ext=AddExtension(wfilename,'_WND.sav')
if wfilename_ext ne wfilename then widget_control,WFileWidID,SET_VALUE = wfilename_ext
save, nmperframe, aa, lambda_vac, nd_water, nd_oil, wind_range, cal_lookup_data, cal_lookup_zz, filename=wfilename_ext
end
;
;-----------------------------------------------------------------
;
pro On_Convert_Frame_to_Z, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

;z_media_multiplier depends on objective NA and media index. This is ratio which determines by how much the focal plane of the (air) objective shifts in the media for a unit shift of the objective along the axis.

WIDID_TEXT_ZCalStep = Widget_Info(Event.top, find_by_uname='WID_TEXT_ZCalStep_Astig')
widget_control,WIDID_TEXT_ZCalStep,GET_VALUE = nmperframe_txt
nmperframe = float(nmperframe_txt[0])

FrNum_ind = min(where(RowNames eq 'Frame Number'))                        ; CGroupParametersGP[9,*] - frame number
Z_ind=min(where(RowNames eq 'Z Position'))
nmperframe_in_medium = nmperframe * z_media_multiplier
Fr_Max = max(CGroupParams[FrNum_ind,*])
;CGroupParams[Z_ind,*] = (CGroupParams[FrNum_ind,*]) * nmperframe_in_medium
CGroupParams[Z_ind,*] = (Fr_Max - (CGroupParams[FrNum_ind,*])) * nmperframe_in_medium

if bridge_exists then begin
	print,'Reloading the Bridge Array'
	CATCH, Error_status
	CGroupParams_bridge = SHMVAR(shmName_data)
	CGroupParams_bridge[Z_ind,*] = CGroupParams[Z_ind,*]
	IF Error_status NE 0 THEN BEGIN
		PRINT, 'Bridge Refresh Error: Drift Correction:',!ERROR_STATE.MSG
		PRINT, 'System Error Message:',!ERROR_STATE.SYS_MSG
		bridge_exists = 0
		SHMUnmap, shmName_data
		SHMUnmap, shmName_filter
		for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
		CATCH,/CANCEL
	ENDIF
	CATCH,/CANCEL
	print,'Finished Reloading the Bridge Array'
endif

ReloadParamlists, Event, [Z_ind]

end
;
;-----------------------------------------------------------------
;
pro ExtractSubsetZ_Astig, Event, zdrift, use_multiple_GS	;Pulls out subset of data from param limits and fits z vs frames
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(CGroupParams) lt 1 then begin

	return      ; if data not loaded return
endif

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	z=dialog_message('Error: Drift Correction: '+string(!ERROR_STATE.MSG))
	CATCH,/CANCEL
	!p.multi=[0,0,0,0,0]
	return
ENDIF

Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size = min(where(RowNames eq '24 Group Size'))
FrNum_ind = min(where(RowNames eq 'Frame Number'))
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak

!p.multi=[0,1,2,0,0]

WidSldFitOrderID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_Fit')
widget_control,WidSldFitOrderID,get_value=fitorder

WidSmWidthID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_Sm_Width')
widget_control,WidSmWidthID,get_value=SmWidth

WidDListDispFitMethodID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z_Fit_Method')
FitMethod = widget_info(WidDListDispFitMethodID,/DropList_Select)

use_multiple_GS_DH_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs_DH')
use_multiple_GS_DH = widget_info(use_multiple_GS_DH_id,/button_set)

GS_anc_file_info=FILE_INFO(GS_anc_fname)
; if using multiple Guide Stars - load their X-Y coordinates
use_multiple_ANC = (use_multiple_GS or use_multiple_GS_DH) and GS_anc_file_info.exists and (strlen(GS_anc_fname) gt 0)
GSAnchorPnts=dblarr(2,AnchPnts_MaxNum)
GSAnchorPnts_line=dblarr(6)
if use_multiple_ANC  then begin
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

NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[FrNum_ind,*])+1))	: long64(max(CGroupParams[FrNum_ind,*])+1)
FR=dindgen(NFrames)

; original procedure (treat each Guide Star separetly and then average)
if (NOT use_multiple_GS_DH) then begin
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
		if cnt lt 1 then begin
			z=dialog_message('Z- GuideStar subset has '+string(cnt)+' points')
			!p.multi=[0,0,0,0,0]
			return      ; if data not loaded return
		endif
		print, 'Z- GuideStar subset has ',cnt,' points'
		if cnt gt 0 then begin
			subset=CGroupParams[*,subsetindex]
			if FitMethod eq 0 then begin
				zcoef=poly_fit(subset[FrNum_ind,*],subset[Z_ind,*],fitorder,YFIT=fit_to_x)
				ZFit=poly(FR,zcoef)
			endif else begin
				indecis=uniq(subset[FrNum_ind,*])
				Zsmooth=smooth(subset[Nph_ind,indecis]*subset[Z_ind,indecis],SmWidth,/EDGE_TRUNCATE)/smooth(subset[Nph_ind,indecis],SmWidth,/EDGE_TRUNCATE)
				Zfit=interpol(Zsmooth,subset[FrNum_ind,indecis],FR)
				frame_low = min(subset[FrNum_ind,indecis])
				ind_low=where(FR[*] lt frame_low,c)
				if c ge 1 then Zfit[ind_low]=Zfit[frame_low]
				frame_high = max(subset[FrNum_ind,indecis])
				ind_high=where(FR[*] gt frame_high,c)
				if c ge 1 then	Zfit[ind_high]=Zfit[frame_high]
			endelse

			if (n_elements(cal_lookup_zz) gt 2) then Ellipt_Fit = cal_lookup_data[(VALUE_LOCATE(cal_lookup_zz,Zfit)>0)]
			;Ellipt_Fit = cal_lookup_data[(VALUE_LOCATE(cal_lookup_zz,Zfit)>0)]

			zdrift=Zfit-Zfit[0]
			zdrift_mult = (jp eq 0) ? transpose(zdrift) : [zdrift_mult,transpose(zdrift)]

			if jp eq 0 then begin
				!P.NOERASE=0
				plot,FR,ZFit,xtitle='Frame',ytitle='Z-DRIFT (nm)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1,yrange=Paramlimits[Z_ind,0:1],ystyle=1
				oplot,subset[FrNum_ind,*],subset[Z_ind,*],psym=3
				if FitMethod eq 0 then for i =0,fitorder do xyouts,0.8,0.95-0.02*i,zcoef[i],/normal
				if (n_elements(Ellipt_Fit) gt 2) and (total(abs(subset[Ell_ind,0:(100 < ((size(subset))[2]-1))])) ne 0.0) then begin
					plot,FR,Ellipt_Fit,xtitle='Frame',ytitle='X-Y Ellipticity',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1,yrange=Paramlimits[Ell_ind,0:1],ystyle=1
					oplot,subset[FrNum_ind,*],subset[Ell_ind,*],psym=3
				endif
				!P.NOERASE=1
			endif else begin
				!p.multi=[0,1,2,0,0]
				plot,FR,ZFit,xtitle='Frame',ytitle='Z-DRIFT (nm)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1,yrange=Paramlimits[Z_ind,0:1],ystyle=1
				col=(250-jp*50)>50
				oplot,FR,ZFit,col=col
				oplot,subset[FrNum_ind,*],subset[Z_ind,*],psym=3,col=col
				if FitMethod eq 0 then for i =0,fitorder do xyouts,0.8,0.95-0.02*i,zcoef[i],/normal
					if (n_elements(Ellipt_Fit) gt 2) and (total(abs(subset[Ell_ind,0:(100 <((size(subset))[2]-1))])) ne 0.0) then begin
					!p.multi=[1,1,2,0,0]
					plot,FR,Ellipt_Fit,xtitle='Frame',ytitle='X-Y Ellipticity',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1,yrange=Paramlimits[Ell_ind,0:1],ystyle=1
					oplot,FR,Ellipt_Fit,col=col
					oplot,subset[FrNum_ind,*],subset[Ell_ind,*],psym=3,col=col
				endif
			endelse
		endif
	endfor
endif else begin
; New procedure (DPH) - average first
	superset_Z = dblarr(ip_cnt, NFrames)
	superset_Nph = superset_Z
	superset_FR = superset_Z
	zdr_plotted=0

	for jp=0,(ip_cnt-1) do begin
		filter=filter0

		ParamLimits[2,2] = GSAnchorPnts[0,jp]
		ParamLimits[2,0] = GSAnchorPnts[0,jp]-GS_radius
		ParamLimits[2,1] = GSAnchorPnts[0,jp]+GS_radius
		ParamLimits[3,2] = GSAnchorPnts[1,jp]
		ParamLimits[3,0] = GSAnchorPnts[1,jp]-GS_radius
		ParamLimits[3,1] = GSAnchorPnts[1,jp]+GS_radius

		FilterIt
		subsetindex=where(filter eq 1,cnt)
		print, jp,'  Z- GuideStar subset has ',cnt,' points',  ',   X=',ParamLimits[2,2],', Y=',ParamLimits[3,2]
		if cnt gt 0 then begin
			subset = double(CGroupParams[*,subsetindex])
			subset[Z_ind,*] = subset[Z_ind,*] - mean(subset[Z_ind,*],/DOUBLE)
			; remove secondary peaks in frames
			x=subset[FrNum_ind,*]
			ind = reverse(n_elements(X)-uniq(reverse(X))-1)
			subset = temporary(subset[*,ind])
			superset_Z[jp,subset[FrNum_ind,*]] = subset[Z_ind,*]
			superset_Nph[jp,subset[FrNum_ind,*]] = subset[Nph_ind,*]
			superset_FR[jp,*] = FR
			if zdr_plotted eq 0 then begin
				zdr_plotted=1
				!P.NOERASE=0
				yrng0 = [1.5*min(subset[Z_ind,*])<0, 1.5*max(subset[Z_ind,*])>0]
				plot,subset[FrNum_ind,*],subset[Z_ind,*],xtitle='Frame',ytitle='Z-DRIFT (nm)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1, yrange=yrng0,ystyle=1, psym=3
				if Ell_ind ge 0 then  begin
					plot,subset[FrNum_ind,*],subset[Ell_ind,*],xtitle='Frame',ytitle='X-Y Ellipticity',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1,yrange=Paramlimits[Ell_ind,0:1],ystyle=1, psym=3
				endif
				!P.NOERASE=1
			endif else begin
				col=(250-jp*25)>50
				!p.multi=[0,1,2,0,0]
				plot,subset[FrNum_ind,*],subset[Z_ind,*],xtitle='Frame',ytitle='Z-DRIFT (nm)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1, yrange=yrng0,ystyle=1, psym=3
				oplot,subset[FrNum_ind,*],subset[Z_ind,*],col=col,psym=3
				if Ell_ind ge 0 then  begin
					!p.multi=[1,1,2,0,0]
					plot,subset[FrNum_ind,*],subset[Ell_ind,*],xtitle='Frame',ytitle='X-Y Ellipticity',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1,yrange=Paramlimits[Ell_ind,0:1],ystyle=1, psym=3
					oplot,subset[FrNum_ind,*],subset[Ell_ind,*],col=col,psym=3
				endif
			endelse
		endif
	endfor

	if FitMethod eq 0 then begin
		non_zero_ind=where(superset_Nph gt 0)
		zcoef=poly_fit(superset_Fr[non_zero_ind], superset_Z[non_zero_ind], fitorder, YFIT=fit_to_x)
		ZFit=poly(FR, zcoef)
	endif else begin
		Nph_arr = total(superset_Nph,1)
		Z_arr = total(superset_Z*superset_Nph,1)
		non_zero_ind=where(Nph_arr gt 0)
		Zsmooth=smooth(Z_arr[non_zero_ind]/Nph_arr[non_zero_ind],SmWidth,/EDGE_TRUNCATE)
		Zfit=interpol(Zsmooth,FR[non_zero_ind],FR)
	endelse

	!p.multi=[0,1,2,0,0]
	plot,FR,ZFit,xtitle='Frame',ytitle='Z-DRIFT (nm)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1,yrange=yrng0,ystyle=1, thick=2
	zdrift=Zfit-Zfit[0]

endelse

!p.multi=[0,0,0,0,0]
!p.background=0
!P.NOERASE=0

if use_multiple_GS and (NOT use_multiple_GS_DH) then begin
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
CATCH,/CANCEL
!p.multi=[0,0,0,0,0]
return
end
;
;-----------------------------------------------------------------
;
pro OnTestZDrift_Astig, Event
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius

use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs')
if use_multiple_GS_id gt 0 then use_multiple_GS = widget_info(use_multiple_GS_id,/button_set) else use_multiple_GS=0
;use_multiple_GS_DH_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs_DH')
;if use_multiple_GS_DH_id gt 0 then use_multiple_GS_DH = widget_info(use_multiple_GS_DH_id,/button_set) else use_multiple_GS_DH=0
;use_multiple_GS = use_multiple_GS or use_multiple_GS_DH

WID_TEXT_GuideStarAncFilename_Astig_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_GuideStarAncFilename_Astig')
widget_control,WID_TEXT_GuideStarAncFilename_Astig_ID,GET_VALUE = GS_anc_fname
GS_anc_file_info=FILE_INFO(GS_anc_fname)
use_multiple_GS = use_multiple_GS and GS_anc_file_info.exists
ExtractSubsetZ_Astig, Event, zdift, use_multiple_GS
end
;
;-----------------------------------------------------------------
;
pro OnWriteZDrift_Astig, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size = min(where(RowNames eq '24 Group Size'))
Frame_Number = min(where(RowNames eq 'Frame Number'))

use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs')
if use_multiple_GS_id gt 0 then use_multiple_GS = widget_info(use_multiple_GS_id,/button_set) else use_multiple_GS=0
;use_multiple_GS_DH_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs_DH')
;if use_multiple_GS_DH_id gt 0 then use_multiple_GS_DH = widget_info(use_multiple_GS_DH_id,/button_set) else use_multiple_GS_DH=0
;use_multiple_GS = use_multiple_GS or use_multiple_GS_DH

WID_TEXT_GuideStarAncFilename_Astig_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_GuideStarAncFilename_Astig')
widget_control,WID_TEXT_GuideStarAncFilename_Astig_ID,GET_VALUE = GS_anc_fname
GS_anc_file_info=FILE_INFO(GS_anc_fname)
use_multiple_GS = use_multiple_GS and GS_anc_file_info.exists

ExtractSubsetZ_Astig, Event, zdrift, use_multiple_GS

print,"Performing Z Drift Correction"
CGroupParams[Z_ind,*] = CGroupParams[Z_ind,*] - zdrift[CGroupParams[Frame_Number,*]]
CGroupParams[GrZ_ind,*] = CGroupParams[GrZ_ind,*] - zdrift[CGroupParams[Frame_Number,*]]
GuideStarDrift[0].zdrift = zdrift

if bridge_exists then begin
	print,'Reloading the Bridge Array'
	CATCH, Error_status
	CGroupParams_bridge = SHMVAR(shmName_data)
	CGroupParams_bridge[Z_ind,*] = CGroupParams[Z_ind,*]
	CGroupParams_bridge[GrZ_ind,*] = CGroupParams[GrZ_ind,*]
	IF Error_status NE 0 THEN BEGIN
		PRINT, 'Bridge Refresh Error: Drift Correction:',!ERROR_STATE.MSG
		PRINT, 'System Error Message:',!ERROR_STATE.SYS_MSG
		bridge_exists = 0
		SHMUnmap, shmName_data
		SHMUnmap, shmName_filter
		for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
		CATCH,/CANCEL
	ENDIF
	CATCH,/CANCEL
	print,'Finished Reloading the Bridge Array'
endif

ReloadParamlists, Event, [Ell_ind, Z_ind, Gr_Ell_ind, GrZ_ind]

end
;
;-----------------------------------------------------------------
;
pro OnRemoveTilt_Astig, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

Z_ind=min(where(RowNames eq 'Z Position'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))

Remove_XYZ_tilt

if bridge_exists then begin
	print,'Reloading the Bridge Array'
	CATCH, Error_status
	CGroupParams_bridge = SHMVAR(shmName_data)
	CGroupParams_bridge[Z_ind,*] = CGroupParams[Z_ind,*]
	CGroupParams_bridge[GrZ_ind,*] = CGroupParams[GrZ_ind,*]
	IF Error_status NE 0 THEN BEGIN
		PRINT, 'Bridge Refresh Error: Tilt Removal:',!ERROR_STATE.MSG
		PRINT, 'System Error Message:',!ERROR_STATE.SYS_MSG
		bridge_exists = 0
		SHMUnmap, shmName_data
		SHMUnmap, shmName_filter
		for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
		CATCH,/CANCEL
	ENDIF
	CATCH,/CANCEL
	print,'Finished Reloading the Bridge Array'
endif

end
;
;-----------------------------------------------------------------
;
pro OnAddOffset_Astig, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

WidZPhaseOffsetID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_phase_offset_Astig')
widget_control,WidZPhaseOffsetID,get_value=phase_offset

print,'phase offset=',phase_offset

Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))

CGroupParams[Z_ind,*]= Cgroupparams[Z_ind,*] + phase_offset
CGroupParams[GrZ_ind,*]= Cgroupparams[GrZ_ind,*] + phase_offset

ind_l = [Z_ind, GrZ_ind ]

	for k=0,(n_elements(ind_l)-1) do begin
		i = ind_l[k]
		valid_cgp=WHERE(FINITE(CGroupParams[i,*]),cnt)
		ParamLimits[i,0]=1.1*min(CGroupParams[i,valid_cgp]) < 0.9*min(CGroupParams[i,valid_cgp])
		ParamLimits[i,1]=1.1*max(CGroupParams[i,valid_cgp])
		ParamLimits[i,3]=ParamLimits[i,1] - ParamLimits[i,0]
		ParamLimits[i,2]=(ParamLimits[i,1] + ParamLimits[i,0])/2.
	endfor

	LimitUnwrapZ

	wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
	widget_control,wtable,get_value=ini_table, use_table_select=[0,Z_ind,3,GrZ_ind]
	ini_table[0:3,0]=ParamLimits[Z_ind,0:3]
	ini_table[0:3,3]=ParamLimits[GrZ_ind,0:3]
	widget_control,wtable,set_value=ini_table, use_table_select=[0,Z_ind,3,GrZ_ind]
	widget_control, wtable, /editable,/sensitive

end
;
;-----------------------------------------------------------------
;
pro OnButtonClose_Astig, Event
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnPickCalFile_Astig, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
wfilename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.sav'],title='Select *WND.sav file to open')
if wfilename ne '' then begin
	cd,fpath
	WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WindFilename_Astig')
	widget_control,WFileWidID,SET_VALUE = wfilename
	wfile_info=FILE_INFO(wfilename)
		if wfile_info.exists then begin
			restore,filename=wfilename
			WIDID_TEXT_ZCalStep = Widget_Info(Event.Top, find_by_uname='WID_TEXT_ZCalStep_Astig')
			nmperframe_txt=string(nmperframe,FORMAT='(F8.2)')
			widget_control, WIDID_TEXT_ZCalStep, SET_VALUE = nmperframe_txt
		endif
endif
end
;
;-----------------------------------------------------------------
;
pro OnPickGuideStarAncFile_Astig, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, names, modalList
GS_anc_fname = Dialog_Pickfile(/read,get_path=fpath,filter=['*.anc'],title='Select *.anc file to open')
GS_anc_file_info=FILE_INFO(GS_anc_fname)
if (GS_anc_fname ne '') and GS_anc_file_info.exists then begin
	cd,fpath
	WID_TEXT_GuideStarAncFilename_Astig_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_GuideStarAncFilename_Astig')
	widget_control,WID_TEXT_GuideStarAncFilename_Astig_ID,SET_VALUE = GS_anc_fname
	BASE_GuideStar_num = min(where(names eq 'WID_BASE_GuideStar'))
	if BASE_GuideStar_num gt 0 then begin
		BASE_GuideStarID=ids[BASE_GuideStar_num]
		if BASE_GuideStarID gt 0 then begin
			WID_TEXT_XY_GuideStarAncFilename_ID = Widget_Info(BASE_GuideStarID, find_by_uname='WID_TEXT_XY_GuideStarAncFilename')
			if WID_TEXT_XY_GuideStarAncFilename_ID gt 0 then widget_control,WID_TEXT_XY_GuideStarAncFilename_ID,SET_VALUE = GS_anc_fname
		endif
	endif
endif else GS_anc_fname = ''
end
;
;-----------------------------------------------------------------
;
pro WriteGudeStarRadius_Astig, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, names, modalList
WID_TEXT_GuideStar_Radius_AstigID = Widget_Info(Event.top, find_by_uname='WID_TEXT_GuideStar_Radius_Astig')
widget_control,WID_TEXT_GuideStar_Radius_AstigID,GET_VALUE = GS_radius_txt
BASE_GuideStar_num=min(where(names eq 'WID_BASE_GuideStar'))
if BASE_GuideStar_num ge 0 then begin
	BASE_GuideStarID=ids[BASE_GuideStar_num]
	WID_TEXT_XY_GuideStar_Radius_ID = Widget_Info(BASE_GuideStarID, find_by_uname='WID_TEXT_XY_GuideStar_Radius')
	if WID_TEXT_XY_GuideStar_Radius_ID gt 0 then widget_control,WID_TEXT_XY_GuideStar_Radius_ID,SET_VALUE = GS_radius_txt
endif
GS_radius = float(GS_radius_txt[0])
end

