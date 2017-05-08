;
; Empty stub procedure used for autoloading.
;
pro Zoperations_Astig_Wid_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_Z_operations_Astig, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
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

WIDID_TEXT_ZCalStep = Widget_Info(wWidget, find_by_uname='WID_TEXT_ZCalStep_Astig')
nmperframe_txt=string(nmperframe,FORMAT='(F8.2)')
widget_control,WIDID_TEXT_ZCalStep,SET_VALUE = nmperframe_txt

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
			z_unwrap_coeff = [0.0,0.0,0.0]
			restore,filename=wfilename
		endif
	endif else begin
		z_unwrap_coeff = [0.0,0.0,0.0]
	endelse
endif else begin
	z_unwrap_coeff = [0.0,0.0,0.0]
endelse

FitmethodID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Z_Fit_Method')
widget_control,FitmethodID,SET_DROPLIST_SELECT=1

WPhaseSliderID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Z_phase_offset')
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
Pro ExtractEllipticityCalib_Astig, Event		; perform z-calibration on filtered data set
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common InfoFit, pth, filen, thisfitcond, saved_pks_filenam

common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size = min(where(RowNames eq '24 Group Size'))
Frame_Number = min(where(RowNames eq 'Frame Number'))
sz=size(CGroupParams)

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

FilterIt

subsetindex=where(filter eq 1,cnt)
print, 'subset has ',cnt,' points'
if cnt le 0 then return

WIDID_TEXT_ZCalStep = Widget_Info(Event.top, find_by_uname='WID_TEXT_ZCalStep_Astig')
widget_control,WIDID_TEXT_ZCalStep,GET_VALUE = nmperframe_txt
nmperframe = float(nmperframe_txt[0])

NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[Frame_Number,*])+1))	: long64(max(CGroupParams[Frame_Number,*])+1)
ellipticity=transpose(CGroupParams[Ell_ind,subsetindex])
zz=(CGroupParams[Frame_Number,subsetindex]-NFrames/2.)*nmperframe * nd_oil/nd_water
zzmin=round(min(zz))
zzmax=round(max(zz))
cal_lookup_zz=zzmin+findgen(zzmax-zzmin)

WidSldFitOrderID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Zastig_Fit')
widget_control,WidSldFitOrderID,get_value=fitorder
z_unwrap_coeff = poly_fit(zz,ellipticity,fitorder)
cal_lookup_data=poly(cal_lookup_zz,z_unwrap_coeff)

	plot,zz,ellipticity,xtitle='Z position (nm)',ytitle='Ellipticity',ystyle=1,psym=6
	oplot,cal_lookup_zz,cal_lookup_data,col=col
	for i =0,(n_elements(z_unwrap_coeff)-1) do xyouts,0.8,0.90-0.02*i,z_unwrap_coeff[i],/normal

!p.background=0

return
end
;
;-----------------------------------------------------------------
;
pro OnExtractZCoord_Astig, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

if wfilename eq '' then return

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

CGroupParams[Z_ind,*] = cal_lookup_zz[(VALUE_LOCATE(cal_lookup_data,CGroupParams[Ell_ind,*])>0)]

if (mean(CGroupParams[Gr_Size,*]) ne 0) then  $
CGroupParams[GrZ_ind,*] = cal_lookup_zz[(VALUE_LOCATE(cal_lookup_data,CGroupParams[Gr_Ell_ind,*])>0)]

ReloadParamlists, Event, [Z_ind, GrZ_ind]
end
;
;-----------------------------------------------------------------
;
pro OnSaveEllipticityCal, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
BKGRND= 'Black'
WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WindFilename_Astig')
widget_control,WFileWidID,GET_VALUE = wfilename
if wfilename eq '' then return
wfilename_ext=AddExtension(wfilename,'_WND.sav')
if wfilename_ext ne wfilename then widget_control,WFileWidID,SET_VALUE = wfilename_ext
save, nmperframe, z_unwrap_coeff, lambda_vac, nd_water, nd_oil, wind_range, cal_lookup_data, cal_lookup_zz, filename=wfilename_ext
end
;
;-----------------------------------------------------------------
;
pro ExtractSubsetZ_Astig, Event, zdrift, use_multiple_GS	;Pulls out subset of data from param limits and fits z vs frames
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, thisfitcond, saved_pks_filenam
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size = min(where(RowNames eq '24 Group Size'))
Frame_Number = min(where(RowNames eq 'Frame Number'))
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak

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
		;NFrames=long64(max(CGroupParams[Frame_Number,*]))
		NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[Frame_Number,*])+1))	: long64(max(CGroupParams[Frame_Number,*])+1)
		subset=CGroupParams[*,subsetindex]
		FR=findgen(NFrames)

		if FitMethod eq 0 then begin
			WidSldFitOrderID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_Fit')
			widget_control,WidSldFitOrderID,get_value=fitorder
			zcoef=poly_fit(subset[Frame_Number,*],subset[Z_ind,*],fitorder,YFIT=fit_to_x)
			ZFit=poly(FR,zcoef)
		endif else begin
			WidSmWidthID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_Sm_Width')
			widget_control,WidSmWidthID,get_value=SmWidth
			indecis=uniq(subset[Frame_Number,*])
			Zsmooth=smooth(subset[Nph_ind,indecis]*subset[Z_ind,indecis],SmWidth,/EDGE_TRUNCATE)/smooth(subset[Nph_ind,indecis],SmWidth,/EDGE_TRUNCATE)
			Zfit=interpol(Zsmooth,subset[Frame_Number,indecis],FR)
			frame_low = min(subset[Frame_Number,indecis])
			ind_low=where(FR[*] lt frame_low,c)
			if c ge 1 then Zfit[ind_low]=Zfit[frame_low]
			frame_high = max(subset[Frame_Number,indecis])
			ind_high=where(FR[*] gt frame_high,c)
			if c ge 1 then	Zfit[ind_high]=Zfit[frame_high]
		endelse

		if (n_elements(cal_lookup_zz) gt 2) then Ellipt_Fit = cal_lookup_data[(VALUE_LOCATE(cal_lookup_zz,Zfit)>0)]
		;Ellipt_Fit = cal_lookup_data[(VALUE_LOCATE(cal_lookup_zz,Zfit)>0)]

		zdrift=Zfit-Zfit[0]
		zdrift_mult = (jp eq 0) ? transpose(zdrift) : [zdrift_mult,transpose(zdrift)]

		if jp eq 0 then begin
			!P.NOERASE=0
			plot,FR,ZFit,xtitle='frame',ytitle='Z-DRIFT (nm)',xrange=Paramlimits[Frame_Number,0:1],xstyle=1,yrange=Paramlimits[Z_ind,0:1],ystyle=1
			oplot,subset[Frame_Number,*],subset[Z_ind,*],psym=3
			if FitMethod eq 0 then for i =0,fitorder do xyouts,0.8,0.95-0.02*i,zcoef[i],/normal
			if (n_elements(Ellipt_Fit) gt 2) and (total(abs(subset[Ell_ind,0:(100 < ((size(subset))[2]-1))])) ne 0.0) then begin
				plot,FR,Ellipt_Fit,xtitle='frame',ytitle='X-Y Ellipticity',xrange=Paramlimits[Frame_Number,0:1],xstyle=1,yrange=Paramlimits[Ell_ind,0:1],ystyle=1
				oplot,subset[Frame_Number,*],subset[Ell_ind,*],psym=3
			endif
			!P.NOERASE=1
		endif else begin
			!p.multi=[0,1,2,0,0]
			plot,FR,ZFit,xtitle='frame',ytitle='Z-DRIFT (nm)',xrange=Paramlimits[Frame_Number,0:1],xstyle=1,yrange=Paramlimits[Z_ind,0:1],ystyle=1
			col=(250-jp*50)>50
			oplot,FR,ZFit,col=col
			oplot,subset[Frame_Number,*],subset[Z_ind,*],psym=3,col=col
			if FitMethod eq 0 then for i =0,fitorder do xyouts,0.8,0.95-0.02*i,zcoef[i],/normal
				if (n_elements(Ellipt_Fit) gt 2) and (total(abs(subset[Ell_ind,0:(100 <((size(subset))[2]-1))])) ne 0.0) then begin
				!p.multi=[1,1,2,0,0]
				plot,FR,Ellipt_Fit,xtitle='frame',ytitle='X-Y Ellipticity',xrange=Paramlimits[Frame_Number,0:1],xstyle=1,yrange=Paramlimits[Ell_ind,0:1],ystyle=1
				oplot,FR,Ellipt_Fit,col=col
				oplot,subset[Frame_Number,*],subset[Ell_ind,*],psym=3,col=col
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
pro OnTestZDrift_Astig, Event
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius

use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs')
use_multiple_GS = widget_info(use_multiple_GS_id,/button_set)
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
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size = min(where(RowNames eq '24 Group Size'))
Frame_Number = min(where(RowNames eq 'Frame Number'))

use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs')
use_multiple_GS = widget_info(use_multiple_GS_id,/button_set)
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

ReloadParamlists, Event, [Ell_ind, Z_ind, Gr_Ell_ind, GrZ_ind]

end
;
;-----------------------------------------------------------------
;
pro OnRemoveTilt_Astig, Event
Remove_XYZ_tilt
end
;
;-----------------------------------------------------------------
;
pro OnAddOffset_Astig, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

WidZPhaseOffsetID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_phase_offset')
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
pro OnButtonClose, Event
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnPickCalFile, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
wfilename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.sav'],title='Select *WND.sav file to open')
if wfilename ne '' then begin
	cd,fpath
	WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WindFilename_Astig')
	widget_control,WFileWidID,SET_VALUE = wfilename
	wfile_info=FILE_INFO(wfilename)
		if wfile_info.exists then begin
			z_unwrap_coeff = transpose([0.0,0.0,0.0])
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
pro OnPickGuideStarAncFile, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
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
pro WriteGudeStarRadius, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
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