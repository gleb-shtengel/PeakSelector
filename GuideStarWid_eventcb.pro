;
; IDL Event Callback Procedures
; GuideStar_eventcb
;
; Generated on:	11/02/2007 13:46.33
;
;
; Empty stub procedure used for autoloading.
;
pro GuideStarWid_eventcb
end
;
;-----------------------------------------------------------------
;
pro OnButtonClose, Event				; Closes the Menu Widget
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnTestGuideStarXY, Event	;Shows fit to guide star wo changing data:	XY coordinates
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius

use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleGuideStars_XY')
use_multiple_GS = widget_info(use_multiple_GS_id,/button_set)
use_multiple_GS_XY_DH_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleGuideStars_XY_DH')
use_multiple_GS_XY_DH = widget_info(use_multiple_GS_XY_DH_id,/button_set)
use_multiple_GS = use_multiple_GS or use_multiple_GS_XY_DH
WID_TEXT_XY_GuideStarAncFilename_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_XY_GuideStarAncFilename')
widget_control,WID_TEXT_XY_GuideStarAncFilename_ID,GET_VALUE = GS_anc_fname
GS_anc_file_info=FILE_INFO(GS_anc_fname)
use_multiple_GS = use_multiple_GS and GS_anc_file_info.exists
ExtractSubset, Event, xdrift, ydrift, use_multiple_GS

end
;
;-----------------------------------------------------------------
;
pro OnWriteGuideStarXY, Event	;Drift corrects x and y pix to constant guide star coordinates
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

X_ind = min(where(RowNames eq 'X Position'))                            ; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))                            ; CGroupParametersGP[3,*] - Peak Y  Position
FrNum_ind = min(where(RowNames eq 'Frame Number'))                        ; CGroupParametersGP[9,*] - frame number
GrX_ind = min(where(RowNames eq 'Group X Position'))                    ; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))                    ; CGroupParametersGP[20,*] - average y - position in the group

use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleGuideStars_XY')
use_multiple_GS = widget_info(use_multiple_GS_id,/button_set)
use_multiple_GS_XY_DH_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleGuideStars_XY_DH')
use_multiple_GS_XY_DH = widget_info(use_multiple_GS_XY_DH_id,/button_set)
use_multiple_GS = use_multiple_GS or use_multiple_GS_XY_DH

WID_TEXT_XY_GuideStarAncFilename_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_XY_GuideStarAncFilename')
widget_control,WID_TEXT_XY_GuideStarAncFilename_ID,GET_VALUE = GS_anc_fname
GS_anc_file_info=FILE_INFO(GS_anc_fname)
use_multiple_GS = use_multiple_GS and GS_anc_file_info.exists

WidDListDispFitMethodID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_XY_Fit_Method')
FitMethod = widget_info(WidDListDispFitMethodID,/DropList_Select)
if FitMethod eq 2 then begin
	drift_filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.sav'],title='Select *.sav file to load drift data')
	drift_file_info = FILE_INFO(drift_filename)
	if (drift_filename ne '') and drift_file_info.exists then begin
		cd, fpath
		restore, drift_filename
		xdrift = GStarCoeff.xdrift
		ydrift = GStarCoeff.ydrift
	endif else return
endif else ExtractSubset, Event, xdrift, ydrift, use_multiple_GS

CGroupParams[X_ind,*] = CGroupParams[X_ind,*] - xdrift[CGroupParams[FrNum_ind,*]]
CGroupParams[Y_ind,*] = CGroupParams[Y_ind,*] - ydrift[CGroupParams[FrNum_ind,*]]
CGroupParams[GrX_ind,*] = CGroupParams[GrX_ind,*] - xdrift[CGroupParams[FrNum_ind,*]]
CGroupParams[GrY_ind,*] = CGroupParams[GrY_ind,*] - ydrift[CGroupParams[FrNum_ind,*]]

if bridge_exists then begin
	print,'Reloading the Bridge Array'
	CATCH, Error_status
	CGroupParams_bridge = SHMVAR(shmName_data)
	CGroupParams_bridge[X_ind,*] = CGroupParams[X_ind,*]
	CGroupParams_bridge[Y_ind,*] = CGroupParams[Y_ind,*]
	CGroupParams_bridge[GrX_ind,*] = CGroupParams[GrX_ind,*]
	CGroupParams_bridge[GrY_ind,*] = CGroupParams[GrY_ind,*]
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

;NFrames=long64(max(CGroupParams[FrNum_ind,*]))
NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[FrNum_ind,*])+1))	: long64(max(CGroupParams[FrNum_ind,*])+1)
if n_elements(GuideStarDrift) eq 0 then begin
	GuideStarDrift={present:0B,xdrift:fltarr(NFrames),ydrift:fltarr(NFrames),zdrift:fltarr(NFrames)}
endif else begin
	if n_elements(GuideStarDrift[0].xdrift) ne NFrames then GuideStarDrift={present:0B,xdrift:fltarr(NFrames),ydrift:fltarr(NFrames),zdrift:fltarr(NFrames)}
endelse
GuideStarDrift[0].present = 1
GuideStarDrift[0].xdrift = GuideStarDrift[0].xdrift + xdrift
GuideStarDrift[0].ydrift = GuideStarDrift[0].ydrift + ydrift
end
;
;-----------------------------------------------------------------
;
pro ExtractSubset, Event, xdrift, ydrift, use_multiple_GS	;Pulls out subset of data from param limits and fits x,y vs frames
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

X_ind = min(where(RowNames eq 'X Position'))                            ; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))                            ; CGroupParametersGP[3,*] - Peak Y  Position
FrNum_ind = min(where(RowNames eq 'Frame Number'))                        ; CGroupParametersGP[9,*] - frame number
Nph_ind = min(where(RowNames eq '6 N Photons'))                            ; CGroupParametersGP[6,*] - Number of Photons in the Peak

!p.multi=[0,1,2,0,0]

WidSldFitOrderID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_XY_Fit')
widget_control,WidSldFitOrderID,get_value=fitorder

WidSmWidthID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_XY_Sm_Width')
widget_control,WidSmWidthID,get_value=SmWidth

WidDListDispFitMethodID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_XY_Fit_Method')
FitMethod = widget_info(WidDListDispFitMethodID,/DropList_Select)

use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleGuideStars_XY')
use_multiple_GS = widget_info(use_multiple_GS_id,/button_set)
use_multiple_GS_XY_DH_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleGuideStars_XY_DH')
use_multiple_GS_XY_DH = widget_info(use_multiple_GS_XY_DH_id,/button_set)

GS_anc_file_info=FILE_INFO(GS_anc_fname)
; if using multiple Guide Stars - load their X-Y coordinates
use_multiple_ANC = (use_multiple_GS or use_multiple_GS_XY_DH) and GS_anc_file_info.exists and (strlen(GS_anc_fname) gt 0)
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
FR=findgen(NFrames)

; original procedure (treat each Guide Star separetly and then average)
if (NOT use_multiple_GS_XY_DH) then begin
	for jp=0,(ip_cnt-1) do begin
		print,'fiducial #',jp
		filter=filter0
		if use_multiple_GS then begin
			ParamLimits[X_ind,2] = GSAnchorPnts[0,jp]
			ParamLimits[X_ind,0] = GSAnchorPnts[0,jp]-GS_radius
			ParamLimits[X_ind,1] = GSAnchorPnts[0,jp]+GS_radius
			ParamLimits[Y_ind,2] = GSAnchorPnts[1,jp]
			ParamLimits[Y_ind,0] = GSAnchorPnts[1,jp]-GS_radius
			ParamLimits[Y_ind,1] = GSAnchorPnts[1,jp]+GS_radius
		endif

		FilterIt

		subsetindex=where(filter eq 1,cnt)
		;print, 'subset has ',cnt,' points'

		if cnt gt 0 then begin
			subset=CGroupParams[*,subsetindex]
			if FitMethod eq 0 then begin
				xcoef=poly_fit(subset[FrNum_ind,*],subset[X_ind,*],fitorder,YFIT=fit_to_x)
				ycoef=poly_fit(subset[FrNum_ind,*],subset[Y_ind,*],fitorder,YFIT=fit_to_y)
				;print,xcoef,'=xcoef		',ycoef,'=ycoef'
				XFit=poly(FR,xcoef)
				YFit=poly(FR,ycoef)
			endif else begin
				indecis=uniq(subset[FrNum_ind,*])
				n_ele=n_elements(indecis)
				X1=subset[X_ind,indecis]
				Xsmooth=smooth(subset[6,indecis]*subset[X_ind,indecis],SmWidth,/EDGE_TRUNCATE)/smooth(subset[6,indecis],SmWidth,/EDGE_TRUNCATE)
				Xfit=interpol_gs(Xsmooth,subset[FrNum_ind,indecis],FR)
				Y1=subset[Y_ind,indecis]
				Ysmooth=smooth(subset[6,indecis]*subset[Y_ind,indecis],SmWidth,/EDGE_TRUNCATE)/smooth(subset[6,indecis],SmWidth,/EDGE_TRUNCATE)
				Yfit=interpol_gs(Ysmooth,subset[FrNum_ind,indecis],FR)
				frame_low = min(subset[FrNum_ind,indecis])
				ind_low=where(FR[*] lt frame_low,c)
				if c ge 1 then begin
					Yfit[ind_low]=Yfit[frame_low]
					Xfit[ind_low]=Xfit[frame_low]
				endif
				frame_high = max(subset[FrNum_ind,indecis])
				ind_high=where(FR[*] gt frame_high,c)
				if c ge 1 then begin
					Xfit[ind_high]=Xfit[frame_high]
					Yfit[ind_high]=Yfit[frame_high]
				endif
			endelse
			xdrift=Xfit-Xfit[0]
			ydrift=Yfit-Yfit[0]
			xdrift_mult = (jp eq 0) ? transpose(xdrift) : [xdrift_mult,transpose(xdrift)]
			ydrift_mult = (jp eq 0) ? transpose(ydrift) : [ydrift_mult,transpose(ydrift)]

			if jp eq 0 then begin
				!P.NOERASE=0
				xrange=Paramlimits[FrNum_ind,0:1]
				yrange_top=(Paramlimits[X_ind,0:1]-XFit[0])
				yrange_bot=(Paramlimits[Y_ind,0:1]-YFit[0])
				plot,FR,(XFit-XFit[0]),xtitle='frame',ytitle='xdrift(pixels)',xrange=xrange,xstyle=1,yrange=yrange_top,ystyle=1
				oplot,subset[FrNum_ind,*],(subset[X_ind,*]-XFit[0]),psym=3
				if FitMethod eq 0 then for i =0,fitorder do xyouts,0.8,0.95-0.02*i,xcoef[i],/normal
				plot,FR,(YFit-YFit[0]),xtitle='frame',ytitle='ydrift(pixels)',xrange=xrange,xstyle=1,yrange=yrange_bot,ystyle=1
				oplot,subset[FrNum_ind,*],(subset[Y_ind,*]-YFit[0]),psym=3
				if FitMethod eq 0 then for i =0,fitorder do xyouts,0.8,0.45-0.02*i,ycoef[i],/normal
				!P.NOERASE=1
			endif else begin
				!p.multi=[0,1,2,0,0]
				col=(250-jp*50)>50
				plot,FR,(XFit-XFit[0]),xtitle='frame',ytitle='xdrift(pixels)',xrange=xrange,xstyle=1,yrange=yrange_top,ystyle=1
				oplot,FR,(XFit-XFit[0]),col=col
				oplot,subset[FrNum_ind,*],(subset[X_ind,*]-XFit[0]),psym=3,col=col
				!p.multi=[1,1,2,0,0]
				plot,FR,(YFit-YFit[0]),xtitle='frame',ytitle='ydrift(pixels)',xrange=xrange,xstyle=1,yrange=yrange_bot,ystyle=1
				oplot,FR,(YFit-YFit[0]),col=col
				oplot,subset[FrNum_ind,*],(subset[Y_ind,*]-YFit[0]),psym=3,col=col
			endelse
		endif
	endfor
endif else begin
; New procedure (DPH) - average first
	superset_X = dblarr(ip_cnt, NFrames)
	superset_Y = dblarr(ip_cnt, NFrames)
	superset_Nph = superset_X
	superset_FR = superset_X
	dr_plotted=0

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
		print, jp,'  GuideStar subset has ',cnt,' points',  ',   X=',ParamLimits[2,2],', Y=',ParamLimits[3,2]
		if cnt gt 0 then begin
			subset = double(CGroupParams[*,subsetindex])
			subset[X_ind,*] = subset[X_ind,*] - mean(subset[X_ind,*],/DOUBLE)
			subset[Y_ind,*] = subset[Y_ind,*] - mean(subset[Y_ind,*],/DOUBLE)
			; remove secondary peaks in frames
			XY=subset[FrNum_ind,*]
			ind = reverse(n_elements(XY)-uniq(reverse(XY))-1)
			subset = temporary(subset[*,ind])
			superset_X[jp,subset[FrNum_ind,*]] = subset[X_ind,*]
			superset_Y[jp,subset[FrNum_ind,*]] = subset[Y_ind,*]
			superset_Nph[jp,subset[FrNum_ind,*]] = subset[Nph_ind,*]
			superset_FR[jp,*] = FR
			if dr_plotted eq 0 then begin
				dr_plotted=1
				!P.NOERASE=0
				yrng_X = [1.5*min(subset[X_ind,*])<0, 1.5*max(subset[X_ind,*])>0]
				plot,subset[FrNum_ind,*],subset[X_ind,*],xtitle='Frame',ytitle='X-drift (pixels)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1, yrange=yrng_X,ystyle=1, psym=3
				yrng_Y = [1.5*min(subset[Y_ind,*])<0, 1.5*max(subset[Y_ind,*])>0]
				plot,subset[FrNum_ind,*],subset[Y_ind,*],xtitle='Frame',ytitle='Y-drift (pixels)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1,yrange=yrng_Y,ystyle=1, psym=3
				!P.NOERASE=1
			endif else begin
				col=(250-jp*25)>50
				!p.multi=[0,1,2,0,0]
				plot,subset[FrNum_ind,*],subset[X_ind,*],xtitle='Frame',ytitle='X-drift (pixels)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1, yrange=yrng_X,ystyle=1, psym=3
				oplot,subset[FrNum_ind,*],subset[X_ind,*],col=col,psym=3
				!p.multi=[1,1,2,0,0]
				plot,subset[FrNum_ind,*],subset[Y_ind,*],xtitle='Frame',ytitle='Y-drift (pixels)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1,yrange=yrng_Y,ystyle=1, psym=3
				oplot,subset[FrNum_ind,*],subset[Y_ind,*],col=col,psym=3
			endelse
		endif
	endfor


	print,'SmWidth=', SmWidth

	if FitMethod eq 0 then begin
		non_zero_ind=where(superset_Nph gt 0, cnt_nz)
		xcoef=poly_fit(superset_Fr[non_zero_ind], superset_X[non_zero_ind], fitorder, YFIT=fit_to_x)
		XFit=poly(FR, xcoef)
		ycoef=poly_fit(superset_Fr[non_zero_ind], superset_Y[non_zero_ind], fitorder, YFIT=fit_to_x)
		YFit=poly(FR, ycoef)
	endif else begin
		Nph_arr = total(superset_Nph,1)
		X_arr = total(superset_X*superset_Nph,1)
		Y_arr = total(superset_Y*superset_Nph,1)
		non_zero_ind=where(Nph_arr gt 0, cnt_nz)
		Xsmooth=smooth(X_arr[non_zero_ind]/Nph_arr[non_zero_ind],SmWidth,/EDGE_TRUNCATE)
		Xfit=interpol_gs(Xsmooth,FR[non_zero_ind],FR)    ; interpol_gs is the same as interpol but forces the edge values outseide the range
		;Xfit[0:non_zero_ind[0]]=Xfit[non_zero_ind[0]]
		;Xfit[non_zero_ind[(cnt_nz-1)]:(NFrames-1)]=Xfit[non_zero_ind[(cnt_nz-1)]]
		Ysmooth=smooth(Y_arr[non_zero_ind]/Nph_arr[non_zero_ind],SmWidth,/EDGE_TRUNCATE)
		Yfit=interpol_gs(Ysmooth,FR[non_zero_ind],FR)
		;Yfit[0:non_zero_ind[0]]=Yfit[non_zero_ind[0]]
		;Yfit[non_zero_ind[(cnt_nz-1)]:(NFrames-1)]=Yfit[non_zero_ind[(cnt_nz-1)]]
	endelse

	!p.multi=[0,1,2,0,0]
	plot,FR,XFit,xtitle='Frame',ytitle='X-drift (pixels)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1, yrange=yrng_X, ystyle=1, thick=2
	oplot,FR,XFit, thick=2
	xdrift=Xfit-Xfit[0]
	!p.multi=[1,1,2,0,0]
	plot,FR,YFit,xtitle='Frame',ytitle='Y-drift (pixels)',xrange=Paramlimits[FrNum_ind,0:1],xstyle=1, yrange=yrng_Y ,ystyle=1, thick=2
	ydrift=Yfit-Yfit[0]
endelse
!p.multi=[0,0,0,0,0]
!p.background=0
!P.NOERASE=0

if use_multiple_ANC then begin
	if (NOT use_multiple_GS_XY_DH) then begin
		xdrift=total(xdrift_mult,1)/ip_cnt
		x_residual=dblarr(ip_cnt)
		for i=0,(ip_cnt-1) do x_residual[i]=(max(xdrift_mult[i,*]-xdrift)-min(xdrift_mult[i,*]-xdrift))*nm_per_pixel
		print,'residual x-uncertainties (nm):   ',x_residual
		xyouts,0.12,0.96,'X- residual uncertainties (nm):   ' + string(x_residual,FORMAT='(10(F8.2," "))'),color=200, charsize=1.5,CHARTHICK=1.0,/NORMAL
		ydrift=total(ydrift_mult,1)/ip_cnt
		y_residual=dblarr(ip_cnt)
		for i=0,(ip_cnt-1) do y_residual[i]=(max(ydrift_mult[i,*]-ydrift)-min(ydrift_mult[i,*]-ydrift))*nm_per_pixel
		print,'residual y-uncertainties (nm):   ',y_residual
		xyouts,0.12,0.46,'Y- residual uncertainties (nm):   ' + string(y_residual,FORMAT='(10(F8.2," "))'),color=200, charsize=1.5,CHARTHICK=1.0,/NORMAL
	endif else begin
		fri = Paramlimits[FrNum_ind,0]
		fra = Paramlimits[FrNum_ind,1]
		max_xdrift=(max(xdrift[fri:fra])-min(xdrift[fri:fra]))*nm_per_pixel
		xyouts,0.12,0.96,'X- drift (nm):   ' + string(max_xdrift,FORMAT='(F8.2)'),color=250, charsize=1.5,CHARTHICK=1.0,/NORMAL
		max_ydrift=(max(ydrift[fri:fra])-min(ydrift[fri:fra]))*nm_per_pixel
		xyouts,0.12,0.46,'Y- drift (nm):   ' + string(max_ydrift,FORMAT='(F8.2)'),color=250, charsize=1.5,CHARTHICK=1.0,/NORMAL
	endelse

endif else begin
	fri = Paramlimits[FrNum_ind,0]
	fra = Paramlimits[FrNum_ind,1]
	max_xdrift=(max(xdrift[fri:fra])-min(xdrift[fri:fra]))*nm_per_pixel
	xyouts,0.12,0.96,'X- drift (nm):   ' + string(max_xdrift,FORMAT='(F8.2)'),color=250, charsize=1.5,CHARTHICK=1.0,/NORMAL
	max_ydrift=(max(ydrift[fri:fra])-min(ydrift[fri:fra]))*nm_per_pixel
	xyouts,0.12,0.46,'Y- drift (nm):   ' + string(max_ydrift,FORMAT='(F8.2)'),color=250, charsize=1.5,CHARTHICK=1.0,/NORMAL
endelse

ParamLimits = ParamLimits0
filter = filter0

return
end
;
;-----------------------------------------------------------------
;
pro OnPick_XYGuideStarAncFile, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, names, modalList
GS_anc_fname = Dialog_Pickfile(/read,get_path=fpath,filter=['*.anc'],title='Select *.anc file to open')
GS_anc_file_info=FILE_INFO(GS_anc_fname)
if (GS_anc_fname ne '') and GS_anc_file_info.exists then begin
	cd,fpath
	WID_TEXT_XY_GuideStarAncFilename_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_XY_GuideStarAncFilename')
	widget_control,WID_TEXT_XY_GuideStarAncFilename_ID,SET_VALUE = GS_anc_fname
	WID_BASE_Z_operations_num=min(where(names eq 'WID_BASE_Z_operations'))
	if (WID_BASE_Z_operations_num ge 0) then begin
		WID_BASE_Z_operations_ID=ids[WID_BASE_Z_operations_num]
		WID_TEXT_GuideStarAncFilename_ID = Widget_Info(WID_BASE_Z_operations_ID, find_by_uname='WID_TEXT_GuideStarAncFilename')
		widget_control,WID_TEXT_GuideStarAncFilename_ID,SET_VALUE = GS_anc_fname
	endif
	WID_BASE_Z_operations_Astig_num=min(where(names eq 'WID_BASE_Z_operations_Astig'))
	if (WID_BASE_Z_operations_Astig_num ge 0) then begin
		WID_BASE_Z_operations_Astig_ID=ids[WID_BASE_Z_operations_Astig_num]
		WID_TEXT_GuideStarAncFilename_Astig_ID = Widget_Info(WID_BASE_Z_operations_Astig_ID, find_by_uname='WID_TEXT_GuideStarAncFilename_Astig')
		widget_control,WID_TEXT_GuideStarAncFilename_Astig_ID,SET_VALUE = GS_anc_fname
	endif
endif else GS_anc_fname = ''
end
;
;-----------------------------------------------------------------
;
pro Write_XY_GudeStarRadius, Event
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, names, modalList
WID_TEXT_XY_GuideStar_RadiusID = Widget_Info(Event.top, find_by_uname='WID_TEXT_XY_GuideStar_Radius')
widget_control,WID_TEXT_XY_GuideStar_RadiusID,GET_VALUE = GS_radius_txt
WID_BASE_Z_operations_num=min(where(names eq 'WID_BASE_Z_operations'))
if WID_BASE_Z_operations_num ge 0 then begin
	WID_BASE_Z_operations_ID=ids[WID_BASE_Z_operations_num]
	WID_TEXT_GuideStar_Radius_ID = Widget_Info(WID_BASE_Z_operations_ID, find_by_uname='WID_TEXT_GuideStar_Radius')
	widget_control,WID_TEXT_GuideStar_Radius_ID,SET_VALUE = GS_radius_txt
endif
WID_BASE_Z_operations_Astig_num=min(where(names eq 'WID_BASE_Z_operations_Astig'))
if (WID_BASE_Z_operations_Astig_num ge 0) then begin
	WID_BASE_Z_operations_Astig_ID=ids[WID_BASE_Z_operations_Astig_num]
	WID_TEXT_GuideStar_Radius_Astig_ID = Widget_Info(WID_BASE_Z_operations_Astig_ID, find_by_uname='WID_TEXT_GuideStar_Radius_Astig')
	widget_control,WID_TEXT_GuideStar_Radius_Astig_ID,SET_VALUE = GS_radius_txt
endif
GS_radius = float(GS_radius_txt[0])
end
;
;-----------------------------------------------------------------
;
pro Initialize_XY_GuideStar, wWidget
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

WID_TEXT_XY_GuideStar_RadiusID = Widget_Info(wWidget, find_by_uname='WID_TEXT_XY_GuideStar_Radius')
GS_radius_txt=string(GS_radius[0],FORMAT='(F6.2)')
widget_control,WID_TEXT_XY_GuideStar_RadiusID,SET_VALUE = GS_radius_txt

WID_TEXT_XY_GuideStarAncFilename_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_XY_GuideStarAncFilename')
widget_control,WID_TEXT_XY_GuideStarAncFilename_ID,SET_VALUE = GS_anc_fname

end
;
;-----------------------------------------------------------------
;
pro OnButtonPress_UseMultipleGuideStars_XY, Event
	use_multiple_GS_XY_DH_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleGuideStars_XY_DH')
	if use_multiple_GS_XY_DH_id gt 0 then if Event.select then widget_control, use_multiple_GS_XY_DH_id, set_button=0
end
;
;-----------------------------------------------------------------
;
pro OnButtonPress_UseMultipleGuideStars_XY_DH, Event
	use_multiple_GS_id = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_UseMultipleGuideStars_XY')
	if use_multiple_GS_id gt 0 then if Event.select then widget_control, use_multiple_GS_id, set_button=0
end
