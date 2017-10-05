;
; Empty stub procedure used for autoloading.
;
pro Save_data_ASCII_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_Save_Data_ASCII, wWidget
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WID_Base0_Peakselector, ID:TopID, TOP:TopID, HANDLER:TopID }

	WID_ID_DROPLIST_Save_ASCII_Filter = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Save_ASCII_Filter')
	widget_control,WID_ID_DROPLIST_Save_ASCII_Filter,SET_DROPLIST_SELECT = SaveASCII_Filter
	WID_ID_WID_DROPLIST_Save_ASCII_XY = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Save_ASCII_XY')
	widget_control,WID_ID_WID_DROPLIST_Save_ASCII_XY,SET_DROPLIST_SELECT = SaveASCII_units
	WID_ID_WID_DROPLIST_Save_ASCII_Parameters = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Save_ASCII_Parameters')
	widget_control,WID_ID_WID_DROPLIST_Save_ASCII_Parameters,SET_DROPLIST_SELECT = SaveASCII_ParamChoice

	WidLabel0 = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
	widget_control,WidLabel0,get_value=dfilename
	SaveASCII_Filename = AddExtension(dfilename,'_ASCII.txt')
	WidID_TEXT_ASCII_Filename = Widget_Info(wWidget, find_by_uname='WID_TEXT_ASCII_Filename')
	widget_control,WidID_TEXT_ASCII_Filename,SET_VALUE = SaveASCII_Filename,/EDITABLE

	WidID_TEXT_ASCII_Save_Parameter_List = Widget_Info(wWidget, find_by_uname='WID_TEXT_ASCII_Save_Parameter_List')
	widget_control,WidID_TEXT_ASCII_Save_Parameter_List,SET_VALUE = string(SaveASCII_ParamList),/NO_NEWLINE,/EDITABLE
end
;
;-----------------------------------------------------------------
;
pro On_Select_SaveASCII_Filter, Event
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
	WID_ID_DROPLIST_Save_ASCII_Filter = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Save_ASCII_Filter')
	SaveASCII_Filter = widget_info(WID_ID_DROPLIST_Save_ASCII_Filter,/DropList_Select)
end
;
;-----------------------------------------------------------------
;
pro On_Select_SaveASCII_units, Event
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
	WID_ID_WID_DROPLIST_Save_ASCII_XY = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Save_ASCII_XY')
	SaveASCII_units = widget_info(WID_ID_WID_DROPLIST_Save_ASCII_XY,/DropList_Select)
end
;
;-----------------------------------------------------------------
;
pro On_Select_SaveASCII_ParamChoice, Event
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
	WID_ID_WID_DROPLIST_Save_ASCII_Parameters = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Save_ASCII_Parameters')
	SaveASCII_ParamChoice = widget_info(WID_ID_WID_DROPLIST_Save_ASCII_Parameters,/DropList_Select)
end
;
;-----------------------------------------------------------------
;
pro On_ASCII_Filename_change, Event
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
	WidID_TEXT_ASCII_Filename = Widget_Info(Event.Top, find_by_uname='WID_TEXT_ASCII_Filename')
	widget_control,WidID_TEXT_ASCII_Filename,GET_VALUE = SaveASCII_Filename
end
;
;-----------------------------------------------------------------
;
pro On_ASCII_ParamList_change, Event
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
	WidID_TEXT_ASCII_Save_Parameter_List = Widget_Info(Event.Top, find_by_uname='WID_TEXT_ASCII_Save_Parameter_List')
	widget_control,WidID_TEXT_ASCII_Save_Parameter_List,GET_VALUE = SaveASCII_ParamString
	len=strlen(SaveASCII_ParamString)
	i=0 & j=0
	while i lt len-1 do begin
		chr=STRMID(SaveASCII_ParamString,i,1)
		;print,i,'  chr=',chr,' byte=' , byte(chr)
		if (byte(chr) ne 32) and  (byte(chr) ne 9) then begin
			ParamStr=chr
			while ((i lt len) and (byte(chr) ne 32) and  (byte(chr) ne 9)) do begin
				i++
				chr=STRMID(SaveASCII_ParamString,i,1)
				ParamStr=ParamStr+chr
			endwhile
			if j eq 0 then SaveASCII_ParamList = fix(strcompress(ParamStr)) else SaveASCII_ParamList = [SaveASCII_ParamList,fix(strcompress(ParamStr))]
			j++
		endif
		i++
	endwhile
end
;
;-----------------------------------------------------------------
;
pro OnCancel_SAVE_ASCII, Event
	widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro Save_ASCII, Event		;Save the presently loaded & modified peak parameters into an ascii file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common Zdisplay, Z_scale_multiplier, vbar_top
common Offset, PkWidth_offset
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames

X_ind = min(where(RowNames eq 'X Position'))
Y_ind = min(where(RowNames eq 'Y Position'))
Xwid_ind = min(where(RowNames eq 'X Peak Width'))
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))
GrX_ind = min(where(RowNames eq 'Group X Position'))
GrY_ind = min(where(RowNames eq 'Group Y Position'))
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))

lat_ind = [X_ind, Y_ind, Xwid_ind, Ywid_ind, SigNphX_ind, SigNphY_ind, SigX_ind, SigY_ind, $
					GrX_ind, GrY_ind, GrSigX_ind, GrSigY_ind]

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

if SaveASCII_Filename eq '' then begin
	z=dialog_message('Empty filename')
	return
endif

if SaveASCII_Filter then begin
	GroupFilterIt
	FilteredIndex=where((filter eq 1) and (CGroupParams[25,*] eq 1))
endif else begin
	FilterIt
	FilteredIndex=where(filter eq 1)
endelse

if FilteredIndex[0] eq -1 then begin
	z=dialog_message('Zero Peaks/groups selected')
	return
endif					;If no peaks in filter then return

FGroupParams=CGroupParams[*,FilteredIndex]

if SaveASCII_units then begin
	for i=0, (n_elements(lat_ind)-1) do begin
		if lat_ind[i] ge 0 then FGroupParams[lat_ind[i]]*=nm_per_pixel
	endfor
endif

if SaveASCII_ParamChoice ge 1 then indecis = SaveASCII_ParamList[where(SaveASCII_ParamList lt CGrpSize)] else indecis = indgen(CGrpSize)
if n_elements(indecis) eq 0 then begin
	z=dialog_message('Select Valid indecis')
	return
endif

if SaveASCII_ParamChoice eq 2 then begin
	CGrpSize = n_elements(indecis)
	RowNames = RowNames[indecis]
	ParamLimits = ParamLimits[indecis,*]
	CGroupParams = FGroupParams[indecis,*]
	filename=AddExtension(SaveASCII_Filename,'_IDL.sav')
	save, CGroupParams, CGrpSize, ParamLimits, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames, MLRawFilenames,$
		GuideStarDrift, FiducialCoeff, FlipRotate, thisfitcond, RowNames, $
		lambda_vac,nd_water, nd_oil, wind_range, z_unwrap_coeff, ellipticity_slopes, $
		nm_per_pixel, wfilename, PkWidth_offset, Z_scale_multiplier, grouping_gap, grouping_radius100, $
		lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames, sp_dispersion,  sp_offset, filename=filename
endif else begin
	Title_String=RowNames[indecis[0]]
	for i=1,(n_elements(indecis)-1) do Title_String=Title_String+'	'+RowNames[indecis[i]]
	openw,1,SaveASCII_Filename,width=1024
	printf,1,Title_String
	printf,1,FGroupParams[indecis,*],FORMAT='('+strtrim((n_elements(indecis)-1),2)+'(E13.5,"'+string(9B)+'"),E13.5)'
	close,1
endelse
end
;-----------------------------------------------------------------


