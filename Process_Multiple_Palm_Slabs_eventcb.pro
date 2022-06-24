;
; IDL Event Callback Procedures
; Process_Multiple_Palm_Slabs_eventcb
;
; Generated on:	11/15/2017 21:21.14
;
;-----------------------------------------------------------------
; Notify Realize Callback Procedure.
; Argument:
;   wWidget - ID number of specific widget.
;
;
;   Retrieve the IDs of other widgets in the widget hierarchy using
;       id=widget_info(Event.top, FIND_BY_UNAME=name)
;
;-----------------------------------------------------------------
;
;
; Empty stub procedure used for autoloading.
;
pro Process_Multiple_Palm_Slabs_eventcb
end
;
;-----------------------------------------------------------------
;
pro OnCancel_ZvsV_mSlabs, Event
	widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro Initialize_Process_Multiple_PALM_Slabs, wWidget
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius

	if n_elements(Rundat_Filename) gt 0 then begin
		RunDat_File_WidID = Widget_Info(wWidget, find_by_uname='WID_TXT_RunDat_Filename_mSlabs')
		widget_control,RunDat_File_WidID,SET_VALUE = Rundat_Filename
	endif

	WID_TEXT_ZvsV_Slope_mSlabs_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_ZvsV_Slope_mSlabs')
	ZvsV_Slope_txt=string(ZvsV_Slope,FORMAT='(F8.2)')
	widget_control,WID_TEXT_ZvsV_Slope_mSlabs_ID,SET_VALUE = ZvsV_Slope_txt

	if n_elements(State_Voltages) gt 0 then begin
		WidTableID = Widget_Info(wWidget, find_by_uname='WID_Parameters_mSlabs')
		widget_control,WidTableID,ROW_LABELS=RNames, TABLE_YSIZE=n_elements(State_Voltages), $
			COLUMN_WIDTH=[100,100,130,100], $
			set_value=transpose([[State_Voltages], [State_Frames], [Transition_Frames], [State_Zs]])
	endif

end
;
;-----------------------------------------------------------------
;
pro On_Select_RunDat_File_mSlabs, Event
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope
	Rundat_Filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.dat'],title='Select Run Setup .dat file to open')
	if Rundat_Filename ne '' then begin
		cd,fpath
		RunDat_File_WidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_RunDat_Filename_mSlabs')
		widget_control,RunDat_File_WidID,SET_VALUE = Rundat_Filename
		On_Load_RunDat_mSlabs, Event
	endif
end
;
;-----------------------------------------------------------------
;
pro On_Load_RunDat_mSlabs, Event
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope
	RunDat_File_info=FILE_INFO(Rundat_Filename)
	FID=12
	if RunDat_File_info.exists then begin
		;Res = READ_ASCII(Rundat_Filename)
		text=''
		text2=''
		text3=''
		close,FID
		openr,FID,Rundat_Filename
		while (~ EOF(FID)) and (text ne 'Voltage States:')  do begin
			if (~ EOF(FID)) then readf,FID,text
			print,text
		endwhile
		if EOF(FID) then return
		j=0
		while (~ EOF(FID)) do begin
			vtarg= 'V'+strtrim(j,2)+'='
			ntarg= 'Nfr'+strtrim(j,2)+'='
			trtarg= 'Transition'+strtrim(j,2)+'='
			RNames = j eq 0 ? 'Z State '+strtrim(j,2) : [RNames, 'Z State '+strtrim(j,2)]
			;print, vtarg
			id = strmatch(text, vtarg+'*')
			while (~ EOF(FID)) and (id eq 0)  do begin
				if (~ EOF(FID)) then readf,FID,text
				id = strmatch(text, vtarg+'*')
				print,text
			endwhile
			if (~ EOF(FID)) then begin
				x = strmid(text,max(strsplit(text,'=')))
				V = float(strmid(x,0,strpos(x,'V')))
				State_Voltages = j eq 0 ? V : [State_Voltages, V]
				readf,FID,text2
				print,text2
				Nfr = fix(strmid(text2, strlen(ntarg)))
				State_Frames = j eq 0 ? Nfr : [State_Frames, Nfr]
				readf,FID,text3
				print,text3
				y = strmid(text3,max(strsplit(text3,'=')))
				Trt = fix(round(float(strmid(y,0,strpos(y,'s')))/0.025))
				Transition_Frames = j eq 0 ? Trt : [Transition_Frames, Trt]
				j++
			endif
		endwhile
		close,FID
;		Transition_Frames = State_Frames * 0
		State_Zs = State_Voltages * ZvsV_Slope
		State_Zs =  -1.0*State_Zs + State_Zs[0]
		WidTableID = Widget_Info(Event.top, find_by_uname='WID_Parameters_mSlabs')
		widget_control,WidTableID,ROW_LABELS=RNames,TABLE_YSIZE=n_elements(State_Voltages), $
			set_value=transpose([[State_Voltages], [State_Frames], [Transition_Frames], [State_Zs]])
	endif
end
;
;-----------------------------------------------------------------
;
pro On_Change_ZvsV_Slope_mSlabs, Event
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope
	widget_control, event.id, GET_VALUE = ZvsV_Slope_txt
	ZvsV_Slope = float(ZvsV_Slope_txt[0])
	State_Zs = State_Voltages * ZvsV_Slope
	State_Zs =  -1.0*State_Zs + State_Zs[0]
	WidTableID = Widget_Info(Event.top, find_by_uname='WID_Parameters_mSlabs')
	widget_control, WidTableID, ROW_LABELS=RNames, TABLE_YSIZE=n_elements(State_Voltages), $
		set_value=transpose([[State_Voltages], [State_Frames], [Transition_Frames], [State_Zs]])
end
;
;-----------------------------------------------------------------
;
pro Do_Change_Params_mSlabs, Event
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope
	widget_control, event.id, get_value=thevalue
	State_Voltages = thevalue[0,*]
	State_Frames = thevalue[1,*]
	Transition_Frames = thevalue[2,*]
	State_Zs = thevalue[3,*]
end
;
;-----------------------------------------------------------------
;
pro Assign_zStates_and_Shift_ZvsV_mSlabs, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[where(names eq 'WID_BASE_0_PeakSelector')]

	if n_elements(State_Voltages) le 1 then return ; no need to do anything if it is just one state or no states are loaded

	if n_elements(CGroupParams) le 2 then begin
		print,'No data loaded'
		return      ; if data not loaded return
	endif

	FrNum_ind = min(where(RowNames eq 'Frame Number'))                        ; CGroupParametersGP[9,*] - frame number
	Zindex = max(where(RowNames eq 'Z Position'))
	UnwZindex = max(where(RowNames eq 'Unwrapped Z'))
	GrZindex = max(where(RowNames eq 'Group Z Position'))
	UnwGrZindex = max(where(RowNames eq 'Unwrapped Group Z'))
	ZState_ind = max(where(RowNames eq 'Z State'))
	Zst_params=[ZState_ind,Zindex,UnwZindex,GrZindex,UnwGrZindex]

	Ncycle=total(State_Frames)
	offset = CGroupParams[UnwGrZindex,*]*0.0
	CGroupParams[ZState_ind,*]= 0
	fr = Cgroupparams[FrNum_ind,*] mod Ncycle

	for i=1, (n_elements(State_Voltages)-1) do begin
		Npref = total(State_Frames[0:(i-1)])
		ind = where((fr ge Npref) and (fr lt (Npref+State_Frames[i])))
		offset[ind] = State_Zs[i]
		CGroupParams[ZState_ind,ind]= i
	endfor
	CGroupParams[UnwZindex,*]= Cgroupparams[UnwZindex,*] + offset
	CGroupParams[UnwGrZindex,*]= Cgroupparams[UnwGrZindex,*] + offset

	sz=size(CGroupParams)
	nZpar=n_elements(Zst_params)-1
	for j=0,nZpar do ParamLimits[Zst_params[j],2]=Median(CGroupParams[Zst_params[j],*])
	ParamLimits[41,2]=(ParamLimits[41,2]>ParamLimits[35,2])
	for j=0,nZpar do ParamLimits[Zst_params[j],0] = min(CGroupParams[Zst_params[j],*])
	for j=0,nZpar do ParamLimits[Zst_params[j],1] = max(CGroupParams[Zst_params[j],*])
	for j=0,nZpar do ParamLimits[Zst_params[j],3] = ParamLimits[Zst_params[j],1] - ParamLimits[Zst_params[j],0]
	for j=0,nZpar do ParamLimits[Zst_params[j],2] = (ParamLimits[Zst_params[j],1] + ParamLimits[Zst_params[j],0])/2.

	LimitUnwrapZ

	print,'Z Operations: finished: Assigne Z-States and Voltages'
	TopIndex = (CGrpSize-1)
	TopID=ids[where(names eq 'WID_BASE_0_PeakSelector')]
	wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
	widget_control,wtable,set_value=transpose(ParamLimits[0:TopIndex,0:3]), use_table_select=[0,0,3,TopIndex]
	widget_control,wtable,set_value=transpose(CGroupParams[0:TopIndex,sz[2]/2]), use_table_select=[4,0,4,TopIndex]

end
;
;-----------------------------------------------------------------
;
pro Assign_zStates_mSlabs, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[where(names eq 'WID_BASE_0_PeakSelector')]

	if n_elements(State_Voltages) le 1 then return ; no need to do anything if it is just one state or no states are loaded

	if n_elements(CGroupParams) le 2 then begin
		print,'No data loaded'
		return      ; if data not loaded return
	endif

	FrNum_ind = min(where(RowNames eq 'Frame Number'))                        ; CGroupParametersGP[9,*] - frame number
	Zindex = max(where(RowNames eq 'Z Position'))
	UnwZindex = max(where(RowNames eq 'Unwrapped Z'))
	GrZindex = max(where(RowNames eq 'Group Z Position'))
	UnwGrZindex = max(where(RowNames eq 'Unwrapped Group Z'))
	ZState_ind = max(where(RowNames eq 'Z State'))
	Zst_params=[ZState_ind,Zindex,UnwZindex,GrZindex,UnwGrZindex]

	Ncycle=total(State_Frames)
;	offset = CGroupParams[UnwGrZindex,*]*0.0
	CGroupParams[ZState_ind,*]= 0
	fr = Cgroupparams[FrNum_ind,*] mod Ncycle

	for i=1, (n_elements(State_Voltages)-1) do begin
		Npref = total(State_Frames[0:(i-1)])
		ind = where((fr ge Npref) and (fr lt (Npref+State_Frames[i])))
	;	offset[ind] = State_Zs[i]
		CGroupParams[ZState_ind,ind]= i
	endfor
	;CGroupParams[UnwZindex,*]= Cgroupparams[UnwZindex,*] + offset
	;CGroupParams[UnwGrZindex,*]= Cgroupparams[UnwGrZindex,*] + offset

	print,'Z Operations: finished: Assigne Z-States'
	sz=size(CGroupParams)
	nZpar=n_elements(Zst_params)-1
	for j=0,nZpar do ParamLimits[Zst_params[j],2]=Median(CGroupParams[Zst_params[j],*])
	ParamLimits[41,2]=(ParamLimits[41,2]>ParamLimits[35,2])
	for j=0,nZpar do ParamLimits[Zst_params[j],0] = min(CGroupParams[Zst_params[j],*])
	for j=0,nZpar do ParamLimits[Zst_params[j],1] = max(CGroupParams[Zst_params[j],*])
	for j=0,nZpar do ParamLimits[Zst_params[j],3] = ParamLimits[Zst_params[j],1] - ParamLimits[Zst_params[j],0]
	for j=0,nZpar do ParamLimits[Zst_params[j],2] = (ParamLimits[Zst_params[j],1] + ParamLimits[Zst_params[j],0])/2.

	LimitUnwrapZ

	TopIndex = (CGrpSize-1)
	TopID=ids[where(names eq 'WID_BASE_0_PeakSelector')]
	wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
	widget_control,wtable,set_value=transpose(ParamLimits[0:TopIndex,0:3]), use_table_select=[0,0,3,TopIndex]
	widget_control,wtable,set_value=transpose(CGroupParams[0:TopIndex,sz[2]/2]), use_table_select=[4,0,4,TopIndex]

end
;
;-----------------------------------------------------------------
;
pro Assign_Transition_Frames_mSlabs, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[where(names eq 'WID_BASE_0_PeakSelector')]

	if n_elements(State_Voltages) le 1 then return ; no need to do anything if it is just one state or no states are loaded

	if n_elements(CGroupParams) le 2 then begin
		print,'No data loaded'
		return      ; if data not loaded return
	endif

	Assign_zStates_mSlabs, Event

	FrNum_ind = min(where(RowNames eq 'Frame Number'))                        ; CGroupParametersGP[9,*] - frame number
	Zindex = max(where(RowNames eq 'Z Position'))
	UnwZindex = max(where(RowNames eq 'Unwrapped Z'))
	GrZindex = max(where(RowNames eq 'Group Z Position'))
	UnwGrZindex = max(where(RowNames eq 'Unwrapped Group Z'))
	ZState_ind = max(where(RowNames eq 'Z State'))
	Zst_params=[ZState_ind,Zindex,UnwZindex,GrZindex,UnwGrZindex]

	Ncycle=total(State_Frames)
	print, Transition_Frames

	fr = Cgroupparams[FrNum_ind,*] mod Ncycle

	for i=0, (n_elements(State_Voltages)-1) do begin
		Npref = (i eq 0) ? 0 : total(State_Frames[0:(i-1)])
		ind = where((fr ge Npref) and (fr lt (Npref+Transition_Frames[i])))
		print,i, n_elements(ind)
		if min(ind) ge 0 then CGroupParams[ZState_ind,ind]= -1
	endfor

	sz=size(CGroupParams)
	nZpar=n_elements(Zst_params)-1
	for j=0,nZpar do ParamLimits[Zst_params[j],2]=Median(CGroupParams[Zst_params[j],*])
	ParamLimits[41,2]=(ParamLimits[41,2]>ParamLimits[35,2])
	for j=0,nZpar do ParamLimits[Zst_params[j],0] = min(CGroupParams[Zst_params[j],*])
	for j=0,nZpar do ParamLimits[Zst_params[j],1] = max(CGroupParams[Zst_params[j],*])
	for j=0,nZpar do ParamLimits[Zst_params[j],3] = ParamLimits[Zst_params[j],1] - ParamLimits[Zst_params[j],0]
	for j=0,nZpar do ParamLimits[Zst_params[j],2] = (ParamLimits[Zst_params[j],1] + ParamLimits[Zst_params[j],0])/2.

	LimitUnwrapZ

	print,'Z Operations: finished: Assign Transition Frames to Z State = -1'
	TopIndex = (CGrpSize-1)
	TopID=ids[where(names eq 'WID_BASE_0_PeakSelector')]
	wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
	widget_control,wtable,set_value=transpose(ParamLimits[0:TopIndex,0:3]), use_table_select=[0,0,3,TopIndex]
	widget_control,wtable,set_value=transpose(CGroupParams[0:TopIndex,sz[2]/2]), use_table_select=[4,0,4,TopIndex]

end
