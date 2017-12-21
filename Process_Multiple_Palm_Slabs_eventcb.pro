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
pro OnCancel_Macro_mSlabs, Event
	widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro Initialize_Process_Multiple_PALM_Slabs, wWidget
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
	Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs

	WID_BUTTON_PerformFiltering_mSlabs_ID = Widget_Info(wWidget, find_by_uname = 'WID_BUTTON_PerformFiltering_mSlabs')
		widget_control, WID_BUTTON_PerformFiltering_mSlabs_ID, set_button = DoFilter
	WID_BUTTON_PerformPurging_mSlabs_ID = Widget_Info(wWidget, find_by_uname = 'WID_BUTTON_PerformPurging_mSlabs')
		widget_control, WID_BUTTON_PerformPurging_mSlabs_ID, set_button = DoPurge
	WID_BUTTON_AutoFindFiducials_mSlabs_ID = Widget_Info(wWidget, find_by_uname = 'WID_BUTTON_AutoFindFiducials_mSlabs')
		widget_control, WID_BUTTON_AutoFindFiducials_mSlabs_ID, set_button = DoAutoFindFiducials

	WID_BUTTON_DriftCorrect_mSlabs_ID = Widget_Info(wWidget, find_by_uname = 'WID_BUTTON_DriftCorrect_mSlabs')
		widget_control, WID_BUTTON_DriftCorrect_mSlabs_ID, set_button = DoDriftCottect
	WID_BUTTON_Group_mSlabs_ID = Widget_Info(wWidget, find_by_uname = 'WID_BUTTON_Group_mSlabs')
		widget_control, WID_BUTTON_Group_mSlabs_ID, set_button = DoGrouping
	WID_BUTTON_RegisterToScaffold_mSlabss_ID = Widget_Info(wWidget, find_by_uname = 'WID_BUTTON_RegisterToScaffold_mSlabs')
		widget_control, WID_BUTTON_RegisterToScaffold_mSlabss_ID, set_button = DoScaffoldRegister

	WID_Filter_Parameters_mSlabs_ID = Widget_Info(wWidget, find_by_uname = 'WID_Filter_Parameters_mSlabs')
		n_par = n_elements(Filter_RowNames)
		widget_control, WID_Filter_Parameters_mSlabs_ID, ROW_LABELS = Filter_RowNames, TABLE_YSIZE = n_par
		widget_control, WID_Filter_Parameters_mSlabs_ID, COLUMN_WIDTH=[160,85,85],use_table_select = [ -1, 0, 1, (n_par-1) ]
		widget_control, WID_Filter_Parameters_mSlabs_ID, set_value=transpose(Filter_Params);, use_table_select=[0,0,3,(CGrpSize-1)]
		widget_control, WID_Filter_Parameters_mSlabs_ID, /editable,/sensitive

	WID_Purge_Parameters_mSlabs_ID = Widget_Info(wWidget, find_by_uname = 'WID_Purge_Parameters_mSlabs')
		n_par = n_elements(Filter_RowNames)
		widget_control, WID_Purge_Parameters_mSlabs_ID, ROW_LABELS = Purge_RowNames_mSlabs, TABLE_YSIZE = n_par
		widget_control, WID_Purge_Parameters_mSlabs_ID, COLUMN_WIDTH=[160,85,85],use_table_select = [ -1, 0, 1, (n_par-1) ]
		widget_control, WID_Purge_Parameters_mSlabs_ID, set_value=transpose(Purge_Params_mSlabs);, use_table_select=[0,0,3,(CGrpSize-1)]
		widget_control, WID_Purge_Parameters_mSlabs_ID, /editable,/sensitive

	WID_AutoFindFiducials_Parameters_mSlabs_ID = Widget_Info(wWidget, find_by_uname = 'WID_AutoFindFiducials_Parameters_mSlabs')
		widget_control, WID_AutoFindFiducials_Parameters_mSlabs_ID, set_value=transpose(AutoFindFiducial_Params)
		widget_control, WID_AutoFindFiducials_Parameters_mSlabs_ID, /editable,/sensitive

	WidDListGroupEngine = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_GroupEngine_mSlabs')
		widget_control,WidDListGroupEngine,SET_DROPLIST_SELECT = TransformEngine			;Set the default value to Local for Windows, and Cluster for UNIX
	wGrpGapID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Group_Gap_mSlabs')
		widget_control,wGrpGapID,set_value = grouping_gap
	wGrpGapID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Grouping_Radius_mSlabs')
		widget_control,wGrpGapID,set_value=grouping_radius100

	WID_SLIDER_FramesPerNode_ID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_FramesPerNode_mSlabs')
		increment = 500
		widget_control,WID_SLIDER_FramesPerNode_ID,set_value=increment

	WID_TEXT_ZStep_mSlabs_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_ZStep_mSlabs')
		ZStep_mSlabs_txt=string(ZStep_mSlabs,FORMAT='(F8.2)')
		widget_control,WID_TEXT_ZStep_mSlabs_ID,SET_VALUE = ZStep_mSlabs_txt
end
;
;-----------------------------------------------------------------
;
pro Do_Change_Filter_Params_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
	Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	widget_control,event.id,get_value=thevalue
	Filter_Params[event.y,event.x]=thevalue[event.x,event.y]
	widget_control, event.id, set_value=transpose(Filter_Params)
end
;
;-----------------------------------------------------------------
;
pro DoInsert_Autodetect_Param_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
	Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	widget_control,event.id,get_value=thevalue
	AutoFindFiducial_Params[event.y]=thevalue[event.x,event.y]
	widget_control,event.id,set_value=transpose(AutoFindFiducial_Params)
end
;
;-----------------------------------------------------------------
;
pro Do_Change_Purge_Params_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
	Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	widget_control,event.id,get_value=thevalue
	Purge_Params_mSlabs[event.y,event.x]=thevalue[event.x,event.y]
	widget_control, event.id, set_value=transpose(Purge_Params_mSlabs)
end
;
;-----------------------------------------------------------------
;
pro On_Change_ZStep_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
	Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	WID_TEXT_ZStep_mSlabs_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_ZStep_mSlabs')
	widget_control,WID_TEXT_ZStep_mSlabs_ID,GET_VALUE = ZStep_mSlabs_txt
	ZStep_mSlabs = float(ZStep_mSlabs_txt[0])
	print,"Set ZStep_mSlabs=", ZStep_mSlabs
end
;
;-----------------------------------------------------------------
;
pro OnSet_Button_PerformFiltering_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
		Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	DoFilter = Event.select
	print,"Set DoFilter=", DoFilter
end
;
;-----------------------------------------------------------------
;
pro OnSet_Button_PerformPurging_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
		Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	DoPurge = Event.select
	print,"Set DoPurge=", DoPurge
end
;
;-----------------------------------------------------------------
;
pro OnSet_Button_AutoFindFiducials_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
		Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	DoAutoFindFiducials = Event.select
	print,"Set DoAutoFindFiducials=", DoAutoFindFiducials
end
;
;-----------------------------------------------------------------
;
pro OnSet_Button_DriftCorrect_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
		Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	DoDriftCottect = Event.select
	print,"Set DoDriftCottect=", DoDriftCottect
end
;
;-----------------------------------------------------------------
;
pro OnSet_Button_PerfromGrouping_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
		Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	DoGrouping = Event.select
	print,"Set DoGrouping=", DoGrouping
end
;
;-----------------------------------------------------------------
;
pro OnSet_Button_RegistertoScaffold_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
		Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	DoScaffoldRegister = Event.select
	print,"Set DoScaffoldRegister=", DoScaffoldRegister
end
;
;-----------------------------------------------------------------
;
pro OnPick_ScaffoldFiducials_File, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
		Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
	Scaffold_Fid_FName = Dialog_Pickfile(/read,get_path=fpath,filter=['*.anc'],title='Select *.anc file to open')
	if Scaffold_Fid_FName ne '' then cd,fpath
	AncFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_ScaffoldFiducialsFile_mSlabs')
	widget_control, AncFileWidID, SET_VALUE = Scaffold_Fid_FName

	ScFileInfo=file_info(Scaffold_Fid_FName)
	if ~ScFileInfo.exists then return
	AnchorPnts=dblarr(6,AnchPnts_MaxNum)
	AnchorPnts_line=dblarr(6)
	ZPnts=dblarr(3,AnchPnts_MaxNum)
	ZPnts_line=dblarr(3)
	close,5
	openr,5,Scaffold_Fid_FName
	ip=0
	while (not EOF(5)) and (ip lt AnchPnts_MaxNum) do begin
		readf,5,AnchorPnts_line
		AnchorPnts[*,ip] = AnchorPnts_line
		ip+=1
	endwhile
	close,5

	Sc_Z_FileInfo=file_info(Scaffold_Fid_FName+'z')
	if Sc_Z_FileInfo.exists then begin
		ip=0
		openr,5,(Scaffold_Fid_FName+'z')
		while (not EOF(5)) and (ip lt AnchPnts_MaxNum)  do begin
			readf,5,ZPnts_line
			ZPnts[*,ip] = ZPnts_line
			ip+=1
		endwhile
		close,5
	endif
end
;
;-----------------------------------------------------------------
;
pro On_Select_Directory_mSlabs, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
	fpath = Dialog_Pickfile(/read,/DIRECTORY)
	WID_TXT_mSlabs_Directory_ID = Widget_Info(Event.Top, find_by_uname='WID_TXT_mSlabs_Directory')
		widget_control,WID_TXT_mSlabs_Directory_ID,SET_VALUE = fpath
		if fpath ne '' then cd,fpath
end
;
;-----------------------------------------------------------------
;
pro On_ReFind_Files_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
		Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

WID_TXT_mSlabs_Directory_ID = Widget_Info(Event.Top, find_by_uname='WID_TXT_mSlabs_Directory')
	widget_control,WID_TXT_mSlabs_Directory_ID,GET_VALUE = fpath

WID_TXT_mSlabs_FileMask_ID = Widget_Info(Event.Top, find_by_uname='WID_TXT_mSlabs_FileMask')
	widget_control,WID_TXT_mSlabs_FileMask_ID,GET_VALUE = fmask

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	PRINT, 'Error index: ', Error_status
	PRINT, 'Error message: ', !ERROR_STATE.MSG
	CATCH, /CANCEL
	return
ENDIF

widget_control,/hourglass
mSlab_Filenames = FILE_SEARCH(fpath + sep + fmask )

WID_LIST_Process_mSlabs_ID = Widget_Info(Event.Top, find_by_uname='WID_LIST_Process_mSlabs')
	widget_control,WID_LIST_Process_mSlabs_ID,SET_VALUE = mSlab_Filenames
WID_LABEL_nfiles_mSlabs_ID = Widget_Info(Event.Top, find_by_uname='WID_LABEL_nfiles_mSlabs')
	nfiles_text = string(n_elements(mSlab_Filenames))+' files (total)'
	widget_control,WID_LABEL_nfiles_mSlabs_ID,SET_VALUE=nfiles_text

CATCH, /CANCEL

end
;
;-----------------------------------------------------------------
;
pro On_Remove_Selected_mSlabs, Event
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
		Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs

	CATCH, Error_status
  	WID_LIST_Process_mSlabs_ID = Widget_Info(Event.Top, find_by_uname='WID_LIST_Process_mSlabs')
		ind_selected = widget_info(WID_LIST_Process_mSlabs_ID, /LIST_SELECT)

	if n_elements(ind_selected) eq 1 then if ind_selected eq -1 then return
	mSlab_Filenames[ind_selected] = '-1'
	mSlab_Filenames = mSlab_Filenames[where(mSlab_Filenames ne '-1')]

	widget_control,WID_LIST_Process_mSlabs_ID,SET_VALUE = mSlab_Filenames
		new_sel = max(ind_selected)+1 < n_elements(mSlab_Filenames)-1
	widget_control,WID_LIST_Process_mSlabs_ID, SET_LIST_SELECT=new_sel

	nfiles_text = string(n_elements(mSlab_Filenames))+' files'
	WID_LABEL_nfiles_mSlabs_ID = Widget_Info(Event.Top, find_by_uname='WID_LABEL_nfiles_mSlabs')
		widget_control,WID_LABEL_nfiles_mSlabs_ID, SET_VALUE=nfiles_text
	IF Error_status NE 0 THEN BEGIN
		print,text, 'cannot remove files'
		PRINT, 'Error index: ', Error_status
		PRINT, 'Error message: ', !ERROR_STATE.MSG
		CATCH, /CANCEL
		return
	ENDIF
end
;
;-----------------------------------------------------------------
;
pro Start_Macro_mSlabs, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
		Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs

n_files = n_elements(mSlab_Filenames)
if n_files le 0 then begin
	print,'Select '
	return
endif



for i=0, n_files-1 do begin

print,'started processing slab #,',i,'of ',(n_files-1)
; Step 1: Initial filtering
	if NOT DoFilter then begin
		print, 'Skipping the step "Perform Filtering"'
	endif else begin
		print, 'Starting the step "Perform Filtering"'
		;	filter the data set
		;   Filter_RowNames, Filter_Params
		param_num = n_elements(Filter_RowNames)
		if param_num gt 0 then begin
			params = intarr(param_num)
			for j = 0, param_num-1 do params[j] = min(where(RowNames eq Filter_RowNames[j]))
            CGPsz = size(CGroupParams)
            low  = Filter_Params[*,0]#replicate(1,CGPsz[2])
            high = Filter_Params[*,1]#replicate(1,CGPsz[2])
            filter = (CGroupParams[params,*] ge low) and (CGroupParams[params,*] le high)
		endif
	endelse



; Step 2: Auto Detect Fiducials
	if NOT DoAutoFindFiducials then begin
		print, 'Skipping the step "Auto Detect Fiducials"'
	endif else begin
		print, 'Starting the step "Auto Detect Fiducials"'


	endelse



; Step 3: Perform Drift Correction
	if NOT DoDriftCottect then begin
		print, 'Skipping the step "Perform Drift Correction"'
	endif else begin
		print, 'Starting the step "Perform Drift Correction"'


	endelse



; Step 4: Perform Purging
	if NOT DoPurge then begin
		print, 'Skipping the step "Perform Purging"'
	endif else begin
		print, 'Starting the step "Perform Purging"'
		; purge the data set
		; Purge_RowNames_mSlabs, Purge_Params_mSlabs
		param_num = n_elements(Purge_RowNames_mSlabs)
		if param_num gt 0 then begin
			params = intarr(param_num)
			for j = 0, param_num-1 do params[j] = min(where(RowNames eq Purge_RowNames_mSlabs[j]))
            CGPsz = size(CGroupParams)
            low  = Purge_Params_mSlabs[*,0]#replicate(1,CGPsz[2])
            high = Purge_Params_mSlabs[*,1]#replicate(1,CGPsz[2])
            filter = (CGroupParams[params,*] ge low) and (CGroupParams[params,*] le high)
            indecis = where (floor(total(temporary(filter), 1) / n_elements(params)) gt 0)
			CGroupParams = CGroupParams[*, indecis]
		endif
	endelse



; Step 5: Perform Grouping
	if NOT DoGrouping then begin
		print, 'Skipping the step "Perform Grouping"'
	endif else begin
		print, 'Starting the step "Perform Grouping"'


	endelse



; Step 6: Perform Register to Scaffold
	if NOT DoScaffoldRegister then begin
		print, 'Skipping the step "Register to Scaffold"'
	endif else begin
		print, 'Starting the step "Register to Scaffold"'


		print, 'Concatenating summary array'
		if i eq 0 then begin
			if n_files ne 1 then CGroupParams_tot = CGroupParams
		endif else begin
			if i ne (n_files-1) then CGroupParams_tot = [CGroupParams_tot, CGroupParams] else CGroupParams = [CGroupParams_tot, CGroupParams]
		endelse
	endelse

print,'finished processing slab #,',i,'of ',(n_files-1)
endfor

end




