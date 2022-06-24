;
; Empty stub procedure used for autoloading.
;
pro Transform_Extract_ReExtract_Filter_GetZ_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_Transform_Extract_ReExtract_Filter_GetZ_iPALM, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters

nlbls=3
RawFilenames = strarr(nlbls)
current_thisfitcond=thisfitcond
LoadThiFitCond,ini_filename,thisfitcond

IF LMGR(/VM) then TransformEngine=0	; Set TransformEngine=0 if  IDL is in Virtual Machine Mode

if n_tags(thisfitcond) eq n_tags(current_thisfitcond) then thisfitcond=current_thisfitcond; check if the original thisfitcond is up to date, if not - load default.

	WidDListDispLevel = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_TransformEngine_iPALM')
	widget_control,WidDListDispLevel,SET_DROPLIST_SELECT = TransformEngine
	WID_Use_InfoFile_Flip = Widget_Info(wWidget, find_by_uname = 'WID_Use_InfoFile_Flip')
	;widget_control,WID_Use_InfoFile_Flip,set_button = !VERSION.OS_family eq 'unix' ? 1 : 0
	widget_control,WID_Use_InfoFile_Flip,set_button = 1
	AnchorPnts=dblarr(6,AnchPnts_MaxNum)
	ZPnts=dblarr(3,AnchPnts_MaxNum)

	increment=2500 ; Frames Per Node for grouping
	WidFramesPerNode = Widget_Info(wWidget, find_by_uname='WID_SLIDER_FramesPerNode_iPALM')
	widget_control,WidFramesPerNode,set_value=increment
	wGrpGapID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Group_Gap_iPALM')
	widget_control,wGrpGapID,set_value=grouping_gap
	wGrpRadID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Grouping_Radius_iPALM')
	widget_control,wGrpRadID,set_value=grouping_radius100

	; iPALM_MacroParameters = [Nph,	Full Sigma X Min. (pix.),	Full Sigma X Min. (pix.),	Sigma Z Max. (nm), Coherence Min.,	Coherence Max,	...
	; iPALM_MacroParameters = ...	Max A for (Wx-A)*(Wx-A)<B,	B for (Wx-A)*(Wx-A)<B,	Min A  for (Wx-A)*(Wx-A)>B,	B for (Wx-A)*(Wx-A)>B ]
	iPALM_MacroParameters = thisfitcond.SigmaSym ?	iPALM_MacroParameters_XY	:	iPALM_MacroParameters_R
	;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep

	WID_iPALM_MacroParameters_ID = Widget_Info(wWidget, find_by_uname='WID_Filter_Parameters_iPALM_Macro')
	widget_control,WID_iPALM_MacroParameters_ID,set_value=transpose(iPALM_MacroParameters), use_table_select=[0,0,0,(n_elements(iPALM_MacroParameters)-1)]
	widget_control,WID_iPALM_MacroParameters_ID,COLUMN_WIDTH=[180,120],use_table_select = [ -1, 0, 0, (n_elements(iPALM_MacroParameters)-1) ]

	TABLE_InfoFile_iPALM_macro_ID = Widget_Info(wWidget, find_by_uname='WID_TABLE_InfoFile_iPALM_macro')
	if !VERSION.OS_family eq 'unix' then widget_control,TABLE_InfoFile_iPALM_macro_ID,COLUMN_WIDTH=[150,100],use_table_select = [ -1, 0, 0, 25 ]
	Event={ID:wWidget,TOP:wWidget,HANDLER:wWidget}
	Fill_Parameters_iPALM_Macro, Event

end
;
;-----------------------------------------------------------------
;
pro OnCanceliPALM_Macro, Event
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnPickCam1TxtFile, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
filter_to_read = !VERSION.OS_family eq 'unix' ? ['*.txt'] : ['*.txt']
text = Dialog_Pickfile(/read,filter=filter_to_read,title='Pick Raw Data #1 File *.txt')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
File1WidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_Cam1Filename')
widget_control,File1WidID,SET_VALUE = text
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	print,text, ' cannot be loaded, or has incorrect structure'
	PRINT, 'Error index: ', Error_status
	PRINT, 'Error message: ', !ERROR_STATE.MSG
	CATCH, /CANCEL
	return
ENDIF
if strmatch(text,'*.txt',/fold_case) then begin
	ReadThisFitCond, text, pth, filen, ini_filename, thisfitcond
	Fill_Parameters_iPALM_Macro, Event
endif

pos = strpos(text,'.',/reverse_search,/reverse_offset)
RawFilenames0Info=file_info(text)
if ~RawFilenames0Info.exists then begin
	z=dialog_message('CAM1 File does not exist')
	return
endif
RawFilenames[0] = strmid(text,0,pos)

if product(strlen(RawFilenames)) ne 0 then begin
	Recalculate_Thisfitconds, RawFilenames, ThisFitConds
	Fill_Parameters_iPALM_Macro, Event
endif

end
;
;-----------------------------------------------------------------
;
pro OnPickCam2TxtFile, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
filter_to_read = !VERSION.OS_family eq 'unix' ? ['*.txt'] : ['*.txt']
text = Dialog_Pickfile(/read,filter=filter_to_read,title='Pick Raw Data #2 File *.txt')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
File2WidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_Cam2Filename')
widget_control,File2WidID,SET_VALUE = text
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	print,text, ' cannot be loaded, or has incorrect structure'
	PRINT, 'Error index: ', Error_status
	PRINT, 'Error message: ', !ERROR_STATE.MSG
	CATCH, /CANCEL
	return
ENDIF
if strmatch(text,'*.txt',/fold_case) then begin
	ReadThisFitCond, text, pth, filen, ini_filename, thisfitcond
	Fill_Parameters_iPALM_Macro, Event
endif

pos = strpos(text,'.',/reverse_search,/reverse_offset)
RawFilenames1Info=file_info(text)
if ~RawFilenames1Info.exists then begin
	z=dialog_message('CAM2 File does not exist')
	return
endif
RawFilenames[1] = strmid(text,0,pos)

if product(strlen(RawFilenames)) ne 0 then begin
	Recalculate_Thisfitconds, RawFilenames, ThisFitConds
	Fill_Parameters_iPALM_Macro, Event
endif
end
;
;-----------------------------------------------------------------
;
pro OnPickCam3TxtFile, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
filter_to_read = !VERSION.OS_family eq 'unix' ? ['*.txt'] : ['*.txt']
text = Dialog_Pickfile(/read,filter=filter_to_read,title='Pick Raw Data #3 File *.txt')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
File3WidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_Cam3Filename')
widget_control,File3WidID,SET_VALUE = text
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	print,text, ' cannot be loaded, or has incorrect structure'
	PRINT, 'Error index: ', Error_status
	PRINT, 'Error message: ', !ERROR_STATE.MSG
	CATCH, /CANCEL
	return
ENDIF
if strmatch(text,'*.txt',/fold_case) then begin
	ReadThisFitCond, text, pth, filen, ini_filename, thisfitcond
	Fill_Parameters_iPALM_Macro, Event
endif
pos = strpos(text,'.',/reverse_search,/reverse_offset)
RawFilenames2Info=file_info(text)
if ~RawFilenames2Info.exists then begin
	z=dialog_message('CAM3 File does not exist')
	return
endif
RawFilenames[2] = strmid(text,0,pos)

if product(strlen(RawFilenames)) ne 0 then begin
	Recalculate_Thisfitconds, RawFilenames, ThisFitConds
	Fill_Parameters_iPALM_Macro, Event
endif
end
;
;-----------------------------------------------------------------
;
pro OnPickANCFile_iPALM, Event
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
AnchorFile = Dialog_Pickfile(/read,get_path=fpath,filter=['*.anc'],title='Select *.anc file to open')
if AnchorFile ne '' then begin
	cd,fpath
	AncFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_ANCFilename')
	widget_control,AncFileWidID,SET_VALUE = AnchorFile
	CATCH, Error_status
	IF Error_status NE 0 THEN BEGIN
		print,'Fiducials cannot be loaded'
		PRINT, 'Error index: ', Error_status
		PRINT, 'Error message: ', !ERROR_STATE.MSG
		CATCH, /CANCEL
		close,5
		return
	ENDIF
	AnchorPnts=dblarr(6,AnchPnts_MaxNum)
	AnchorPnts_line=dblarr(6)
	close,5
	openr,5,AnchorFile
	ip=0
	while not EOF(5) do begin
		readf,5,AnchorPnts_line
		AnchorPnts[*,ip] = AnchorPnts_line
		ip+=1
	endwhile
	close,5
endif
end
;
;-----------------------------------------------------------------
;
pro OnPickWINDFile_iPALM, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
wfilename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.sav'],title='Select *WND.sav file to open')
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	print,'Z-calibrations cannot be loaded'
	PRINT, 'Error index: ', Error_status
	PRINT, 'Error message: ', !ERROR_STATE.MSG
	CATCH, /CANCEL
	return
ENDIF
if wfilename ne '' then begin
	cd,fpath
	WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_WindFilename')
	widget_control,WFileWidID,SET_VALUE = wfilename
	wfile_info=FILE_INFO(wfilename)
		if wfile_info.exists then begin
			wind_range = 0
			z_unwrap_coeff = transpose([0.0,0.0,0.0])
			restore,filename=wfilename
		endif
endif
end
;
;-----------------------------------------------------------------
;
pro DoInsertInfo_iPALM_Macro, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
widget_control,event.id,get_value=thevalue
value=float(reform(thevalue))
CASE event.y OF
	0:thisfitcond.zerodark=value[event.y]
	1:thisfitcond.xsz=value[event.y]
	2:thisfitcond.ysz=value[event.y]
	3:thisfitcond.Nframesmax=value[event.y]
	4:thisfitcond.Frm0=value[event.y]
	5:thisfitcond.FrmN=value[event.y]
	6:thisfitcond.Thresholdcriteria=value[event.y]
	7:thisfitcond.filetype=value[event.y]
	8:thisfitcond.LimBotA1=value[event.y]
	9:thisfitcond.LimTopA1=value[event.y]
	10:thisfitcond.LimBotSig=value[event.y]
	11:thisfitcond.LimTopSig=value[event.y]
	12:thisfitcond.LimChiSq=value[event.y]
	13:thisfitcond.Cntpere=value[event.y]
	14:thisfitcond.maxcnt1=value[event.y]
	15:thisfitcond.maxcnt2=value[event.y]
	16:thisfitcond.fliphor=value[event.y]
	17:thisfitcond.flipvert=value[event.y]
	18:thisfitcond.MaskSize=value[event.y]
	19:thisfitcond.GaussSig=value[event.y]
	20:thisfitcond.MaxBlck=value[event.y]
	21:thisfitcond.SparseOversampling=value[event.y]
	22:thisfitcond.SparseLambda=value[event.y]
	23:thisfitcond.SparseDelta=value[event.y]
	24:thisfitcond.SpError=value[event.y]
	25:thisfitcond.SpMaxIter=value[event.y]
ENDCASE
widget_control,event.id,set_value=transpose(value),use_table_select=[0,0,0,25]
end
;
;-----------------------------------------------------------------
;
pro Fill_Parameters_iPALM_Macro, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters

TABLE_InfoFile_iPALM_macro_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_InfoFile_iPALM_macro')
values=[thisfitcond.zerodark,  thisfitcond.xsz,  thisfitcond.ysz,  thisfitcond.Nframesmax,$
		thisfitcond.Frm0,  thisfitcond.FrmN,  thisfitcond.Thresholdcriteria,$
		thisfitcond.filetype,  thisfitcond.LimBotA1,  thisfitcond.LimTopA1,$
		thisfitcond.LimBotSig,  thisfitcond.LimTopSig,  thisfitcond.LimChiSq,$
		thisfitcond.Cntpere, thisfitcond.maxcnt1, thisfitcond.maxcnt2,$
		thisfitcond.fliphor,thisfitcond.flipvert, thisfitcond.MaskSize,$
		thisfitcond.GaussSig,thisfitcond.MaxBlck,thisfitcond.SparseOversampling,$
		thisfitcond.SparseLambda,thisfitcond.SparseDelta,thisfitcond.SpError,thisfitcond.SpMaxIter]
widget_control,TABLE_InfoFile_iPALM_macro_ID,set_value=transpose(values), use_table_select=[0,0,0,25]

WID_iPALM_MacroParameters_ID = Widget_Info(Event.top, find_by_uname='WID_Filter_Parameters_iPALM_Macro')
iPALM_MacroParameters = thisfitcond.SigmaSym ?	iPALM_MacroParameters_XY	:	iPALM_MacroParameters_R
widget_control,WID_iPALM_MacroParameters_ID,set_value=transpose(iPALM_MacroParameters), use_table_select=[0,0,0,(n_elements(iPALM_MacroParameters)-1)]

WidDListSigmaSym = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_SetSigmaFitSym_iPALM')
widget_control,WidDListSigmaSym,SET_DROPLIST_SELECT=thisfitcond.SigmaSym

WidDListDispLevel = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_TransformEngine_iPALM')
widget_control,WidDListDispLevel,SET_DROPLIST_SELECT=TransformEngine
end
;
;-----------------------------------------------------------------
;
pro Set_SigmaFitSym_iPALM, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
	SigmaSym=widget_info(event.id,/DropList_Select)		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
	thisfitcond.SigmaSym = SigmaSym
	iPALM_MacroParameters = thisfitcond.SigmaSym ?	iPALM_MacroParameters_XY	:	iPALM_MacroParameters_R
	;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
	WID_iPALM_MacroParameters_ID = Widget_Info(Event.top, find_by_uname='WID_Filter_Parameters_iPALM_Macro')
	widget_control,WID_iPALM_MacroParameters_ID,set_value=transpose(iPALM_MacroParameters), use_table_select=[0,0,0,(n_elements(iPALM_MacroParameters)-1)]
end
;
;-----------------------------------------------------------------
;
pro Set_TransformEngine_iPALM, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
	TransformEngine = widget_info(event.id,/DropList_Select)
	print,'Set TransformEngine to ',TransformEngine
end
;
;-----------------------------------------------------------------
;
pro Start_iPALM_Macro, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common transformfilenames, lab_filenames, sum_filename
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
print,"Use Confirm and Start Fast instead"
return

TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
thisfitcond_orig = thisfitcond
TransformEngine_orig = TransformEngine
Initialization_PeakSelector, TopID

TransformEngine = TransformEngine_orig

WID_iPALM_MacroParameters_ID = Widget_Info(Event.Top, find_by_uname='WID_Filter_Parameters_iPALM_Macro')
widget_control,WID_iPALM_MacroParameters_ID,get_value=iPALM_MacroParameters

thisfitcond=thisfitcond_orig
print,'Set Thisfitcond to',thisfitcond

print,'Start_iPALM_Macro: iPALM_MacroParameters=',transpose(iPALM_MacroParameters)

WID_ID_Use_SkipTransformation = Widget_Info(Event.Top, find_by_uname='WID_Use_SkipTransformation')
SkipTransformation=widget_info(WID_ID_Use_SkipTransformation,/button_set)

def_wind=!D.window
nlbls=3
lab_filenames=strarr(nlbls)
Orig_RawFilenames = strarr(nlbls)
RawFilenames = strarr(nlbls)
disp_increment=500					; frame interval for progress display
Start_Time= SYSTIME(/SECONDS)

; Load the Original Filename1, and check if it exists
File1WidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_Cam1Filename')
widget_control,File1WidID,GET_VALUE = text
pos = strpos(text,'.',/reverse_search,/reverse_offset)
Orig_RawFilenames[0] = text
Orig_RawFilenames0Info=file_info(text)
if ~Orig_RawFilenames0Info.exists then begin
	z=dialog_message('CAM1 File does not exist')
	return
endif
Orig_RawFilenames[0] = strmid(text,0,pos)

; Load the Original Filename2, and check if it exists
File2WidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_Cam2Filename')
widget_control,File2WidID,GET_VALUE = text
pos = strpos(text,'.',/reverse_search,/reverse_offset)
Orig_RawFilenames[1] = text
Orig_RawFilenames1Info=file_info(text)
if ~Orig_RawFilenames1Info.exists then begin
	z=dialog_message('CAM2 File does not exist')
	return
endif
Orig_RawFilenames[1] = strmid(text,0,pos)

; Load the Original Filename3, and check if it exists
File3WidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_Cam3Filename')
widget_control,File3WidID,GET_VALUE = text
pos = strpos(text,'.',/reverse_search,/reverse_offset)
Orig_RawFilenames[2] = text
Orig_RawFilenames2Info=file_info(text)
if ~Orig_RawFilenames2Info.exists then begin
	z=dialog_message('CAM3 File does not exist')
	return
endif

Orig_RawFilenames[2] = strmid(text,0,pos)

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
Raw_dir = strmid(Orig_RawFilenames[2],0,strpos(Orig_RawFilenames[2],sep,/REVERSE_OFFSET,/REVERSE_SEARCH))
cd,Raw_dir				; work out the directory where the CAM3 data File resides

; Check if ANC and WND files exist and load the data
AncFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_ANCFilename')
widget_control,AncFileWidID,GET_VALUE = AnchorFile
AncFileInfo=file_info(AnchorFile)
AnchorPnts_line=dblarr(6)
if ~AncFileInfo.exists then begin
	z=dialog_message('*.ANC File does not exist')
	return
endif
;AncSAV_File=AddExtension(AnchorFile,'_anc.sav')
;AncSAV_FileInfo=file_info(AncSAV_File)
;if AncFileInfo.exists then restore,filename=AncSAV_File else Transf_Meth=0
close,5
openr,5,AnchorFile
ip=0
while not EOF(5) do begin
	readf,5,AnchorPnts_line
	AnchorPnts[*,ip] = AnchorPnts_line
	ip+=1
endwhile
close,5
WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_WindFilename')
widget_control,WFileWidID,GET_VALUE = wfilename
WfilenameInfo=file_info(wfilename)
if ~WfilenameInfo.exists then begin
	z=dialog_message('*.WND File does not exist')
	return
endif
wfile_info=FILE_INFO(wfilename)
if wfile_info.exists then begin
	wind_range = 0
	z_unwrap_coeff = transpose([0.0,0.0,0.0])
	restore,filename=wfilename
endif

; Load other settings
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

WidFramesPerNode = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_FramesPerNode_iPALM')
widget_control,WidFramesPerNode,get_value=increment
wGrpGapID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Group_Gap_iPALM')
widget_control,wGrpGapID,get_value=grouping_gap
wGrpRadID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Grouping_Radius_iPALM')
widget_control,wGrpRadID,get_value=grouping_radius100
spacer=grouping_gap+2
grouping_radius=FLOAT(grouping_radius100)/100	; in CCD pixel units

WID_Use_InfoFile_Flip = Widget_Info(Event.Top, find_by_uname='WID_Use_InfoFile_Flip')
Flip_Using_InfoFile=widget_info(WID_Use_InfoFile_Flip,/button_set)


; Create "Transformed" and "Sum" filenames, load original Thisfitconds
for lbl_ind =0 , (nlbls-1) do begin
	ReadThisFitCond, (Orig_RawFilenames[lbl_ind]+'.txt'), pth, filen, ini_filename, thisfitcond
	pos=max(strsplit(Orig_RawFilenames[lbl_ind],sep))
	fpath=strmid(Orig_RawFilenames[lbl_ind],0,pos-1)
	pos1=strpos(Orig_RawFilenames[lbl_ind],'cam'+strtrim(strmid((lbl_ind+1),0),1))
	pos2=strpos(Orig_RawFilenames[lbl_ind],'c'+strtrim(strmid((lbl_ind+1),0),1))
	if pos1 gt 1 then begin
		lab_filenames[lbl_ind]=strmid(Orig_RawFilenames[lbl_ind],0,pos1+4)+'trn2'+strmid(Orig_RawFilenames[lbl_ind],pos1+4,(strlen(Orig_RawFilenames[lbl_ind])-pos1-4))+'.dat'
	endif else begin
	if pos2 gt 1 then begin
		lab_filenames[lbl_ind]=strmid(Orig_RawFilenames[lbl_ind],0,pos2+2)+'trn2'+strmid(Orig_RawFilenames[lbl_ind],pos2+2,(strlen(Orig_RawFilenames[lbl_ind])-pos2-2))+'.dat'
	endif else		lab_filenames[lbl_ind]=strmid(Orig_RawFilenames[lbl_ind],pos)+'_transformed.dat'
	endelse
	if lbl_ind eq 0 then begin
		ThisFitConds=replicate(thisfitcond,nlbls)
		FlipRot=replicate({frt,present:0B,transp:0B,flip_h:0B,flip_v:0B},nlbls)
		NFrames=thisfitcond.Nframesmax    ;
		GStarDrifts=replicate({present:0B,xdrift:dblarr(Nframes),ydrift:dblarr(Nframes),zdrift:dblarr(Nframes)},nlbls)
		FidCoeffs=replicate({fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)},nlbls)
		first_file=strmid(Orig_RawFilenames[0],pos)
		cam_pos=strpos(first_file,'cam')
		if cam_pos gt 1 then begin
			sum_filename=strmid(first_file,0,cam_pos+3)+'123_sum'+strmid(first_file,cam_pos+4,strlen(first_file)-cam_pos-4)+'.dat'
		endif else begin
			cam_pos1=strpos(first_file,'c1')
			if cam_pos1 gt 1 then begin
				sum_filename=strmid(first_file,0,cam_pos1+1)+'123_sum'+strmid(first_file,cam_pos1+2,strlen(first_file)-cam_pos1-2)+'.dat'
			endif else sum_filename=first_file
		endelse
		sum_filename=fpath + sep + sum_filename
	endif else ThisFitConds[lbl_ind]=thisfitcond
		if Flip_Using_InfoFile and thisfitcond.fliphor then begin
		FlipRot[lbl_ind].present=1
		FlipRot[lbl_ind].flip_h=1
	endif
	if Flip_Using_InfoFile and thisfitcond.flipvert then begin
		FlipRot[lbl_ind].present=1
		FlipRot[lbl_ind].flip_v=1
	endif
endfor


AncSav=addextension(AnchorFile,'.sav')
AncSav_FileInfo=file_info(AncSav)
if AncSav_FileInfo.exists then restore, filename=AncSav
;Transf_Meth	;0 - linear regression 1 - polywarp; 2 - pivot and average transform (in case of three fiducials only)
if n_elements(transf_meth) eq 0 then transf_meth=0

; Calculate the Fiducial Transformation Coefficients: Red to Green
LabelToTransform=1
LabelTarget=2
DatFid=AnchorPnts[(2*LabelToTransform-2):(2*LabelToTransform-1),*]
TargFid=AnchorPnts[(2*LabelTarget-2):(2*LabelTarget-1),*]
anc_ind=where(DatFid[0,*] ne 0)
Xo=reform(DatFid[0,anc_ind])
Yo=reform(DatFid[1,anc_ind])
XYo=complex(Xo,Yo)
Xi=reform(TargFid[0,anc_ind])
Yi=reform(TargFid[1,anc_ind])
XYi=complex(Xi,Yi)
Zo=XYo
Zi=XYi

print,'Using Transformation Method:  ',Transf_Meth
print,'0 - linear regression 1 - polywarp; 2 - pivot and average transform (in case of three fiducials only)'

if Transf_Meth eq 1 then begin
	print,'Using Polywarp Degree:  ',PW_deg
	polywarp,Xo,Yo,Xi,Yi,PW_deg,P,Q				;Xi=sum(kxij#Xo^jYo^i)   Yi=sum(kyij#Xo^jYo^i)
	; transformation for actual data is inverse of that for the fiducials, see the difference betweem POLY_2D and POLYWRAP
endif else begin
	Complex_Linear_Regression, Zo, Zi, P,Q
endelse
FidCoeffs[0].present=1
FidCoeffs[0].P=P
FidCoeffs[0].Q=Q
print,'Red-to-Green P=',P
print,'Red-to-Green Q=',Q

; Calculate the Fiducial Transformation Coefficients: Blue to Green
LabelToTransform=3
LabelTarget=2
DatFid=AnchorPnts[(2*(LabelToTransform-1)):(2*LabelToTransform-1),*]
TargFid=AnchorPnts[(2*(LabelTarget-1)):(2*LabelTarget-1),*]
anc_ind=where(DatFid[0,*] ne 0)
Xo=reform(DatFid[0,anc_ind])
Yo=reform(DatFid[1,anc_ind])
XYo=complex(Xo,Yo)
Xi=reform(TargFid[0,anc_ind])
Yi=reform(TargFid[1,anc_ind])
XYi=complex(Xi,Yi)
Zo=XYo
Zi=XYi
if Transf_Meth eq 1 then begin
	polywarp,Xo,Yo,Xi,Yi,PW_deg,P,Q				;Xi=sum(kxij#Xo^jYo^i)   Yi=sum(kyij#Xo^jYo^i)
	; transformation for actual data is inverse of that for the fiducials, see the difference betweem POLY_2D and POLYWRAP
endif else begin
	Complex_Linear_Regression, Zo, Zi, P,Q
endelse
FidCoeffs[2].present=1
FidCoeffs[2].P=P
FidCoeffs[2].Q=Q
print,'Blue-to-Green P=',P
print,'Blue-to-Green Q=',Q

if ~SkipTransformation then begin
	;Perform transformations, create "Transformed" and "Sum" files
	print,'iPALM Macro: Start Data Transformation and Saving'
	TransformRaw_Save_SaveSum, sum_filename, lab_filenames, Orig_RawFilenames, GStarDrifts, FidCoeffs, FlipRot
	print,'iPALM Macro: Finished transformation and saving'
endif else begin
	print,'iPALM Macro: Load Thisfitcond for Existing SUM file'
	sum_filename_txt=addextension(sum_filename,'.txt')
	ReadThisFitCond, sum_filename_txt, pth, filen, ini_filename, thisfitcond
endelse
wset,def_wind

DispType = 1
if TransformEngine gt 0 then DispType = 3
;DispType = TransformEngine ? 3 : 1	;set to 0 - min display while fitting 1 - some display 2 - full display  3 - Cluster
DispType0=DispType

fpath=pth
filen=strmid(sum_filename,strlen(fpath))			;the filename
filen=strmid(filen,0,strlen(filen)-4)			;the filename wo extension
print,'iPALM Macro: Started Sum file Extraction'
ReadRawLoop6,DispType		;goes through data and fits peaks
print,'iPALM Macro: Finished Sum file Extraction'

MLRawFilenames=lab_filenames
nlbls=n_elements(lab_filenames)
for i=0,nlbls-1 do begin
	interm_file=strmid(lab_filenames[i],strlen(fpath))
	MLRawFilenames[i]=pth+strmid(interm_file,0,strlen(interm_file)-4)
endfor

OKindex=where(CGroupParams[8,*] eq 1 or CGroupParams[8,*] eq 2,OKcnt)
CGroupParams=temporary(CGroupParams[*,OKindex])
sz=size(CGroupParams)
CGroupParams[11,*]=lindgen(sz[2])

	;iPALM_MacroParameters = [100.0,	0.4,	0.4,	40.0,	0.1,	1.3,	1.4,	0.01,	0.6,	0.01]
	; iPALM_MacroParameters = [Nph,	Full Sigma X Max. (pix.),	Full Sigma Y Max. (pix.),	Sigma Z Max. (nm), Coherence Min.,	Coherence Max,	...
	; iPALM_MacroParameters = ...	Max A for (Wx-A)*(Wx-A)<B,	B for (Wx-A)*(Wx-A)<B,	Min A  for (Wx-A)*(Wx-A)>B,	B for (Wx-A)*(Wx-A)>B ]

Nph_ind = min(where(RowNames eq '6 N Photons'))
Wx_ind = min(where(RowNames eq 'X Peak Width'))
Wy_ind = min(where(RowNames eq 'Y Peak Width'))
FullSigmaX_ind = min(where(RowNames eq 'Sigma X Pos Full'))
FullSigmaY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))
SigmaZ_ind = min(where(RowNames eq 'Sigma Z'))
Coherence_ind = min(where(RowNames eq '36 Coherence'))

filter1 = (CGroupParams[Nph_ind,*] ge iPALM_MacroParameters[0])	AND (CGroupParams[FullSigmaX_ind,*] le iPALM_MacroParameters[1]) $
	AND (CGroupParams[FullSigmaY_ind,*] le iPALM_MacroParameters[2])	$
	AND (((CGroupParams[Wx_ind,*]-iPALM_MacroParameters[6]) > 0.0)*((CGroupParams[Wy_ind,*]-iPALM_MacroParameters[6])> 0.0) le iPALM_MacroParameters[7])	$
	AND (((CGroupParams[Wx_ind,*]-iPALM_MacroParameters[8]) > 0.0)*((CGroupParams[Wy_ind,*]-iPALM_MacroParameters[8])> 0.0) ge iPALM_MacroParameters[9])
pk_indecis=where(filter1,cnt)

if cnt lt 1 then begin
	z=dialog_message('Filter returned no valid peaks')
	return      ; if data not loaded return
endif
CGroupParams=temporary(CGroupParams[*,pk_indecis])

if (size(CGroupParams))[0] ne 0 then begin
	ReloadParamlists, Event1
	OnUnZoomButton, Event1
endif
print,'iPALM Macro: purged not OK peaks and applied filters (all but z, coherence)'
print,'Total number of peaks=',cnt

SigmaSym = thisfitcond.SigmaSym
DispType =DispType0
print,'iPALM Macro: Started ReExtractig using the files:  ',MLRawFilenames
print,'iPALM Macro: using Disptype and SygmaSym: ',DispType,SigmaSym
ReadRawLoopMultipleLabel,DispType		;goes through data and fits peaks
print,'iPALM Macro: Finished ReExtractig multilabel'

x_center=((size(TotalRawData))[0] ne 0) ? (size(TotalRawData))[1]/2.0 : 0
y_center=((size(TotalRawData))[0] ne 0) ? (size(TotalRawData))[2]/2.0 : 0
ellipticity_slopes = [x_center,0.0,y_center,0.0]

print,'iPALM Macro: Start First Extracting Z-coordinate'
OnExtractZCoord, Event1
print,'iPALM Macro: Finished First Extracting Z-coordinate'
; filter and purge bad peaks: Sigma Z
filter = filter AND (CGroupParams[SigmaZ_ind,*] le iPALM_MacroParameters[3]) AND (CGroupParams[Coherence_ind,*] ge iPALM_MacroParameters[4]) AND (CGroupParams[Coherence_ind,*] le iPALM_MacroParameters[5])
pk_indecis=where(filter,cnt)
if cnt lt 1 then begin
	z=dialog_message('Filter returned no valid peaks')
	return      ; if data not loaded return
endif
CGroupParams=temporary(CGroupParams[*,pk_indecis])

if (size(CGroupParams))[0] ne 0 then begin
	ReloadParamlists, Event1
	OnUnZoomButton, Event1
endif
print,'iPALM Macro: purged based on Sigma Z and  coherence filters'
print,'Total number of peaks=',cnt

GroupEngine = TransformEngine
maxgrsize=10						; not absolute max group size: max group size for arrayed processing (groups with elements>maxgroupsize are later analyzed separately)
print,'iPALM Macro: Started Grouping'
if GroupEngine eq 0 then begin
	GroupDisplay=1						; 0 for cluster, 1 for local
	GoodPeaks=where(filter ne 0,OKpkcnt)
	CGroupParamsGP=CGroupParams[*,GoodPeaks]
	GroupPeaksCore,CGroupParamsGP,CGrpSize,OKpkcnt,grouping_radius,spacer,maxgrsize,disp_increment,GroupDisplay
	CGroupParams[*,GoodPeaks]=CGroupParamsGP
endif else begin
	framefirst=long(ParamLimits[9,0])
	framelast=long(ParamLimits[9,1])
	nloops=long(ceil((framelast-framefirst+1.0)/increment))
	GroupDisplay=0						; 0 for cluster, 1 for local
	if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR') else idl_pwd=pref_get('IDL_WDE_START_DIR')
	cd,current=curr_pwd
	FILE_MKDIR,curr_pwd+'/temp'
	save, curr_pwd,idl_pwd, CGroupParams, CGrpSize, ParamLimits, increment, nloops, spacer, grouping_radius,maxgrsize,disp_increment,GroupDisplay, filename='temp/temp.sav'		;save variables for cluster cpu access
	GroupPeaksCluster_ReadBack, interrupt_load
	file_delete,'temp/temp.sav'
	file_delete,'temp'
	cd,curr_pwd
endelse
ReloadParamlists, Event1
OnGroupCentersButton, Event1
print,'iPALM Macro: Finished Grouping'

print,'iPALM Macro: Start Second Extracting Z-coordinate'
OnExtractZCoord, Event1
print,'iPALM Macro: Finished Second Extracting Z-coordinate'

results_filename = AddExtension(RawFilenames[0],'_IDL.sav')
print,'iPALM Macro: Saving the data into file:',results_filename
SavFilenames=results_filename
save, CGroupParams, CGrpSize, ParamLimits, Image, b_set, xydsz, TotalRawData, RawFilenames, SavFilenames, MLRawFilenames,$
		GuideStarDrift, FiducialCoeff, FlipRotate, thisfitcond, RowNames, $
		lambda_vac,nd_water, nd_oil, wind_range, z_unwrap_coeff, ellipticity_slopes, $
		nm_per_pixel, wfilename, PkWidth_offset, Z_scale_multiplier, grouping_gap, grouping_radius100, filename=results_filename

wlabel = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,wlabel,set_value=results_filename
print,'iPALM Macro: Finished'
widget_control,event.top,/destroy
AnchorPnts=dblarr(6,AnchPnts_MaxNum)
ZPnts=dblarr(3,AnchPnts_MaxNum)
print,'iPALM Macro: total time (sec)', (SYSTIME(/SECONDS)-Start_Time)
end
;
;-----------------------------------------------------------------
;
pro Start_Transformation_Macro, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common transformfilenames, lab_filenames, sum_filename
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
thisfitcond_orig = thisfitcond
TransformEngine_orig = TransformEngine
Initialization_PeakSelector, TopID

TransformEngine = 0

WID_iPALM_MacroParameters_ID = Widget_Info(Event.Top, find_by_uname='WID_Filter_Parameters_iPALM_Macro')
widget_control,WID_iPALM_MacroParameters_ID,get_value=iPALM_MacroParameters

thisfitcond=thisfitcond_orig
print,'Set Thisfitcond to',thisfitcond

print,'Start_iPALM_Macro: iPALM_MacroParameters=',transpose(iPALM_MacroParameters)

WID_ID_Use_SkipTransformation = Widget_Info(Event.Top, find_by_uname='WID_Use_SkipTransformation')
SkipTransformation=widget_info(WID_ID_Use_SkipTransformation,/button_set)

def_wind=!D.window
nlbls=3
lab_filenames=strarr(nlbls)
Orig_RawFilenames = strarr(nlbls)
RawFilenames = strarr(nlbls)
disp_increment=500					; frame interval for progress display
Start_Time= SYSTIME(/SECONDS)

; Load the Original Filename1, and check if it exists
File1WidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_Cam1Filename')
widget_control,File1WidID,GET_VALUE = text
pos = strpos(text,'.',/reverse_search,/reverse_offset)
Orig_RawFilenames[0] = text
Orig_RawFilenames0Info=file_info(text)
if ~Orig_RawFilenames0Info.exists then begin
	z=dialog_message('CAM1 File does not exist')
	return
endif
Orig_RawFilenames[0] = strmid(text,0,pos)

; Load the Original Filename2, and check if it exists
File2WidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_Cam2Filename')
widget_control,File2WidID,GET_VALUE = text
pos = strpos(text,'.',/reverse_search,/reverse_offset)
Orig_RawFilenames[1] = text
Orig_RawFilenames1Info=file_info(text)
if ~Orig_RawFilenames1Info.exists then begin
	z=dialog_message('CAM2 File does not exist')
	return
endif
Orig_RawFilenames[1] = strmid(text,0,pos)

; Load the Original Filename3, and check if it exists
File3WidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_Cam3Filename')
widget_control,File3WidID,GET_VALUE = text
pos = strpos(text,'.',/reverse_search,/reverse_offset)
Orig_RawFilenames[2] = text
Orig_RawFilenames2Info=file_info(text)
if ~Orig_RawFilenames2Info.exists then begin
	z=dialog_message('CAM3 File does not exist')
	return
endif

Orig_RawFilenames[2] = strmid(text,0,pos)

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
Raw_dir = strmid(Orig_RawFilenames[2],0,strpos(Orig_RawFilenames[2],sep,/REVERSE_OFFSET,/REVERSE_SEARCH))
cd,Raw_dir				; work out the directory where the CAM3 data File resides

; Check if ANC and WND files exist and load the data
AncFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_ANCFilename')
widget_control,AncFileWidID,GET_VALUE = AnchorFile
AncFileInfo=file_info(AnchorFile)
AnchorPnts_line=dblarr(6)
if ~AncFileInfo.exists then begin
	z=dialog_message('*.ANC File does not exist')
	return
endif
;AncSAV_File=AddExtension(AnchorFile,'_anc.sav')
;AncSAV_FileInfo=file_info(AncSAV_File)
;if AncFileInfo.exists then restore,filename=AncSAV_File else Transf_Meth=0
close,5
openr,5,AnchorFile
ip=0
while not EOF(5) do begin
	readf,5,AnchorPnts_line
	AnchorPnts[*,ip] = AnchorPnts_line
	ip+=1
endwhile
close,5

; Load other settings
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

WID_Use_InfoFile_Flip = Widget_Info(Event.Top, find_by_uname='WID_Use_InfoFile_Flip')
Flip_Using_InfoFile=widget_info(WID_Use_InfoFile_Flip,/button_set)

; Create "Transformed" and "Sum" filenames, load original Thisfitconds
for lbl_ind =0 , (nlbls-1) do begin
	ReadThisFitCond, (Orig_RawFilenames[lbl_ind]+'.txt'), pth, filen, ini_filename, thisfitcond
	pos=max(strsplit(Orig_RawFilenames[lbl_ind],sep))
	fpath=strmid(Orig_RawFilenames[lbl_ind],0,pos-1)
	pos1=strpos(Orig_RawFilenames[lbl_ind],'cam'+strtrim(strmid((lbl_ind+1),0),1))
	pos2=strpos(Orig_RawFilenames[lbl_ind],'c'+strtrim(strmid((lbl_ind+1),0),1))
	if pos1 gt 1 then begin
		lab_filenames[lbl_ind]=strmid(Orig_RawFilenames[lbl_ind],0,pos1+4)+'trn2'+strmid(Orig_RawFilenames[lbl_ind],pos1+4,(strlen(Orig_RawFilenames[lbl_ind])-pos1-4))+'.dat'
	endif else begin
	if pos2 gt 1 then begin
		lab_filenames[lbl_ind]=strmid(Orig_RawFilenames[lbl_ind],0,pos2+2)+'trn2'+strmid(Orig_RawFilenames[lbl_ind],pos2+2,(strlen(Orig_RawFilenames[lbl_ind])-pos2-2))+'.dat'
	endif else		lab_filenames[lbl_ind]=strmid(Orig_RawFilenames[lbl_ind],pos)+'_transformed.dat'
	endelse
	if lbl_ind eq 0 then begin
		ThisFitConds=replicate(thisfitcond,nlbls)
		FlipRot=replicate({frt,present:0B,transp:0B,flip_h:0B,flip_v:0B},nlbls)
		NFrames=thisfitcond.Nframesmax    ;
		GStarDrifts=replicate({present:0B,xdrift:dblarr(Nframes),ydrift:dblarr(Nframes),zdrift:dblarr(Nframes)},nlbls)
		FidCoeffs=replicate({fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)},nlbls)
		first_file=strmid(Orig_RawFilenames[0],pos)
		cam_pos=strpos(first_file,'cam')
		if cam_pos gt 1 then begin
			sum_filename=strmid(first_file,0,cam_pos+3)+'123_sum'+strmid(first_file,cam_pos+4,strlen(first_file)-cam_pos-4)+'.dat'
		endif else begin
			cam_pos1=strpos(first_file,'c1')
			if cam_pos1 gt 1 then begin
				sum_filename=strmid(first_file,0,cam_pos1+1)+'123_sum'+strmid(first_file,cam_pos1+2,strlen(first_file)-cam_pos1-2)+'.dat'
			endif else sum_filename=first_file
		endelse
		sum_filename=fpath + sep + sum_filename
	endif else ThisFitConds[lbl_ind]=thisfitcond
		if Flip_Using_InfoFile and thisfitcond.fliphor then begin
		FlipRot[lbl_ind].present=1
		FlipRot[lbl_ind].flip_h=1
	endif
	if Flip_Using_InfoFile and thisfitcond.flipvert then begin
		FlipRot[lbl_ind].present=1
		FlipRot[lbl_ind].flip_v=1
	endif
endfor


AncSav=addextension(AnchorFile,'.sav')
AncSav_FileInfo=file_info(AncSav)
if AncSav_FileInfo.exists then restore, filename=AncSav
;Transf_Meth	;0 - linear regression 1 - polywarp; 2 - pivot and average transform (in case of three fiducials only)
if n_elements(transf_meth) eq 0 then transf_meth=0

; Calculate the Fiducial Transformation Coefficients: Red to Green
LabelToTransform=1
LabelTarget=2
DatFid=AnchorPnts[(2*LabelToTransform-2):(2*LabelToTransform-1),*]
TargFid=AnchorPnts[(2*LabelTarget-2):(2*LabelTarget-1),*]
anc_ind=where(DatFid[0,*] ne 0)
Xo=reform(DatFid[0,anc_ind])
Yo=reform(DatFid[1,anc_ind])
XYo=complex(Xo,Yo)
Xi=reform(TargFid[0,anc_ind])
Yi=reform(TargFid[1,anc_ind])
XYi=complex(Xi,Yi)
Zo=XYo
Zi=XYi

print,'Using Transformation Method:  ',Transf_Meth
print,'0 - linear regression 1 - polywarp; 2 - pivot and average transform (in case of three fiducials only)'

if Transf_Meth eq 1 then begin
	print,'Using Polywarp Degree:  ',PW_deg
	polywarp,Xo,Yo,Xi,Yi,PW_deg,P,Q				;Xi=sum(kxij#Xo^jYo^i)   Yi=sum(kyij#Xo^jYo^i)
	; transformation for actual data is inverse of that for the fiducials, see the difference betweem POLY_2D and POLYWRAP
endif else begin
	Complex_Linear_Regression, Zo, Zi, P,Q
endelse
FidCoeffs[0].present=1
FidCoeffs[0].P=P
FidCoeffs[0].Q=Q
print,'Red-to-Green P=',P
print,'Red-to-Green Q=',Q

; Calculate the Fiducial Transformation Coefficients: Blue to Green
LabelToTransform=3
LabelTarget=2
DatFid=AnchorPnts[(2*(LabelToTransform-1)):(2*LabelToTransform-1),*]
TargFid=AnchorPnts[(2*(LabelTarget-1)):(2*LabelTarget-1),*]
anc_ind=where(DatFid[0,*] ne 0)
Xo=reform(DatFid[0,anc_ind])
Yo=reform(DatFid[1,anc_ind])
XYo=complex(Xo,Yo)
Xi=reform(TargFid[0,anc_ind])
Yi=reform(TargFid[1,anc_ind])
XYi=complex(Xi,Yi)
Zo=XYo
Zi=XYi
if Transf_Meth eq 1 then begin
	polywarp,Xo,Yo,Xi,Yi,PW_deg,P,Q				;Xi=sum(kxij#Xo^jYo^i)   Yi=sum(kyij#Xo^jYo^i)
	; transformation for actual data is inverse of that for the fiducials, see the difference betweem POLY_2D and POLYWRAP
endif else begin
	Complex_Linear_Regression, Zo, Zi, P,Q
endelse
FidCoeffs[2].present=1
FidCoeffs[2].P=P
FidCoeffs[2].Q=Q
print,'Blue-to-Green P=',P
print,'Blue-to-Green Q=',Q

if ~SkipTransformation then begin
	;Perform transformations, create "Transformed" and "Sum" files
	print,'iPALM Macro: Start Data Transformation and Saving'
	TransformRaw_Save_SaveSum, sum_filename, lab_filenames, Orig_RawFilenames, GStarDrifts, FidCoeffs, FlipRot
	print,'iPALM Macro: Finished transformation and saving'
endif else begin
	print,'iPALM Macro: Load Thisfitcond for Existing SUM file'
	sum_filename_txt=addextension(sum_filename,'.txt')
	ReadThisFitCond, sum_filename_txt, pth, filen, ini_filename, thisfitcond
endelse
widget_control,event.top,/destroy
TransformEngine = TransformEngine_orig
print,'iPALM Transformation Macro: total time (sec)', (SYSTIME(/SECONDS)-Start_Time)
end
;
;-----------------------------------------------------------------
;
pro Start_iPALM_Macro_Fast, Event			; !!! Only works with cluster or IDL bridge  (TransformEngine = 1 or 2)  !!!!
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common transformfilenames, lab_filenames, sum_filename
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #

IF LMGR(/VM) or LMGR(/DEMO) or LMGR(/TRIAL) then begin	; Cannot run this Macro if  IDL is in Virtual Machine Mode
          z=dialog_message('Cannot run this Macro with IDL in Virtual Machine / DEMO / TRIAL Mode')
          return      ; if data not loaded return
endif

TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
thisfitcond_orig=thisfitcond
TransformEngine_orig=TransformEngine

Initialization_PeakSelector, TopID

def_wind=!D.window
TransformEngine = TransformEngine_orig
If TransformEngine eq 0 then TransformEngine=2
WidDListDispLevel = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_TransformEngine_iPALM')
widget_control,WidDListDispLevel,SET_DROPLIST_SELECT = TransformEngine

nlbls=3

WidDListDispLevel = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_TransformEngine_iPALM')
widget_control,WidDListDispLevel,SET_DROPLIST_SELECT=TransformEngine
print,'Force TransformEngine to ',TransformEngine

WID_iPALM_MacroParameters_ID = Widget_Info(Event.Top, find_by_uname='WID_Filter_Parameters_iPALM_Macro')
widget_control,WID_iPALM_MacroParameters_ID,get_value=iPALM_MacroParameters

print,'Start_iPALM_Macro_New: iPALM_MacroParameters=',transpose(iPALM_MacroParameters)

WID_ID_Use_SkipTransformation = Widget_Info(Event.Top, find_by_uname='WID_Use_SkipTransformation')
SkipTransformation=widget_info(WID_ID_Use_SkipTransformation,/button_set)

def_wind=!D.window
disp_increment=500					; frame interval for progress display
Start_Time= SYSTIME(/SECONDS)

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
Raw_dir = strmid(RawFilenames[2],0,strpos(RawFilenames[2],sep,/REVERSE_OFFSET,/REVERSE_SEARCH))
cd,Raw_dir				; work out the directory where the CAM3 data File resides

; Check if ANC and WND files exist and load the data
AncFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_ANCFilename')
widget_control,AncFileWidID,GET_VALUE = AnchorFile
AncFileInfo=file_info(AnchorFile)
AnchorPnts_line=dblarr(6)
if ~AncFileInfo.exists then begin
	z=dialog_message('*.ANC File does not exist')
	return
endif

close,5
openr,5,AnchorFile
ip=0
while not EOF(5) do begin
	readf,5,AnchorPnts_line
	AnchorPnts[*,ip] = AnchorPnts_line
	ip+=1
endwhile
close,5
WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_WindFilename')
widget_control,WFileWidID,GET_VALUE = wfilename
WfilenameInfo=file_info(wfilename)
if ~WfilenameInfo.exists then begin
	z=dialog_message('*.WND File does not exist')
	return
endif else begin
	wind_range = 0
	z_unwrap_coeff = transpose([0.0,0.0,0.0])
	restore,filename=wfilename
endelse

; Load other settings
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

WidFramesPerNode = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_FramesPerNode_iPALM')
widget_control,WidFramesPerNode,get_value=grouping_increment
wGrpGapID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Group_Gap_iPALM')
widget_control,wGrpGapID,get_value=grouping_gap
wGrpRadID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Grouping_Radius_iPALM')
widget_control,wGrpRadID,get_value=grouping_radius100
spacer=grouping_gap+2
grouping_radius=FLOAT(grouping_radius100)/100	; in CCD pixel units

WID_Use_InfoFile_Flip = Widget_Info(Event.Top, find_by_uname='WID_Use_InfoFile_Flip')
Flip_Using_InfoFile=widget_info(WID_Use_InfoFile_Flip,/button_set)

; Create "Transformed" and "Sum" filenames, load original Thisfitconds
for lbl_ind =0 , (nlbls-1) do begin
	ReadThisFitCond, (RawFilenames[lbl_ind]+'.txt'), pth, filen, ini_filename, thisfitcond
	pos=max(strsplit(RawFilenames[lbl_ind],sep))
	fpath=strmid(RawFilenames[lbl_ind],0,pos-1)
	pos1=strpos(RawFilenames[lbl_ind],'cam'+strtrim(strmid((lbl_ind+1),0),1))
	;pos2=strpos(RawFilenames[lbl_ind],'c'+strtrim(strmid((lbl_ind+1),0),1))
	pos2=strpos(RawFilenames[lbl_ind],('c'+strtrim(strmid((lbl_ind+1),0),1)),/REVERSE_SEARCH) > strpos(RawFilenames[lbl_ind],('C'+strtrim(strmid((lbl_ind+1),0),1)), /REVERSE_SEARCH)


	if lbl_ind eq 0 then begin
		ThisFitConds=replicate(thisfitcond,nlbls)
		FlipRot=replicate({frt,present:0B,transp:0B,flip_h:0B,flip_v:0B},nlbls)
		NFrames=thisfitcond.Nframesmax    ;
		GStarDrifts=replicate({present:0B,xdrift:dblarr(Nframes),ydrift:dblarr(Nframes),zdrift:dblarr(Nframes)},nlbls)
		FidCoeffs=replicate({fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)},nlbls)
		first_file=strmid(RawFilenames[0],pos)
		cam_pos=strpos(first_file,'cam')
		if cam_pos gt 1 then begin
			sum_filename=strmid(first_file,0,cam_pos+3)+'123_sum'+strmid(first_file,cam_pos+4,strlen(first_file)-cam_pos-4)+'.dat'
		endif else begin
			;cam_pos1=strpos(first_file,'c1')
			cam_pos1=strpos(first_file,'c1',/REVERSE_SEARCH) > strpos(first_file,'C1',/REVERSE_SEARCH)
			if cam_pos1 gt 1 then begin
				sum_filename=strmid(first_file,0,cam_pos1+1)+'123_sum'+strmid(first_file,cam_pos1+2,strlen(first_file)-cam_pos1-2)+'.dat'
			endif else sum_filename=first_file
		endelse
		sum_filename=fpath + sep + sum_filename
	endif else ThisFitConds[lbl_ind]=thisfitcond
		if Flip_Using_InfoFile and thisfitcond.fliphor then begin
		FlipRot[lbl_ind].present=1
		FlipRot[lbl_ind].flip_h=1
	endif
	if Flip_Using_InfoFile and thisfitcond.flipvert then begin
		FlipRot[lbl_ind].present=1
		FlipRot[lbl_ind].flip_v=1
	endif
endfor

thisfitcond=thisfitcond_orig

AncSav=addextension(AnchorFile,'.sav')
AncSav_FileInfo=file_info(AncSav)
if AncSav_FileInfo.exists then restore, filename=AncSav
;Transf_Meth	;0 - linear regression 1 - polywarp; 2 - pivot and average transform (in case of three fiducials only)
if n_elements(transf_meth) eq 0 then transf_meth=0

; Calculate the Fiducial Transformation Coefficients: Red to Green
LabelToTransform=1
LabelTarget=2
DatFid=AnchorPnts[(2*LabelToTransform-2):(2*LabelToTransform-1),*]
TargFid=AnchorPnts[(2*LabelTarget-2):(2*LabelTarget-1),*]
anc_ind=where(DatFid[0,*] ne 0)
Xo=reform(DatFid[0,anc_ind])
Yo=reform(DatFid[1,anc_ind])
XYo=complex(Xo,Yo)
Xi=reform(TargFid[0,anc_ind])
Yi=reform(TargFid[1,anc_ind])
XYi=complex(Xi,Yi)
Zo=XYo
Zi=XYi

print,'iPALM Macro Fast:Using Transformation Method:  ',Transf_Meth
print,'0 - linear regression 1 - polywarp; 2 - pivot and average transform (in case of three fiducials only)'

if Transf_Meth eq 1 then begin
	print,'Using Polywarp Degree:  ',PW_deg
	polywarp,Xo,Yo,Xi,Yi,PW_deg,P,Q				;Xi=sum(kxij#Xo^jYo^i)   Yi=sum(kyij#Xo^jYo^i)
	; transformation for actual data is inverse of that for the fiducials, see the difference betweem POLY_2D and POLYWRAP
endif else begin
	Complex_Linear_Regression, Zo, Zi, P,Q
endelse
FidCoeffs[0].present=1
FidCoeffs[0].P=P
FidCoeffs[0].Q=Q
print,'Red-to-Green P=',P
print,'Red-to-Green Q=',Q

; Calculate the Fiducial Transformation Coefficients: Blue to Green
LabelToTransform=3
LabelTarget=2
DatFid=AnchorPnts[(2*(LabelToTransform-1)):(2*LabelToTransform-1),*]
TargFid=AnchorPnts[(2*(LabelTarget-1)):(2*LabelTarget-1),*]
anc_ind=where(DatFid[0,*] ne 0)
Xo=reform(DatFid[0,anc_ind])
Yo=reform(DatFid[1,anc_ind])
XYo=complex(Xo,Yo)
Xi=reform(TargFid[0,anc_ind])
Yi=reform(TargFid[1,anc_ind])
XYi=complex(Xi,Yi)
Zo=XYo
Zi=XYi
if Transf_Meth eq 1 then begin
	polywarp,Xo,Yo,Xi,Yi,PW_deg,P,Q				;Xi=sum(kxij#Xo^jYo^i)   Yi=sum(kyij#Xo^jYo^i)
	; transformation for actual data is inverse of that for the fiducials, see the difference betweem POLY_2D and POLYWRAP
endif else begin
	Complex_Linear_Regression, Zo, Zi, P,Q
endelse
FidCoeffs[2].present=1
FidCoeffs[2].P=P
FidCoeffs[2].Q=Q
print,'Blue-to-Green P=',P
print,'Blue-to-Green Q=',Q

;Perform transformations, create "Transformed" and "Sum" files
print,'RawFilenames:   ',RawFilenames
print,'iPALM Macro Fast: Start Data Transformation and Saving'
if TransformEngine eq 1 then begin
	iPALM_Macro_Fast, RawFilenames, ThisFitConds, GStarDrifts, FidCoeffs, FlipRot, iPALM_MacroParameters
endif else iPALM_Macro_Fast_Bridge, RawFilenames, ThisFitConds, GStarDrifts, FidCoeffs, FlipRot, iPALM_MacroParameters

print,'iPALM Macro Fast: Finished transformations'

	restore,'temp/temp.sav'
	tot_fr=0
	n_ele = 0
	for nlps=0,nloops-1 do begin			;reassemble little pks files from all the workers into on big one
		framefirst=	thisfitcond.Frm0 + (nlps)*increment						;first frame in batch
		framelast=((thisfitcond.Frm0 + (nlps+1)*increment-1)<thisfitcond.Nframesmax) < thisfitcond.FrmN
		print,'iPALM Macro Fast: concatenating the segment',(nlps+1),'   of ',nloops
		temp_file_info=file_info(temp_idl_fnames[Nlps])
		if temp_file_info.exists then begin
			CGroupParams=0
			restore,filename=temp_idl_fnames[Nlps]
			;print,'iPALM Macro Fast: delete',(nlps+1), '  file:',temp_idl_fnames[Nlps]
			file_delete,temp_idl_fnames[Nlps],  /ALLOW_NONEXISTENT,/QUIET
			n_ele_nlps = n_elements(CGroupParams)

			if (n_ele eq 0) and (n_ele_nlps ge CGrpSize) then begin
				n_ele = n_ele_nlps
				CGroupParams_tot=transpose(CGroupParams)
				TotalRawData =totdat
			endif else begin
				Nframes=framelast-framefirst+1L
				tot_fr=tot_fr+Nframes
				if n_ele_nlps ge CGrpSize then begin
					n_ele = n_ele + n_ele_nlps
					CGroupParams_tot=[CGroupParams_tot, transpose(CGroupParams)]
				endif
				TotalRawData = TotalRawData/tot_fr*(tot_fr-Nframes) + totdat/tot_fr*Nframes
			endelse
		endif
	endfor
	file_delete,'temp/temp.sav',  /ALLOW_NONEXISTENT,/QUIET
	file_delete,'temp',  /ALLOW_NONEXISTENT,/QUIET
	file_delete,'npks_det.sav',  /ALLOW_NONEXISTENT,/QUIET
	print,'Concatenation complete'
	;help,CGroupParams_tot
	CGroupParams=transpose(CGroupParams_tot)

wset,def_wind

sz=size(CGroupParams)
CGroupParams[11,*]=lindgen(sz[2])
print,'iPALM Macro Fast: Reloading Parameters and displaying'
if (size(CGroupParams))[0] ne 0 then begin
	ReloadParamlists, Event1
	OnUnZoomButton, Event1
endif

x_center=((size(TotalRawData))[0] ne 0) ? (size(TotalRawData))[1]/2.0 : 0
y_center=((size(TotalRawData))[0] ne 0) ? (size(TotalRawData))[2]/2.0 : 0
ellipticity_slopes = [x_center,0.0,y_center,0.0]

RawFilenames=strarr(3)
pos = strpos(sum_filename,'.',/reverse_search,/reverse_offset)
RawFilenames[0]=strmid(sum_filename,0,pos)
results_filename = AddExtension(sum_filename,'_IDL.sav')
print,'iPALM Macro Fast: Saving preliminary data into file:',results_filename

SavFilenames=results_filename
save, CGroupParams, CGrpSize, ParamLimits, Image, b_set, xydsz, TotalRawData, RawFilenames, SavFilenames, MLRawFilenames,$
		GuideStarDrift, FiducialCoeff, FlipRotate, thisfitcond, RowNames, $
		lambda_vac,nd_water, nd_oil, wind_range, z_unwrap_coeff, ellipticity_slopes, $
		nm_per_pixel, wfilename, PkWidth_offset, Z_scale_multiplier, grouping_gap, grouping_radius100, filename=results_filename

wlabel = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,wlabel,set_value=results_filename

GroupEngine = TransformEngine
maxgrsize=10						; not absolute max group size: max group size for arrayed processing (groups with elements>maxgroupsize are later analyzed separately)
increment=grouping_increment
print,'iPALM Macro Fast: Started Grouping'

framefirst=long(ParamLimits[9,0])
framelast=long(ParamLimits[9,1])
nloops=long(ceil((framelast-framefirst+1.0)/increment))
if TransformEngine eq 2 then begin
	nloops=(long(ceil((framelast-framefirst+1.0)/increment)) < 	!CPU.HW_NCPU ) < n_br_max
		; don't allow more bridge processes than there are CPU's
	increment = long(ceil((framelast-framefirst+1.0)/nloops))
	nloops = long(ceil((framelast-framefirst+1.0)/increment)) > 1L
	print,' Will start '+strtrim(nloops,2)+' bridge child processes'
endif

GroupDisplay=0						; 0 for cluster, 1 for local
if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR') else idl_pwd=pref_get('IDL_WDE_START_DIR')
td = 'temp' + strtrim(ulong(SYSTIME(/seconds)),2)
temp_dir=curr_pwd + sep + td
FILE_MKDIR,temp_dir
interrupt_load = 0
if TransformEngine eq 1 then begin
		save, curr_pwd,idl_pwd, CGrpSize, ParamLimits, increment, nloops, spacer, grouping_radius, maxgrsize, disp_increment, GroupDisplay, RowNames, filename=td + sep + 'temp.sav'		;save variables for cluster cpu access
		spawn,'sh '+idl_pwd+'/group_initialize_jobs.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd+' '+temp_dir			;Spawn grouping

		oStatusBar = obj_new('PALM_StatusBar', $
        	COLOR=[0,0,255], $
        	TEXT_COLOR=[255,255,255], $
        	CANCEL_BUTTON_PRESENT = 1, $
       	 	TITLE='Starting grouped data processing...', $
      		TOP_LEVEL_BASE=tlb)
		fraction_complete_last=0.0D
		pr_bar_inc=0.01D

		nlps = 0L

		while (nlps lt nloops) and (interrupt_load eq 0) do begin
			framestart=	framefirst + (nlps)*increment						;first frame in batch
			framestop=(framefirst + (nlps+1L)*increment-1)<framelast
			GoodPeaks=where((CGroupParams[FrNum_ind,*] ge framestart) and (CGroupParams[FrNum_ind,*] le framestop),OKpkcnt)
			GPmin = GoodPeaks[0]
			GPmax = GoodPeaks[n_elements(GoodPeaks)-1]
			fname_nlps=temp_dir+'/temp'+strtrim(Nlps,2)+'.sav'
			;CGroupParamsGP = CGroupParams[*,GoodPeaks]	; slow (?)
			CGroupParamsGP = CGroupParams[*,GPmin:GPmax]	; faster
			save, curr_pwd,idl_pwd, CGroupParamsGP, CGrpSize, ParamLimits, increment, nloops, spacer, grouping_radius,maxgrsize,disp_increment,GroupDisplay,RowNames, filename=fname_nlps		;save variables for cluster cpu access
			wait,0.1
			fraction_complete=FLOAT(nlps)/FLOAT((nloops-1.0))
			if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
				fraction_complete_last=fraction_complete
				oStatusBar -> UpdateStatus, fraction_complete
			endif
			spawn,'sh '+idl_pwd+'/group_start_single_job.sh '+strtrim(nlps,2)+' '+curr_pwd+' '+idl_pwd+' '+temp_dir			;Spawn grouping workers in cluster
			nlps++
			interrupt_load = oStatusBar -> CheckCancel()
		endwhile

		obj_destroy, oStatusBar
		;print,'interrupt_load = ', interrupt_load
		if interrupt_load eq 1 then print,'Grouping aborted, cleaning up...'
		if interrupt_load eq 0 then begin
			print,'starting GroupPeaksCluster_ReadBack'
			GroupPeaksCluster_ReadBack, interrupt_load  ;reassemble little pks files from all the workers into on big one
		endif
		CATCH, Error_status
		file_delete,td + sep + 'temp.sav'
		file_delete,td
		IF Error_status NE 0 THEN BEGIN
		    PRINT, 'Error index: ', Error_status
			PRINT, 'Error message: ', !ERROR_STATE.MSG
			CATCH,/CANCEL
		ENDIF
		CATCH,/CANCEL
		if interrupt_load eq 1 then print,'Finished cleaning up...'
endif else begin
	iPALM_data_cnt=n_elements(CGroupParams)
	save, curr_pwd,idl_pwd, iPALM_data_cnt, CGrpSize, ParamLimits, increment, nloops, spacer, grouping_radius, maxgrsize, disp_increment, GroupDisplay, RowNames, filename=td + sep + 'temp.sav'		;save variables for cluster cpu access
	GroupPeaks_Bridge, CGroupParams, temp_dir
	file_delete,td + sep + 'temp.sav', /ALLOW_NONEXISTENT, /QUIET
	file_delete, td, /ALLOW_NONEXISTENT, /QUIET
endelse

cd,curr_pwd

ReloadParamlists, Event1
OnGroupCentersButton, Event1
print,'iPALM Macro Fast: Finished Grouping'

Wid_ZExctractEngine_ID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_ZExctractEngine_iPALM')
ZExctractEngine = widget_info(Wid_ZExctractEngine_ID,/DropList_Select)
;if ZExctractEngine=1 then perform extraction on bridge, otherwise locally
if ZExctractEngine eq 1 then begin
	print,'iPALM Macro Fast: Start Second Extracting Z-coordinate (local)'
	OnExtractZCoord_Bridge
endif else begin
	print,'iPALM Macro Fast: Start Second Extracting Z-coordinate (bridge)'
	OnExtractZCoord, Event1
endelse

print,'iPALM Macro Fast: Finished Second Extracting Z-coordinate'
ReloadParamlists, Event1
OnUnZoomButton, Event1

print,'iPALM Macro Fast: Finished Second Extracting Z-coordinate'

FlipRotate=replicate({frt,present:0B,transp:0B,flip_h:0B,flip_v:0B},3)
NFrames=thisfitcond.Nframesmax    ;   long64(max(CGroupParams[9,*]))
GuideStarDrift=replicate({present:0B,xdrift:dblarr(Nframes),ydrift:dblarr(Nframes),zdrift:dblarr(Nframes)},3)
FiducialCoeff=replicate({fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)},3)

print,'iPALM Macro Fast: Saving the final data into file:',results_filename
SavFilenames=results_filename
save, CGroupParams, CGrpSize, ParamLimits, Image, b_set, xydsz, TotalRawData, RawFilenames, SavFilenames, MLRawFilenames,$
		GuideStarDrift, FiducialCoeff, FlipRotate, thisfitcond, RowNames, $
		lambda_vac,nd_water, nd_oil, wind_range, z_unwrap_coeff, ellipticity_slopes, $
		nm_per_pixel, wfilename, PkWidth_offset, Z_scale_multiplier, grouping_gap, grouping_radius100, filename=results_filename

print,'iPALM Macro Fast: Finished'
widget_control,event.top,/destroy
AnchorPnts=dblarr(6,AnchPnts_MaxNum)
ZPnts=dblarr(3,AnchPnts_MaxNum)
print,'iPALM Macro Fast: total time (sec)', (SYSTIME(/SECONDS)-Start_Time)
end
;
;-----------------------------------------------------------------
;
pro Recalculate_Thisfitconds, RawDataFiles, ThisFitConds
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
	sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
	if n_elements(filen) ne 0 then orig_filen=filen
	nlabels=n_elements(RawDataFiles)
	def_wind=!D.window
	for i=0,nlabels-1 do begin
		pos=strpos(RawDataFiles[i],sep,/REVERSE_OFFSET,/REVERSE_SEARCH)
		pth=strmid(RawDataFiles[i],0,pos)
		RawDataFileTxt=strmid(RawDataFiles[i],(pos+1))+'.txt'
		ReadThisFitCond, RawDataFileTxt, pth, filen, ini_filename, thisfitcond
		thisfitcond.fliphor=0
		thisfitcond.flipvert=0
		if i eq 0 then ThisFitConds=replicate(thisfitcond,nlabels) else ThisFitConds[i]=thisfitcond
	endfor
	thisfitcond.zerodark=total(ThisFitConds[*].zerodark)
	thisfitcond.thresholdcriteria=total(ThisFitConds[*].thresholdcriteria)*0.6
	thisfitcond.LimChiSq=total(ThisFitConds[*].LimChiSq)
	thisfitcond.cntpere=mean(ThisFitConds[*].cntpere)
	thisfitcond.fliphor=0
	thisfitcond.flipvert=0
end
;
;-----------------------------------------------------------------
;
pro iPALM_Macro_Fast, RawDataFiles, ThisFitConds, GStarDrifts, FidCoeffs, FlipRot,iPALM_MacroParameters		; Transformation core called by TransformRaw_Save_SaveSum_MenuItem
;This version (Short) only transformes chunks and leaves them in TEMP directory without combining into large transformed files.
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
	def_wind=!D.window
	sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
	nlabels=n_elements(RawDataFiles)
	thisfitcond_sum=thisfitcond

	if n_elements(orig_filen) ne 0 then filen=orig_filen
	xsz=thisfitcond.xsz													;number of x pixels
	ysz=thisfitcond.ysz													;number of y pixels
	;if xsz gt 256 then increment = 100
	increment = 50*(fix(384./sqrt(float(xsz)*float(ysz)))>1)				;setup loopback conditions to write multiple files

	if thisfitcond.FrmN le 500 then increment=thisfitcond.FrmN-thisfitcond.Frm0+1
	print,'Max cluster nodes to be used:', n_cluster_nodes_max
	;n_cluster_nodes_max = 256
	nloops = Fix((thisfitcond.FrmN-thisfitcond.Frm0+1)/increment) < n_cluster_nodes_max			;nloops=Fix((framelast-framefirst)/increment)
		;don't allow to use more then n_cluster_nodes_max cluster cores
	increment = long(floor((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/nloops))
	nloops = fix(ceil((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/increment))

	frm_first_erase=''
	frm_last_erase=''
	totfr=strtrim(thisfitcond.FrmN,2)

		DisplayType=-1			;turns of all displays during processing
		if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else	idl_pwd=pref_get('IDL_WDE_START_DIR')
		cd,current=curr_pwd
		FILE_MKDIR,curr_pwd+'/temp'
		do_save_sum=1
		pos=intarr(nlabels)
		temp_data_files=strarr(nlabels,nloops)
		sum_data_files=strarr(nloops)
		temp_idl_fnames=strarr(nloops)
		for nlps=0,nloops-1 do begin			;reassemble little dat files from all the workers into on big one
			framefirst=	thisfitcond.Frm0 + (nlps)*increment					;first frame in batch
			framelast=((thisfitcond.Frm0 + (nlps+1L)*increment-1L)<thisfitcond.Nframesmax) < thisfitcond.FrmN
			Nframes=(framelast-framefirst+1) 								;number of frames to extract in file

			for i=0,nlabels-1 do begin
				pos[i]=strpos(RawDataFiles[i],sep,/REVERSE_OFFSET,/REVERSE_SEARCH)
				temp_data_files[i,nlps]=curr_pwd+'/temp/Cam'+strtrim((i+1),2)+'_trn_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.dat'
			endfor
			sum_data_files[nlps]=curr_pwd+'/temp/Sum_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.dat'
			temp_idl_fnames[nlps]=curr_pwd+'/temp/Sum_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'_IDL.sav'
		endfor
		save, curr_pwd, idl_pwd, pth, nlabels, ThisFitConds, nloops, increment, do_save_sum, temp_data_files, sum_data_files, temp_idl_fnames, iPALM_MacroParameters, $
				ini_filename, thisfitcond, RawDataFiles, GStarDrifts, FidCoeffs, FlipRot, $
				aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius,$
				filename='temp/temp.sav'		;save variables for cluster cpu access
		print,'sh '+idl_pwd+'/ipalm_macro.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd		;Spawn workers in cluster
		spawn,'sh '+idl_pwd+'/ipalm_macro.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd		;Spawn workers in cluster
		cd,curr_pwd
 	wset,def_wind
end
;
;------------------------------------------------------------------------------------
;
Pro	ipalm_macro_worker,nlps,data_dir						;spawn mulitple copies of this programs for cluster
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
Nlps=ulong((COMMAND_LINE_ARGS())[0])
data_dir=(COMMAND_LINE_ARGS())[1]
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
cd,data_dir
restore,'temp/temp.sav'
print,'worker started',nlabels,nloops, pth,thisfitcond,RawDataFiles, FidCoeffs, FlipRot
;save, curr_pwd, idl_pwd, pth, nlabels, ThisFitConds, nloops, increment, do_save_sum, temp_data_files, sum_data_files, temp_idl_fnames, iPALM_MacroParameters,$
;				ini_filename, thisfitcond, RawDataFiles, GStarDrifts, FidCoeffs, FlipRot, aa, wind_range, filename='temp/temp.sav'		;save variables for cluster cpu access

for i=5,20 do close,i

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	print,(' Cluster Error:  '+!ERROR_STATE.msg)
	print,(' Cluster Error:  '+!ERROR_STATE.sys_msg)
	CATCH, /CANCEL
	return
ENDIF

xsz=thisfitcond.xsz												;number of x pixels
ysz=thisfitcond.ysz												;number of y pixels
framefirst=	thisfitcond.Frm0 + nlps*increment					;first frame in batch
framelast=((thisfitcond.Frm0 + (nlps+1)*increment-1)<thisfitcond.Nframesmax) < thisfitcond.FrmN
Nframes=(framelast-framefirst+1) 								;number of frames to extract in file
print,'Total number of Frames in this subset:  ',Nframes
data=uintarr(xsz,ysz,Nframes)

tempfiles=temp_data_files[*,Nlps]
for i=0,nlabels-1 do begin
	openr,10+i,RawDataFiles[i]+'.dat'
endfor
sumfile=sum_data_files[Nlps]

print, 'Opened all files to read'

for i=0,nlabels-1 do begin
	point_lun,10+i,2ull*xsz*ysz*framefirst
	readu,10+i,data
	if Nframes eq 1 then begin	; only one fame in a subset
		if FlipRot[i].present then begin
			if FlipRot[i].transp then data=transpose(temporary(data))
			if FlipRot[i].flip_h then begin
				data=transpose(temporary(data))
				data=reverse(temporary(data),2)
				data=transpose(temporary(data))
			endif
			if FlipRot[i].flip_v then data=reverse(temporary(data),2)
		endif
		if GStarDrifts[i].present then begin
 			P=[[GStarDrifts[i].xdrift[framefirst],0],[1,0]]
			Q=[[GStarDrifts[i].ydrift[framefirst],1],[0,0]]
			data[*,*]=POLY_2D(temporary(data[*,*]),P,Q,1)
		endif
		if FidCoeffs[i].present then $
				data[*,*]=POLY_2D(temporary(data[*,*]),FidCoeffs[i].P,FidCoeffs[i].Q,1)
	endif else begin		; more then one frame
		if FlipRot[i].present then begin
			if FlipRot[i].transp then data=transpose(temporary(data),[1,0,2])
			if FlipRot[i].flip_h then begin
				data=transpose(temporary(data),[1,0,2])
				data=reverse(temporary(data),2)
				data=transpose(temporary(data),[1,0,2])
			endif
			if FlipRot[i].flip_v then data=reverse(temporary(data),2)
		endif
		for k=0,nframes-1 do begin
			if GStarDrifts[i].present then begin
 				P=[[GStarDrifts[i].xdrift[framefirst+k],0],[1,0]]
				Q=[[GStarDrifts[i].ydrift[framefirst+k],1],[0,0]]
				data[*,*,k]=POLY_2D(temporary(data[*,*,k]),P,Q,1)
			endif
			if FidCoeffs[i].present then $
				data[*,*,k]=POLY_2D(temporary(data[*,*,k]),FidCoeffs[i].P,FidCoeffs[i].Q,1)
		endfor
	endelse
	if i eq 0 then Cam1_data=data
	if i eq 1 then Cam2_data=data
	if i eq 2 then Cam3_data=data

	SumData=(i eq 0) ? data : data+SumData
endfor
print, 'Finished the data transformation, closed data files'

;**************------------------  steps below are only done as a part of "fast-track" ----------------------------------------------

	print,'started the data processing'
	if n_elements(CGrpSize) eq 0 then CGrpSize=49
	file_dot_pos=strpos(sum_data_files[Nlps],'.',/REVERSE_OFFSET,/REVERSE_SEARCH)
	thefile_no_exten = strmid(sum_data_files[Nlps],0,file_dot_pos); strip file extension

	DisplayType=-1													;set to no displays
	DisplaySet=DisplayType											;set to no displays
	print,'Started processinmg frames  '+strtrim(framefirst,2)+'-'+strtrim(framelast,2)
	;data=float(ReadData(thefile_no_exten,thisfitcond,framefirst,Nframes))	;Reads thefile and returns data (bunch of frames) in (units of photons)
	zerodark=thisfitcond.zerodark										;zero dark count in CCD counts
	counts_per_e=thisfitcond.cntpere									;counts per electron CCD sensitivity
	data=((float(temporary(SumData))-zerodark)/counts_per_e)>0.
	print,'Converted the data'
	if Nframes gt 1 then totdat=total(data[*,*,0L:Nframes-1L],3)/Nframes else totdat=data
	Apeakparams = ParamsofShortStackofFrames(data,DisplayType,thisfitcond,framefirst)
	print, 'Extracted peaks'
	xydsz=[xsz,ysz]
	sz=size(Apeakparams)
	CGroupParams=dblarr(CGrpSize,sz[1])
	CGroupParams[0:1,*]=ApeakParams.A[0:1]
	CGroupParams[2,*]=ApeakParams.peakx
	CGroupParams[3,*]=ApeakParams.peaky
	CGroupParams[4:5,*]=ApeakParams.A[2:3]
	CGroupParams[6,*]=ApeakParams.NPhot
	CGroupParams[7,*]=ApeakParams.ChiSq
	CGroupParams[8,*]=ApeakParams.FitOK
	CGroupParams[9,*]=ApeakParams.FrameIndex
	CGroupParams[10,*]=ApeakParams.PeakIndex
	CGroupParams[11,*]=dindgen(sz[1])
	CGroupParams[12,*]=ApeakParams.A[2]*ApeakParams.A[3]
	CGroupParams[13,*]=ApeakParams.Sigma2[1]
	CGroupParams[14:15,*]=ApeakParams.Sigma2[4:5]
	CGroupParams[16:17,*]=ApeakParams.Sigma2[2:3]
	CGroupParams[32,*]=ApeakParams.A[6]
	TotalRawData = totdat
	if CGrpSize ge 49 then CGroupParams[43,*]=(ApeakParams.A[2]-ApeakParams.A[3])/(ApeakParams.A[2]+ApeakParams.A[3])


		;iPALM_MacroParameters = [100.0,	0.4,	0.4,	40.0,	0.1,	1.3,	1.4,	0.01,	0.6,	0.01]
	; iPALM_MacroParameters = [Nph,	Full Sigma X Max. (pix.),	Full Sigma Y Max. (pix.),	Sigma Z Max. (nm), Coherence Min.,	Coherence Max,	...
	; iPALM_MacroParameters = ...	Max A for (Wx-A)*(Wx-A)<B,	B for (Wx-A)*(Wx-A)<B,	Min A  for (Wx-A)*(Wx-A)>B,	B for (Wx-A)*(Wx-A)>B ]

	filter1 = ((CGroupParams[8,*] eq 1) or (CGroupParams[8,*] eq 2)) $
		AND (CGroupParams[6,*] ge iPALM_MacroParameters[0])	AND (CGroupParams[16,*] le iPALM_MacroParameters[1]) $
		AND (CGroupParams[17,*] le iPALM_MacroParameters[2])	$
		AND (((CGroupParams[4,*]-iPALM_MacroParameters[6]) > 0.0)*((CGroupParams[5,*]-iPALM_MacroParameters[6])> 0.0) le iPALM_MacroParameters[7])	$
		AND (((CGroupParams[4,*]-iPALM_MacroParameters[8]) > 0.0)*((CGroupParams[5,*]-iPALM_MacroParameters[8])> 0.0) ge iPALM_MacroParameters[9])
	pk_indecis=where(filter1,cnt)

	if cnt lt 1 then begin
		print,('Filter returned no valid peaks')
		return      ; if data not loaded return
	endif
	CGroupParams=temporary(CGroupParams[*,pk_indecis])

	print,'iPALM Macro Fast: purged not OK peaks and applied filters (all but z, coherence)'
	print,'Total number of peaks=',cnt
	print,'iPALM Macro Fast: using Disptype and SygmaSym: ',DisplayType,thisfitcond.SigmaSym

	d = thisfitcond.MaskSize		; d=5.								half size of region of a selected peak
																;setup twinkle structure with all the fitted peak information
	peakparams = {twinkle,frameindex:0l,peakindex:0l,fitOK:0,peakx:0.0,peaky:0.0,A:fltarr(7),sigma1:fltarr(7),sigma2:fltarr(7),chisq:0.0,Nphot:0l}
	peakparams.A=[0.0,1.0,1.2,1.2,d,d,0.]
	print,'starting the worker process, Frames=',framefirst,framelast
	firstpeak = -1
	lastpeak = 0
	for Label=0,nlabels-1 do begin
		;data=ReadData(MLRawFilenames[Label],ThisFitConds[Label],framefirst,Nframes)	;Reads thefile and returns data (bunch of frames) in (units of photons)
		if Label eq 0 then data = Cam1_data
		if Label eq 1 then data = Cam2_data
		if Label eq 2 then data = Cam3_data
		zerodark = ThisFitConds[Label].zerodark
		counts_per_e=ThisFitConds[Label].cntpere
		data=((float(temporary(data))-zerodark)/counts_per_e)>0.
		for frameindx=0l,Nframes-1 do begin
			if (frameindx mod 50) eq 0 then print,'iPALM Macro Fast: Re-extract peaks: Frameindex=',frameindx
			if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data ;clip from transformed dat, frame with correct label
			peaks_in_frame=long(where((CGroupParams[9,*] eq (frameindx+framefirst)),num_peaks))
			if (firstpeak eq -1) and (num_peaks gt 0) then firstpeak = min(peaks_in_frame) > 0
			if lastpeak lt max(peaks_in_frame) then lastpeak = max(peaks_in_frame)
			for ii=0L,num_peaks-1 do begin
					peakparams.A[0:1]=CGroupParams[0:1,peaks_in_frame[ii]]/3.			;base & amplitude
					peakparams.A[2:3]=CGroupParams[4:5,peaks_in_frame[ii]]				;Fitted sigma x and y of Gaussian
					peakparams.A[4:5]=CGroupParams[2:3,peaks_in_frame[ii]]				;Fitted x and y center of Gaussian
					Dispxy=[CGroupParams[10,peaks_in_frame[ii]],Label]					;tell index of chosen frame and label
					peakx=fix(peakparams.A[4])											;fitted x center of totaled triplet
					peaky=fix(peakparams.A[5])											;fitted y center of totaled triplet
					peakparams.A[4:5]=peakparams.A[4:5]-[peakx,peaky]+d-0.5
					fita = [1,1,1,1,0,0]												;fit only the base and amplitude
					FindnWackaPeak, clip, d, peakparams, fita, result, ThisFitConds[Label], DisplaySet, peakx, peaky, criteria, Dispxy, 0		;Find,fit,remove target peak & return fit parameters
					CGroupParams[27+Label,peaks_in_frame[ii]]= peakparams.A[1]			; Fitted Peak Amplitude
					CGroupParams[30,peaks_in_frame[ii]]=+peakparams.fitOK*10^Label											; New FitOK
					;CGroupParams[31+Label,peaks_in_frame[ii]]=peakparams.sigma2[1]
			endfor
		endfor
	endfor

	print,'iPALM Macro Fast: Finished ReExtractig multilabel'

	print,'iPALM Macro Fast: Start First Extracting Z-coordinate'
	OnExtractZCoord_Core, CGroupParams, 0, 0	; peaks only, do not display progress reports
	print,'iPALM Macro Fast: Finished First Extracting Z-coordinate'
	; filter and purge bad peaks: Sigma Z
	filter = (CGroupParams[35,*] le iPALM_MacroParameters[3]) AND (CGroupParams[36,*] ge iPALM_MacroParameters[4]) AND (CGroupParams[36,*] le iPALM_MacroParameters[5])
	pk_indecis=where(filter,cnt)
	if cnt lt 1 then begin
		print,('Filter returned no valid peaks')
		return      ; if data not loaded return
	endif
	CGroupParams=temporary(CGroupParams[*,pk_indecis])

	save,CGroupParams,xydsz,totdat,filename=temp_idl_fnames[Nlps]


;spawn,'sync'
;spawn,'sync'
print,'Wrote file '+temp_idl_fnames[Nlps]
end


;
;-----------------------------------------------------------------
;
pro iPALM_Macro_Fast_Bridge, RawDataFiles, ThisFitConds, GStarDrifts, FidCoeffs, FlipRot,iPALM_MacroParameters		; Transformation core called by TransformRaw_Save_SaveSum_MenuItem
;This version (Short) only transformes chunks and leaves them in TEMP directory without combining into large transformed files.
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

	def_wind=!D.window
	sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
	nlabels=n_elements(RawDataFiles)
	thisfitcond_sum=thisfitcond

	if n_elements(orig_filen) ne 0 then filen=orig_filen
	xsz=thisfitcond.xsz													;number of x pixels
	ysz=thisfitcond.ysz													;number of y pixels
	increment = 50*(fix(384./sqrt(float(xsz)*float(ysz)))>1)				;setup loopback conditions to write multiple files

	if thisfitcond.FrmN le 500 then increment=thisfitcond.FrmN-thisfitcond.Frm0+1

	ncores_cluster = fix(strtrim(GETENV('LSB_DJOB_NUMPROC'),2))
	n_br_loops = ncores_cluster gt 0 ? ncores_cluster : !CPU.HW_NCPU
	print, n_br_loops, '  CPU cores are present'
	nloops = (Fix((thisfitcond.FrmN-thisfitcond.Frm0+1)/increment) < n_br_loops) < n_br_max		;nloops=Fix((framelast-framefirst)/increment)
	print, 'will set up ', nloops, ' bridge processes'
	; don't allow more bridge processes than there are CPU's
	increment = long(ceil((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/nloops))
	nloops = fix(ceil((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/increment))
	print,'Will start'+strtrim(nloops)+' bridge child processes''
	DisplayType=-1			;turns of all displays during processing
	if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else	idl_pwd=pref_get('IDL_WDE_START_DIR')
	cd,current=curr_pwd
	FILE_MKDIR,curr_pwd+'/temp'
	print,'data_dir:',curr_pwd
	print,'IDL_dir:',IDL_pwd

	obridge=obj_new("IDL_IDLBridge", output='')
	for i=1, nloops-1 do obridge=[obridge, obj_new("IDL_IDLBridge", output='')]

	frm_first_erase=''
	frm_last_erase=''
	totfr=strtrim(thisfitcond.FrmN,2)

	do_save_sum=1
	pos=intarr(nlabels)
	temp_data_files=strarr(nlabels,nloops)
	sum_data_files=strarr(nloops)
	temp_idl_fnames=strarr(nloops)
	for nlps=0,nloops-1 do begin			;reassemble little dat files from all the workers into on big one
		framefirst=	thisfitcond.Frm0 + (nlps)*increment					;first frame in batch
		framelast=((thisfitcond.Frm0 + (nlps+1L)*increment-1L)<thisfitcond.Nframesmax) < thisfitcond.FrmN
		Nframes=(framelast-framefirst+1) 								;number of frames to extract in file

		for i=0,nlabels-1 do begin
			pos[i]=strpos(RawDataFiles[i],sep,/REVERSE_OFFSET,/REVERSE_SEARCH)
			temp_data_files[i,nlps]=curr_pwd+'/temp/Cam'+strtrim((i+1),2)+'_trn_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.dat'
		endfor
		sum_data_files[nlps]=curr_pwd+'/temp/Sum_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.dat'
		temp_idl_fnames[nlps]=curr_pwd+'/temp/Sum_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'_IDL.sav'
	endfor
	save, curr_pwd, idl_pwd, pth, nlabels, ThisFitConds, nloops, increment, do_save_sum, temp_data_files, sum_data_files, temp_idl_fnames, iPALM_MacroParameters, $
				ini_filename, thisfitcond, RawDataFiles, GStarDrifts, FidCoeffs, FlipRot, $
				aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius,$
				filename='temp/temp.sav'		;save variables for cluster cpu access

	shmName='Status_reports'
	max_len=300
	SHMMAP,shmName,/BYTE, Dimension=nloops*max_len, GET_OS_HANDLE=OS_handle_val
	Reports=SHMVAR(shmName)

	for i=0, nloops-1 do begin
		obridge[i]->setvar, 'nlps',i
		obridge[i]->setvar, 'data_dir',curr_pwd
		obridge[i]->setvar, 'IDL_dir',IDL_pwd
		obridge[i]->setvar, 'OS_handle_val',OS_handle_val
		print,'bridge ',i,'  set variables'
		obridge[i]->execute,'cd, IDL_dir'
		print,'bridge ',i,'  changed directory'
		obridge[i]->execute,"restore,'iPalm_Macro_Worker_Bridge.sav'"
		obridge[i]->execute,'iPalm_Macro_Worker_Bridge,nlps,data_dir,OS_handle_val',/NOWAIT
		print,'bridge ',i,'  started'
	endfor

	Alldone = 0
	bridge_done=intarr(nloops)
	bridge_done_prev=bridge_done
	Reports_prev=Reports
	while alldone EQ 0 do begin
		wait,5
		Alldone = 1
		for i=0, nloops-1 do begin
			bridge_done[i]=obridge[i]->Status()
			Alldone = Alldone * (bridge_done[i] ne 1)
		endfor
		rep_new=~ARRAY_EQUAL(Reports,Reports_prev)
		bridge_done_new=~ARRAY_EQUAL(bridge_done,bridge_done_prev)
		if rep_new or bridge_done_new then begin
			bridge_done_prev=bridge_done
			Reports_prev=Reports
			for i=0, nloops-1 do print,'Bridge',i,'  status:',bridge_done[i],';    ',string(Reports[(i*max_len):((i+1)*max_len-1)])
		endif
	endwhile
	SHMUnmap, shmName

	for i=0, nloops-1 do obj_destroy, obridge[i]

	cd,curr_pwd
 	wset,def_wind
end
;
;------------------------------------------------------------------------------------
;
Pro	iPalm_Macro_Worker_Bridge,nlps,data_dir,OS_handle_val						;spawn mulitple copies of this programs for IDL bridge
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
;Profiler
cd,data_dir
restore,'temp/temp.sav'
print,'worker started',nlabels,nloops, pth,thisfitcond,RawDataFiles, FidCoeffs, FlipRot
;save, curr_pwd, idl_pwd, pth, nlabels, ThisFitConds, nloops, increment, do_save_sum, temp_data_files, sum_data_files, temp_idl_fnames, iPALM_MacroParameters,$
;				ini_filename, thisfitcond, RawDataFiles, GStarDrifts, FidCoeffs, FlipRot, aa, wind_range, filename='temp/temp.sav'		;save variables for cluster cpu access

for i=5,20 do close,i

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	rep=' iPalm_Macro_Worker_Bridge Error:  '+!ERROR_STATE.msg
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	CATCH, /CANCEL
	return
ENDIF
shmName='Status_reports'
max_len=300

SHMMAP,shmName,/BYTE, Dimension=nloops*max_len,OS_Handle=OS_handle_val
Reports=SHMVAR(shmName)

rep_i=nlps*max_len

xsz=thisfitcond.xsz												;number of x pixels
ysz=thisfitcond.ysz												;number of y pixels
framefirst=	thisfitcond.Frm0 + nlps*increment					;first frame in batch
framelast=((thisfitcond.Frm0 + (nlps+1)*increment-1)<thisfitcond.Nframesmax) < thisfitcond.FrmN
Nframes=(framelast-framefirst+1) 								;number of frames to extract in file
rep='Total number of Frames in this subset:  '+string(Nframes)
if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
data=uintarr(xsz,ysz,Nframes)

tempfiles=temp_data_files[*,Nlps]
for i=0,nlabels-1 do begin
	openr,10+i,RawDataFiles[i]+'.dat'
endfor
sumfile=sum_data_files[Nlps]

rep = 'Opened all files to read'
if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)

for i=0,nlabels-1 do begin
	point_lun,10+i,2ull*xsz*ysz*framefirst
	readu,10+i,data
	if Nframes eq 1 then begin	; only one fame in a subset
		if FlipRot[i].present then begin
			if FlipRot[i].transp then data=transpose(temporary(data))
			if FlipRot[i].flip_h then begin
				data=transpose(temporary(data))
				data=reverse(temporary(data),2)
				data=transpose(temporary(data))
			endif
			if FlipRot[i].flip_v then data=reverse(temporary(data),2)
		endif
		if GStarDrifts[i].present then begin
 			P=[[GStarDrifts[i].xdrift[framefirst],0],[1,0]]
			Q=[[GStarDrifts[i].ydrift[framefirst],1],[0,0]]
			data[*,*]=POLY_2D(temporary(data[*,*]),P,Q,1)
		endif
		if FidCoeffs[i].present then $
				data[*,*]=POLY_2D(temporary(data[*,*]),FidCoeffs[i].P,FidCoeffs[i].Q,1)
	endif else begin		; more then one frame
		if FlipRot[i].present then begin
			if FlipRot[i].transp then data=transpose(temporary(data),[1,0,2])
			if FlipRot[i].flip_h then begin
				data=transpose(temporary(data),[1,0,2])
				data=reverse(temporary(data),2)
				data=transpose(temporary(data),[1,0,2])
			endif
			if FlipRot[i].flip_v then data=reverse(temporary(data),2)
		endif
		for k=0,nframes-1 do begin
			if GStarDrifts[i].present then begin
 				P=[[GStarDrifts[i].xdrift[framefirst+k],0],[1,0]]
				Q=[[GStarDrifts[i].ydrift[framefirst+k],1],[0,0]]
				data[*,*,k]=POLY_2D(temporary(data[*,*,k]),P,Q,1)
			endif
			if FidCoeffs[i].present then $
				data[*,*,k]=POLY_2D(temporary(data[*,*,k]),FidCoeffs[i].P,FidCoeffs[i].Q,1)
		endfor
	endelse
	if i eq 0 then Cam1_data=data
	if i eq 1 then Cam2_data=data
	if i eq 2 then Cam3_data=data

	SumData=(i eq 0) ? data : data+SumData
endfor

rep = 'Finished the data transformation, closed data files'
if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)

;**************------------------  steps below are only done as a part of "fast-track" ----------------------------------------------

	rep = 'started the data processing'
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	if n_elements(CGrpSize) eq 0 then CGrpSize=49
	file_dot_pos=strpos(sum_data_files[Nlps],'.',/REVERSE_OFFSET,/REVERSE_SEARCH)
	thefile_no_exten = strmid(sum_data_files[Nlps],0,file_dot_pos); strip file extension

	DisplayType=-1													;set to no displays
	DisplaySet=DisplayType											;set to no displays
	rep = 'Started processinmg frames  '+strtrim(framefirst,2)+'-'+strtrim(framelast,2)
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	zerodark=thisfitcond.zerodark										;zero dark count in CCD counts
	counts_per_e=thisfitcond.cntpere									;counts per electron CCD sensitivity
	data=((float(temporary(SumData))-zerodark)/counts_per_e)>0.
	rep = 'Converted the data'
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	if Nframes gt 1 then totdat=total(data[*,*,0L:Nframes-1L],3)/Nframes else totdat=data
	Apeakparams = ParamsofShortStackofFrames(data,DisplayType,thisfitcond,framefirst)
	rep = 'Extracted peaks'
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	xydsz=[xsz,ysz]
	sz=size(Apeakparams)
	CGroupParams=dblarr(CGrpSize,sz[1])
	CGroupParams[0:1,*]=ApeakParams.A[0:1]
	CGroupParams[2,*]=ApeakParams.peakx
	CGroupParams[3,*]=ApeakParams.peaky
	CGroupParams[4:5,*]=ApeakParams.A[2:3]
	CGroupParams[6,*]=ApeakParams.NPhot
	CGroupParams[7,*]=ApeakParams.ChiSq
	CGroupParams[8,*]=ApeakParams.FitOK
	CGroupParams[9,*]=ApeakParams.FrameIndex
	CGroupParams[10,*]=ApeakParams.PeakIndex
	CGroupParams[11,*]=dindgen(sz[1])
	CGroupParams[12,*]=ApeakParams.A[2]*ApeakParams.A[3]
	CGroupParams[13,*]=ApeakParams.Sigma2[1]
	CGroupParams[14:15,*]=ApeakParams.Sigma2[4:5]
	CGroupParams[16:17,*]=ApeakParams.Sigma2[2:3]
	CGroupParams[32,*]=ApeakParams.A[6]
	TotalRawData = totdat
	if CGrpSize ge 49 then CGroupParams[43,*]=(ApeakParams.A[2]-ApeakParams.A[3])/(ApeakParams.A[2]+ApeakParams.A[3])

	filter1 = ((CGroupParams[8,*] eq 1) or (CGroupParams[8,*] eq 2)) $
		AND (CGroupParams[6,*] ge iPALM_MacroParameters[0])	AND (CGroupParams[16,*] le iPALM_MacroParameters[1]) $
		AND (CGroupParams[17,*] le iPALM_MacroParameters[2])	$
		AND (((CGroupParams[4,*]-iPALM_MacroParameters[6]) > 0.0)*((CGroupParams[5,*]-iPALM_MacroParameters[6])> 0.0) le iPALM_MacroParameters[7])	$
		AND (((CGroupParams[4,*]-iPALM_MacroParameters[8]) > 0.0)*((CGroupParams[5,*]-iPALM_MacroParameters[8])> 0.0) ge iPALM_MacroParameters[9])
	pk_indecis=where(filter1,cnt)

	if cnt lt 1 then begin
		rep = 'Filter returned no valid peaks'
		if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
		Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
		return      ; if data not loaded return
	endif
	CGroupParams=temporary(CGroupParams[*,pk_indecis])

	rep = 'iPALM Macro Fast: purged not OK peaks and applied filters (all but z, coherence)'
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	rep = 'Total number of peaks='+strtrim(cnt,2)
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	rep = 'iPALM Macro Fast: using Disptype and SygmaSym: '+ strtrim(DisplayType,2)+','+strtrim(thisfitcond.SigmaSym,2)
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)

	d = thisfitcond.MaskSize		; d=5.								half size of region of a selected peak
																;setup twinkle structure with all the fitted peak information
	peakparams = {twinkle,frameindex:0l,peakindex:0l,fitOK:0,peakx:0.0,peaky:0.0,A:fltarr(7),sigma1:fltarr(7),sigma2:fltarr(7),chisq:0.0,Nphot:0l}
	peakparams.A=[0.0,1.0,1.2,1.2,d,d,0.]
	rep = 'starting the worker process, Frames='+strtrim(framefirst,2)+'-'+strtrim(framelast,2)
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	firstpeak = -1
	lastpeak = 0
	for Label=0,nlabels-1 do begin
		;data=ReadData(MLRawFilenames[Label],ThisFitConds[Label],framefirst,Nframes)	;Reads thefile and returns data (bunch of frames) in (units of photons)
		if Label eq 0 then data = Cam1_data
		if Label eq 1 then data = Cam2_data
		if Label eq 2 then data = Cam3_data
		zerodark = ThisFitConds[Label].zerodark
		counts_per_e=ThisFitConds[Label].cntpere
		data=((float(temporary(data))-zerodark)/counts_per_e)>0.
		for frameindx=0l,Nframes-1 do begin
			if (frameindx mod 50) eq 0 then begin
				rep='iPALM Macro Fast: Re-extract peaks: Frameindex='+strtrim(frameindx,2)
				if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
				Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
			endif
			if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data ;clip from transformed dat, frame with correct label
			peaks_in_frame=long(where((CGroupParams[9,*] eq (frameindx+framefirst)),num_peaks))
			if (firstpeak eq -1) and (num_peaks gt 0) then firstpeak = min(peaks_in_frame) > 0
			if lastpeak lt max(peaks_in_frame) then lastpeak = max(peaks_in_frame)
			for ii=0L,num_peaks-1 do begin
					peakparams.A[0:1]=CGroupParams[0:1,peaks_in_frame[ii]]/3.			;base & amplitude
					peakparams.A[2:3]=CGroupParams[4:5,peaks_in_frame[ii]]				;Fitted sigma x and y of Gaussian
					peakparams.A[4:5]=CGroupParams[2:3,peaks_in_frame[ii]]				;Fitted x and y center of Gaussian
					Dispxy=[CGroupParams[10,peaks_in_frame[ii]],Label]					;tell index of chosen frame and label
					peakx=fix(peakparams.A[4])											;fitted x center of totaled triplet
					peaky=fix(peakparams.A[5])											;fitted y center of totaled triplet
					peakparams.A[4:5]=peakparams.A[4:5]-[peakx,peaky]+d-0.5
					fita = [1,1,1,1,0,0]												;fit only the base and amplitude
					FindnWackaPeak, clip, d, peakparams, fita, result, ThisFitConds[Label], DisplaySet, peakx, peaky, criteria, Dispxy, 0		;Find,fit,remove target peak & return fit parameters
					CGroupParams[27+Label,peaks_in_frame[ii]]= peakparams.A[1]			; Fitted Peak Amplitude
					CGroupParams[30,peaks_in_frame[ii]]=+peakparams.fitOK*10^Label											; New FitOK
					;CGroupParams[31+Label,peaks_in_frame[ii]]=peakparams.sigma2[1]
			endfor
		endfor
	endfor

	rep = 'iPALM Macro Fast: Finished ReExtractig multilabel'
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)

	rep = 'iPALM Macro Fast: Start First Extracting Z-coordinate'
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)

	OnExtractZCoord_Core, CGroupParams, 0, 0	; peaks only, do not display progress reports

	rep = 'iPALM Macro Fast: Finished First Extracting Z-coordinate'
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)

	; filter and purge bad peaks: Sigma Z
	filter = (CGroupParams[35,*] le iPALM_MacroParameters[3]) AND (CGroupParams[36,*] ge iPALM_MacroParameters[4]) AND (CGroupParams[36,*] le iPALM_MacroParameters[5])
	pk_indecis=where(filter,cnt)
	if cnt lt 1 then begin
		rep = 'Filter returned no valid peaks'
		if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
		Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
		return      ; if data not loaded return
	endif
	CGroupParams=temporary(CGroupParams[*,pk_indecis])

	save,CGroupParams,xydsz,totdat,filename=temp_idl_fnames[Nlps]


;**************------------------ end of fast-track ---------------------------------------------------------------------------------

rep = 'Wrote file '+temp_idl_fnames[Nlps]
if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)

;Profiler,/report,output=output
;out_file='output'+strtrim(nloops,2)+'.txt'
;openw,nlps+3,out_file
;printf,nlps+3,output
;close,nlps+3
end
;
;-----------------------------------------------------------------
;
pro OnExtractZCoord_Bridge		; IDL Bridge: Extracts Z-coordinates using the fit parameters from selected WND file.
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

print,!CPU.HW_NCPU,'  CPU cores are present
nloops	=	!CPU.HW_NCPU < n_br_max
		; don't allow more bridge processes than there are CPU's
print, 'will start', nloops,' bridge child processes'
framefirst=long(ParamLimits[9,0])
framelast=long(ParamLimits[9,1])

increment = long(ceil((framelast-framefirst+1.0)/nloops))
nloops = long(ceil((framelast-framefirst+1.0)/increment)) > 1L
print,' Will start '+strtrim(nloops,2)+' bridge child processes'

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

shmName='Status_Rep_Zextr'
max_len=150
SHMMAP,shmName,/BYTE, Dimension=nloops*max_len, GET_OS_HANDLE=OS_handle_val1
Reports=SHMVAR(shmName)
shmName_data='iPALM_data'
iPALM_data_cnt=n_elements(CGroupParams)
SHMMAP,shmName_data,/FLOAT, Dimension=[CGrpSize,(iPALM_data_cnt/CGrpSize)], GET_OS_HANDLE=OS_handle_val2
CGroupParams_bridge=SHMVAR(shmName_data)
CGroupParams_bridge[0,0]=CGroupParams

save, curr_pwd,idl_pwd, iPALM_data_cnt, CGrpSize, ParamLimits, increment, nloops, $
	aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius,$
		filename='temp/temp.sav'

for i=0, nloops-1 do begin
	obridge[i]->setvar, 'nlps',i
	obridge[i]->setvar, 'data_dir',curr_pwd
	obridge[i]->setvar, 'IDL_dir',IDL_pwd
	obridge[i]->setvar, 'OS_handle_val1',OS_handle_val1
	obridge[i]->setvar, 'OS_handle_val2',OS_handle_val2
	print,'bridge ',i,'  set variables'
	obridge[i]->execute,'cd, IDL_dir'
	print,'bridge ',i,'  changed directory'
	obridge[i]->execute,"restore,'OnExtractZCoord_Bridge_Worker.sav'"
	obridge[i]->execute,'OnExtractZCoord_Bridge_Worker,nlps,data_dir,OS_handle_val1,OS_handle_val2',/NOWAIT
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
SHMUnmap, shmName

CGroupParams = CGroupParams_bridge
SHMUnmap, shmName_data
for nlps=0L,nloops-1 do	obj_destroy, obridge[nlps]

file_delete,'temp/temp.sav'
file_delete,'temp'
cd,curr_pwd
return
end
;
;-----------------------------------------------------------------
;
pro OnExtractZCoord_Bridge_Worker, nlps,data_dir,OS_handle_val1,OS_handle_val2		;spawn mulitple copies of this programs for IDL bridge
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius

cd,data_dir
restore,'temp/temp.sav'
print,'worker started, nloops=',nloops

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	rep=' OnExtractZCoord_Bridge_Worker Error:  '+!ERROR_STATE.msg
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	CATCH, /CANCEL
	return
ENDIF
shmName='Status_Rep_Zextr'
max_len=150
SHMMAP,shmName,/BYTE, Dimension=nloops*max_len,OS_Handle=OS_handle_val1
Reports=SHMVAR(shmName)
rep_i=nlps*max_len

shmName_data='iPALM_data'
SHMMAP,shmName_data,/FLOAT, Dimension=[CGrpSize,(iPALM_data_cnt/CGrpSize)],OS_Handle=OS_handle_val2
CGroupParams_bridge=SHMVAR(shmName_data)

framefirst=long(ParamLimits[9,0])
framelast=long(ParamLimits[9,1])
framestart=	framefirst + nlps*increment					;first frame in batch
framestop=(framestart+increment-1)<framelast

GoodPeaks=where((CGroupParams_bridge[9,*] ge framestart) and (CGroupParams_bridge[9,*] le framestop),OKpkcnt)
rep='Starting OnExtractZCoord_Core for framefirst='+strtrim(framestart,2)+',  framelast='+strtrim(framestop,2)+',  OKpkcnt='+strtrim(OKpkcnt,2)
if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
if OKpkcnt ge 1 then begin
	CGroupParamsGP=CGroupParams_bridge[*,GoodPeaks]
	OnExtractZCoord_Core, CGroupParamsGP, 1	, 0 ; groups only, do not display progress reports
	CGroupParams_bridge[*,GoodPeaks]=CGroupParamsGP
	rep='Re-saved the CGroupParams for frames:'+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)
endif else rep='No peaks detected for frames:'+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)

if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
return
end
