;
; IDL Event Callback Procedures
; Extract_Peaks_Multiple_TIFFs_eventcb
;
; Created by Gleb Shtengel:	05/08/2017 15:23.13
;

;
;-----------------------------------------------------------------
; Empty stub procedure used for autoloading.
;
pro Extract_Peaks_Multiple_TIFFs_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_Extract_Peaks_mTIFFs, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
common Glob, UseGlobIni_mTIFFs, GlobINI_FileName, Glob_lines
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
Common Multiple_PALM_TIFFs, DoPurge_mTIFFs, Purge_RowNames_mTIFFs, Purge_Params_mTIFFs

cd,current = pth

WID_TXT_mTIFFS_Directory_ID = Widget_Info(wWidget, find_by_uname='WID_TXT_mTIFFS_Directory')
widget_control,WID_TXT_mTIFFS_Directory_ID,SET_VALUE = pth

wtable_id = Widget_Info(wWidget, find_by_uname='WID_TABLE_InfoFile_mTIFFS')
if !VERSION.OS_family eq 'unix' then widget_control,wtable_id,COLUMN_WIDTH=[200,100],use_table_select = [ -1, 0, 0, 22 ]

IF LMGR(/VM) then TransformEngine=0	; Set TransformEngine=0 if  IDL is in Virtual Machine Mode

WidDListDispLevel = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_TransformEngine_mTIFFS')
widget_control,WidDListDispLevel,SET_DROPLIST_SELECT = TransformEngine

LoadThiFitCond,ini_filename,thisfitcond
thisfitcond.filetype = 1
thisfitcond.xsz = 0
thisfitcond.ysz = 0

WID_TABLE_InfoFile__mTIFFS_ID = Widget_Info(wWidget, find_by_uname='WID_TABLE_InfoFile_mTIFFS')
Fill_Fitting_Parameters_mTIFFS, WID_TABLE_InfoFile__mTIFFS_ID

WID_DROPLIST_SetSigmaFitSym_mTIFFS_ID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_SetSigmaFitSym_mTIFFS')
widget_control,WID_DROPLIST_SetSigmaFitSym_mTIFFS_ID,SET_DROPLIST_SELECT=thisfitcond.SigmaSym

WID_DROPLIST_TransformEngine_mTIFFS_ID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_TransformEngine_mTIFFS')
widget_control,WID_DROPLIST_TransformEngine_mTIFFS_ID,SET_DROPLIST_SELECT=TransformEngine

nfiles_text = ''
WID_LABEL_nfiles_mTIFFs_ID = Widget_Info(wWidget, find_by_uname='WID_LABEL_nfiles_mTIFFs')
widget_control,WID_LABEL_nfiles_mTIFFs_ID,SET_VALUE=nfiles_text

WID_LABEL_nfiles_Glob_mTIFFs_ID = Widget_Info(wWidget, find_by_uname='WID_LABEL_nfiles_Glob_mTIFFs')
widget_control,WID_LABEL_nfiles_Glob_mTIFFs_ID,SET_VALUE=nfiles_text

UseGlobIni_mTIFFs = 0

if strlen(wfilename) gt 1 then begin
	WFileWidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_WindFilename_Astig_MultiTIFF')
	widget_control,WFileWidID,SET_VALUE = wfilename
endif

WID_Filter_Parameters_mTIFFs_ID = Widget_Info(wWidget, find_by_uname = 'WID_Filter_Parameters_mTIFFs')
	n_par = n_elements(Purge_RowNames_mTIFFs)
	widget_control, WID_Filter_Parameters_mTIFFs_ID, ROW_LABELS = Purge_RowNames_mTIFFs, TABLE_YSIZE = n_par
	widget_control, WID_Filter_Parameters_mTIFFs_ID, COLUMN_WIDTH=[160,85,85],use_table_select = [ -1, 0, 1, (n_par-1) ]
	widget_control, WID_Filter_Parameters_mTIFFs_ID, set_value=transpose(Purge_Params_mTIFFs);, use_table_select=[0,0,3,(CGrpSize-1)]
	widget_control, WID_Filter_Parameters_mTIFFs_ID, /editable,/sensitive

end
;
;-----------------------------------------------------------------
;
pro Do_Change_Filter_Params_mTIFFs, Event
Common Multiple_PALM_TIFFs, DoPurge_mTIFFs, Purge_RowNames_mTIFFs, Purge_Params_mTIFFs
	widget_control,event.id,get_value=thevalue
	Purge_Params_mTIFFs[event.y,event.x]=thevalue[event.x,event.y]
	widget_control, event.id, set_value=transpose(Purge_Params_mTIFFs)
end
;
;-----------------------------------------------------------------
;
pro OnPickCalFile_Astig_MultiTIFF, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
	wfilename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.sav'],title='Select *WND.sav file to open')
	if wfilename ne '' then begin
		restore,filename=wfilename
		print,'Astigmatic Fit coefficients:', aa
		cd,fpath
		WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_WindFilename_Astig_MultiTIFF')
		widget_control,WFileWidID,SET_VALUE = wfilename
	endif
end
;
;-----------------------------------------------------------------
;
pro OnCancel_Extract_mTIFFS, Event
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro Set_SigmaFitSym_mTIFFS, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R
	SigmaSym=widget_info(event.id,/DropList_Select)		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
	thisfitcond.SigmaSym = SigmaSym
	;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
end
;
;-----------------------------------------------------------------
;
pro On_Select_Directory, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
fpath = Dialog_Pickfile(/read,/DIRECTORY)
WID_TXT_mTIFFS_Directory_ID = Widget_Info(Event.Top, find_by_uname='WID_TXT_mTIFFS_Directory')
widget_control,WID_TXT_mTIFFS_Directory_ID,SET_VALUE = fpath
if fpath ne '' then cd,fpath
end
;
;-----------------------------------------------------------------
;
pro On_Select_GolbIni, Event
filters = ['*.ini','*.txt']
GlobINI_FileName = Dialog_Pickfile(/read, FILTER = filters)
WID_TXT_mTIFFS_GlobINI_File_ID = Widget_Info(Event.Top, find_by_uname='WID_TXT_mTIFFS_GlobINI_File')
widget_control, WID_TXT_mTIFFS_GlobINI_File_ID, SET_VALUE = GlobINI_FileName
end
;
;-----------------------------------------------------------------
;
pro OnPickWINDFile_mTIFFS, Event
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
	WFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TXT_WindFilename_mTIFFS')
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
pro Set_TransformEngine_mTIFFS, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
	TransformEngine = widget_info(Event.id,/DropList_Select)
	print,'Set TransformEngine to ',TransformEngine
end
;
;-----------------------------------------------------------------
;
function extract_filenames_glob, filenames_to_search, glob_line
	return,filenames_to_search[where(strmatch(filenames_to_search, glob_line) eq 1)]
end
;
;-----------------------------------------------------------------
;
pro Set_UseGlobIni_mTIFFs, Event
common Glob, UseGlobIni_mTIFFs, GlobINI_FileName, Glob_lines
	UseGlobIni_mTIFFs = Widget_Info(event.id, /BUTTON_SET)
	;print,'Set UseGlobIni_mTIFFs = ',UseGlobIni_mTIFFs
end

;
;-----------------------------------------------------------------
;
pro On_ReFind_Files_mTIFFS, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Glob, UseGlobIni_mTIFFs, GlobINI_FileName, Glob_lines
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

WID_TXT_mTIFFS_Directory_ID = Widget_Info(Event.Top, find_by_uname='WID_TXT_mTIFFS_Directory')
widget_control,WID_TXT_mTIFFS_Directory_ID,GET_VALUE = fpath

WID_TXT_mTIFFS_FileMask_ID = Widget_Info(Event.Top, find_by_uname='WID_TXT_mTIFFS_FileMask')
widget_control,WID_TXT_mTIFFS_FileMask_ID,GET_VALUE = fmask

WID_TXT_mTIFFS_GlobINI_File_ID = Widget_Info(Event.Top, find_by_uname='WID_TXT_mTIFFS_GlobINI_File')
widget_control, WID_TXT_mTIFFS_GlobINI_File_ID, GET_VALUE = GlobINI_FileName

WID_BUTTON_Include_Subdirectories_ID = Widget_Info(Event.Top, find_by_uname='WID_BUTTON_Include_Subdirectories_mTIFFs')
include_subdir = Widget_Info(WID_BUTTON_Include_Subdirectories_ID, /BUTTON_SET)

WID_BUTTON_Excl_PKS_ID = Widget_Info(Event.Top, find_by_uname='WID_BUTTON_Excl_PKS')
Excl_PKS = Widget_Info(WID_BUTTON_Excl_PKS_ID, /BUTTON_SET)

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	print,text, ' cannot be loaded, or has incorrect structure'
	PRINT, 'Error index: ', Error_status
	PRINT, 'Error message: ', !ERROR_STATE.MSG
	CATCH, /CANCEL
	return
ENDIF

;fmask = "*.tif"

widget_control,/hourglass

if UseGlobIni_mTIFFs then begin
	if n_elements(RawFilenames) eq 0 then $
		if include_subdir then RawFilenames = FILE_SEARCH(fpath, fmask) else RawFilenames = FILE_SEARCH(fpath + sep + fmask)
    GlobINI_FileInfo=FILE_INFO(GlobINI_FileName)
	if ~(GlobINI_FileInfo.exists) then return
	openr, lun, GlobINI_FileName, /GET_LUN,ERROR = err
  	; If err is nonzero, something happened. Print the error message to
	IF (err NE 0) then PRINT,!ERROR_STATE.MSG
	eof_lun = EOF(lun)
 	array = ''
	line = ''
	wcnt=0
	WHILE eof_lun ne 1 DO BEGIN
		if eof_lun ne 1 then begin
			READF, lun, line
  			Glob_line = '*'+line
  			print,Glob_line
  			ind = where((strmatch(RawFilenames, Glob_line) eq 1),cnt)
  			print,'number of matching files:',cnt
			if cnt ge 1  then begin
  				if n_elements(Files) eq 0 then begin
				;Files = extract_filenames_glob(RawFilenames, Glob_line)
					Files = RawFilenames[ind]
  				endif else begin
  		 			;Files = [Files, extract_filenames_glob(RawFilenames, Glob_line)]
  					Files = [Files, RawFilenames[ind]]
 				endelse
			endif
			Glob_lines = wcnt eq 0	?	[Glob_line] : [Glob_lines, Glob_line]
			wcnt+=1
			eof_lun = EOF(lun)
			;print,eof_lun
		endif
	ENDWHILE
	print,'Found files:',n_elements(Files)
	FREE_LUN, lun
	if n_elements(Files) gt 0  then RawFilenames = Files
endif else begin
	if include_subdir then RawFilenames = FILE_SEARCH(fpath, fmask) else RawFilenames = FILE_SEARCH(fpath + sep + fmask )
endelse

; This is to allow for restart if the Peakselector crashed first time but some *.pks files already esist

MLRawFilenames = RawFilenames	; complete set of RawFileNames (including those for which the .pks had already been created
if Excl_PKS then begin
	for i=0,(n_elements(RawFilenames)-1) do begin
		pks_f = file_info(addextension(RawFilenames[i],'_IDL.pks'))
		if ~pks_f.exists then begin
			if n_elements(RawFiles_new) eq 0 then RawFiles_new = RawFilenames[i] else RawFiles_new = [RawFiles_new, RawFilenames[i]]
		endif
	endfor
	RawFilenames = (n_elements(RawFiles_new) ne 0)	?	RawFiles_new	: ['']
endif

CATCH, /CANCEL

WID_LIST_Extract_mTIFFS_ID = Widget_Info(Event.Top, find_by_uname='WID_LIST_Extract_mTIFFS')
widget_control,WID_LIST_Extract_mTIFFS_ID,SET_VALUE = RawFilenames

nfiles = (RawFilenames ne ['']) ? n_elements(RawFilenames)	:	0

if UseGlobIni_mTIFFs then begin
	nfiles_text = string(n_elements(RawFilenames))+' files (Glob)'
	WID_LABEL_nfiles_Glob_mTIFFs_ID = Widget_Info(Event.Top, find_by_uname='WID_LABEL_nfiles_Glob_mTIFFs')
	widget_control,WID_LABEL_nfiles_Glob_mTIFFs_ID,SET_VALUE=nfiles_text
endif else begin
	nfiles_text = string(n_elements(RawFilenames))+' files (total)'
	WID_LABEL_nfiles_mTIFFs_ID = Widget_Info(Event.Top, find_by_uname='WID_LABEL_nfiles_mTIFFs')
	widget_control,WID_LABEL_nfiles_mTIFFs_ID,SET_VALUE=nfiles_text
endelse

end
;
;-----------------------------------------------------------------
;
pro On_Remove_Selected_mTIFFS, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate

CATCH, Error_status
WID_LIST_Extract_mTIFFS_ID = Widget_Info(Event.Top, find_by_uname='WID_LIST_Extract_mTIFFS')
ind_selected = widget_info(WID_LIST_Extract_mTIFFS_ID, /LIST_SELECT)

if n_elements(ind_selected) eq 1 then if ind_selected eq -1 then return
RawFilenames[ind_selected] = '-1'
RawFilenames = RawFilenames[where(RawFilenames ne '-1')]

widget_control,WID_LIST_Extract_mTIFFS_ID,SET_VALUE = RawFilenames
new_sel = max(ind_selected)+1 < n_elements(RawFilenames)-1
widget_control,WID_LIST_Extract_mTIFFS_ID, SET_LIST_SELECT=new_sel

nfiles_text = string(n_elements(RawFilenames))+' files'
WID_LABEL_nfiles_mTIFFs_ID = Widget_Info(Event.Top, find_by_uname='WID_LABEL_nfiles_mTIFFs')
widget_control,WID_LABEL_nfiles_mTIFFs_ID, SET_VALUE=nfiles_text
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
pro On_Read_TIFF_Info_mTIFFS, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
if n_elements(RawFilenames) eq 0 then begin
	z=dialog_message('Select at least 1 TIF file')
	return      ; if data not loaded return
endif

tif_filename = RawFilenames[0]
test0=file_info(tif_filename)
test1=QUERY_TIFF(tif_filename, tif_info)

if test1 eq 0 then begin
	z=dialog_message('Incorrect TIF file')
	return      ; if data not loaded return
endif

;LoadThiFitCond,ini_filename,thisfitcond
pos=max(strsplit(tif_filename,sep))
pth= strmid(tif_filename,0,(pos-1))
fname=strmid(tif_filename,strlen(pth))
filen=StripExtension(fname)

thisfitcond.f_info = filen											; filename wo extension
thisfitcond.xsz = tif_info.dimensions[0]							; x-size (pixels)
thisfitcond.ysz = tif_info.dimensions[1]							; y-size (pixels)
thisfitcond.filetype = 1
print,thisfitcond
WID_TABLE_InfoFile_mTIFFS_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_InfoFile_mTIFFS')
Fill_Fitting_Parameters_mTIFFS, WID_TABLE_InfoFile_mTIFFS_ID

end
;
;-----------------------------------------------------------------
;
pro Do_Change_TIFF_Info_mTIFFS, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
widget_control,event.id,get_value=thevalue
value=float(reform(thevalue))
CASE event.y OF
	0:thisfitcond.zerodark=value[event.y]
	1:thisfitcond.xsz=value[event.y]
	2:thisfitcond.ysz=value[event.y]
	3:thisfitcond.Thresholdcriteria=value[event.y]
	4:thisfitcond.filetype=value[event.y]
	5:thisfitcond.LimBotA1=value[event.y]
	6:thisfitcond.LimTopA1=value[event.y]
	7:thisfitcond.LimBotSig=value[event.y]
	8:thisfitcond.LimTopSig=value[event.y]
	9:thisfitcond.LimChiSq=value[event.y]
	10:thisfitcond.Cntpere=value[event.y]
	11:thisfitcond.maxcnt1=value[event.y]
	12:thisfitcond.maxcnt2=value[event.y]
	13:thisfitcond.fliphor=value[event.y]
	14:thisfitcond.flipvert=value[event.y]
	15:thisfitcond.MaskSize=value[event.y]
	16:thisfitcond.GaussSig=value[event.y]
	17:thisfitcond.MaxBlck=value[event.y]
	18:thisfitcond.SparseOversampling=value[event.y]
	19:thisfitcond.SparseLambda=value[event.y]
	20:thisfitcond.SparseDelta=value[event.y]
	21:thisfitcond.SpError=value[event.y]
	22:thisfitcond.SpMaxIter=value[event.y]
ENDCASE
widget_control,event.id,set_value=transpose(value),use_table_select=[0,0,0,(n_elements(value)-1)]
end
;
;-----------------------------------------------------------------
;
pro Do_Change_Astig_Macroparams_mTIFFS, Event
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
widget_control,event.id,get_value=thevalue
value=float(reform(thevalue))
Astig_MacroParameters[event.y]=value[event.y]
widget_control,event.id,set_value=transpose(value),use_table_select=[0,0,0,(n_elements(value)-1)]
print,'New Asig Macroparameters: ', Astig_MacroParameters
end
;
;-----------------------------------------------------------------
;
pro Fill_Fitting_Parameters_mTIFFS, WidgetID
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
values = [thisfitcond.zerodark,	$
	thisfitcond.xsz,	$
	thisfitcond.ysz,	$
	thisfitcond.Thresholdcriteria,  $
	thisfitcond.filetype,	$
	thisfitcond.LimBotA1,	$
	thisfitcond.LimTopA1,	$
	thisfitcond.LimBotSig,  $
	thisfitcond.LimTopSig,	$
	thisfitcond.LimChiSq,	$
	thisfitcond.Cntpere, 	$
	thisfitcond.maxcnt1, 	$
	thisfitcond.maxcnt2,	$
	thisfitcond.fliphor,	$
	thisfitcond.flipvert, 	$
	thisfitcond.MaskSize,	$
	thisfitcond.GaussSig,	$
	thisfitcond.MaxBlck,	$
	thisfitcond.SparseOversampling,	$
	thisfitcond.SparseLambda,	$
	thisfitcond.SparseDelta,	$
	thisfitcond.SpError,	$
	thisfitcond.SpMaxIter]
widget_control,WidgetID,set_value=transpose(values), use_table_select=[0,0,0,(n_elements(values)-1)]
end
;
;-----------------------------------------------------------------
;
pro Start_mTIFFS_Extract, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
										;use SigmaSym as a flag to indicate xsigma and ysigma are not independent and locked together in the fit
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common Glob, UseGlobIni_mTIFFs, GlobINI_FileName, Glob_lines
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
Common Multiple_PALM_TIFFs, DoPurge_mTIFFs, Purge_RowNames_mTIFFs, Purge_Params_mTIFFs
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
Chi_ind = min(where(RowNames eq 'ChiSquared'))							; CGroupParametersGP[7,*] - Chi Squared
FitOK_ind = min(where(RowNames eq 'FitOK'))								; CGroupParametersGP[8,*] - Original FitOK
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))					; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigAmp_ind = min(where(RowNames eq 'Sigma Amplitude'))
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))				; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))				; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity

DisplayType = TransformEngine ? 3 : 1

DisplayType_nonVM = (DisplayType eq 3)
IF LMGR(/VM) and DisplayType_nonVM then begin	; Cannot run this Macro if  IDL is in Virtual Machine Mode
	z=dialog_message('Cannot run this procedure with IDL in Virtual Machine Mode')
	return      ; if data not loaded return
endif

if (n_elements(RawFilenames) eq 0) then begin
	z=dialog_message('Select TIF Files to process')
	return      ; if data not loaded return
endif

if (thisfitcond.xsz eq 0) or (thisfitcond.ysz eq 0) then begin
	z=dialog_message('Frame size is 0. Read TIF File info or define the frame size manually')
	return      ; if data not loaded return
endif

if (n_elements(filen) eq 0) or (n_elements(pth) eq 0) then begin
	z=dialog_message('Read TIF File info to load the file data')
	return      ; if data not loaded return
endif

;WID_DROPLIST_SetSigmaFitSym_mTIFFS_ID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_SetSigmaFitSym_mTIFFS')
;SigmaSym=widget_info(WID_DROPLIST_SetSigmaFitSym_mTIFFS_ID,/DropList_Select)		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
;thisfitcond.SigmaSym = SigmaSym

print,'start of ReadRawLoop6,  thisfitcond=',thisfitcond
print,'Path: ',pth
print,'First file name: ',filen

Start_Time = SYSTIME(/SECONDS)
thefile_no_exten=pth+filen
DisplaySet=0
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz													;number of y pixels

MaxBlckSize=thisfitcond.MaxBlck+thisfitcond.MaskSize*2
Tiling = ((xsz ge MaxBlckSize) or  (ysz ge MaxBlckSize))			; check if every frame needs to be split into tiles and processed separately.

wxsz=1024 & wysz=1024
dsz=xsz>ysz
mgw=(wxsz<wysz)/dsz
if mgw eq 0 then mgw=float(wxsz<wysz)/dsz
mg_scl=2L		;	size reduction for frame display
scl=4.			; 	brightness increase for frame display ;intensity scaling Range = scl* # electrons
print,'DisplayType',DisplayType
;nloops = n_elements(RawFilenames)			;nloops=long((framelast-framefirst)/increment)
nloops = (RawFilenames ne ['']) ? n_elements(RawFilenames)	:	0
print,'nloops=',nloops

if DisplayType eq 3 then begin 	;set to 3 (--> -1) - Cluster
	DisplayType=-1			;turns of all displays during processing
	;if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR') else idl_pwd=pref_get('IDL_WDE_START_DIR')
	cd,current=curr_pwd
	td = 'temp' + strtrim(ulong(SYSTIME(/seconds)),2)
	temp_dir=curr_pwd + sep + td
	FILE_MKDIR,temp_dir
	save, curr_pwd, idl_pwd, temp_dir, pth, filen, ini_filename, thisfitcond, aa, RawFilenames, nloops, Astig_MacroParameters, $
		GlobINI_FileName, UseGlobIni_mTIFFs, DoPurge_mTIFFs, Purge_RowNames_mTIFFs, Purge_Params_mTIFFs, filename=td + sep + 'temp.sav'		;save variables for cluster cpu access
	ReadRawLoopCluster_mTIFFs, Event
	file_delete,td + sep + 'temp.sav'
	file_delete,td
	cd,curr_pwd
endif else begin
		for nlps=0L,nloops-1 do begin											;loop for all file chunks
			thefile_no_exten = StripExtension(RawFilenames[Nlps])
			framefirst=0
			test1=QUERY_TIFF(RawFilenames[nlps], tif_info)
			Nframes = tif_info.num_images
			print,'Started processing file:  ', RawFilenames[nlps]
			data=float(ReadData(thefile_no_exten,thisfitcond,framefirst,Nframes))	;Reads thefile and returns data (bunch of frames) in (units of photons)
			print,'Read the data'

			if DisplayType eq 2 then Showframes, data,xsz,ysz,mgw, Nframes,scl						;Shows time movie of data

			if Nframes gt 1 then totdat=float(total(data[*,*,0:Nframes-1],3)/Nframes) else totdat=float(data)
			if DisplayType ge 1 then begin
				if mgw ge 1 then	totaldata=rebin(totdat*Nframes,xsz*mgw,ysz*mgw,/sample)$
				else totaldata=congrid(totdat*Nframes,round(xsz*mgw),round(ysz*mgw))
				ShowIt,totaldata,mag=mgw,wait=1.0
			endif

			;Get parameters of one bunch of frames
			if thisfitcond.LocalizationMethod eq 0 then begin
				if DisplayType lt 4 then begin
					;Apeakparams=ParamsofShortStackofFrames(data,DisplayType,thisfitcond,framefirst)
					if thisfitcond.SigmaSym le 1 then Apeakparams = ParamsofShortStackofFrames(data,DisplayType,thisfitcond,framefirst)
					if thisfitcond.SigmaSym ge 2 then Apeakparams = ParamsofShortStackofFrames_AstigmaticZ(data,DisplayType,thisfitcond,aa,framefirst)
				endif
				if DisplayType eq 4 then begin
					print,'loaded data block',nlps+1,'
					t0 = SYSTIME(/SECONDS)
					Apeakparams=ParamsofLongStackofFrames(data,DisplayType,thisfitcond,framefirst)
				endif
			endif else if thisfitcond.LocalizationMethod eq 1 then begin
				Apeakparams=ParamsofShortStackofFrames_SparseSampling(data,DisplayType,thisfitcond,framefirst)
			endif

			;Apeakparams[*].frameindex+=framefirst						;adjust frame index to include batch offset
			; no need to adjust, it is adjusted inside the ParamsofLongStackofFrames

			loc=fltarr(xsz*mgw/mg_scl,ysz*mgw/mg_scl)
			filter=((Apeakparams.fitok eq 1) or (Apeakparams.fitok eq 2))
			loc[[mgw*Apeakparams.peakx],[mgw*Apeakparams.peaky]]=255*filter
			image=float(loc)

			If DisplayType ge 1 then begin
				ShowIt,totaldata,mag=mgw,wait=1.0
				ShowIt,loc,mag=mgw,wait=1.0
			endif

			;--------------------------
			if nlps eq 0 then begin
				tot_fr = Nframes
				Apeakparams_tot = Apeakparams
			endif else begin
				ApeakParams.FrameIndex += tot_fr
				Apeakparams_tot = [Apeakparams_tot, Apeakparams]
				tot_fr += Nframes
				totdat=Ntotdat/tot_fr*Nframes + totdat/tot_fr*(tot_fr-Nframes)
				image=Nimage/tot_fr*Nframes + image/tot_fr*(tot_fr-Nframes)
				save,Apeakparams,image,xsz,ysz,totdat,filename=thefile_no_exten+'_'+strtrim(thisfitcond.Frm0,2)+'-'+strtrim(framelast,2)+'_IDL.pks',thefile_no_exten
				file_delete,thefile_no_exten+'_'+strtrim(thisfitcond.Frm0,2)+'-'+strtrim(framefirst-1,2)+'_IDL.pks'
			endelse
			saved_pks_filename=thefile_no_exten+'_'+strtrim(thisfitcond.Frm0,2)+'-'+strtrim(framelast,2)+'_IDL.pks'
			print,'Wrote file '+saved_pks_filename
			if framelast gt thisfitcond.FrmN then return
		endfor

	xydsz=[xsz,ysz]
	sz=size(Apeakparams)
	CGroupParams=fltarr(CGrpSize,sz[1])
	CGroupParams[Off_ind:Amp_ind,*]=ApeakParams.A[0:1]
	CGroupParams[X_ind,*]=ApeakParams.peakx
	CGroupParams[Y_ind,*]=ApeakParams.peaky
	CGroupParams[Xwid_ind:Ywid_ind,*]=ApeakParams.A[2:3]
	CGroupParams[Nph_ind,*]=ApeakParams.NPhot
	CGroupParams[Chi_ind,*]=ApeakParams.ChiSq
	CGroupParams[FitOK_ind,*]=ApeakParams.FitOK
	CGroupParams[FrNum_ind,*]=ApeakParams.FrameIndex
	if PkInd_ind gt 0 then CGroupParams[PkInd_ind,*]=ApeakParams.PeakIndex
	if PkGlInd_ind gt 0 then CGroupParams[PkGlInd_ind,*]=dindgen(sz[1])
	if Par12_ind gt 0 then CGroupParams[Par12_ind,*]=ApeakParams.A[2]*ApeakParams.A[3]
	if SigAmp_ind gt 0 then CGroupParams[SigAmp_ind,*]=ApeakParams.Sigma2[1]
	if SigNphX_ind gt 0 then CGroupParams[SigNphX_ind:SigNphY_ind,*]=ApeakParams.Sigma2[4:5]
	if SigX_ind gt 0 then CGroupParams[SigX_ind:SigY_ind,*]=ApeakParams.Sigma2[2:3]
	if Ell_ind gt 0 then CGroupParams[Ell_ind,*]=(ApeakParams.A[2]-ApeakParams.A[3])/(ApeakParams.A[2]+ApeakParams.A[3])

	TotalRawData = totdat
	FlipRotate=replicate({frt,present:0B,transp:0B,flip_h:0B,flip_v:0B},3)
	NFrames=thisfitcond.Nframesmax    ;   long64(max(CGroupParams[9,*]))

	GuideStarDrift=replicate({present:0B,xdrift:dblarr(Nframes),ydrift:dblarr(Nframes),zdrift:dblarr(Nframes)},3)
	FiducialCoeff=replicate({fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)},3)

	for i=0,(n_elements(RawFilenames)-1) do RawFilenames[i] = StripExtension(RawFilenames[i])
endelse


print,'PALM Peak Extraction: total time (sec)', (SYSTIME(/SECONDS)-Start_Time)
widget_control,event.top,/destroy

if n_elements(CGroupParams) gt 10 then begin
ReloadParamlists, Event1

wlabel = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,wlabel,set_value = saved_pks_filename
OnUnZoomButton, Event1
peak_index=0L
ReloadPeakColumn,peak_index
endif

end
;
;------------------------------------------------------------------------------------
;
pro Reassemble_PKS_Files, Event, SumFilename, npks_det
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common Zdisplay, Z_scale_multiplier, vbar_top
common Offset, PkWidth_offset
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
Common Multiple_PALM_TIFFs, DoPurge_mTIFFs, Purge_RowNames_mTIFFs, Purge_Params_mTIFFs
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames

WID_BUTTON_Excl_PKS_ID = Widget_Info(Event.Top, find_by_uname='WID_BUTTON_Excl_PKS')
Excl_PKS = Widget_Info(WID_BUTTON_Excl_PKS_ID, /BUTTON_SET)

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
Chi_ind = min(where(RowNames eq 'ChiSquared'))							; CGroupParametersGP[7,*] - Chi Squared
FitOK_ind = min(where(RowNames eq 'FitOK'))								; CGroupParametersGP[8,*] - Original FitOK
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))				; CGroupParametersGP[10,*]
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))				; CGroupParametersGP[11,*]
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))					; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigAmp_ind = min(where(RowNames eq 'Sigma Amplitude'))					; CGroupParametersGP[13,*]
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))				; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))				; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
AmpL1_ind = min(where(RowNames eq 'Amplitude L1'))						; CGroupParametersGP[27,*] - Label 1 Amplitude
AmpL2_ind = min(where(RowNames eq 'Amplitude L2'))						; CGroupParametersGP[28,*] - Label 2 Amplitude
AmpL3_ind = min(where(RowNames eq 'Amplitude L3'))						; CGroupParametersGP[29,*] - Label 3 Amplitude
SigL1_ind = min(where(RowNames eq 'Zeta0'))								; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))						; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))						; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))							; CGroupParametersGP[35,*] - Sigma Z
Coh_ind = min(where(RowNames eq '36 Coherence'))						; CGroupParametersGP[36,*] - Coherence
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

xi = 0ul
npktot = total(ulong(npks_det), /PRESERVE_TYPE)
print,'Total Peaks Detected: ',npktot,'    for data set: ', SumFilename

nloops = n_elements(RawFilenames)

	for nlps=0,nloops-1 do begin			;reassemble little pks files from all the workers into on big one
		print,'PALM cluster processing:  concatenating the segment',(nlps+1),'   of ',nloops
		current_file = 	AddExtension(RawFilenames[nlps],'_IDL.pks')
		test1=file_info(current_file)
		if ~test1.exists then begin
			print, 'file does not exist:',current_file
		endif else begin
			restore,filename = current_file
			if (size(Apeakparams))[2] ne 0 then begin
				if ((size(Apeakparams_tot))[2] eq 0) and (npks_det[nlps] gt 0) then begin
					Apeakparams_tot = replicate(Apeakparams[0],npktot)
					xa = xi + npks_det[nlps]-1uL
					Apeakparams_tot[xi:xa] = Apeakparams
					xi = xa + 1ul
					totdat_tot = totdat
					image_tot = image
					tot_fr = max(Apeakparams.frameindex)+1ul
				endif else begin
					if npks_det[nlps] gt 0 then begin
						Nframes=max(Apeakparams.frameindex)+1ul
						ApeakParams.FrameIndex += tot_fr
						xa = xi + npks_det[nlps]-1uL
						Apeakparams_tot[xi] = Apeakparams	; fast
						xi = xa + 1ul
						tot_fr += Nframes
						totdat_tot=totdat_tot/tot_fr*(tot_fr-Nframes) + totdat/tot_fr*Nframes
						image_tot=image_tot/tot_fr*(tot_fr-Nframes)+image/tot_fr*Nframes
					endif
				endelse
			endif
			;file_delete, current_file, /QUIET
		endelse
	endfor

	if n_elements(Apeakparams_tot) ge 1 then begin
		Apeakparams = Apeakparams_tot
		xydsz=[xsz,ysz]
		image=image_tot
		ind_good = where((ApeakParams.NPhot gt 50) and (ApeakParams.FitOK eq 1), sz)
		CGroupParams=fltarr(CGrpSize,sz)

		CGroupParams[Off_ind:Amp_ind,*]=ApeakParams[ind_good].A[0:1]
		CGroupParams[X_ind,*]=ApeakParams[ind_good].peakx
		CGroupParams[Y_ind,*]=ApeakParams[ind_good].peaky
		if Chi_ind gt 0 then CGroupParams[Chi_ind,*]=ApeakParams[ind_good].ChiSq

		if thisfitcond.SigmaSym le 1 then begin
			CGroupParams[Xwid_ind:Ywid_ind,*]=ApeakParams[ind_good].A[2:3]
			if SigNphX_ind gt 0 then CGroupParams[SigNphX_ind:SigNphY_ind,*]=ApeakParams[ind_good].Sigma2[4:5]
			if Par12_ind gt 0 then CGroupParams[Par12_ind,*]=ApeakParams[ind_good].A[2]*ApeakParams[ind_good].A[3]
			if SigAmp_ind gt 0 then CGroupParams[SigAmp_ind,*]=ApeakParams[ind_good].Sigma2[1]
			if SigX_ind gt 0 then CGroupParams[SigX_ind:SigY_ind,*]=ApeakParams[ind_good].Sigma2[2:3]
			if Ell_ind gt 0 then CGroupParams[Ell_ind,*]=(ApeakParams[ind_good].A[2]-ApeakParams[ind_good].A[3])/(ApeakParams[ind_good].A[2]+ApeakParams[ind_good].A[3])
		endif else begin
			CGroupParams[Xwid_ind,*] = ApeakParams[ind_good].peak_widx
			CGroupParams[Ywid_ind,*] = ApeakParams[ind_good].peak_widy
			if Z_ind gt 0 then CGroupParams[Z_ind,*]=ApeakParams[ind_good].A[4]
			if SigX_ind gt 0 then CGroupParams[SigX_ind:SigY_ind,*]=ApeakParams[ind_good].Sigma[2:3]
			if SigZ_ind gt 0 then CGroupParams[SigZ_ind,*]=ApeakParams[ind_good].Sigma[4]
			if Ell_ind gt 0 then CGroupParams[Ell_ind,*]=(ApeakParams[ind_good].peak_widx-ApeakParams[ind_good].peak_widy)/(ApeakParams[ind_good].peak_widx+ApeakParams[ind_good].peak_widy)
			if thisfitcond.SigmaSym eq 4 then if Par12_ind gt 0 then CGroupParams[Par12_ind,*]=ApeakParams[ind_good].A[5]
		endelse
		CGroupParams[Nph_ind,*]=ApeakParams[ind_good].NPhot
		CGroupParams[FitOK_ind,*]=ApeakParams[ind_good].FitOK
		CGroupParams[FrNum_ind,*]=ApeakParams[ind_good].FrameIndex
		if PkInd_ind gt 0 then CGroupParams[PkInd_ind,*]=ApeakParams[ind_good].PeakIndex
		if PkGlInd_ind gt 0 then CGroupParams[PkGlInd_ind,*]=dindgen(sz[1])

		;	purge the data sets before saving them
		param_num = n_elements(Purge_RowNames_mTIFFs)
		if DoPurge_mTIFFs and (param_num gt 0)then begin
			;Purge_RowNames_mTIFFs, Purge_Params_mTIFFs
			params = intarr(param_num)
			for i = 0, param_num-1 do params[i] = min(where(RowNames eq Purge_RowNames_mTIFFs[i]))
            CGPsz = size(CGroupParams)
            low  = Purge_Params_mTIFFs[*,0]#replicate(1,CGPsz[2])
            high = Purge_Params_mTIFFs[*,1]#replicate(1,CGPsz[2])
            filter = (CGroupParams[params,*] ge low) and (CGroupParams[params,*] le high)
            indecis = where (floor(total(temporary(filter), 1) / n_elements(params)) gt 0)
			CGroupParams = CGroupParams[*, indecis]
		endif

		TotalRawData = totdat_tot
		FlipRotate={frt,present:0B,transp:0B,flip_h:0B,flip_v:0B}
		NFrames=long64(max(CGroupParams[FrNum_ind,*]))+1
		GuideStarDrift={present:0B,xdrift:dblarr(Nframes),ydrift:dblarr(Nframes),zdrift:dblarr(Nframes)}
		FiducialCoeff={fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)}
		for i=0,(n_elements(RawFilenames)-1) do RawFilenames[i] = StripExtension(RawFilenames[i])

		save, CGroupParams, CGrpSize, ParamLimits, Image, xydsz, TotalRawData, DIC, RawFilenames,$
			GuideStarDrift, FiducialCoeff, FlipRotate, thisfitcond, RowNames, $
			lambda_vac,nd_water, nd_oil, nmperframe, wind_range, z_unwrap_coeff, ellipticity_slopes, $
			nm_per_pixel, wfilename, PkWidth_offset, Z_scale_multiplier, grouping_gap, grouping_radius100, filename=SumFilename

		print,'Wrote result file '+ SumFilename
		print,''

	endif else print,'No peaks detected, did not save anything for the data set ' + SumFilename
	print,'Started deleting original PKS files'
		for nlps=0,nloops-1 do file_delete, current_file, /QUIET
	print,'Finished deleting original PKS files. Done'
end
;
;------------------------------------------------------------------------------------
;
Pro ReadRawLoopCluster_mTIFFs, Event			;Master program to read data and loop through processing for cluster
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Glob, UseGlobIni_mTIFFs, GlobINI_FileName, Glob_lines
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
Common Multiple_PALM_TIFFs, DoPurge_mTIFFs, Purge_RowNames_mTIFFs, Purge_Params_mTIFFs
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

WID_BUTTON_Excl_PKS_ID = Widget_Info(Event.Top, find_by_uname='WID_BUTTON_Excl_PKS')
Excl_PKS = Widget_Info(WID_BUTTON_Excl_PKS_ID, /BUTTON_SET)

restore,(temp_dir+'/temp.sav')

if nloops gt 0 then begin
	print,'sh '+idl_pwd+'/runme_mTIFFs.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd+' '+temp_dir		;Spawn workers in cluster
	spawn,'sh '+idl_pwd+'/runme_mTIFFs.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd+' '+temp_dir		;Spawn workers in cluster
endif

thefile_no_exten=pth+filen

; We previously stored all filenames into MLRawFilenames array
; and the filenames without corresponding .pks processed files into RawFilenames
; now we just need to re-assemble all MLRawFilenames.
print,'We previously stored all filenames into MLRawFilenames array'
print,'and the filenames without corresponding .pks processed files into RawFilenames'
print,'now we just need to re-assemble all MLRawFilenames.'
if RawFilenames ne [''] then print,'n_elements(RawFilenames):',n_elements(RawFilenames) else print,'n_elements(RawFilenames): 0'
print,'n_elements(MLRawFilenames):',n_elements(MLRawFilenames)
RawFilenames = MLRawFilenames
nloops = n_elements(RawFilenames)
if Excl_PKS then begin ; process was previously interrupted and npks_det data DEOS NOT exist, need to create it
	print,'process was previously interrupted and npks_det data DEOS NOT exist, need to create it'
	npks_det = ulonarr(nloops)

	oStatusBar = obj_new('PALM_StatusBar', $	;********* Status Bar Initialization  ******************
						COLOR=[0,0,255], $
						TEXT_COLOR=[255,255,255],  $
						TITLE='Re-setting npks_det...', $
						TOP_LEVEL_BASE=tlb)
	fraction_complete_last=0.0
	pr_bar_inc=0.01

	for nlps=0,nloops-1 do begin			;reassemble little pks files from all the workers into on big one
		;print,'Finding the peak number for the segment',(nlps+1),'   of ',nloops
		current_file = 	AddExtension(RawFilenames[nlps],'_IDL.pks')
		test1=file_info(current_file)
		if ~test1.exists then begin
			npks_det[nlps] = 0
			print, 'file does not exist:',current_file
		endif else begin
			restore,filename = current_file
			npks_det[nlps] = n_elements(Apeakparams)
		endelse
		fraction_complete = float(nlps+1)/float(nloops)
		if (fraction_complete - fraction_complete_last) ge pr_bar_inc then begin ; *********** Status Bar Update **********
 			oStatusBar -> UpdateStatus, fraction_complete
 			fraction_complete_last = fraction_complete
 		endif
	endfor
	obj_destroy, oStatusBar ;********* Status Bar Close ******************
endif else restore,(temp_dir+'/npks_det.sav')

file_delete,(temp_dir+'/npks_det.sav'), /QUIET

if UseGlobIni_mTIFFs then begin
	print,'Glob_lines:', Glob_lines
	test = where(strmatch(RawFilenames, Glob_lines[0]) eq 1)
endif
print, 'n_files(Slab0):',n_elements(test)
print, 'n_elements(npks_det):',n_elements(npks_det)
print, 'total(npks_det):',total(npks_det)
print, 'n_elements(RawFilenames):',n_elements(RawFilenames)


if NOT UseGlobIni_mTIFFs then begin
; stnadard procedure - no globs
	SumFilename = pth+'/PALM_data_IDL.sav'
	RawFilenames_subset = RawFilenames
	Reassemble_PKS_Files, Event, SumFilename, npks_det
endif else begin
; if using globs
	npks_det_tot = npks_det
	RawFilenames_tot = RawFilenames
	for glob_id =0, n_elements(Glob_lines)-1 do begin
		SumFilename = AddExtension(GlobINI_FileName, ('_'+strtrim(glob_id,2)+'_IDL.sav'))
		indices_loc = where(strmatch(RawFilenames_tot, Glob_lines[glob_id]) eq 1)
		RawFilenames = RawFilenames_tot[indices_loc]
		npks_det = npks_det_tot[indices_loc]
		Reassemble_PKS_Files, Event, SumFilename, npks_det
	endfor
	RawFilenames = RawFilenames_tot
endelse
wlabel = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,wlabel,set_value=SumFilename
end
;
;------------------------------------------------------------------------------------
;
Pro	ReadRawWorker_mTIFFs,nlps,data_dir,temp_dir						;spawn mulitple copies of this programs for cluster
Nlps=ulong((COMMAND_LINE_ARGS())[0])
data_dir=(COMMAND_LINE_ARGS())[1]
temp_dir=(COMMAND_LINE_ARGS())[2]
cd,data_dir
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
    PRINT, 'ReadRawWorker_mTIFFs Error index: ', Error_status
    PRINT, 'ReadRawWorker_mTIFFs Error message: ', !ERROR_STATE.MSG
	CATCH, /CANCEL
	return
ENDIF
restore,temp_dir + '/temp.sav'
thefile_no_exten = StripExtension(RawFilenames[Nlps])
DisplayType=-1													;set to no displays
xsz=thisfitcond.xsz												;number of x pixels
ysz=thisfitcond.ysz												;number of y pixels

framefirst=0
test1=QUERY_TIFF(RawFilenames[Nlps], tif_info)
Nframes = tif_info.num_images
print,'Started processing file:  ', RawFilenames[Nlps]
data=float(ReadData(thefile_no_exten,thisfitcond,framefirst,Nframes))	;Reads thefile and returns data (bunch of frames) in (units of photons)

print,'Read the data'
wxsz=1024 & wysz=1024
dsz=xsz>ysz
mg=((wxsz<wysz))/dsz

if Nframes gt 1 then totdat=float(total(data[*,*,0L:Nframes-1L],3)/Nframes) else totdat=float(data)
;if thisfitcond.LocalizationMethod eq 0 then Apeakparams = ParamsofShortStackofFrames(data,DisplayType,thisfitcond,framefirst)
if thisfitcond.LocalizationMethod eq 0 then begin
	if thisfitcond.SigmaSym le 1 then Apeakparams = ParamsofShortStackofFrames(data,DisplayType,thisfitcond,framefirst)
	if thisfitcond.SigmaSym ge 2 then Apeakparams = ParamsofShortStackofFrames_AstigmaticZ(data,DisplayType,thisfitcond,aa,framefirst)
endif
if thisfitcond.LocalizationMethod eq 1 then Apeakparams = ParamsofShortStackofFrames_SparseSampling(data,DisplayType,thisfitcond,framefirst)
print,'ReadRawWorker: Finished ParamsofShortStackofFrames'
mg=2
loc=fltarr(xsz*mg,ysz*mg)
if n_elements(Apeakparams) gt 0 then begin
;	filter = (Apeakparams.fitok eq 1) $
;		AND (ApeakParams.NPhot ge Astig_MacroParameters[0])	$
;		AND (ApeakParams.Sigma2[4] le Astig_MacroParameters[1]) $
;		AND (ApeakParams.Sigma2[5] le Astig_MacroParameters[2])
	filter=((Apeakparams.fitok eq 1) and (Apeakparams.NPhot ge 50))
	loc[[mg*Apeakparams.peakx],[mg*Apeakparams.peaky]]=255*filter
	image=float(loc)
	Apeakparams = Apeakparams(where(filter))
	npks_det = total(ulong(filter),/PRESERVE_TYPE)
endif else npks_det = 0uL
image=float(loc)
current_file = 	AddExtension(RawFilenames[nlps],'_IDL.pks')
save,Apeakparams,image,xsz,ysz,totdat,filename=current_file
spawn,'sync'
spawn,'sync'
print,'Wrote file '+ current_file +'   Peaks Detected:'+strtrim(npks_det,2)
return
end
;
;-----------------------------------------------------------------
;