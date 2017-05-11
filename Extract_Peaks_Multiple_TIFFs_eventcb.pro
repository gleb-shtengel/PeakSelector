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

cd,current = pth

WID_TXT_mTIFFS_Directory_ID = Widget_Info(wWidget, find_by_uname='WID_TXT_mTIFFS_Directory')
widget_control,WID_TXT_mTIFFS_Directory_ID,SET_VALUE = pth

wtable_id = Widget_Info(wWidget, find_by_uname='WID_TABLE_InfoFile_mTIFFS')
if !VERSION.OS_family eq 'unix' then widget_control,wtable_id,COLUMN_WIDTH=[200,100],use_table_select = [ -1, 0, 0, 22 ]
wtable2_id = Widget_Info(wWidget, find_by_uname='WID_Filter_Parameters_mTIFFS')
if !VERSION.OS_family eq 'unix' then widget_control,wtable2_id,COLUMN_WIDTH=[200,100],use_table_select = [ -1, 0, 0, 2 ]

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

WID_Filter_Parameters_mTIFFS_ID = Widget_Info(wWidget, find_by_uname='WID_Filter_Parameters_mTIFFS')
widget_control,WID_Filter_Parameters_mTIFFS_ID,set_value=transpose(Astig_MacroParameters), use_table_select=[0,0,0,(n_elements(Astig_MacroParameters)-1)]

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
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
fpath = Dialog_Pickfile(/read,/DIRECTORY)
WID_TXT_mTIFFS_Directory_ID = Widget_Info(Event.Top, find_by_uname='WID_TXT_mTIFFS_Directory')
widget_control,WID_TXT_mTIFFS_Directory_ID,SET_VALUE = fpath
if fpath ne '' then cd,fpath
end
;
;-----------------------------------------------------------------
;
pro OnPickWINDFile_mTIFFS, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
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
pro On_ReFind_Files_mTIFFS, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

WID_TXT_mTIFFS_Directory_ID = Widget_Info(Event.Top, find_by_uname='WID_TXT_mTIFFS_Directory')
widget_control,WID_TXT_mTIFFS_Directory_ID,GET_VALUE = fpath

WID_BUTTON_Include_Subdirectories_ID = Widget_Info(Event.Top, find_by_uname='WID_BUTTON_Include_Subdirectories_mTIFFs')
include_subdir = Widget_Info(WID_BUTTON_Include_Subdirectories_ID, /BUTTON_SET)

CATCH, Error_status
if include_subdir then RawFilenames = FILE_SEARCH(fpath, "*.tif" ) else RawFilenames = FILE_SEARCH(fpath + sep + "*.tif" )
WID_LIST_Extract_mTIFFS_ID = Widget_Info(Event.Top, find_by_uname='WID_LIST_Extract_mTIFFS')
widget_control,WID_LIST_Extract_mTIFFS_ID,SET_VALUE = RawFilenames

nfiles_text = string(n_elements(RawFilenames))+' files'
WID_LABEL_nfiles_mTIFFs_ID = Widget_Info(Event.Top, find_by_uname='WID_LABEL_nfiles_mTIFFs')
widget_control,WID_LABEL_nfiles_mTIFFs_ID,SET_VALUE=nfiles_text
;print,RawFilenames
IF Error_status NE 0 THEN BEGIN
	print,text, ' cannot be loaded, or has incorrect structure'
	PRINT, 'Error index: ', Error_status
	PRINT, 'Error message: ', !ERROR_STATE.MSG
	CATCH, /CANCEL
	return
ENDIF
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

LoadThiFitCond,ini_filename,thisfitcond
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
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }

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


print,'start of ReadRawLoop6,  thisfitcond=',thisfitcond
print,'Path: ',pth
print,'First file name: ',filen

Start_Time= SYSTIME(/SECONDS)
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
nloops = n_elements(RawFilenames)			;nloops=long((framelast-framefirst)/increment)
print,'nloops=',nloops

if DisplayType eq 3 then begin 	;set to 3 (--> -1) - Cluster
	DisplayType=-1			;turns of all displays during processing
	;if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR') else idl_pwd=pref_get('IDL_WDE_START_DIR')
	cd,current=curr_pwd
	temp_dir=curr_pwd+'/temp'
	FILE_MKDIR,temp_dir
	save, curr_pwd, idl_pwd, temp_dir, pth, filen, ini_filename, thisfitcond, RawFilenames, nloops, Astig_MacroParameters, filename='temp/temp.sav'		;save variables for cluster cpu access
	ReadRawLoopCluster_mTIFFs
	file_delete,'temp/temp.sav'
	file_delete,'temp'
	cd,curr_pwd
	restore,saved_pks_filename
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
				if DisplayType lt 4 then Apeakparams=ParamsofShortStackofFrames(data,DisplayType,thisfitcond,framefirst)
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
	endelse

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

print,'PALM Peak Extraction: total time (sec)', (SYSTIME(/SECONDS)-Start_Time)
widget_control,event.top,/destroy

for i=0,(n_elements(RawFilenames)-1) do RawFilenames[i] = StripExtension(RawFilenames[i])

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
Pro ReadRawLoopCluster_mTIFFs			;Master program to read data and loop through processing for cluster
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
restore,'temp/temp.sav'
print,'sh '+idl_pwd+'/runme_mTIFFs.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd		;Spawn workers in cluster
spawn,'sh '+idl_pwd+'/runme_mTIFFs.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd		;Spawn workers in cluster
thefile_no_exten=pth+filen
restore,'temp/npks_det.sav'
npktot = total(ulong(npks_det), /PRESERVE_TYPE)
print,'Total Peaks Detected: ',npktot
file_delete,'temp/npks_det.sav'
xi = 0ul
for nlps=0,nloops-1 do begin			;reassemble little pks files from all the workers into on big one
	print,'PALM cluster processing: concatenating the segment',(nlps+1),'   of ',nloops
	current_file = 	AddExtension(RawFilenames[nlps],'_IDL.pks')
	test1=file_info(current_file)
	if ~test1.exists then stop
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
	file_delete, current_file
endfor
if n_elements(Apeakparams_tot) ge 1 then Apeakparams = Apeakparams_tot
totdat=totdat_tot
image=image_tot
saved_pks_filename=pth+'/PALM_data_IDL.pks'
save,Apeakparams,image,xsz,ysz,totdat,thefile_no_exten, filename=saved_pks_filename
print,'Wrote result file '+saved_pks_filename

return
end

;------------------------------------------------------------------------------------
;
Pro	ReadRawWorker_mTIFFs,nlps,data_dir						;spawn mulitple copies of this programs for cluster
Nlps=ulong((COMMAND_LINE_ARGS())[0])
data_dir=(COMMAND_LINE_ARGS())[1]
cd,data_dir
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
    PRINT, 'ReadRawWorker_mTIFFs Error index: ', Error_status
    PRINT, 'ReadRawWorker_mTIFFs Error message: ', !ERROR_STATE.MSG
	CATCH, /CANCEL
	return
ENDIF
restore,'temp/temp.sav'
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
if thisfitcond.LocalizationMethod eq 0 then Apeakparams = ParamsofShortStackofFrames(data,DisplayType,thisfitcond,framefirst)
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

