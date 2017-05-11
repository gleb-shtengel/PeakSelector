;
; Empty stub procedure used for autoloading.
;
pro ExtractMultiLabelWid_eventcb
end
;
;-----------------------------------------------------------------
;
pro StartMLExtract, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
Initialization_PeakSelector, TopID

WidDListSigmaSym = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_SetSigmaFitSym_ML')
thisfitcond.SigmaSym = widget_info(WidDListSigmaSym,/DropList_Select)		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'



File1WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam1Filename')
widget_control,File1WidID,GET_VALUE = text
if strlen(text) eq 0 then begin
	z=dialog_message('Camera 1 Filename cannot be empty')
	return
endif
MLRawFilenames = [text]

File2WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam2Filename')
widget_control,File2WidID,GET_VALUE = text
if strlen(text) ge 1 then MLRawFilenames = [MLRawFilenames,text];

File3WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam3Filename')
widget_control,File3WidID,GET_VALUE = text
if strlen(text) ge 1 then MLRawFilenames = [MLRawFilenames,text];

MLRawFilenames=MLRawFilenames[where(MLRawFilenames ne '',MLcnt)]
print,MLRawFilenames

pos=max(strsplit(MLRawFilenames[0],sep))
fpath=strmid(MLRawFilenames[0],0,pos)
cd,fpath
widget_control,/hourglass

WidDListDispLevel = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_FitDisplayType')
DispType_ML = widget_info(WidDListDispLevel,/DropList_Select)	;set to 0 - min display while fitting 1 - some display 2 - full display  3 - Cluster
TransformEngine = (DispType_ML eq 3) ? 1 : 0

WID_Use_InfoFile_Flip = Widget_Info(Event.Top, find_by_uname='WID_Use_InfoFile_Flip')
Flip_Using_InfoFile=widget_info(WID_Use_InfoFile_Flip,/button_set)
;read sifs extract peaks, and save the *.sav files
for i=0,MLcnt-1 do begin
	pos=max(strsplit(MLRawFilenames[i],sep))
	fpath=strmid(MLRawFilenames[i],0,pos)
	cd,fpath
	pth=fpath
	fname=strmid(MLRawFilenames[i],strlen(fpath))
	fname0=strmid(MLRawFilenames[i],0,strlen(MLRawFilenames[i])-4)
	conf_example='Andor_iXon_SN_1880.ini'
	ini_path=file_which(conf_example,/INCLUDE_CURRENT_DIR)
	if !VERSION.OS_family eq 'unix' then	configpath=pref_get('IDL_MDE_START_DIR')+'/Andor_iXon_ini'	else	configpath=pref_get('IDL_WDE_START_DIR')+'\Andor_iXon_ini'
	if strmatch(fname,'*.sif',/fold_case) then begin
		Read_sif,fpath,fname,configpath,ini_filename, thisfitcond
		fname=fname0+'.txt'
	endif
	if strmatch(fname,'*.tif',/fold_case) then begin
		Read_tif,fpath,fname,configpath,ini_filename, thisfitcond
		fname=fname0+'.txt'
	endif
	if ~strmatch(fname,'*.txt',/fold_case) then return

	ReadThisFitCond, fname, pth, filen, ini_filename, thisfitcond
	pth1=file_which(fpath,strmid(fname,0,strlen(fname)-4)+'.txt')
	if pth1 ne '' then pth=fpath

	DispType = DispType_ML
	ReadRawLoop6,DispType		;goes through data and fits peaks

	print,Flip_Using_InfoFile,thisfitcond.fliphor,thisfitcond.flipvert

	if Flip_Using_InfoFile and thisfitcond.fliphor then OnFlipHorizontal, Event1
	if Flip_Using_InfoFile and thisfitcond.flipvert then OnFlipVertical, Event1

	filename=AddExtension(fname0,'_IDL.sav')
	SavFilenames = filename
	save, CGroupParams, CGrpSize, ParamLimits, RowNames, Image, b_set, xydsz, TotalRawData, RawFilenames, SavFilenames, MLRawFilenames,$
		GuideStarDrift, FiducialCoeff, FlipRotate, thisfitcond, lambda_vac, nd_water, nd_oil, PkWidth_offset, filename=filename

endfor


; displays all labels together
for i=0,MLcnt-1 do begin
	fname0=strmid(MLRawFilenames[i],0,strlen(MLRawFilenames[i])-4)
	filename=AddExtension(fname0,'_IDL.sav')
	if i eq 0 then begin

		if filename eq '' then return
		Restore,filename=filename
		SavFilenames = filename
		TotalRawData0 = TotalRawData
		ReloadParamlists, Event
		NFrames=long64(max(CGroupParams[9,*]))+1
		sz=size(CGroupParams)
		if n_elements(CGroupParams) lt 1 then begin
			z=dialog_message('Invalid data file')
			return      ; if data not loaded return
		endif
		CGrpSize = sz[1]; < 43
		wtable = Widget_Info(Event1.Top, find_by_uname='WID_TABLE_0')
		widget_control,wtable,set_value=transpose(ParamLimits[0:(CGrpSize-1),0:3]), use_table_select=[0,0,3,(CGrpSize-1)]
		widget_control,wtable,set_value=transpose(CGroupParams[0:(CGrpSize-1),sz[2]/2]), use_table_select=[4,0,4,(CGrpSize-1)]
		widget_control, wtable, /editable,/sensitive
		im_sz=size(image)
		if im_sz[0] eq 3 then begin
			wDROPLISTLabel = Widget_Info(Event1.Top, find_by_uname='WID_DROPLIST_Label')
			widget_control,wDROPLISTLabel,set_droplist_select=1				;set to red
		endif
		if im_sz[0] eq 2 then begin
			wDROPLISTLabel = Widget_Info(Event1.Top, find_by_uname='WID_DROPLIST_Label')
			widget_control,wDROPLISTlabel,set_droplist_select=0				;set to null
		endif
		wlabel = Widget_Info(Event1.Top, find_by_uname='WID_LABEL_0')
		widget_control,wlabel,set_value=filename
		peak_index=0L
		ReloadParamlists, Event
		OnUnZoomButton, Event1
		WidFrameNumber = Widget_Info(Event1.Top, find_by_uname='WID_SLIDER_RawFrameNumber')
		WidPeakNumber = Widget_Info(Event1.Top, find_by_uname='WID_SLIDER_RawPeakIndex')
		widget_control,WidFrameNumber,set_value=0
		widget_control,WidPeakNumber,set_value=0
		SetRawSliders,Event1
		; check if the "RawFileName" points to a non-local file
		; if the local file with the same name exists, chasnge RawFileName to point to it
		sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
		for ii=0,n_elements(RawFileNames)-1 do begin
			pos_rawfilename_wind=strpos(RawFileNames[ii],'\',/reverse_search,/reverse_offset)
			pos_rawfilename_unix=strpos(RawFileNames[ii],'/',/reverse_search,/reverse_offset)
			pos_rawfilename=max([pos_rawfilename_wind,pos_rawfilename_unix])
			pos_filename_wind=strpos(filename,'\',/reverse_search,/reverse_offset)
			pos_filename_unix=strpos(filename,'/',/reverse_search,/reverse_offset)
			pos_filename=max([pos_filename_wind,pos_filename_unix])
			if (pos_rawfilename gt 0) and (pos_filename gt 0) then begin
				local_rawfilename=strmid(filename,0,pos_filename)+sep+strmid(RawFileNames[ii],pos_rawfilename+1,strlen(RawFileNames[ii])-pos_rawfilename-1)
				conf_info=file_info(local_rawfilename+'.dat')
				if conf_info.exists then RawFileNames[ii]=local_rawfilename
			endif
		endfor
		;print,'i=',i,'  RawFilenames=',RawFilenames
		existing_ind1=where((RawFilenames ne ''),nls)
		FlipRotate_tot     = FlipRotate[existing_ind1]
		GuideStarDrift_tot = GuideStarDrift[existing_ind1]
		FiducialCoeff_tot  = FiducialCoeff[existing_ind1]
		RawFilenames_tot   = RawFilenames[existing_ind1]
		SavFilenames_tot   = [filename]

	endif else begin
		if n_elements(CGroupParams) eq 0 then return
		P1=ParamLimits
		C1=CGroupParams
		C1sz=size(C1)
		if filename eq '' then return
		cd,fpath
		C1[26,*]=C1[26,*] > 1
		NLabels=C1[26,C1sz[2]-1]			; label of last point

		restore,filename=filename

		; check if the "RawFileName" points to a non-local file
		; if the local file with the same name exists, chasnge RawFileName to point to it
		sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
		for ii=0,n_elements(RawFileNames)-1 do begin
			pos_rawfilename_wind=strpos(RawFileNames[ii],'\',/reverse_search,/reverse_offset)
			pos_rawfilename_unix=strpos(RawFileNames[ii],'/',/reverse_search,/reverse_offset)
			pos_rawfilename=max([pos_rawfilename_wind,pos_rawfilename_unix])
			pos_filename_wind=strpos(filename,'\',/reverse_search,/reverse_offset)
			pos_filename_unix=strpos(filename,'/',/reverse_search,/reverse_offset)
			pos_filename=max([pos_filename_wind,pos_filename_unix])
			if (pos_rawfilename gt 0) and (pos_filename gt 0) then begin
				local_rawfilename=strmid(filename,0,pos_filename)+sep+strmid(RawFileNames[ii],pos_rawfilename+1,strlen(RawFileNames[ii])-pos_rawfilename-1)
				conf_info=file_info(local_rawfilename+'.dat')
				if conf_info.exists then RawFileNames[ii]=local_rawfilename
			endif
		endfor
		n_frames_c1=max(C1[9,*])-min(C1[9,*])+1
		n_frames_cgr=max(CGroupParams[9,*])-min(CGroupParams[9,*])+1
		;print,'i=',i,'  RawFilenames=',RawFilenames
		existing_ind2=where((RawFilenames ne ''),nls)
		FlipRotate_tot     = [FlipRotate_tot, FlipRotate[existing_ind2]]
		GuideStarDrift_tot = [GuideStarDrift_tot, GuideStarDrift[existing_ind2]]
		FiducialCoeff_tot  = [FiducialCoeff_tot, FiducialCoeff[existing_ind2]]
		RawFilenames_tot   = [RawFilenames_tot, RawFilenames[existing_ind2]]
		SavFilenames_tot   = [SavFilenames_tot, filename[existing_ind2]]

		CGroupParams[26,*]=NLabels+1						;write in next label index
		CGroupParams=[[C1],[CgroupParams]]					;append data of next label (fluorescent label)
		sz=size(CGroupParams)
		WidFrameNumber = Widget_Info(Event1.Top, find_by_uname='WID_SLIDER_RawFrameNumber')
		WidPeakNumber = Widget_Info(Event1.Top, find_by_uname='WID_SLIDER_RawPeakIndex')
		widget_control,WidFrameNumber,set_value=0
		widget_control,WidPeakNumber,set_value=0
		SetRawSliders,Event1
		wlabel = Widget_Info(Event1.Top, find_by_uname='WID_LABEL_0')
		widget_control,wlabel,set_value=filename
		ReloadParamlists, Event
		OnUnZoomButton, Event1
	endelse
endfor

FlipRotate      = FlipRotate_tot
GuideStarDrift  = GuideStarDrift_tot
FiducialCoeff   = FiducialCoeff_tot
RawFilenames    = RawFilenames_tot
SavFilenames    = SavFilenames_tot

TotalRawData = TotalRawData0
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnCancelReExtract, Event
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnPickCam1DatFile, Event		; Picks the label 1 (camera 1) fit condition (.sif) file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
filter_to_read = !VERSION.OS_family eq 'unix' ? ['*.txt','*.sif','*.tif'] : ['*.sif','*.txt','*.tif']
text = Dialog_Pickfile(/read,filter=filter_to_read,title='Pick Raw Data #1 File *.sif, *tif, or *.txt')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
File1WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam1Filename')
widget_control,File1WidID,SET_VALUE = text

if strmatch(text,'*.txt',/fold_case) then begin
	ReadThisFitCond, text, pth, filen, ini_filename, thisfitcond
	WidDListSigmaSym = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_SetSigmaFitSym_ML')
	widget_control,WidDListSigmaSym,SET_DROPLIST_SELECT=thisfitcond.SigmaSym		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
endif
end
;
;-----------------------------------------------------------------
;
pro OnPickCam2DatFile, Event		; Picks the label 2 (camera 2) fit condition (.sif) file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
filter_to_read = !VERSION.OS_family eq 'unix' ? ['*.txt','*.sif','*.tif'] : ['*.sif','*.txt','*.tif']
text = Dialog_Pickfile(/read,filter=filter_to_read,title='Pick Raw Data #2 File *.sif, *tif, or *.txt')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
File2WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam2Filename')
widget_control,File2WidID,SET_VALUE = text
end
;
;-----------------------------------------------------------------
;
pro OnPickCam3DatFile, Event		; Picks the label 3 (camera 3) fit condition (.sif) file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
filter_to_read = !VERSION.OS_family eq 'unix' ? ['*.txt','*.sif','*.tif'] : ['*.sif','*.txt','*.tif']
text = Dialog_Pickfile(/read,filter=filter_to_read,title='Pick Raw Data #3 File *.sif, *tif, or *.txt')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
File3WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam3Filename')
widget_control,File3WidID,SET_VALUE = text
end
;
;-----------------------------------------------------------------
;
pro Initialize_ExtractMultiLabel, wWidget
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
	WidDListDispLevel = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_FitDisplayType')
	widget_control,WidDListDispLevel,SET_DROPLIST_SELECT = TransformEngine ? 3 : 1
	WID_Use_InfoFile_Flip = Widget_Info(wWidget, find_by_uname = 'WID_Use_InfoFile_Flip')
	widget_control,WID_Use_InfoFile_Flip,set_button = !VERSION.OS_family eq 'unix' ? 1 : 0
	AnchorPnts=dblarr(6,AnchPnts_MaxNum)
	ZPnts=dblarr(3,AnchPnts_MaxNum)
	WidDListSigmaSym = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_SetSigmaFitSym_ML')
	widget_control,WidDListSigmaSym,SET_DROPLIST_SELECT=thisfitcond.SigmaSym		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
end
;
;-----------------------------------------------------------------
;
pro SetSigmaSym_ML, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
	WidDListSigmaSym = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_SetSigmaFitSym_ML')
	SigmaSym=widget_info(WidDListSigmaSym,/DropList_Select)		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
	thisfitcond.SigmaSym = SigmaSym
   print,'SetSigmaSym',thisfitcond
end
