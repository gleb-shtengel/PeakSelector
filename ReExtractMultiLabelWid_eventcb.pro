;
; IDL Event Callback Procedures
; ReExtractMultiLabelWid_eventcb
;
; Generated on:	11/12/2007 21:37.40
;
;-----------------------------------------------------------------
;

pro StartReExtract, Event		; Starts Re-extracting Peaks for multilabel (3D) processing
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

File1WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam1Filename')
widget_control,File1WidID,GET_VALUE = text
if strlen(text) eq 0 then begin
	z=dialog_message('Camera 1 Filename cannot be empty')
	return
endif

pos = strpos(text,'.',/reverse_search,/reverse_offset)
MLRawFilenames = [strmid(text,0,pos)]

;File1WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam1Filename')
;widget_control,File1WidID,GET_VALUE = text
;MLRawFilenames[0] = strmid(text,0,pos)

File2WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam2Filename')
widget_control,File2WidID,GET_VALUE = text
pos = strpos(text,'.',/reverse_search,/reverse_offset)
;MLRawFilenames[1] = strmid(text,0,pos)
if strlen(text) ge 1 then MLRawFilenames = [MLRawFilenames,strmid(text,0,pos)]

File3WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam3Filename')
widget_control,File3WidID,GET_VALUE = text
pos = strpos(text,'.',/reverse_search,/reverse_offset)
;MLRawFilenames[2] = strmid(text,0,pos)
if strlen(text) ge 1 then MLRawFilenames = [MLRawFilenames,strmid(text,0,pos)]

print,'step1, ',MLRawFileNames

pos=max(strsplit(MLRawFilenames[0],sep))
fpath=strmid(MLRawFilenames[0],0,pos-1)
cd,fpath

widget_control,/hourglass

WidDListDispLevel = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_FitDisplayType')
DispType = widget_info(WidDListDispLevel,/DropList_Select)	;set to 0 - min display while fitting 1 - some display 2 - full display  3 - Cluster

TransformEngine = (DispType eq 3) ? 1 : 0

WidDListSigmaSym = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_SetSigmaFitSym_Reextract')
SigmaSym = widget_info(WidDListSigmaSym,/DropList_Select)		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
thisfitcond.SigmaSym = SigmaSym

ReadRawLoopMultipleLabel,DispType		;goes through data and fits peaks
widget_control,event.top,/destroy

end
;
;-----------------------------------------------------------------
;
pro OnCancelReExtract, Event		; Cancels and closes the menu widget
widget_control,event.top,/destroy
end
;
; Empty stub procedure used for autoloading.
;
pro ReExtractMultiLabelWid_eventcb
end

;
;-----------------------------------------------------------------
;
pro OnPickCam1File, Event		; Picks the label 1 (camera 1) fit condition (.txt) file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
text = Dialog_Pickfile(/read,filter=['*.txt'],title='Pick Raw Data #1 File *.txt')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
File1WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam1Filename')
widget_control,File1WidID,SET_VALUE = text
end
;
;-----------------------------------------------------------------
;
pro OnPickCam2File, Event		; Picks the label 2 (camera 2) fit condition (.txt) file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
text = Dialog_Pickfile(/read,filter=['*.txt'],title='Pick Raw Data #2 File *.txt')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
File2WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam2Filename')
widget_control,File2WidID,SET_VALUE = text
end
;
;-----------------------------------------------------------------
;
pro OnPickCam3File, Event		; Picks the label 3 (camera 3) fit condition (.txt) file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
text = Dialog_Pickfile(/read,filter=['*.txt'],title='Pick Raw Data #3 File *.txt')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
File3WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Cam3Filename')
widget_control,File3WidID,SET_VALUE = text
end
;
;-----------------------------------------------------------------
;
pro SetSigmaSym_Reextract, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
	WidDListSigmaSym = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_SetSigmaFitSym_Reextract')
	SigmaSym=widget_info(WidDListSigmaSym,/DropList_Select)		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
	thisfitcond.SigmaSym = SigmaSym
end
;
;-----------------------------------------------------------------
;
pro Initialize_ReExtractMultiLabel, wWidget				; Initializes filenames and the Engine Selection (local for Windows, cluster for UNIX)
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common transformfilenames, lab_filenames, sum_filename
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster

WidDListDispLevel = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_FitDisplayType')
widget_control,WidDListDispLevel,SET_DROPLIST_SELECT = TransformEngine ? 3 : 1

WidDListSigmaSym = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_SetSigmaFitSym_Reextract')
if (size(thisfitcond))[2] eq 8 then $
widget_control,WidDListSigmaSym, SET_DROPLIST_SELECT=thisfitcond.SigmaSym		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep

if ((size(RawFilenames))[2] le 0) or ((size(lab_filenames))[2] le 0) or ((size(sum_filename))[2] le 0) then return

pos=strpos(sum_filename,'.', /REVERSE_OFFSET, /REVERSE_SEARCH)

if RawFilenames[0] ne strmid(sum_filename,0,pos) then return

if n_elements(lab_filenames) eq 3 then begin
	f1_info=file_info(lab_filenames[0])
	f2_info=file_info(lab_filenames[1])
	f3_info=file_info(lab_filenames[2])
	if f1_info.exists and f2_info.exists and f3_info.exists then begin
		File1WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Cam1Filename')
		pos0=strpos(lab_filenames[0],'.', /REVERSE_OFFSET, /REVERSE_SEARCH)
		text=strmid(lab_filenames[0],0,pos0)+'.txt'
		widget_control,File1WidID,SET_VALUE = text
		File2WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Cam2Filename')
		pos1=strpos(lab_filenames[1],'.', /REVERSE_OFFSET, /REVERSE_SEARCH)
		text=strmid(lab_filenames[1],0,pos1)+'.txt'
		widget_control,File2WidID,SET_VALUE = text
		File3WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Cam3Filename')
		pos2=strpos(lab_filenames[2],'.', /REVERSE_OFFSET, /REVERSE_SEARCH)
		text=strmid(lab_filenames[2],0,pos2)+'.txt'
		widget_control,File3WidID,SET_VALUE = text
	endif
endif

end
;
;
;------------------------------------------------------------------------------------
;
Pro ReadRawLoopMultiLabelCluster			;Master program to read data and loop through processing for cluster
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
restore,'temp/temp.sav'
print,'sh '+idl_pwd+'/multilabel_runme.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd			;Spawn workers in cluster
spawn,'sh '+idl_pwd+'/multilabel_runme.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd		;Spawn workers in cluster
print,'ReadRawLoopMultiLabelCluster cluster jobs have completed'
;spawn,'sync'
;spawn,'sync'
cd,curr_pwd
for nlps=0l,nloops-1 do begin			;reassemble little pks files from all the workers into on big one
	framefirst=	thisfitcond.Frm0 + (nlps)*increment						;first frame in batch
	framelast=((thisfitcond.Frm0 + (nlps+1)*increment-1)<thisfitcond.Nframesmax) < thisfitcond.FrmN
	file_fragment = 'temp/ReExtract_peaks_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.sav'
	file_fragment_info = file_info(file_fragment)
	if file_fragment_info.exists then begin
		restore , filename = file_fragment
		CGroupParams[27:33,firstpeak:lastpeak] = CGroupParamsML
		file_delete , file_fragment
	endif
endfor
file_delete,'npks_det.sav'
return
end
;
;------------------------------------------------------------------------------------
;
Pro	ReadRawMultilabelWorker,nlps,data_dir						;spawn mulitple copies of this programs for cluster
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	PRINT, 'ReadRawMultilabelWorker: Error index: ', Error_status
	PRINT, 'ReadRawMultilabelWorker: Error :',!ERROR_STATE.MSG
	CATCH,/CANCEL
	stop
ENDIF
Nlps=long((COMMAND_LINE_ARGS())[0])
data_dir=(COMMAND_LINE_ARGS())[1]
cd,data_dir
restore,'temp/temp.sav'
thisfitcond=ThisFitConds[nlbls-1]
DisplayType=-1													;set to no displays
DisplaySet=DisplayType
xsz=thisfitcond.xsz												;number of x pixels
ysz=thisfitcond.ysz												;number of y pixels
framefirst=	thisfitcond.Frm0 + (nlps)*increment					;first frame in batch
framelast=((thisfitcond.Frm0 + (nlps+1L)*increment-1L)<thisfitcond.Nframesmax) < thisfitcond.FrmN
Nframes=long(framelast-framefirst+1l)								;number of frames to extract in file
d = thisfitcond.MaskSize		; d=5.								half size of region of a selected peak
																;setup twinkle structure with all the fitted peak information
peakparams = {twinkle,frameindex:0l,peakindex:0l,fitOK:0,peakx:0.0,peaky:0.0,A:fltarr(7),sigma1:fltarr(7),sigma2:fltarr(7),chisq:0.0,Nphot:0l}
peakparams.A=[0.0,1.0,1.2,1.2,d,d,0.]
print,'ReadRawMultilabelWorker: starting the worker process, Frames=',framefirst,framelast
firstpeak = -1
lastpeak = 0
for Label=0,nlbls-1 do begin
	;print, 'ReadRawMultilabelWorker: MLRawFilenames[Label]:', MLRawFilenames[Label]
	;print, 'ReadRawMultilabelWorker: ThisFitConds[Label]:',ThisFitConds[Label]
	data=ReadData(MLRawFilenames[Label],ThisFitConds[Label],framefirst,Nframes)	;Reads thefile and returns data (bunch of frames) in (units of photons)
	print,MLRawFilenames[Label]
	for frameindx=0l,Nframes-1 do begin
		if (frameindx mod 50) eq 0 then print,'Frameindex=',frameindx
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
print,'peaks:',firstpeak,lastpeak
if (firstpeak ne -1) then begin
	CGroupParamsML=CgrouPParams[27:33,firstpeak:lastpeak]
	save,firstpeak,lastpeak,CGroupParamsML,filename='temp/ReExtract_peaks_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.sav'
	;spawn,'sync'
	;spawn,'sync'
	print,'Wrote file temp/ReExtract_peaks_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.sav'
endif else	print,'Wrote file - nothing to write'
return
end

;
;------------------------------------------------------------------------------------
;
Pro ReadRawLoopMultiLabel_Bridge			;
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster

restore,'temp/temp.sav'

print,'Starting IDL bridge worker routines'
;Starting IDL bridge workers
obridge=obj_new("IDL_IDLBridge", output='')
for i=1, nloops-1 do obridge=[obridge, obj_new("IDL_IDLBridge", output='')]
print,'data_dir:	',curr_pwd
print,'IDL_dir:		',IDL_pwd

shmName='Status_rep_ML'
max_len=150
SHMMAP,shmName,/BYTE, Dimension=nloops*max_len, GET_OS_HANDLE=OS_handle_val1
Reports=SHMVAR(shmName)
shmName_data='iPALM_data'
SHMMAP,shmName_data,/FLOAT, Dimension=[CGrpSize,iPALM_data_cnt], GET_OS_HANDLE=OS_handle_val2
CGroupParams_bridge=SHMVAR(shmName_data)
CGroupParams_bridge[0,0]=CGroupParams

for i=0, nloops-1 do begin
	obridge[i]->setvar, 'nlps',i
	obridge[i]->setvar, 'data_dir',curr_pwd
	obridge[i]->setvar, 'IDL_dir',IDL_pwd
	obridge[i]->setvar, 'OS_handle_val1',OS_handle_val1
	obridge[i]->setvar, 'OS_handle_val2',OS_handle_val2
	print,'bridge ',i,'  set variables'
	obridge[i]->execute,'cd, IDL_dir'
	print,'bridge ',i,'  changed directory'
	obridge[i]->execute,"restore,'ReadRawLoopMultiLabel_Bridge_Worker.sav'"
	obridge[i]->execute,'ReadRawLoopMultiLabel_Bridge_Worker,nlps,data_dir,OS_handle_val1,OS_handle_val2',/NOWAIT
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

CGroupParams = CGroupParams_bridge

SHMUnmap, shmName
SHMUnmap, shmName_data
for nlps=0L,nloops-1 do	obj_destroy, obridge[nlps]

cd,curr_pwd
return

end
;
;------------------------------------------------------------------------------------
;
Pro	ReadRawLoopMultiLabel_Bridge_Worker,nlps,data_dir,OS_handle_val1,OS_handle_val2						;spawn mulitple copies of this programs for cluster
Error_status=0
cd,data_dir
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
    rep='ReadRawLoopMultiLabel_Bridge_Worker Error: '+!ERROR_STATE.msg
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	CATCH, /CANCEL
	close,1
	return
ENDIF

restore,'temp/temp.sav'

shmName='Status_rep_ML'
max_len=150
SHMMAP,shmName,/BYTE, Dimension=nloops*max_len,OS_Handle=OS_handle_val1
Reports=SHMVAR(shmName)
rep_i=nlps*max_len

shmName_data='iPALM_data'
SHMMAP,shmName_data,/FLOAT, Dimension=[CGrpSize,iPALM_data_cnt],OS_Handle=OS_handle_val2
CGroupParams_bridge=SHMVAR(shmName_data)

thisfitcond=ThisFitConds[nlbls-1]
DisplayType=-1													;set to no displays
DisplaySet=DisplayType
xsz=thisfitcond.xsz												;number of x pixels
ysz=thisfitcond.ysz												;number of y pixels
framefirst=	thisfitcond.Frm0 + (nlps)*increment					;first frame in batch
framelast=((thisfitcond.Frm0 + (nlps+1L)*increment-1L)<thisfitcond.Nframesmax) < thisfitcond.FrmN
Nframes=long(framelast-framefirst+1l)								;number of frames to extract in file
d = thisfitcond.MaskSize		; d=5.								half size of region of a selected peak
																;setup twinkle structure with all the fitted peak information
peakparams = {twinkle,frameindex:0l,peakindex:0l,fitOK:0,peakx:0.0,peaky:0.0,A:fltarr(7),sigma1:fltarr(7),sigma2:fltarr(7),chisq:0.0,Nphot:0l}
peakparams.A=[0.0,1.0,1.2,1.2,d,d,0.]
print,'starting the worker process, Frames=',framefirst,framelast
firstpeak = -1
lastpeak = 0
for Label=0,nlbls-1 do begin
	data=ReadData(MLRawFilenames[Label],ThisFitConds[Label],framefirst,Nframes)	;Reads thefile and returns data (bunch of frames) in (units of photons)
	print,MLRawFilenames[Label]
	for frameindx=0l,Nframes-1 do begin
		if (frameindx mod 50) eq 0 then print,'Frameindex=',frameindx
		if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data ;clip from transformed dat, frame with correct label
		peaks_in_frame=long(where((CGroupParams_bridge[9,*] eq (frameindx+framefirst)),num_peaks))
		if (firstpeak eq -1) and (num_peaks gt 0) then firstpeak = min(peaks_in_frame) > 0
		if lastpeak lt max(peaks_in_frame) then lastpeak = max(peaks_in_frame)
		for ii=0L,num_peaks-1 do begin
				peakparams.A[0:1]=CGroupParams_bridge[0:1,peaks_in_frame[ii]]/3.			;base & amplitude
				peakparams.A[2:3]=CGroupParams_bridge[4:5,peaks_in_frame[ii]]				;Fitted sigma x and y of Gaussian
				peakparams.A[4:5]=CGroupParams_bridge[2:3,peaks_in_frame[ii]]				;Fitted x and y center of Gaussian
				Dispxy=[CGroupParams_bridge[10,peaks_in_frame[ii]],Label]					;tell index of chosen frame and label
				peakx=fix(peakparams.A[4])											;fitted x center of totaled triplet
				peaky=fix(peakparams.A[5])											;fitted y center of totaled triplet
				peakparams.A[4:5]=peakparams.A[4:5]-[peakx,peaky]+d-0.5
				fita = [1,1,1,1,0,0]												;fit only the base and amplitude
				FindnWackaPeak, clip, d, peakparams, fita, result, ThisFitConds[Label], DisplaySet, peakx, peaky, criteria, Dispxy, 0		;Find,fit,remove target peak & return fit parameters
				CGroupParams_bridge[27+Label,peaks_in_frame[ii]]= peakparams.A[1]			; Fitted Peak Amplitude
				CGroupParams_bridge[30,peaks_in_frame[ii]]=+peakparams.fitOK*10^Label											; New FitOK
		endfor
	endfor
endfor

return
end
;
;------------------------------------------------------------------------------------
;
pro ReadRawLoopMultipleLabel, DisplayType			;Master program to read data and loop through processing
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
										;use SigmaSym as a flag to indicate xsigma and ysigma are not independent and locked together in the fit
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
nlbls=n_elements(MLRawFilenames)
print,'MLRawFilenames:    ',MLRawFilenames
for i=0,nlbls-1 do begin
	ReadThisFitCond, (MLRawFilenames[i]+'.txt'), pth, filen, ini_filename, thisfitcond
	if i eq 0 then ThisFitConds=replicate(thisfitcond,nlbls) else ThisFitConds[i]=thisfitcond
endfor
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz													;number of y pixels

xy_sz=sqrt(float(xsz)*float(ysz))
min_frames_per_node= long(max((thisfitcond.FrmN-thisfitcond.Frm0)/500.00))>1L
increment = (thisfitcond.LocalizationMethod gt 0) ?	min_frames_per_node	:	long((500*(256.0/xy_sz)))>min_frames_per_node				;setup loopback conditions to write multiple files
;if DisplayType eq 4 then increment=5000
if (thisfitcond.LocalizationMethod eq 0) and (thisfitcond.FrmN le 500) then increment = thisfitcond.FrmN-thisfitcond.Frm0+1
increment = long(round(increment * 125.0 / thisfitcond.maxcnt1)) > 1L
if DisplayType eq 4 and (!CPU.HW_NCPU gt 1) then increment = (long((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/(!CPU.HW_NCPU < n_br_max)))>1L

n_cluster_nodes_max = 256
nloops = long((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/increment) < n_cluster_nodes_max			;nloops=long((framelast-framefirst)/increment)
;don't allow to use more then n_cluster_nodes_max cluster cores
if (DisplayType eq 4) then begin
	print,!CPU.HW_NCPU,'  CPU cores are present'
	nloops = (nloops < !CPU.HW_NCPU) < n_br_max
		; don't allow more bridge processes than there are CPU's
	print, 'will start', nloops,' bridge child processes'
endif

increment = long(ceil((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/nloops))
print,'increment=',increment
nloops = long(ceil((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/increment)) > 1L
print,'nloops=',nloops

d = thisfitcond.MaskSize			;d=5.					half size of region of a selected peak
																;setup twinkle structure with all the fitted peak information
peakparams = {twinkle,frameindex:0l,peakindex:0,fitOK:0,peakx:0.0,peaky:0.0,A:fltarr(7),sigma1:fltarr(7),sigma2:fltarr(7),chisq:0.0,Nphot:0l}
peakparams.A=[0.0,1.0,1.2,1.2,d,d,0.]

;case DisplayType of		; 0 - local, 3-cluster, 4-IDL bridge

if DisplayType eq 3 then begin 	;set to 3 (--> -1) - Cluster
	DisplayType=-1			;turns of all displays during processing
	;if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else idl_pwd=pref_get('IDL_WDE_START_DIR')
	cd,current=curr_pwd
	;print,'ReadRawLoopMultipleLabel, curr_pwd:', curr_pwd
	temp_dir = curr_pwd+'/temp'
	print,'nloops=',nloops
	FILE_MKDIR,temp_dir
	save, CGroupParams, nlbls, curr_pwd, idl_pwd, pth, ini_filename, thisfitcond, MLRawFilenames, ThisFitConds, increment, nloops, filename='temp/temp.sav'		;save variables for cluster cpu access
	ReadRawLoopMultiLabelCluster
	file_delete,'temp/temp.sav'
	file_delete,'temp'
	cd,curr_pwd
endif	else begin
	if DisplayType eq 4 then begin	; 4-IDL bridge
		DisplayType=-1			;turns of all displays during processing
		;if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else idl_pwd=pref_get('IDL_WDE_START_DIR')
		cd,current=curr_pwd
		FILE_MKDIR,curr_pwd+'/temp'
		iPALM_data_cnt=n_elements(CGroupParams)/CGrpSize
		save, nlbls, curr_pwd, idl_pwd, iPALM_data_cnt, CGrpSize, pth, ini_filename, thisfitcond, MLRawFilenames, ThisFitConds, increment, nloops, filename='temp/temp.sav'		;save variables for cluster cpu access
		ReadRawLoopMultiLabel_Bridge
		file_delete,'temp/temp.sav'
		file_delete,'temp'
		cd,curr_pwd
	endif else begin
		trash=''
		for nlps=0L,nloops-1 do begin											;loop for all file chunks
			framefirst=	thisfitcond.Frm0 + (nlps)*increment					;first frame in batch
			framelast=((thisfitcond.Frm0 + (nlps+1L)*increment-1L)<thisfitcond.Nframesmax) < thisfitcond.FrmN
			Nframes=long(framelast-framefirst+1L) 								;number of frames to extract in file

			for Label=0,nlbls-1 do begin
				data=ReadData(MLRawFilenames[Label],ThisFitConds[Label],framefirst,Nframes)	;Reads thefile and returns data (bunch of frames) in (units of photons)
		;		if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data
				for frameindx=0l,(Nframes-1l) do begin
					if (DisplayType ge 1) and ((frameindx mod 100) eq 0) then DisplaySet=2 else DisplaySet=DisplayType
					if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data ;clip from transformed dat, frame with correct label
					If (DisplaySet gt 1) or (DisplaySet ge 1)*((frameindx mod 20) eq 1) then begin
						xyouts,0.85,0.92,trash,/normal,col=0
						xyouts,0.05,0.02,trash,/normal,col=0
						xyouts,0.85,0.92,string(frameindx+framefirst)+'#',/normal
						xyouts,0.05,0.02,string(frameindx+framefirst)+'#',/normal
						trash=string(frameindx+framefirst)+'#'
					endif
					peaks_in_frame=where((CGroupParams[9,*] eq (frameindx+framefirst)),num_peaks)
					for ii=0L,num_peaks-1L do begin
						peakparams.A[0:1]=CGroupParams[0:1,peaks_in_frame[ii]]/3.			;base & amplitude
						peakparams.A[2:3]=CGroupParams[4:5,peaks_in_frame[ii]]				;Fitted sigma x and y of Gaussian
						peakparams.A[4:5]=CGroupParams[2:3,peaks_in_frame[ii]]				;Fitted x and y center of Gaussian
						Dispxy=[CGroupParams[10,peaks_in_frame[ii]],Label]					;tell index of chosen frame and label
						peakx=fix(peakparams.A[4])											;fitted x center of totaled triplet
						peaky=fix(peakparams.A[5])											;fitted y center of totaled triplet
						peakparams.A[4:5]=peakparams.A[4:5]-[peakx,peaky]+d-0.5
						fita = [1,1,1,1,0,0]
						;fita = [1,1,0,0,0,0]												;fit only the base and amplitude
						FindnWackaPeak, clip, d, peakparams, fita, result, ThisFitConds[Label], DisplaySet, peakx, peaky, criteria, Dispxy, 0		;Find,fit,remove target peak & return fit parameters
						CGroupParams[27+Label,peaks_in_frame[ii]]= peakparams.A[1]			; Fitted Peak Amplitude
						CGroupParams[30,peaks_in_frame[ii]]=+peakparams.fitOK*10^Label											; New FitOK
						;CGroupParams[31+Label,peaks_in_frame[ii]]=peakparams.sigma2[1]
					endfor
				endfor
			endfor
		endfor
	endelse
endelse
return
end
