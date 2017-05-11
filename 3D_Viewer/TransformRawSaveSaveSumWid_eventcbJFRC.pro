;
; IDL Event Callback Procedures
; TransformRawSaveSaveSum_eventcb
;
; Generated on:	11/19/2007 14:38.00
;
;-----------------------------------------------------------------
;
; Empty stub procedure used for autoloading.
;
pro TransformRawSaveSaveSumWid_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_TransformedFilenames, wWidget, _EXTRA=_VWBExtra_			; Initialize initial guess values for filenames
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, RawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, thisfitcond, saved_pks_filename
common transformfilenames, lab_filenames, sum_filename
if (size(RawFilenames))[2] le 0 then return
sep = !VERSION.OS eq 'Win32' ? '\': '/'
existing_ind=where((RawFilenames ne ''),nlabels)
lab_filenames=strarr(nlabels)
tr_tar=strtrim(strmid(min(where(FiducialCoeff[existing_ind[0:(nlabels-1)]].present eq 0))+1,0),1)
for jj=0,nlabels-1 do begin
	lbl_ind=existing_ind[jj]

    	pos=max(strsplit(RawFilenames[lbl_ind],sep))
		fpath=strmid(RawFilenames[lbl_ind],0,pos-1)
		pos1=strpos(RawFilenames[lbl_ind],'cam'+strtrim(strmid((lbl_ind+1),0),1))
		pos2=strpos(RawFilenames[lbl_ind],'c'+strtrim(strmid((lbl_ind+1),0),1))
		if pos1 gt 1 then begin
			lab_filenames[jj]=strmid(RawFilenames[lbl_ind],0,pos1+4)+'trn'+tr_tar+strmid(RawFilenames[lbl_ind],pos1+4,(strlen(RawFilenames[lbl_ind])-pos1-4))+'.dat'
		endif else begin
			if pos2 gt 1 then begin
					lab_filenames[jj]=strmid(RawFilenames[lbl_ind],0,pos2+2)+'trn'+tr_tar+strmid(RawFilenames[lbl_ind],pos2+2,(strlen(RawFilenames[lbl_ind])-pos2-2))+'.dat'
			endif else		lab_filenames[jj]=strmid(RawFilenames[lbl_ind],pos)+'_transformed.dat'
		endelse
endfor
pos=max(strsplit(RawFilenames[0],sep))
fpath=strmid(RawFilenames[0],0,pos-1)
first_file=strmid(RawFilenames[min(existing_ind)],pos)
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
Label1WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Label1Filename')
widget_control,Label1WidID,SET_VALUE = lab_filenames[0]

if n_elements(lab_filenames) ge 2 then begin
	Label2WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Label2Filename')
	widget_control,Label2WidID,SET_VALUE = lab_filenames[1]

	if n_elements(lab_filenames) ge 3 then begin
		Label3WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Label3Filename')
		widget_control,Label3WidID,SET_VALUE = lab_filenames[2]
	endif

	LabelSumWidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_SumFilename')
	widget_control,LabelSumWidID,SET_VALUE = sum_filename
endif

WidDListSigmaSym = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_SetSigmaFitSym')
widget_control,WidDListSigmaSym,SET_DROPLIST_SELECT=thisfitcond.SigmaSym		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep

end
;
;-----------------------------------------------------------------
;
pro StartSaveTransformed, Event			; Starts the transforms
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, RawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, thisfitcond, saved_pks_filename
common transformfilenames, lab_filenames, sum_filename

sep = !VERSION.OS eq 'Win32' ? '\': '/'
if (size(RawFilenames))[2] le 0 then return
existing_ind=where((RawFilenames ne ''),nlabels)

Label1WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Label1Filename')
widget_control,Label1WidID,GET_VALUE = text
lab_filenames[0]=text

if n_elements(lab_filenames) ge 2 then begin
	Label2WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Label2Filename')
	widget_control,Label2WidID,GET_VALUE = text
	lab_filenames[1]=text

	if n_elements(lab_filenames) ge 3 then begin
		Label3WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Label3Filename')
		widget_control,Label3WidID,GET_VALUE = text
		lab_filenames[2]=text
	endif

	LabelSumWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_SumFilename')
	widget_control,LabelSumWidID,GET_VALUE = sum_filename
endif

if (size(sum_filename))[2] le 0 then return
if sum_filename eq '' then return
FlipRot=FlipRotate[existing_ind]
GStarDrifts=GuideStarDrift[existing_ind]
FidCoeffs=FiducialCoeff[existing_ind]
RawDataFiles=RawFilenames[existing_ind]

widget_control,/hourglass
TransformRaw_Save_SaveSum, sum_filename, lab_filenames, RawDataFiles, GStarDrifts, FidCoeffs, FlipRot

widget_control,event.top,/destroy

end
;
;-----------------------------------------------------------------
;
pro OnCancelSave, Event		; close
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro TransformRaw_Save_SaveSum, sum_filename, filenames, RawDataFiles, GStarDrifts, FidCoeffs, FlipRot		; Transformation core called by TransformRaw_Save_SaveSum_MenuItem
common InfoFit, pth, filen, thisfitcond, saved_pks_filename
	sep = !VERSION.OS eq 'Win32' ? '\': '/'
	if n_elements(filen) ne 0 then orig_filen=filen
	nlabels=n_elements(filenames)
	def_wind=!D.window
	window,10,xsize=800,ysize=100,xpos=50,ypos=450,Title='Transforming and Saving Raw Data'
	xyouts,0.1,0.5,'Started Reading and Transforming data',CHARSIZE=2.0,/NORMAL
	wait,0.01
	openw,5,sum_filename
	for i=0,nlabels-1 do begin
		pos=strpos(RawDataFiles[i],sep,/REVERSE_OFFSET,/REVERSE_SEARCH)
		pth=strmid(RawDataFiles[i],0,pos)
		RawDataFileTxt=strmid(RawDataFiles[i],(pos+1))+'.txt'
		ReadThisFitCond, RawDataFileTxt, pth, filen, thisfitcond
		thisfitcond.fliphor=0
		thisfitcond.flipvert=0
		if i eq 0 then ThisFitConds=replicate(thisfitcond,nlabels) else ThisFitConds[i]=thisfitcond
		openw,6+i,filenames[i]
		targ_file_withext=strmid(filenames[i],max(strsplit(filenames[i],sep)))
		filen=strmid(targ_file_withext,0,strlen(targ_file_withext)-4)
		pos_rawfilename=strpos(RawDataFiles[i],sep,/reverse_search,/reverse_offset)
		pth=strmid(RawDataFiles[i],0,pos_rawfilename+1)
		WriteInfoFile
	endfor
	thisfitcond.zerodark=total(ThisFitConds[*].zerodark)
	thisfitcond.thresholdcriteria=total(ThisFitConds[*].thresholdcriteria)*0.6
	thisfitcond.LimChiSq=total(ThisFitConds[*].LimChiSq)
	thisfitcond.cntpere=mean(ThisFitConds[*].cntpere)
	thisfitcond.fliphor=0
	thisfitcond.flipvert=0
	targ_file_withext=strmid(sum_filename,max(strsplit(sum_filename,sep)))
	filen=strmid(targ_file_withext,0,strlen(targ_file_withext)-4)
	WriteInfoFile
	if n_elements(orig_filen) ne 0 then filen=orig_filen
	xsz=thisfitcond.xsz													;number of x pixels
	ysz=thisfitcond.ysz													;number of y pixels
	if xsz gt 256 then increment = 100
	if xsz le 256 then increment = 500*fix(384./sqrt(float(xsz)*float(ysz)))				;setup loopback conditions to write multiple files
	nloops=long((thisfitcond.FrmN-thisfitcond.Frm0)/increment)			;nloops=Fix((framelast-framefirst)/increment)
	frm_first_erase=''
	frm_last_erase=''
	totfr=strtrim(thisfitcond.FrmN,2)
	for nlps=0L,nloops do begin											;loop for all file chunks
		framefirst=	thisfitcond.Frm0 + (nlps)*increment					;first frame in batch
		framelast=((thisfitcond.Frm0 + (nlps+1L)*increment-1L)<thisfitcond.Nframesmax) < thisfitcond.FrmN
		Nframes=(framelast-framefirst+1) 								;number of frames to extract in file
		if nlps eq 0 then window,10,xsize=800,ysize=100,xpos=50,ypos=450,Title='Transforming and Saving Raw Data'
		xyouts,0.1,0.5,'Transforming Frames ' + frm_first_erase+' to '+frm_last_erase+' of total '+totfr,CHARSIZE=2.0,/NORMAL,col=0
		xyouts,0.1,0.5,'Transforming Frames ' + strtrim(framefirst,2)+' to '+strtrim(framelast,2)+' of total '+totfr,CHARSIZE=2.0,/NORMAL
		wait,0.01
		frm_first_erase=strtrim(framefirst,2)
		frm_last_erase=strtrim(framelast,2)
		for i=0,nlabels-1 do begin
			data=ReadData(RawDataFiles[i],ThisFitConds[i],framefirst,Nframes)	;Reads thefile and returns data (bunch of frames) in (units of photons)
			if FlipRot[i].present then begin
				if FlipRot[i].transp then data=transpose(temporary(data),[1,0,2])
				if FlipRot[i].flip_h then data=reverse(temporary(data),1)
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
			SumData=(i eq 0) ? data : data+SumData

			data=uint((temporary(data)*ThisFitConds[i].cntpere+ThisFitConds[i].zerodark)>0)
			writeu,6+i,data

		endfor
		SumData=uint((temporary(SumData)*thisfitcond.cntpere+thisfitcond.zerodark)>0)
		writeu,5,SumData
	endfor
	for i=5,nlabels+5 do close,i
	wdelete,10
	wset,def_wind
end
;
;-----------------------------------------------------------------
;
pro ApplyTransforms, Event			;Manual File Select & apply saved transforms to Selected Raw file (-.dat)
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, RawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
FlipRot={frt,present:0B,transp:0B,flip_h:0B,flip_v:0B}
NFrames=long64(max(CGroupParams[9,*]))
GStarDrift={present:0B,xdrift:dblarr(Nframes+1),ydrift:dblarr(Nframes+1),zdrift:dblarr(Nframes+1)}
FidCoeff={fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)}
RawOriginalFname=Dialog_Pickfile(/read,filter=['*.dat'],title='Pick Raw Data *.dat file')
if RawOriginalFname eq '' then return
RawOriginal_NoExt=strmid(RawOriginalFname,0,strlen(RawOriginalFname)-4)
sep = !VERSION.OS eq 'Win32' ? '\': '/'
pos=max(strsplit(RawOriginal_NoExt,sep))
fpath=strmid(RawOriginal_NoExt,0,pos-1)
cd,fpath
ffile=RawOriginal_NoExt+'_transformed.dat'
filename = Dialog_Pickfile(/write,path=fpath,file=ffile,filter=['*.dat'],title='Transform '+ RawOriginalFname+' and save into *.dat file')
if filename eq '' then return
FRfilename = Dialog_Pickfile(/read,path=fpath,filter=['*.sav'],title='Load Flip/Rotation Transformation *.sav file')
if FRfilename ne '' then restore,filename=FRfilename
GSfilename = Dialog_Pickfile(/read,path=fpath,filter=['*.sav'],title='Load GuideStar Transformation *.sav file')
if GSfilename ne '' then restore,filename=GSfilename
FDfilename = Dialog_Pickfile(/read,path=fpath,filter=['*.sav'],title='Load Fiducial Transformation *.sav file')
if FDfilename ne '' then restore,filename=FDfilename
if GStarDrift.present or FidCoeff.present then TransformRaw_Save, filename, RawOriginal_NoExt, GStarDrift, FidCoeff, FlipRot
end
;
;-----------------------------------------------------------------
;
pro TransformRaw_Save, filename, RawDataFile, GStarDrift, FidCoeff, FlipRot		; Transformation core called by ApplyTransforms
common InfoFit, pth, filen, thisfitcond, saved_pks_filename
	sep = !VERSION.OS eq 'Win32' ? '\': '/'
	openw,5,filename
	def_wind=!D.window
	window,10,xsize=500,ysize=100,xpos=50,ypos=250,Title='Transforming and Saving Raw Data'
	xyouts,0.1,0.5,'Started Reading and Transforming data',CHARSIZE=2.0,/NORMAL
	wait,0.01
	ReadThisFitCond, (RawDataFile+'.txt'), pth, filen, thisfitcond
	orig_filen=filen
	targ_file_withext=strmid(filename,max(strsplit(filename,sep)))
	pth=strmid(filename,0,(strlen(filename)-strlen(targ_file_withext)))
	filen=strmid(targ_file_withext,0,strlen(targ_file_withext)-4)
	WriteInfoFile
	filen=orig_filen
	xsz=thisfitcond.xsz													;number of x pixels
	ysz=thisfitcond.ysz													;number of y pixels
	if xsz gt 256 then increment = 100
	if xsz le 256 then increment = 500*fix(384./sqrt(float(xsz)*float(ysz)))				;setup loopback conditions to write multiple files
	nloops=long((thisfitcond.FrmN-thisfitcond.Frm0)/increment)			;nloops=Fix((framelast-framefirst)/increment)
	frm_first_erase=''
	frm_last_erase=''
	totfr=strtrim(thisfitcond.FrmN,2)
	for nlps=0L,nloops do begin											;loop for all file chunks
		framefirst=	thisfitcond.Frm0 + (nlps)*increment					;first frame in batch
		framelast=((thisfitcond.Frm0 + (nlps+1)*increment-1)<thisfitcond.Nframesmax) < thisfitcond.FrmN
		Nframes=(framelast-framefirst+1) 								;number of frames to extract in file
		if nlps eq 0 then window,10,xsize=800,ysize=100,xpos=50,ypos=250,Title='Transforming and Saving Raw Data'
		xyouts,0.1,0.5,'Transforming Frames ' + frm_first_erase+' to '+frm_last_erase+' of total '+totfr,CHARSIZE=2.0,/NORMAL,col=0
		xyouts,0.1,0.5,'Transforming Frames ' + strtrim(framefirst,2)+' to '+strtrim(framelast,2)+' of total '+totfr,CHARSIZE=2.0,/NORMAL
		wait,0.01
		frm_first_erase=strtrim(framefirst,2)
 		frm_last_erase=strtrim(framelast,2)
		data=ReadData(RawDataFile,thisfitcond,framefirst,Nframes)	;Reads thefile and returns data (bunch of frames) in (units of photons)
		if FlipRot.present then begin
				if FlipRot.transp then data=transpose(temporary(data),[1,0,2])
				if FlipRot.flip_h then data=reverse(temporary(data),1)
				if FlipRot.flip_v then data=reverse(temporary(data),2)
		endif
		for k=0,nframes-1 do begin
			if GStarDrift.present then begin
				P=[[GStarDrift.xdrift[framefirst+k],0],[1,0]]
				Q=[[GStarDrift.ydrift[framefirst+k],1],[0,0]]
				data[*,*,k]=POLY_2D(temporary(data[*,*,k]),P,Q,1)
			endif
			if FidCoeff.present then $
				data[*,*,k]=POLY_2D(temporary(data[*,*,k]),FidCoeff.P,FidCoeff.Q,1)
		endfor
		data=uint((temporary(data)*thisfitcond.cntpere+thisfitcond.zerodark)>0)
		writeu,5,data
	endfor
	close,5
	wdelete,10
	wset,def_wind
end
;
;-----------------------------------------------------------------
; This is a MACRO which, after you select the filenames for the transformed files, transforms and saves the files, extract
; peaks from the new summary file, re-extracts peaks multi-label, and the groups peaks.
;
pro StartTransformAndReExtract, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, RawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, thisfitcond, saved_pks_filename
common transformfilenames, lab_filenames, sum_filename
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
tstart=systime(/seconds)
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
sep = !VERSION.OS eq 'Win32' ? '\': '/'
;SigmaSym=0		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
WidDListSigmaSym = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_SetSigmaFitSym')
SigmaSym = widget_info(WidDListSigmaSym,/DropList_Select)		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
widget_control,/hourglass
StartSaveTransformed, Event
print,'finished transformation and saving'
DispType=(!VERSION.OS eq 'Win32')? 1 : 3	;set to 0 - min display while fitting 1 - some display 2 - full display  3 - Cluster
DispType0=DispType

fpath=pth
filen=strmid(sum_filename,strlen(fpath))			;the filename
filen=strmid(filen,0,strlen(filen)-4)			;the filename wo extension
ReadRawLoop6,DispType,SigmaSym		;goes through data and fits peaks
MLRawFilenames=lab_filenames
nlbls=n_elements(lab_filenames)
for i=0,nlbls-1 do begin
	interm_file=strmid(lab_filenames[i],strlen(fpath))
	MLRawFilenames[i]=pth+strmid(interm_file,0,strlen(interm_file)-4)
endfor
print,'finished peak extraction on sum file'
OnPurgeButton, Event1
print,'purged not OK peaks'
DispType=DispType0
print,MLRawFilenames,DispType,SigmaSym
ReadRawLoopMultipleLabel,MLRawFilenames,DispType,SigmaSym		;goes through data and fits peaks
CGroupParams[34,*]=atan(sqrt(CGroupParams[27,*]/CGroupParams[28,*]))
if (size(CGroupParams))[0] ne 0 then begin
	ReloadParamlists
	OnUnZoomButton, Event1
endif
print,'finished reprocessing multilabel'
;WidDListDispLevel = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_GroupEngine')
;GroupEngine = widget_info(WidDListDispLevel,/DropList_Select)
GroupEngine=(!VERSION.OS eq 'Win32')? 0 : 1
gap=6
grouping_radius100=40
grouping_radius=double(grouping_radius100)/100	; in CCD pixel units
spacer=gap+2
maxgrsize=10						; not absolute max group size: max group size for arrayed processing (groups with elements>maxgroupsize are later analyzed separately)
disp_increment=500					; frame interval for progress display

if GroupEngine eq 0 then begin
	GroupDisplay=1						; 0 for cluster, 1 for local
	GoodPeaks=where(filter ne 0,OKpkcnt)
	CGroupParamsGP=CGroupParams[*,GoodPeaks]
	GroupPeaksCore,CGroupParamsGP,CGrpSize,OKpkcnt,grouping_radius,spacer,maxgrsize,disp_increment,GroupDisplay
	CGroupParams[*,GoodPeaks]=CGroupParamsGP
endif else begin
	increment=2500L
	framefirst=long(ParamLimits[9,0])
	framelast=long(ParamLimits[9,1])
	nloops=long(floor((framelast-framefirst)/increment))
	GroupDisplay=0						; 0 for cluster, 1 for local
	if !VERSION.OS eq 'Win32' then	idl_pwd=pref_get('IDL_WDE_START_DIR') else idl_pwd=pref_get('IDL_MDE_START_DIR')
	cd,current=curr_pwd
	FILE_MKDIR,curr_pwd+'/temp'
	save, curr_pwd,idl_pwd, CGroupParams, CGrpSize, ParamLimits, increment, nloops, spacer, grouping_radius,maxgrsize,disp_increment,GroupDisplay, filename='temp/temp.sav'		;save variables for cluster cpu access
	GroupPeaksCluster
	file_delete,'temp/temp.sav'
	file_delete,'temp'
	cd,curr_pwd
endelse
print,'finished grouping'
ReloadParamlists
OnGroupCentersButton, Event1
end
