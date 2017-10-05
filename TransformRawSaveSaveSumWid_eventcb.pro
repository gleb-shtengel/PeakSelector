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
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common transformfilenames, lab_filenames, sum_filename
if (size(RawFilenames))[2] le 0 then return
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
existing_ind=where((RawFilenames ne ''),nlabels)
lab_filenames=strarr(nlabels)
tr_tar=strtrim(strmid(min(where(FiducialCoeff[existing_ind[0:(nlabels-1)]].present eq 0))+1,0),1)
for jj=0,nlabels-1 do begin
	lbl_ind=existing_ind[jj]
    	pos=max(strsplit(RawFilenames[lbl_ind],sep))
		fpath=strmid(RawFilenames[lbl_ind],0,pos-1)
		pos1=strpos(RawFilenames[lbl_ind],'cam'+strtrim(strmid((lbl_ind+1),0),1))
		pos2=strpos(RawFilenames[lbl_ind],('c'+strtrim(strmid((lbl_ind+1),0),1)),/REVERSE_SEARCH) > strpos(RawFilenames[lbl_ind],('C'+strtrim(strmid((lbl_ind+1),0),1)), /REVERSE_SEARCH)
		if pos1 gt 1 then begin
			lab_filenames[jj]=strmid(RawFilenames[lbl_ind],0,pos1+4)+'trn'+tr_tar+strmid(RawFilenames[lbl_ind],pos1+4,(strlen(RawFilenames[lbl_ind])-pos1-4))+'.dat'
		endif else begin
			if pos2 gt 1 then begin
					lab_filenames[jj]=strmid(RawFilenames[lbl_ind],0,pos2+2)+'trn'+tr_tar+strmid(RawFilenames[lbl_ind],pos2+2,(strlen(RawFilenames[lbl_ind])-pos2-2))+'.dat'
			endif else	lab_filenames[jj]=strmid(RawFilenames[lbl_ind],pos)+'_transformed.dat'
		endelse
endfor
pos=max(strsplit(RawFilenames[0],sep))
fpath=strmid(RawFilenames[0],0,pos-1)
first_file=strmid(RawFilenames[min(existing_ind)],pos)
cam_pos=strpos(first_file,'cam') > strpos(first_file,'Cam')
sum_suffix = (nlabels eq 3) ? '123_sum' : '12_sum'

if cam_pos gt 1 then begin
	sum_filename=strmid(first_file,0,cam_pos+3)+sum_suffix+strmid(first_file,cam_pos+4,strlen(first_file)-cam_pos-4)+'.dat'
endif else begin
	cam_pos1=strpos(first_file,'c',/REVERSE_SEARCH) > strpos(first_file,'C',/REVERSE_SEARCH)
	if cam_pos1 gt 1 then begin
		sum_filename=strmid(first_file,0,cam_pos1+1)+sum_suffix+strmid(first_file,cam_pos1+2,strlen(first_file)-cam_pos1-2)+'.dat'
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

WidDListSigmaSym = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_SetSigmaFitSym_TRS')
if (size(thisfitcond))[2] eq 8 then $
widget_control,WidDListSigmaSym,SET_DROPLIST_SELECT=thisfitcond.SigmaSym		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep

TransformEngineID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_TransformEngine')
widget_control,TransformEngineID,SET_DROPLIST_SELECT = TransformEngine			;Set the default value to Local or Cluster

wGrpGapID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Group_Gap_tr')
widget_control,wGrpGapID,set_value=grouping_gap

wGrpRadID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Grouping_Radius_tr')
widget_control,wGrpRadID,set_value=grouping_radius100

end
;
;-----------------------------------------------------------------
;
pro StartSaveTransformed, Event			; Starts the transforms
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common transformfilenames, lab_filenames, sum_filename
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
if (size(RawFilenames))[2] le 0 then return
existing_ind=where((RawFilenames ne ''),nlabels)

if n_elements(wind_range) ge 1 then wind_range=wind_range[0]

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

WidDListDispLevel = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_TransformEngine')
TransformEngine = widget_info(WidDListDispLevel,/DropList_Select)

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
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
	Start_Time= SYSTIME(/SECONDS)
	sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

	if n_elements(filen) ne 0 then orig_filen=filen
	nlabels=n_elements(filenames)
	def_wind=!D.window
	window,10,xsize=800,ysize=100,xpos=50,ypos=450,Title='Transforming and Saving Raw Data'
	prev_message='Started Reading and Transforming data'
	xyouts,0.1,0.5,prev_message,CHARSIZE=2.0,/NORMAL
	wait,0.01
	for i=5,15 do close,i
	openw,5,sum_filename
	for i=0,nlabels-1 do begin
		pos=strpos(RawDataFiles[i],sep,/REVERSE_OFFSET,/REVERSE_SEARCH)
		pth=strmid(RawDataFiles[i],0,pos)
		RawDataFileTxt=strmid(RawDataFiles[i],(pos+1))+'.txt'
		ReadThisFitCond, RawDataFileTxt, pth, filen, ini_filename, thisfitcond
		thisfitcond.fliphor=0
		thisfitcond.flipvert=0
		if i eq 0 then ThisFitConds=replicate(thisfitcond,nlabels) else ThisFitConds[i]=thisfitcond
		openw,6+i,filenames[i]
		if (TransformEngine eq 0) or (TransformEngine eq 2) then openr,10+i,strmid(RawDataFiles[i],(pos+1))+'.dat'
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
	;if xsz gt 256 then increment = 100
	increment = 100*(fix(384./sqrt(float(xsz)*float(ysz)))>1)				;setup loopback conditions to write multiple files

	if thisfitcond.FrmN le 500 then increment=thisfitcond.FrmN-thisfitcond.Frm0+1

	n_cluster_nodes_max = 128
	if TransformEngine eq 1 then begin
		nloops = Fix((thisfitcond.FrmN-thisfitcond.Frm0+1)/increment) < n_cluster_nodes_max			;nloops=Fix((framelast-framefirst)/increment)
		;don't allow to use more then n_cluster_nodes_max cluster cores
		increment = long(floor((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/nloops))
	endif

	nloops = fix(ceil((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/increment))

	frm_first_erase=''
	frm_last_erase=''
	totfr=strtrim(thisfitcond.FrmN,2)
	if TransformEngine eq 1 then begin 	;set to 1 - Cluster
		DisplayType=-1			;turns of all displays during processing
		if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else	idl_pwd=pref_get('IDL_WDE_START_DIR')
		cd,current=curr_pwd
		FILE_MKDIR,curr_pwd+'/temp'
		do_save_sum=0
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
				temp_data_files[i,nlps]=curr_pwd+'/temp/'+strmid(RawDataFiles[i],(pos[i]+1))+'_trn_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.dat'
			endfor
			sum_data_files[nlps]=curr_pwd+'/temp/Sum_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.dat'
			temp_idl_fnames[nlps]=curr_pwd+'/temp/Sum_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'_IDL.sav'
		endfor
		save, curr_pwd, idl_pwd, pth, nlabels, nloops, increment, do_save_sum, temp_data_files, sum_data_files, temp_idl_fnames, $
				ini_filename, thisfitcond, sum_filename, filenames, RawDataFiles, GStarDrifts, FidCoeffs, FlipRot, filename='temp/temp.sav'		;save variables for cluster cpu access
		xyouts,0.1,0.5,prev_message,CHARSIZE=2.0,/NORMAL,col=0
		TransformRaw_Save_SaveSum_Cluster
		file_delete,'temp/temp.sav'
		file_delete,'temp'
		file_delete,'npks_det.sav'
		cd,curr_pwd
	endif else begin
		TransformLocal=0
		if !VERSION.OS_family eq 'unix' then begin
			if (scope_varfetch('HAVECUDAPOLY3D', LEVEL=1)) then TransformLocal=1
			if (scope_varfetch('haveGPUlib', LEVEL=1)) then TransformLocal=2
		endif
		case TransformLocal of
			0: print, "Local Transformation Started"
			1: print, "CUDA Local Transformation Started"
			2: print, "GPU Local Transformation placeholder, Standard Local Transformation Started"
		endcase

		for nlps=0L,nloops-1 do begin											;loop for all file chunks
			framefirst=	thisfitcond.Frm0 + (nlps)*increment					;first frame in batch
			framelast=((thisfitcond.Frm0 + (nlps+1L)*increment-1L)<thisfitcond.Nframesmax) < thisfitcond.FrmN
			Nframes=(framelast-framefirst+1) 								;number of frames to extract in file
			if nlps eq 0 then window,10,xsize=800,ysize=100,xpos=50,ypos=450,Title='Transforming and Saving Raw Data'
			xyouts,0.1,0.5,prev_message,CHARSIZE=2.0,/NORMAL,col=0
			prev_message = 'Transforming Frames ' + strtrim(framefirst,2)+' to '+strtrim(framelast,2)+' of total '+totfr
			xyouts,0.1,0.5,prev_message,CHARSIZE=2.0,/NORMAL
			wait,0.01
			data=uintarr(ThisFitConds[0].xsz,ThisFitConds[0].ysz,Nframes)
			for i=0,nlabels-1 do begin
				point_lun,10+i,2ull*xsz*ysz*framefirst
				readu,10+i,data
				if FlipRot[i].present then begin
					if FlipRot[i].transp then data=transpose(temporary(data),[1,0,2])
					if FlipRot[i].flip_h then begin
						;data=reverse(temporary(data),1)
						data=transpose(temporary(data),[1,0,2])
						data=reverse(temporary(data),2)
						data=transpose(temporary(data),[1,0,2])
					endif
					if FlipRot[i].flip_v then data=reverse(temporary(data),2)
				endif

			case TransformLocal of

            	1:	begin ; CUDA
                	if FidCoeffs[i].present then begin
                    	P = reform(reform(FidCoeffs[i].P,4) # replicate(1,nframes), 2,2,nframes)
                    	Q = reform(reform(FidCoeffs[i].Q,4) # replicate(1,nframes), 2,2,nframes)
						sz=size(data)
						if (sz[0] eq 2) or ((sz[0] eq 3) and (sz[3] eq 1)) then begin
							data=POLY_2D(temporary(reform(data)),reform(P),reform(Q),1)
							data=uint(reform(data,sz[1],sz[2],1))
						endif else data = uint(cu_poly_3d(fix(data), float(P), float(Q)))			;############
					endif
                	if GStarDrifts[i].present then begin
                   		P = reform([0,0,1.0,0] # replicate(1,nframes), 2,2,nframes)
                    	Q = reform([0,1.0,0,0] # replicate(1,nframes), 2,2,nframes)
                    	P[0,*,*] = GStarDrifts[i].xdrift[framefirst:(framefirst+nframes-1)]
                    	Q[0,*,*] = GStarDrifts[i].ydrift[framefirst:(framefirst+nframes-1)]
						sz=size(data)
						if (sz[0] eq 2) or ((sz[0] eq 3) and (sz[3] eq 1)) then begin
							data=POLY_2D(temporary(reform(data)),reform(P),reform(Q),1)
							data=uint(reform(data,sz[1],sz[2],1))
						endif else data = uint(cu_poly_3d(fix(data), float(P), float(Q)))			;###############
					endif
					end

           	2:  begin ;GPUlib present but code unimplemented.  Using plain IDL instead!
                    for k=0,nframes-1 do begin
                       	if GStarDrifts[i].present then begin
                           	P=[[GStarDrifts[i].xdrift[framefirst+k],0],[1,0]]
                           	Q=[[GStarDrifts[i].ydrift[framefirst+k],1],[0,0]]
                           	data[*,*,k]=POLY_2D(temporary(data[*,*,k]),P,Q,1)
                       	endif
                       	if FidCoeffs[i].present then $
                           	data[*,*,k]=POLY_2D(temporary(data[*,*,k]),FidCoeffs[i].P,FidCoeffs[i].Q,1)
                    endfor
                	end

             	0:  begin ;  plain IDL
                    for k=0,nframes-1 do begin
                        if GStarDrifts[i].present then begin
                            P=[[GStarDrifts[i].xdrift[framefirst+k],0],[1,0]]
                            Q=[[GStarDrifts[i].ydrift[framefirst+k],1],[0,0]]
                            data[*,*,k]=POLY_2D(temporary(data[*,*,k]),P,Q,1)
                        endif
                        if FidCoeffs[i].present then $
                            data[*,*,k]=POLY_2D(temporary(data[*,*,k]),FidCoeffs[i].P,FidCoeffs[i].Q,1)
                    endfor
                    end

            endcase
			SumData=(i eq 0) ? data : data+SumData
			writeu,6+i,data
			endfor
			writeu,5,SumData
		endfor
	endelse

	for i=5,15 do close,i
	wdelete,10
	wset,def_wind
	print,'Transfrom Time = ',SYSTIME(/SECONDS)-Start_Time,'   seconds'
end
;
;------------------------------------------------------------------------------------
;
Pro TransformRaw_Save_SaveSum_Cluster			;Read data and loop through transforming for cluster
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
restore,'temp/temp.sav'
pos=intarr(nlabels)
for i=0,nlabels-1 do pos[i]=strpos(RawDataFiles[i],sep,/REVERSE_OFFSET,/REVERSE_SEARCH)
prev_message = 'Starting Cluster Transformation'
xyouts,0.1,0.5,prev_message,CHARSIZE=2.0,/NORMAL
;save, curr_pwd, idl_pwd, pth, nlabels, nloops, increment, do_save_sum, temp_data_files, sum_data_files, $
;				ini_filename, thisfitcond, sum_filename, filenames, RawDataFiles, GStarDrifts, FidCoeffs, FlipRot, filename='temp/temp.sav'		;save variables for cluster cpu access
print,'sh '+idl_pwd+'/transform.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd		;Spawn workers in cluster
spawn,'sh '+idl_pwd+'/transform.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd		;Spawn workers in cluster

for nlps=0,nloops-1 do begin			;reassemble little dat files from all the workers into on big one
	framefirst=	thisfitcond.Frm0 + (nlps)*increment					;first frame in batch
	framelast=((thisfitcond.Frm0 + (nlps+1L)*increment-1L)<thisfitcond.Nframesmax) < thisfitcond.FrmN
	Nframes=(framelast-framefirst+1) 								;number of frames to extract in file
	data=uintarr(thisfitcond[0].xsz,thisfitcond[0].ysz,Nframes)
	xyouts,0.1,0.5,prev_message,CHARSIZE=2.0,/NORMAL,col=0
	prev_message = 'Re-Saving Frames ' + strtrim(framefirst,2)+' to '+strtrim(framelast,2)+' of total '+strtrim(thisfitcond.FrmN,2)
	xyouts,0.1,0.5,prev_message,CHARSIZE=2.0,/NORMAL
	wait,0.01
	for i=0,nlabels-1 do begin
		openr,1, pth+'/temp/'+strmid(RawDataFiles[i],(pos[i]+1))+'_trn_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.dat'
		readu,1,data
		SumData=(i eq 0) ? data : data+SumData
		writeu,6+i,data
		close,1
		file_delete, pth+'/temp/'+strmid(RawDataFiles[i],(pos[i]+1))+'_trn_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'.dat'
	endfor
	writeu,5,SumData
endfor

print,'Finished Transformations'
end
;
;------------------------------------------------------------------------------------
;
Pro	TransformRaw_Save_SaveSum_Worker,nlps,data_dir						;spawn mulitple copies of this programs for cluster
Nlps=ulong((COMMAND_LINE_ARGS())[0])
data_dir=(COMMAND_LINE_ARGS())[1]
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
cd,data_dir
restore,'temp/temp.sav'
print,'worker started',nlabels,nloops, pth,thisfitcond,sum_filename,filenames,RawDataFiles, FidCoeffs, FlipRot
;save, curr_pwd, idl_pwd, pth, nlabels, nloops, increment, do_save_sum, temp_data_files, sum_data_files, $
;				ini_filename, thisfitcond, sum_filename, filenames, RawDataFiles, GStarDrifts, FidCoeffs, FlipRot, filename='temp/temp.sav'		;save variables for cluster cpu access
if n_elements(do_save_sum) eq 0 then do_save_sum=0
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
	openw,6+i,filenames[i]
	openr,10+i,RawDataFiles[i]+'.dat'
	openw,15+i,tempfiles[i]
endfor
if do_save_sum then begin
	sumfile=sum_data_files[Nlps]
	openw,15+nlabels,sumfile
endif

for i=0,nlabels-1 do begin
	point_lun,10+i,2ull*xsz*ysz*framefirst
	readu,10+i,data
	;print,'data size:  ',size(data)
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
				;data=reverse(temporary(data),1)
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
	if do_save_sum then SumData=(i eq 0) ? data : data+SumData
	writeu,15+i,data
endfor
if do_save_sum then writeu,15+nlabels,SumData
for i=5,(15+nlabels) do close,i
print, 'Finished the data transformation, closed data files'

spawn,'sync'
spawn,'sync'
print,'Wrote file '+tempfiles[0]
end
;
;-----------------------------------------------------------------
;
pro ApplyTransforms, Event			;Manual File Select & apply saved transforms to Selected Raw file (-.dat)
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
FlipRot={frt,present:0B,transp:0B,flip_h:0B,flip_v:0B}
NFrames=long64(max(CGroupParams[9,*]))+1
GStarDrift={present:0B,xdrift:dblarr(Nframes),ydrift:dblarr(Nframes),zdrift:dblarr(Nframes)}
FidCoeff={fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)}
RawOriginalFname=Dialog_Pickfile(/read,filter=['*.dat'],title='Pick Raw Data *.dat file')
if RawOriginalFname eq '' then return
RawOriginal_NoExt=strmid(RawOriginalFname,0,strlen(RawOriginalFname)-4)
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
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
print,FlipRot.present, GStarDrift.present, FidCoeff.present
if FlipRot.present or GStarDrift.present or FidCoeff.present then TransformRaw_Save, filename, RawOriginal_NoExt, GStarDrift, FidCoeff, FlipRot
end
;
;-----------------------------------------------------------------
;
pro TransformRaw_Save, filename, RawDataFile, GStarDrift, FidCoeff, FlipRot		; Transformation core called by ApplyTransforms
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
	sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
	close,5
	close,6
	openw,5,filename
	openr,6,RawDataFile+'.dat'
	def_wind=!D.window
	window,10,xsize=500,ysize=100,xpos=50,ypos=250,Title='Transforming and Saving Raw Data'
	xyouts,0.1,0.5,'Started Reading and Transforming data',CHARSIZE=2.0,/NORMAL
	wait,0.01
	ReadThisFitCond, (RawDataFile+'.txt'), pth, filen, ini_filename, thisfitcond
	orig_filen=filen
	targ_file_withext=strmid(filename,max(strsplit(filename,sep)))
	pth=strmid(filename,0,(strlen(filename)-strlen(targ_file_withext)))
	filen=strmid(targ_file_withext,0,strlen(targ_file_withext)-4)
	WriteInfoFile
	filen=orig_filen
	xsz=thisfitcond.xsz													;number of x pixels
	ysz=thisfitcond.ysz													;number of y pixels
	if xsz gt 256 then increment = 100
	if xsz le 256 then increment = 500*(fix(384./sqrt(float(xsz)*float(ysz)))>1)				;setup loopback conditions to write multiple files
	nloops=long((thisfitcond.FrmN-thisfitcond.Frm0)/increment)			;nloops=Fix((framelast-framefirst)/increment)
	frm_first_erase=''
	frm_last_erase=''
	totfr=strtrim(thisfitcond.FrmN,2)
	TransformLocal=0
	if !VERSION.OS_family eq 'unix' then begin
		if (scope_varfetch('HAVECUDAPOLY3D', LEVEL=1)) then TransformLocal=1
		if (scope_varfetch('haveGPUlib', LEVEL=1)) then TransformLocal=2
	endif
	case TransformLocal of
		0: print, "Local Transformation Started"
		1: print, "CUDA Local Transformation Started"
		0: print, "CUDA Local Transformation placeholder, Standard Local Transformation Started"
	endcase
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
 		data=uintarr(thisfitcond.xsz,thisfitcond.ysz,Nframes)
		point_lun,6,2ull*thisfitcond.xsz*thisfitcond.ysz*framefirst
		readu,6,data						;Reads thefile and returns data (bunch of frames)
		if FlipRot.present then begin
				if FlipRot.transp then data=transpose(temporary(data),[1,0,2])
				if FlipRot.flip_h then data=reverse(temporary(data),1)
				if FlipRot.flip_v then data=reverse(temporary(data),2)
		endif

        case TransformLocal of
           1: begin
              case (1) of		;  CUDA
              FidCoeff.present: begin
                P = reform(reform(FidCoeff.P,4) # replicate(1,nframes), 2,2,nframes)
                Q = reform(reform(FidCoeff.Q,4) # replicate(1,nframes), 2,2,nframes)
			sz=size(data)
			if (sz[0] eq 2) or ((sz[0] eq 3) and (sz[3] eq 1)) then begin
				data=POLY_2D(temporary(reform(data)),reform(P),reform(Q),1)
				data=uint(reform(data,sz[1],sz[2],1))
			endif else data = uint(cu_poly_3d(fix(data), float(P), float(Q)))			;###############
               end
              GStarDrift.present: begin
                P = reform([0,0,1.0,0] # replicate(1,nframes), 2,2,nframes)
                Q = reform([0,1.0,0,0] # replicate(1,nframes), 2,2,nframes)
                P[0,*,*] = GStarDrift.xdrift[framefirst:(framefirst+nframes-1)]
                Q[0,*,*] = GStarDrift.ydrift[framefirst:(framefirst+nframes-1)]
			sz=size(data)
			if (sz[0] eq 2) or ((sz[0] eq 3) and (sz[3] eq 1)) then begin
				data=POLY_2D(temporary(reform(data)),reform(P),reform(Q),1)
				data=uint(reform(data,sz[1],sz[2],1))
			endif else data = uint(cu_poly_3d(fix(data), float(P), float(Q)))			;###############
              end
              else:  ; do nothing
            endcase
        end
        0:      begin		;'Using plain IDL '
                for k=0,nframes-1 do begin
                    if GStarDrift.present then begin
                        P=[[GStarDrift.xdrift[framefirst+k],0],[1,0]]
                        Q=[[GStarDrift.ydrift[framefirst+k],1],[0,0]]
                        data[*,*,k]=POLY_2D(temporary(data[*,*,k]),P,Q,1)
                    endif
                    if FidCoeff.present then $
                        data[*,*,k]=POLY_2D(temporary(data[*,*,k]),FidCoeff.P,FidCoeff.Q,1)
                endfor
         end
         2:      begin		;'Using plain IDL '
                for k=0,nframes-1 do begin
                    if GStarDrift.present then begin
                        P=[[GStarDrift.xdrift[framefirst+k],0],[1,0]]
                        Q=[[GStarDrift.ydrift[framefirst+k],1],[0,0]]
                        data[*,*,k]=POLY_2D(temporary(data[*,*,k]),P,Q,1)
                    endif
                    if FidCoeff.present then $
                        data[*,*,k]=POLY_2D(temporary(data[*,*,k]),FidCoeff.P,FidCoeff.Q,1)
                endfor
         end
         endcase

		writeu,5,data
	endfor
	close,5
	close,6
	wdelete,10
	wset,def_wind
end
;
;-----------------------------------------------------------------
; This is a MACRO which, after you select the filenames for the transformed files, transforms and saves the files, extract
; peaks from the new summary file, re-extracts peaks multi-label, and the groups peaks.
;
pro StartTransformAndReExtract, Event

common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster, transformengine, grouping_gap,grouping_radius
common transformfilenames, lab_filenames, sum_filename
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common Offset, PkWidth_offset
common Zdisplay, Z_scale_multiplier, vbar_top

COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
tstart=systime(/seconds)
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
WidDListDispLevel = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_TransformEngine')
TransformEngine = widget_info(WidDListDispLevel,/DropList_Select)
;SigmaSym=0		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep

WidDListSigmaSym = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_SetSigmaFitSym_TRS')
SigmaSym = widget_info(WidDListSigmaSym,/DropList_Select)		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
widget_control,/hourglass

disp_increment=500					; frame interval for progress display
;increment=2500L
WidFramesPerNode = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_FramesPerNode_tr')
widget_control,WidFramesPerNode,get_value=increment
;gap=6
wGrpGapID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Group_Gap_tr')
widget_control,wGrpGapID,get_value=gap
grouping_gap=gap
spacer=gap+2
;grouping_radius100=40
wGrpRadID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Grouping_Radius_tr')
widget_control,wGrpRadID,get_value=grouping_radius100
grouping_radius=FLOAT(grouping_radius100)/100	; in CCD pixel units

StartSaveTransformed, Event
print,'StartTransformAndReExtract Macro: finished transformation and saving'

case TransformEngine of
	0: DispType = 1	;set to 0 - min display while fitting 1 - some display 2 - full display  3 - Cluster; 4 - IDL Bridge
	1: DispType = 3
	2: DispType = 4
endcase

DispType0=DispType

fpath=pth
filen=strmid(sum_filename,strlen(fpath))			;the filename
filen=strmid(filen,0,strlen(filen)-4)			;the filename wo extension
ReadRawLoop6,DispType		;goes through data and fits peaks
MLRawFilenames=lab_filenames
nlbls=n_elements(lab_filenames)
for i=0,nlbls-1 do begin
	interm_file=strmid(lab_filenames[i],strlen(fpath))
	MLRawFilenames[i]=pth+strmid(interm_file,0,strlen(interm_file)-4)
endfor
print,'StartTransformAndReExtract Macro: finished peak extraction on sum file'
OnPurgeButton, Event1
print,'purged not OK peaks'
DispType=DispType0
;print,MLRawFilenames,DispType,SigmaSym
ReadRawLoopMultipleLabel,DispType		;goes through data and fits peaks
CGroupParams[34,*]=atan(sqrt(CGroupParams[27,*]/CGroupParams[28,*]))
if (size(CGroupParams))[0] ne 0 then begin
	ReloadParamlists, Event
	OnUnZoomButton, Event1
endif
print,'StartTransformAndReExtract Macro: finished reprocessing multilabel'
;GroupEngine = TransformEngine
;maxgrsize=10						; not absolute max group size: max group size for arrayed processing (groups with elements>maxgroupsize are later analyzed separately)
;if GroupEngine eq 0 then begin
;	GroupDisplay=1						; 0 for cluster, 1 for local
;	GoodPeaks=where(filter ne 0,OKpkcnt)
;	CGroupParamsGP=CGroupParams[*,GoodPeaks]
;	GroupPeaksCore,CGroupParamsGP,CGrpSize,OKpkcnt,grouping_radius,spacer,maxgrsize,disp_increment,GroupDisplay
;	CGroupParams[*,GoodPeaks]=CGroupParamsGP
;endif else begin
;
;	framefirst=long(ParamLimits[9,0])
;	framelast=long(ParamLimits[9,1])
;	nloops=long(ceil((framelast-framefirst+1.0)/increment))
;	GroupDisplay=0						; 0 for cluster, 1 for local
;	if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR') else idl_pwd=pref_get('IDL_WDE_START_DIR')
;	cd,current=curr_pwd
;	FILE_MKDIR,curr_pwd+'/temp'
;	save, curr_pwd,idl_pwd, CGroupParams, CGrpSize, ParamLimits, increment, nloops, spacer, grouping_radius,maxgrsize,disp_increment,GroupDisplay, filename='temp/temp.sav'		;save variables for cluster cpu access
;	GroupPeaksCluster
;	file_delete,'temp/temp.sav'
;	file_delete,'temp'
;	cd,curr_pwd
;endelse
;print,'StartTransformAndReExtract Macro: finished grouping'
;ReloadParamlists, Event
;OnGroupCentersButton, Event1

results_filename = AddExtension(RawFilenames[0],'_IDL.sav')
print,'StartTransformAndReExtract Macro: Saving the data into file:',results_filename
save, CGroupParams, CGrpSize, ParamLimits, Image, b_set, xydsz, TotalRawData, RawFilenames, MLRawFilenames,$
		GuideStarDrift, FiducialCoeff, FlipRotate, thisfitcond, RowNames, $
		lambda_vac,nd_water, nd_oil, wind_range, z_unwrap_coeff, ellipticity_slopes, $
		nm_per_pixel, wfilename, PkWidth_offset, Z_scale_multiplier, grouping_gap, grouping_radius100, filename=results_filename

wlabel = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,wlabel,set_value=results_filename
print,'StartTransformAndReExtract Macro: Finished'

end
