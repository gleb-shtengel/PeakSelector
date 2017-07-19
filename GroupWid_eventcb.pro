;
; IDL Event Callback Procedures
; GroupWid_eventcb
; Generated on:	10/26/2007 20:54.09
;
;-----------------------------------------------------------------
;
;
;
pro Initialize_GroupPeaks, wWidget
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
WidDListGroupEngine = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_GroupEngine')
widget_control,WidDListGroupEngine,SET_DROPLIST_SELECT = TransformEngine			;Set the default value to Local for Windows, and Cluster for UNIX
wGrpGapID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Group_Gap')
widget_control,wGrpGapID,set_value = grouping_gap
wGrpGapID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Grouping_Radius')
widget_control,wGrpGapID,set_value=grouping_radius100
end
;
;-----------------------------------------------------------------
;
pro OnGroupingInfoOK, Event		; Starts Grouping according to selected choices
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

WidDListDispLevel = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_GroupEngine')
GroupEngine = widget_info(WidDListDispLevel,/DropList_Select)
TransformEngine = GroupEngine
CGroupParams[Gr_ind,*]=0
interrupt_load = 0

case GroupEngine of

	0: begin
		widget_control,/hourglass
		GroupPeaksLocal, Event
	end

	1: begin	; Cluster
		WidFramesPerNode = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_FramesPerNode')
		widget_control,WidFramesPerNode,get_value=increment
		framefirst=long(ParamLimits[FrNum_ind,0])
		framelast=long(ParamLimits[FrNum_ind,1])
		nloops=long(ceil((framelast-framefirst+1.0)/increment))

		wGrpGapID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Group_Gap')
		widget_control,wGrpGapID,get_value=grouping_gap

		wGrpGapID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Grouping_Radius')
		widget_control,wGrpGapID,get_value=grouping_radius100
		grouping_radius=FLOAT(grouping_radius100)/100	; in CCD pixel units
		spacer = grouping_gap+2
		maxgrsize = 10						; not absolute max group size: max group size for arrayed processing (groups with elements>maxgroupsize are later analyzed separately)
		disp_increment = 100				; frame interval for progress display
		GroupDisplay = 0					; 0 for cluster, 1 for local
		if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else	idl_pwd=pref_get('IDL_WDE_START_DIR')
		cd,current=curr_pwd

		cd,current=curr_pwd
		td = 'temp' + strtrim(ulong(SYSTIME(/seconds)),2)
		temp_dir=curr_pwd + sep + td
		FILE_MKDIR,temp_dir

		save, curr_pwd,idl_pwd, CGrpSize, ParamLimits, increment, nloops, spacer, grouping_radius, maxgrsize, disp_increment, GroupDisplay, RowNames, filename=td + sep + 'temp.sav'		;save variables for cluster cpu access
		spawn,'sh '+idl_pwd+'/group_initialize_jobs.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd+' '+temp_dir			;Spawn grouping workers in cluster

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

		if interrupt_load eq 1 then print,'Grouping aborted, cleaning up...'

		if interrupt_load eq 0 then begin
			GroupPeaksCluster_ReadBack, interrupt_load
		endif
		print,'removing temp directory'
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
		cd,curr_pwd
	end

	2: begin	; IDL Bridge
		widget_control,/hourglass
		print,!CPU.HW_NCPU,'  CPU cores are present, will start as many bridge child processes'
		WidFramesPerNode = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_FramesPerNode')
		widget_control, WidFramesPerNode, get_value = increment
		framefirst = long(ParamLimits[FrNum_ind,0])
		framelast = long(ParamLimits[FrNum_ind,1])
		nloops = long(ceil((framelast-framefirst+1.0)/increment)) < 	!CPU.HW_NCPU
		; don't allow more bridge processes than there are CPU's
		increment = long(ceil((framelast-framefirst+1.0)/nloops))
		nloops = long(ceil((framelast-framefirst+1.0)/increment)) > 1L
		print,' Will start '+strtrim(nloops,2)+' bridge child processes'

		wGrpGapID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Group_Gap')
		widget_control, wGrpGapID, get_value=grouping_gap

		wGrpGapID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Grouping_Radius')
		widget_control, wGrpGapID, get_value=grouping_radius100
		grouping_radius = FLOAT(grouping_radius100)/100	; in CCD pixel units
		spacer = grouping_gap + 2
		maxgrsize = 10						; not absolute max group size: max group size for arrayed processing (groups with elements>maxgroupsize are later analyzed separately)
		disp_increment = 100					; frame interval for progress display
		GroupDisplay = 0						; 0 for cluster, 1 for local
		if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else	idl_pwd=pref_get('IDL_WDE_START_DIR')
		cd, current = curr_pwd
		FILE_MKDIR, curr_pwd+'/temp'
		iPALM_data_cnt=n_elements(CGroupParams)
		save, curr_pwd, idl_pwd, iPALM_data_cnt, CGrpSize, ParamLimits, increment, nloops, spacer, grouping_radius, maxgrsize, disp_increment, GroupDisplay, RowNames, filename='temp/temp.sav'		;save variables for cluster cpu access
		GroupPeaks_Bridge, CGroupParams
		;file_delete, 'temp/temp.sav'
		;file_delete, 'temp'
		file_delete,'temp', /RECURSIVE, /ALLOW_NONEXISTENT
		cd, curr_pwd
	end
endcase

if interrupt_load eq 0 then begin
	wait,0.5
	print,'Finished Grouing'

	if bridge_exists gt 0 then begin
		print,'Resetting Bridge Structure'
		CATCH, Error_status1
		SHMUnmap, shmName_data
		IF Error_status1 NE 0 THEN BEGIN
			PRINT, 'Grouping: Error while unmapping', shmName_data,':  ',!ERROR_STATE.MSG
		ENDIF
		CATCH,/CANCEL
		CATCH, Error_status2
		SHMUnmap, shmName_filter
		IF Error_status2 NE 0 THEN BEGIN
			PRINT, 'Grouping: Error while unmapping', shmName_filter,':  ',!ERROR_STATE.MSG
		ENDIF
		CATCH,/CANCEL
		CATCH, Error_status3
		for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
		IF Error_status3 NE 0 THEN BEGIN
			PRINT, 'Grouping: Error while destroying fbr_arr objects:  ',!ERROR_STATE.MSG
		ENDIF
		CATCH,/CANCEL
		bridge_exists = 0
	endif

	indices0 = [Gr_ind, GrX_ind, GrY_ind, GrSigX_ind, GrSigY_ind, GrNph_ind, Gr_size_ind, GrInd_ind, $
							GrAmpL1_ind, GrAmpL2_ind, GrAmpL3_ind, GrZ_ind, GrSigZ_ind, GrCoh_ind, Gr_Ell_ind, UnwGrZ_ind, UnwGrZErr_ind]
	indices = indices0[where(indices0 ge 0)]
	ReloadParamlists, Event, indices
	OnGroupCentersButton, Event
endif

widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro GroupPeaksLocal, Event			; Perform Grouping locally using new fast grouping core
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
;FilterIt

;************ 1	Parameters read from GUI
wGrpGapID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Group_Gap')
widget_control,wGrpGapID,get_value=grouping_gap
wGrpGapID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Grouping_Radius')
widget_control,wGrpGapID,get_value=grouping_radius100
grouping_radius=FLOAT(grouping_radius100)/100	; in CCD pixel units
spacer=grouping_gap+2
maxgrsize=10						; not absolute max group size: max group size for arrayed processing (groups with elements>maxgroupsize are later analyzed separately)
disp_increment=100					; frame interval for progress display
GroupDisplay=1						; 0 for cluster, 1 for local
;
;************ 2	Grouping

OKpkcnt = (size(CGroupParams))[2]
GroupPeaksCore,CGroupParams,CGrpSize,OKpkcnt,grouping_radius,spacer,maxgrsize,disp_increment,GroupDisplay,RowNames

return
end
;
;------------------------------------------------------------------------------------
;
Pro GroupPeaksCluster_ReadBack, interrupt_load			;Master program to loop through group processing for cluster, using same fast new group processing core
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common test, test1, nlps, test2, test3
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #

restore,(temp_dir+'/temp.sav')
spawn,'sh '+idl_pwd+'/group_status_check.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd+' '+temp_dir			;Start monitoring the grouping workers on cluster

framefirst=long(ParamLimits[FrNum_ind,0])
framelast=long(ParamLimits[FrNum_ind,1])

file_delete,(temp_dir+'/npks_det.sav')

print,'Grouping finished, reading back the data...'
oStatusBar = obj_new('PALM_StatusBar', $
        	COLOR=[0,0,255], $
        	TEXT_COLOR=[255,255,255], $
        	CANCEL_BUTTON_PRESENT = 1, $
       	 	TITLE='Reading back  grouped data...', $
      		TOP_LEVEL_BASE=tlb)
fraction_complete_last=0.0D
pr_bar_inc=0.01D
interrupt_load = 0
nlps = 0L

while (nlps lt nloops) and (interrupt_load eq 0) do begin			;reassemble little pks files from all the workers into on big one
	framestart=	framefirst + (nlps)*increment						;first frame in batch
	framestop=(framefirst + (nlps+1L)*increment-1)<framelast
	par_fname = temp_dir+'/group_parameters_'+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)+'_cgr.par'
	print,'Reading the data back from: ',par_fname
	test1=file_info(par_fname)
	if ~test1.exists then begin
		print,'Did not found file:',test1.name
	endif else begin
		restore,filename = par_fname
		if n_elements(CGroupParamsGP) gt 2 then begin
			;print,'File returned peaks:', size(CGroupParamsGP)
			if nlps gt 0 then CGroupParamsGP[Gr_ind,*]=CGroupParamsGP[Gr_ind,*]+max(CGroupParams[Gr_ind,*])+1
			ind0 = (where(CGroupParams[FrNum_ind,*] ge framestart))[0]
			;CGroupParams[*,indecis]=CGroupParamsGP  ;- slow!
			CGroupParams[0,ind0]=CGroupParamsGP	; should be 4x faster!
		endif else print,'File retuned no valid peaks: ',par_fname

		file_delete, par_fname
	endelse
	fraction_complete=FLOAT(nlps)/FLOAT((nloops-1.0))
	if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
		fraction_complete_last=fraction_complete
		oStatusBar -> UpdateStatus, fraction_complete
	endif
	nlps++
	interrupt_load = oStatusBar -> CheckCancel()
endwhile
obj_destroy, oStatusBar
print,'Finished reading the data back from Cluster'
return
end
;
;------------------------------------------------------------------------------------
;
Pro	GroupPeaksWorker, nlps, data_dir, temp_dir						;spawn mulitple copies of this programs for cluster

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
    PRINT, 'GroupPeaksWorker Error index: ', Error_status
    PRINT, 'GroupPeaksWorker Error message: ', !ERROR_STATE.MSG
	CATCH, /CANCEL
	return
ENDIF

Nlps=long((COMMAND_LINE_ARGS())[0])
data_dir=(COMMAND_LINE_ARGS())[1]
temp_dir=(COMMAND_LINE_ARGS())[2]
cd,data_dir
;restore,'temp/temp.sav'
;restore,'temp/temp'+strtrim(Nlps,2)+'.sav'
temp_data_file = temp_dir + '/temp'+strtrim(Nlps,2)+'.sav'
print,'Starting GroupPeaksWorker, restore deta from file: ', temp_data_file
restore, temp_data_file
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
framefirst=long(ParamLimits[FrNum_ind,0])
framelast=long(ParamLimits[FrNum_ind,1])
framestart=	framefirst + nlps*increment						;first frame in batch
framestop=(framefirst + (nlps+1L)*increment-1L)<framelast

CGPsz = size(CGroupParamsGP)
CGrpSize = CGPsz[1]
OKpkcnt = CGPsz[2]
print,'Starting GroupPeaksCore for framefirst=',framestart,',  framelast=',framestop,',  OKpkcnt=',OKpkcnt
if OKpkcnt ge 1 then begin
	;CGroupParamsGP=CGroupParams[*,GoodPeaks]
	print,'Nlps=',Nlps
	print,'CGroupParamsGP size:', size(CGroupParamsGP)
	print,'CGrpSize: ', CGrpSize
	print,'OKpkcnt: ', OKpkcnt
	print,'grouping_radius: ', grouping_radius
	print,'spacer: ', spacer
	print,'maxgrsize: ', maxgrsize
	print,'disp_increment: ', disp_increment
	print,'GroupDisplay: ', GroupDisplay
	GroupPeaksCore, CGroupParamsGP, CGrpSize, OKpkcnt, grouping_radius, spacer,maxgrsize, disp_increment, GroupDisplay, RowNames
endif else CGroupParamsGP=0
save,CGroupParamsGP,filename=temp_dir+'/group_parameters_'+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)+'_cgr.par'
spawn,'sync'
spawn,'sync'
print,'Wrote file '+temp_dir+'/group_parameters_'+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)+'_cgr.par'
file_delete, temp_data_file
return
end
;
;------------------------------------------------------------------------------------
;
Pro GroupPeaks_Bridge,CGroupParams			;Master program to loop through group processing for cluster, using same fast new group processing core

restore,'temp/temp.sav'
print,'Starting IDL bridge worker routines'
;Starting IDL bridge workers
obridge=obj_new("IDL_IDLBridge", output='')
for i=1, nloops-1 do obridge=[obridge, obj_new("IDL_IDLBridge", output='')]
print,'data_dir:	',curr_pwd
print,'IDL_dir:		',IDL_pwd

shmName='Status_rep_GR'
max_len=150
SHMMAP,shmName,/BYTE, Dimension=nloops*max_len, GET_OS_HANDLE=OS_handle_val1
Reports=SHMVAR(shmName)
shmName_data='iPALM_data'
SHMMAP,shmName_data,/FLOAT, Dimension=[CGrpSize,(iPALM_data_cnt/CGrpSize)], GET_OS_HANDLE=OS_handle_val2
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
	obridge[i]->execute,"restore,'GroupPeaksWorker_Bridge.sav'"
	obridge[i]->execute,'GroupPeaksWorker_Bridge, nlps, data_dir, OS_handle_val1, OS_handle_val2',/NOWAIT
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
Pro	GroupPeaksWorker_Bridge,nlps,data_dir,OS_handle_val1,OS_handle_val2						;spawn mulitple copies of this programs for cluster
Error_status=0
cd,data_dir
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
    rep='GroupPeaksWorker_Bridge Error index: '+!ERROR_STATE.msg
	if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
	Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
	CATCH, /CANCEL
	close,1
	return
ENDIF

restore,'temp/temp.sav'
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number

shmName='Status_rep_GR'
max_len=150
SHMMAP,shmName,/BYTE, Dimension=nloops*max_len,OS_Handle=OS_handle_val1
Reports=SHMVAR(shmName)
rep_i=nlps*max_len

shmName_data='iPALM_data'
SHMMAP,shmName_data,/FLOAT, Dimension=[CGrpSize,(iPALM_data_cnt/CGrpSize)],OS_Handle=OS_handle_val2
CGroupParams_bridge=SHMVAR(shmName_data)

framefirst=long(ParamLimits[FrNum_ind,0])
framelast=long(ParamLimits[FrNum_ind,1])
framestart=	framefirst + nlps*increment						;first frame in batch
framestop=(framefirst + (nlps+1L)*increment-1L)<framelast

GoodPeaks=where((CGroupParams_bridge[FrNum_ind,*] ge framestart) and (CGroupParams_bridge[FrNum_ind,*] le framestop), OKpkcnt)
ind0 = GoodPeaks[0]
ind1 = GoodPeaks[n_elements(GoodPeaks)-1]
rep='Starting GroupPeaksCore for framefirst='+strtrim(framestart,2)+',  framelast='+strtrim(framestop,2)+',  OKpkcnt='+strtrim(OKpkcnt,2)
if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
if OKpkcnt ge 1 then begin
	;CGroupParamsGP=CGroupParams_bridge[*,GoodPeaks]	; slow
	CGroupParamsGP=CGroupParams_bridge[*,ind0:ind1]	; fast
	GroupPeaksCore, CGroupParamsGP, CGrpSize, OKpkcnt, grouping_radius, spacer, maxgrsize, disp_increment, GroupDisplay, RowNames
	;CGroupParams_bridge[*,GoodPeaks] = CGroupParamsGP	; - slow!
	CGroupParams_bridge[0,ind0] = CGroupParamsGP  ; - should be 4x faster
endif else CGroupParamsGP=0
rep='Grouped and saved data, frames '+strtrim(string(framestart,format='(i)'),2)+'-'+strtrim(string(framestop,format='(i)'),2)+', total peaks='+strtrim(OKpkcnt,2)
if strlen(rep) ge max_len then rep=strmid(rep,0,max_len) else rep=(rep+string(bytarr(max_len-strlen(rep))+32B))
Reports[rep_i:(rep_i+strlen(rep)-1)]=byte(rep)
return
end
;
;
; NEW Grouping Procedure (fast)- GS 06.25.07
;----------------------------------------------------------------------------------
;
Pro GroupPeaksCore,CGroupParamsGP,CGrpSize,OKpkcnt,grouping_radius,spacer,maxgrsize,disp_increment,GroupDisplay,RowNames	;New Fast Group peaks based on min distance in frames to next peak, & write into param data 17-25

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
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude

Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
Z_ind = min(where(RowNames eq 'Z Position'))                            ; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))                            ; CGroupParametersGP[35,*] - Sigma Z
GrZ_ind = min(where(RowNames eq 'Group Z Position'))                    ; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))                    ; CGroupParametersGP[41,*] - Group Sigma Z

CGroupParamsGP[Gr_ind,*]=-1											; initialize group ID to -1
CGroupParamsGP[GrInd_ind,*]=1										; imitialize the group frame index in the group to 1 (they are indexed from 1, not from 0)
CGroupParamsGP[Gr_size_ind,*]=1										; initialize the group peak count to 1
CGroupParamsGP[GrNph_ind,*]=CGroupParamsGP[Nph_ind,*]				; initialize the group photon count to peak photon count
CGroupParamsGP[GrX_ind:GrY_ind,*]=CGroupParamsGP[X_ind:Y_ind,*]		; initialize the group position to peak position
if GrZ_ind gt 0 then CGroupParamsGP[GrZ_ind,*]=CGroupParamsGP[Z_ind,*]					; initialize the group position to peak position
CGroupParamsGP[GrSigX_ind:GrSigY_ind,*]=CGroupParamsGP[SigX_ind:SigY_ind,*]				; initialize the group X-Y sigmas to peak X-Y sigmas
if GrSigZ_ind gt 0 then CGroupParamsGP[GrSigZ_ind,*]=CGroupParamsGP[SigZ_ind,*]			; initialize the group Z sigma to peak Z sigma
if AmpL1_ind gt 0 then CGroupParamsGP[GrAmpL1_ind:GrAmpL3_ind,*]=CgroupParamsGP[AmpL1_ind:AmpL3_ind,*]
if Gr_Ell_ind gt 0 then CGroupParamsGP[Gr_Ell_ind,*]=CgroupParamsGP[Ell_ind,*]	;	03/13/09 GES: initialize Group Ellipticity values with peak Ellipticity values

uniq_frames=CGroupParamsGP[FrNum_ind,UNIQ(CGroupParamsGP[FrNum_ind,*],sort(CGroupParamsGP[FrNum_ind,*]))]
uniq_frame_num=n_elements(uniq_frames)
UnTermedGroupCount=0
GroupNumber=0L

GrPeaksX=fltarr(maxgrsize+1,OKpkcnt)	; these are 2D arrays where
GrPeaksY=GrPeaksX						; the grouped peak parameters will be added during analysis, columns are
GrPeaksZ=GrPeaksX						; separate groups, up to maxgrsize+1 elements in each
GrPksStdX=GrPeaksX						; element [maxgrsize,*] is a place where the data for larger groups is damped
GrPksStdY=GrPksStdX						; simialr array for Number of Photons for each peak in the group
GrPksStdZ=GrPeaksX
GrGaussWidthX=GrPksStdX
GrGaussWidthY=GrPksStdX
GrNumPhotons=GrPksStdX

if AmpL1_ind ge 0 then GrA1=GrPksStdX						; similar array for L1 amplitude
if AmpL2_ind ge 0 then GrA2=GrPksStdX
if AmpL3_ind ge 0 then GrA3=GrPksStdX

CGP_Ind=lonarr(maxgrsize+1,OKpkcnt)		; indecis of the peaks within CGroupParamsGP array
Groups=fltarr(5,OKpkcnt)
Groups[*,*]=0					;Groups[0,*] - number of elements in the group
Groups[0,*]=1					;Groups[1,*] - Group Terminamtion Countdown
								;Groups[2,*] - running average of x-position
								;Groups[3,*] - running average of y-position
								;Groups[4,*] - number of photons in the group
tstart=systime(/seconds)
;
;*********** Analysis of the peaks
for i=0L,uniq_frame_num-1 do begin		; cycle over all frames containing "good peaks"
	frame_peak_ind=where(CGroupParamsGP[FrNum_ind,*] eq uniq_frames[i],frame_peak_number)
	if UntermedGroupCount eq 0 then begin
		; if there are no untermed groups, the entire list of peaks is to form new groups
		Groups[2:3,(GroupNumber+lindgen(frame_peak_number))]=CGroupParamsGP[X_ind:Y_ind,frame_peak_ind]
		Groups[4,(GroupNumber+lindgen(frame_peak_number))]=CGroupParamsGP[Nph_ind,frame_peak_ind]
		CGroupParamsGP[Gr_ind,frame_peak_ind]=GroupNumber+lindgen(frame_peak_number)
		Groups[1,GroupNumber+lindgen(frame_peak_number)]=spacer

		GrPeaksX[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[X_ind,frame_peak_ind]
		GrPeaksY[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[Y_ind,frame_peak_ind]
		GrPeaksZ[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[Z_ind,frame_peak_ind]
		GrPksStdX[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[SigX_ind,frame_peak_ind]
		GrPksStdY[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[SigY_ind,frame_peak_ind]
		GrPksStdZ[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[SigZ_ind,frame_peak_ind]
		GrGaussWidthX[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[Xwid_ind,frame_peak_ind]
		GrGaussWidthY[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[Ywid_ind,frame_peak_ind]
		GrNumPhotons[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[Nph_ind,frame_peak_ind]

		if AmpL1_ind ge 0 then GrA1[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[AmpL1_ind,frame_peak_ind]
		if AmpL2_ind ge 0 then GrA2[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[AmpL2_ind,frame_peak_ind]
		if AmpL3_ind ge 0 then GrA3[0,GroupNumber+lindgen(frame_peak_number)]=CGroupParamsGP[AmpL3_ind,frame_peak_ind]

		CGP_Ind[0,GroupNumber+lindgen(frame_peak_number)]=frame_peak_ind
		GroupNumber+=frame_peak_number
	endif else begin
		UntermedInd=where(Groups[1,*] gt 0,size1)
		; build 2D arrays of replicas of X-Y coordinates (as COMPLEX) of Unterminated Groups and Peaks in this Frame
		Groups2D=transpose(replicate(COMPLEX(1,0),frame_peak_number)#COMPLEX(Groups[2,UntermedInd],Groups[3,UntermedInd]))
		Peaks2D=replicate(COMPLEX(1,0),size1)#COMPLEX(CGroupParamsGP[X_ind,frame_peak_ind],CGroupParamsGP[Y_ind,frame_peak_ind])
		Group_Peak_dist=abs(Peaks2D-Groups2D)
		Group_peak_dist_min=min(Group_Peak_dist,grp_i,DIMENSION=1)		; for each peak find the unterminated grop that is closest to it.
		abs_ind=grp_i[where(Group_peak_dist_min lt grouping_radius,num_found)>0]	; this needed to make sure that if there are two groups within the grouping criteria distance, group it with closest
		if num_found gt 0 then begin					; assign the peaks to existing groups if within range (if any)
			indii=ARRAY_INDICES(Group_Peak_dist,abs_ind)
			if size(Peaks2D,/N_DIMENSIONS) eq 1 then begin
				peak_indices=frame_peak_ind
				Matched_Group_Indecis=UntermedInd[indii]
			endif else begin
				peak_indices=frame_peak_ind[indii[1,*]]
				Matched_Group_Indecis=UntermedInd[indii[0,*]]
			endelse
			Groups[0,Matched_Group_Indecis] += 1			; increment number of elements in the group
			grcnt = Groups[0,Matched_Group_Indecis]			; new number of elements in this group
			Groups[1,Matched_Group_Indecis] = spacer		; reset countdown
			CGroupParamsGP[Gr_ind,peak_indices] = Matched_Group_Indecis
			CGroupParamsGP[GrInd_ind,peak_indices] = grcnt			; frame index in the group
					; update running averages ofx- and y- positions in the group to account for a new group member
			Groups[2,Matched_Group_Indecis]=(Groups[2,Matched_Group_Indecis]*Groups[4,Matched_Group_Indecis]+CGroupParamsGP[X_ind,peak_indices]*CGroupParamsGP[Nph_ind,peak_indices])/(Groups[4,Matched_Group_Indecis]+CGroupParamsGP[Nph_ind,peak_indices])
			Groups[3,Matched_Group_Indecis]=(Groups[3,Matched_Group_Indecis]*Groups[4,Matched_Group_Indecis]+CGroupParamsGP[Y_ind,peak_indices]*CGroupParamsGP[Nph_ind,peak_indices])/(Groups[4,Matched_Group_Indecis]+CGroupParamsGP[Nph_ind,peak_indices])
			Groups[4,Matched_Group_Indecis]+=CGroupParamsGP[Nph_ind,peak_indices]
			IndHere=((grcnt-1)<maxgrsize)+Matched_Group_Indecis*(maxgrsize+1)
			GrPeaksX[IndHere]=CGroupParamsGP[X_ind,peak_indices]
			GrPeaksY[IndHere]=CGroupParamsGP[Y_ind,peak_indices]
			GrPeaksZ[IndHere]=CGroupParamsGP[Z_ind,peak_indices]
			GrPksStdX[IndHere]=CGroupParamsGP[SigX_ind,peak_indices]
			GrPksStdY[IndHere]=CGroupParamsGP[SigY_ind,peak_indices]
			GrPksStdZ[IndHere]=CGroupParamsGP[SigZ_ind,peak_indices]
			GrGaussWidthX[IndHere]=CGroupParamsGP[Xwid_ind,peak_indices]
			GrGaussWidthY[IndHere]=CGroupParamsGP[Ywid_ind,peak_indices]
			GrNumPhotons[IndHere]=CGroupParamsGP[Nph_ind,peak_indices]
			if AmpL1_ind ge 0 then GrA1[IndHere]=CGroupParamsGP[AmpL1_ind,peak_indices]
			if AmpL2_ind ge 0 then GrA2[IndHere]=CGroupParamsGP[AmpL2_ind,peak_indices]
			if AmpL3_ind ge 0 then GrA3[IndHere]=CGroupParamsGP[AmpL3_ind,peak_indices]
			CGP_Ind[IndHere]=peak_indices
		endif
		if num_found lt frame_peak_number then begin	; start new groups for all "unidentified" remaining peaks (if any)
			remaining_ind=where(min(Group_Peak_dist,DIMENSION=1) gt grouping_radius,num_remaining); find remaining peaks
			peak_indices=frame_peak_ind[remaining_ind]
			Groups[2:3,(GroupNumber+lindgen(num_remaining))]=CGroupParamsGP[X_ind:Y_ind,peak_indices]
			CGroupParamsGP[Gr_ind,peak_indices]=GroupNumber+lindgen(num_remaining)
			Groups[1,GroupNumber+lindgen(num_remaining)]=spacer
			Groups[4,(GroupNumber+lindgen(num_remaining))]=CGroupParamsGP[Nph_ind,peak_indices]
			GrPeaksX[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[X_ind,peak_indices]
			GrPeaksY[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[Y_ind,peak_indices]
			GrPeaksZ[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[Z_ind,peak_indices]
			GrPksStdX[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[SigX_ind,peak_indices]
			GrPksStdY[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[SigY_ind,peak_indices]
			GrPksStdZ[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[SigZ_ind,peak_indices]
			GrGaussWidthX[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[Xwid_ind,peak_indices]
			GrGaussWidthY[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[Ywid_ind,peak_indices]
			GrNumPhotons[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[Nph_ind,peak_indices]
			if CGrpSize ge 43 then begin
				GrA1[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[AmpL1_ind,peak_indices]
				GrA2[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[AmpL2_ind,peak_indices]
				GrA3[0,GroupNumber+lindgen(num_remaining)]=CGroupParamsGP[AmpL3_ind,peak_indices]
			endif
			CGP_Ind[0,GroupNumber+lindgen(num_remaining)]=peak_indices
			GroupNumber+=num_remaining					; increment the number of groups
		endif
	endelse
	Groups[1,*]-=1
	Groups[1,*]>=0
	UnTermedGroupCount=total(Groups[1,*])
	if (i mod disp_increment eq 1) then begin
		workingFrame=uniq_frames[i]
		if GroupDisplay le 0 then print,'Analyzing CGroupParams at ' + string(workingFrame)+' of  '+string(max(CGroupParamsGP[FrNum_ind,*])) + '  Frames'
		if GroupDisplay ge 1 then xyouts,0.01,0.02,'Analyzing CGroupParams at ' + string(workingFrame-disp_increment)+' of  '+string(max(CGroupParamsGP[FrNum_ind,*])) + '  Frames',/normal,col=0
		if GroupDisplay ge 1 then xyouts,0.01,0.02,'Analyzing CGroupParams at ' + string(workingFrame)+' of  '+string(max(CGroupParamsGP[FrNum_ind,*])) + '  Frames',/normal
		wait,0.01
	endif
endfor
tint=systime(/seconds)
if GroupDisplay ge 1 then print,'total group sorting time (sec)=',tint-tstart

GrInd=where((Groups[0,*] gt 1) and (Groups[0,*] le maxgrsize),GrCnt) ;indices of groups with 1<elements<=maxgroupsize
if grcnt gt 0 then begin
	GrPeaksX=GrPeaksX[0:(maxgrsize-1),GrInd]			; elements of these 2D arrays are either valid peak cordinates or zeroes
	GrPeaksY=GrPeaksY[0:(maxgrsize-1),GrInd]          	; only the groups (columns) with number of (non-zero) elements >1  and <maxgrsize
	GrPeaksZ=GrPeaksZ[0:(maxgrsize-1),GrInd]			; are analyzed at this step
	GrPksStdX=GrPksStdX[0:(maxgrsize-1),GrInd]
	GrPksStdY=GrPksStdY[0:(maxgrsize-1),GrInd]
	GrPksStdZ=GrPksStdZ[0:(maxgrsize-1),GrInd]
	GrNumPhotons=GrNumPhotons[0:(maxgrsize-1),GrInd]
	GrGaussWidthX=GrGaussWidthX[0:(maxgrsize-1),GrInd]
	GrGaussWidthY=GrGaussWidthY[0:(maxgrsize-1),GrInd]

	if AmpL1_ind ge 0 then GrA1=GrA1[0:(maxgrsize-1),GrInd]
	if AmpL2_ind ge 0 then GrA2=GrA2[0:(maxgrsize-1),GrInd]
	if AmpL3_ind ge 0 then GrA3=GrA3[0:(maxgrsize-1),GrInd]

	ValidPks=GrPeaksX ne 0								; Array with "1" elements in the positions of valid peaks
	Pts=where(reform(ValidPks,maxgrsize*GrCnt) ne 0)
	CGP_Ind_1D=reform(CGP_Ind[0:(maxgrsize-1),GrInd],maxgrsize*GrCnt)	; 1D array of individual peaks in CGroupParameters,
																	; which belong to groups with elements >1  and <maxgrsize
	Mult=replicate(1.0,maxgrsize)
	GrMeanX=Mult#Groups[2,GrInd]
	GrMX=reform(GrMeanX,maxgrsize*GrCnt)
	GrMeanY=Mult#Groups[3,GrInd]
	GrMY=reform(GrMeanY,maxgrsize*GrCnt)
	GrNP=reform(Mult#Groups[4,GrInd],maxgrsize*GrCnt)
	GrNE=reform(replicate(1,maxgrsize)#Groups[0,GrInd],maxgrsize*GrCnt)		; Group Nymber of Elements
	GrStdX=GrPeaksX
	GrStdY=GrStdX
	GrStdZ=GrStdX
	cnt=total(ValidPks,1)
;	GrStdX=sqrt(total((GrPeaksX-GrMeanX)^2*GrNumPhotons*ValidPks,1)/(Groups[4,GrInd]-1+total(GrPksStdX^2*GrNumPhotons,1)/(Groups[4,GrInd])^2)/sqrt(2)
	GrStdX=sqrt(total((GrPeaksX-GrMeanX)^2*GrNumPhotons*ValidPks,1)/(Groups[4,GrInd]*cnt)+total(GrPksStdX^2*GrNumPhotons,1)/(Groups[4,GrInd]*cnt))/sqrt(2)
	GrStdXLimit=sqrt(total(GrGaussWidthX^2*GrNumPhotons,1))/Groups[4,GrInd]	;theoretical limit - mean of gauss widths of each peak devided by sqrt of photons)
;	GrStdY=sqrt(total((GrPeaksY-GrMeanY)^2*GrNumPhotons*ValidPks,1)/(Groups[4,GrInd]-1)+total(GrPksStdY^2*GrNumPhotons,1)/(Groups[4,GrInd])^2)/sqrt(2)
	GrStdY=sqrt(total((GrPeaksY-GrMeanY)^2*GrNumPhotons*ValidPks,1)/(Groups[4,GrInd]*cnt)+total(GrPksStdY^2*GrNumPhotons,1)/(Groups[4,GrInd]*cnt))/sqrt(2)
	GrStdYLimit=sqrt(total(GrGaussWidthX^2*GrNumPhotons,1))/Groups[4,GrInd]
	GrMeanZ = Mult # (total(GrPeaksZ*GrNumPhotons,1) / total(GrNumPhotons,1))
	GrMZ = reform(GrMeanZ ,maxgrsize*GrCnt)
	;GrStdZ=sqrt(total((GrPeaksZ-GrMeanZ)^2*GrNumPhotons*ValidPks,1)/(Groups[4,GrInd]*cnt))/sqrt(2)
	GrStdZ=sqrt(total((GrPeaksZ-GrMeanZ)^2*GrNumPhotons*ValidPks,1)/(Groups[4,GrInd]*cnt)+total(GrPksStdZ^2*GrNumPhotons,1)/(Groups[4,GrInd]*cnt))/sqrt(2)

	;GrSX=reform(Mult#GrStd,maxgrsize*GrCnt)
	;GrSY=reform(Mult#GrStd,maxgrsize*GrCnt)
	;GrStd=GrStdX>GrStdXLimit>GrStdY>GrStdYLimit
	GrSX=reform(Mult#(GrStdX>GrStdXLimit),maxgrsize*GrCnt)
	GrSY=reform(Mult#(GrStdY>GrStdYLimit),maxgrsize*GrCnt)
	GrSZ=reform(Mult#GrStdZ,maxgrsize*GrCnt)
	if Gr_Ell_ind gt 0 then begin
		GrGWx=total(GrGaussWidthX*GrNumPhotons,1)/Groups[4,GrInd]		; GES 03/13/09 calculate the Group Gaussian Width in X direction
		GrGWy=total(GrGaussWidthY*GrNumPhotons,1)/Groups[4,GrInd]		; GES 03/13/09 calculate the Group Gaussian Width in Y direction
		GrEllipticity=(GrGWx-GrGWy)/(GrGWx+GrGWy)						; GES 03/13/09 calculate the Group Ellipticity
		GrEllipticityS=reform(Mult#GrEllipticity,maxgrsize*GrCnt)
	endif
	CGroupParamsGP[Gr_size_ind,CGP_Ind_1D[Pts]] = GrNE[Pts]	; total peaks forming the group
	CGroupParamsGP[GrNph_ind,CGP_Ind_1D[Pts]]= GrNP[Pts]		; total photons forming the group
	CGroupParamsGP[GrX_ind,CGP_Ind_1D[Pts]] = GrMX[Pts]		; averaged x position
	CGroupParamsGP[GrY_ind,CGP_Ind_1D[Pts]] = GrMY[Pts]		; averaged y position
	CGroupParamsGP[GrZ_ind,CGP_Ind_1D[Pts]]= GrMZ[Pts]			; averaged z position
	CGroupParamsGP[GrSigX_ind,CGP_Ind_1D[Pts]] = GrSX[Pts]	; new x sigma
	CGroupParamsGP[GrSigY_ind,CGP_Ind_1D[Pts]] = GrSY[Pts]	; new y sigma
	CGroupParamsGP[GrSigZ_ind,CGP_Ind_1D[Pts]] = GrSZ[Pts]		; new z sigma

	if AmpL1_ind gt 0 then begin
		GrA1S=total(GrA1*GrNumPhotons*ValidPks,1)/Groups[4,GrInd]
		GrA1R=reform(Mult#GrA1S,maxgrsize*GrCnt)
		CGroupParamsGP[GrAmpL1_ind,CGP_Ind_1D[Pts]]=GrA1R[Pts]	; summed A1
	endif
	if AmpL2_ind gt 0 then begin
		GrA2S=total(GrA2*GrNumPhotons*ValidPks,1)/Groups[4,GrInd]
		GrA2R=reform(Mult#GrA2S,maxgrsize*GrCnt)
		CGroupParamsGP[GrAmpL2_ind,CGP_Ind_1D[Pts]]=GrA2R[Pts]	; averaged A2
	endif
	if AmpL3_ind gt 0 then begin
		GrA3S=total(GrA3*GrNumPhotons*ValidPks,1)/Groups[4,GrInd]
		GrA3R=reform(Mult#GrA3S,maxgrsize*GrCnt)
		CGroupParamsGP[GrAmpL3_ind,CGP_Ind_1D[Pts]]=GrA3R[Pts]	; averaged A3
	endif

	if Gr_Ell_ind gt 0 then CGroupParamsGP[Gr_Ell_ind,CGP_Ind_1D[Pts]]=float(GrEllipticityS[Pts])	; GES 03/13/09 assign the Group Ellipticity to all elements in the group
	tint=systime(/seconds)
	if GroupDisplay ge 1 then print,'total time after processing groups up to 10 (sec)=',tint-tstart
endif

IndToBeTermed=where((Groups[0,*] gt maxgrsize),NumToBeTermed)	; all groups with more then "maxgrsize" elements to be analyzed
if NumToBeTermed ge 1 then begin
	for ii=0L,NumToBeTermed-1 do begin
		ThisGroup=where(CGroupParamsGP[Gr_ind,*] eq IndToBeTermed[ii],cnt)
		CGroupParamsGP[Gr_size_ind,ThisGroup] = cnt						; Total # of peaks forming group

		Ntot = total(CGroupParamsGP[Nph_ind,ThisGroup])		; Total number of photons in the group
		CGroupParamsGP[GrNph_ind,ThisGroup] = Ntot				;
		CGroupParamsGP[GrX_ind,ThisGroup]=Groups[2,IndToBeTermed[ii]]		;averaged x position
		CGroupParamsGP[GrY_ind,ThisGroup]=Groups[3,IndToBeTermed[ii]]		;averaged y position)
;		CGroupParamsGP[GrSigX_ind,ThisGroup]=$
;			sqrt(total((CGroupParamsGP[X_ind,ThisGroup]-CGroupParamsGP[GrX_ind,ThisGroup])^2*CGroupParamsGP[Nph_ind,ThisGroup])/(CGroupParamsGP[GrNph_ind,ThisGroup]-1) $
;			+ total(CGroupParamsGP[SigX_ind,ThisGroup]^2*CGroupParamsGP[Nph_ind,ThisGroup]/CGroupParamsGP[GrNph_ind,ThisGroup]^2))/sqrt(2)		;new x sigma
;		CGroupParamsGP[GrSigY_ind,ThisGroup]=$
;			sqrt(total((CGroupParamsGP[Y_ind,ThisGroup]-CGroupParamsGP[GrY_ind,ThisGroup])^2*CGroupParamsGP[Nph_ind,ThisGroup])/(CGroupParamsGP[GrNph_ind,ThisGroup]-1) $
;			+ total(CGroupParamsGP[SigY_ind,ThisGroup]^2*CGroupParamsGP[Nph_ind,ThisGroup]/CGroupParamsGP[GrNph_ind,ThisGroup]^2))/sqrt(2)		;new y sigma
		Wgt= CGroupParamsGP[Nph_ind,ThisGroup]/(Ntot*cnt)		;Nphot in ith frame/(Nphot total in group * N peaks in Group)  ##10.61
		CGroupParamsGP[GrSigX_ind,ThisGroup]=$					;  ##10.61
			sqrt( total( ((CGroupParamsGP[X_ind,ThisGroup]-CGroupParamsGP[GrX_ind,ThisGroup])^2 + CGroupParamsGP[SigX_ind,ThisGroup]^2)*Wgt ) )/sqrt(2)		;new x sigma ##10.61
		CGroupParamsGP[GrSigY_ind,ThisGroup]=$					;  ##10.61
			sqrt( total( ((CGroupParamsGP[Y_ind,ThisGroup]-CGroupParamsGP[GrY_ind,ThisGroup])^2 + CGroupParamsGP[SigY_ind,ThisGroup]^2)*Wgt ) )/sqrt(2)		;new y sigma ##10.61
		GrStdXLimit=sqrt(total(CGroupParamsGP[Xwid_ind,ThisGroup]^2*CGroupParamsGP[Nph_ind,ThisGroup])/CGroupParamsGP[GrNph_ind,ThisGroup]^2)
		GrStdYLimit=sqrt(total(CGroupParamsGP[Ywid_ind,ThisGroup]^2*CGroupParamsGP[Nph_ind,ThisGroup])/CGroupParamsGP[GrNph_ind,ThisGroup]^2)
		CGroupParamsGP[GrSigX_ind,ThisGroup] = CGroupParamsGP[GrSigX_ind,ThisGroup]>CGroupParamsGP[GrSigY_ind,ThisGroup]>GrStdXLimit>GrStdYLimit
		CGroupParamsGP[GrSigY_ind,ThisGroup] = CGroupParamsGP[GrSigX_ind,ThisGroup]
		Group_MeanZ = total(CGroupParamsGP[Z_ind,ThisGroup]*CGroupParamsGP[Nph_ind,ThisGroup])/Ntot
		CGroupParamsGP[GrZ_ind,ThisGroup] = Group_MeanZ
		CGroupParamsGP[GrSigZ_ind,ThisGroup] = sqrt(total((CGroupParamsGP[Z_ind,ThisGroup]-Group_MeanZ)^2*CGroupParamsGP[Nph_ind,ThisGroup])/(2*Ntot*cnt))

		if AmpL1_ind gt 0 then CGroupParamsGP[GrAmpL1_ind,ThisGroup] = total(CGroupParamsGP[AmpL1_ind,ThisGroup]*CGroupParamsGP[Nph_ind,ThisGroup])/total(CGroupParamsGP[Nph_ind,ThisGroup])    ; group A1
		if AmpL2_ind gt 0 then CGroupParamsGP[GrAmpL2_ind,ThisGroup] = total(CGroupParamsGP[AmpL2_ind,ThisGroup]*CGroupParamsGP[Nph_ind,ThisGroup])/total(CGroupParamsGP[Nph_ind,ThisGroup])    ; group A2
		if AmpL3_ind gt 0 then CGroupParamsGP[GrAmpL3_ind,ThisGroup] = total(CGroupParamsGP[AmpL3_ind,ThisGroup]*CGroupParamsGP[Nph_ind,ThisGroup])/total(CGroupParamsGP[Nph_ind,ThisGroup])    ; group A3
		if Gr_Ell_ind gt 0 then begin
			GrGWx = total(CGroupParamsGP[Xwid_ind,ThisGroup]*CGroupParamsGP[Nph_ind,ThisGroup])/total(CGroupParamsGP[Nph_ind,ThisGroup]) ; GES 03/13/09 calculate the Group Gaussian Width in X direction
			GrGWy = total(CGroupParamsGP[Ywid_ind,ThisGroup]*CGroupParamsGP[Nph_ind,ThisGroup])/total(CGroupParamsGP[Nph_ind,ThisGroup]) ; GES 03/13/09 calculate the Group Gaussian Width in Y direction
			CGroupParamsGP[Gr_Ell_ind,ThisGroup] = (GrGWx-GrGWy)/(GrGWx+GrGWy)                                         		; GES 03/13/09 calculate the Group Ellipticity
		endif
	endfor
endif

tstop=systime(/seconds)
print,'total new grouping time (sec)=',tstop-tstart
return
end
;
;
;-----------------------------------------------------------------
; Delete everything below this line for 2D version
;
;
;-----------------------------------------------------------------
;
;NEW Grouping Procedure - GS 06.18.07
;-----------------------------------------------------------------
;
Pro GroupPeaksCore_understandable,CGroupParamsGP,OKpkcnt,grouping_radius,spacer,maxgrsize,disp_increment,GroupDisplay			;Group peaks based on min distance in frames to next peak, & write into param data 17-25

;CGroupParametersGP[18,*] - group #
;CGroupParametersGP[24,*] - total peaks in the group
;CGroupParametersGP[25,*] - frame index in the group
;CGroupParametersGP[23,*] - total Photons in the group
;CGroupParametersGP[19,*] - average x - position in the group
;CGroupParametersGP[20,*] - average y - position in teh group
;CGroupParametersGP[21,*] - new x - position sigma
;CGroupParametersGP[22,*] - new y - position sigma
CGroupParamsGP[18,*]=-1								; initialize group ID to -1
CGroupParamsGP[25,*]=1								; imitialize the group frame index in the group to 1
CGroupParamsGP[24,*]=1								; initialize the group peak count to 1
CGroupParamsGP[23,*]=CGroupParamsGP[6,*]			; initialize the group photon count to peak photon count
CGroupParamsGP[19:20,*]=CGroupParamsGP[2:3,*]		; initialize the group position to peak position
CGroupParamsGP[21:22,*]=CGroupParamsGP[16:17,*]		; initialize the group position sigma to peak position sigma
uniq_frames=CGroupParamsGP[9,UNIQ(CGroupParamsGP[9,*],sort(CGroupParamsGP[9,*]))]
uniq_frame_num=n_elements(uniq_frames)
UnTermedGroupCount=0
GroupNumber=0L
Groups=dblarr(5,OKpkcnt)
Groups[*,*]=0					;Groups[0,*] - number of elements in the group
Groups[0,*]=1					;Groups[1,*] - Group Terminamtion Countdown
								;Groups[2,*] - running average of x-position
								;Groups[3,*] - running average of y-position
								;Groups[4,*] - number of photons in the group
tstart=systime(/seconds)
for i=0L,uniq_frame_num-1 do begin		; cycle over all frames containing "good peaks"
	frame_peak_ind=where(CGroupParamsGP[9,*] eq uniq_frames[i],frame_peak_number)
	if UntermedGroupCount eq 0 then begin
		; if there are no untermed groups, the entire list of peaks is to form new groups
		Groups[2:3,(GroupNumber+lindgen(frame_peak_number))]=CGroupParamsGP[2:3,frame_peak_ind]
		Groups[4,(GroupNumber+lindgen(frame_peak_number))]=CGroupParamsGP[6,frame_peak_ind]
		CGroupParamsGP[18,frame_peak_ind]=GroupNumber+lindgen(frame_peak_number)
		Groups[1,GroupNumber+lindgen(frame_peak_number)]=spacer
		GroupNumber+=frame_peak_number
	endif else begin
		UntermedInd=where(Groups[1,*] gt 0,size1)
		; build 2D arrays of replicas of X-Y coordinates (as COMPLEX) of Unterminated Groups and Peaks in this Frame
		GroupXY=replicate(DCOMPLEX(1,0),frame_peak_number)#COMPLEX(Groups[2,UntermedInd],Groups[3,UntermedInd])
		PeaksXY=replicate(DCOMPLEX(1,0),size1)#COMPLEX(CGroupParamsGP[2,frame_peak_ind],CGroupParamsGP[3,frame_peak_ind])
		Group_Peak_dist=abs(PeaksXY-transpose(GroupXY))
		Group_peak_dist1=transpose(min(Group_Peak_dist,DIMENSION=1)#replicate(1,size1))
		Group_dist_Check=(Group_Peak_dist-Group_peak_dist1)			; this needed to make sure that if there are two groups close to a peak, group it with closest
		abs_ind=where(Group_Peak_dist lt grouping_radius and Group_dist_Check eq 0, num_found); find all
		if num_found gt 0 then begin					; assign the peaks to existing groups if within range (if any)
			indii=ARRAY_INDICES(Group_Peak_dist,abs_ind)
			if size(PeaksXY,/N_DIMENSIONS) eq 1 then begin
				peak_indices=frame_peak_ind
				Matched_Group_Indecis=UntermedInd[indii]
			endif else begin
				peak_indices=frame_peak_ind[indii[1,*]]
				Matched_Group_Indecis=UntermedInd[indii[0,*]]
			endelse
			Groups[0,Matched_Group_Indecis]+=1			; increment number of elements in the group
			grcnt=Groups[0,Matched_Group_Indecis]			; new number of elements in this group
			Groups[1,Matched_Group_Indecis]=spacer		; reset countdown
			CGroupParamsGP[18,peak_indices]=Matched_Group_Indecis
			CGroupParamsGP[25,peak_indices]=grcnt			; frame index in the group
					; update running averages ofx- and y- positions in the group to account for a new group member
			; Old running average update without weighted averaging
			;Groups[2,Matched_Group_Indecis]=(Groups[2,Matched_Group_Indecis]*(grcnt-1)+CGroupParamsGP[2,peak_indices])/grcnt
			;Groups[3,Matched_Group_Indecis]=(Groups[3,Matched_Group_Indecis]*(grcnt-1)+CGroupParamsGP[3,peak_indices])/grcnt
			;
			; new weighted running position averaging
			Groups[2,Matched_Group_Indecis]=(Groups[2,Matched_Group_Indecis]*Groups[4,Matched_Group_Indecis]+CGroupParamsGP[2,peak_indices]*CGroupParamsGP[6,peak_indices])/(Groups[4,Matched_Group_Indecis]+CGroupParamsGP[6,peak_indices])
			Groups[3,Matched_Group_Indecis]=(Groups[3,Matched_Group_Indecis]*Groups[4,Matched_Group_Indecis]+CGroupParamsGP[3,peak_indices]*CGroupParamsGP[6,peak_indices])/(Groups[4,Matched_Group_Indecis]+CGroupParamsGP[6,peak_indices])
			Groups[4,Matched_Group_Indecis]+=CGroupParamsGP[6,peak_indices]
		endif
		if num_found lt frame_peak_number then begin	; start new groups for all "unidentified" remaining peaks (if any)
			remaining_ind=where(min(Group_Peak_dist,DIMENSION=1) gt grouping_radius,num_remaining); find remaining peaks
			peak_indices=frame_peak_ind[remaining_ind]
			Groups[2:3,(GroupNumber+lindgen(num_remaining))]=CGroupParamsGP[2:3,peak_indices]
			Groups[4,(GroupNumber+lindgen(num_remaining))]=CGroupParamsGP[6,peak_indices]
			CGroupParamsGP[18,peak_indices]=GroupNumber+lindgen(num_remaining)
			Groups[1,GroupNumber+lindgen(num_remaining)]=spacer
			GroupNumber+=num_remaining					; increment the number of groups
		endif
	endelse
	Groups[1,*]-=1
	Groups[1,*]>=0
	UnTermedGroupCount=total(Groups[1,*])
	if (i mod 1000 eq 1) then begin
		workingFrame=uniq_frames[i]
		if GroupDisplay le 0 then print,'Analyzing CGroupParams at ' + string(workingFrame)+' of  '+string(max(CGroupParamsGP[9,*])) + '  Frames'
		if GroupDisplay ge 1 then xyouts,0.01,0.02,'Analyzing CGroupParams at ' + string(workingFrame-disp_increment)+' of  '+string(max(CGroupParamsGP[9,*])) + '  Frames',/normal,col=0
		if GroupDisplay ge 1 then xyouts,0.01,0.02,'Analyzing CGroupParams at ' + string(workingFrame)+' of  '+string(max(CGroupParamsGP[9,*])) + '  Frames',/normal
		wait,0.01
	endif
endfor
tint=systime(/seconds)
if GroupDisplay le 0 then print,'total group sorting time (sec)=',tint-tstart
if GroupDisplay ge 1 then xyouts,0.01,0.02,'Analyzing CGroupParams at ' + string(workingFrame)+' of  '+string(max(CGroupParamsGP[9,*])) + '  Frames',/normal,col=0

IndToBeTermed=where((Groups[0,*] gt 1),NumToBeTermed)	; all groups with more then 1 elements to be analyzed
if NumToBeTermed ge 1 then begin
	for ii=0L,NumToBeTermed-1 do begin
		ThisGroup=where(CGroupParamsGP[18,*] eq IndToBeTermed[ii],cnt)
		CGroupParamsGP[24,ThisGroup]=cnt						; total peaks forming group
		Ntot= total(CGroupParamsGP[6,ThisGroup])		; Total number of photons in the group
		CGroupParamsGP[23,ThisGroup]=Ntot				; Total peaks forming group

;	******* Old group parameter recalculation with no weighted averaging of group position and sigma
;		mean_variancex = moment(CGroupParamsGP[2,ThisGroup],maxmoment=2)
;		mean_variancey = moment(CGroupParamsGP[3,ThisGroup],maxmoment=2)
;		CGroupParamsGP[19,ThisGroup]=mean_variancex[0]		;averaged x position
;		CGroupParamsGP[20,ThisGroup]=mean_variancey[0]		;averaged y position)
;		CGroupParamsGP[21,ThisGroup]=$
;			sqrt(mean_variancex[1]+total((CGroupParamsGP[16,ThisGroup])^2)/cnt^2)/sqrt(2)		;new x sigma
;		CGroupParamsGP[22,ThisGroup]=$
;			sqrt(mean_variancey[1]+total((CGroupParamsGP[17,ThisGroup])^2)/cnt^2)/sqrt(2)		;new y sigma

;  ******* New Group parameter recalculation with weighted averaging of group position and sigmas. Worst sigma is assigned to both sigma x and sigma y
		CGroupParamsGP[19,ThisGroup]=Groups[2,IndToBeTermed[ii]]		;averaged x position
		CGroupParamsGP[20,ThisGroup]=Groups[3,IndToBeTermed[ii]]		;averaged y position)
;		CGroupParamsGP[21,ThisGroup]=$
;			sqrt(total((CGroupParamsGP[2,ThisGroup]-CGroupParamsGP[19,ThisGroup])^2*CGroupParamsGP[6,ThisGroup])/(CGroupParamsGP[23,ThisGroup]-1) $
;			+ total(CGroupParamsGP[16,ThisGroup]^2*CGroupParamsGP[6,ThisGroup]/CGroupParamsGP[23,ThisGroup]^2))/sqrt(2)		;new x sigma
;		CGroupParamsGP[22,ThisGroup]=$
;			sqrt(total((CGroupParamsGP[3,ThisGroup]-CGroupParamsGP[20,ThisGroup])^2*CGroupParamsGP[6,ThisGroup])/(CGroupParamsGP[23,ThisGroup]-1) $
;			+ total(CGroupParamsGP[17,ThisGroup]^2*CGroupParamsGP[6,ThisGroup]/CGroupParamsGP[23,ThisGroup]^2))/sqrt(2)		;new y sigma
		Wgt= CGroupParamsGP[6,ThisGroup]/(Ntot*cnt)		;Nphot in ith frame/(Nphot total in group * N peaks in Group)
		CGroupParamsGP[21,ThisGroup]=$
			sqrt( total( ((CGroupParamsGP[2,ThisGroup]-CGroupParamsGP[19,ThisGroup])^2 + CGroupParamsGP[16,ThisGroup]^2)*Wgt ) )/sqrt(2)		;new x sigma
		CGroupParamsGP[22,ThisGroup]=$
			sqrt( total( ((CGroupParamsGP[3,ThisGroup]-CGroupParamsGP[20,ThisGroup])^2 + CGroupParamsGP[17,ThisGroup]^2)*Wgt ) )/sqrt(2)		;new y sigma
		GrStdXLimit=sqrt(total(CGroupParamsGP[4,ThisGroup]^2*CGroupParamsGP[6,ThisGroup])/CGroupParamsGP[23,ThisGroup]^2)
		GrStdYLimit=sqrt(total(CGroupParamsGP[5,ThisGroup]^2*CGroupParamsGP[6,ThisGroup])/CGroupParamsGP[23,ThisGroup]^2)
		CGroupParamsGP[21,ThisGroup]=CGroupParamsGP[21,ThisGroup]>CGroupParamsGP[22,ThisGroup]>GrStdXLimit>GrStdYLimit
		CGroupParamsGP[22,ThisGroup]=CGroupParamsGP[21,ThisGroup]

		if (ii mod 1000 eq 1) then begin
			xx=ii
			if GroupDisplay ge 1 then xyouts,0.01,0.02,'Calculating CGroupParams for Group' + string(xx-1000)+' of  '+string(NumToBeTermed) + '  Groups (with more then 1 peak)',/normal,col=0
			if GroupDisplay ge 1 then xyouts,0.01,0.02,'Calculating CGroupParams for Group' + string(xx)+' of  '+string(NumToBeTermed) + '  Groups (with more then 1 peak)',/normal
			wait,0.01
		endif
	endfor
	if GroupDisplay ge 1 then xyouts,0.01,0.02,'Calculating CGroupParams for Group' + string(xx)+' of  '+string(NumToBeTermed) + '  Groups (with more then 1 peak)',/normal,col=0
endif

tstop=systime(/seconds)
if GroupDisplay le 0 then print,'total new grouping time (sec)=',tstop-tstart
return
end
;
; Empty stub procedure used for autoloading.
;
pro GroupWid_eventcb
end
