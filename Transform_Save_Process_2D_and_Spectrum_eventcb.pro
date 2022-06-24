;
;-----------------------------------------------------------------
;
;
; Empty stub procedure used for autoloading.
;
pro Transform_Save_Process_2D_and_Spectrum_eventcb
end
;
;-----------------------------------------------------------------
;
pro OnCancelSave, Event	; close
	widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro Initialize_Transform_2D_and_spectrum, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w

COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
Sp_Cal_Table_ID=widget_info(wWidget,FIND_BY_UNAME='WID_SP_Cal_Table')
sp_window=!D.window

if n_elements(sp_d) lt 3 then begin
	sp_d = [6,20,1]		; SIZE OF THE SPECTRAL WINDOW (in CCD pixel counts)
	Max_sp_num = 6		; max number of spectra allowed
	cal_frames = [20,30]; frames to be used for calibration
	sp_size=sp_d[0]+sp_d[1]+1
endif

if n_elements(cal_spectra) eq 0 then cal_spectra=dblarr(Max_sp_num,sp_size)

Update_Sp_Cal_display,wWidget

if (size(RawFilenames))[1] le 0 then begin
	z=dialog_message('Please load X-Y and Spectrum files')
	return
endif

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
existing_ind=where((RawFilenames ne ''),nlabels)

if n_elements(lab_filenames) lt 2 then begin
	lab_filenames=strarr(2)
	lab_filenames[0]=RawFilenames[0]
	lab_filenames[1]=RawFilenames[1]+'_trn'
endif

tr_tar=strtrim(strmid(min(where(FiducialCoeff[existing_ind[0:(nlabels-1)]].present eq 0))+1,0),1)

XY_file_WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_XY_Filename')
widget_control,XY_file_WidID,SET_VALUE = lab_filenames[0]+'.dat'

TrnSp_file_WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_TrnSpFilename')
if n_elements(lab_filenames) gt 1 then  widget_control,TrnSp_file_WidID,SET_VALUE = lab_filenames[1]+'.dat'

WID_DROPLIST_FitDisplayType_Spectrum_ID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_FitDisplayType_Spectrum')
widget_control,WID_DROPLIST_FitDisplayType_Spectrum_ID,SET_DROPLIST_SELECT = TransformEngine ? 3 : 1			;Set the default value to Local for Windows, and Cluster for UNIX


Cal_Frames_Table_ID=widget_info(wWidget,FIND_BY_UNAME='WID_TABLE_WlCal_FrStart_FrStop')

widget_control,Cal_Frames_Table_ID,set_value=transpose(cal_frames), use_table_select=[0,0,0,1]
end
;
;-----------------------------------------------------------------
;
pro OnPickXYTxtFile, Event
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
XY_file_WidID = Widget_Info(Event.top, find_by_uname='WID_TEXT_XY_Filename')

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
filter_to_read = ['*.dat']
text = Dialog_Pickfile(/read,filter=filter_to_read,title='Pick X-Y Data File *.dat')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
widget_control,XY_file_WidID,SET_VALUE = text
lab_filenames[0] = StripExtension(text)
end
;
;-----------------------------------------------------------------
;
pro OnPickSPTxtFile, Event
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
SP_file_WidID = Widget_Info(Event.top, find_by_uname='WID_TEXT_TrnSpFilename')

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
filter_to_read = ['*.dat']
text = Dialog_Pickfile(/read,filter=filter_to_read,title='Pick Spectral Data File *.dat')
fpath = strmid(text,0,(max(strsplit(text,sep))-1))
if text ne '' then cd,fpath
widget_control,SP_file_WidID,SET_VALUE = text
if n_elements(lab_filenames) gt 1 then  lab_filenames[1] = StripExtension(text) else lab_filenames = [lab_filenames, StripExtension(text)]
end
;
;-----------------------------------------------------------------
;
pro OnPickSpCalFile, Event
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
Max_sp_num=10
spcal_filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.txt'],title='Select *spcal.txt file to open')
if spcal_filename ne '' then begin
	sp_cal_file = spcal_filename
	cd,fpath
	SpCalFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_SpCalFilename')
	widget_control,SpCalFileWidID,SET_VALUE = sp_cal_file
	sp_cal_file_info=FILE_INFO(sp_cal_file)
	sp_size=sp_d[0]+sp_d[1]+1
	if sp_cal_file_info.exists then begin
		cal_spectra=dblarr(Max_sp_num,sp_size)
		cal_spectra_line=dblarr(sp_size)
		close,5
		openr,5,sp_cal_file
		ip=0
		while not EOF(5) do begin
			readf,5,cal_spectra_line
			cal_spectra[ip,*] = cal_spectra_line
			if total(cal_spectra_line) gt 0.00001 then ip+=1
		endwhile
		close,5
		Max_sp_num=ip
		cal_spectra=cal_spectra[0:(Max_sp_num-1),*]
	endif
	Update_Sp_Cal_display,Event.top
endif
end
;
;-----------------------------------------------------------------
;
pro WlShift_Single_Sp_Cal, Event
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
WID_TEXT_RemoveNumber_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_CalSpNum')
widget_control,WID_TEXT_RemoveNumber_ID,GET_VALUE = Remove_Index
WID_TEXT_WlShift_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_WlShift')
widget_control,WID_TEXT_WlShift_ID,GET_VALUE = Wl_Shift
sp_size=sp_d[0]+sp_d[1]+1
cal_spectra_single = cal_spectra[Remove_Index,*]
new_wl=findgen(sp_size)-(float(Wl_Shift))[0]
cal_spectra_new=INTERPOLATE(cal_spectra_single, new_wl , CUBIC=-0.5)>0.0
cal_spectra[Remove_Index,*]=cal_spectra_new
Update_Sp_Cal_display,Event.top
end
;
;-----------------------------------------------------------------
;
pro OnSaveSpCalFile, Event
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
SpCalFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_SpCalFilename')
widget_control,SpCalFileWidID,GET_VALUE = sp_cal_file
if (sp_cal_file eq '') then return
openw,1,sp_cal_file,width=512
printf,1,transpose(cal_spectra)
close,1
end
;
;-----------------------------------------------------------------
;
pro Remove_Single_Sp_Cal, Event
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
WID_TEXT_RemoveNumber_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_CalSpNum')
widget_control,WID_TEXT_RemoveNumber_ID,GET_VALUE = Remove_Index
sp_size=sp_d[0]+sp_d[1]+1
cal_spectra1 = dblarr((Max_sp_num-1),sp_size)
if Remove_Index ge 1 then cal_spectra1[0:(Remove_Index-1),*] = cal_spectra[0:(Remove_Index-1),*]
if Remove_Index lt (Max_sp_num-1) then cal_spectra1[(Remove_Index):(Max_sp_num-2),*] = cal_spectra[(Remove_Index+1):(Max_sp_num-1),*]
cal_spectra=cal_spectra1
Max_sp_num --
Update_Sp_Cal_display,Event.top
end
;
;-----------------------------------------------------------------
;
pro Do_Edit_Sp_Cal, Event
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
widget_control,event.id,get_value=thevalue
cal_spectra[event.x,event.y]=thevalue[event.x,event.y]
Update_Sp_Cal_display,event.id
end
;
;-----------------------------------------------------------------
;
pro Do_Edit_Cal_Frames, Event
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
widget_control,event.id,get_value=thevalue
cal_frames[event.y]=thevalue[event.x,event.y]
end
;
;-----------------------------------------------------------------
;
pro Update_Sp_Cal_display,wwidget
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
Sp_Cal_Table_ID=widget_info(wWidget,FIND_BY_UNAME='WID_SP_Cal_Table')
sp_size=sp_d[0]+sp_d[1]+1
Max_sp_num=(size(cal_spectra))[1]
if Max_sp_num eq 1 then begin
	labels=strarr(1)
	labels[0]=string(indgen(Max_sp_num),FORMAT='("Sp_",I1)')
endif else labels=string(indgen(Max_sp_num),FORMAT='("Sp_",I1)')
widget_control,Sp_Cal_Table_ID,TABLE_XSIZE = Max_sp_num, TABLE_YSIZE = sp_size,  COLUMN_LABELS=labels
widget_control, set_table_select=[0,0,(Max_sp_num-1),(sp_size-1)], /use_table_select
widget_control,Sp_Cal_Table_ID,set_value=(cal_spectra);, use_table_select=[0,0,(Max_sp_num-1),(sp_size-1)]
wset, sp_window
sp_size=sp_d[0]+sp_d[1]+1
x=indgen(sp_size)
plot,x,cal_spectra[0,*], yrange=[(min(cal_spectra)*0.9+max(cal_spectra)*0.1),(max(cal_spectra)*0.9-min(cal_spectra)*0.1)], title='Calibration Spectra',xtitle='CCD pixels',ytitle='Amplitude'
for i=1,(Max_sp_num-1)do oplot,x,cal_spectra[i,*], color=(255-40*i)
wset, def_w
end
;
;-----------------------------------------------------------------
;
pro Create_Single_Sp_Cal, Event
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
;DisplayType set to 0 - min display while fitting 1 - some display, 2 - full display,  3 - Cluster, 4 - GPU
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
										;use SigmaSym as a flag to indicate xsigma and ysigma are not independent and locked together in the fit

WID_DROPLIST_FitDisplayType_Spectrum_ID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_FitDisplayType_Spectrum')
DisplayType=widget_info(WID_DROPLIST_FitDisplayType_Spectrum_ID,/DropList_Select)	;set to 0 - min display while fitting 1 - some display 2 - full display  3 - Cluster  4- GPU

WID_TEXT_RemoveNumber_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_CalSpNum')
widget_control,WID_TEXT_RemoveNumber_ID,GET_VALUE = Remove_Index

WID_Button_TransformFirst = Widget_Info(Event.Top, find_by_uname='WID_BUTTON_TransformFirstForCal')
TransformFirst=widget_info(WID_Button_TransformFirst,/button_set)

if TransformFirst then begin
	print,'2D X-Y+color: Started Data Transformation and Saving'
	TransformRaw_Save, (lab_filenames[1]+'.dat'), RawFilenames[1], GuideStarDrift[1], FiducialCoeff[1], FlipRotate[1]
	print,'2D X-Y+color: Finished Data Transformation and Saving'
endif

ReadThisFitCond, (lab_filenames[1]+'.txt'), pth, filen, ini_filename, thisfitcond
sp_size=sp_d[0]+sp_d[1]+1
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz													;number of y pixels
sp_tot=dblarr(sp_size)
xy_sz=sqrt(float(xsz)*float(ysz))

if Remove_Index gt (Max_sp_num-1) then begin
	cal_spectra_old = cal_spectra
	cal_spectra = dblarr(Remove_Index+1,sp_size)
	cal_spectra[0:(Max_sp_num-1),*] = cal_spectra_old
	Max_sp_num = Remove_Index + 1
endif
sc_mag=1
trash=''
if (DisplayType eq 0) then begin
	trash=string(framefirst)+'#'
	xyouts,0.85,0.92,trash,/normal		;0.85,0.92
	xyouts,0.05,0.02,trash,/normal		;0.85,0.92
endif

SVDC, cal_spectra, Wsp, Usp, Vsp
N = N_ELEMENTS(Wsp)
WPsp = FLTARR(N, N)
FOR K = 0, N-1 DO   IF ABS(Wsp(K)) GE 1.0e-5 THEN WPsp(K, K) = 1.0/Wsp(K)

trash=''
											;loop for all file chunks
	framefirst=	cal_frames[0]
	framelast = cal_frames[1]
	Nframes=long(framelast-framefirst+1L) 								;number of frames to extract in file

	data=ReadData(lab_filenames[1],thisfitcond,framefirst,Nframes)	;Reads thefile and returns data (bunch of frames) in (units of photons)
	;	if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data
	for frameindx=0l,(Nframes-1l) do begin
			;print,'Frame #',frameindx
			if (DisplayType ge 1) and ((frameindx mod 100) eq 0) then DisplaySet=2 else DisplaySet=DisplayType
			if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data ;clip from transformed dat, frame with correct label
			clipsize=size(clip)
			if DisplaySet gt 1 then begin
				mg=2*(1 + (xsz lt 250))
				ofs=(min(clip) - 50) > 0
				tempclip=clip-ofs
				tempclip=200.0/max(tempclip)*temporary(tempclip)
				newframe=rebin(tempclip,xsz*mg,ysz*mg,/sample)
				ShowIt,newframe,mag=mg,wait=0.0
			endif
            process_display=(DisplaySet gt 1) ; or (DisplaySet ge 1)*((frameindx mod 20) eq 1)
			if process_display then begin
				xyouts,0.85,0.92,trash,/normal,col=0
				xyouts,0.05,0.02,trash,/normal,col=0
				trash=string(frameindx+framefirst)+'#'
				xyouts,0.85,0.92,trash,/normal		;0.85,0.92
				scl=300.0/max(clip)
				tv,scl*clip<255							;intensity scaling Range = scl* # electrons
				xyouts,0.05,0.02,trash,/normal		;0.85,0.92
			endif

			peaks_in_frame=where((CGroupParams[9,*] eq (frameindx+framefirst)) and (filter eq 1),num_peaks)
			;print, num_peaks
			if num_peaks ge 1 then begin
				peakx_array=fix(CGroupParams[2,peaks_in_frame])							;fitted x centers
				peaky_array=fix(CGroupParams[3,peaks_in_frame])							;fitted y centers
				SubtractBacground,clip,peakx_array,peaky_array,num_peaks,sp_d
				tempclip1=clip*300.0/max(clip)
				for ii=0L,num_peaks-1L do begin
					Dispxy=[ii,0]
					if process_display then begin
						xi=((peakx_array[ii]-sp_d[0]-2)>0)
						xa=((peakx_array[ii]+sp_d[1]+2)<(clipsize[1])-1)
						yi=((peaky_array[ii]-sp_d[2]-2)>0)
						tempclip1[xi:xa,yi]=255.
						tv,tempclip1<255.										;show it
					endif
					pr_disp=process_display*2
					ExtractSpectrum, clip, sp_d, peakx_array[ii], peaky_array[ii], spectrum, pr_disp, Dispxy, sc_mag			;
					sp_tot+=spectrum
				endfor
			endif
			if process_display then wait,1
		endfor
cal_spectra[Remove_Index,*]=sp_tot/max(sp_tot)>0.0
Update_Sp_Cal_display,Event.top
end
;
;-----------------------------------------------------------------
;
pro StartTransform, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }

WID_DROPLIST_FitDisplayType_Spectrum_ID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_FitDisplayType_Spectrum')
DisplayType=widget_info(WID_DROPLIST_FitDisplayType_Spectrum_ID,/DropList_Select)	;set to 0 - min display while fitting 1 - some display 2 - full display  3 - Cluster  4- GPU
TransformEngine = (DisplayType eq 3) ? 1 : 0
print,'2D X-Y+color: Started Data Transformation and Saving'
TransformRaw_Save, (lab_filenames[1]+'.dat'), RawFilenames[1], GuideStarDrift[1], FiducialCoeff[1], FlipRotate[1]
print,'2D X-Y+color: Finished Data Transformation and Saving'
non_spectral_peaks=where((CGroupParams[26,*] eq 1),num_peaks)
CGroupParams=CGroupParams[*,non_spectral_peaks]
CGroupParams[26,*]=0
ReloadParamlists, Event1
OnUnZoomButton, Event1
;widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro Start_Process_2DSpectrum, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }

WID_DROPLIST_FitDisplayType_Spectrum_ID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_FitDisplayType_Spectrum')
DisplayType=widget_info(WID_DROPLIST_FitDisplayType_Spectrum_ID,/DropList_Select)	;set to 0 - min display while fitting 1 - some display 2 - full display  3 - Cluster  4- GPU
TransformEngine = (DisplayType eq 3) ? 1 : 0
print,'2D X-Y+color: Starting Spectral Data Analysis'
ReadRawLoop2DSpectrum, DisplayType
ReloadParamlists, Event1
print,'2D X-Y+color: Finished Spectral Data Analysis'
;widget_control,event.top,/destroy
end
;
;------------------------------------------------------------------------------------
;
pro ReadRawLoop2DSpectrum, DisplayType			;Master program to read data and loop through processing
;DisplayType set to 0 - min display while fitting 1 - some display, 2 - full display,  3 - Cluster, 4 - GPU
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
										;use SigmaSym as a flag to indicate xsigma and ysigma are not independent and locked together in the fit
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames

ReadThisFitCond, (lab_filenames[1]+'.txt'), pth, filen, ini_filename, thisfitcond
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz													;number of y pixels
sp_tot=dblarr(sp_d[0]+sp_d[1]+1)
xy_sz=sqrt(float(xsz)*float(ysz))
framefirst=thisfitcond.Frm0
min_frames_per_node=fix(max((thisfitcond.FrmN-thisfitcond.Frm0)/500.00))>50
increment = long(fix(500*(256.0/xy_sz)))>long(min_frames_per_node)
if thisfitcond.FrmN le 500 then increment=thisfitcond.FrmN-thisfitcond.Frm0+1

n_cluster_nodes_max = 128
nloops = Fix((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/increment) < n_cluster_nodes_max			;nloops=Fix((framelast-framefirst)/increment)
;don't allow to use more then n_cluster_nodes_max cluster cores
increment = long(floor((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/nloops))
nloops = fix(ceil((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/increment))

trash=''
if (DisplayType eq 0) then begin
	trash=string(framefirst)+'#'
	xyouts,0.85,0.92,trash,/normal		;0.85,0.92
	xyouts,0.05,0.02,trash,/normal		;0.85,0.92
endif

sc_mag=1

SVDC, cal_spectra, Wsp, Usp, Vsp
N = N_ELEMENTS(Wsp)
WPsp = FLTARR(N, N)
FOR K = 0, N-1 DO   IF ABS(Wsp(K)) GE 1.0e-5 THEN WPsp(K, K) = 1.0/Wsp(K)

trash=''
for nlps=0L,nloops-1 do begin											;loop for all file chunks
	framefirst=	thisfitcond.Frm0 + (nlps)*increment					;first frame in batch
	framelast=((thisfitcond.Frm0 + (nlps+1L)*increment-1L)<thisfitcond.Nframesmax) < thisfitcond.FrmN
	Nframes=long(framelast-framefirst+1L) 								;number of frames to extract in file
	print,'Loop #',nlps
	data=ReadData(lab_filenames[1],thisfitcond,framefirst,Nframes)	;Reads thefile and returns data (bunch of frames) in (units of photons)
	;	if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data
	for frameindx=0l,(Nframes-1l) do begin

			if (DisplayType ge 1) and ((frameindx mod 100) eq 0) then DisplaySet=2 else DisplaySet=DisplayType
			if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data ;clip from transformed dat, frame with correct label
			clipsize=size(clip)
			if DisplaySet gt 1 then begin
				mg=2*(1 + (xsz lt 250))
				ofs=(min(clip) - 50) > 0
				tempclip=clip-ofs
				tempclip=200.0/max(tempclip)*temporary(tempclip)
				newframe=rebin(tempclip,xsz*mg,ysz*mg,/sample)
				ShowIt,newframe,mag=mg,wait=0.0
			endif
            process_display=(DisplaySet gt 1) ; or (DisplaySet ge 1)*((frameindx mod 20) eq 1)
			if process_display then begin
				xyouts,0.85,0.92,trash,/normal,col=0
				xyouts,0.05,0.02,trash,/normal,col=0
				trash=string(frameindx+framefirst)+'#'
				xyouts,0.85,0.92,trash,/normal		;0.85,0.92
				scl=300.0/max(clip)
				tv,scl*clip<255							;intensity scaling Range = scl* # electrons
				xyouts,0.05,0.02,trash,/normal		;0.85,0.92
				print, "Frame	Index	Coefficients'
			endif

			peaks_in_frame=where((CGroupParams[9,*] eq (frameindx+framefirst))and (filter eq 1),num_peaks)
			;if num_peaks eq 0 then stop
			if num_peaks ge 1 then begin
				peakx_array=fix(CGroupParams[2,peaks_in_frame])							;fitted x centers
				peaky_array=fix(CGroupParams[3,peaks_in_frame])							;fitted y centers
				SubtractBacground,clip,peakx_array,peaky_array,num_peaks,sp_d
				tempclip1=clip*300.0/max(clip)
				for ii=0L,num_peaks-1L do begin
					Dispxy=[ii,0]
					if process_display then begin
						xi=((peakx_array[ii]-sp_d[0]-2)>0)
						xa=((peakx_array[ii]+sp_d[1]+2)<(clipsize[1])-1)
						yi=((peaky_array[ii]-sp_d[2]-2)>0)
						tempclip1[xi:xa,yi]=255.
						tv,tempclip1<255.					;show it
					endif
					ExtractSpectrum, clip, sp_d, peakx_array[ii], peaky_array[ii], spectrum, process_display, Dispxy, sc_mag				;
					sp_tot+=spectrum
					AnalyzeSpectrum, spectrum, Vsp,WPsp,Usp, process_display, Dispxy, sc_mag, coeffs, resid
					;coeffs = Vsp ## WPsp ## TRANSPOSE(Usp) ## transpose(spectrum)
					if process_display then print, frameindx, ii, resid, transpose(coeffs)
					CGroupParams[18,peaks_in_frame[ii]]=resid
					CGroupParams[19:(18+Max_sp_num),peaks_in_frame[ii]]=coeffs
				endfor
			endif
			if process_display then wait,1
		endfor
endfor
end
;
;------------------------------------------------------------------------------------
;
pro SubtractBacground,clip,peakx_array,peaky_array,num_peaks,sp_d
bckg_clip=clip
clipsize=size(clip)
for jj=0L,num_peaks-1L do begin
	xi=((peakx_array[jj]-sp_d[0]-2)>0)
	xa=((peakx_array[jj]+sp_d[1]+2)<(clipsize[1])-1)
	yi=((peaky_array[jj]-sp_d[2]-2)>0)
	ya=((peaky_array[jj]+sp_d[2]+2)<(clipsize[2])-1)
	;print,xi,xa,yi,ya
	bckg_clip[xi:xa,yi:ya] = 0.0

	xii=((peakx_array[jj]-sp_d[0]-4)>0)
	xaa=((peakx_array[jj]+sp_d[1]+4)<(clipsize[1])-1)
	yii=((peaky_array[jj]-sp_d[2]-4)>0)
	yaa=((peaky_array[jj]+sp_d[2]+4)<(clipsize[2])-1)

	Z0 = bckg_clip[xii:xaa,yii:yaa]
	nz_ind=where(Z0 gt 1,cnt)
	Zc = Z0[nz_ind]
	if cnt gt 0 then begin
		if cnt lt total(sp_d) then begin
			bckg_clip[xi:xa,yi:ya]=mean(Zc)
		endif else begin
			XY_arr=ARRAY_INDICES(Z0,nz_ind)
			xc=XY_arr[0,*]
			yc=XY_arr[1,*]
			A=[[total(Xc*Xc),total(Xc*Yc),total(Xc)],[total(Xc*Yc),total(Yc*Yc),total(Yc)],[total(Xc),total(Yc),n_elements(Xc)]]
			B=[total(Xc*Zc),total(Yc*Zc),total(Zc)]
			LUDC, A, INDEX
			Plane_coeff = LUSOL(A, INDEX, B)
			ind_x = (indgen(xa-xi+1)+(xaa-xa))#((indgen(ya-yi+1)+1)/(indgen(ya-yi+1)+1))
			ind_y = ((indgen(xa-xi+1)+1)/(indgen(xa-xi+1)+1)) # indgen(ya-yi+1)+(yaa-ya)
			bckg_clip[xi:xa,yi:ya]= ind_x* Plane_coeff[0] + ind_y * Plane_coeff[1] + Plane_coeff[2]
		endelse
	endif
endfor
clip = (clip-bckg_clip)>0
end
;
;------------------------------------------------------------------------------------
;
pro ExtractSpectrum, clip, sp_d, peakx, peaky, spectrum, process_display, Dispxy, sc_mag	;
	clipsize=size(clip)
	spd_len=sp_d[0]+sp_d[1]+1
	spd_ht=sp_d[2]*2+1
	region=dblarr(spd_len,spd_ht)			;define the region in the spectral array to analyze the spectrum corresponding to the peak

	xic = (peakx-sp_d[0])>0									;starting left point of the region withn the clip
	xir	= (sp_d[0]-peakx)>0									;starting left point within the standalone region
	xac = (peakx+sp_d[1])<(clipsize[1]-1)								;last right point of the region withn the clip
	xar	= ((sp_d[0]+sp_d[1])-((peakx+sp_d[1]-clipsize[1]+1)>0))		;last right point within the standalone region

	yic = (peaky-sp_d[2])>0									;starting vert point of the region withn the clip
	yir	= (sp_d[2]-peaky)>0									;starting vert point within the standalone region
	yac = (peaky+sp_d[2])<(clipsize[2]-1)								;last vert point of the region withn the clip
	yar	= (2*sp_d[2]-((peaky+sp_d[2]-clipsize[2]+1)>0))		;last vert point within the standalone region

	region[xir:xar,yir:yar]=clip[xic:xac,yic:yac]

	spectrum = total(region,2)
	gscl=4.*sc_mag
	scl=0.5

	if process_display gt 0 then begin
		xtvpeak=(spd_len*gscl*dispxy[0] mod (fix(1024.0/spd_len/gscl)*spd_len*gscl))
		ytvpeak=512+(3*dispxy[1])*spd_ht*gscl
		tv,50+scl*rebin(region-min(region),spd_len*gscl,spd_ht*gscl,/sample)<255,xtvpeak,ytvpeak				;tv slected peak region
		if process_display gt 1 then begin
			position1=[(xtvpeak+1),(ytvpeak+spd_ht*gscl+3),(xtvpeak+spd_len*gscl-3),(ytvpeak+(spd_ht+spd_len)*gscl-1)]/1024.
			emptyblock=intarr(spd_len*gscl,spd_len*gscl)
			tv,emptyblock,xtvpeak,ytvpeak+spd_ht*gscl+1
			plot,spectrum, position=position1,xtickname=[' ',' ',' ',' ',' ',' ',' ',' ',' '],ytickname=[' ',' ',' ',' ',' ',' ',' ',' ',' '], /NOERASE
		endif
	endif
end
;
;------------------------------------------------------------------------------------
;
pro AnalyzeSpectrum, spectrum, Vsp,WPsp,Usp, process_display, Dispxy, sc_mag, coeffs, resid
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
	coeffs = abs(Vsp ## WPsp ## TRANSPOSE(Usp) ## transpose(spectrum))
	sum_fit=coeffs#cal_spectra
	ind=where(spectrum ne 0)
	resid=sqrt(total(((spectrum[ind]-sum_fit[ind])^2))/total(spectrum[ind]^2))
	if resid eq !VALUES.D_infinity or resid eq !VALUES.F_infinity then stop
	if process_display then begin
		gscl=4.*sc_mag
		scl=0.5
		spd_len=sp_d[0]+sp_d[1]+1
		spd_ht=sp_d[2]*2+1
		xtvpeak=(spd_len*gscl*dispxy[0] mod (fix(1024.0/spd_len/gscl)*spd_len*gscl))
		ytvpeak=512+(3*dispxy[1])*spd_ht*gscl

		position1=[(xtvpeak+1),(ytvpeak+spd_ht*gscl+3),(xtvpeak+spd_len*gscl-3),(ytvpeak+(spd_ht+spd_len)*gscl-1)]/1024.
		emptyblock=intarr(spd_len*gscl,spd_len*gscl)
		tv,emptyblock,xtvpeak,ytvpeak+spd_ht*gscl+1
		plot,spectrum, position=position1,xtickname=[' ',' ',' ',' ',' ',' ',' ',' ',' '],ytickname=[' ',' ',' ',' ',' ',' ',' ',' ',' '], /NOERASE
		oplot,sum_fit, thick=2, color=150

		position=[(xtvpeak+1),(ytvpeak+(spd_ht+spd_len)*gscl+8),(xtvpeak+spd_len*gscl-3),(ytvpeak+(spd_ht+spd_len*2)*gscl-1)]/1024.
		emptyblock=intarr(spd_len*gscl,spd_len*gscl+10)
		tv,emptyblock,xtvpeak,(ytvpeak+(spd_ht+spd_len)*gscl+1)
		plot,spectrum, psym=4, symsize=0.5, position=position,xtickname=[' ',' ',' ',' ',' ',' ',' ',' ',' '],ytickname=[' ',' ',' ',' ',' ',' ',' ',' ',' '], /NOERASE
		for i=0,(Max_sp_num-1)do oplot,coeffs[i]*cal_spectra[i,*], thick=2, color=(255-40*i)
		wait,0.1
	endif
end
;
;-----------------------------------------------------------------
;
pro OnStopSpectralProcessing, Event

end
;
;-----------------------------------------------------------------
;
pro Plot_Spectral_Weigths_Distributions, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames

if (n_elements(CGroupParams) lt 1) then begin
	void = dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

nsize=size(cal_spectra)
if (nsize[0] eq 0) then begin
	void = dialog_message('Please load spectra file and process data')
	return      ; if data not loaded return
endif

nbns=50
mn = 0.0
mx = 1.0
binsize=(mx-mn)/(nbns-1.0)
xx=fltarr(2*nbns)
histhist=fltarr(2*nbns)
evens=2*indgen(nbns)
odds=evens+1
x=findgen(nbns)/nbns*(mx-mn)+mn
dx=0.5 * (mx-mn) / nbns
xx[evens]=x-dx
xx[odds]=x+dx

n_sp=nsize[1]

ind_pks=where(filter,pk_cnt)
sum=total(CGroupParams[19:(18+n_sp),ind_pks],1)
weights=dblarr(n_sp,pk_cnt)
hist_count=intarr(n_sp)
lbl_color = intarr(n_sp)

for il=0,(n_sp-1) do begin
	lbl_color[il]=255-40*(il)
	weights[il,*]=CGroupParams[(19+il),ind_pks]/sum
	hist=histogram(weights[il,*],min=mn,max=mx,nbins=nbns)
	histhist[evens]=hist
	histhist[odds]=hist
	if il eq 0 then histhist_multilable=transpose(histhist) else histhist_multilable = [histhist_multilable, transpose(histhist)]
	hist_count[il]=max(histhist)
	xcoord=xx
endfor

yrange_hist=[0, max(hist_count)*1.1]
xrange_hist = [mn,mx]
;stop
;device,decompose=0
;loadct,12
tk=1.5
plot,xx,histhist_multilable[0,*],xstyle=1, xtitle='Relative Weight', ytitle='Molecule Count', $
		thick=1.0, xthick=1.0, ythick=1.0, charsize=tk, charthick=tk, xrange = xrange_hist, yrange = yrange_hist
oplot,xx,histhist_multilable[0,*], color=lbl_color[0]
oplot,xx,histhist_multilable[1,*], color=lbl_color[1]
oplot,xx,histhist_multilable[2,*], color=lbl_color[2]
oplot,xx,histhist_multilable[3,*], color=lbl_color[3]

;device,decompose=0
;loadct,3

;bmp_filename=AddExtension(dfilename,'_fiducial_coloc.bmp')
;presentimage=tvrd(true=1)
;write_bmp,bmp_filename,presentimage,/rgb

end
