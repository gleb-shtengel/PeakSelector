;
; Empty stub procedure used for autoloading.
;
pro Analyze_Plot_Save_Spectra_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_Analyze_Plot_Save_Spectra, wWidget

common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event={ WIDGET, ID:wWidget, TOP:wWidget, HANDLER:TopID }
Sp_Cal_Table_ID=widget_info(wWidget,FIND_BY_UNAME='WID_SP_Cal_Table')
sp_win=!D.window
if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	OnCancel      ; if data not loaded return
endif

BG_top=round(ParamLimits[3,1])
BG_bot=round(ParamLimits[3,0])
SP_top=BG_top-2
SP_bot=BG_bot+2
SP_top=(SP_top > SP_bot) < BG_top
SP_bot=(SP_bot < SP_top) > BG_bot
BG_subtr_params = [BG_top,SP_top,SP_bot,BG_bot,1]
WID_BG_Subtr_Params_ID = Widget_Info(wWidget, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]

SP_left=round(ParamLimits[2,0])
WidFrameNumber_top = Widget_Info(TopID, find_by_uname='WID_SLIDER_RawFrameNumber')
widget_control,WidFrameNumber_top,get_value=RawFrameNumber
fr_min_max=widget_info(WidFrameNumber_top,/SLIDER_MIN_MAX)

peak_indecis=where(filter,cnt)
if cnt lt 1 then begin
	z=dialog_message('No peaks selected')
	OnCancel      ; if data not loaded return
endif

Frame_Indecis = uniq(CGroupParams[9,peak_indecis])
Peak_Indecis=peak_indecis[frame_indecis]
npks=n_elements(peak_indecis)

if npks lt 1 then begin
	z=dialog_message('No peaks selected')
	OnCancel      ; if data not loaded return
endif

WidFrameNumber = Widget_Info(wWidget, find_by_uname='WID_SLIDER_RawFrameNumber_Spectral')
widget_control,WidFrameNumber,set_value=RawFrameNumber,SET_SLIDER_MIN=fr_min_max[0],SET_SLIDER_MAX=fr_min_max[1]

id=where(Frame_Indecis eq RawFrameNumber,cn_id)
RawPeakIndex=0
if cn_id then RawPeakIndex=Peak_Indecis[id]
WidPeakIndex = Widget_Info(wWidget, find_by_uname='WID_SLIDER_RawPeak_Index_Spectral')
widget_control,WidPeakIndex,set_value=RawFrameNumber,SET_SLIDER_MIN=0,SET_SLIDER_MAX=(npks-1)


Last_SAV_filename_label_ID = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,Last_SAV_filename_label_ID,get_value=Last_SAV_filename
IDL_pos=strpos(Last_SAV_filename,'_IDL')
if IDL_pos gt 0 then Last_SAV_filename_s = strmid(Last_SAV_filename,0,IDL_pos) else $
Last_SAV_filename_s = strmid(Last_SAV_filename,0,strlen(Last_SAV_filename)-4)
sp_filename = Last_SAV_filename_s+'_spectrum_fr'+strtrim(RawFrameNumber,2)+'_x'+strtrim(SP_left,2)+'_y'+strtrim(BG_bot,2)
SpFilename_WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_SpFilename')
widget_control,SpFilename_WidID,SET_VALUE = sp_filename
if n_elements(sp_calc_method) eq 0 then sp_calc_method=0
WidID_Sp_Calc_Meth = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Spectrum_Calc')
widget_control,WidID_Sp_Calc_Meth,SET_DROPLIST_SELECT=sp_calc_method	; 0 - Frame, with BG subtraction
																		; 1 - Frame, no BG subtraction
																		; 2 - Total, with BG subtraction


																		; 3 - Total, no BG subtraction
Reload_SP_data,event
Calculate_Plot_Spectrum, Event

end
;
;-----------------------------------------------------------------
;
pro OnSaveSpectrum, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename

if sp_filename eq '' then begin
	sp_filename = Dialog_Pickfile(/write,get_path=fpath)
	if strlen(fpath) ne 0 then cd,fpath
endif
if sp_filename eq '' then return

presentimage=reverse(tvrd(true=1),3)
TIFF_filename=AddExtension(sp_filename,'.tiff')
write_tiff,TIFF_filename,presentimage,orientation=1


ASCII_filename=AddExtension(sp_filename,'.txt')
Title_String='Wavelength (nm)	Amplitude'

Calculate_Spectrum, wl_arr, sp_arr
openw,1,ASCII_filename,width=1024
printf,1,Title_String
printf,1,[transpose(wl_arr),transpose(sp_arr)],FORMAT='(E13.5,"'+string(9B)+'",E13.5)'
close,1

end
;
;-----------------------------------------------------------------
;
pro OnCancel, Event
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
	widget_control,event.top,/destroy
	wset,def_w
end
;
;-----------------------------------------------------------------
;
pro Do_Edit_BG_Subtraction_Params, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename

widget_control,event.id,get_value=thevalue
BG_subtr_params[event.y]=thevalue[event.x,event.y]
widget_control,event.id,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
if event.y lt 4 then Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro Change_Spectrum_Calc, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename

sp_calc_method = widget_info(event.id,/DropList_Select)	; 0 - Frame, with BG subtraction
														; 1 - Frame, no BG subtraction
														; 2 - Total, with BG subtraction
														; 3 - Total, no BG subtraction
print,'Set sp_calc_method=',sp_calc_method
Reload_SP_data,event
Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro OnRawFrameNumber_Spectral, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

RawFrameNumber=event.value
WidFrameNumber_top = Widget_Info(TopID, find_by_uname='WID_SLIDER_RawFrameNumber')
widget_control,WidFrameNumber_top,set_value=RawFrameNumber

id=where(Frame_Indecis eq RawFrameNumber,cn_id)
RawPeakIndex=0
if cn_id then RawPeakIndex=Peak_Indecis[id]
WidPeakIndex = Widget_Info(Event.top, find_by_uname='WID_SLIDER_RawPeak_Index_Spectral')
widget_control,WidPeakIndex,set_value=RawFrameNumber

Reload_SP_data,event
Calculate_Plot_Spectrum, Event

end
;
;-----------------------------------------------------------------
;
pro OnRawPeakIndex_Spectral, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

RawPeakIndex=event.value
RawFrameNumber=CGroupParams[9,Peak_Indecis[RawPeakIndex]]

WidFrameNumber_Spectral = Widget_Info(Event.top, find_by_uname='WID_SLIDER_RawFrameNumber_Spectral')
widget_control,WidFrameNumber_Spectral,set_value=RawFrameNumber

WidFrameNumber_top = Widget_Info(TopID, find_by_uname='WID_SLIDER_RawFrameNumber')
widget_control,WidFrameNumber_top,set_value=RawFrameNumber

Reload_SP_data,event
Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro Reload_SP_data,event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

if  sp_calc_method ge 2 then sp_2D_data=TotalRawData else begin		; 0 - Frame, with BG subtraction
																	; 1 - Frame, no BG subtraction
																	; 2 - Total, with BG subtraction
																	; 3 - Total, no BG subtraction

	WidFrameNumber = Widget_Info(TopID, find_by_uname='WID_SLIDER_RawFrameNumber')
	widget_control,WidFrameNumber,get_value=RawFrameNumber
	RawFileNameWidID = Widget_Info(TopID, find_by_uname='WID_DROPLIST_RawFileName')
	Raw_File_Index=(widget_info(RawFileNameWidID,/DropList_Select))[0]
	raw_file_extension = thisfitcond.filetype ? '.tif' : '.dat'
	if Raw_File_Index eq -1 then begin
		z=dialog_message('Raw data file not found  '+RawFilenames)
		return
	endif
	file_conf=file_info(AddExtension(RawFilenames[Raw_File_Index],raw_file_extension))
	if ~file_conf.exists then begin
		z=dialog_message('Raw data file not found:  '+(RawFilenames[Raw_File_Index]+raw_file_extension))
		return
	endif
	reffilename=AddExtension(RawFilenames[Raw_File_Index],'.txt')
	ReadThisFitCond, reffilename, pth, filen, ini_filename, thisfitcond
	;print,'loading frame #', RawFrameNumber
	clip=ReadData(RawFilenames[Raw_File_Index],thisfitcond,RawFrameNumber,1)
	sp_2D_data=clip
endelse

end
;
;-----------------------------------------------------------------
;
pro Calculate_Plot_Spectrum, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename

WidID_DRAW_Spectra = Widget_Info(event.top, find_by_uname='WID_DRAW_Spectra')
draw_info=Widget_Info(WidID_DRAW_Spectra,/GEOMETRY)
wxsz=draw_info.scr_xsize
wysz=draw_info.scr_xsize

dxmn = paramlimits[2,0]
dxmx = paramlimits[2,1]
dymn = BG_subtr_params[3]
dymx = BG_subtr_params[0]
loc=fltarr(wxsz,wysz)
mgw=((wxsz /(dxmx-dxmn+1))<(wysz /(dymx-dymn+1)))*(0.98-0.12)

newx=(dxmx-dxmn+1)*mgw
newy=(dymx-dymn+1)*mgw

Plot_Raw_Spectrum, Event

Calculate_Spectrum, wl_arr, sp_arr

!p.noerase=1

plot_pos=[0.12,(newy/wysz+0.08),0.98,0.99]
tk=2.0
plot,wl_arr,sp_arr,position=plot_pos, thick=1.0, xtitle='Wavelength (nm)', ytitle='Intensity', $
	xrange=[min(wl_arr),max(wl_arr)],xstyle=1,xthick=1.0, ythick=1.0, charsize=tk, charthick=tk
!p.noerase=0
end
;
;-----------------------------------------------------------------
;
pro Calculate_Spectrum, wl_arr, sp_arr
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename

BG_top=BG_subtr_params[0]
SP_top=BG_subtr_params[1]
SP_bot=BG_subtr_params[2]
BG_bot=BG_subtr_params[3]

tot_size=BG_top-BG_bot+1
sp_size=SP_top-SP_bot+1
bg_size=tot_size-sp_size
tot=dblarr(tot_size)+1.0
sp=dblarr(tot_size)+1.0
bg=(dblarr(tot_size)+1.0)*sp_size/bg_size
if (SP_bot-BG_bot-1) ge 0 then sp[0:(SP_bot-BG_bot-1)]=0
if (SP_top-BG_bot+1) le (BG_top-BG_bot) then sp[(SP_top-BG_bot+1):(BG_top-BG_bot)]=0
if (SP_bot-BG_bot) le (SP_top-BG_bot) then bg[(SP_bot-BG_bot):(SP_top-BG_bot)]=0

if (sp_calc_method eq 1) or (sp_calc_method eq 3) then sp_arr=spectra#tot else sp_arr=spectra#(sp-bg)
wl_arr=(indgen(n_elements(sp_arr))+paramlimits[2,0])*sp_dispersion+sp_offset

end
;
;-----------------------------------------------------------------
;
pro On_Save_All_Spectra, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
WidFrameNumber_top = Widget_Info(TopID, find_by_uname='WID_SLIDER_RawFrameNumber')
WidFrameNumber = Widget_Info(Event.top, find_by_uname='WID_SLIDER_RawFrameNumber_Spectral')
peak_indecis=where(filter)
npks=n_elements(peak_indecis)

if npks lt 1 then begin
	z=dialog_message('No peaks selected')
	OnCancel      ; if data not loaded return
endif

On_Plot_All_Spectra, Event, wl_arr, all_spectra
Last_SAV_filename_label_ID = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,Last_SAV_filename_label_ID,get_value=Last_SAV_filename
IDL_pos=strpos(Last_SAV_filename,'_IDL')
SP_left=round(ParamLimits[2,0])
BG_bot=BG_subtr_params[3]
if IDL_pos gt 0 then Last_SAV_filename_s = strmid(Last_SAV_filename,0,IDL_pos) else $
	Last_SAV_filename_s = strmid(Last_SAV_filename,0,strlen(Last_SAV_filename)-4)
all_frames_filename = Last_SAV_filename_s+'_selected_frames__x'+strtrim(SP_left,2)+'_y'+strtrim(BG_bot,2)

presentimage=reverse(tvrd(true=1),3)
TIFF_filename=AddExtension(all_frames_filename,'.tiff')
write_tiff,TIFF_filename,presentimage,orientation=1

ASCII_filename=AddExtension(all_frames_filename,'.txt')
all_data=[transpose(wl_arr),transpose(all_spectra)]
Title_String='Wavelength (nm)'
for i=0,(npks-1) do Title_String=Title_String+'	Frame'+strtrim(fix(CGroupParams[9,peak_indecis[i]]),2)
openw,1,ASCII_filename,width=1024
printf,1,Title_String
printf,1,all_data,FORMAT='('+strtrim(npks,2)+'(E13.5,"'+string(9B)+'"),E13.5)'
close,1


end
;
;-----------------------------------------------------------------
;
pro On_Plot_All_Spectra, Event, wl_arr, all_spectra
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

WidFrameNumber_top = Widget_Info(TopID, find_by_uname='WID_SLIDER_RawFrameNumber')
WidFrameNumber = Widget_Info(Event.top, find_by_uname='WID_SLIDER_RawFrameNumber_Spectral')


WidID_DRAW_Spectra = Widget_Info(event.top, find_by_uname='WID_DRAW_Spectra')
draw_info=Widget_Info(WidID_DRAW_Spectra,/GEOMETRY)
wxsz=draw_info.scr_xsize
wysz=draw_info.scr_xsize

dxmn = paramlimits[2,0]
dxmx = paramlimits[2,1]
dymn = BG_subtr_params[3]
dymx = BG_subtr_params[0]
loc=fltarr(wxsz,wysz)
mgw=((wxsz /(dxmx-dxmn+1))<(wysz /(dymx-dymn+1)))*(0.98-0.12)

newx=(dxmx-dxmn+1)*mgw
newy=(dymx-dymn+1)*mgw

npks=n_elements(Peak_Indecis)
sp_size=size(spectra)
all_spectra=dblarr(sp_size[1],npks)
col_arr=indgen(npks)*round(200/npks)

RawFrameNumber_ini=RawFrameNumber
sp_calc_method_ini=sp_calc_method

;switch to individual frame mode and process all peaks
if sp_calc_method eq 2 then sp_calc_method=0
if sp_calc_method eq 3 then sp_calc_method=1

plot_pos=[0.12,(newy/wysz+0.08),0.98,0.99]
tk=2.0

for i=0,(npks-1) do begin
	RawFrameNumber=CGroupParams[9,peak_indecis[i]]
	print,'processing frame #',RawFrameNumber
	widget_control,WidFrameNumber_top,set_value=RawFrameNumber
	widget_control,WidFrameNumber,set_value=RawFrameNumber
	Reload_SP_data,event
	Plot_Raw_Spectrum, Event
	Calculate_Spectrum, wl_arr, sp_arr
	if i eq 0 then sp_3D_data = sp_2D_data else sp_3D_data=sp_3D_data+sp_2D_data
	all_spectra[*,i]=sp_arr
	!p.noerase=1
	plot,wl_arr,sp_arr,position=plot_pos, thick=1.0, xtitle='Wavelength (nm)', ytitle='Intensity', $	; color=col_arr[i],
			xrange=[min(wl_arr),max(wl_arr)],xstyle=1,xthick=1.0, ythick=1.0, charsize=tk, charthick=tk
	wait,0.02
	!p.noerase=0
endfor

; plot avraged spectra

sp_2D_data=sp_3D_data/npks
Plot_Raw_Spectrum, Event
!p.noerase=1
ymin=min(all_spectra)
ymax=max(all_spectra)
sp_arr=all_spectra[*,0]
plot,wl_arr,sp_arr,position=plot_pos, thick=1.0, xtitle='Wavelength (nm)', ytitle='Intensity', $	; color=col_arr[i],
			xrange=[min(wl_arr),max(wl_arr)],yrange=[min(all_spectra),max(all_spectra)],xstyle=1,xthick=1.0, ythick=1.0, charsize=tk, charthick=tk

for i=0,(npks-1) do begin
	sp_arr=all_spectra[*,i]
	oplot,wl_arr,sp_arr,color=col_arr[i]
endfor
!p.noerase=0

;switch back to original mode
sp_calc_method=sp_calc_method_ini
RawFrameNumber=RawFrameNumber_ini
widget_control,WidFrameNumber_top,set_value=RawFrameNumber
widget_control,WidFrameNumber,set_value=RawFrameNumber

end
;
;-----------------------------------------------------------------
;
pro Plot_Raw_Spectrum, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w

WidID_DRAW_Spectra = Widget_Info(event.top, find_by_uname='WID_DRAW_Spectra')
draw_info=Widget_Info(WidID_DRAW_Spectra,/GEOMETRY)
wxsz=draw_info.scr_xsize
wysz=draw_info.scr_xsize

dxmn = paramlimits[2,0]
dxmx = paramlimits[2,1]
dymn = BG_subtr_params[3]
dymx = BG_subtr_params[0]
loc=fltarr(wxsz,wysz)
mgw=((wxsz /(dxmx-dxmn+1))<(wysz /(dymx-dymn+1)))*(0.98-0.12)

newx=(dxmx-dxmn+1)*mgw
newy=(dymx-dymn+1)*mgw

dxmn_f=floor(dxmn)>0
dymn_f=dymn
dxmx_f=(ceil(dxmx)+1)<(xydsz[0]-1)
dymx_f=dymx

dx=round((dxmx_f-dxmn_f+1)*mgw)
dy=round((dymx_f-dymn_f+1)*mgw)

if n_elements(sp_2D_data) le 1 then Reload_SP_data,event
spectra=sp_2D_data[dxmn_f:dxmx_f,dymn_f:dymx_f]
FnewImage=Congrid(spectra,dx,dy)
lim=size(FnewImage)

dxmn_l=(dxmn-dxmn_f)*mgw>0
dymn_l=(dymn-dymn_f)*mgw>0
dxmx_l=(dxmn_l+newx-1)<(lim[1]-1)
dymx_l=(dymn_l+newy-1)<(lim[2]-1)

Fimage=FnewImage[dxmn_l:dxmx_l,dymn_l:dymx_l]
sizef=size(fimage)

xp=round(-1*dxmn*mgw>0)
yp=round(-1*dymn*mgw>0)
if xp gt 0 then begin
	Fimage1=intarr((sizef[1]+xp),sizef[2])
	Fimage1[xp:(sizef[1]+xp-1),*]=Fimage
	Fimage=Fimage1
	sizef=size(fimage)
endif
if yp gt 0 then begin
	Fimage1=intarr(sizef[1],(sizef[2]+yp))
	Fimage1[*,yp:(sizef[2]+yp-1)]=Fimage
	Fimage=Fimage1
endif

sp_2D_image=Fimage

tv,bytarr(wxsz,wysz)
tvscl,sp_2D_image,0.12*wxsz,0.01*wysz
image=sp_2D_image				;tvrd(true=1)

;AdjustContrastnDisplay, Event

xc=transpose([dxmn,(dxmx+1)])
bg_bot=transpose([BG_subtr_params[3], BG_subtr_params[3]])
bg_top=transpose([BG_subtr_params[0], BG_subtr_params[0]])+1
Draw_single_line, Event, [xc,bg_top], 0	; Line_color : 0=RED, 1=GREEN, 2=BLUE
Draw_single_line, Event, [xc,bg_bot], 0
sp_bot=transpose([BG_subtr_params[2], BG_subtr_params[2]])
sp_top=transpose([BG_subtr_params[1], BG_subtr_params[1]])+1
Draw_single_line, Event, [xc,sp_top], 1
Draw_single_line, Event, [xc,sp_bot], 1

end
;
;-----------------------------------------------------------------
;
pro Draw_single_line, Event, Line_Coordinates, Line_color	; Line_color : 0=RED, 1=GREEN, 2=BLUE
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename

WidID_DRAW_Spectra = Widget_Info(event.top, find_by_uname='WID_DRAW_Spectra')
draw_info=Widget_Info(WidID_DRAW_Spectra,/GEOMETRY)
wxsz=draw_info.scr_xsize
wysz=draw_info.scr_xsize

dxmn = paramlimits[2,0]
dxmx = paramlimits[2,1]
dymn = BG_subtr_params[3]
dymx = BG_subtr_params[0]
loc=fltarr(wxsz,wysz)
;mgw=(wxsz /(dxmx-dxmn+1))<(wysz /(dymx-dymn+1))
mgw=((wxsz /(dxmx-dxmn+1))<(wysz /(dymx-dymn+1)))*(0.98-0.12)

TVLCT,R0,G0,B0,/GET
TVLCT, [[255], [0], [0]], 0
TVLCT, [[0], [255], [0]], 1
TVLCT, [[0], [0], [255]], 2

x_loc_rgb=mgw * (Line_Coordinates[0,*] - dxmn)+round(0.12*wxsz)
y_loc_rgb=mgw * (Line_Coordinates[1,*] - dymn)+round(0.01*wysz)

plots,x_loc_rgb,y_loc_rgb,/device,color=Line_color
TVLCT,R0,G0,B0
end
;
;-----------------------------------------------------------------
;
pro Set_def_window, wWidget
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
wset,def_w
end
;
;-----------------------------------------------------------------
;
pro BG_Top_Up, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
BG_subtr_params[0]=BG_subtr_params[0]+BG_subtr_params[4]
WID_BG_Subtr_Params_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro BG_Top_Down, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
BG_subtr_params[0]=BG_subtr_params[0]-BG_subtr_params[4]
BG_subtr_params[1]=BG_subtr_params[1]<BG_subtr_params[0]
BG_subtr_params[2]=BG_subtr_params[2]<BG_subtr_params[1]
BG_subtr_params[3]=BG_subtr_params[3]<BG_subtr_params[2]
WID_BG_Subtr_Params_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro Sp_Top_Up, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
BG_subtr_params[1]=BG_subtr_params[1]+BG_subtr_params[4]
BG_subtr_params[0]=BG_subtr_params[0]>BG_subtr_params[1]
WID_BG_Subtr_Params_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro Sp_Top_Down, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
BG_subtr_params[1]=BG_subtr_params[1]-BG_subtr_params[4]
BG_subtr_params[2]=BG_subtr_params[2]<BG_subtr_params[1]
BG_subtr_params[3]=BG_subtr_params[3]<BG_subtr_params[2]
WID_BG_Subtr_Params_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro BG_Bot_Up, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
BG_subtr_params[3]=BG_subtr_params[3]+BG_subtr_params[4]
BG_subtr_params[2]=BG_subtr_params[2]>BG_subtr_params[3]
BG_subtr_params[1]=BG_subtr_params[1]>BG_subtr_params[2]
BG_subtr_params[0]=BG_subtr_params[0]>BG_subtr_params[1]
WID_BG_Subtr_Params_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro BG_Bot_Down, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
BG_subtr_params[3]=BG_subtr_params[3]-BG_subtr_params[4]
WID_BG_Subtr_Params_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro Sp_Bot_Up, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
BG_subtr_params[2]=BG_subtr_params[2]+BG_subtr_params[4]
BG_subtr_params[1]=BG_subtr_params[1]>BG_subtr_params[2]
BG_subtr_params[0]=BG_subtr_params[0]>BG_subtr_params[1]
WID_BG_Subtr_Params_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro Sp_Bot_Down, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
BG_subtr_params[2]=BG_subtr_params[2]-BG_subtr_params[4]
BG_subtr_params[3]=BG_subtr_params[3]<BG_subtr_params[2]
WID_BG_Subtr_Params_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
Calculate_Plot_Spectrum, Event
end
;
;
;-----------------------------------------------------------------
;
pro All_Up, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
BG_subtr_params[0:3]=BG_subtr_params[0:3]+BG_subtr_params[4]
WID_BG_Subtr_Params_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
Calculate_Plot_Spectrum, Event
end
;
;-----------------------------------------------------------------
;
pro All_Down, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
BG_subtr_params[0:3]=BG_subtr_params[0:3]-BG_subtr_params[4]
WID_BG_Subtr_Params_ID = Widget_Info(Event.top, find_by_uname='WID_TABLE_BG_Subtr_Params')
widget_control,WID_BG_Subtr_Params_ID,set_value=transpose(BG_subtr_params), use_table_select=[0,0,0,4]
Calculate_Plot_Spectrum, Event
end
;-----------------------------------------------------------------
