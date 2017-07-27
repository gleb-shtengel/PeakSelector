;
; IDL Event Callback Procedures
; EditPeakSelectorIni_eventcb
;
; Generated on:	12/01/2010 13:45.06
;
;-----------------------------------------------------------------
; Notify Realize Callback Procedure.
; Argument:
;   wWidget - ID number of specific widget.
;
;
;   Retrieve the IDs of other widgets in the widget hierarchy using
;       id=widget_info(Event.top, FIND_BY_UNAME=name)
;
;-----------------------------------------------------------------
;
;-----------------------------------------------------------------
;
; Empty stub procedure used for autoloading.
;
pro EditPeakSelectorIni_eventcb
end
;
;-----------------------------------------------------------------
;
pro DoRealize_PeakSelector_INI, wWidget
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir
; TransformEngine : 0=Local, 1=Cluster, TransformEngine, grouping_gap,grouping_radius100; TransformEngine : 0=Local, 1=Cluster
if ini_filename ne '' then begin
	INIFileWidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_INI_Filename')
	widget_control,INIFileWidID,SET_VALUE = ini_filename

	close,1
	openr, 1, ini_filename
	c=0b
	cc=32b
	while (~ EOF(1)) do begin
		readu,1,c
		cc=[[cc],c]
	endwhile
	PeakSelector_INI_text=strtrim(cc,2)
	close,1

	PeakSelector_INI_WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_PeakSelector_INI')
	widget_control,PeakSelector_INI_WidID,SET_VALUE = PeakSelector_INI_text
endif
end
;
;-----------------------------------------------------------------
;
pro Save_INI_File, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir
; TransformEngine : 0=Local, 1=Cluster, TransformEngine, grouping_gap,grouping_radius100; TransformEngine : 0=Local, 1=Cluster
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

INIFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_INI_Filename')
widget_control,INIFileWidID,GET_VALUE = ini_filename0
if ini_filename0 eq ''then return

PeakSelector_INI_WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_PeakSelector_INI')
widget_control,PeakSelector_INI_WidID,GET_VALUE = PeakSelector_INI_text

print,'Saving INI file'
close,1
openw, 1, ini_filename0
printf, 1, PeakSelector_INI_text, FORMAT='(A)'
close,1

;Parse_INI_Text, Event
ini_file = ini_filename
Initialize_Common_parameters, ini_file
ReloadMainTableColumns, TopID

end
;
;-----------------------------------------------------------------
;
pro OnPickINIFile, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir
; TransformEngine : 0=Local, 1=Cluster, TransformEngine, grouping_gap,grouping_radius100; TransformEngine : 0=Local, 1=Cluster
ini_filename0 = Dialog_Pickfile(/read,get_path=fpath,filter=['*.ini'],title='Select *.ini file to open')
if ini_filename0 ne '' then begin
	cd,fpath
	INIFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_INI_Filename')
	widget_control,INIFileWidID,SET_VALUE = ini_filename0
	ini_file_info=FILE_INFO(ini_filename0)
	if ini_file_info.exists then ini_filename=ini_filename0
endif
end
;
;-----------------------------------------------------------------
;
pro Load_INI_File, Event	; initialises menue tables, droplists, IDL starting directory, material and other parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common Offset, PkWidth_offset
common Zdisplay, Z_scale_multiplier, vbar_top
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir
; TransformEngine : 0=Local, 1=Cluster, TransformEngine, grouping_gap,grouping_radius100; TransformEngine : 0=Local, 1=Cluster
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

INIFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_INI_Filename')
widget_control,INIFileWidID,GET_VALUE = ini_filename0
if ini_filename0 eq ''then return
ini_file_info=FILE_INFO(ini_filename0)
if ~(ini_file_info.exists) then return

close,1
openr, 1, ini_filename0

c=0b
cc=32b
while (~ EOF(1)) do begin
	readu,1,c
	cc=[[cc],c]
endwhile
PeakSelector_INI_text=strtrim(cc,2)
close,1

PeakSelector_INI_WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_PeakSelector_INI')
widget_control,PeakSelector_INI_WidID,SET_VALUE = PeakSelector_INI_text

;Parse_INI_Text, Event
ini_file = ini_filename
Initialize_Common_parameters, ini_file
ReloadMainTableColumns, TopID

end
;
;-----------------------------------------------------------------
;
pro Parse_INI_Text, Event	; initialises menue tables, droplists, IDL starting directory, material and other parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common Offset, PkWidth_offset
common Zdisplay, Z_scale_multiplier, vbar_top
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir
; TransformEngine : 0=Local, 1=Cluster, TransformEngine, grouping_gap,grouping_radius100; TransformEngine : 0=Local, 1=Cluster

COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

PeakSelector_INI_WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_PeakSelector_INI')
widget_control,PeakSelector_INI_WidID,GET_VALUE = PeakSelector_INI_text

sz_ini=size(PeakSelector_INI_text)
nlines_ini=sz_ini[1]
if nlines_ini le 0 then return

CATCH, Error_status
line_i='start'
IF Error_status NE 0 THEN BEGIN
      PRINT, 'Incorrect INI format. Could not parse the settings from the line:   ',line_i
      PRINT, i,k,'   ',byte(line_i),'   ',line_i
	  CATCH, /CANCEL
	  return
ENDIF

i=0
k=0
while (i lt nlines_ini) and (strmid(line_i,0,7) ne 'RowLbls') do begin
	line_i=strjoin(PeakSelector_INI_text[i,*])
	if (strmid(line_i,0,7) ne 'RowLbls') and ((byte(line_i))[0] ne 0) then begin
		;print,i,'   ',(byte(line_i))[0],'   ',line_i
		x=execute(line_i)
	endif
	i++
endwhile

while (i lt nlines_ini) and (k lt CGrpSize) do begin
	line_i=strjoin(PeakSelector_INI_text[i,*])
	if (strmid(line_i,0,7) ne 'RowLbls') and ((byte(line_i))[0] ne 0b) and ((byte(line_i))[0] ne 13b) then begin
		;print,i,'   ',(byte(line_i))[0],'   ',line_i,'   ',(byte(line_i))
		pos13b=strpos(line_i,string(13b))
		if pos13b gt 0 then line_i=strmid(strtrim(line_i),0,pos13b)
		RowLbls = (k eq 0) ? line_i : [ RowLbls , line_i ]
	endif
	i++
	k++
endwhile

CATCH, /CANCEL

RowNames=RowLbls[0:(CGrpSize-1)]

wWidget=TopID
	WidTableID = Widget_Info(wWidget, find_by_uname='WID_TABLE_0')
	widget_control,WidTableID,ROW_LABELS=RowNames,TABLE_YSIZE=CGrpSize
	if !VERSION.OS_family eq 'unix' then widget_control,WidTableID,COLUMN_WIDTH=[120,105,105,105,105],use_table_select = [ -1, 0, 4, (CGrpSize-1) ]

	WID_TABLE_StartReadSkip_ID = Widget_Info(wWidget, find_by_uname='WID_TABLE_StartReadSkip')
	widget_control,WID_TABLE_StartReadSkip_ID,COLUMN_WIDTH=[1,70,70,70],use_table_select = [ -1, 0, 2, 1 ]

	WidDrXID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select=9
	WidDrYID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select=3
	WidDrZID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select=2

	WidDrFunct = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Function')
	widget_control,WidDrFunct, Set_Droplist_Select=1
	WidDrFilter = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Filter')
	widget_control,WidDrFilter, Set_Droplist_Select=1
	WidDrAccum = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Accumulate')
	widget_control,WidDrAccum, Set_Droplist_Select=1

	labelcontrast=intarr(3,4)			;stretch top, gamma, stretch bottom	rows x blank, red, green, blue columns
	for i =0, 3 do labelcontrast[*,i]=[50,50,0]
	;
	WidID_StartReadSkip = Widget_Info(wWidget, find_by_uname='WID_TABLE_StartReadSkip')
	widget_control,WidID_StartReadSkip,set_value=transpose([0,0,0])
	widget_control, WidID_StartReadSkip, /editable,/sensitive


XZ_swap_menue_ID = Widget_Info(wWidget, find_by_uname='W_MENU_56')
widget_control,XZ_swap_menue_ID,set_button=0,set_value='Swap X-Z (now X)'

YZ_swap_menue_ID = Widget_Info(wWidget, find_by_uname='W_MENU_57')
widget_control,YZ_swap_menue_ID,set_button=0,set_value='Swap Y-Z (now Y)'

Z_unwrap_swap_menue_ID = Widget_Info(wWidget, find_by_uname='W_MENU_65')
widget_control,Z_unwrap_swap_menue_ID,set_button=0,set_value='Swap Z with Unwrapped Z'

end
;
;-----------------------------------------------------------------
;
pro OnCancel_INI_Edit, Event
	widget_control,event.top,/destroy
end
