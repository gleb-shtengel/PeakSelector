;
;-----------------------------------------------------------------
; Empty stub procedure used for autoloading.
;
pro Import_data_ASCII_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_Import_Data_ASCII, wWidget
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WID_Base0_Peakselector, ID:TopID, TOP:TopID, HANDLER:TopID }

	WID_ID_WID_DROPLIST_Import_ASCII_XY = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Import_ASCII_XY')
	widget_control,WID_ID_WID_DROPLIST_Import_ASCII_XY,SET_DROPLIST_SELECT = ImportASCII_units

	WidID_TEXT_ASCII_Import_Parameter_List = Widget_Info(wWidget, find_by_uname='WID_TEXT_ASCII_Import_Parameter_List')
	widget_control,WidID_TEXT_ASCII_Import_Parameter_List,SET_VALUE = string(ImportASCII_ParamList),/NO_NEWLINE,/EDITABLE

	WID_ID_TEXT_Import_ASCII_nm_per_pixel = Widget_Info(wWidget, find_by_uname='WID_TEXT_Import_ASCII_nm_per_pixel')
	nm_per_pixel_txt=string(ImportASCII_nm_per_pixel,FORMAT='(F8.2)')
	widget_control,WID_ID_TEXT_Import_ASCII_nm_per_pixel,SET_VALUE = nm_per_pixel_txt

end
;
;-----------------------------------------------------------------
;
pro OnPickASCIIFile, Event
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
;sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
sep=PATH_SEP()
ImportASCII_Filename = Dialog_Pickfile(/read,title='Pick ASCII File')
fpath = strmid(ImportASCII_Filename,0,(max(strsplit(ImportASCII_Filename,sep))-1))
if ImportASCII_Filename ne '' then cd,fpath
WidID_TEXT_Import_ASCII_Filename = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Import_ASCII_Filename')
widget_control,WidID_TEXT_Import_ASCII_Filename,SET_VALUE = ImportASCII_Filename
end
;
;-----------------------------------------------------------------
;
pro On_ASCII_Filename_change, Event
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
	WidID_TEXT_Import_ASCII_Filename = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Import_ASCII_Filename')
	widget_control,WidID_TEXT_Import_ASCII_Filename,GET_VALUE = ImportASCII_Filename
end
;
;-----------------------------------------------------------------
;
pro On_Import_ASCII_ParamList_change, Event
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
	WidID_TEXT_ASCII_Import_Parameter_List = Widget_Info(Event.Top, find_by_uname='WID_TEXT_ASCII_Import_Parameter_List')
	widget_control,WidID_TEXT_ASCII_Import_Parameter_List,GET_VALUE = ImportASCII_ParamString
	len=strlen(ImportASCII_ParamString)
	i=0 & j=0
	while i lt len-1 do begin
		chr=STRMID(ImportASCII_ParamString,i,1)
		;print,i,'  chr=',chr,' byte=' , byte(chr)
		if (byte(chr) ne 32) and  (byte(chr) ne 9) then begin
			ParamStr=chr
			while ((i lt len) and (byte(chr) ne 32) and  (byte(chr) ne 9)) do begin
				i++
				chr=STRMID(ImportASCII_ParamString,i,1)
				ParamStr=ParamStr+chr
			endwhile
			if j eq 0 then ImportASCII_ParamList = fix(strcompress(ParamStr)) else ImportASCII_ParamList = [ImportASCII_ParamList,fix(strcompress(ParamStr))]
			j++
		endif
		i++
	endwhile
end
;
;-----------------------------------------------------------------
;
pro On_Select_ImportASCII_units, Event
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
	WID_ID_WID_DROPLIST_Import_ASCII_XY = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Import_ASCII_XY')
	ImportASCII_units = widget_info(WID_ID_WID_DROPLIST_Import_ASCII_XY,/DropList_Select)
end
;
;-----------------------------------------------------------------
;
pro Change_Import_ASCII_nm_per_pixel, Event
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
	WID_ID_TEXT_Import_ASCII_nm_per_pixel = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Import_ASCII_nm_per_pixel')
	widget_control,WID_ID_TEXT_Import_ASCII_nm_per_pixel,GET_VALUE = nm_per_pixel_txt
	ImportASCII_nm_per_pixel=float(nm_per_pixel_txt[0])
end
;
;-----------------------------------------------------------------
;
pro Import_ASCII, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common calib, aa, wind_range, nmperframe, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common Zdisplay, Z_scale_multiplier, vbar_top
common Offset, PkWidth_offset
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WID_Base0_Peakselector, ID:TopID, TOP:TopID, HANDLER:TopID }

file_ascii_info=file_info(ImportASCII_Filename)
if ~file_ascii_info.exists then begin
	z=dialog_message('ASCII data file not found:  '+ImportASCII_Filename)
	return
endif

n_params=n_elements(ImportASCII_ParamList)
params_single=transpose(dblarr(n_params))
cnt=0uL

window,10,xsize=400,ysize=100,xpos=50,ypos=250,Title='Importing ASCII file'
window_txt='Started Reading ASCII file'
xyouts,0.1,0.5,window_txt,CHARSIZE=2.0,/NORMAL
wait,0.01

fraction_complete_last=0
skipfirstline_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_SkipFirstLine_ImportASCII')
skipfirstline=widget_info(skipfirstline_id,/button_set)
title=strarr(1)						;dummy title line
close,1
openr,1,ImportASCII_Filename
	if skipfirstline then readf,1,title
	point_lun,-1,Cur_pos0
	while (~ EOF(1)) do begin
		readf,1,params_single
		point_lun,-1,Cur_pos
		if cnt eq 0 then begin
			delta=cur_pos-cur_pos0-2
			n_pts=ulong(file_ascii_info.size/delta)+1
			params=fltarr(n_params,n_pts)
			params[*,0]=float(params_single)
		endif else params[*,cnt] = float(params_single)
		cnt+=1
		fraction_complete=float(Cur_pos)/file_ascii_info.size*100
		if	(fraction_complete-fraction_complete_last) gt 1 then begin
			fraction_complete_last=fraction_complete
			xyouts,0.1,0.5,window_txt,CHARSIZE=2.0,/NORMAL,col=0
	 		window_txt=strtrim(string(fraction_complete,FORMAT='(F10.1)'),2)+' % Completed  '
	 		xyouts,0.1,0.5,window_txt,CHARSIZE=2.0,/NORMAL
	 	endif
	endwhile
close,1
wdelete,10
wset,def_w
;params=transpose(params)

CGroupParams=MAKE_ARRAY(CGrpSize,cnt,/FLOAT,VALUE=1.0)
CGroupParams[26,*]=0.0
nm_per_pixel=ImportASCII_nm_per_pixel;

XY_related_inices=[2,3,4,5,14,15,16,17,19,20,21,22]		; indecis of CGroupParameters that are kept in CCD pixel units. If the user ASCII file has them in nm units, they need to be scaled appropriately
for j=0, (n_elements(ImportASCII_ParamList)-1) do begin
	if 	ImportASCII_ParamList[j] ne -1 then begin
		xx=where(ImportASCII_ParamList[j] eq XY_related_inices, is_XY_related)
		if ImportASCII_units and is_XY_related then CGroupParams[ImportASCII_ParamList[j],*]=params[j,0:(cnt-1)]/nm_per_pixel else CGroupParams[ImportASCII_ParamList[j],*]=params[j,0:(cnt-1)]
	endif
endfor

xmax=max(CGroupParams[2,*])
ymax=max(CGroupParams[3,*])
xydsz=[xmax,ymax]
TotalRawData=fltarr(xmax,ymax)
RawFilenames=ImportASCII_Filename
ReloadParamlists
OnUnZoomButton, Event1
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnCancel_Import_ASCII, Event
	widget_control,event.top,/destroy
end
