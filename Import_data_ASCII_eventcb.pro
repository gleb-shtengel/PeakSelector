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
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common Zdisplay, Z_scale_multiplier, vbar_top
common Offset, PkWidth_offset
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
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

Off_ind = min(where(RowNames eq 'Offset'))                                ; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))                            ; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))                            ; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))                            ; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))                        ; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))                        ; CGroupParametersGP[5,*] - Peak Y Gaussian Width

Nph_ind = min(where(RowNames eq '6 N Photons'))                            ; CGroupParametersGP[6,*] - Number of Photons in the Peak
Chi_ind = min(where(RowNames eq 'ChiSquared'))                            ; CGroupParametersGP[7,*] - Chi Squared
FitOK_ind = min(where(RowNames eq 'FitOK'))                                ; CGroupParametersGP[8,*] - Original FitOK
FrNum_ind = min(where(RowNames eq 'Frame Number'))                        ; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))                    ; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigAmp_ind = min(where(RowNames eq 'Sigma Amplitude'))
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))                ; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))                ; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))                    ; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))                    ; CGroupParametersGP[17,*] - y - sigma

Gr_ind = min(where(RowNames eq '18 Grouped Index'))                        ; CGroupParametersGP[18,*] - group #
GrX_ind = min(where(RowNames eq 'Group X Position'))                    ; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))                    ; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))                ; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))                ; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))                    ; CGroupParametersGP[23,*] - total Photons in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))                    ; CGroupParametersGP[24,*] - total peaks in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))                ; CGroupParametersGP[25,*] - Frame Index in the Group
LabelSet_ind = min(where(RowNames eq 'Label Set'))                        ; CGroupParametersGP[26,*] - Label Number
AmpL1_ind = min(where(RowNames eq 'Amplitude L1'))                        ; CGroupParametersGP[27,*] - Label 1 Amplitude
AmpL2_ind = min(where(RowNames eq 'Amplitude L2'))                        ; CGroupParametersGP[28,*] - Label 2 Amplitude
AmpL3_ind = min(where(RowNames eq 'Amplitude L3'))                        ; CGroupParametersGP[29,*] - Label 3 Amplitude
SigL1_ind = min(where(RowNames eq 'Zeta0'))                                ; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))                        ; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))                        ; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
Z_ind = min(where(RowNames eq 'Z Position'))                            ; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))                            ; CGroupParametersGP[35,*] - Sigma Z
Coh_ind = min(where(RowNames eq '36 Coherence'))                        ; CGroupParametersGP[36,*] - Coherence
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))                        ; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))                        ; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))                        ; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))                    ; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))                    ; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))                ; CGroupParametersGP[42,*] - Group Coherence
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))                        ; CGroupParametersGP[43,*] - XY Ellipticity
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))                        ; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))                ; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))                ; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))                ; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))        ; CGroupParametersGP[48,*] - Group Z Position Error

n_params=n_elements(ImportASCII_ParamList)
params_single=transpose(dblarr(n_params))
cnt=0uL

;********* Status Bar Initialization  ******************
oStatusBar = obj_new('PALM_StatusBar', $
		COLOR=[0,0,255], $
		TEXT_COLOR=[255,255,255], $
		CANCEL_BUTTON_PRESENT = 1, $
		TITLE='Reading ASCII data...', $
		TOP_LEVEL_BASE=tlb)
fraction_complete_last=0.0D
pr_bar_inc=0.01D
interrupt_load = 0
; ********************************************************

skipfirstline_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_SkipFirstLine_ImportASCII')
skipfirstline=widget_info(skipfirstline_id,/button_set)
title=strarr(1)						;dummy title line
close,1
openr,1,ImportASCII_Filename
	if skipfirstline then readf,1,title
	point_lun,-1,Cur_pos0
	while (~ EOF(1)) and (interrupt_load eq 0) do begin
		readf,1,params_single
		point_lun,-1,Cur_pos
		if cnt eq 0 then begin
			delta=cur_pos-cur_pos0-2.0
			n_pts=ulong(file_ascii_info.size/delta*1.1)+1   ; put some overhead on possible size
			params=fltarr(n_params,n_pts)
			params[*,0]=float(params_single)
		endif else if (cnt lt n_pts) then params[*,cnt] = float(params_single)

		cnt+=1

; *********** Status Bar Update (inside FOR loop) **********
        fraction_complete = float(Cur_pos)/file_ascii_info.size
        if (fraction_complete - fraction_complete_last) ge pr_bar_inc then begin
             oStatusBar -> UpdateStatus, fraction_complete
             fraction_complete_last = fraction_complete
        endif
        interrupt_load = oStatusBar -> CheckCancel()

	endwhile
obj_destroy, oStatusBar ;********* Status Bar Close (after FOR loop)******************
close,1

if interrupt_load eq 1 then begin
    print,'Reading aborted, cleaning up...'
    return
endif

cnt = min([cnt,n_pts])
;wdelete,10
wset,def_w
;params=transpose(params)

CGroupParams=MAKE_ARRAY(CGrpSize,cnt,/FLOAT,VALUE=1.0)
if where(ImportASCII_ParamList eq LabelSet_ind) eq -1 then CGroupParams[LabelSet_ind,*]=0.0 ; if imported set has only one label, set LabelSet_ind to 0
nm_per_pixel=ImportASCII_nm_per_pixel;

XY_related_inices=[	X_ind, $
					Y_ind, $
					Xwid_ind, $
					Ywid_ind, $
					SigNphX_ind, $
					SigNphY_ind, $
					SigX_ind, $
					SigY_ind, $
					GrX_ind, $
					GrY_ind, $
					GrSigX_ind, $
					GrSigY_ind]		; indecis of CGroupParameters that are kept in CCD pixel units. If the user ASCII file has them in nm units, they need to be scaled appropriately

for j=0, (n_elements(ImportASCII_ParamList)-1) do begin
	if 	ImportASCII_ParamList[j] ne -1 then begin
		xx=where(ImportASCII_ParamList[j] eq XY_related_inices, is_XY_related)
		if ImportASCII_units and is_XY_related then CGroupParams[ImportASCII_ParamList[j],*]=params[j,0:(cnt-1)]/nm_per_pixel $
												else CGroupParams[ImportASCII_ParamList[j],*]=params[j,0:(cnt-1)]
	endif
endfor

xmax=max(CGroupParams[X_ind,*])
ymax=max(CGroupParams[Y_ind,*])
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
