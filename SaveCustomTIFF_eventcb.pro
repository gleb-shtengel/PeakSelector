
; Empty stub procedure used for autoloading.
;
pro SaveCustomTIFF_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_Custom_TIFF, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $		; and their names
			modalList		; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }

Cust_TIFF_window=!D.window


Z_UnwZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_65')
Z_UnwZ_swaped=Widget_Info(Z_UnwZ_swap_menue_ID,/button_set)

X_ind = min(where(RowNames eq 'X Position'))                            ; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))                            ; CGroupParametersGP[3,*] - Peak Y  Position
Z_ind = min(where(RowNames eq 'Z Position'))                            ; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))                    ; CGroupParametersGP[40,*] - Group Z Position
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))                        ; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))                ; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)


Cust_TIFF_3D = 0
Cust_TIFF_volume_image = 0
Cust_TIFF_Z_multiplier = 1.0

Top_AccumId=widget_info(Event1.top, FIND_BY_UNAME='WID_DROPLIST_Accumulate')
Cust_TIFF_Accumulation=widget_info(Top_AccumId,/DropList_Select)
Cust_TIFF_AccumId=widget_info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Accumulate_cust_TIFF')
widget_control,Cust_TIFF_AccumId,SET_DROPLIST_SELECT=Cust_TIFF_Accumulation

Top_FunctionId=widget_info(Event1.top, FIND_BY_UNAME='WID_DROPLIST_Function')
Cust_TIFF_Function=widget_info(Top_FunctionId,/DropList_Select)
Cust_TIFF_FunctionId=widget_info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Function_cust_TIFF')
widget_control,Cust_TIFF_FunctionId,SET_DROPLIST_SELECT=Cust_TIFF_Function

Top_FilterId=widget_info(Event1.top, FIND_BY_UNAME='WID_DROPLIST_Filter')
Cust_TIFF_Filter=widget_info(Top_FilterId,/DropList_Select)
Cust_TIFF_FilterId=widget_info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Filter_cust_TIFF')
widget_control,Cust_TIFF_FilterId,SET_DROPLIST_SELECT=Cust_TIFF_Filter

WID_IMAGE_SCALING_Parameters_ID = Widget_Info(wWidget, find_by_uname='WID_IMAGE_SCALING_Parameters')
Cust_TIFF_Pix_X = ParamLimits[X_ind,1] - ParamLimits[X_ind,0]
Cust_TIFF_Pix_Y = ParamLimits[Y_ind,1] - ParamLimits[Y_ind,0]
cust_nm_per_pix = nm_per_pixel
params=[cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y]
widget_control,WID_IMAGE_SCALING_Parameters_ID,set_value=transpose(params), use_table_select=[0,0,0,2]

Wid_WID_DRAW_Custom_TIFF_IS = Widget_Info(wWidget, find_by_uname='WID_DRAW_Custom_TIFF')
widget_control,Wid_WID_DRAW_Custom_TIFF_IS, DRAW_XSIZE =Cust_TIFF_Pix_X,DRAW_YSIZE=Cust_TIFF_Pix_Y

WID_IMAGE_Zcoord_Parameters_ID = Widget_Info(wWidget, find_by_uname='WID_IMAGE_Zcoord_Parameters')
if Z_UnwZ_swaped then begin
	Cust_TIFF_Z_start=ParamLimits[UnwZ_ind,0]
	Cust_TIFF_Z_stop=ParamLimits[UnwZ_ind,1]
endif else begin
	Cust_TIFF_Z_start=ParamLimits[Z_ind,0]
	Cust_TIFF_Z_stop=ParamLimits[Z_ind,1]
endelse
Zstep=cust_nm_per_pix/Cust_TIFF_Z_multiplier
params=[Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Zstep, Cust_TIFF_Z_multiplier]
widget_control,WID_IMAGE_Zcoord_Parameters_ID,set_value=transpose(params), use_table_select=[0,0,0,3]

WID_TEXT_Z_subvolume_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Z_subvolume')
Cust_TIFF_Z_subvol_nm_txt=string(Cust_TIFF_Z_subvol_nm,FORMAT='(F8.2)')
widget_control,WID_TEXT_Z_subvolume_ID,SET_VALUE = Cust_TIFF_Z_subvol_nm_txt

WID_TEXT_XY_subvolume_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_XY_subvolume')
Cust_TIFF_XY_subvol_nm_txt=string(Cust_TIFF_XY_subvol_nm,FORMAT='(F8.2)')
widget_control,WID_TEXT_XY_subvolume_ID,SET_VALUE = Cust_TIFF_XY_subvol_nm_txt

WID_IMAGE_SCALING_Parameters_ID = Widget_Info(wWidget, find_by_uname='WID_IMAGE_SCALING_Parameters')
widget_control,WID_IMAGE_SCALING_Parameters_ID,COLUMN_WIDTH=[180,150],use_table_select = [ -1, 0, 0, 3 ]
WID_IMAGE_Zcoord_Parameters_ID = Widget_Info(wWidget, find_by_uname='WID_IMAGE_Zcoord_Parameters')
widget_control,WID_IMAGE_Zcoord_Parameters_ID,COLUMN_WIDTH=[180,150],use_table_select = [ -1, 0, 0, 4 ]

WidDL_LabelID = Widget_Info(TopID, find_by_uname='WID_DROPLIST_Label')
selectedlabel = widget_info(WidDL_LabelID,/DropList_Select)
WidDL_LabelID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Label_cust_TIFF')
widget_control,WidDL_LabelID,SET_DROPLIST_SELECT=SelectedLabel

WidSldTop_cust_TIFF_ID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Top_cust_TIFF')
widget_control,WidSldTop_cust_TIFF_ID,set_value=labelContrast[0,selectedlabel]

WidSldGamma_cust_TIFF_ID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Gamma_cust_TIFF')
widget_control,WidSldGamma_cust_TIFF_ID,set_value=labelContrast[1,selectedlabel]

WidSldBot_cust_TIFF_ID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Bot_cust_TIFF')
widget_control,WidSldBot_cust_TIFF_ID,set_value=labelContrast[2,selectedlabel]

wset,def_w
end
;
;-----------------------------------------------------------------
;
pro CustomTIFF_Draw_Realize, wWidget
device,decompose=0
loadct,3
end
;
;-----------------------------------------------------------------
;
pro Cust_TIFF_Select_Accumulation, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Cust_TIFF_AccumId=widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Accumulate_cust_TIFF')
Cust_TIFF_Accumulation=widget_info(Cust_TIFF_AccumId,/DropList_Select)
Top_AccumId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Accumulate')
widget_control,Top_AccumId,SET_DROPLIST_SELECT=Cust_TIFF_Accumulation
end
;
;-----------------------------------------------------------------
;
pro Cust_TIFF_Select_Filter, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Cust_TIFF_FilterId=widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Filter_cust_TIFF')
Cust_TIFF_Filter=widget_info(Cust_TIFF_FilterId,/DropList_Select)
Top_FilterId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Filter')
widget_control,Top_FilterId,SET_DROPLIST_SELECT=Cust_TIFF_Filter
end
;
;-----------------------------------------------------------------
;
pro Cust_TIFF_Select_Function, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Cust_TIFF_FunctionId=widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Function_cust_TIFF')
Cust_TIFF_Function=widget_info(Cust_TIFF_FunctionId,/DropList_Select)
Top_FunctionId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Function')
widget_control,Top_FunctionId,SET_DROPLIST_SELECT=Cust_TIFF_Function
end
;
;-----------------------------------------------------------------
;
pro OnLabelDropList_cust_TIFF, Event			;Change color scale sliders to match label
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
wset,Cust_TIFF_window
SelectedLabel=event.index
WidDL_LabelID = Widget_Info(TopID, find_by_uname='WID_DROPLIST_Label')
widget_control,WidDL_LabelID,SET_DROPLIST_SELECT=SelectedLabel

WidSldTop_cust_TIFF_ID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Top_cust_TIFF')
widget_control,WidSldTop_cust_TIFF_ID,set_value=labelContrast[0,selectedlabel]
WidSldTopID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Top')
widget_control,WidSldTopID,set_value=labelContrast[0,selectedlabel]

WidSldGamma_cust_TIFF_ID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Gamma_cust_TIFF')
widget_control,WidSldGamma_cust_TIFF_ID,set_value=labelContrast[1,selectedlabel]
WidSldGammaID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Gamma')
widget_control,WidSldGammaID,set_value=labelContrast[1,selectedlabel]

WidSldBot_cust_TIFF_ID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Bot_cust_TIFF')
widget_control,WidSldBot_cust_TIFF_ID,set_value=labelContrast[2,selectedlabel]
WidSldBotID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Bot')
widget_control,WidSldBotID,set_value=labelContrast[2,selectedlabel]
wset,def_w
end
;
;-----------------------------------------------------------------
;
pro OnStretchTop_cust_TIFF, Event				;Set max of color range
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
WidDL_Label_cust_TIFF_ID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Label_cust_TIFF')
selectedlabel = widget_info(WidDL_Label_cust_TIFF_ID,/DropList_Select)

;WID_BUTTON_Tie_RGB_ID = Widget_Info(TopID, find_by_uname='WID_BUTTON_Tie_RGB')
;Tie_RGB = Widget_Info(WID_BUTTON_Tie_RGB_ID, /BUTTON_SET)
WID_BUTTON_Tie_RGB_CustTIFF_ID = Widget_Info(Event.top, find_by_uname='WID_BUTTON_Tie_RGB_CustTIFF')
Tie_RGB_CustTIFF = Widget_Info(WID_BUTTON_Tie_RGB_CustTIFF_ID, /BUTTON_SET)
if Tie_RGB_CustTIFF then labelContrast[0,*]=Event.value else labelContrast[0,selectedlabel]=Event.value
;labelContrast[0,selectedlabel]=Event.value

wset,Cust_TIFF_window
widget_control, /HOURGLASS	;  Show the hourglass
if Cust_TIFF_3D eq 0 then begin	; 2D image, perfrom standard AdjustContrastnDisplay and adjust main Peakselector window slider
	WidDL_Label_ID = Widget_Info(TopID, find_by_uname='WID_DROPLIST_Label')
	widget_control,WidDL_Label_ID,SET_DROPLIST_SELECT=selectedlabel
	WidSldTopID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Top')
	widget_control,WidSldTopID,set_value=Event.value
	AdjustContrastnDisplay,Event
	endif else begin		; if 3D volume image is present, work on it.
	Display_Zslice, Event
endelse
wset,def_w
end
;
;-----------------------------------------------------------------
;
pro OnGamma_cust_TIFF, Event					;Set gamma of color range
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
WidDL_Label_cust_TIFF_ID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Label_cust_TIFF')
selectedlabel = widget_info(WidDL_Label_cust_TIFF_ID,/DropList_Select)
WID_BUTTON_Tie_RGB_CustTIFF_ID = Widget_Info(Event.top, find_by_uname='WID_BUTTON_Tie_RGB_CustTIFF')
Tie_RGB_CustTIFF = Widget_Info(WID_BUTTON_Tie_RGB_CustTIFF_ID, /BUTTON_SET)
if Tie_RGB_CustTIFF then labelContrast[1,*]=Event.value else labelContrast[1,selectedlabel]=Event.value
;labelContrast[1,selectedlabel]=Event.value
wset,Cust_TIFF_window
widget_control, /HOURGLASS	;  Show the hourglass
if Cust_TIFF_3D eq 0 then begin	; 2D image, perfrom standard AdjustContrastnDisplay and adjust main Peakselector window slider
	WidDL_Label_ID = Widget_Info(TopID, find_by_uname='WID_DROPLIST_Label')
	widget_control,WidDL_Label_ID,SET_DROPLIST_SELECT=selectedlabel
	WidSldGammaID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Gamma')
	widget_control,WidSldGammaID,set_value=Event.value
	AdjustContrastnDisplay, Event
endif else begin		; if 3D volume image is present, work on it.
	Display_Zslice, Event
endelse
wset,def_w
end
;
;-----------------------------------------------------------------
;
pro OnStretchBottom_cust_TIFF, Event			;Set min of color range
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
WidDL_Label_cust_TIFF_ID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Label_cust_TIFF')
selectedlabel = widget_info(WidDL_Label_cust_TIFF_ID,/DropList_Select)
WID_BUTTON_Tie_RGB_CustTIFF_ID = Widget_Info(Event.top, find_by_uname='WID_BUTTON_Tie_RGB_CustTIFF')
Tie_RGB_CustTIFF = Widget_Info(WID_BUTTON_Tie_RGB_CustTIFF_ID, /BUTTON_SET)
if Tie_RGB_CustTIFF then labelContrast[2,*]=Event.value else labelContrast[2,selectedlabel]=Event.value
;labelContrast[2,selectedlabel]=Event.value
wset,Cust_TIFF_window
widget_control, /HOURGLASS	;  Show the hourglass
if Cust_TIFF_3D eq 0 then begin	; 2D image, perfrom standard AdjustContrastnDisplay and adjust main Peakselector window slider
	WidDL_Label_ID = Widget_Info(TopID, find_by_uname='WID_DROPLIST_Label')
	widget_control,WidDL_Label_ID,SET_DROPLIST_SELECT=selectedlabel
	WidSldBotID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Bot')
	widget_control,WidSldBotID,set_value=Event.value
	AdjustContrastnDisplay, Event
	endif else begin		; if 3D volume image is present, work on it.
	Display_Zslice, Event
endelse
wset,def_w
end
;
;-----------------------------------------------------------------
;
pro DoInsert_Cust_TIFF_Scale_Param, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
widget_control,event.id,get_value=thevalue
CASE event.y OF
	0:	new_cust_nm_per_pix=thevalue[event.y] ; change nm per pixel in the image window
	1:	new_cust_nm_per_pix = cust_nm_per_pix * Cust_TIFF_Pix_X / thevalue[event.y]  ; change image pixel size: X
	2:	new_cust_nm_per_pix = cust_nm_per_pix * Cust_TIFF_Pix_Y / thevalue[event.y]  ; change image pixel size: Y
ENDCASE
Cust_TIFF_Pix_X=Cust_TIFF_Pix_X * cust_nm_per_pix / new_cust_nm_per_pix
Cust_TIFF_Pix_Y=Cust_TIFF_Pix_Y * cust_nm_per_pix / new_cust_nm_per_pix
cust_nm_per_pix=new_cust_nm_per_pix
params=[cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y]
widget_control,event.id,set_value=transpose(params), use_table_select=[0,0,0,2]
Wid_WID_DRAW_Custom_TIFF_IS = Widget_Info(Event.Top, find_by_uname='WID_DRAW_Custom_TIFF')
widget_control,Wid_WID_DRAW_Custom_TIFF_IS, DRAW_XSIZE =Cust_TIFF_Pix_X,DRAW_YSIZE=Cust_TIFF_Pix_Y

WID_IMAGE_Zcoord_Parameters_ID = Widget_Info(Event.Top, find_by_uname='WID_IMAGE_Zcoord_Parameters')
Zstep=cust_nm_per_pix/Cust_TIFF_Z_multiplier
params=[Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Zstep, Cust_TIFF_Z_multiplier]
widget_control,WID_IMAGE_Zcoord_Parameters_ID,set_value=transpose(params), use_table_select=[0,0,0,3]
;if !VERSION.OS_family ne 'Windows' then begin
;	wset,Cust_TIFF_window
;	wait,0.1
;	DEVICE, RETAIN=2
;	wset,def_w
;endif
end
;
;-----------------------------------------------------------------
;
pro DoInsert_Cust_TIFF_ZScale_Param, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
widget_control,event.id,get_value=thevalue
CASE event.y OF
	0:	Cust_TIFF_Z_start = thevalue[event.y] ; change Zstart
	1:	Cust_TIFF_Z_stop = thevalue[event.y]  ; change Zstop
	2:	Cust_TIFF_Z_multiplier = cust_nm_per_pix / thevalue[event.y]  ; change Zstep - results in a change of Z-X scale ratio (Z multiplier)
	3:	Cust_TIFF_Z_multiplier = thevalue[event.y]
ENDCASE
Zstep=cust_nm_per_pix/Cust_TIFF_Z_multiplier
params=[Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Zstep, Cust_TIFF_Z_multiplier]
widget_control,event.id,set_value=transpose(params), use_table_select=[0,0,0,3]
end
;
;-----------------------------------------------------------------
;
pro Change_XY_Subvolume, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm

		WID_TEXT_XY_subvolume_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_XY_subvolume')
		widget_control,WID_TEXT_XY_subvolume_ID,GET_VALUE = Cust_TIFF_XY_subvol_nm_txt
		Cust_TIFF_XY_subvol_nm=float(Cust_TIFF_XY_subvol_nm_txt[0])
end
;
;-----------------------------------------------------------------
;
pro Change_Z_Subvolume, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm

		WID_TEXT_Z_subvolume_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Z_subvolume')
		widget_control,WID_TEXT_Z_subvolume_ID,GET_VALUE = Cust_TIFF_Z_subvol_nm_txt
		Cust_TIFF_Z_subvol_nm=float(Cust_TIFF_Z_subvol_nm_txt[0])
end

;
;-----------------------------------------------------------------
;
pro Render_cust_TIFF, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
Cust_TIFF_3D=0
wset,Cust_TIFF_window
OnRenderButton, Event
wset,def_w
end
;
;-----------------------------------------------------------------
;
pro OnAddScaleBarButton_cust_TIFF, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
wset,Cust_TIFF_window
OnAddScaleBarButton, Event
wset,def_w
end
;
;-----------------------------------------------------------------
;
pro Save_cust_TIFF, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
wset,Cust_TIFF_window
SaveImageTIFF, Event
;if !VERSION.OS_family eq 'Windows' then SaveImageTIFF, Event else SaveImageTIFF_cust, Event
wset,def_w
end
;
;-----------------------------------------------------------------
;
pro Save_cust_TIFF_float, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
	filename = Dialog_Pickfile(/write,get_path=fpath)
	if strlen(fpath) ne 0 then cd,fpath
	if filename eq '' then return
	if (size(image))[0] eq 2 then begin
		filename=AddExtension(filename,'.tif')		; monochrome
		write_tiff,filename,reverse(image,2), /float
	endif else begin
		filename_r=AddExtension(filename,'_r.tif')		; monochrome
		write_tiff,filename_r,reverse(image[*,*,0],2), /float
		filename_g=AddExtension(filename,'_g.tif')		; monochrome
		write_tiff,filename_g,reverse(image[*,*,1],2), /float
		filename_b=AddExtension(filename,'_b.tif')		; monochrome
		write_tiff,filename_b,reverse(image[*,*,2],2), /float
	endelse
end
;
;-----------------------------------------------------------------
;
pro SaveImageTIFF_cust, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
wset,Cust_TIFF_window
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

filename = Dialog_Pickfile(/write,get_path=fpath)
if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
;presentimage=reverse(tvrd(true=1),3)
filename=AddExtension(filename,'.tiff')


WidSldTopID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Top')
;widget_control,WidSldTopID,get_value=topV
WidSldBotID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Bot')
;widget_control,WidSldBotID,get_value=botV
WidSldGammaID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Gamma')
;widget_control,WidSldGammaID,get_value=gamma

WidDL_LabelID = Widget_Info(TopID, find_by_uname='WID_DROPLIST_Label')
Label=widget_info(WidDL_LabelID,/DropList_Select)
botV=labelContrast[2,Label]
gamma=labelContrast[1,Label]
topV=labelContrast[0,Label]
if topV le botV then begin
	topV = botV+1
	widget_control,WidSldTopID,set_value=TopV
endif
sz_im=size(image)
if (sz_im[0] eq 2) and (label eq 0) then begin
	Timage=image ^(gamma/1000.)
	rng=Max(Timage)-Min(Timage)
	presentimage=bytscl(Timage,min=(botV/1000.)*rng+Min(Timage),max=Max(Timage)-(1.-topV/1000.)*rng)
	write_tiff,filename,presentimage,orientation=1
endif
mx=max(CGroupParams[26,*])
wID = Widget_Info(TopID, find_by_uname='W_MENU_54')
widget_control,wID,get_value=z_color
if z_color eq 'Z using Hue Scale' then mx=4

if (sz_im[0] eq 3) and (mx gt 1) then begin
	Labelcontrast[*,label]=[topV,gamma,botV]
	Timage=image
	mx<=3
	for i=0,mx-1 do begin
		gamma=labelcontrast[1,i+1]
		topV=labelcontrast[0,i+1]
		botV=labelcontrast[2,i+1]
		Timage[*,*,i]=image[*,*,i]^(gamma/1000.)
		rng=Max(Timage[*,*,i])-Min(Timage[*,*,i])
		Timage[*,*,i]=bytscl(Timage[*,*,i],min=(botV/1000.)*rng+Min(Timage[*,*,i]),max=Max(Timage[*,*,i])-(1.-topV/1000.)*rng)
	endfor
	tiff_image=bytarr(3,(size(Timage))[1],(size(Timage))[2])
	tiff_image[0,*,*]=reverse(Timage[*,*,0],2)
	tiff_image[1,*,*]=reverse(Timage[*,*,1],2)
	tiff_image[2,*,*]=reverse(Timage[*,*,2],2)
	write_tiff,filename,tiff_image,orientation=1
endif
end
;
;-----------------------------------------------------------------
;
function build_volume, CGroupParams, filter, ParamLimits, render_ind, render_params, render_win, liveupdate

	filterlist = where(filter eq 1,cnt)

	X_ind = render_ind[0]
	Y_ind = render_ind[1]
	Z_ind = render_ind[2]
	Xs_ind = render_ind[3]
	Ys_ind = render_ind[4]
	Zs_ind = render_ind[5]
	Nph_ind = render_ind[6]
	GrNph_ind = render_ind[7]
	FrNum_ind = render_ind[8]
	LabelSet_ind = render_ind[9]

	FilterItem = render_params[0]
	FunctionItem = render_params[1]
	AccumItem = render_params[2]
	rend_z_color = render_params[3]
	lbl_mx = render_params[4]
	testXZ = render_params[5]
	testYZ = render_params[6]
	cust_nm_per_pixel = render_params[7]
	XY_subvol_nm = render_params[8]
	Z_subvol_nm = render_params[9]
	z_multiplier = render_params[10]

	cur_win = render_win[0]
	dxmn = render_win[1]
	dymn = render_win[2]
	dzmn = render_win[3]
	dxmx = render_win[4]
	dymx = render_win[5]
	dzmx = render_win[6]
	hue_scale = render_win[7]
	wxsz = render_win[8]
	wysz = render_win[9]
	vbar_top = render_win[10]

if liveupdate gt 0 then begin
	cancel_button_present = (liveupdate eq 1) ? 1 : 0	; do not allow for cancel button in Bridge
	oStatusBar = obj_new('PALM_StatusBar', $
	COLOR=[0,0,255], $
	TEXT_COLOR=[255,255,255], $
	CANCEL_BUTTON_PRESENT = cancel_button_present, $
	TITLE='Generating  Volume...', $
	TOP_LEVEL_BASE=tlb)
	fraction_complete_last=0.0D
	pr_bar_inc=0.01D
endif

loc=fltarr(wxsz,wysz)
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))
print,'mgw=',mgw,'   cust_nm_per_pixel=',cust_nm_per_pixel
; # of display pixels per CCD pixel = ratio of the display window size to th ethe image size (magnification)

xy_scale = cust_nm_per_pixel ;/ mgw		; x-y display window scale - nm per window pixel
wd=fix(round(XY_subvol_nm / xy_scale))		; x-y radius (in units of display pixels) of sub-window to render Gaussian cloud
wdd=2*wd+1
z_scale = xy_scale / z_multiplier	; z display window scale - nm per window pixel
wdz=fix(round(Z_subvol_nm / z_scale))		; z radius (in units of display pixels) of sub-window to render Gaussian cloud
wddz=2*wdz+1
wzsz = round((dzmx-dzmn) / z_scale)
;allocate 3D image arrays for single or multiple colors
if (lbl_mx gt 0) or rend_z_color then f3image = fltarr(3,wxsz,wysz,wzsz) else	fimage = fltarr(wxsz,wysz,wzsz)

start=systime(2)
wx = findgen(wdd)-wd									;wdd x vector of window pixels (zero is in middle of array)
wy = findgen(wdd)-wd									;wdd y vector
wz = findgen(wddz)-wdz								;wddz z vector

wxpkpos	= mgw * (CGroupParams[x_ind,filterlist]-dxmn)		;x peak positions (in units of display pixels) - vector for filtered peaks
wypkpos	= mgw * (CGroupparams[y_ind,filterlist]-dymn)
wzpkpos	= (CGroupparams[z_ind,filterlist]-dzmn) / z_scale
wxsig	= mgw * CGroupParams[xs_ind,filterlist]				;x sigma  (in units of display pixels) - vector for filtered peaks
wysig	= mgw * CGroupParams[ys_ind,filterlist]
wzsig	= CGroupParams[zs_ind,filterlist] / z_scale
sigzero_loc = where(wzsig eq 0, cnt_zsig_zero)
if cnt_zsig_zero gt 0 then wzsig[sigzero_loc] = (wysig[sigzero_loc] + wysig[sigzero_loc]) *3

wxofs_v	= floor(wxpkpos)					; the offset of the center of the cloud in display units.
wyofs_v	= floor(wypkpos)
wzofs_v	= floor(wzpkpos)

;amp = (1.0d / ((2.*!pi)^1.5*(ssx/df)*(ssy/df)*(ssz/df)))<1.d

A1	=1.d/(2.*!pi)^1.5/(wxsig*wysig*wzsig)*(FunctionItem eq 1) + $			;Gaussian amplitude - vector for filtered peaks
			(CGroupParams[Nph_ind,filterlist]*(FilterItem eq 0)+CGroupParams[GrNph_ind,filterlist]*(FilterItem eq 1))/mgw/mgw*(FunctionItem eq 2)
;FunctionItem	=	(0: Center Locations,  1: Gaussian Normalized,   2: Gaussian Amplitude)
;FilterItem		=	(0: Frame Peaks,   1: Group Peaks)
;AccumItem		=	(0: Envelope,   1: Sum)
;
;default vbar_top= molecular probability of 0.003 molecule per nm^2, then vbar_top/(mgW/nm_per_pixel)^2 is molecular probabilty per pixel
; Calculate scale factor so that scale_factor * vbar_top / (mgW/nm_per_pixel)^2 = 0.5
scale_factor = 0.5 / vbar_top * float(mgw/cust_nm_per_pixel)^2
;
;---------------------------------						setup info in case Hue Scale is set

if rend_z_color then begin
	if ~testXZ and ~testYZ then normZval=(CgroupParams[z_ind,filterlist]-ParamLimits[z_ind,0])/ParamLimits[z_ind,3]
	if testXZ and ~testYZ  then normZval=(CgroupParams[x_ind,filterlist]-ParamLimits[x_ind,0])/ParamLimits[x_ind,3]
	if ~testXZ and testYZ  then normZval=(CgroupParams[y_ind,filterlist]-ParamLimits[y_ind,0])/ParamLimits[y_ind,3]
endif

;---------------------------------
for j=0l,cnt-1 do begin											;loop through all peaks
	wxofs=wxofs_v[j]
	wyofs=wyofs_v[j]
	wzofs=wzofs_v[j]

	dwx = (wx-(wxpkpos[j]-wxofs))/wxsig[j]
	dwy = (wy-(wypkpos[j]-wyofs))/wysig[j]
	dwz = (wz-(wzpkpos[j]-wzofs))/wzsig[j]

    expA=exp(-(((dwx^2)/2.)<100000.))
    expB=exp(-(((dwy^2)/2.)<100000.))
    expC=exp(-(((dwz^2)/2.)<100000.))
    gausscenter=A1[j]*reform(expA#(reform(expB#expC,wdd*wddz)), wdd, wdd, wddz) ; gauss cloud offset by a rounded position center

    xi = (wd-wxofs)>0
    xa = (2*wd+((wxsz-1-wxofs-wd)<0))
    yi = (wd-wyofs)>0
    ya = (2*wd+((wysz-1-wyofs-wd)<0))
    zi = (wdz-wzofs)>0
    za = (2*wdz+((wzsz-1-wzofs-wdz)<0))
	;if (xi gt xa) or (yi gt ya) or (zi gt za) then print,xi,xa,yi,ya,zi,za

	if zi le za then begin
		gausscenter=gausscenter[xi:xa,yi:ya,zi:za]
		if rend_z_color eq 0 then begin	; DO NOT use Hue for z-coordinate
			if (lbl_mx eq 0)	then begin	; single label
				;q=fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),(wzofs-wdz)>0:(wzofs+wdz)<(wzsz-1)]
				if (AccumItem eq 1) then fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),(wzofs-wdz)>0:(wzofs+wdz)<(wzsz-1)]+=gausscenter 	;+ q	;Sum
				if (AccumItem eq 0) then fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),(wzofs-wdz)>0:(wzofs+wdz)<(wzsz-1)]>=gausscenter 	;> q	;Envelope
			endif else	begin					; multiple labels
				labelindex=CGroupParams[LabelSet_ind,filterlist[j]]
					;q=f3image[(labelindex-1),(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),(wzofs-wdz)>0:(wzofs+wdz)<(wzsz-1)]
				if (AccumItem eq 1) then f3image[(labelindex-1),(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),(wzofs-wdz)>0:(wzofs+wdz)<(wzsz-1)]+=gausscenter ;+ q		;Sum
				if (AccumItem eq 0) then f3image[(labelindex-1),(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),(wzofs-wdz)>0:(wzofs+wdz)<(wzsz-1)]>=gausscenter ;> q		;Envelope
			endelse
		endif else begin								; use Hue for z-coordinate

			hue=hue_scale*(normZval[j]-fix(normZval[j]))							;multiply hue_scale * normalized z to range
			huep=hue/60.0
			V= scale_factor * gausscenter
			rgb_gauss=dblarr(3,(size(V))[1],(size(V))[2],(size(V))[3])
			X=V*(1.0-abs((huep mod 2)-1.0))

			if (huep ge 0) and (huep lt 1) then begin
				rgb_gauss[0,*,*,*] = V
			 	rgb_gauss[1,*,*,*] = X
			endif

			if (huep ge 1) and (huep lt 2) then begin
				rgb_gauss[0,*,*,*] = X
				rgb_gauss[1,*,*,*] = V
			endif

			if (huep ge 2) and (huep lt 3) then begin
				rgb_gauss[1,*,*,*] = V
				rgb_gauss[2,*,*,*] = X
			endif

			if (huep ge 3) and (huep lt 4) then begin
				rgb_gauss[1,*,*,*] = X
				rgb_gauss[2,*,*,*] = V
			endif

			if (huep ge 4) and (huep lt 5) then begin
				rgb_gauss[0,*,*,*] = X
				rgb_gauss[2,*,*,*] = V
			endif

			if (huep ge 5) and (huep lt 6) then begin
				rgb_gauss[0,*,*,*] = V
				rgb_gauss[2,*,*,*] = X
			endif

			if (AccumItem eq 1) then f3image[*,(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),(wzofs-wdz)>0:(wzofs+wdz)<(wzsz-1)] += rgb_gauss; + q		;Sum
			if (AccumItem eq 0) then f3image[*,(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),(wzofs-wdz)>0:(wzofs+wdz)<(wzsz-1)] >= rgb_gauss; > q		;Envelope
		endelse
	endif
		; liveupdate = 1 - all updates, liveupdate = 2 - only progress bar
	if liveupdate gt 0 then begin
		if oStatusBar -> CheckCancel() then begin
			if liveupdate gt 0 then obj_destroy, oStatusBar
			if (lbl_mx gt 0) or rend_z_color then volume_image = f3image	else	volume_image = fimage
			return, volume_image
		endif
		fraction_complete=FLOAT(j)/FLOAT((cnt-1.0))
		if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
			fraction_complete_last=fraction_complete
			oStatusBar -> UpdateStatus, fraction_complete
		endif
	endif
endfor
if liveupdate gt 0 then obj_destroy, oStatusBar
if (lbl_mx gt 0) or rend_z_color then volume_image = f3image	else	volume_image = fimage
return, volume_image
end
;
;-----------------------------------------------------------------
;
pro Generate_3D_Volume, wxsz, wysz, cust_nm_per_pixel, Zstart, Zstop, z_multiplier, XY_subvol_nm, Z_subvol_nm, volume_image, WARN_on_NOGROUPS			;Render the display according to function filter & accum settings (maybe slow)
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
common Zdisplay, Z_scale_multiplier, vbar_top
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
cur_win=!D.window

if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('No points to display')
	return      ; if data not loaded return
endif

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
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
SigL1_ind = min(where(RowNames eq 'Zeta0'))								; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))						; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))						; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))							; CGroupParametersGP[35,*] - Sigma Z
Coh_ind = min(where(RowNames eq '36 Coherence'))						; CGroupParametersGP[36,*] - Coherence
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

FunctionId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Function')
FunctionItem=widget_info(FunctionId,/DropList_Select)						;	(0: Center Locations,  1: Gaussian Normalized,   2: Gaussian Amplitude)
FilterId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Filter')
FilterItem=widget_info(FilterId,/DropList_Select)							;	(0: Frame Peaks,   1: Group Peaks)
AccumId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Accumulate')
AccumItem=widget_info(AccumId,/DropList_Select)								;	(0: Envelope,   1: Sum)

ParamLimits0=ParamLimits
ParamLimits[Z_ind,0]=Zstart
ParamLimits[GrZ_ind,0]=Zstart
ParamLimits[Z_ind,1]=Zstop
ParamLimits[GrZ_ind,1]=Zstop
if FilterItem eq 0 then 	FilterIt
if FilterItem eq 1 then begin
	GroupFilterIt
	filter=filter*(CGroupParams[GrInd_ind,*] eq 1)
endif
if (FunctionItem eq 0) and (FilterItem eq 0) then begin
	OnPeakCentersButton, Event1
	return
endif
if (FunctionItem eq 0) and (FilterItem eq 1) then begin
	OnGroupCentersButton, Event1
	return
endif

t_start = FLOAT(systime(2))

filterlist=where(filter eq 1,cnt)					; indcis of the peaks/groups to be displayed
if cnt le 1 then begin
	if WARN_on_NOGROUPS then z=dialog_message('No valid Peaks/Groups')
	return      ; if data not loaded return
endif
lbl_mx=max(CGroupParams[LabelSet_ind,filterlist])
wID = Widget_Info(TopID, find_by_uname='W_MENU_54')
widget_control,wID,get_value=z_color
rend_z_color = z_color eq 'Z using Hue Scale'
XZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapXZ')
testXZ=Widget_Info(XZ_swap_menue_ID,/button_set)
YZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapYZ')
testYZ=Widget_Info(YZ_swap_menue_ID,/button_set)
Z_UnwZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_65')
Z_UnwZ_swaped=Widget_Info(Z_UnwZ_swap_menue_ID,/button_set)

print, 'FilterItem=',FilterItem,(FilterItem ? '     (Groups)' : '     (Peaks)')
X_ind = FilterItem ? GrX_ind : X_ind
Y_ind = FilterItem ? GrY_ind : Y_ind
Z_ind = Z_UnwZ_swaped ? UnwZ_ind : Z_ind
GrZ_ind = Z_UnwZ_swaped ? UnwGrZ_ind : GrZ_ind
Z_ind = FilterItem ? GrZ_ind : Z_ind
xs_ind = FilterItem ? GrSigX_ind : SigX_ind
ys_ind = FilterItem ? GrSigY_ind : SigY_ind
zs_ind = FilterItem ? GrSigZ_ind : SigZ_ind

dxsz=xydsz[0] & dysz=xydsz[1]
dxmn = paramlimits[x_ind,0]							; image size
dymn = paramlimits[y_ind,0]
dzmn = paramlimits[z_ind,0]
dxmx = paramlimits[x_ind,1]
dymx = paramlimits[y_ind,1]
dzmx = paramlimits[z_ind,1]

render_ind = [X_ind, Y_ind, Z_ind, Xs_ind, Ys_ind, Zs_ind, Nph_ind, GrNph_ind, FrNum_ind, LabelSet_ind]
render_params = [FilterItem, FunctionItem, AccumItem, rend_z_color, lbl_mx, testXZ, testYZ, cust_nm_per_pixel, XY_subvol_nm, Z_subvol_nm, z_multiplier]
render_win = [cur_win, dxmn, dymn, dzmn, dxmx, dymx, dzmx, hue_scale, wxsz, wysz, vbar_top]

if (NOT LMGR(/VM)) and (NOT LMGR(/DEMO)) and (NOT LMGR(/TRIAL)) and allow_bridge then begin
; ***** IDL Bridge Version ******************************
		print,'Starting Bridge rendering, no intermediate display updates'
		;common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
		npk_sub = ceil(npk_tot/n_br_loops)
		for i=0, n_br_loops-1 do begin
			if i eq 0 then begin
				fbr_arr[i]->setvar, 'liveupdate',2
			endif else fbr_arr[i]->setvar, 'liveupdate',0
			fbr_arr[i]->setvar, 'nlps', i
			fbr_arr[i]->setvar, 'render_ind', render_ind
			fbr_arr[i]->setvar, 'render_params', render_params
			fbr_arr[i]->setvar, 'render_win', render_win
			fbr_arr[i]->setvar, 'ParamLimits', ParamLimits
			fbr_arr[i]->execute,'volume_i = build_volume (CGroupParams_bridge, filter_bridge, ParamLimits, render_ind, render_params, render_win, liveupdate)', /NOWAIT
		endfor

		Alldone = 0
		while alldone EQ 0 do begin
			wait,0.5
			Alldone = 1
			for i=0, n_br_loops-1 do begin
				bridge_done=fbr_arr[i]->Status()
				print,'Bridge',i,'  status0:',bridge_done
				Alldone = Alldone * (bridge_done ne 1)
			endfor
		endwhile

		volume_image = fbr_arr[0]->getvar('volume_i')
		if n_br_loops gt 1 then begin
			for i=1, n_br_loops-1 do begin
				volume_i = fbr_arr[i]->getvar('volume_i')
				if (AccumItem eq 1) then volume_image += volume_i; Sum
				if (AccumItem eq 0) then volume_image >= volume_i; Envelope
			endfor
		endif

endif else begin
;***** Non Bridge Version **************** loop through all peaks
	liveupdate = 1
	volume_image = build_volume (CGroupParams, filter, ParamLimits, render_ind, render_params, render_win, liveupdate)
endelse
ParamLimits=ParamLimits0
print,'finshed rendering 3D Volume',FLOAT(systime(2))-t_start,'  seconds render time'
end
;
;-----------------------------------------------------------------
;
pro On_Generate3D, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }

WARN_on_NOGROUPS=0
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

widget_control, /HOURGLASS   ;  Show the hourglass
Cust_TIFF_volume_image = 0
Cust_TIFF_3D=0
Generate_3D_Volume, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y, cust_nm_per_pix, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_Z_multiplier, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm, Cust_TIFF_volume_image, WARN_on_NOGROUPS
Cust_TIFF_3D=1
cust_size=size(Cust_TIFF_volume_image)
if n_elements(cust_size) le 3 then begin
	z=dialog_message('Data not grouped?, cust_size='+strtrim(n_elements(cust_size),2))
	return      ; if data not loaded return
endif
if cust_size[0] eq 4 then begin
	t0=systime(/SECONDS)
	Cust_TIFF_max = max(max(max(Cust_TIFF_volume_image,dimension=4,/NAN),dimension=3,/NAN),dimension=2,/NAN)
	;print,'search for max took  ',(systime(/SECONDS)-t0),'  seconds'
	Num_frames = cust_size[4]
endif else begin
	Cust_TIFF_max = max(Cust_TIFF_volume_image,/NAN)
	Num_frames = cust_size[3]
endelse
WID_SLIDER_Z_slice_ID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_slice')
SLIDER_MIN_MAX=widget_info(WID_SLIDER_Z_slice_ID,/SLIDER_MIN_MAX)
if SLIDER_MIN_MAX[1] ne (Num_frames-1) then widget_control,WID_SLIDER_Z_slice_ID,SET_SLIDER_MAX=(Num_frames-1),SET_VALUE=(Num_frames-1)/2
Display_Zslice, Event
wset,def_w
end
;
;-----------------------------------------------------------------
;
pro Display_Zslice, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
if Cust_TIFF_3D eq 0 then begin
	z=dialog_message('No 3D Volume image present')
	return      ; if data not loaded return
endif
wset,Cust_TIFF_window
WID_SLIDER_Z_slice_ID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Z_slice')
widget_control,WID_SLIDER_Z_slice_ID,GET_VALUE=slice_ID

create_cust_slice, slice_ID, slice

if n_elements(Cust_TIFF_max) gt 1 then begin
	tv,slice,true=3
endif else begin
	cust_size=size(Cust_TIFF_volume_image)
	if cust_size[0] eq 4 then tv,slice,true=1 else tv,slice
endelse

wset,def_w
end
;
;-----------------------------------------------------------------
;
pro create_cust_slice, slice_ID, slice
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
if n_elements(Cust_TIFF_max) gt 1 then begin
	g=reform(labelcontrast[1,1:3]/1000.0,3)
	t=reform(labelcontrast[0,1:3]/1000.0,3)
	b=reform(labelcontrast[2,1:3]/1000.0,3)
	Cust_TIFF_scl=255.0/(((Cust_TIFF_max-Cust_TIFF_max*b)>0.0)<t*Cust_TIFF_max)^g
	inf_ind=where(Cust_TIFF_scl eq !VALUES.F_INFINITY,inf_cnt)
	if inf_cnt ge 1 then Cust_TIFF_scl[inf_ind]=0
	Cust_TIFF_slice=((((Cust_TIFF_volume_image[*,*,*,slice_ID]-rebin(b*Cust_TIFF_max,[3,Cust_TIFF_Pix_X,Cust_TIFF_Pix_Y]))>0.0)<rebin(t*Cust_TIFF_max,[3,Cust_TIFF_Pix_X,Cust_TIFF_Pix_Y]))$
		^rebin(g,[3,Cust_TIFF_Pix_X,Cust_TIFF_Pix_Y]))*rebin(Cust_TIFF_scl,[3,Cust_TIFF_Pix_X,Cust_TIFF_Pix_Y])
	slice=transpose(Cust_TIFF_slice,[1,2,0])
endif else begin
	g=labelcontrast[1,0]/1000.0
	t=labelcontrast[0,0]/1000.0
	b=labelcontrast[2,0]/1000.0
	Cust_TIFF_scl=255.0/(((Cust_TIFF_max-b*Cust_TIFF_max)>0.0)<t*Cust_TIFF_max)^g
	inf_ind=where(Cust_TIFF_scl eq !VALUES.F_INFINITY,inf_cnt)
	if inf_cnt ge 1 then Cust_TIFF_scl[inf_ind]=0
	Cust_TIFF_slice=((((Cust_TIFF_volume_image[*,*,slice_ID]-Cust_TIFF_max*b)>0.0)<Cust_TIFF_max*t)^g)*Cust_TIFF_scl
	slice=Cust_TIFF_slice
	cust_size=size(Cust_TIFF_volume_image)
endelse
end
;
;-----------------------------------------------------------------
;
pro Save_Volume_TIFF, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
vol_size=size(Cust_TIFF_volume_image)
if vol_size[0] lt 3 then begin
	z=dialog_message('No 3D Volume image present')
	return      ; if data not loaded return
endif
filename = Dialog_Pickfile(/write,get_path=fpath)
if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
widget_control, /HOURGLASS	;  Show the hourglass
filename=AddExtension(filename,'.tiff')
g=reform(labelcontrast[1,1:3]/1000.0,3)
Cust_TIFF_scl=255.0/Cust_TIFF_max^g
inf_ind=where(Cust_TIFF_scl eq !VALUES.F_INFINITY,inf_cnt)
if inf_cnt ge 1 then Cust_TIFF_scl[inf_ind]=0

cust_size=size(Cust_TIFF_volume_image)
if cust_size[0] eq 4 then Num_frames = cust_size[4] else Num_frames = cust_size[3]
; cust_size[0] = 4 for multi-color image; cust_size[0] = 3  for single-color image

;********* Status Bar Initialization  ******************
oStatusBar = obj_new('PALM_StatusBar', COLOR=[0,0,255], TEXT_COLOR=[255,255,255], TITLE='Saving into a Multi-Frame TIFF file...', TOP_LEVEL_BASE=tlb)
fraction_complete_last=0.0
pr_bar_inc=0.01
;stop
for slice_ID=0, Num_frames-1 do begin
	print,'Slice_ID=',slice_ID
	create_cust_slice, slice_ID, slice
	if cust_size[0] eq 3 then Cust_TIFF_slice=fltarr(3,(size(slice))[1],(size(slice))[2])
	if cust_size[0] eq 4 then Cust_TIFF_slice=reverse(transpose(slice,[2,0,1]),3) else Cust_TIFF_slice[0,*,*] = reverse(slice,2)
	if slice_ID eq 0 then WRITE_TIFF,filename,Cust_TIFF_slice, orientation=1 $
		 else WRITE_TIFF,filename,Cust_TIFF_slice, orientation=1, /append
	fraction_complete=float(slice_ID)/(Num_frames-1.0)
	if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
		fraction_complete_last=fraction_complete
		oStatusBar -> UpdateStatus, fraction_complete
	endif
endfor
obj_destroy, oStatusBar	;********* Status Bar Close ******************

end
;
;-----------------------------------------------------------------
;
pro Save_Volume_TIFF_Monochrome, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
	vol_size=size(Cust_TIFF_volume_image)
	if vol_size[0] lt 3 then begin
		z=dialog_message('No 3D Volume image present')
		return      ; if data not loaded return
	endif
	if vol_size[0] eq 4 then Num_frames = vol_size[4] else Num_frames = vol_size[3]
	; vol_size[0] = 4 for multi-color image; vol_size[0] = 3  for single-color image
	filename = Dialog_Pickfile(/write,get_path=fpath)
	filename=AddExtension(filename,'.tif')		; monochrome
	filename_r=AddExtension(filename,'_r.tif')		; monochrome
	filename_g=AddExtension(filename,'_g.tif')		; monochrome
	filename_b=AddExtension(filename,'_b.tif')		; monochrome

	if strlen(fpath) ne 0 then cd,fpath
	if filename eq '' then return
	widget_control, /HOURGLASS	;  Show the hourglass

	;********* Status Bar Initialization  ******************
	oStatusBar = obj_new('PALM_StatusBar', COLOR=[0,0,255], TEXT_COLOR=[255,255,255], TITLE='Saving into a Multi-Frame TIFF file...', TOP_LEVEL_BASE=tlb)
	fraction_complete_last=0.0
	pr_bar_inc=0.01
	for slice_ID=0, Num_frames-1 do begin
		print,'Slice_ID=',slice_ID
		if vol_size[0] eq 3 then begin
			slice = Cust_TIFF_volume_image[*,*,slice_ID]
			if slice_ID eq 0 then write_tiff, filename, reverse(slice,2), /float, orientation=1 $
				else write_tiff, filename, reverse(slice,2), /float, orientation=1, /append
		endif else begin
			slice = Cust_TIFF_volume_image[*,*,*,slice_ID]
			if slice_ID eq 0 then begin
				write_tiff, filename_r, reverse(slice[*,*,0],2), /float, orientation=1
				write_tiff, filename_g, reverse(slice[*,*,1],2), /float, orientation=1
				write_tiff, filename_b, reverse(slice[*,*,2],2), /float, orientation=1
			endif else begin
				write_tiff, filename_r, reverse(slice[*,*,0],2), /float, orientation=1, /append
				write_tiff, filename_g, reverse(slice[*,*,1],2), /float, orientation=1, /append
				write_tiff, filename_b, reverse(slice[*,*,2],2), /float, orientation=1, /append
			endelse
		endelse

		fraction_complete=float(slice_ID)/(Num_frames-1.0)
		if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
			fraction_complete_last=fraction_complete
			oStatusBar -> UpdateStatus, fraction_complete
		endif
	endfor
	obj_destroy, oStatusBar	;********* Status Bar Close ******************

end
;
;-----------------------------------------------------------------
;
function Save_Slices_bridge, subvolume, start_Sclice_ID, file_name_prefix, liveupdate
CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	;z=dialog_message(' Save_Slices_bridge:  '+!ERROR_STATE.msg)
	z = !ERROR_STATE.msg
	CATCH, /CANCEL
	return, z
ENDIF
	vol_size = size(subvolume)
	;z=dialog_message(string(vol_size))
	if vol_size[0] eq 4 then Num_frames = vol_size[4] else Num_frames = vol_size[3]
	;********* Status Bar Initialization  ******************
	if liveupdate then begin
		oStatusBar = obj_new('PALM_StatusBar', COLOR=[0,0,255], TEXT_COLOR=[255,255,255], TITLE='Saving slices into TIF files...', TOP_LEVEL_BASE=tlb)
		fraction_complete_last=0.0
		pr_bar_inc=0.01
	endif
	for slice_ID=0, Num_frames-1 do begin
		print,'Slice_ID=',slice_ID + start_Sclice_ID
		ext_new='_slice_'+strtrim(string((slice_ID + start_Sclice_ID),FORMAT='(i3.3)'),2)
		filename =AddExtension(file_name_prefix,(ext_new+'.tif'))
		filename_r=AddExtension(file_name_prefix,('_r'+ext_new+'.tif'))
		filename_g=AddExtension(file_name_prefix,('_g'+ext_new+'.tif'))
		filename_b=AddExtension(file_name_prefix,('_b'+ext_new+'.tif'))
		if vol_size[0] eq 3 then begin
			slice = subvolume[*,*,slice_ID]
			write_tiff, filename, reverse(slice,2), /float, orientation=1
		endif else begin
			slice = subvolume[*,*,*,slice_ID]
			write_tiff, filename_r, reverse(slice[*,*,0],2), /float, orientation=1
			write_tiff, filename_g, reverse(slice[*,*,1],2), /float, orientation=1
			write_tiff, filename_b, reverse(slice[*,*,2],2), /float, orientation=1
		endelse
		if liveupdate then begin
			fraction_complete=float(slice_ID)/(Num_frames-1.0)
			if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
				fraction_complete_last=fraction_complete
				oStatusBar -> UpdateStatus, fraction_complete
			endif
		endif
	endfor
	if liveupdate then obj_destroy, oStatusBar	;********* Status Bar Close ******************
CATCH, /CANCEL
return,!ERROR_STATE.msg
end
;
;-----------------------------------------------------------------
;
pro Save_Volume_TIFF_separate_files_Monochrome, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
common Zdisplay, Z_scale_multiplier, vbar_top
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }
cur_win=!D.window

WARN_on_NOGROUPS=0
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
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
SigL1_ind = min(where(RowNames eq 'Zeta0'))								; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))						; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))						; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))							; CGroupParametersGP[35,*] - Sigma Z
Coh_ind = min(where(RowNames eq '36 Coherence'))						; CGroupParametersGP[36,*] - Coherence
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

FunctionId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Function')
FunctionItem=widget_info(FunctionId,/DropList_Select)						;	(0: Center Locations,  1: Gaussian Normalized,   2: Gaussian Amplitude)
FilterId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Filter')
FilterItem=widget_info(FilterId,/DropList_Select)							;	(0: Frame Peaks,   1: Group Peaks)
AccumId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Accumulate')
AccumItem=widget_info(AccumId,/DropList_Select)								;	(0: Envelope,   1: Sum)


filename0 = Dialog_Pickfile(/write,get_path=fpath)
if strlen(fpath) ne 0 then cd,fpath
if filename0 eq '' then return
widget_control, /HOURGLASS	;  Show the hourglass

t_start = FLOAT(systime(2))

if FilterItem eq 0 then 	FilterIt
if FilterItem eq 1 then begin
	GroupFilterIt
	filter=filter*(CGroupParams[GrInd_ind,*] eq 1)
endif
if (FunctionItem eq 0) and (FilterItem eq 0) then begin
	OnPeakCentersButton, Event1
	return
endif
if (FunctionItem eq 0) and (FilterItem eq 1) then begin
	OnGroupCentersButton, Event1
	return
endif

filterlist=where(filter eq 1,cnt)					; indcis of the peaks/groups to be displayed
if cnt le 1 then begin
	if WARN_on_NOGROUPS then z=dialog_message('No valid Peaks/Groups')
	return      ; if data not loaded return
endif
lbl_mx=max(CGroupParams[LabelSet_ind,filterlist])
wID = Widget_Info(TopID, find_by_uname='W_MENU_54')
widget_control,wID,get_value=z_color
rend_z_color = z_color eq 'Z using Hue Scale'
XZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapXZ')
testXZ=Widget_Info(XZ_swap_menue_ID,/button_set)
YZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapYZ')
testYZ=Widget_Info(YZ_swap_menue_ID,/button_set)
Z_UnwZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_65')
Z_UnwZ_swaped=Widget_Info(Z_UnwZ_swap_menue_ID,/button_set)

print, 'FilterItem=',FilterItem,(FilterItem ? '     (Groups)' : '     (Peaks)')
X_ind = FilterItem ? GrX_ind : X_ind
Y_ind = FilterItem ? GrY_ind : Y_ind
Z_ind = Z_UnwZ_swaped ? UnwZ_ind : Z_ind
GrZ_ind = Z_UnwZ_swaped ? UnwGrZ_ind : GrZ_ind
Z_ind = FilterItem ? GrZ_ind : Z_ind
xs_ind = FilterItem ? GrSigX_ind : SigX_ind
ys_ind = FilterItem ? GrSigY_ind : SigY_ind
zs_ind = FilterItem ? GrSigZ_ind : SigZ_ind

dxsz=xydsz[0] & dysz=xydsz[1]
dxmn = paramlimits[x_ind,0]							; image size
dymn = paramlimits[y_ind,0]
dzmn = paramlimits[z_ind,0]
dxmx = paramlimits[x_ind,1]
dymx = paramlimits[y_ind,1]
dzmx = paramlimits[z_ind,1]
wxsz = Cust_TIFF_Pix_X
wysz = Cust_TIFF_Pix_Y

mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))
print,'mgw=',mgw,'   cust_nm_per_pixel=',cust_nm_per_pix
; # of display pixels per CCD pixel = ratio of the display window size to th ethe image size (magnification)
z_scale = cust_nm_per_pix / Cust_TIFF_Z_multiplier	; z display window scale - nm per window pixel
wzsz = round((dzmx-dzmn) / z_scale)

render_ind = [X_ind, Y_ind, Z_ind, Xs_ind, Ys_ind, Zs_ind, Nph_ind, GrNph_ind, FrNum_ind, LabelSet_ind]
render_params = [FilterItem, FunctionItem, AccumItem, rend_z_color, lbl_mx, testXZ, testYZ, cust_nm_per_pix, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm, Cust_TIFF_Z_multiplier]
render_win0 = [cur_win, dxmn, dymn, dzmn, dxmx, dymx, dzmx, hue_scale, wxsz, wysz, vbar_top]

if (NOT LMGR(/VM)) and (NOT LMGR(/DEMO)) and (NOT LMGR(/TRIAL)) and allow_bridge then begin

	dz_bridge = floor(wzsz / n_br_loops)+1
	Z_min = dzmn + findgen(n_br_loops)*dz_bridge*z_scale
	Z_max = (Z_min + dz_bridge*z_scale) < dzmx
	ID_start = findgen(n_br_loops)*dz_bridge
	; ***** IDL Bridge Version ******************************
	print,'Starting Bridge rendering, no intermediate display updates'
	;common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
	for i=0, n_br_loops-1 do begin
		if i eq 0 then begin
			fbr_arr[i]->setvar, 'liveupdate',2
		endif else fbr_arr[i]->setvar, 'liveupdate',0
		render_win = render_win0
		render_win[3] = Z_min[i]
		render_win[6] = Z_max[i]
		fbr_arr[i]->setvar, 'filename0', filename0
		fbr_arr[i]->setvar, 'render_ind', render_ind
		fbr_arr[i]->setvar, 'render_params', render_params
		fbr_arr[i]->setvar, 'render_win', render_win
		fbr_arr[i]->setvar, 'ParamLimits', ParamLimits
		fbr_arr[i]->setvar, 'ID_start', ID_start
		fbr_arr[i]->execute,'Z_ind = render_ind[2]'
		fbr_arr[i]->execute,'Z_subvol_nm = render_params[9]'
		fbr_arr[i]->execute,'Zmin = render_win[3] - 2*Z_subvol_nm'
		fbr_arr[i]->execute,'Zmax = render_win[6] + 2*Z_subvol_nm'
		fbr_arr[i]->execute,'filter_bridge_i = filter_bridge and (CGroupParams_bridge[Z_ind,*] ge Zmin) and (CGroupParams_bridge[Z_ind,*] le Zmax)'
		fbr_arr[i]->execute,'volume_i = build_volume (CGroupParams_bridge, filter_bridge_i, ParamLimits, render_ind, render_params, render_win, liveupdate)', /NOWAIT
	endfor

print,'step1 finished'
	Alldone = 0
	while (alldone EQ 0) do begin
		wait,0.5
		Alldone = 1
		for i=0, n_br_loops-1 do begin
			bridge_done = fbr_arr[i]->Status(ERROR=ErrorString)
			err_str = strlen(ErrorString) eq 0 ? '' : (',   Error String: '+ErrorString)
			print,'Step 2: Bridge',i,'  status: ', bridge_done, err_str
			Alldone = Alldone * (bridge_done eq 0)
		endfor
	endwhile


bridge_starts = intarr(n_br_loops)
	Alldone = 0
	while (alldone EQ 0) or (product(bridge_starts) eq 0) do begin
		wait,0.5
		Alldone = 1
		for i=0, n_br_loops-1 do begin
			bridge_done=fbr_arr[i]->Status(ERROR=ErrorString)
			err_str = strlen(ErrorString) eq 0 ? '' : (',   Error String: '+ErrorString)
			print,'Step 3: Bridge',i,'  status: ', bridge_done, err_str
			Alldone = Alldone * (bridge_done ne 1)
			if (bridge_done eq 0) and (bridge_starts[i] eq 0) then begin
				fbr_arr[i]->execute,'test=size(volume_i)'
				test = fbr_arr[i]->getvar('test')
				print,'Bridge',i,'  returned array:',string(test)
				fbr_arr[i]->setvar, 'nlps', i
				fbr_arr[i]->execute,'error_message = Save_Slices_bridge(volume_i, ID_start[nlps], filename0, liveupdate)', /NOWAIT
				bridge_starts[i] =1				; indicate that the second part had started
			endif
		endfor
	endwhile

	Alldone = 0
	while (alldone EQ 0) do begin
		wait,0.5
		Alldone = 1
		for i=0, n_br_loops-1 do begin
			bridge_done=fbr_arr[i]->Status(ERROR=ErrorString)
			err_str = strlen(ErrorString) eq 0 ? '' : (',   Error String: '+ErrorString)
			print,'Step 3: Bridge',i,'  status: ', bridge_done, err_str
			Alldone = Alldone * (bridge_done ne 1)
		endfor
	endwhile

endif else begin
;***** Non Bridge Version **************** loop through all peaks
	liveupdate = 1
	volume_image = build_volume (CGroupParams, filter, ParamLimits, render_ind, render_params, render_win0, liveupdate)
	error_message = Save_Slices_bridge(volume_image, 0, filename0, liveupdate)
endelse

print,'finshed saving monochrome slices',FLOAT(systime(2))-t_start,'  seconds render time'
end
;
;-----------------------------------------------------------------
;
pro Save_Volume_TIFF_separate_files_Monochrome_original, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
	vol_size=size(Cust_TIFF_volume_image)
	if vol_size[0] lt 3 then begin
		z=dialog_message('No 3D Volume image present')
		return      ; if data not loaded return
	endif
	if vol_size[0] eq 4 then Num_frames = vol_size[4] else Num_frames = vol_size[3]
	; vol_size[0] = 4 for multi-color image; vol_size[0] = 3  for single-color image
	filename0 = Dialog_Pickfile(/write,get_path=fpath)

	if strlen(fpath) ne 0 then cd,fpath
	if filename0 eq '' then return
	widget_control, /HOURGLASS	;  Show the hourglass

	;********* Status Bar Initialization  ******************
	oStatusBar = obj_new('PALM_StatusBar', COLOR=[0,0,255], TEXT_COLOR=[255,255,255], TITLE='Saving into a Multi-Frame TIFF file...', TOP_LEVEL_BASE=tlb)
	fraction_complete_last=0.0
	pr_bar_inc=0.01
	for slice_ID=0, Num_frames-1 do begin
		print,'Slice_ID=',slice_ID
		ext_new='_slice_'+strtrim(string(slice_ID,FORMAT='(i3.3)'),2)
		filename =AddExtension(filename0,(ext_new+'.tif'))
		filename_r=AddExtension(filename0,('_r'+ext_new+'.tif'))
		filename_g=AddExtension(filename0,('_g'+ext_new+'.tif'))
		filename_b=AddExtension(filename0,('_b'+ext_new+'.tif'))
		if vol_size[0] eq 3 then begin
			slice = Cust_TIFF_volume_image[*,*,slice_ID]
			write_tiff, filename, reverse(slice,2), /float, orientation=1
		endif else begin
			slice = Cust_TIFF_volume_image[*,*,*,slice_ID]
			write_tiff, filename_r, reverse(slice[*,*,0],2), /float, orientation=1
			write_tiff, filename_g, reverse(slice[*,*,1],2), /float, orientation=1
			write_tiff, filename_b, reverse(slice[*,*,2],2), /float, orientation=1
		endelse

		fraction_complete=float(slice_ID)/(Num_frames-1.0)
		if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
			fraction_complete_last=fraction_complete
			oStatusBar -> UpdateStatus, fraction_complete
		endif
	endfor
	obj_destroy, oStatusBar	;********* Status Bar Close ******************

end
;
;-----------------------------------------------------------------
;
pro Save_Volume_TIFF_separate_files, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
vol_size=size(Cust_TIFF_volume_image)
if vol_size[0] lt 3 then begin
	z=dialog_message('No 3D Volume image present')
	return      ; if data not loaded return
endif
filename0 = Dialog_Pickfile(/write,get_path=fpath)
if strlen(fpath) ne 0 then cd,fpath
if filename0 eq '' then return
widget_control, /HOURGLASS	;  Show the hourglass
Zstep=cust_nm_per_pix/Cust_TIFF_Z_multiplier
g=reform(labelcontrast[1,1:3]/1000.0,3)
Cust_TIFF_scl=255.0/Cust_TIFF_max^g
inf_ind=where(Cust_TIFF_scl eq !VALUES.F_INFINITY,inf_cnt)
if inf_cnt ge 1 then Cust_TIFF_scl[inf_ind]=0

cust_size=size(Cust_TIFF_volume_image)
if cust_size[0] eq 4 then Num_frames = cust_size[4] else Num_frames = cust_size[3]
; cust_size[0] = 4 for multi-color image; cust_size[0] = 3  for single-color image

;********* Status Bar Initialization  ******************
oStatusBar = obj_new('PALM_StatusBar', COLOR=[0,0,255], TEXT_COLOR=[255,255,255], TITLE='Saving into Multiple TIFF files...', TOP_LEVEL_BASE=tlb)
fraction_complete_last=0.0
pr_bar_inc=0.01

for slice_ID=0, Num_frames-1 do begin
	print,'Slice_ID=',slice_ID
	Zmin=Cust_TIFF_Z_start+slice_ID*Zstep
	Zmax=Zmin+Zstep
	ext_new='_slice.'+strtrim(string(slice_ID,FORMAT='(i3.3)'),2)+'.tiff'
	filename=AddExtension(filename0,ext_new)

	create_cust_slice, slice_ID, slice

	if cust_size[0] eq 4 then Cust_TIFF_slice=reverse(transpose(slice,[2,0,1]),3) else Cust_TIFF_slice = reverse(slice,2)

	WRITE_TIFF,filename,Cust_TIFF_slice, orientation=1
		;********* Status Bar Update ******************
	fraction_complete=float(slice_ID)/(Num_frames-1.0)
	if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
		fraction_complete_last=fraction_complete
		oStatusBar -> UpdateStatus, fraction_complete
	endif
endfor
obj_destroy, oStatusBar	;********* Status Bar Close ******************
end
;
;-----------------------------------------------------------------
;
pro Save_Volume_PNG_separate_files, Event
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
vol_size=size(Cust_TIFF_volume_image)
if vol_size[0] lt 3 then begin
	z=dialog_message('No 3D Volume image present')
	return      ; if data not loaded return
endif
filename0 = Dialog_Pickfile(/write,get_path=fpath)
if strlen(fpath) ne 0 then cd,fpath
if filename0 eq '' then return
widget_control, /HOURGLASS	;  Show the hourglass
Zstep=cust_nm_per_pix/Cust_TIFF_Z_multiplier
g=reform(labelcontrast[1,1:3]/1000.0,3)
Cust_TIFF_scl=255.0/Cust_TIFF_max^g
inf_ind=where(Cust_TIFF_scl eq !VALUES.F_INFINITY,inf_cnt)
if inf_cnt ge 1 then Cust_TIFF_scl[inf_ind]=0

cust_size=size(Cust_TIFF_volume_image)
if cust_size[0] eq 4 then Num_frames = cust_size[4] else Num_frames = cust_size[3]
; cust_size[0] = 4 for multi-color image; cust_size[0] = 3  for single-color image

;********* Status Bar Initialization  ******************
oStatusBar = obj_new('PALM_StatusBar', COLOR=[0,0,255], TEXT_COLOR=[255,255,255], TITLE='Saving into Multiple PNG files...', TOP_LEVEL_BASE=tlb)
fraction_complete_last=0.0
pr_bar_inc=0.01

for slice_ID=0, Num_frames-1 do begin
	print,'Slice_ID=',slice_ID
	Zmin=Cust_TIFF_Z_start+slice_ID*Zstep
	Zmax=Zmin+Zstep
	ext_new='_slice.'+strtrim(string(slice_ID,FORMAT='(i3.3)'),2)+'.png'
	filename=AddExtension(filename0,ext_new)

	create_cust_slice, slice_ID, slice
	if cust_size[0] eq 3 then Cust_TIFF_slice_RGB=fltarr(3,(size(slice))[1],(size(slice))[2])
	if cust_size[0] eq 4 then Cust_TIFF_slice_RGB=reverse(transpose(slice,[2,0,1]),3) else Cust_TIFF_slice_RGB[0,*,*] = reverse(slice,2)

	color_convert,Cust_TIFF_slice_RGB,Cust_TIFF_slice_HSV,/rgb_hsv
	Alpha_channel = (Cust_TIFF_slice_HSV[2,*,*]*255.0<255.0)>0.0
	Cust_TIFF_slice_HSV[2,*,*]=0.98
	color_convert,Cust_TIFF_slice_HSV,Cust_TIFF_slice_RGB_new,/hsv_rgb
	Cust_TIFF_slice_RGBA=fix(round([Cust_TIFF_slice_RGB_new,Alpha_channel]),type=1)

	WRITE_PNG,filename,Cust_TIFF_slice_RGBA
		;********* Status Bar Update ******************
	fraction_complete=float(slice_ID)/(Num_frames-1.0)
	if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
		fraction_complete_last=fraction_complete
		oStatusBar -> UpdateStatus, fraction_complete
	endif
endfor
obj_destroy, oStatusBar	;********* Status Bar Close ******************
end
;
;-----------------------------------------------------------------
;
pro Overlay_DIC_cust_TIFF, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
if (n_elements(DIC) le 1) then begin
	z=dialog_message('DIC image not loaded')
	return
endif
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

wset,Cust_TIFF_window
present_image=tvrd(true=3)

DICx=(size(DIC))[1] & DICy=(size(DIC))[2]
wxsz=Cust_TIFF_Pix_X & wysz=Cust_TIFF_Pix_Y
dxmn = paramlimits[2,0]
dymn = paramlimits[3,0]
dxmx = paramlimits[2,1]
dymx = paramlimits[3,1]
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))
rmg=FLOAT(max((size(DIC))[1:2])) / (DICx > DICy)		; real magnification

XI=fix(floor(rmg*dxmn))>0
XA=fix(floor(rmg*(dxmx))) < (DICx - 1)
YI=fix(floor(rmg*dymn)) > 0
YA=fix(floor(rmg*(dymx))) < (DICy - 1)

botV=labelContrast[2,4]/1000.
gamma=labelContrast[1,4]/1000.
topV=labelContrast[0,4]/1000.

Fimage=DIC[XI : XA, YI : YA] ^gamma
;rng=(Max(Fimage)-Min(Fimage))
Fimage= ((Fimage - botV*Max(Fimage))>0) < topV*Max(Fimage)
Fimage=Fimage/max(Fimage)*255
fimagex=fix(float(XA-XI+1)*mgw/rmg)
fimagey=fix(float(YA-YI+1)*mgw/rmg)
Fimage=Congrid(Fimage,fimagex,fimagey)
xs=(fimagex-1) < ((size(present_image))[1]-1)
ys=(fimagey-1) < ((size(present_image))[2]-1)

tv,bytarr(wxsz,wysz)

Overl_Image=present_image
Overl_Image[0:xs,0:ys,0]=Overl_Image[0:xs,0:ys,0]/2.0+Fimage[0:xs,0:ys]/2.0
Overl_Image[0:xs,0:ys,1]=Overl_Image[0:xs,0:ys,1]/2.0+Fimage[0:xs,0:ys]/2.0
Overl_Image[0:xs,0:ys,2]=Overl_Image[0:xs,0:ys,2]/2.0+Fimage[0:xs,0:ys]/2.0

tvscl,Overl_Image,true=3
wset,def_w
end
;
;-----------------------------------------------------------------
;
pro Draw_DIC_only_cust_TIFF, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
if (n_elements(DIC) le 1) then begin
	z=dialog_message('DIC image not loaded')
	return
endif
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

wset,Cust_TIFF_window

DICx=(size(DIC))[1] & DICy=(size(DIC))[2]
wxsz=Cust_TIFF_Pix_X & wysz=Cust_TIFF_Pix_Y
dxmn = paramlimits[2,0]
dymn = paramlimits[3,0]
dxmx = paramlimits[2,1]
dymx = paramlimits[3,1]
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))
rmg=FLOAT(max((size(DIC))[1:2])) / (DICx > DICy)		; real magnification

XI=fix(floor(rmg*dxmn))>0
XA=fix(floor(rmg*(dxmx))) < (DICx - 1)
YI=fix(floor(rmg*dymn)) > 0
YA=fix(floor(rmg*(dymx))) < (DICy - 1)

botV=labelContrast[2,4]/1000.
gamma=labelContrast[1,4]/1000.
topV=labelContrast[0,4]/1000.

Fimage=DIC[XI : XA, YI : YA] ^gamma
rng=(Max(Fimage)-Min(Fimage))
Fimage= ((Fimage - botV*rng)>0) < (Max(Fimage)-(1.-topV)*rng)
Fimage=Fimage/max(Fimage)*255
fimagex=fix(float(XA-XI+1)*mgw/rmg)
fimagey=fix(float(YA-YI+1)*mgw/rmg)
Fimage=Congrid(Fimage,fimagex,fimagey)
xs=(fimagex-1) < ((size(present_image))[1]-1)
ys=(fimagey-1) < ((size(present_image))[2]-1)

tv,bytarr(wxsz,wysz)
tvscl,Fimage
image=fimage
AdjustContrastnDisplay, Event

wset,def_w
end
;
;-----------------------------------------------------------------
;
pro OnTotalRawDataButton_cust, Event

common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

wset,Cust_TIFF_window

wxsz=Cust_TIFF_Pix_X & wysz=Cust_TIFF_Pix_Y
dxmn = paramlimits[2,0]
dymn = paramlimits[3,0]
dxmx = paramlimits[2,1]
dymx = paramlimits[3,1]
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))
rmg=FLOAT(max((size(TotalRawData))[1:2])) / (max(xydsz))		; real magnification

XI=long(floor(rmg*dxmn))>0
XA=long(floor(rmg*(dxmx))) < ((size(TotalRawData))[1]-1)
YI=long(floor(rmg*dymn)) > 0
YA=long(floor(rmg*(dymx))) < ((size(TotalRawData))[2]-1)
Fimage=TotalRawData[XI : XA, YI : YA]
fimagex=fix(float(XA-XI+1)*mgw/rmg)
fimagey=fix(float(YA-YI+1)*mgw/rmg)

Fimage=Congrid(Fimage,fimagex,fimagey)

tv,bytarr(wxsz,wysz)
tvscl,Fimage
image=fimage				;tvrd(true=1)
AdjustContrastnDisplay, Event

wset,def_w
end;-----------------------------------------------------------------


;-----------------------------------------------------------------
pro Set_Tie_RGB_CustTIFF, Event

end
