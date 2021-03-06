; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	07/18/2017 17:06.41
; 
pro WID_BASE_SaveCustomTIFF_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_SaveCustomTIFF'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_Separate_TIFFs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_Volume_TIFF_separate_files, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DRAW_Custom_TIFF'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_IMAGE_SCALING_Parameters'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        DoInsert_Cust_TIFF_Scale_Param, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Render_cust_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Render_cust_TIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Accumulate_cust_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        Cust_TIFF_Select_Accumulation, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Filter_cust_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        Cust_TIFF_Select_Filter, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Function_cust_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        Cust_TIFF_Select_Function, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_Bot_cust_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        OnStretchBottom_cust_TIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_Gamma_cust_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        OnGamma_cust_TIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_Top_cust_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        OnStretchTop_cust_TIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Label_cust_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        OnLabelDropList_cust_TIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_cust_TIFF_0'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_cust_TIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_ScaleBar_cust_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAddScaleBarButton_cust_TIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Generate3D'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Generate3D, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_Z_slice'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        Display_Zslice, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_Multiframe_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_Volume_TIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_IMAGE_Zcoord_Parameters'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        DoInsert_Cust_TIFF_ZScale_Param, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_XY_subvolume'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        Change_XY_Subvolume, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_STR' )then $
        Change_XY_Subvolume, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_Separate_PNGs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_Volume_PNG_separate_files, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Overlay_DIC_EM_cust_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Overlay_DIC_cust_TIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_Multiframe_Monochrome_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_Volume_TIFF_Monochrome, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_TotalRawData_cust'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnTotalRawDataButton_cust, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Render_cust_DIC'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Draw_DIC_only_cust_TIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Tie_RGB_CustTIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_Tie_RGB_CustTIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_cust_TIFF_float'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_cust_TIFF_float, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_Volume_Multiple_Monochrome_TIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_Volume_TIFF_separate_files_Monochrome, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_Z_subvolume'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        Change_Z_Subvolume, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_STR' )then $
        Change_Z_Subvolume, Event
    end
    else:
  endcase

end

pro WID_BASE_SaveCustomTIFF, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'SaveCustomTIFF_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_SaveCustomTIFF = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_SaveCustomTIFF' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=1446 ,SCR_YSIZE=1071  $
      ,NOTIFY_REALIZE='Initialize_Custom_TIFF' ,TITLE='Save Custom'+ $
      ' TIFF files' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_BUTTON_Save_Separate_TIFFs =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Save_Separate_TIFFs' ,XOFFSET=51 ,YOFFSET=815  $
      ,SCR_XSIZE=280 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Save Volume'+ $
      ' as Separate TIFF files')

  
  WID_DRAW_Custom_TIFF = Widget_Draw(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_DRAW_Custom_TIFF' ,FRAME=1 ,XOFFSET=380  $
      ,SCR_XSIZE=1024 ,SCR_YSIZE=1024  $
      ,NOTIFY_REALIZE='CustomTIFF_Draw_Realize' ,/SCROLL ,XSIZE=1024  $
      ,YSIZE=1024)

  
  WID_IMAGE_SCALING_Parameters =  $
      Widget_Table(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_IMAGE_SCALING_Parameters' ,XOFFSET=21 ,YOFFSET=340  $
      ,SCR_XSIZE=335 ,SCR_YSIZE=110 ,/EDITABLE ,/RESIZEABLE_COLUMNS  $
      ,COLUMN_LABELS=[ 'Value' ] ,ROW_LABELS=[ 'NM per Image Pixel',  $
      'Total Image Pixels X', 'Total Image Pixels Y' ] ,XSIZE=1  $
      ,YSIZE=3)

  
  WID_BUTTON_Render_cust_TIFF =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Render_cust_TIFF' ,XOFFSET=20 ,YOFFSET=105  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Render')

  
  WID_DROPLIST_Accumulate_cust_TIFF =  $
      Widget_Droplist(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_DROPLIST_Accumulate_cust_TIFF' ,XOFFSET=10  $
      ,YOFFSET=75 ,SCR_XSIZE=170 ,SCR_YSIZE=25 ,TITLE='Accumulation'  $
      ,VALUE=[ 'Envelope', 'Sum' ])

  
  WID_DROPLIST_Filter_cust_TIFF =  $
      Widget_Droplist(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_DROPLIST_Filter_cust_TIFF' ,XOFFSET=10 ,YOFFSET=40  $
      ,SCR_XSIZE=170 ,SCR_YSIZE=25 ,TITLE='Filter' ,VALUE=[ 'Frame'+ $
      ' Peaks', 'Grouped Peaks' ])

  
  WID_DROPLIST_Function_cust_TIFF =  $
      Widget_Droplist(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_DROPLIST_Function_cust_TIFF' ,XOFFSET=10 ,YOFFSET=5  $
      ,SCR_XSIZE=170 ,SCR_YSIZE=25 ,TITLE='Function' ,VALUE=[ 'Center'+ $
      ' Locations', 'Gaussian Normalized', 'Gaussian Amplitude' ])

  
  WID_SLIDER_Bot_cust_TIFF = Widget_Slider(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_SLIDER_Bot_cust_TIFF' ,XOFFSET=204 ,YOFFSET=156  $
      ,SCR_XSIZE=160 ,SCR_YSIZE=46 ,TITLE='Stretch Bottom'  $
      ,MAXIMUM=1000)

  
  WID_SLIDER_Gamma_cust_TIFF = Widget_Slider(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_SLIDER_Gamma_cust_TIFF' ,XOFFSET=204 ,YOFFSET=101  $
      ,SCR_XSIZE=160 ,SCR_YSIZE=46 ,TITLE='Gamma' ,MAXIMUM=1000  $
      ,VALUE=500)

  
  WID_SLIDER_Top_cust_TIFF = Widget_Slider(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_SLIDER_Top_cust_TIFF' ,XOFFSET=205 ,YOFFSET=46  $
      ,SCR_XSIZE=160 ,SCR_YSIZE=46 ,TITLE='Stretch Top' ,MAXIMUM=1000  $
      ,VALUE=500)

  
  WID_DROPLIST_Label_cust_TIFF =  $
      Widget_Droplist(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_DROPLIST_Label_cust_TIFF' ,XOFFSET=250 ,YOFFSET=5  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=25 ,TITLE='Label' ,VALUE=[ '', 'Red',  $
      'Green', 'Blue', 'DIC / EM' ])

  
  WID_BUTTON_Save_cust_TIFF_0 =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Save_cust_TIFF_0' ,XOFFSET=30 ,YOFFSET=260  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Save TIFF')

  
  WID_BUTTON_ScaleBar_cust_TIFF =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_ScaleBar_cust_TIFF' ,XOFFSET=222 ,YOFFSET=251  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Add Scale'+ $
      ' Bar1')

  
  WID_BUTTON_Generate3D = Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Generate3D' ,XOFFSET=81 ,YOFFSET=670  $
      ,SCR_XSIZE=193 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Generate 3D'+ $
      ' Volume')

  
  WID_SLIDER_Z_slice = Widget_Slider(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_SLIDER_Z_slice' ,XOFFSET=16 ,YOFFSET=710  $
      ,SCR_XSIZE=350 ,SCR_YSIZE=55 ,TITLE='Z slice #' ,MAXIMUM=100  $
      ,VALUE=50)

  
  WID_BUTTON_Save_Multiframe_TIFF =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Save_Multiframe_TIFF' ,XOFFSET=51  $
      ,YOFFSET=770 ,SCR_XSIZE=280 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Save Volume as Multi-frame TIFF file')

  
  WID_IMAGE_Zcoord_Parameters = Widget_Table(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_IMAGE_Zcoord_Parameters' ,XOFFSET=21 ,YOFFSET=460  $
      ,SCR_XSIZE=335 ,SCR_YSIZE=125 ,/EDITABLE ,COLUMN_LABELS=[  $
      'Value' ] ,ROW_LABELS=[ 'Z start (nm)', 'Z stop (nm)', 'Z step'+ $
      ' (nm)', 'Z - X scaling' ] ,XSIZE=1 ,YSIZE=4)

  
  WID_TEXT_XY_subvolume = Widget_Text(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_TEXT_XY_subvolume' ,XOFFSET=281 ,YOFFSET=590  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=30 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,VALUE=[ '100.0', '' ] ,XSIZE=20 ,YSIZE=2)

  
  WID_LABEL_XY_subvolume_txt = Widget_Label(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_LABEL_XY_subvolume_txt' ,XOFFSET=21 ,YOFFSET=600  $
      ,SCR_XSIZE=250 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='XY Gauss.'+ $
      ' Cloud Radius (subvol) (nm)')

  
  WID_BUTTON_Save_Separate_PNGs =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Save_Separate_PNGs' ,XOFFSET=51 ,YOFFSET=860  $
      ,SCR_XSIZE=280 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Save Volume'+ $
      ' as Separate PNG files')

  
  WID_BUTTON_Overlay_DIC_EM_cust_TIFF =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Overlay_DIC_EM_cust_TIFF' ,XOFFSET=20  $
      ,YOFFSET=180 ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Overlay DIC/EM')

  
  WID_BUTTON_Save_Multiframe_Monochrome_TIFF =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Save_Multiframe_Monochrome_TIFF' ,XOFFSET=20  $
      ,YOFFSET=905 ,SCR_XSIZE=340 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Save Volume as Monochrome Multi-frame TIFF stack')

  
  WID_BUTTON_TotalRawData_cust =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_TotalRawData_cust' ,XOFFSET=20 ,YOFFSET=140  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Total Raw'+ $
      ' Data')

  
  WID_BUTTON_Render_cust_DIC = Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Render_cust_DIC' ,XOFFSET=20 ,YOFFSET=220  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Render Only'+ $
      ' DIC/EM')

  
  WID_BASE_Tie_RGB_CustTIFF = Widget_Base(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BASE_Tie_RGB_CustTIFF' ,XOFFSET=236 ,YOFFSET=211  $
      ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Tie_RGB_CustTIFF =  $
      Widget_Button(WID_BASE_Tie_RGB_CustTIFF,  $
      UNAME='WID_BUTTON_Tie_RGB_CustTIFF' ,/ALIGN_LEFT ,VALUE='Tie'+ $
      ' RGB')

  
  WID_BUTTON_Save_cust_TIFF_float =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Save_cust_TIFF_float' ,XOFFSET=30  $
      ,YOFFSET=300 ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Save TIFF float')

  
  WID_BUTTON_Save_Volume_Multiple_Monochrome_TIFF =  $
      Widget_Button(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_BUTTON_Save_Volume_Multiple_Monochrome_TIFF'  $
      ,XOFFSET=10 ,YOFFSET=965 ,SCR_XSIZE=360 ,SCR_YSIZE=30  $
      ,/ALIGN_CENTER ,VALUE='Generate Volume + save into Monochrome'+ $
      ' TIFF files')

  
  WID_LABEL_dont_generate_Volume =  $
      Widget_Label(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_LABEL_dont_generate_Volume' ,XOFFSET=10  $
      ,YOFFSET=1000 ,SCR_XSIZE=365 ,SCR_YSIZE=20 ,/ALIGN_LEFT  $
      ,VALUE='David, do NOT press "Generate 3D Volume" with this')

  
  WID_LABEL_Z_subvolume_txt = Widget_Label(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_LABEL_Z_subvolume_txt' ,XOFFSET=20 ,YOFFSET=635  $
      ,SCR_XSIZE=250 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Z Gauss.'+ $
      ' Cloud Radius (subvol) (nm)')

  
  WID_TEXT_Z_subvolume = Widget_Text(WID_BASE_SaveCustomTIFF,  $
      UNAME='WID_TEXT_Z_subvolume' ,XOFFSET=280 ,YOFFSET=630  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=30 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,VALUE=[ '100.0' ] ,XSIZE=20 ,YSIZE=2)

  Widget_Control, /REALIZE, WID_BASE_SaveCustomTIFF

  XManager, 'WID_BASE_SaveCustomTIFF', WID_BASE_SaveCustomTIFF, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro SaveCustomTIFF, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_SaveCustomTIFF, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
