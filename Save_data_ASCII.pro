; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	01/10/2017 09:19.45
; 
pro WID_BASE_Save_Data_ASCII_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Save_Data_ASCII'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_ASCII_Filename'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        On_ASCII_Filename_change, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_STR' )then $
        On_ASCII_Filename_change, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_ASCII_Save_Parameter_List'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        On_ASCII_ParamList_change, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_STR' )then $
        On_ASCII_ParamList_change, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Save_ASCII_Filter'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        On_Select_SaveASCII_Filter, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Save_ASCII_XY'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        On_Select_SaveASCII_units, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Save_ASCII_Parameters'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        On_Select_SaveASCII_ParamChoice, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_ASCII'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_ASCII, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel_Save_ASCII'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancel_SAVE_ASCII, Event
    end
    else:
  endcase

end

pro WID_BASE_Save_Data_ASCII, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Save_data_ASCII_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Save_Data_ASCII = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Save_Data_ASCII' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=535 ,SCR_YSIZE=362  $
      ,NOTIFY_REALIZE='Initialize_Save_Data_ASCII' ,TITLE='Save Data'+ $
      ' into Tab Delimited ASCII file' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_TEXT_ASCII_Filename = Widget_Text(WID_BASE_Save_Data_ASCII,  $
      UNAME='WID_TEXT_ASCII_Filename' ,XOFFSET=6 ,YOFFSET=40  $
      ,SCR_XSIZE=511 ,SCR_YSIZE=44 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_TEXT_ASCII_Save_Parameter_List =  $
      Widget_Text(WID_BASE_Save_Data_ASCII,  $
      UNAME='WID_TEXT_ASCII_Save_Parameter_List' ,XOFFSET=9  $
      ,YOFFSET=195 ,SCR_XSIZE=513 ,SCR_YSIZE=57 ,/SCROLL ,/EDITABLE  $
      ,/ALL_EVENTS ,XSIZE=20 ,YSIZE=1)

  
  WID_DROPLIST_Save_ASCII_Filter =  $
      Widget_Droplist(WID_BASE_Save_Data_ASCII,  $
      UNAME='WID_DROPLIST_Save_ASCII_Filter' ,XOFFSET=21 ,YOFFSET=95  $
      ,SCR_XSIZE=170 ,SCR_YSIZE=30 ,TITLE='Filter' ,VALUE=[ 'Frame'+ $
      ' Peaks', 'Grouped Peaks' ])

  
  WID_DROPLIST_Save_ASCII_XY =  $
      Widget_Droplist(WID_BASE_Save_Data_ASCII,  $
      UNAME='WID_DROPLIST_Save_ASCII_XY' ,XOFFSET=306 ,YOFFSET=95  $
      ,SCR_XSIZE=170 ,SCR_YSIZE=30 ,TITLE='X-Y  Coord. Units'  $
      ,VALUE=[ 'Pixels', 'nm' ])

  
  WID_DROPLIST_Save_ASCII_Parameters =  $
      Widget_Droplist(WID_BASE_Save_Data_ASCII,  $
      UNAME='WID_DROPLIST_Save_ASCII_Parameters' ,XOFFSET=98  $
      ,YOFFSET=140 ,SCR_XSIZE=315 ,SCR_YSIZE=30 ,TITLE='Save'+ $
      ' Parameters' ,VALUE=[ 'All', 'From the list below', 'From the'+ $
      ' list below into binary .sav' ])

  
  WID_BUTTON_Save_ASCII = Widget_Button(WID_BASE_Save_Data_ASCII,  $
      UNAME='WID_BUTTON_Save_ASCII' ,XOFFSET=70 ,YOFFSET=269  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Save')

  
  WID_BUTTON_Cancel_Save_ASCII =  $
      Widget_Button(WID_BASE_Save_Data_ASCII,  $
      UNAME='WID_BUTTON_Cancel_Save_ASCII' ,XOFFSET=276 ,YOFFSET=269  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Close')

  
  WID_LABEL_Save_ASCII_FileName_Text =  $
      Widget_Label(WID_BASE_Save_Data_ASCII,  $
      UNAME='WID_LABEL_Save_ASCII_FileName_Text' ,XOFFSET=5  $
      ,YOFFSET=23 ,SCR_XSIZE=519 ,SCR_YSIZE=18 ,/ALIGN_LEFT  $
      ,VALUE='Edit / Enter in the field below the filename (including'+ $
      ' file ext.) where you want to save the data')

  
  WID_LABEL_SAVE_ASCII_ParamList_Explanation =  $
      Widget_Label(WID_BASE_Save_Data_ASCII,  $
      UNAME='WID_LABEL_SAVE_ASCII_ParamList_Explanation' ,XOFFSET=7  $
      ,YOFFSET=176 ,SCR_XSIZE=515 ,SCR_YSIZE=17 ,/ALIGN_LEFT  $
      ,VALUE='Enter (space separated) the indecis of the parameters'+ $
      ' that you want to have saved')

  Widget_Control, /REALIZE, WID_BASE_Save_Data_ASCII

  XManager, 'WID_BASE_Save_Data_ASCII', WID_BASE_Save_Data_ASCII, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Save_data_ASCII, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Save_Data_ASCII, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
