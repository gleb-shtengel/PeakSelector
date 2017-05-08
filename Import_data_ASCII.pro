; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	10/21/2011 09:35.58
; 
pro WID_BASE_Import_Data_ASCII_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Import_Data_ASCII'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_Import_ASCII_Filename'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        On_ASCII_Filename_change, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_STR' )then $
        On_ASCII_Filename_change, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_ASCII_Import_Parameter_List'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        On_Import_ASCII_ParamList_change, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_STR' )then $
        On_Import_ASCII_ParamList_change, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Import_ASCII_XY'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        On_Select_ImportASCII_units, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Import_ASCII'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Import_ASCII, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel_Import_ASCII'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancel_Import_ASCII, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickASCIIFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickASCIIFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_Import_ASCII_nm_per_pixel'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        Change_Import_ASCII_nm_per_pixel, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_STR' )then $
        Change_Import_ASCII_nm_per_pixel, Event
    end
    else:
  endcase

end

pro WID_BASE_Import_Data_ASCII, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Import_data_ASCII_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Import_Data_ASCII = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Import_Data_ASCII' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=535 ,SCR_YSIZE=412  $
      ,NOTIFY_REALIZE='Initialize_Import_Data_ASCII' ,TITLE='Import'+ $
      ' Data from Tab Delimited ASCII file' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_TEXT_Import_ASCII_Filename =  $
      Widget_Text(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_TEXT_Import_ASCII_Filename' ,XOFFSET=9 ,YOFFSET=40  $
      ,SCR_XSIZE=366 ,SCR_YSIZE=44 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_TEXT_ASCII_Import_Parameter_List =  $
      Widget_Text(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_TEXT_ASCII_Import_Parameter_List' ,XOFFSET=9  $
      ,YOFFSET=195 ,SCR_XSIZE=513 ,SCR_YSIZE=57 ,/SCROLL ,/EDITABLE  $
      ,/ALL_EVENTS ,XSIZE=20 ,YSIZE=1)

  
  WID_DROPLIST_Import_ASCII_XY =  $
      Widget_Droplist(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_DROPLIST_Import_ASCII_XY' ,XOFFSET=24 ,YOFFSET=103  $
      ,SCR_XSIZE=170 ,SCR_YSIZE=30 ,TITLE='X-Y  Coord. Units'  $
      ,VALUE=[ 'Pixels', 'nm' ])

  
  WID_BUTTON_Import_ASCII = Widget_Button(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_BUTTON_Import_ASCII' ,XOFFSET=68 ,YOFFSET=320  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Import')

  
  WID_BUTTON_Cancel_Import_ASCII =  $
      Widget_Button(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_BUTTON_Cancel_Import_ASCII' ,XOFFSET=274  $
      ,YOFFSET=320 ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Close')

  
  WID_LABEL_Import_ASCII_FileName_Text =  $
      Widget_Label(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_LABEL_Import_ASCII_FileName_Text' ,XOFFSET=8  $
      ,YOFFSET=13 ,SCR_XSIZE=519 ,SCR_YSIZE=18 ,/ALIGN_LEFT  $
      ,VALUE='Pick / Edit / Enter in the field below the filename'+ $
      ' (including file ext.) of the data file')

  
  WID_LABEL_Import_ASCII_ParamList_Explanation1 =  $
      Widget_Label(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_LABEL_Import_ASCII_ParamList_Explanation1'  $
      ,XOFFSET=5 ,YOFFSET=140 ,SCR_XSIZE=515 ,SCR_YSIZE=20  $
      ,/ALIGN_LEFT ,VALUE='Enter the PeakSelector parameter indecis'+ $
      ' corresponding to the columns in your data file.')

  
  WID_BUTTON_PickASCIIFile =  $
      Widget_Button(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_BUTTON_PickASCIIFile' ,XOFFSET=395 ,YOFFSET=44  $
      ,SCR_XSIZE=120 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick ASCII'+ $
      ' File')

  
  WID_LABEL_Import_ASCII_ParamList_Explanation1_0 =  $
      Widget_Label(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_LABEL_Import_ASCII_ParamList_Explanation1_0'  $
      ,XOFFSET=5 ,YOFFSET=162 ,SCR_XSIZE=515 ,SCR_YSIZE=20  $
      ,/ALIGN_LEFT ,VALUE='If the column in your data file does not'+ $
      ' have a corresponding PeakSelector parameter, enter -1.')

  
  WID_TEXT_Import_ASCII_nm_per_pixel =  $
      Widget_Text(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_TEXT_Import_ASCII_nm_per_pixel' ,XOFFSET=424  $
      ,YOFFSET=98 ,SCR_XSIZE=70 ,SCR_YSIZE=30 ,/EDITABLE ,/ALL_EVENTS  $
      ,/WRAP ,VALUE=[ '133.33' ] ,XSIZE=20 ,YSIZE=2)

  
  WID_BASE_SkipFirstLine_ImportASCII =  $
      Widget_Base(WID_BASE_Import_Data_ASCII,  $
      UNAME='WID_BASE_SkipFirstLine_ImportASCII' ,XOFFSET=55  $
      ,YOFFSET=270 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_SkipFirstLine_ImportASCII =  $
      Widget_Button(WID_BASE_SkipFirstLine_ImportASCII,  $
      UNAME='WID_BUTTON_SkipFirstLine_ImportASCII' ,/ALIGN_LEFT  $
      ,VALUE='Skip First Line (check if titles line is present)')

  Widget_Control, /REALIZE, WID_BASE_Import_Data_ASCII

  XManager, 'WID_BASE_Import_Data_ASCII', WID_BASE_Import_Data_ASCII, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Import_data_ASCII, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Import_Data_ASCII, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
