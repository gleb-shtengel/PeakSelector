; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	07/01/2011 09:10.35
; 
pro WID_BASE_AnalyzeMultiplePeaks_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_AnalyzeMultiplePeaks'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_AnalyzeMultiple_OK'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_AnalyzeMultiple_Start, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickPeaksFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickPeaksFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Analyze_Multiple_Filter'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        On_Select_Peaks_Filter, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Save_Peak_ASCII_Parameters'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        On_Select_Peak_SaveASCII_ParamChoice, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Save_Peak_ASCII_XY'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        On_Select_Peak_SaveASCII_units, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_ASCII_Peak_Save_Parameter_List'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        On_ASCII_Peak_ParamList_change, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_STR' )then $
        On_ASCII_Peak_ParamList_change, Event
    end
    else:
  endcase

end

pro WID_BASE_AnalyzeMultiplePeaks, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

; Floating or modal bases must have a Group Leader.

  if(N_Elements(wGroup) eq 0)then $
     Message,'Group leader must be specified for Modal or Floating'+ $
      ' top level bases'

  Resolve_Routine, 'AnalyzeMultiplePeaks_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_AnalyzeMultiplePeaks = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_AnalyzeMultiplePeaks' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=566 ,SCR_YSIZE=555  $
      ,NOTIFY_REALIZE='Initialize_AnalizeMultiplePeaks'  $
      ,TITLE='Analyze Multiple Peaks' ,SPACE=3 ,XPAD=3 ,YPAD=3  $
      ,/MODAL)

  
  WID_BUTTON_AnalyzeMultiple_OK =  $
      Widget_Button(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_BUTTON_AnalyzeMultiple_OK' ,XOFFSET=79 ,YOFFSET=460  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Confirm and'+ $
      ' Start')

  
  WID_LABEL_Xrange = Widget_Label(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_LABEL_Xrange' ,XOFFSET=48 ,YOFFSET=110 ,SCR_XSIZE=85  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='X Range (pixels)')

  
  WID_TEXT_Xrange = Widget_Text(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_TEXT_Xrange' ,XOFFSET=149 ,YOFFSET=100  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '1.5' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_TEXT_Yrange = Widget_Text(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_TEXT_Yrange' ,XOFFSET=149 ,YOFFSET=135  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '1.5' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_LABEL_YRange = Widget_Label(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_LABEL_YRange' ,XOFFSET=49 ,YOFFSET=145 ,SCR_XSIZE=85  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Y Range (pixels)')

  
  WID_LABEL_Status = Widget_Label(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_LABEL_Status' ,XOFFSET=189 ,YOFFSET=470  $
      ,SCR_XSIZE=35 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Status')

  
  WID_TEXT_Status = Widget_Text(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_TEXT_Status' ,XOFFSET=224 ,YOFFSET=460  $
      ,SCR_XSIZE=190 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ 'Press'+ $
      ' ' ] ,XSIZE=20 ,YSIZE=2)

  
  WID_TEXT_PeaksFilename = Widget_Text(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_TEXT_PeaksFilename' ,XOFFSET=26 ,YOFFSET=49  $
      ,SCR_XSIZE=331 ,SCR_YSIZE=48 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BUTTON_PickPeaksFile =  $
      Widget_Button(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_BUTTON_PickPeaksFile' ,XOFFSET=200 ,YOFFSET=12  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick *.anc'+ $
      ' File')

  
  WID_LABEL_anc_explanation =  $
      Widget_Label(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_LABEL_anc_explanation' ,XOFFSET=10 ,YOFFSET=14  $
      ,SCR_XSIZE=177 ,SCR_YSIZE=23 ,/ALIGN_LEFT ,VALUE='Use *.anc'+ $
      ' file to list peaks')

  
  WID_LABEL_MinNpeaks = Widget_Label(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_LABEL_MinNpeaks' ,XOFFSET=48 ,YOFFSET=180  $
      ,SCR_XSIZE=85 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Min # of'+ $
      ' Peaks')

  
  WID_TEXT_MinNPeaks = Widget_Text(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_TEXT_MinNPeaks' ,XOFFSET=149 ,YOFFSET=170  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '200' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_DROPLIST_Analyze_Multiple_Filter =  $
      Widget_Droplist(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_DROPLIST_Analyze_Multiple_Filter' ,XOFFSET=120  $
      ,YOFFSET=220 ,SCR_XSIZE=170 ,SCR_YSIZE=30 ,TITLE='Filter'  $
      ,VALUE=[ 'Frame Peaks', 'Grouped Peaks' ])

  
  WID_LABEL_SAVE_ASCII_ParamList_Explanation =  $
      Widget_Label(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_LABEL_SAVE_ASCII_ParamList_Explanation' ,XOFFSET=72  $
      ,YOFFSET=349 ,SCR_XSIZE=515 ,SCR_YSIZE=17 ,/ALIGN_LEFT  $
      ,VALUE='Enter (space separated) the indecis of the parameters'+ $
      ' that you want to have saved')

  
  WID_DROPLIST_Save_Peak_ASCII_Parameters =  $
      Widget_Droplist(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_DROPLIST_Save_Peak_ASCII_Parameters' ,XOFFSET=163  $
      ,YOFFSET=313 ,SCR_XSIZE=315 ,SCR_YSIZE=30 ,TITLE='Save'+ $
      ' Parameters' ,VALUE=[ 'All', 'From the list below' ])

  
  WID_DROPLIST_Save_Peak_ASCII_XY =  $
      Widget_Droplist(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_DROPLIST_Save_Peak_ASCII_XY' ,XOFFSET=371  $
      ,YOFFSET=268 ,SCR_XSIZE=170 ,SCR_YSIZE=30 ,TITLE='X-Y  Coord.'+ $
      ' Units' ,VALUE=[ 'Pixels', 'nm' ])

  
  WID_TEXT_ASCII_Peak_Save_Parameter_List =  $
      Widget_Text(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_TEXT_ASCII_Peak_Save_Parameter_List' ,XOFFSET=16  $
      ,YOFFSET=368 ,SCR_XSIZE=513 ,SCR_YSIZE=57 ,/SCROLL ,/EDITABLE  $
      ,/ALL_EVENTS ,XSIZE=20 ,YSIZE=1)

  
  WID_BASE_Save_Each_Peak_Distribution =  $
      Widget_Base(WID_BASE_AnalyzeMultiplePeaks,  $
      UNAME='WID_BASE_Save_Each_Peak_Distribution' ,XOFFSET=20  $
      ,YOFFSET=267 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Save_Each_Peak_Distribution =  $
      Widget_Button(WID_BASE_Save_Each_Peak_Distribution,  $
      UNAME='WID_BUTTON_Save_Each_Peak_Distribution' ,/ALIGN_LEFT  $
      ,VALUE='Save Each Peak Distribution into Sepearte ASCII file')

  Widget_Control, /REALIZE, WID_BASE_AnalyzeMultiplePeaks

  XManager, 'WID_BASE_AnalyzeMultiplePeaks', WID_BASE_AnalyzeMultiplePeaks, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro AnalyzeMultiplePeaks, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_AnalyzeMultiplePeaks, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
