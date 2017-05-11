; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	11/05/2010 13:26.41
; 
pro WID_BASE_AnalyzePhaseUnwrap_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_AnalyzePhaseUnwrap'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Analyze_PhaseUnwrap_OK'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Analyze_PhaseUnwrap_Start, Event
    end
    else:
  endcase

end

pro WID_BASE_AnalyzePhaseUnwrap, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'AnalyzePhaseUnwrap_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_AnalyzePhaseUnwrap = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_AnalyzePhaseUnwrap' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=348 ,SCR_YSIZE=259  $
      ,NOTIFY_REALIZE='Initialize_AnalyzePhaseUnwrap' ,TITLE='Analyze'+ $
      ' Phase Unwrapping and Localizations' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_BUTTON_Analyze_PhaseUnwrap_OK =  $
      Widget_Button(WID_BASE_AnalyzePhaseUnwrap,  $
      UNAME='WID_BUTTON_Analyze_PhaseUnwrap_OK' ,XOFFSET=115  $
      ,YOFFSET=117 ,SCR_XSIZE=120 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Confirm and Start')

  
  WID_LABEL_NumFramesPerStep =  $
      Widget_Label(WID_BASE_AnalyzePhaseUnwrap,  $
      UNAME='WID_LABEL_NumFramesPerStep' ,XOFFSET=20 ,YOFFSET=25  $
      ,SCR_XSIZE=175 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Number of'+ $
      ' Frames Per Step')

  
  WID_number_frames_per_step =  $
      Widget_Text(WID_BASE_AnalyzePhaseUnwrap,  $
      UNAME='WID_number_frames_per_step' ,XOFFSET=220 ,YOFFSET=15  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '100' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_nm_per_step = Widget_Text(WID_BASE_AnalyzePhaseUnwrap,  $
      UNAME='WID_nm_per_step' ,XOFFSET=220 ,YOFFSET=65 ,SCR_XSIZE=100  $
      ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '8' ] ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_LABEL_nm_per_step = Widget_Label(WID_BASE_AnalyzePhaseUnwrap,  $
      UNAME='WID_LABEL_nm_per_step' ,XOFFSET=25 ,YOFFSET=75  $
      ,SCR_XSIZE=175 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='NM per step')

  
  WID_TEXT_ResultsFilename = Widget_Text(WID_BASE_AnalyzePhaseUnwrap,  $
      UNAME='WID_TEXT_ResultsFilename' ,XOFFSET=14 ,YOFFSET=168  $
      ,SCR_XSIZE=311 ,SCR_YSIZE=53 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  Widget_Control, /REALIZE, WID_BASE_AnalyzePhaseUnwrap

  XManager, 'WID_BASE_AnalyzePhaseUnwrap', WID_BASE_AnalyzePhaseUnwrap, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro AnalyzePhaseUnwrap, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_AnalyzePhaseUnwrap, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
