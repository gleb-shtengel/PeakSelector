; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	12/02/2010 15:25.38
; 
pro WID_BASE_PeakSelector_INI_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_PeakSelector_INI'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_INI'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_INI_File, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Load_ANC'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Load_INI_File, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Pick_INI'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickINIFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel_INI_Edit'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancel_INI_Edit, Event
    end
    else:
  endcase

end

pro WID_BASE_PeakSelector_INI, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

; Floating or modal bases must have a Group Leader.

  if(N_Elements(wGroup) eq 0)then $
     Message,'Group leader must be specified for Modal or Floating'+ $
      ' top level bases'

  Resolve_Routine, 'EditPeakSelectorIni_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_PeakSelector_INI = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_PeakSelector_INI' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=917 ,SCR_YSIZE=726  $
      ,NOTIFY_REALIZE='DoRealize_PeakSelector_INI'  $
      ,TITLE='PeakSelector Settings' ,SPACE=3 ,XPAD=3 ,YPAD=3  $
      ,/MODAL)

  
  WID_BUTTON_Save_INI = Widget_Button(WID_BASE_PeakSelector_INI,  $
      UNAME='WID_BUTTON_Save_INI' ,XOFFSET=450 ,YOFFSET=600  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Force'+ $
      ' Settings + Save (*.ini)')

  
  WID_BUTTON_Load_ANC = Widget_Button(WID_BASE_PeakSelector_INI,  $
      UNAME='WID_BUTTON_Load_ANC' ,XOFFSET=230 ,YOFFSET=600  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Load'+ $
      ' (*.ini) + Force Settings')

  
  WID_TEXT_INI_Filename = Widget_Text(WID_BASE_PeakSelector_INI,  $
      UNAME='WID_TEXT_INI_Filename' ,XOFFSET=25 ,YOFFSET=646  $
      ,SCR_XSIZE=868 ,SCR_YSIZE=40 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BUTTON_Pick_INI = Widget_Button(WID_BASE_PeakSelector_INI,  $
      UNAME='WID_BUTTON_Pick_INI' ,XOFFSET=30 ,YOFFSET=600  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Pick *.ini'+ $
      ' File')

  
  WID_TEXT_PeakSelector_INI = Widget_Text(WID_BASE_PeakSelector_INI,  $
      UNAME='WID_TEXT_PeakSelector_INI' ,XOFFSET=10 ,YOFFSET=8  $
      ,SCR_XSIZE=880 ,SCR_YSIZE=577 ,/SCROLL ,/EDITABLE ,XSIZE=20  $
      ,YSIZE=1)

  
  WID_BUTTON_Cancel_INI_Edit =  $
      Widget_Button(WID_BASE_PeakSelector_INI,  $
      UNAME='WID_BUTTON_Cancel_INI_Edit' ,XOFFSET=700 ,YOFFSET=600  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Close')

  Widget_Control, /REALIZE, WID_BASE_PeakSelector_INI

  XManager, 'WID_BASE_PeakSelector_INI', WID_BASE_PeakSelector_INI, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro EditPeakSelectorIni, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_PeakSelector_INI, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
