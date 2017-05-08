; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	08/25/2016 11:43.35
; 
pro WID_BASE_Process_SRM_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Process_SRM'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Start_SRM_processing'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Start_SRM_Processing, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel_SRM_processing'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancelReExtract, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Pick_SRM_File'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPick_SRM_File, Event
    end
    else:
  endcase

end

pro WID_BASE_Process_SRM, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Process_SRM_Wid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Process_SRM = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Process_SRM' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=674 ,SCR_YSIZE=162  $
      ,NOTIFY_REALIZE='Initialize_Process_SRM' ,TITLE='Process SRM'+ $
      ' File and create data set for PeakSelector' ,SPACE=3 ,XPAD=3  $
      ,YPAD=3)

  
  WID_BUTTON_Start_SRM_processing =  $
      Widget_Button(WID_BASE_Process_SRM,  $
      UNAME='WID_BUTTON_Start_SRM_processing' ,XOFFSET=253  $
      ,YOFFSET=63 ,SCR_XSIZE=150 ,SCR_YSIZE=55 ,/ALIGN_CENTER  $
      ,VALUE='Confirm and Start')

  
  WID_BUTTON_Cancel_SRM_processing =  $
      Widget_Button(WID_BASE_Process_SRM,  $
      UNAME='WID_BUTTON_Cancel_SRM_processing' ,XOFFSET=413  $
      ,YOFFSET=63 ,SCR_XSIZE=150 ,SCR_YSIZE=55 ,/ALIGN_CENTER  $
      ,VALUE='Cancel')

  
  WID_BUTTON_Pick_SRM_File = Widget_Button(WID_BASE_Process_SRM,  $
      UNAME='WID_BUTTON_Pick_SRM_File' ,XOFFSET=509 ,YOFFSET=10  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick SRM'+ $
      ' File')

  
  WID_TEXT_SRM_Filename = Widget_Text(WID_BASE_Process_SRM,  $
      UNAME='WID_TEXT_SRM_Filename' ,XOFFSET=4 ,YOFFSET=9  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BASE_0 = Widget_Base(WID_BASE_Process_SRM, UNAME='WID_BASE_0'  $
      ,XOFFSET=47 ,YOFFSET=75 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_Make_DAT_Duplicates = Widget_Button(WID_BASE_0,  $
      UNAME='WID_Make_DAT_Duplicates' ,/ALIGN_LEFT ,VALUE='Make'+ $
      ' Duplicate .DAT files')

  Widget_Control, /REALIZE, WID_BASE_Process_SRM

  XManager, 'WID_BASE_Process_SRM', WID_BASE_Process_SRM, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Process_SRM_Wid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Process_SRM, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
