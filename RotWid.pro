
;
; IDL Widget Interface Procedures. This Code is automatically
;     generated and should not be modified.

;
; Generated on:	11/12/2007 10:29.33
;
pro WID_BASE_Rot_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Rot'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Rotate'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnRotate, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancel, Event
    end
    else:
  endcase

end

pro WID_BASE_Rot, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'RotWid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines

  WID_BASE_Rot = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Rot' ,XOFFSET=5 ,YOFFSET=5 ,SCR_XSIZE=399  $
      ,SCR_YSIZE=203 ,NOTIFY_REALIZE='OnRealizeRotz'  $
      ,TITLE='Rotation' ,SPACE=3 ,XPAD=3 ,YPAD=3)


  WID_SLIDER_Rot = Widget_Slider(WID_BASE_Rot, UNAME='WID_SLIDER_Rot'  $
      ,XOFFSET=50 ,YOFFSET=40 ,SCR_XSIZE=300 ,SCR_YSIZE=50  $
      ,TITLE='Angle (degrees)' ,MINIMUM=-90 ,MAXIMUM=90 ,VALUE=0)


  WID_BUTTON_Rotate = Widget_Button(WID_BASE_Rot,  $
      UNAME='WID_BUTTON_Rotate' ,XOFFSET=47 ,YOFFSET=113  $
      ,SCR_XSIZE=112 ,SCR_YSIZE=36 ,/ALIGN_CENTER ,VALUE='Rotate')


  WID_BUTTON_Cancel = Widget_Button(WID_BASE_Rot,  $
      UNAME='WID_BUTTON_Cancel' ,XOFFSET=201 ,YOFFSET=113  $
      ,SCR_XSIZE=112 ,SCR_YSIZE=36 ,/ALIGN_CENTER ,VALUE='Cancel')

  Widget_Control, /REALIZE, WID_BASE_Rot

  XManager, 'WID_BASE_Rot', WID_BASE_Rot, /NO_BLOCK

end
;
; Empty stub procedure used for autoloading.
;
pro RotWid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Rot, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
