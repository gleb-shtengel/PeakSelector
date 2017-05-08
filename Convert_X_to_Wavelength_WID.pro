; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	04/17/2015 10:35.41
; 
pro WID_BASE_Convert_X_to_Wavelength_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Convert_X_to_Wavelength'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Convert_X_to_Wavelength'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnRecalculate_Wavelength, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancel, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_offset_value'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        Change_offset_text, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_dispersion_value'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        Change_dispersion_text, Event
    end
    else:
  endcase

end

pro WID_BASE_Convert_X_to_Wavelength, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

; Floating or modal bases must have a Group Leader.

  if(N_Elements(wGroup) eq 0)then $
     Message,'Group leader must be specified for Modal or Floating'+ $
      ' top level bases'

  Resolve_Routine, 'Convert_X_to_Wavelength_WID_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Convert_X_to_Wavelength = Widget_Base(  $
      GROUP_LEADER=wGroup, UNAME='WID_BASE_Convert_X_to_Wavelength'  $
      ,XOFFSET=5 ,YOFFSET=5 ,SCR_XSIZE=399 ,SCR_YSIZE=203  $
      ,NOTIFY_REALIZE='Initialize_Convert_X_to_Wavelength'  $
      ,TITLE='Calculate Wavelength from X-coordinate' ,SPACE=3  $
      ,XPAD=3 ,YPAD=3 ,/MODAL)

  
  WID_BUTTON_Convert_X_to_Wavelength =  $
      Widget_Button(WID_BASE_Convert_X_to_Wavelength,  $
      UNAME='WID_BUTTON_Convert_X_to_Wavelength' ,XOFFSET=10  $
      ,YOFFSET=113 ,SCR_XSIZE=200 ,SCR_YSIZE=36 ,/ALIGN_CENTER  $
      ,VALUE='Recalculate CGroupParams[12,*]')

  
  WID_BUTTON_Cancel = Widget_Button(WID_BASE_Convert_X_to_Wavelength,  $
      UNAME='WID_BUTTON_Cancel' ,XOFFSET=250 ,YOFFSET=113  $
      ,SCR_XSIZE=112 ,SCR_YSIZE=36 ,/ALIGN_CENTER ,VALUE='Cancel')

  
  WID_TEXT_offset_value =  $
      Widget_Text(WID_BASE_Convert_X_to_Wavelength,  $
      UNAME='WID_TEXT_offset_value' ,XOFFSET=300 ,YOFFSET=55  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=32 ,/EDITABLE ,/ALL_EVENTS ,VALUE=[  $
      '0.0' ] ,XSIZE=20 ,YSIZE=1)

  
  WID_LABEL_Offset = Widget_Label(WID_BASE_Convert_X_to_Wavelength,  $
      UNAME='WID_LABEL_Offset' ,XOFFSET=240 ,YOFFSET=63 ,SCR_XSIZE=48  $
      ,SCR_YSIZE=20 ,/ALIGN_LEFT ,VALUE='Offset')

  
  WID_LABEL_Wavelength_Conversion =  $
      Widget_Label(WID_BASE_Convert_X_to_Wavelength,  $
      UNAME='WID_LABEL_Wavelength_Conversion' ,XOFFSET=10 ,YOFFSET=18  $
      ,SCR_XSIZE=360 ,SCR_YSIZE=23 ,/ALIGN_LEFT  $
      ,VALUE='CGroupParams[12,*] = CGroupParams[2,*] * Dispersion +'+ $
      ' Offset')

  
  WID_LABEL_Dispersion =  $
      Widget_Label(WID_BASE_Convert_X_to_Wavelength,  $
      UNAME='WID_LABEL_Dispersion' ,XOFFSET=10 ,YOFFSET=60  $
      ,SCR_XSIZE=120 ,SCR_YSIZE=20 ,/ALIGN_LEFT ,VALUE='Dispersion'+ $
      ' (nm/pix)')

  
  WID_TEXT_dispersion_value =  $
      Widget_Text(WID_BASE_Convert_X_to_Wavelength,  $
      UNAME='WID_TEXT_dispersion_value' ,XOFFSET=130 ,YOFFSET=55  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=32 ,/EDITABLE ,/ALL_EVENTS ,VALUE=[  $
      '0.0' ] ,XSIZE=20 ,YSIZE=1)

  Widget_Control, /REALIZE, WID_BASE_Convert_X_to_Wavelength

  XManager, 'WID_BASE_Convert_X_to_Wavelength', WID_BASE_Convert_X_to_Wavelength, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Convert_X_to_Wavelength_WID, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Convert_X_to_Wavelength, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
