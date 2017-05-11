; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	07/23/2009 06:46.24
; 
pro WID_BASE_Recalc_PeakW_Product_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Recalc_PeakW_Product'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Recalculate'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnRecalculate, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancel, Event
    end
    else:
  endcase

end

pro WID_BASE_Recalc_PeakW_Product, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

; Floating or modal bases must have a Group Leader.

  if(N_Elements(wGroup) eq 0)then $
     Message,'Group leader must be specified for Modal or Floating'+ $
      ' top level bases'

  Resolve_Routine, 'Recalc_XPkWYPkW_product_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Recalc_PeakW_Product = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Recalc_PeakW_Product' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=399 ,SCR_YSIZE=203  $
      ,NOTIFY_REALIZE='Initialize_Recalculate_Menu'  $
      ,TITLE='Recalculate the Peakwidth Product' ,SPACE=3 ,XPAD=3  $
      ,YPAD=3 ,/MODAL)

  
  WID_BUTTON_Recalculate =  $
      Widget_Button(WID_BASE_Recalc_PeakW_Product,  $
      UNAME='WID_BUTTON_Recalculate' ,XOFFSET=10 ,YOFFSET=113  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=36 ,/ALIGN_CENTER ,VALUE='Recalculate'+ $
      ' CGroupParams[12,*]')

  
  WID_BUTTON_Cancel = Widget_Button(WID_BASE_Recalc_PeakW_Product,  $
      UNAME='WID_BUTTON_Cancel' ,XOFFSET=250 ,YOFFSET=113  $
      ,SCR_XSIZE=112 ,SCR_YSIZE=36 ,/ALIGN_CENTER ,VALUE='Cancel')

  
  WID_TEXT_offset_value = Widget_Text(WID_BASE_Recalc_PeakW_Product,  $
      UNAME='WID_TEXT_offset_value' ,XOFFSET=180 ,YOFFSET=57  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=32 ,/EDITABLE ,VALUE=[ '0.0' ]  $
      ,XSIZE=20 ,YSIZE=1)

  
  WID_LABEL_Offset_V = Widget_Label(WID_BASE_Recalc_PeakW_Product,  $
      UNAME='WID_LABEL_Offset_V' ,XOFFSET=130 ,YOFFSET=65  $
      ,SCR_XSIZE=48 ,SCR_YSIZE=21 ,/ALIGN_LEFT ,VALUE='Offset')

  
  WID_LABEL_0 = Widget_Label(WID_BASE_Recalc_PeakW_Product,  $
      UNAME='WID_LABEL_0' ,XOFFSET=1 ,YOFFSET=22 ,SCR_XSIZE=389  $
      ,SCR_YSIZE=23 ,/ALIGN_LEFT ,VALUE='CGroupParams[12,*] ='+ $
      ' (CGroupParams[4,*] - Offset) * (CGroupParams[5,*] - Offset)')

  Widget_Control, /REALIZE, WID_BASE_Recalc_PeakW_Product

  XManager, 'WID_BASE_Recalc_PeakW_Product', WID_BASE_Recalc_PeakW_Product, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Recalc_XPkWYPkW_product, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Recalc_PeakW_Product, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
