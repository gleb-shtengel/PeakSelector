; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	10/03/2011 13:29.10
; 
pro WID_BASE_SaveZslicesTIFF_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_SaveZslicesTIFF'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_ZslicesTIFF_OK'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Zslices_TIFF_Start, Event
    end
    else:
  endcase

end

pro WID_BASE_SaveZslicesTIFF, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

; Floating or modal bases must have a Group Leader.

  if(N_Elements(wGroup) eq 0)then $
     Message,'Group leader must be specified for Modal or Floating'+ $
      ' top level bases'

  Resolve_Routine, 'SaveZslicesTIFF_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_SaveZslicesTIFF = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_SaveZslicesTIFF' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=357 ,SCR_YSIZE=283  $
      ,NOTIFY_REALIZE='Initialize_ZslicesTIFF' ,TITLE='Save Z-slices'+ $
      ' into Individual TIFF files' ,SPACE=3 ,XPAD=3 ,YPAD=3 ,/MODAL)

  
  WID_DROPLIST_Normalization =  $
      Widget_Droplist(WID_BASE_SaveZslicesTIFF,  $
      UNAME='WID_DROPLIST_Normalization' ,XOFFSET=20 ,YOFFSET=12  $
      ,SCR_XSIZE=250 ,SCR_YSIZE=22 ,TITLE='Image Normalization'  $
      ,VALUE=[ 'Compound Image (fast)', 'Run Z-slices twice (slow)'  $
      ])

  
  WID_BUTTON_ZslicesTIFF_OK = Widget_Button(WID_BASE_SaveZslicesTIFF,  $
      UNAME='WID_BUTTON_ZslicesTIFF_OK' ,XOFFSET=82 ,YOFFSET=220  $
      ,SCR_XSIZE=193 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Confirm and'+ $
      ' Start')

  
  WID_LABEL_Zstart = Widget_Label(WID_BASE_SaveZslicesTIFF,  $
      UNAME='WID_LABEL_Zstart' ,XOFFSET=60 ,YOFFSET=50 ,SCR_XSIZE=75  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Z start (nm)')

  
  WID_TEXT_Zstart = Widget_Text(WID_BASE_SaveZslicesTIFF,  $
      UNAME='WID_TEXT_Zstart' ,XOFFSET=150 ,YOFFSET=45 ,SCR_XSIZE=100  $
      ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '0.00' ] ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_TEXT_Zstop = Widget_Text(WID_BASE_SaveZslicesTIFF,  $
      UNAME='WID_TEXT_Zstop' ,XOFFSET=150 ,YOFFSET=85 ,SCR_XSIZE=100  $
      ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '250.00' ] ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_LABEL_Zstop = Widget_Label(WID_BASE_SaveZslicesTIFF,  $
      UNAME='WID_LABEL_Zstop' ,XOFFSET=60 ,YOFFSET=90 ,SCR_XSIZE=75  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Z stop (nm)')

  
  WID_LABEL_Zstep = Widget_Label(WID_BASE_SaveZslicesTIFF,  $
      UNAME='WID_LABEL_Zstep' ,XOFFSET=60 ,YOFFSET=130 ,SCR_XSIZE=75  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Z step (nm)')

  
  WID_TEXT_Zstep = Widget_Text(WID_BASE_SaveZslicesTIFF,  $
      UNAME='WID_TEXT_Zstep' ,XOFFSET=150 ,YOFFSET=125 ,SCR_XSIZE=100  $
      ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '5.00' ] ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_LABEL_Status = Widget_Label(WID_BASE_SaveZslicesTIFF,  $
      UNAME='WID_LABEL_Status' ,XOFFSET=10 ,YOFFSET=177 ,SCR_XSIZE=35  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Status')

  
  WID_TEXT_Status = Widget_Text(WID_BASE_SaveZslicesTIFF,  $
      UNAME='WID_TEXT_Status' ,XOFFSET=59 ,YOFFSET=170 ,SCR_XSIZE=286  $
      ,SCR_YSIZE=35 ,/EDITABLE ,/WRAP ,VALUE=[ 'Press ' ] ,XSIZE=20  $
      ,YSIZE=2)

  Widget_Control, /REALIZE, WID_BASE_SaveZslicesTIFF

  XManager, 'WID_BASE_SaveZslicesTIFF', WID_BASE_SaveZslicesTIFF, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro SaveZslicesTIFF, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_SaveZslicesTIFF, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
