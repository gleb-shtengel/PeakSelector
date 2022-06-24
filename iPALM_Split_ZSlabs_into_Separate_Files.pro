; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	11/11/2021 08:51.47
; 
pro WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SplitZSlabsOK'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSplitZSlabsOK, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickFilePrefix'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickFilePrefix, Event
    end
    else:
  endcase

end

pro WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'iPALM_Split_ZSlabs_into_Separate_Files_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files = Widget_Base(  $
      GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files'  $
      ,XOFFSET=5 ,YOFFSET=5 ,SCR_XSIZE=552 ,SCR_YSIZE=386  $
      ,NOTIFY_REALIZE='Initialize_SplitZslabs_Wid' ,TITLE='Split Data'+ $
      ' into Separate Z-Slab Files' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_BUTTON_SplitZSlabsOK =  $
      Widget_Button(WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files,  $
      UNAME='WID_BUTTON_SplitZSlabsOK' ,XOFFSET=364 ,YOFFSET=249  $
      ,SCR_XSIZE=129 ,SCR_YSIZE=46 ,/ALIGN_CENTER ,VALUE='Confirm and'+ $
      ' Start')

  
  WID_SLIDER_FramesPerZSlab =  $
      Widget_Slider(WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files,  $
      UNAME='WID_SLIDER_FramesPerZSlab' ,XOFFSET=50 ,YOFFSET=250  $
      ,SCR_XSIZE=250 ,SCR_YSIZE=50 ,TITLE='Frames per Z-Slab'  $
      ,MINIMUM=1 ,MAXIMUM=1000 ,VALUE=250)

  
  WID_SLIDER_Number_of_ZSlabs =  $
      Widget_Slider(WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files,  $
      UNAME='WID_SLIDER_Number_of_ZSlabs' ,XOFFSET=50 ,YOFFSET=150  $
      ,SCR_XSIZE=250 ,SCR_YSIZE=50 ,TITLE='Number of Z-Slabs (Sample'+ $
      ' Piezo Z-levels)' ,MINIMUM=1 ,MAXIMUM=25 ,VALUE=5)

  
  WID_BUTTON_PickFilePrefix =  $
      Widget_Button(WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files,  $
      UNAME='WID_BUTTON_PickFilePrefix' ,XOFFSET=400 ,YOFFSET=40  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick File'+ $
      ' Prefix')

  
  WID_TEXT_FilePrefix =  $
      Widget_Text(WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files,  $
      UNAME='WID_TEXT_FilePrefix' ,XOFFSET=20 ,YOFFSET=40  $
      ,SCR_XSIZE=350 ,SCR_YSIZE=40 ,/EDITABLE ,/WRAP ,VALUE=[ 'Select'+ $
      ' *.anc File' ] ,XSIZE=20 ,YSIZE=2)

  Widget_Control, /REALIZE, WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files

  XManager, 'WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files', WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro iPALM_Split_ZSlabs_into_Separate_Files, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
