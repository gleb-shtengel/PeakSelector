; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	03/10/2017 13:19.01
; 
pro WID_BASE_GroupPeaks_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_GroupPeaks'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GroupingOK'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnGroupingInfoOK, Event
    end
    else:
  endcase

end

pro WID_BASE_GroupPeaks, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'GroupWid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_GroupPeaks = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_GroupPeaks' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=357 ,SCR_YSIZE=248  $
      ,NOTIFY_REALIZE='Initialize_GroupPeaks' ,TITLE='Group Peaks'  $
      ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_DROPLIST_GroupEngine = Widget_Droplist(WID_BASE_GroupPeaks,  $
      UNAME='WID_DROPLIST_GroupEngine' ,XOFFSET=57 ,YOFFSET=84  $
      ,SCR_XSIZE=181 ,SCR_YSIZE=22 ,TITLE='Grouping Engine' ,VALUE=[  $
      'Local', 'Cluster', 'IDL Bridge' ])

  
  WID_SLIDER_Grouping_Radius = Widget_Slider(WID_BASE_GroupPeaks,  $
      UNAME='WID_SLIDER_Grouping_Radius' ,XOFFSET=176 ,YOFFSET=11  $
      ,SCR_XSIZE=146 ,SCR_YSIZE=48 ,TITLE='Grouping Radius*100'  $
      ,MAXIMUM=200 ,VALUE=25)

  
  WID_SLIDER_Group_Gap = Widget_Slider(WID_BASE_GroupPeaks,  $
      UNAME='WID_SLIDER_Group_Gap' ,XOFFSET=11 ,YOFFSET=12  $
      ,SCR_XSIZE=149 ,SCR_YSIZE=48 ,TITLE='Group Gap' ,MAXIMUM=256  $
      ,VALUE=3)

  
  WID_SLIDER_FramesPerNode = Widget_Slider(WID_BASE_GroupPeaks,  $
      UNAME='WID_SLIDER_FramesPerNode' ,XOFFSET=10 ,YOFFSET=130  $
      ,SCR_XSIZE=142 ,SCR_YSIZE=48 ,TITLE='Frames per Node (Cluster)'  $
      ,MINIMUM=0 ,MAXIMUM=10000 ,VALUE=2500)

  
  WID_BUTTON_GroupingOK = Widget_Button(WID_BASE_GroupPeaks,  $
      UNAME='WID_BUTTON_GroupingOK' ,XOFFSET=195 ,YOFFSET=128  $
      ,SCR_XSIZE=129 ,SCR_YSIZE=46 ,/ALIGN_CENTER ,VALUE='Confirm and'+ $
      ' Start')

  Widget_Control, /REALIZE, WID_BASE_GroupPeaks

  XManager, 'WID_BASE_GroupPeaks', WID_BASE_GroupPeaks, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro GroupWid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_GroupPeaks, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
