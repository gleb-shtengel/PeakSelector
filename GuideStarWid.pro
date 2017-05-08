; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	04/19/2011 16:21.24
; 
pro WID_BASE_GuideStar_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_GuideStar'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_Guide_Star'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnWriteGuideStarXY, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Test_Guide_Star'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnTestGuideStarXY, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Close'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButtonClose, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickAncFile_MultipleGS_XY'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPick_XYGuideStarAncFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_GuideStarRadius'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Write_XY_GudeStarRadius, Event
    end
    else:
  endcase

end

pro WID_BASE_GuideStar, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'GuideStarWid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_GuideStar = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_GuideStar' ,XOFFSET=5 ,YOFFSET=5 ,SCR_XSIZE=332  $
      ,SCR_YSIZE=492 ,NOTIFY_REALIZE='Initialize_XY_GuideStar'  $
      ,TITLE='Test/Write XY Guide Star' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_BUTTON_Write_Guide_Star = Widget_Button(WID_BASE_GuideStar,  $
      UNAME='WID_BUTTON_Write_Guide_Star' ,XOFFSET=168 ,YOFFSET=362  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Write Guide'+ $
      ' Star')

  
  WID_SLIDER_XY_Fit = Widget_Slider(WID_BASE_GuideStar,  $
      UNAME='WID_SLIDER_XY_Fit' ,XOFFSET=13 ,YOFFSET=50  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=48 ,TITLE='Guide Star Polyn. Fit'+ $
      ' Order' ,MINIMUM=1 ,MAXIMUM=10 ,VALUE=5)

  
  WID_BUTTON_Test_Guide_Star = Widget_Button(WID_BASE_GuideStar,  $
      UNAME='WID_BUTTON_Test_Guide_Star' ,XOFFSET=20 ,YOFFSET=362  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Test Guide'+ $
      ' Star')

  
  WID_SLIDER_XY_Sm_Width = Widget_Slider(WID_BASE_GuideStar,  $
      UNAME='WID_SLIDER_XY_Sm_Width' ,XOFFSET=14 ,YOFFSET=110  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=48 ,TITLE='Guide Star Smoothing'+ $
      ' Width' ,MINIMUM=1 ,MAXIMUM=1000 ,VALUE=100)

  
  WID_DROPLIST_XY_Fit_Method = Widget_Droplist(WID_BASE_GuideStar,  $
      UNAME='WID_DROPLIST_XY_Fit_Method' ,XOFFSET=15 ,YOFFSET=16  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=30 ,TITLE='GuideStar Drift'+ $
      ' Correction' ,VALUE=[ 'Polynomial', 'Weighted Smoothing' ])

  
  WID_BUTTON_Close = Widget_Button(WID_BASE_GuideStar,  $
      UNAME='WID_BUTTON_Close' ,XOFFSET=90 ,YOFFSET=414  $
      ,SCR_XSIZE=145 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Close')

  
  WID_TEXT_XY_GuideStarAncFilename = Widget_Text(WID_BASE_GuideStar,  $
      UNAME='WID_TEXT_XY_GuideStarAncFilename' ,XOFFSET=8  $
      ,YOFFSET=231 ,SCR_XSIZE=296 ,SCR_YSIZE=49 ,/EDITABLE ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_BUTTON_PickAncFile_MultipleGS_XY =  $
      Widget_Button(WID_BASE_GuideStar,  $
      UNAME='WID_BUTTON_PickAncFile_MultipleGS_XY' ,XOFFSET=160  $
      ,YOFFSET=194 ,SCR_XSIZE=150 ,SCR_YSIZE=32 ,/ALIGN_CENTER  $
      ,VALUE='Pick ANC File')

  
  WID_BASE_XY_Multiple_Guidestars = Widget_Base(WID_BASE_GuideStar,  $
      UNAME='WID_BASE_XY_Multiple_Guidestars' ,XOFFSET=14  $
      ,YOFFSET=198 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_UseMultipleGuideStars_XY =  $
      Widget_Button(WID_BASE_XY_Multiple_Guidestars,  $
      UNAME='WID_BUTTON_UseMultipleGuideStars_XY' ,/ALIGN_LEFT  $
      ,VALUE='Use Multiple GuideStars')

  
  WID_BUTTON_Write_GuideStarRadius =  $
      Widget_Button(WID_BASE_GuideStar,  $
      UNAME='WID_BUTTON_Write_GuideStarRadius' ,XOFFSET=9  $
      ,YOFFSET=296 ,SCR_XSIZE=218 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Set Guidestar Area Radius (pix)')

  
  WID_TEXT_XY_GuideStar_Radius = Widget_Text(WID_BASE_GuideStar,  $
      UNAME='WID_TEXT_XY_GuideStar_Radius' ,XOFFSET=237 ,YOFFSET=297  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  Widget_Control, /REALIZE, WID_BASE_GuideStar

  XManager, 'WID_BASE_GuideStar', WID_BASE_GuideStar, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro GuideStarWid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_GuideStar, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
