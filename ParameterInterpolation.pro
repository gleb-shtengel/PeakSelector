; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	12/03/2010 15:31.00
; 
pro WID_BASE_ParameterInterpolation_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_ParameterInterpolation'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_XY_Interp'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnWrite_XY_Interp, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Test_Interp'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnTest_XY_Interpolation, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Close_XY_Interp'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButtonClose_XY_Interp, Event
    end
    else:
  endcase

end

pro WID_BASE_ParameterInterpolation, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'ParameterInterpolation_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_ParameterInterpolation = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_ParameterInterpolation' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=337 ,SCR_YSIZE=362  $
      ,NOTIFY_REALIZE='Initialize_XY_Interp_Menu' ,TITLE='Interpolate'+ $
      ' / Subtract Trend' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_BUTTON_Write_XY_Interp =  $
      Widget_Button(WID_BASE_ParameterInterpolation,  $
      UNAME='WID_BUTTON_Write_XY_Interp' ,XOFFSET=165 ,YOFFSET=240  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Subtract'+ $
      ' Interpolation')

  
  WID_SLIDER_XY_Poly_Interp_Order =  $
      Widget_Slider(WID_BASE_ParameterInterpolation,  $
      UNAME='WID_SLIDER_XY_Poly_Interp_Order' ,XOFFSET=15  $
      ,YOFFSET=110 ,SCR_XSIZE=300 ,SCR_YSIZE=48 ,TITLE='Polyn.'+ $
      ' Interp. Order' ,MINIMUM=1 ,MAXIMUM=10 ,VALUE=5)

  
  WID_BUTTON_Test_Interp =  $
      Widget_Button(WID_BASE_ParameterInterpolation,  $
      UNAME='WID_BUTTON_Test_Interp' ,XOFFSET=10 ,YOFFSET=240  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Test'+ $
      ' Interpolation')

  
  WID_SLIDER_XY_Interp_Sm_Width =  $
      Widget_Slider(WID_BASE_ParameterInterpolation,  $
      UNAME='WID_SLIDER_XY_Interp_Sm_Width' ,XOFFSET=15 ,YOFFSET=170  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=48 ,TITLE='Smoothing Width'  $
      ,MINIMUM=1 ,MAXIMUM=1000 ,VALUE=100)

  
  WID_DROPLIST_XY_Interp_Method =  $
      Widget_Droplist(WID_BASE_ParameterInterpolation,  $
      UNAME='WID_DROPLIST_XY_Interp_Method' ,XOFFSET=10 ,YOFFSET=75  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=30 ,TITLE='Interp. Method' ,VALUE=[  $
      'Polynomial', 'Weighted Smoothing' ])

  
  WID_BUTTON_Close_XY_Interp =  $
      Widget_Button(WID_BASE_ParameterInterpolation,  $
      UNAME='WID_BUTTON_Close_XY_Interp' ,XOFFSET=83 ,YOFFSET=287  $
      ,SCR_XSIZE=145 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Close')

  
  WID_DROPLIST_Y_Interp =  $
      Widget_Droplist(WID_BASE_ParameterInterpolation,  $
      UNAME='WID_DROPLIST_Y_Interp' ,XOFFSET=50 ,YOFFSET=40  $
      ,SCR_XSIZE=250 ,SCR_YSIZE=25 ,TITLE='Y Axis' ,VALUE=[ 'Offset',  $
      'Amplitude', 'X Position', 'Y Position', 'X Peak Width', 'Y'+ $
      ' Peak Width', '6 N Photons', 'ChiSquared', 'FitOK', 'Frame'+ $
      ' Number', 'Peak Index of Frame', 'Peak Global Index', '12'+ $
      ' Sigma Offset', 'Sigma Amplitude', 'Sigma X Pos rtNph', 'Sigma'+ $
      ' Y Pos rtNph', 'Sigma X Pos Full', 'Sigma Y Pos Full', '18'+ $
      ' Grouped Index', 'Group X Position', 'Group Y Position',  $
      'Group Sigma X Pos', 'Group Sigma Y Pos', 'Group N Photons',  $
      '24 Group Size', 'Frame Index in Grp', 'Label Set' ])

  
  WID_DROPLIST_X_Interp =  $
      Widget_Droplist(WID_BASE_ParameterInterpolation,  $
      UNAME='WID_DROPLIST_X_Interp' ,XOFFSET=50 ,YOFFSET=5  $
      ,SCR_XSIZE=250 ,SCR_YSIZE=25 ,TITLE='X Axis' ,VALUE=[ 'Offset',  $
      'Amplitude', 'X Position', 'Y Position', 'X Peak Width', 'Y'+ $
      ' Peak Width', '6 N Photons', 'ChiSquared', 'FitOK', 'Frame'+ $
      ' Number', 'Peak Index of Frame', 'Peak Global Index', '12'+ $
      ' Sigma Offset', 'Sigma Amplitude', 'Sigma X Pos rtNph', 'Sigma'+ $
      ' Y Pos rtNph', 'Sigma X Pos Full', 'Sigma Y Pos Full', '18'+ $
      ' Grouped Index', 'Group X Position', 'Group Y Position',  $
      'Group Sigma X Pos', 'Group Sigma Y Pos', 'Group N Photons',  $
      '24 Group Size', 'Frame Index in Grp', 'Label Set' ])

  Widget_Control, /REALIZE, WID_BASE_ParameterInterpolation

  XManager, 'WID_BASE_ParameterInterpolation', WID_BASE_ParameterInterpolation, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro ParameterInterpolation, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_ParameterInterpolation, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
