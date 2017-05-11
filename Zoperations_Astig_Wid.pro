; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	05/11/2017 15:08.13
; 
pro WID_BASE_Z_operations_Astig_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Z_operations_Astig'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_ExtractZ_Astig'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnExtractZCoord_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Add_Offset_Astig'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAddOffset_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Close'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButtonClose_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickCalFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickCalFile_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_Guide_Star_Astig'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnWriteZDrift_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Test_Guide_Star_Astig'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnTestZDrift_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Remove_Tilt_Astig'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnRemoveTilt_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Test_EllipticityOnly'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ExtractEllipticityCalib_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_EllipticityAndWind'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSaveEllipticityCal_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickAncFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickGuideStarAncFile_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_GuideStarRadius'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        WriteGudeStarRadius_Astig, Event
    end
    else:
  endcase

end

pro WID_BASE_Z_operations_Astig, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Zoperations_Astig_Wid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Z_operations_Astig = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Z_operations_Astig' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=601 ,SCR_YSIZE=513  $
      ,NOTIFY_REALIZE='Initialize_Z_operations_Astig'  $
      ,TITLE='Z-coordinate Operations (Astigmatism Only - no'+ $
      ' Interference)' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_SLIDER_Z_phase_offset =  $
      Widget_Slider(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_SLIDER_Z_phase_offset' ,XOFFSET=25 ,YOFFSET=171  $
      ,SCR_XSIZE=167 ,SCR_YSIZE=50 ,TITLE='Z offset (nm)'  $
      ,MINIMUM=-200 ,MAXIMUM=200 ,VALUE=0)

  
  WID_BUTTON_ExtractZ_Astig =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_ExtractZ_Astig' ,XOFFSET=427 ,YOFFSET=37  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Extract Z'+ $
      ' coordinate')

  
  WID_BUTTON_Add_Offset_Astig =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Add_Offset_Astig' ,XOFFSET=206 ,YOFFSET=177  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Add'+ $
      ' Offset')

  
  WID_BUTTON_Close = Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Close' ,XOFFSET=416 ,YOFFSET=420  $
      ,SCR_XSIZE=145 ,SCR_YSIZE=45 ,/ALIGN_CENTER ,VALUE='Close')

  
  WID_BUTTON_PickCalFile = Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_PickCalFile' ,XOFFSET=427 ,SCR_XSIZE=150  $
      ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Pick CAL (WND) File')

  
  WID_TEXT_WindFilename_Astig =  $
      Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_WindFilename_Astig' ,XOFFSET=6 ,YOFFSET=4  $
      ,SCR_XSIZE=380 ,SCR_YSIZE=49 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BUTTON_Write_Guide_Star_Astig =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Write_Guide_Star_Astig' ,XOFFSET=408  $
      ,YOFFSET=360 ,SCR_XSIZE=130 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Write Guide Star')

  
  WID_BUTTON_Test_Guide_Star_Astig =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Test_Guide_Star_Astig' ,XOFFSET=408  $
      ,YOFFSET=320 ,SCR_XSIZE=130 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Test Guide Star')

  
  WID_SLIDER_Z_Sm_Width = Widget_Slider(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_SLIDER_Z_Sm_Width' ,XOFFSET=33 ,YOFFSET=410  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=55 ,TITLE='Guide Star Smoothing'+ $
      ' Width' ,MINIMUM=1 ,MAXIMUM=1000 ,VALUE=250)

  
  WID_SLIDER_Z_Fit = Widget_Slider(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_SLIDER_Z_Fit' ,XOFFSET=34 ,YOFFSET=345  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=55 ,TITLE='Guide Star Polyn. Fit'+ $
      ' Order' ,MINIMUM=1 ,MAXIMUM=10 ,VALUE=5)

  
  WID_DROPLIST_Z_Fit_Method =  $
      Widget_Droplist(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_DROPLIST_Z_Fit_Method' ,XOFFSET=48 ,YOFFSET=315  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=30 ,TITLE='GuideStar Drift'+ $
      ' Correction' ,VALUE=[ 'Polynomial', 'Weighted Smoothing' ])

  
  WID_BUTTON_Remove_Tilt_Astig =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Remove_Tilt_Astig' ,XOFFSET=409 ,YOFFSET=177  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Remove XYZ'+ $
      ' Tilt')

  
  WID_BUTTON_Test_EllipticityOnly =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Test_EllipticityOnly' ,XOFFSET=60  $
      ,YOFFSET=130 ,SCR_XSIZE=180 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Test Ellipticity Calibration')

  
  WID_BUTTON_Save_EllipticityAndWind =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Save_EllipticityAndWind' ,XOFFSET=260  $
      ,YOFFSET=130 ,SCR_XSIZE=180 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Save Ellipticity Calibration')

  
  WID_TEXT_GuideStarAncFilename_Astig =  $
      Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_GuideStarAncFilename_Astig' ,XOFFSET=15  $
      ,YOFFSET=224 ,SCR_XSIZE=344 ,SCR_YSIZE=49 ,/EDITABLE ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_BUTTON_PickAncFile = Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_PickAncFile' ,XOFFSET=380 ,YOFFSET=250  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Pick ANC'+ $
      ' File')

  
  WID_BASE_WriteEllipticityGuideStar_1 =  $
      Widget_Base(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BASE_WriteEllipticityGuideStar_1' ,XOFFSET=379  $
      ,YOFFSET=284 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_UseMultipleANCs =  $
      Widget_Button(WID_BASE_WriteEllipticityGuideStar_1,  $
      UNAME='WID_BUTTON_UseMultipleANCs' ,/ALIGN_LEFT ,VALUE='Use'+ $
      ' Multiple GuideStars')

  
  WID_BUTTON_Write_GuideStarRadius =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Write_GuideStarRadius' ,XOFFSET=19  $
      ,YOFFSET=278 ,SCR_XSIZE=250 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Set Guidestar Area Radius (pix)')

  
  WID_TEXT_GuideStar_Radius_Astig =  $
      Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_GuideStar_Radius_Astig' ,XOFFSET=272  $
      ,YOFFSET=278 ,SCR_XSIZE=70 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_TEXT_ZCalStep_Astig = Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_ZCalStep_Astig' ,XOFFSET=496 ,YOFFSET=82  $
      ,SCR_XSIZE=74 ,SCR_YSIZE=35 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_LABEL_Cal_Step = Widget_Label(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_LABEL_Cal_Step' ,XOFFSET=388 ,YOFFSET=88  $
      ,SCR_XSIZE=112 ,SCR_YSIZE=27 ,/ALIGN_LEFT ,VALUE='Cal. Z Step'+ $
      ' (nm)')

  
  WID_SLIDER_Zastig_Fit = Widget_Slider(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_SLIDER_Zastig_Fit' ,XOFFSET=33 ,YOFFSET=52  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=55 ,TITLE='Polyn. Fit Order for'+ $
      ' Astigmatism vs Z' ,MINIMUM=1 ,MAXIMUM=10 ,VALUE=5)

  Widget_Control, /REALIZE, WID_BASE_Z_operations_Astig

  XManager, 'WID_BASE_Z_operations_Astig', WID_BASE_Z_operations_Astig, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Zoperations_Astig_Wid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Z_operations_Astig, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
