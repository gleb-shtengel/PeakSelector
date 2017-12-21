; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	12/21/2017 11:41.15
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
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_Press_use_multiple_GS, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_GuideStarRadius'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        WriteGudeStarRadius_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Convert_Fr_to_Z'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Convert_Frame_to_Z, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UseMultipleANCs_DH'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_Press_use_multiple_GS_DH, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_ZvsFfame_woffset'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Plot_ZvsFrame_with_offest, Event
    end
    else:
  endcase

end

pro WID_BASE_Z_operations_Astig, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Zoperations_Astig_Wid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Z_operations_Astig = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Z_operations_Astig' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=614 ,SCR_YSIZE=651  $
      ,NOTIFY_REALIZE='Initialize_Z_operations_Astig'  $
      ,TITLE='Z-coordinate Operations (Astigmatism Only - no'+ $
      ' Interference)' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_SLIDER_Z_phase_offset_Astig =  $
      Widget_Slider(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_SLIDER_Z_phase_offset_Astig' ,XOFFSET=23  $
      ,YOFFSET=301 ,SCR_XSIZE=167 ,SCR_YSIZE=50 ,TITLE='Z offset'+ $
      ' (nm)' ,MINIMUM=-200 ,MAXIMUM=200 ,VALUE=0)

  
  WID_BUTTON_ExtractZ_Astig =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_ExtractZ_Astig' ,XOFFSET=427 ,YOFFSET=37  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Extract Z'+ $
      ' coordinate')

  
  WID_BUTTON_Add_Offset_Astig =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Add_Offset_Astig' ,XOFFSET=204 ,YOFFSET=307  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Add'+ $
      ' Offset')

  
  WID_BUTTON_Close = Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Close' ,XOFFSET=414 ,YOFFSET=550  $
      ,SCR_XSIZE=145 ,SCR_YSIZE=45 ,/ALIGN_CENTER ,VALUE='Close')

  
  WID_BUTTON_PickCalFile = Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_PickCalFile' ,XOFFSET=427 ,SCR_XSIZE=150  $
      ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Pick CAL (WND) File')

  
  WID_TEXT_WindFilename_Astig =  $
      Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_WindFilename_Astig' ,XOFFSET=5 ,YOFFSET=4  $
      ,SCR_XSIZE=415 ,SCR_YSIZE=49 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BUTTON_Write_Guide_Star_Astig =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Write_Guide_Star_Astig' ,XOFFSET=406  $
      ,YOFFSET=505 ,SCR_XSIZE=130 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Write Guide Star')

  
  WID_BUTTON_Test_Guide_Star_Astig =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Test_Guide_Star_Astig' ,XOFFSET=406  $
      ,YOFFSET=465 ,SCR_XSIZE=130 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Test Guide Star')

  
  WID_SLIDER_Z_Sm_Width = Widget_Slider(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_SLIDER_Z_Sm_Width' ,XOFFSET=31 ,YOFFSET=540  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=55 ,TITLE='Guide Star Smoothing'+ $
      ' Width' ,MINIMUM=1 ,MAXIMUM=1000 ,VALUE=250)

  
  WID_SLIDER_Z_Fit = Widget_Slider(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_SLIDER_Z_Fit' ,XOFFSET=32 ,YOFFSET=475  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=55 ,TITLE='Guide Star Polyn. Fit'+ $
      ' Order' ,MINIMUM=1 ,MAXIMUM=10 ,VALUE=5)

  
  WID_DROPLIST_Z_Fit_Method =  $
      Widget_Droplist(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_DROPLIST_Z_Fit_Method' ,XOFFSET=46 ,YOFFSET=445  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=30 ,TITLE='GuideStar Drift'+ $
      ' Correction' ,VALUE=[ 'Polynomial', 'Weighted Smoothing' ])

  
  WID_BUTTON_Remove_Tilt_Astig =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Remove_Tilt_Astig' ,XOFFSET=407 ,YOFFSET=307  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Remove XYZ'+ $
      ' Tilt')

  
  WID_BUTTON_Test_EllipticityOnly =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Test_EllipticityOnly' ,XOFFSET=20  $
      ,YOFFSET=221 ,SCR_XSIZE=180 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Test Ellipticity Calibration')

  
  WID_BUTTON_Save_EllipticityAndWind =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Save_EllipticityAndWind' ,XOFFSET=220  $
      ,YOFFSET=221 ,SCR_XSIZE=180 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Save Ellipticity Calibration')

  
  WID_TEXT_GuideStarAncFilename_Astig =  $
      Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_GuideStarAncFilename_Astig' ,XOFFSET=3  $
      ,YOFFSET=354 ,SCR_XSIZE=365 ,SCR_YSIZE=49 ,/EDITABLE ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_BUTTON_PickAncFile = Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_PickAncFile' ,XOFFSET=378 ,YOFFSET=355  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick ANC'+ $
      ' File')

  
  WID_BASE_WriteEllipticityGuideStar_1 =  $
      Widget_Base(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BASE_WriteEllipticityGuideStar_1' ,XOFFSET=378  $
      ,YOFFSET=390 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_UseMultipleANCs =  $
      Widget_Button(WID_BASE_WriteEllipticityGuideStar_1,  $
      UNAME='WID_BUTTON_UseMultipleANCs' ,/ALIGN_LEFT ,VALUE='Use'+ $
      ' Multiple GuideStars')

  
  WID_BUTTON_Write_GuideStarRadius =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Write_GuideStarRadius' ,XOFFSET=17  $
      ,YOFFSET=408 ,SCR_XSIZE=250 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Set Guidestar Area Radius (pix)')

  
  WID_TEXT_GuideStar_Radius_Astig =  $
      Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_GuideStar_Radius_Astig' ,XOFFSET=270  $
      ,YOFFSET=408 ,SCR_XSIZE=70 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_TEXT_ZCalStep_Astig = Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_ZCalStep_Astig' ,XOFFSET=480 ,YOFFSET=130  $
      ,SCR_XSIZE=74 ,SCR_YSIZE=35 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_LABEL_Cal_Step = Widget_Label(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_LABEL_Cal_Step' ,XOFFSET=400 ,YOFFSET=135  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='Z Step (nm)')

  
  WID_SLIDER_Zastig_Fit = Widget_Slider(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_SLIDER_Zastig_Fit' ,XOFFSET=33 ,YOFFSET=52  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=55 ,TITLE='Polyn. Fit Order for'+ $
      ' Astigmatism vs Z' ,MINIMUM=1 ,MAXIMUM=10 ,VALUE=5)

  
  WID_BUTTON_Convert_Fr_to_Z =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Convert_Fr_to_Z' ,XOFFSET=420 ,YOFFSET=221  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Convert Fr'+ $
      ' -> Z')

  
  WID_BASE_WriteEllipticity_MS_GuideStar_DPH =  $
      Widget_Base(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BASE_WriteEllipticity_MS_GuideStar_DPH' ,XOFFSET=379  $
      ,YOFFSET=427 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_UseMultipleANCs_DH =  $
      Widget_Button(WID_BASE_WriteEllipticity_MS_GuideStar_DPH,  $
      UNAME='WID_BUTTON_UseMultipleANCs_DH' ,/ALIGN_LEFT ,VALUE='Use'+ $
      ' Multiple GuideStars (DH)')

  
  WID_TEXT_Zmin_Astig = Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_Zmin_Astig' ,XOFFSET=80 ,YOFFSET=130  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=35 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_LABEL_Cal_Zmin = Widget_Label(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_LABEL_Cal_Zmin' ,XOFFSET=10 ,YOFFSET=135  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='Z min (nm)')

  
  WID_TEXT_Zmax_Astig = Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_Zmax_Astig' ,XOFFSET=270 ,YOFFSET=130  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=35 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_LABEL_Cal_Zmax = Widget_Label(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_LABEL_Cal_Zmax' ,XOFFSET=200 ,YOFFSET=135  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='Z max (nm)')

  
  WID_LABEL_Cal_num_iter = Widget_Label(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_LABEL_Cal_num_iter' ,XOFFSET=10 ,YOFFSET=180  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='# of'+ $
      ' iterations')

  
  WID_TEXT_ZCal_Astig_num_iter =  $
      Widget_Text(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_TEXT_ZCal_Astig_num_iter' ,XOFFSET=110 ,YOFFSET=175  $
      ,SCR_XSIZE=75 ,SCR_YSIZE=35 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_DROPLIST_LegendColor =  $
      Widget_Droplist(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_DROPLIST_LegendColor' ,XOFFSET=235 ,YOFFSET=178  $
      ,SCR_XSIZE=250 ,SCR_YSIZE=31 ,TITLE='Color Order' ,VALUE=[  $
      'Fiducial #', 'Fiducial X', 'Fiducial Y', 'Fiducial Frame#' ])

  
  WID_BUTTON_Plot_ZvsFfame_woffset =  $
      Widget_Button(WID_BASE_Z_operations_Astig,  $
      UNAME='WID_BUTTON_Plot_ZvsFfame_woffset' ,XOFFSET=220  $
      ,YOFFSET=266 ,SCR_XSIZE=180 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Plot Z vs Frame w offset')

  Widget_Control, /REALIZE, WID_BASE_Z_operations_Astig

  XManager, 'WID_BASE_Z_operations_Astig', WID_BASE_Z_operations_Astig, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Zoperations_Astig_Wid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Z_operations_Astig, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
