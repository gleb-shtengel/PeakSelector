;
; IDL Widget Interface Procedures. This Code is automatically
;     generated and should not be modified.

;
; Generated on:	03/04/2014 12:35.28
;
pro WID_BASE_Z_operations_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Z_operations'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Test_Wind_3D'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnTestWindPoint, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_Wind_3D'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnWriteCalibWind, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_ExtractZ'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnExtractZCoord, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Add_Offset_Slope'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAddOffsetSlope, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Close'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButtonClose, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickWINDFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickWINDFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_Guide_Star'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnWriteZDrift, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Test_Guide_Star'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnTestZDrift, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Safe_Wind_ASCII'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnWriteCalibASCII, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Remove_Tilt'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnRemoveTilt, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Read_Wind_Period'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ReadWindPoint, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_Wind_Period'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        WriteWindPoint, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_Zvalue'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Write_Zvalue, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TABLE_EllipticityFitCoeff'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Edit_Ellipticity_Coeff, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PlotEllipticity'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotEllipticityDataAndFit, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Test_EllipticityOnly'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnTestEllipOnly, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_EllipticityAndWind'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSaveEllipAndWind, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UnwrapZ'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        UnwrapZCoord, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_Wind_Period_without_scaling'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        WriteWindPointWithoutScaling, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_WriteEllipticityGuideStar'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Buttonpress_WriteEllipticityGuideStar_Z, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TABLE_EllipticityCorrectionSlope'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Edit_Ellipticity_Correction_Slope, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UnwrapZ_Lookup'): begin
      ;if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
      ;  LookupUnwrapZCoord, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickAncFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickGuideStarAncFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Write_GuideStarRadius'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        WriteGudeStarRadius, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_OptimizeSlopeCorrection'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OptimizeSlopeCorrection, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_WriteEllipticityGuideStar_E'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Buttonpress_WriteEllipticityGuideStar_E, Event
    end
    else:
  endcase

end

pro WID_BASE_Z_operations, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'ZoperationsWid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines

  WID_BASE_Z_operations = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Z_operations' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=601 ,SCR_YSIZE=791  $
      ,NOTIFY_REALIZE='Initialize_Z_operations' ,TITLE='Z-coordinate'+ $
      ' Operations' ,SPACE=3 ,XPAD=3 ,YPAD=3)


  WID_SLIDER_Z_phase_offset = Widget_Slider(WID_BASE_Z_operations,  $
      UNAME='WID_SLIDER_Z_phase_offset' ,XOFFSET=13 ,YOFFSET=108  $
      ,SCR_XSIZE=167 ,SCR_YSIZE=50 ,TITLE='Z offset (nm)'  $
      ,MINIMUM=-200 ,MAXIMUM=200 ,VALUE=0)


  WID_SLIDER_Z_phase_slope = Widget_Slider(WID_BASE_Z_operations,  $
      UNAME='WID_SLIDER_Z_phase_slope' ,XOFFSET=193 ,YOFFSET=108  $
      ,SCR_XSIZE=167 ,SCR_YSIZE=50 ,TITLE='Z slope (nm/1000 frames)'  $
      ,MINIMUM=-200 ,MAXIMUM=200 ,VALUE=0)


  WID_BUTTON_Test_Wind_3D = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Test_Wind_3D' ,XOFFSET=15 ,YOFFSET=8  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Test Wind'+ $
      ' Point 3D')


  WID_BUTTON_Write_Wind_3D = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Write_Wind_3D' ,XOFFSET=195 ,YOFFSET=8  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Write Wind'+ $
      ' Point 3D')


  WID_BUTTON_ExtractZ = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_ExtractZ' ,XOFFSET=395 ,YOFFSET=99  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Extract Z'+ $
      ' coordinate')


  WID_BUTTON_Add_Offset_Slope = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Add_Offset_Slope' ,XOFFSET=19 ,YOFFSET=172  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Add'+ $
      ' Offset/Slope')


  WID_BUTTON_Close = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Close' ,XOFFSET=505 ,YOFFSET=727  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Close')


  WID_BUTTON_PickWINDFile = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_PickWINDFile' ,XOFFSET=395 ,YOFFSET=28  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Pick WND'+ $
      ' File')


  WID_TEXT_WindFilename = Widget_Text(WID_BASE_Z_operations,  $
      UNAME='WID_TEXT_WindFilename' ,XOFFSET=5 ,YOFFSET=50  $
      ,SCR_XSIZE=380 ,SCR_YSIZE=49 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)


  WID_BUTTON_Write_Guide_Star = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Write_Guide_Star' ,XOFFSET=408 ,YOFFSET=360  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Write Guide'+ $
      ' Star')


  WID_BUTTON_Test_Guide_Star = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Test_Guide_Star' ,XOFFSET=408 ,YOFFSET=320  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Test Guide'+ $
      ' Star')


  WID_SLIDER_Z_Sm_Width = Widget_Slider(WID_BASE_Z_operations,  $
      UNAME='WID_SLIDER_Z_Sm_Width' ,XOFFSET=33 ,YOFFSET=395  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=48 ,TITLE='Guide Star Smoothing'+ $
      ' Width' ,MINIMUM=1 ,MAXIMUM=1000 ,VALUE=250)


  WID_SLIDER_Z_Fit = Widget_Slider(WID_BASE_Z_operations,  $
      UNAME='WID_SLIDER_Z_Fit' ,XOFFSET=34 ,YOFFSET=345  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=48 ,TITLE='Guide Star Polyn. Fit'+ $
      ' Order' ,MINIMUM=1 ,MAXIMUM=10 ,VALUE=5)


  WID_DROPLIST_Z_Fit_Method = Widget_Droplist(WID_BASE_Z_operations,  $
      UNAME='WID_DROPLIST_Z_Fit_Method' ,XOFFSET=48 ,YOFFSET=315  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=30 ,TITLE='GuideStar Drift'+ $
      ' Correction' ,VALUE=[ 'Polynomial', 'Weighted Smoothing' ])


  WID_BUTTON_Safe_Wind_ASCII = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Safe_Wind_ASCII' ,XOFFSET=395 ,YOFFSET=148  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Safe Wind'+ $
      ' Curves ASCII')


  WID_TEXT_WindFilename_ASCII = Widget_Text(WID_BASE_Z_operations,  $
      UNAME='WID_TEXT_WindFilename_ASCII' ,XOFFSET=376 ,YOFFSET=188  $
      ,SCR_XSIZE=175 ,SCR_YSIZE=50 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)


  WID_BUTTON_Remove_Tilt = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Remove_Tilt' ,XOFFSET=199 ,YOFFSET=172  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Remove XYZ'+ $
      ' Tilt')


  WID_BUTTON_Read_Wind_Period = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Read_Wind_Period' ,XOFFSET=18 ,YOFFSET=480  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Read Wind'+ $
      ' Period (nm)')


  WID_BUTTON_Write_Wind_Period = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Write_Wind_Period' ,XOFFSET=258 ,YOFFSET=480  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Write Wind'+ $
      ' Period (nm)')


  WID_TEXT_WindPeriod = Widget_Text(WID_BASE_Z_operations,  $
      UNAME='WID_TEXT_WindPeriod' ,XOFFSET=178 ,YOFFSET=480  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)


  WID_TEXT_Zvalue = Widget_Text(WID_BASE_Z_operations,  $
      UNAME='WID_TEXT_Zvalue' ,XOFFSET=13 ,YOFFSET=718 ,SCR_XSIZE=125  $
      ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,XSIZE=20 ,YSIZE=2)


  WID_BUTTON_Write_Zvalue = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Write_Zvalue' ,XOFFSET=284 ,YOFFSET=718  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Write'+ $
      ' Z-value and Z-uncertainty')


  WID_LABEL_Zvalue = Widget_Label(WID_BASE_Z_operations,  $
      UNAME='WID_LABEL_Zvalue' ,XOFFSET=19 ,YOFFSET=703  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Z value (nm)'+ $
      ' to write')


  WID_TEXT_Zuncertainty = Widget_Text(WID_BASE_Z_operations,  $
      UNAME='WID_TEXT_Zuncertainty' ,XOFFSET=151 ,YOFFSET=718  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)


  WID_LABEL_Zuncertainty = Widget_Label(WID_BASE_Z_operations,  $
      UNAME='WID_LABEL_Zuncertainty' ,XOFFSET=153 ,YOFFSET=703  $
      ,SCR_XSIZE=170 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Z uncertainty'+ $
      ' (nm) to write')


  WID_TABLE_EllipticityFitCoeff = Widget_Table(WID_BASE_Z_operations,  $
      UNAME='WID_TABLE_EllipticityFitCoeff' ,XOFFSET=16 ,YOFFSET=520  $
      ,SCR_XSIZE=393 ,SCR_YSIZE=57 ,/EDITABLE ,ROW_LABELS=[  $
      'Ellipticity Fit Coeff.' ] ,XSIZE=3 ,YSIZE=1)


  WID_BUTTON_PlotEllipticity = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_PlotEllipticity' ,XOFFSET=18 ,YOFFSET=585  $
      ,SCR_XSIZE=230 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Plot'+ $
      ' Ellipticity vs. Z: Data + Fit')


  WID_BUTTON_Test_EllipticityOnly =  $
      Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Test_EllipticityOnly' ,XOFFSET=258  $
      ,YOFFSET=585 ,SCR_XSIZE=160 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Test Ellipticity vs Frame')


  WID_BUTTON_Save_EllipticityAndWind =  $
      Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Save_EllipticityAndWind' ,XOFFSET=428  $
      ,YOFFSET=585 ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Save Ellipticity + Wind')


  WID_BUTTON_UnwrapZ = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_UnwrapZ' ,XOFFSET=458 ,YOFFSET=535  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Unwrap Z')


  WID_BUTTON_Write_Wind_Period_without_scaling =  $
      Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Write_Wind_Period_without_scaling'  $
      ,XOFFSET=108 ,YOFFSET=445 ,SCR_XSIZE=230 ,SCR_YSIZE=30  $
      ,/ALIGN_CENTER ,VALUE='Write Wind Period w/o Scaling')


  WID_BASE_WriteEllipticityGuideStar =  $
      Widget_Base(WID_BASE_Z_operations,  $
      UNAME='WID_BASE_WriteEllipticityGuideStar' ,XOFFSET=368  $
      ,YOFFSET=390 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)


  WID_BUTTON_WriteEllipticityGuideStar =  $
      Widget_Button(WID_BASE_WriteEllipticityGuideStar,  $
      UNAME='WID_BUTTON_WriteEllipticityGuideStar' ,/ALIGN_LEFT  $
      ,VALUE='Correct Ellipticity for GuideStar - Z')


  WID_TABLE_EllipticityCorrectionSlope =  $
      Widget_Table(WID_BASE_Z_operations,  $
      UNAME='WID_TABLE_EllipticityCorrectionSlope' ,XOFFSET=10  $
      ,YOFFSET=627 ,SCR_XSIZE=380 ,SCR_YSIZE=57 ,/EDITABLE  $
      ,COLUMN_LABELS=[ 'X cntr. (pix)', 'X slp. (nm/pix)', 'Y cntr.'+ $
      ' (pix)', 'Y slp. (nm/pix)' ] ,ROW_LABELS=[ '', '' ] ,XSIZE=4  $
      ,YSIZE=1)


  WID_BASE_WriteEllipticityGuideStar_0 =  $
      Widget_Base(WID_BASE_Z_operations,  $
      UNAME='WID_BASE_WriteEllipticityGuideStar_0' ,XOFFSET=396  $
      ,YOFFSET=649 ,SCR_XSIZE=180 ,SCR_YSIZE=22 ,TITLE='IDL'  $
      ,COLUMN=1 ,/NONEXCLUSIVE)


  WID_BUTTON_AddEllipticitySlopeCorrection =  $
      Widget_Button(WID_BASE_WriteEllipticityGuideStar_0,  $
      UNAME='WID_BUTTON_AddEllipticitySlopeCorrection' ,SCR_XSIZE=178  $
      ,SCR_YSIZE=22 ,/ALIGN_LEFT ,VALUE='Unwrap with Ell. Slope'+ $
      ' Corr.')


  WID_LABEL_ElliptSlopeCorr = Widget_Label(WID_BASE_Z_operations,  $
      UNAME='WID_LABEL_ElliptSlopeCorr' ,XOFFSET=398 ,YOFFSET=632  $
      ,SCR_XSIZE=170 ,SCR_YSIZE=18 ,/ALIGN_LEFT ,VALUE='Ellipticity'+ $
      ' Slope Correction')


  WID_BUTTON_UnwrapZ_Lookup = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_UnwrapZ_Lookup' ,XOFFSET=431 ,YOFFSET=480  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Lookup '+ $
      ' Unwrap Z')


  WID_DROPLIST_LookupUnwrapDisplayType =  $
      Widget_Droplist(WID_BASE_Z_operations,  $
      UNAME='WID_DROPLIST_LookupUnwrapDisplayType' ,XOFFSET=353  $
      ,YOFFSET=449 ,SCR_XSIZE=225 ,SCR_YSIZE=24 ,TITLE='Lookup'+ $
      ' Display' ,VALUE=[ 'Local - w/o Display', 'Local - with'+ $
      ' Display', 'Cluster - No Display' ])


  WID_TEXT_GuideStarAncFilename = Widget_Text(WID_BASE_Z_operations,  $
      UNAME='WID_TEXT_GuideStarAncFilename' ,XOFFSET=15 ,YOFFSET=224  $
      ,SCR_XSIZE=344 ,SCR_YSIZE=49 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)


  WID_BUTTON_PickAncFile = Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_PickAncFile' ,XOFFSET=380 ,YOFFSET=250  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Pick ANC'+ $
      ' File')


  WID_BASE_WriteEllipticityGuideStar_1 =  $
      Widget_Base(WID_BASE_Z_operations,  $
      UNAME='WID_BASE_WriteEllipticityGuideStar_1' ,XOFFSET=379  $
      ,YOFFSET=284 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)


  WID_BUTTON_UseMultipleANCs =  $
      Widget_Button(WID_BASE_WriteEllipticityGuideStar_1,  $
      UNAME='WID_BUTTON_UseMultipleANCs' ,/ALIGN_LEFT ,VALUE='Use'+ $
      ' Multiple GuideStars')


  WID_BUTTON_Write_GuideStarRadius =  $
      Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_Write_GuideStarRadius' ,XOFFSET=19  $
      ,YOFFSET=278 ,SCR_XSIZE=250 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Set Guidestar Area Radius (pix)')


  WID_TEXT_GuideStar_Radius = Widget_Text(WID_BASE_Z_operations,  $
      UNAME='WID_TEXT_GuideStar_Radius' ,XOFFSET=272 ,YOFFSET=278  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)


  WID_TEXT_WL = Widget_Text(WID_BASE_Z_operations,  $
      UNAME='WID_TEXT_WL' ,XOFFSET=501 ,YOFFSET=62 ,SCR_XSIZE=70  $
      ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '590.0' ] ,XSIZE=20  $
      ,YSIZE=2)


  WID_LABEL_WL = Widget_Label(WID_BASE_Z_operations,  $
      UNAME='WID_LABEL_WL' ,XOFFSET=400 ,YOFFSET=71 ,SCR_XSIZE=100  $
      ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='Wavelength (nm)')


  WID_BUTTON_OptimizeSlopeCorrection =  $
      Widget_Button(WID_BASE_Z_operations,  $
      UNAME='WID_BUTTON_OptimizeSlopeCorrection' ,XOFFSET=398  $
      ,YOFFSET=678 ,SCR_XSIZE=183 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Optimize Slope Corr.')


  WID_BASE_WriteEllipticityGuideStar_E =  $
      Widget_Base(WID_BASE_Z_operations,  $
      UNAME='WID_BASE_WriteEllipticityGuideStar_E' ,XOFFSET=367  $
      ,YOFFSET=417 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)


  WID_BUTTON_WriteEllipticityGuideStar_E =  $
      Widget_Button(WID_BASE_WriteEllipticityGuideStar_E,  $
      UNAME='WID_BUTTON_WriteEllipticityGuideStar_E' ,/ALIGN_LEFT  $
      ,VALUE='Correct Ellipticity for GuideStar - E')


  WID_DROPLIST_Optimization_Mode =  $
      Widget_Droplist(WID_BASE_Z_operations,  $
      UNAME='WID_DROPLIST_Optimization_Mode' ,XOFFSET=235  $
      ,YOFFSET=675 ,SCR_XSIZE=150 ,SCR_YSIZE=24 ,TITLE='Mode'  $
      ,VALUE=[ 'Local Groups', 'Local Peaks', 'Bridge Groups',  $
      'Bridge Peaks' ])

  Widget_Control, /REALIZE, WID_BASE_Z_operations

  XManager, 'WID_BASE_Z_operations', WID_BASE_Z_operations, /NO_BLOCK

end
;
; Empty stub procedure used for autoloading.
;
pro ZoperationsWid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Z_operations, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
