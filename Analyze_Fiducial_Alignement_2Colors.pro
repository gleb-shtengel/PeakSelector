; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	01/22/2013 11:29.17
; 
pro WID_BASE_AnalyzeMultipleFiducials_2Colors_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_AnalyzeMultipleFiducials_2Colors'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Analyze_Fiducial_colocalization_2color_OK'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_AnalyzeMultiple_Fiducial_Colocalization_Start, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickFidFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickFidFile, Event
    end
    else:
  endcase

end

pro WID_BASE_AnalyzeMultipleFiducials_2Colors, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Analyze_Fiducial_Alignement_2Colors_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_AnalyzeMultipleFiducials_2Colors = Widget_Base(  $
      GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_AnalyzeMultipleFiducials_2Colors' ,XOFFSET=5  $
      ,YOFFSET=5 ,SCR_XSIZE=383 ,SCR_YSIZE=373  $
      ,NOTIFY_REALIZE='Initialize_AnalizeMultiple_Fiducials_2Colors'  $
      ,TITLE='Analyze Fiducial Co-Localization ofr  2 Colors'  $
      ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_BUTTON_Analyze_Fiducial_colocalization_2color_OK =  $
      Widget_Button(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_BUTTON_Analyze_Fiducial_colocalization_2color_OK'  $
      ,XOFFSET=250 ,YOFFSET=305 ,SCR_XSIZE=120 ,SCR_YSIZE=30  $
      ,/ALIGN_CENTER ,VALUE='Confirm and Start')

  
  WID_LABEL_Xrange =  $
      Widget_Label(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_LABEL_Xrange' ,XOFFSET=40 ,YOFFSET=109 ,SCR_XSIZE=85  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='X Range (pixels)')

  
  WID_TEXT_Xrange1 =  $
      Widget_Text(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_TEXT_Xrange1' ,XOFFSET=149 ,YOFFSET=104  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '1.5' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_TEXT_Yrange1 =  $
      Widget_Text(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_TEXT_Yrange1' ,XOFFSET=149 ,YOFFSET=139  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '1.5' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_LABEL_YRange =  $
      Widget_Label(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_LABEL_YRange' ,XOFFSET=40 ,YOFFSET=144 ,SCR_XSIZE=85  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Y Range (pixels)')

  
  WID_LABEL_Status =  $
      Widget_Label(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_LABEL_Status' ,XOFFSET=20 ,YOFFSET=275 ,SCR_XSIZE=35  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Status')

  
  WID_TEXT_Status_Fid2Color_Analysis =  $
      Widget_Text(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_TEXT_Status_Fid2Color_Analysis' ,XOFFSET=75  $
      ,YOFFSET=264 ,SCR_XSIZE=280 ,SCR_YSIZE=35 ,/EDITABLE ,/WRAP  $
      ,VALUE=[ 'Press ' ] ,XSIZE=20 ,YSIZE=2)

  
  WID_TEXT_FidFilename =  $
      Widget_Text(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_TEXT_FidFilename' ,XOFFSET=26 ,YOFFSET=49  $
      ,SCR_XSIZE=331 ,SCR_YSIZE=48 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BUTTON_PickFidFile =  $
      Widget_Button(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_BUTTON_PickFidFile' ,XOFFSET=205 ,YOFFSET=12  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick *.anc'+ $
      ' File')

  
  WID_LABEL_anc_explanation =  $
      Widget_Label(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_LABEL_anc_explanation' ,XOFFSET=10 ,YOFFSET=14  $
      ,SCR_XSIZE=177 ,SCR_YSIZE=23 ,/ALIGN_LEFT ,VALUE='Use *.anc'+ $
      ' file to list fiducial peaks')

  
  WID_LABEL_ZRange =  $
      Widget_Label(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_LABEL_ZRange' ,XOFFSET=40 ,YOFFSET=178 ,SCR_XSIZE=85  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Z Range (nm)')

  
  WID_TEXT_ZRange1 =  $
      Widget_Text(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_TEXT_ZRange1' ,XOFFSET=149 ,YOFFSET=174  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '200' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_BASE_0 = Widget_Base(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_BASE_0' ,XOFFSET=2 ,YOFFSET=312 ,TITLE='IDL'  $
      ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_use_green = Widget_Button(WID_BASE_0,  $
      UNAME='WID_BUTTON_use_green' ,/ALIGN_LEFT ,VALUE='Use Green'+ $
      ' Fiducials (Red if not selected)')

  
  WID_LABEL_MinNumber =  $
      Widget_Label(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_LABEL_MinNumber' ,XOFFSET=38 ,YOFFSET=226  $
      ,SCR_XSIZE=85 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Min # of'+ $
      ' peaks')

  
  WID_TEXT_MiNumberofPeaks =  $
      Widget_Text(WID_BASE_AnalyzeMultipleFiducials_2Colors,  $
      UNAME='WID_TEXT_MiNumberofPeaks' ,XOFFSET=149 ,YOFFSET=215  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '50' ]  $
      ,XSIZE=20 ,YSIZE=2)

  Widget_Control, /REALIZE, WID_BASE_AnalyzeMultipleFiducials_2Colors

  XManager, 'WID_BASE_AnalyzeMultipleFiducials_2Colors', WID_BASE_AnalyzeMultipleFiducials_2Colors, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Analyze_Fiducial_Alignement_2Colors, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_AnalyzeMultipleFiducials_2Colors, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
