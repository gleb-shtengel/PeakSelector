; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	11/09/2020 10:41.35
; 
pro WID_BASE_iPALM_MACRO_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_iPALM_MACRO'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_StartTrsnformation_Macro'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Start_Transformation_Macro, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CancelReExtract'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCanceliPALM_Macro, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_SetSigmaFitSym_iPALM'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        Set_SigmaFitSym_iPALM, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_PickFile1'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickCam1TxtFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_PickFile2'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickCam2TxtFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_PickFile3'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickCam3TxtFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_PickANCFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickANCFile_iPALM, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_PickWINDFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickWINDFile_iPALM, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_TransformEngine_iPALM'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        Set_TransformEngine_iPALM, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_StartMLExtract_Fast'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Start_iPALM_Macro_Fast, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TABLE_InfoFile_iPALM_macro'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        DoInsertInfo_iPALM_Macro, Event
    end
    else:
  endcase

end

pro WID_BASE_iPALM_MACRO, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

; Floating or modal bases must have a Group Leader.

  if(N_Elements(wGroup) eq 0)then $
     Message,'Group leader must be specified for Modal or Floating'+ $
      ' top level bases'

  Resolve_Routine, 'Transform_Extract_ReExtract_Filter_GetZ_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_iPALM_MACRO = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_iPALM_MACRO' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=749 ,SCR_YSIZE=870  $
      ,NOTIFY_REALIZE='Initialize_Transform_Extract_ReExtract_Filter_G'+ $
      'etZ_iPALM' ,TITLE='iPALM Macro: Transform, Extract, Reextract,'+ $
      ' Fiter, Group, Extract Z' ,SPACE=3 ,XPAD=3 ,YPAD=3 ,/MODAL)

  
  WID_BUTTON_StartTrsnformation_Macro =  $
      Widget_Button(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_BUTTON_StartTrsnformation_Macro' ,XOFFSET=530  $
      ,YOFFSET=720 ,SCR_XSIZE=180 ,SCR_YSIZE=40 ,/ALIGN_CENTER  $
      ,VALUE='Perform Transformation Only')

  
  WID_BUTTON_CancelReExtract = Widget_Button(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_BUTTON_CancelReExtract' ,XOFFSET=570 ,YOFFSET=780  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=40 ,/ALIGN_CENTER ,VALUE='Cancel')

  
  WID_DROPLIST_SetSigmaFitSym_iPALM =  $
      Widget_Droplist(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_DROPLIST_SetSigmaFitSym_iPALM' ,XOFFSET=280  $
      ,YOFFSET=190 ,SCR_XSIZE=175 ,SCR_YSIZE=30  $
      ,TITLE='SetSigmaFitSymmetry' ,VALUE=[ 'R', 'X Y' ])

  
  WID_BTTN_PickFile1 = Widget_Button(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_BTTN_PickFile1' ,XOFFSET=509 ,YOFFSET=10  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick Cam1'+ $
      ' File')

  
  WID_TXT_Cam1Filename = Widget_Text(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_TXT_Cam1Filename' ,XOFFSET=4 ,YOFFSET=10  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_TXT_Cam2Filename = Widget_Text(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_TXT_Cam2Filename' ,XOFFSET=4 ,YOFFSET=50  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BTTN_PickFile2 = Widget_Button(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_BTTN_PickFile2' ,XOFFSET=509 ,YOFFSET=50  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick Cam2'+ $
      ' File')

  
  WID_TXT_Cam3Filename = Widget_Text(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_TXT_Cam3Filename' ,XOFFSET=3 ,YOFFSET=90  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BTTN_PickFile3 = Widget_Button(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_BTTN_PickFile3' ,XOFFSET=508 ,YOFFSET=90  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick Cam3'+ $
      ' File')

  
  WID_TXT_ANCFilename = Widget_Text(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_TXT_ANCFilename' ,XOFFSET=4 ,YOFFSET=145  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BTTN_PickANCFile = Widget_Button(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_BTTN_PickANCFile' ,XOFFSET=509 ,YOFFSET=144  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick *.anc'+ $
      ' File')

  
  WID_BTTN_PickWINDFile = Widget_Button(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_BTTN_PickWINDFile' ,XOFFSET=565 ,YOFFSET=530  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Pick WND'+ $
      ' File')

  
  WID_TXT_WindFilename = Widget_Text(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_TXT_WindFilename' ,XOFFSET=294 ,YOFFSET=566  $
      ,SCR_XSIZE=430 ,SCR_YSIZE=50 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_SLIDER_Grouping_Radius_iPALM =  $
      Widget_Slider(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_SLIDER_Grouping_Radius_iPALM' ,XOFFSET=435  $
      ,YOFFSET=640 ,SCR_XSIZE=140 ,SCR_YSIZE=50 ,TITLE='Grouping'+ $
      ' Radius*100' ,MAXIMUM=200 ,VALUE=25)

  
  WID_SLIDER_Group_Gap_iPALM = Widget_Slider(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_SLIDER_Group_Gap_iPALM' ,XOFFSET=280 ,YOFFSET=640  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=50 ,TITLE='Group Gap' ,MAXIMUM=32  $
      ,VALUE=3)

  
  WID_SLIDER_FramesPerNode_iPALM =  $
      Widget_Slider(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_SLIDER_FramesPerNode_iPALM' ,XOFFSET=590  $
      ,YOFFSET=640 ,SCR_XSIZE=140 ,SCR_YSIZE=50 ,TITLE='Frames per'+ $
      ' Node (Cluster)' ,MINIMUM=0 ,MAXIMUM=10000 ,VALUE=2500)

  
  WID_DROPLIST_TransformEngine_iPALM =  $
      Widget_Droplist(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_DROPLIST_TransformEngine_iPALM' ,XOFFSET=10  $
      ,YOFFSET=190 ,SCR_XSIZE=225 ,SCR_YSIZE=30  $
      ,TITLE='Transformation Engine' ,VALUE=[ 'Local', 'Cluster',  $
      'IDL Bridge' ])

  
  WID_Filter_Parameters_iPALM_Macro =  $
      Widget_Table(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_Filter_Parameters_iPALM_Macro' ,XOFFSET=380  $
      ,YOFFSET=270 ,SCR_XSIZE=320 ,SCR_YSIZE=234 ,/EDITABLE  $
      ,COLUMN_LABELS=[ 'Value' ] ,ROW_LABELS=[ 'Nph. Min.', 'Full'+ $
      ' Sigma X Max. (pix.)', 'Full Sigma Y Max. (pix.)', 'Sigma Z'+ $
      ' Max. (nm)', 'Coherence Min.', 'Coherence Max.', 'Max. A for'+ $
      ' (Wx-A)*(Wx-A)<B', 'B for (Wx-A)*(Wx-A)<B', 'Min. A  for'+ $
      ' (Wx-A)*(Wx-A)>B', 'B for (Wx-A)*(Wx-A)>B' ] ,XSIZE=1  $
      ,YSIZE=10)

  
  WID_BASE_UseInfoFileToFlip_iPALM =  $
      Widget_Base(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_BASE_UseInfoFileToFlip_iPALM' ,XOFFSET=515  $
      ,YOFFSET=196 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_Use_InfoFile_Flip =  $
      Widget_Button(WID_BASE_UseInfoFileToFlip_iPALM,  $
      UNAME='WID_Use_InfoFile_Flip' ,/ALIGN_LEFT ,VALUE='Use Info'+ $
      ' File to Flip')

  
  WID_BASE_UseInfoFileToFlip_iPALM_0 =  $
      Widget_Base(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_BASE_UseInfoFileToFlip_iPALM_0' ,XOFFSET=515  $
      ,YOFFSET=235 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_Use_SkipTransformation =  $
      Widget_Button(WID_BASE_UseInfoFileToFlip_iPALM_0,  $
      UNAME='WID_Use_SkipTransformation' ,/ALIGN_LEFT ,VALUE='Skip'+ $
      ' Data Transformation')

  
  WID_BUTTON_StartMLExtract_Fast =  $
      Widget_Button(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_BUTTON_StartMLExtract_Fast' ,XOFFSET=330  $
      ,YOFFSET=720 ,SCR_XSIZE=150 ,SCR_YSIZE=40 ,/ALIGN_CENTER  $
      ,VALUE='Confirm and Start Fast')

  
  WID_LABEL_ConfirmStartFast = Widget_Label(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_LABEL_ConfirmStartFast' ,XOFFSET=312 ,YOFFSET=765  $
      ,SCR_XSIZE=249 ,SCR_YSIZE=18 ,/ALIGN_LEFT ,VALUE='Large'+ $
      ' Transformed Files are not creared')

  
  WID_LABEL_ConfirmStartFast_1 = Widget_Label(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_LABEL_ConfirmStartFast_1' ,XOFFSET=311 ,YOFFSET=782  $
      ,SCR_XSIZE=249 ,SCR_YSIZE=18 ,/ALIGN_LEFT ,VALUE='Works only in'+ $
      ' Cluseter or IDL Bridge modes')

  
  WID_TABLE_InfoFile_iPALM_macro = Widget_Table(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_TABLE_InfoFile_iPALM_macro' ,FRAME=1 ,XOFFSET=10  $
      ,YOFFSET=265 ,SCR_XSIZE=260 ,SCR_YSIZE=540 ,/EDITABLE  $
      ,COLUMN_LABELS=[ 'Values' ] ,ROW_LABELS=[ 'Zero Dark Cnt', 'X'+ $
      ' pixels Data', 'Y pixels Data', 'Max # Frames', 'Initial'+ $
      ' Frame', 'Final Frame', 'Peak Threshold Criteria', 'File type'+ $
      ' (0 - .dat, 1 - .tif)', 'Min Peak Ampl.', 'Max Peak Ampl.',  $
      'Min Peak Width', 'Max Peak Width', 'Limit ChiSq', 'Counts Per'+ $
      ' Electron', 'Max # Peaks Iter1', 'Max # Peaks Iter2', 'Flip'+ $
      ' Horizontal', 'Flip Vertical', '(Half) Gauss Size (d)', 'Appr.'+ $
      ' Gauss Sigma', 'Max Block Size', 'Sparse OverSampling',  $
      'Sparse Lambda', 'Sparse Delta', 'Sp. Max Error', 'Sp. Max # of'+ $
      ' Iter.' ] ,XSIZE=1 ,YSIZE=26)

  
  WID_DROPLIST_ZExctractEngine_iPALM =  $
      Widget_Droplist(WID_BASE_iPALM_MACRO,  $
      UNAME='WID_DROPLIST_ZExctractEngine_iPALM' ,XOFFSET=31  $
      ,YOFFSET=222 ,SCR_XSIZE=225 ,SCR_YSIZE=30 ,TITLE='Z-Extraction'+ $
      ' Engine' ,VALUE=[ 'Local', 'IDL Bridge' ])

  Widget_Control, /REALIZE, WID_BASE_iPALM_MACRO

  XManager, 'WID_BASE_iPALM_MACRO', WID_BASE_iPALM_MACRO, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Transform_Extract_ReExtract_Filter_GetZ, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_iPALM_MACRO, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
