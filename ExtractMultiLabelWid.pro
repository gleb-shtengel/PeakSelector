; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	03/10/2014 09:48.55
; 
pro WID_BASE_ExtractMultiLabel_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_ExtractMultiLabel'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_StartMLExtract'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        StartMLExtract, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CancelReExtract'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancelReExtract, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_SetSigmaFitSym_ML'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        SetSigmaSym_ML, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickFile1'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickCam1DatFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickFile2'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickCam2DatFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickFile3'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickCam3DatFile, Event
    end
    else:
  endcase

end

pro WID_BASE_ExtractMultiLabel, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

; Floating or modal bases must have a Group Leader.

  if(N_Elements(wGroup) eq 0)then $
     Message,'Group leader must be specified for Modal or Floating'+ $
      ' top level bases'

  Resolve_Routine, 'ExtractMultiLabelWid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_ExtractMultiLabel = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_ExtractMultiLabel' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=674 ,SCR_YSIZE=251  $
      ,NOTIFY_REALIZE='Initialize_ExtractMultiLabel' ,TITLE='Extract'+ $
      ' Peaks Multilablel' ,SPACE=3 ,XPAD=3 ,YPAD=3 ,/MODAL)

  
  WID_BUTTON_StartMLExtract =  $
      Widget_Button(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_BUTTON_StartMLExtract' ,XOFFSET=350 ,YOFFSET=155  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Confirm and'+ $
      ' Start')

  
  WID_DROPLIST_FitDisplayType =  $
      Widget_Droplist(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_DROPLIST_FitDisplayType' ,XOFFSET=5 ,YOFFSET=156  $
      ,SCR_XSIZE=225 ,SCR_YSIZE=30 ,TITLE='Fit-Display Level'  $
      ,VALUE=[ 'No Display', 'Some Frames/Peaks', 'All Frames/Peaks'+ $
      ' ', 'Cluster - No Display', 'IDL Bridge - No Disp' ])

  
  WID_BUTTON_CancelReExtract =  $
      Widget_Button(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_BUTTON_CancelReExtract' ,XOFFSET=510 ,YOFFSET=155  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Cancel')

  
  WID_DROPLIST_SetSigmaFitSym_ML =  $
      Widget_Droplist(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_DROPLIST_SetSigmaFitSym_ML' ,XOFFSET=5 ,YOFFSET=190  $
      ,SCR_XSIZE=175 ,SCR_YSIZE=30 ,TITLE='SetSigmaFitSymmetry'  $
      ,VALUE=[ 'R', 'X Y' ])

  
  WID_BUTTON_PickFile1 = Widget_Button(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_BUTTON_PickFile1' ,XOFFSET=509 ,YOFFSET=10  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick Cam1'+ $
      ' File')

  
  WID_TEXT_Cam1Filename = Widget_Text(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_TEXT_Cam1Filename' ,XOFFSET=4 ,YOFFSET=9  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_TEXT_Cam2Filename = Widget_Text(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_TEXT_Cam2Filename' ,XOFFSET=4 ,YOFFSET=59  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BUTTON_PickFile2 = Widget_Button(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_BUTTON_PickFile2' ,XOFFSET=509 ,YOFFSET=60  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick Cam2'+ $
      ' File')

  
  WID_TEXT_Cam3Filename = Widget_Text(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_TEXT_Cam3Filename' ,XOFFSET=4 ,YOFFSET=109  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BUTTON_PickFile3 = Widget_Button(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_BUTTON_PickFile3' ,XOFFSET=509 ,YOFFSET=110  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick Cam3'+ $
      ' File')

  
  WID_BASE_0 = Widget_Base(WID_BASE_ExtractMultiLabel,  $
      UNAME='WID_BASE_0' ,XOFFSET=215 ,YOFFSET=190 ,TITLE='IDL'  $
      ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_Use_InfoFile_Flip = Widget_Button(WID_BASE_0,  $
      UNAME='WID_Use_InfoFile_Flip' ,/ALIGN_LEFT ,VALUE='Use Info'+ $
      ' File to Flip')

  Widget_Control, /REALIZE, WID_BASE_ExtractMultiLabel

  XManager, 'WID_BASE_ExtractMultiLabel', WID_BASE_ExtractMultiLabel, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro ExtractMultiLabelWid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_ExtractMultiLabel, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
