; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	03/10/2014 11:16.43
; 
pro WID_BASE_TransformedFilenames_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_TransformedFilenames'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_StartTransform'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        StartSaveTransformed, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CancelReExtract'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancelSave, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_TransformAndReExtract'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        StartTransformAndReExtract, Event
    end
    else:
  endcase

end

pro WID_BASE_TransformedFilenames, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

; Floating or modal bases must have a Group Leader.

  if(N_Elements(wGroup) eq 0)then $
     Message,'Group leader must be specified for Modal or Floating'+ $
      ' top level bases'

  Resolve_Routine, 'TransformRawSaveSaveSumWid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_TransformedFilenames = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_TransformedFilenames' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=618 ,SCR_YSIZE=440  $
      ,NOTIFY_REALIZE='Initialize_TransformedFilenames'  $
      ,TITLE='Select names for Transformed Data Files' ,SPACE=3  $
      ,XPAD=3 ,YPAD=3 ,/MODAL)

  
  WID_BUTTON_StartTransform =  $
      Widget_Button(WID_BASE_TransformedFilenames,  $
      UNAME='WID_BUTTON_StartTransform' ,XOFFSET=6 ,YOFFSET=215  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Confirm and'+ $
      ' Start')

  
  WID_TEXT_Label1Filename =  $
      Widget_Text(WID_BASE_TransformedFilenames,  $
      UNAME='WID_TEXT_Label1Filename' ,XOFFSET=4 ,YOFFSET=20  $
      ,SCR_XSIZE=600 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_TEXT_Label2Filename =  $
      Widget_Text(WID_BASE_TransformedFilenames,  $
      UNAME='WID_TEXT_Label2Filename' ,XOFFSET=4 ,YOFFSET=70  $
      ,SCR_XSIZE=600 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_TEXT_Label3Filename =  $
      Widget_Text(WID_BASE_TransformedFilenames,  $
      UNAME='WID_TEXT_Label3Filename' ,XOFFSET=4 ,YOFFSET=120  $
      ,SCR_XSIZE=600 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_TEXT_SumFilename = Widget_Text(WID_BASE_TransformedFilenames,  $
      UNAME='WID_TEXT_SumFilename' ,XOFFSET=3 ,YOFFSET=170  $
      ,SCR_XSIZE=600 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_LABEL_SumFile = Widget_Label(WID_BASE_TransformedFilenames,  $
      UNAME='WID_LABEL_SumFile' ,XOFFSET=5 ,YOFFSET=155 ,/ALIGN_LEFT  $
      ,VALUE='Sum File:')

  
  WID_LABEL_SumFile_0 = Widget_Label(WID_BASE_TransformedFilenames,  $
      UNAME='WID_LABEL_SumFile_0' ,XOFFSET=5 ,YOFFSET=5 ,/ALIGN_LEFT  $
      ,VALUE='Label 1 File:')

  
  WID_LABEL_SumFile_1 = Widget_Label(WID_BASE_TransformedFilenames,  $
      UNAME='WID_LABEL_SumFile_1' ,XOFFSET=5 ,YOFFSET=55 ,/ALIGN_LEFT  $
      ,VALUE='Label 2 File:')

  
  WID_LABEL_SumFile_2 = Widget_Label(WID_BASE_TransformedFilenames,  $
      UNAME='WID_LABEL_SumFile_2' ,XOFFSET=5 ,YOFFSET=105  $
      ,/ALIGN_LEFT ,VALUE='Label 3 File:')

  
  WID_BUTTON_CancelReExtract =  $
      Widget_Button(WID_BASE_TransformedFilenames,  $
      UNAME='WID_BUTTON_CancelReExtract' ,XOFFSET=449 ,YOFFSET=216  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Cancel')

  
  WID_BUTTON_TransformAndReExtract =  $
      Widget_Button(WID_BASE_TransformedFilenames,  $
      UNAME='WID_BUTTON_TransformAndReExtract' ,XOFFSET=183  $
      ,YOFFSET=215 ,SCR_XSIZE=195 ,SCR_YSIZE=55 ,/ALIGN_CENTER  $
      ,VALUE='Confirm, Transform, ReExtract')

  
  WID_DROPLIST_SetSigmaFitSym_TRS =  $
      Widget_Droplist(WID_BASE_TransformedFilenames,  $
      UNAME='WID_DROPLIST_SetSigmaFitSym_TRS' ,XOFFSET=340  $
      ,YOFFSET=290 ,SCR_XSIZE=225 ,SCR_YSIZE=30  $
      ,TITLE='SetSigmaFitSymmetry' ,VALUE=[ 'R', 'X Y' ])

  
  WID_DROPLIST_TransformEngine =  $
      Widget_Droplist(WID_BASE_TransformedFilenames,  $
      UNAME='WID_DROPLIST_TransformEngine' ,XOFFSET=20 ,YOFFSET=290  $
      ,SCR_XSIZE=225 ,SCR_YSIZE=30 ,TITLE='Transformation Engine'  $
      ,VALUE=[ 'Local', 'Cluster', 'IDL Bridge' ])

  
  WID_SLIDER_FramesPerNode_tr =  $
      Widget_Slider(WID_BASE_TransformedFilenames,  $
      UNAME='WID_SLIDER_FramesPerNode_tr' ,XOFFSET=380 ,YOFFSET=345  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=50 ,TITLE='Frames per Node (Cluster)'  $
      ,MINIMUM=0 ,MAXIMUM=10000 ,VALUE=2500)

  
  WID_SLIDER_Group_Gap_tr =  $
      Widget_Slider(WID_BASE_TransformedFilenames,  $
      UNAME='WID_SLIDER_Group_Gap_tr' ,XOFFSET=20 ,YOFFSET=345  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=50 ,TITLE='Group Gap' ,MAXIMUM=32  $
      ,VALUE=3)

  
  WID_SLIDER_Grouping_Radius_tr =  $
      Widget_Slider(WID_BASE_TransformedFilenames,  $
      UNAME='WID_SLIDER_Grouping_Radius_tr' ,XOFFSET=200 ,YOFFSET=345  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=50 ,TITLE='Grouping Radius*100'  $
      ,MAXIMUM=200 ,VALUE=25)

  Widget_Control, /REALIZE, WID_BASE_TransformedFilenames

  XManager, 'WID_BASE_TransformedFilenames', WID_BASE_TransformedFilenames, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro TransformRawSaveSaveSumWid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_TransformedFilenames, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
