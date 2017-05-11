; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	02/19/2014 15:17.41
; 
pro WID_BASE_Info_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Info'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TABLE_InfoFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        DoInsertInfo, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Info_OK'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnInfoOK, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_FitDisplayType'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_SetSigmaFitSym'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        SetSigmaFitSym, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CancelFit'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancelFit, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Localization_Method'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        SetLocalizationMethod, Event
    end
    else:
  endcase

end

pro WID_BASE_Info, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

; Floating or modal bases must have a Group Leader.

  if(N_Elements(wGroup) eq 0)then $
     Message,'Group leader must be specified for Modal or Floating'+ $
      ' top level bases'

  Resolve_Routine, 'FittingInfoWid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Info = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Info' ,XOFFSET=5 ,YOFFSET=5 ,SCR_XSIZE=330  $
      ,SCR_YSIZE=736 ,NOTIFY_REALIZE='DoRealizeInfo' ,TITLE='Fitting'+ $
      ' Info' ,SPACE=3 ,XPAD=3 ,YPAD=3 ,/MODAL)

  
  WID_TABLE_InfoFile = Widget_Table(WID_BASE_Info,  $
      UNAME='WID_TABLE_InfoFile' ,FRAME=1 ,XOFFSET=29 ,YOFFSET=11  $
      ,SCR_XSIZE=260 ,SCR_YSIZE=510 ,/EDITABLE ,COLUMN_LABELS=[  $
      'Values' ] ,ROW_LABELS=[ 'Zero Dark Cnt', 'X pixels Data', 'Y'+ $
      ' pixels Data', 'Max # Frames', 'Initial Frame', 'Final Frame',  $
      'Peak Threshold Criteria', 'File type (0 - .dat, 1 - .tif)',  $
      'Min Peak Ampl.', 'Max Peak Ampl.', 'Min Peak Width', 'Max Peak'+ $
      ' Width', 'Limit ChiSq', 'Counts Per Electron', 'Max # Peaks'+ $
      ' Iter1', 'Max # Peaks Iter2', 'Flip Horizontal', 'Flip'+ $
      ' Vertical', '(Half) Gauss Size (d)', 'Appr. Gauss Sigma', 'Max'+ $
      ' Block Size', 'Sparse OverSampling', 'Sparse Lambda', 'Sparse'+ $
      ' Delta', 'Sp. Max Error', 'Sp. Max # of Iter.' ] ,XSIZE=1  $
      ,YSIZE=26)

  
  WID_BUTTON_Info_OK = Widget_Button(WID_BASE_Info,  $
      UNAME='WID_BUTTON_Info_OK' ,XOFFSET=171 ,YOFFSET=642  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Confirm and'+ $
      ' Start Fit')

  
  WID_DROPLIST_FitDisplayType = Widget_Droplist(WID_BASE_Info,  $
      UNAME='WID_DROPLIST_FitDisplayType' ,XOFFSET=38 ,YOFFSET=530  $
      ,SCR_XSIZE=225 ,SCR_YSIZE=30  $
      ,NOTIFY_REALIZE='DoRealizeDropListDispType' ,TITLE='Fit-Display'+ $
      ' Level' ,VALUE=[ 'No Display', 'Some Frames/Peaks', 'All'+ $
      ' Frames/Peaks ', 'Cluster - No Display', 'IDL Bridge - No'+ $
      ' Disp' ])

  
  WID_DROPLIST_SetSigmaFitSym = Widget_Droplist(WID_BASE_Info,  $
      UNAME='WID_DROPLIST_SetSigmaFitSym' ,XOFFSET=40 ,YOFFSET=565  $
      ,SCR_XSIZE=225 ,SCR_YSIZE=30 ,TITLE='SetSigmaFitSymmetry'  $
      ,VALUE=[ 'R', 'X Y' ])

  
  WID_BUTTON_CancelFit = Widget_Button(WID_BASE_Info,  $
      UNAME='WID_BUTTON_CancelFit' ,XOFFSET=11 ,YOFFSET=642  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Cancel')

  
  WID_DROPLIST_Localization_Method = Widget_Droplist(WID_BASE_Info,  $
      UNAME='WID_DROPLIST_Localization_Method' ,XOFFSET=43  $
      ,YOFFSET=605 ,SCR_XSIZE=225 ,SCR_YSIZE=30 ,TITLE='Localization'+ $
      ' method' ,VALUE=[ 'Gaussian Fit', 'Sparse Sampling' ])

  Widget_Control, /REALIZE, WID_BASE_Info

  XManager, 'WID_BASE_Info', WID_BASE_Info, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro FittingInfoWid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Info, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
