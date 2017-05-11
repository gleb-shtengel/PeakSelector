; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	05/09/2017 14:37.04
; 
pro WID_BASE_Extract_Peaks_Multiple_TIFFs_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Extract_Peaks_Multiple_TIFFs'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel_Extract_mTIFFS'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancel_Extract_mTIFFS, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_SetSigmaFitSym_mTIFFS'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        Set_SigmaFitSym_mTIFFS, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_SelectDirectory'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Select_Directory, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_TransformEngine_mTIFFS'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        Set_TransformEngine_mTIFFS, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_Filter_Parameters_mTIFFS'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Do_Change_Astig_Macroparams_mTIFFS, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Start_Extract_mTIFFS'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Start_mTIFFS_Extract, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TABLE_InfoFile_mTIFFS'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Do_Change_TIFF_Info_mTIFFS, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_ReFind_Files_mTIFFS'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_ReFind_Files_mTIFFS, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_Remove_Selected_mTIFFS'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Remove_Selected_mTIFFS, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_Read_TIFF_Info_mTIFFS'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Read_TIFF_Info_mTIFFS, Event
    end
    else:
  endcase

end

pro WID_BASE_Extract_Peaks_Multiple_TIFFs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

; Floating or modal bases must have a Group Leader.

  if(N_Elements(wGroup) eq 0)then $
     Message,'Group leader must be specified for Modal or Floating'+ $
      ' top level bases'

  Resolve_Routine, 'Extract_Peaks_Multiple_TIFFs_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Extract_Peaks_Multiple_TIFFs = Widget_Base(  $
      GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Extract_Peaks_Multiple_TIFFs' ,XOFFSET=5  $
      ,YOFFSET=5 ,SCR_XSIZE=749 ,SCR_YSIZE=952  $
      ,NOTIFY_REALIZE='Initialize_Extract_Peaks_mTIFFs'  $
      ,TITLE='Extract Peaks from multiple TIFF files in a single'+ $
      ' directory' ,SPACE=3 ,XPAD=3 ,YPAD=3 ,/MODAL)

  
  WID_BUTTON_Cancel_Extract_mTIFFS =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BUTTON_Cancel_Extract_mTIFFS' ,XOFFSET=579  $
      ,YOFFSET=847 ,SCR_XSIZE=130 ,SCR_YSIZE=40 ,/ALIGN_CENTER  $
      ,VALUE='Cancel')

  
  WID_DROPLIST_SetSigmaFitSym_mTIFFS =  $
      Widget_Droplist(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_DROPLIST_SetSigmaFitSym_mTIFFS' ,XOFFSET=499  $
      ,YOFFSET=653 ,SCR_XSIZE=200 ,SCR_YSIZE=30  $
      ,TITLE='SetSigmaFitSymmetry' ,VALUE=[ 'R', 'X Y' ])

  
  WID_BTTN_SelectDirectory =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BTTN_SelectDirectory' ,XOFFSET=515 ,YOFFSET=10  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Select'+ $
      ' Directory')

  
  WID_TXT_mTIFFS_Directory =  $
      Widget_Text(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_TXT_mTIFFS_Directory' ,XOFFSET=10 ,YOFFSET=10  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=40 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_DROPLIST_TransformEngine_mTIFFS =  $
      Widget_Droplist(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_DROPLIST_TransformEngine_mTIFFS' ,XOFFSET=484  $
      ,YOFFSET=618 ,SCR_XSIZE=225 ,SCR_YSIZE=30  $
      ,TITLE='Transformation Engine' ,VALUE=[ 'Local', 'Cluster',  $
      'IDL Bridge' ])

  
  WID_Filter_Parameters_mTIFFS =  $
      Widget_Table(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_Filter_Parameters_mTIFFS' ,XOFFSET=409 ,YOFFSET=697  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=120 ,/EDITABLE ,COLUMN_LABELS=[  $
      'Value' ] ,ROW_LABELS=[ 'Nph. Min.', 'Full Sigma X Max.'+ $
      ' (pix.)', 'Full Sigma Y Max. (pix.)' ] ,XSIZE=1 ,YSIZE=10)

  
  WID_BUTTON_Start_Extract_mTIFFS =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BUTTON_Start_Extract_mTIFFS' ,XOFFSET=414  $
      ,YOFFSET=847 ,SCR_XSIZE=150 ,SCR_YSIZE=40 ,/ALIGN_CENTER  $
      ,VALUE='Confirm and Start')

  
  WID_TABLE_InfoFile_mTIFFS =  $
      Widget_Table(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_TABLE_InfoFile_mTIFFS' ,FRAME=1 ,XOFFSET=409  $
      ,YOFFSET=104 ,SCR_XSIZE=300 ,SCR_YSIZE=500 ,/EDITABLE  $
      ,COLUMN_LABELS=[ 'Values' ] ,ROW_LABELS=[ 'Zero Dark Cnt', 'X'+ $
      ' pixels Data', 'Y pixels Data', 'Peak Threshold Criteria',  $
      'File type (0 - .dat, 1 - .tif)', 'Min Peak Ampl.', 'Max Peak'+ $
      ' Ampl.', 'Min Peak Width', 'Max Peak Width', 'Limit ChiSq',  $
      'Counts Per Electron', 'Max # Peaks Iter1', 'Max # Peaks'+ $
      ' Iter2', 'Flip Horizontal', 'Flip Vertical', '(Half) Gauss'+ $
      ' Size (d)', 'Appr. Gauss Sigma', 'Max Block Size', 'Sparse'+ $
      ' OverSampling', 'Sparse Lambda', 'Sparse Delta', 'Sp. Max'+ $
      ' Error', 'Sp. Max # of Iter.' ] ,XSIZE=1 ,YSIZE=23)

  
  WID_BTTN_ReFind_Files_mTIFFS =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BTTN_ReFind_Files_mTIFFS' ,XOFFSET=24 ,YOFFSET=64  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Re-Find'+ $
      ' Files')

  
  WID_LIST_Extract_mTIFFS =  $
      Widget_List(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_LIST_Extract_mTIFFS' ,XOFFSET=12 ,YOFFSET=134  $
      ,SCR_XSIZE=370 ,SCR_YSIZE=700 ,XSIZE=11 ,YSIZE=2)

  
  WID_BASE_Include_Subdirectories_mTIFFs =  $
      Widget_Base(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BASE_Include_Subdirectories_mTIFFs' ,XOFFSET=209  $
      ,YOFFSET=64 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Include_Subdirectories_mTIFFs =  $
      Widget_Button(WID_BASE_Include_Subdirectories_mTIFFs,  $
      UNAME='WID_BUTTON_Include_Subdirectories_mTIFFs' ,/ALIGN_LEFT  $
      ,VALUE='Include Subdirectories')

  
  WID_LABEL_nfiles_mTIFFs =  $
      Widget_Label(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_LABEL_nfiles_mTIFFs' ,XOFFSET=14 ,YOFFSET=99  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='')

  
  WID_BTTN_Remove_Selected_mTIFFS =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BTTN_Remove_Selected_mTIFFS' ,XOFFSET=61  $
      ,YOFFSET=849 ,SCR_XSIZE=250 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Remove Selected Files')

  
  WID_BTTN_Read_TIFF_Info_mTIFFS =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BTTN_Read_TIFF_Info_mTIFFS' ,XOFFSET=449 ,YOFFSET=64  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Read TIFF'+ $
      ' Info')

  Widget_Control, /REALIZE, WID_BASE_Extract_Peaks_Multiple_TIFFs

  XManager, 'WID_BASE_Extract_Peaks_Multiple_TIFFs', WID_BASE_Extract_Peaks_Multiple_TIFFs, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Extract_Peaks_Multiple_TIFFs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Extract_Peaks_Multiple_TIFFs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
