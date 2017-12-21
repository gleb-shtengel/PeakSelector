; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	11/20/2017 10:35.16
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
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UseGlobIni_mTIFFs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_UseGlobIni_mTIFFs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_PickGlobIni'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Select_GolbIni, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickCalFile_MultiTIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickCalFile_Astig_MultiTIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_Filter_Parameters_mTIFFs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Do_Change_Filter_Params_mTIFFs, Event
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
      ,YOFFSET=5 ,SCR_XSIZE=749 ,SCR_YSIZE=1054  $
      ,NOTIFY_REALIZE='Initialize_Extract_Peaks_mTIFFs'  $
      ,TITLE='Extract Peaks from multiple TIFF files in a single'+ $
      ' directory' ,SPACE=3 ,XPAD=3 ,YPAD=3 ,/MODAL)

  
  WID_BUTTON_Cancel_Extract_mTIFFS =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BUTTON_Cancel_Extract_mTIFFS' ,XOFFSET=194  $
      ,YOFFSET=915 ,SCR_XSIZE=130 ,SCR_YSIZE=40 ,/ALIGN_CENTER  $
      ,VALUE='Cancel')

  
  WID_DROPLIST_SetSigmaFitSym_mTIFFS =  $
      Widget_Droplist(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_DROPLIST_SetSigmaFitSym_mTIFFS' ,XOFFSET=415  $
      ,YOFFSET=690 ,SCR_XSIZE=290 ,SCR_YSIZE=30  $
      ,TITLE='SetSigmaFitSymmetry' ,VALUE=[ 'R', 'X Y unconstrained',  $
      'X Y constr: SigX(Z), SigY(Z)' ])

  
  WID_BTTN_SelectDirectory =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BTTN_SelectDirectory' ,XOFFSET=415 ,YOFFSET=10  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick'+ $
      ' Directory')

  
  WID_TXT_mTIFFS_Directory =  $
      Widget_Text(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_TXT_mTIFFS_Directory' ,XOFFSET=10 ,YOFFSET=5  $
      ,SCR_XSIZE=400 ,SCR_YSIZE=50 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_DROPLIST_TransformEngine_mTIFFS =  $
      Widget_Droplist(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_DROPLIST_TransformEngine_mTIFFS' ,XOFFSET=480  $
      ,YOFFSET=660 ,SCR_XSIZE=225 ,SCR_YSIZE=30  $
      ,TITLE='Transformation Engine' ,VALUE=[ 'Local', 'Cluster',  $
      'IDL Bridge' ])

  
  WID_BUTTON_Start_Extract_mTIFFS =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BUTTON_Start_Extract_mTIFFS' ,XOFFSET=30  $
      ,YOFFSET=916 ,SCR_XSIZE=150 ,SCR_YSIZE=40 ,/ALIGN_CENTER  $
      ,VALUE='Confirm and Start')

  
  WID_TABLE_InfoFile_mTIFFS =  $
      Widget_Table(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_TABLE_InfoFile_mTIFFS' ,FRAME=1 ,XOFFSET=410  $
      ,YOFFSET=150 ,SCR_XSIZE=300 ,SCR_YSIZE=500 ,/EDITABLE  $
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
      UNAME='WID_BTTN_ReFind_Files_mTIFFS' ,XOFFSET=24 ,YOFFSET=120  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Re-Find'+ $
      ' Files')

  
  WID_LIST_Extract_mTIFFS =  $
      Widget_List(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_LIST_Extract_mTIFFS' ,XOFFSET=12 ,YOFFSET=180  $
      ,SCR_XSIZE=370 ,SCR_YSIZE=680 ,XSIZE=11 ,YSIZE=2)

  
  WID_BASE_Include_Subdirectories_mTIFFs =  $
      Widget_Base(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BASE_Include_Subdirectories_mTIFFs' ,XOFFSET=200  $
      ,YOFFSET=120 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Include_Subdirectories_mTIFFs =  $
      Widget_Button(WID_BASE_Include_Subdirectories_mTIFFs,  $
      UNAME='WID_BUTTON_Include_Subdirectories_mTIFFs' ,/ALIGN_LEFT  $
      ,VALUE='Include Subdirectories')

  
  WID_LABEL_nfiles_mTIFFs =  $
      Widget_Label(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_LABEL_nfiles_mTIFFs' ,XOFFSET=14 ,YOFFSET=152  $
      ,SCR_XSIZE=190 ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='')

  
  WID_BTTN_Remove_Selected_mTIFFS =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BTTN_Remove_Selected_mTIFFS' ,XOFFSET=61  $
      ,YOFFSET=865 ,SCR_XSIZE=230 ,SCR_YSIZE=40 ,/ALIGN_CENTER  $
      ,VALUE='Remove Selected Files')

  
  WID_BTTN_Read_TIFF_Info_mTIFFS =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BTTN_Read_TIFF_Info_mTIFFS' ,XOFFSET=410  $
      ,YOFFSET=117 ,SCR_XSIZE=120 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Read TIFF Info')

  
  WID_BASE_ExclPKS =  $
      Widget_Base(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BASE_ExclPKS' ,XOFFSET=460 ,YOFFSET=63 ,TITLE='IDL'  $
      ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Excl_PKS = Widget_Button(WID_BASE_ExclPKS,  $
      UNAME='WID_BUTTON_Excl_PKS' ,/ALIGN_LEFT ,VALUE='Excl PKS.')

  
  WID_TXT_mTIFFS_FileMask =  $
      Widget_Text(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_TXT_mTIFFS_FileMask' ,XOFFSET=540 ,YOFFSET=5  $
      ,SCR_XSIZE=175 ,SCR_YSIZE=50 ,/EDITABLE ,/WRAP ,VALUE=[ '*.tif'  $
      ] ,XSIZE=20 ,YSIZE=2)

  
  WID_TEXT_ExclPKS_explanation =  $
      Widget_Text(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_TEXT_ExclPKS_explanation' ,XOFFSET=560 ,YOFFSET=63  $
      ,SCR_XSIZE=160 ,SCR_YSIZE=70 ,TAB_MODE=0 ,/SCROLL ,/WRAP  $
      ,VALUE=[ 'Check this to skip', 'processing files that had',  $
      'already been processed', 'and have *.PKS files' ] ,XSIZE=20  $
      ,YSIZE=1)

  
  WID_BASE_Include_UseGobIni_mTIFFs =  $
      Widget_Base(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BASE_Include_UseGobIni_mTIFFs' ,XOFFSET=10  $
      ,YOFFSET=60 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_UseGlobIni_mTIFFs =  $
      Widget_Button(WID_BASE_Include_UseGobIni_mTIFFs,  $
      UNAME='WID_BUTTON_UseGlobIni_mTIFFs' ,/ALIGN_LEFT ,VALUE='Use'+ $
      ' Glob Ini')

  
  WID_TXT_mTIFFS_GlobINI_File =  $
      Widget_Text(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_TXT_mTIFFS_GlobINI_File' ,XOFFSET=130 ,YOFFSET=57  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=60 ,/EDITABLE ,/WRAP ,VALUE=[ '*.ini'  $
      ] ,XSIZE=20 ,YSIZE=2)

  
  WID_BTTN_PickGlobIni =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BTTN_PickGlobIni' ,XOFFSET=340 ,YOFFSET=64  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick Glob'+ $
      ' Ini')

  
  WID_LABEL_nfiles_Glob_mTIFFs =  $
      Widget_Label(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_LABEL_nfiles_Glob_mTIFFs' ,XOFFSET=210 ,YOFFSET=152  $
      ,SCR_XSIZE=190 ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='')

  
  WID_BUTTON_PickCalFile_MultiTIFF =  $
      Widget_Button(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_BUTTON_PickCalFile_MultiTIFF' ,XOFFSET=415  $
      ,YOFFSET=720 ,SCR_XSIZE=150 ,SCR_YSIZE=32 ,/ALIGN_CENTER  $
      ,VALUE='Pick CAL (WND) File')

  
  WID_TEXT_WindFilename_Astig_MultiTIFF =  $
      Widget_Text(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_TEXT_WindFilename_Astig_MultiTIFF' ,XOFFSET=400  $
      ,YOFFSET=750 ,SCR_XSIZE=310 ,SCR_YSIZE=70 ,/EDITABLE ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_Filter_Parameters_mTIFFs =  $
      Widget_Table(WID_BASE_Extract_Peaks_Multiple_TIFFs,  $
      UNAME='WID_Filter_Parameters_mTIFFs' ,XOFFSET=380 ,YOFFSET=830  $
      ,SCR_XSIZE=340 ,SCR_YSIZE=170 ,/EDITABLE ,COLUMN_LABELS=[  $
      'Min', 'Max' ] ,ROW_LABELS=[ 'Amplitude', 'Sigma X Pos Full',  $
      'Sigma Y Pos Full', 'Z Position', 'Sigma Z' ] ,XSIZE=2  $
      ,YSIZE=10)

  Widget_Control, /REALIZE, WID_BASE_Extract_Peaks_Multiple_TIFFs

  XManager, 'WID_BASE_Extract_Peaks_Multiple_TIFFs', WID_BASE_Extract_Peaks_Multiple_TIFFs, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Extract_Peaks_Multiple_TIFFs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Extract_Peaks_Multiple_TIFFs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
