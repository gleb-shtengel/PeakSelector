; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	11/22/2017 10:31.35
; 
pro WID_BASE_Process_Multiple_PALM_Slabs_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Process_Multiple_PALM_Slabs'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel_Macro_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancel_Macro_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_SelectDirectory_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Select_Directory_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_Filter_Parameters_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Do_Change_Filter_Params_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Start_Macro_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Start_Macro_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_ReFind_Files_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_ReFind_Files_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_Remove_Selected_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Remove_Selected_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_AutoFindFiducials_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSet_Button_AutoFindFiducials_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_DriftCorrect_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSet_Button_DriftCorrect_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Group_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSet_Button_PerfromGrouping_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_RegisterToScaffold_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSet_Button_RegistertoScaffold_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_AutoFindFiducials_Parameters_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        DoInsert_Autodetect_Param_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickScaffoldFiducials_mSlab'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPick_ScaffoldFiducials_File, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PerformFiltering_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSet_Button_PerformFiltering_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PerformPurging_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSet_Button_PerformPurging_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_ZStep_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        On_Change_ZStep_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_Purge_Parameters_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Do_Change_Purge_Params_mSlabs, Event
    end
    else:
  endcase

end

pro WID_BASE_Process_Multiple_PALM_Slabs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Process_Multiple_Palm_Slabs_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Process_Multiple_PALM_Slabs = Widget_Base(  $
      GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Process_Multiple_PALM_Slabs' ,XOFFSET=5  $
      ,YOFFSET=5 ,SCR_XSIZE=964 ,SCR_YSIZE=979  $
      ,NOTIFY_REALIZE='Initialize_Process_Multiple_PALM_Slabs'  $
      ,TITLE='Process Multiple PALM Sbals' ,SPACE=3 ,XPAD=3 ,YPAD=3  $
      ,TLB_FRAME_ATTR=1)

  
  WID_BUTTON_Cancel_Macro_mSlabs =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BUTTON_Cancel_Macro_mSlabs' ,XOFFSET=193  $
      ,YOFFSET=879 ,SCR_XSIZE=130 ,SCR_YSIZE=40 ,/ALIGN_CENTER  $
      ,VALUE='Cancel')

  
  WID_BTTN_SelectDirectory_mSlabs =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BTTN_SelectDirectory_mSlabs' ,XOFFSET=415  $
      ,YOFFSET=10 ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Pick Directory')

  
  WID_TXT_mSlabs_Directory =  $
      Widget_Text(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_TXT_mSlabs_Directory' ,XOFFSET=10 ,YOFFSET=5  $
      ,SCR_XSIZE=400 ,SCR_YSIZE=50 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_Filter_Parameters_mSlabs =  $
      Widget_Table(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_Filter_Parameters_mSlabs' ,XOFFSET=580 ,YOFFSET=110  $
      ,SCR_XSIZE=360 ,SCR_YSIZE=157 ,/EDITABLE ,COLUMN_LABELS=[  $
      'Min', 'Max' ] ,ROW_LABELS=[ 'Amplitude', 'Sigma X Pos Full',  $
      'Sigma Y Pos Full', 'Z Position', 'Sigma Z' ] ,XSIZE=2  $
      ,YSIZE=10)

  
  WID_BUTTON_Start_Macro_mSlabs =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BUTTON_Start_Macro_mSlabs' ,XOFFSET=29 ,YOFFSET=880  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=40 ,/ALIGN_CENTER ,VALUE='Confirm and'+ $
      ' Start')

  
  WID_BTTN_ReFind_Files_mSlabs =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BTTN_ReFind_Files_mSlabs' ,XOFFSET=24 ,YOFFSET=120  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Re-Find'+ $
      ' Files')

  
  WID_LIST_Process_mSlabs =  $
      Widget_List(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_LIST_Process_mSlabs' ,XOFFSET=12 ,YOFFSET=182  $
      ,SCR_XSIZE=370 ,SCR_YSIZE=542 ,XSIZE=11 ,YSIZE=2)

  
  WID_BASE_Include_Subdirectories_mSlabs =  $
      Widget_Base(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BASE_Include_Subdirectories_mSlabs' ,XOFFSET=200  $
      ,YOFFSET=120 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Include_Subdirectories_mSlabs =  $
      Widget_Button(WID_BASE_Include_Subdirectories_mSlabs,  $
      UNAME='WID_BUTTON_Include_Subdirectories_mSlabs' ,/ALIGN_LEFT  $
      ,VALUE='Include Subdirectories')

  
  WID_LABEL_nfiles_mSlabs =  $
      Widget_Label(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_LABEL_nfiles_mSlabs' ,XOFFSET=14 ,YOFFSET=152  $
      ,SCR_XSIZE=190 ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='')

  
  WID_BTTN_Remove_Selected_mSlabs =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BTTN_Remove_Selected_mSlabs' ,XOFFSET=60  $
      ,YOFFSET=829 ,SCR_XSIZE=230 ,SCR_YSIZE=40 ,/ALIGN_CENTER  $
      ,VALUE='Remove Selected Files')

  
  WID_TXT_mSlabs_FileMask =  $
      Widget_Text(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_TXT_mSlabs_FileMask' ,XOFFSET=540 ,YOFFSET=5  $
      ,SCR_XSIZE=175 ,SCR_YSIZE=50 ,/EDITABLE ,/WRAP ,VALUE=[  $
      '*_IDL.sav' ] ,XSIZE=20 ,YSIZE=2)

  
  WID_BASE_AutoFindFiducials_mSlabs =  $
      Widget_Base(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BASE_AutoFindFiducials_mSlabs' ,XOFFSET=410  $
      ,YOFFSET=300 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_AutoFindFiducials_mSlabs =  $
      Widget_Button(WID_BASE_AutoFindFiducials_mSlabs,  $
      UNAME='WID_BUTTON_AutoFindFiducials_mSlabs' ,/ALIGN_LEFT  $
      ,VALUE='Auto Detect Fiducials')

  
  WID_BASE_DriftCorrection_mSlabs =  $
      Widget_Base(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BASE_DriftCorrection_mSlabs' ,XOFFSET=409  $
      ,YOFFSET=417 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_DriftCorrect_mSlabs =  $
      Widget_Button(WID_BASE_DriftCorrection_mSlabs,  $
      UNAME='WID_BUTTON_DriftCorrect_mSlabs' ,/ALIGN_LEFT  $
      ,VALUE='Perform Drift Correction')

  
  WID_BASE_Group_mSlabs =  $
      Widget_Base(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BASE_Group_mSlabs' ,XOFFSET=410 ,YOFFSET=650  $
      ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Group_mSlabs = Widget_Button(WID_BASE_Group_mSlabs,  $
      UNAME='WID_BUTTON_Group_mSlabs' ,/ALIGN_LEFT ,VALUE='Perform'+ $
      ' Grouping')

  
  WID_SLIDER_FramesPerNode_mSlabs =  $
      Widget_Slider(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_SLIDER_FramesPerNode_mSlabs' ,XOFFSET=579  $
      ,YOFFSET=719 ,SCR_XSIZE=142 ,SCR_YSIZE=48 ,TITLE='Frames per'+ $
      ' Node (Cluster)' ,MINIMUM=0 ,MAXIMUM=10000 ,VALUE=500)

  
  WID_SLIDER_Group_Gap_mSlabs =  $
      Widget_Slider(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_SLIDER_Group_Gap_mSlabs' ,XOFFSET=579 ,YOFFSET=649  $
      ,SCR_XSIZE=149 ,SCR_YSIZE=48 ,TITLE='Group Gap' ,MAXIMUM=256  $
      ,VALUE=3)

  
  WID_SLIDER_Grouping_Radius_mSlabs =  $
      Widget_Slider(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_SLIDER_Grouping_Radius_mSlabs' ,XOFFSET=759  $
      ,YOFFSET=649 ,SCR_XSIZE=146 ,SCR_YSIZE=48 ,TITLE='Grouping'+ $
      ' Radius*100' ,MAXIMUM=200 ,VALUE=25)

  
  WID_DROPLIST_GroupEngine_mSlabs =  $
      Widget_Droplist(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_DROPLIST_GroupEngine_mSlabs' ,XOFFSET=743  $
      ,YOFFSET=719 ,SCR_XSIZE=181 ,SCR_YSIZE=22 ,TITLE='Grouping'+ $
      ' Engine' ,VALUE=[ 'Local', 'Cluster', 'IDL Bridge' ])

  
  WID_BASE_RegisterToScaffold_mSlabs =  $
      Widget_Base(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BASE_RegisterToScaffold_mSlabs' ,XOFFSET=410  $
      ,YOFFSET=806 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_RegisterToScaffold_mSlabs =  $
      Widget_Button(WID_BASE_RegisterToScaffold_mSlabs,  $
      UNAME='WID_BUTTON_RegisterToScaffold_mSlabs' ,/ALIGN_LEFT  $
      ,VALUE='Register to Scaffold')

  
  WID_AutoFindFiducials_Parameters_mSlabs =  $
      Widget_Table(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_AutoFindFiducials_Parameters_mSlabs' ,XOFFSET=580  $
      ,YOFFSET=300 ,/EDITABLE ,COLUMN_LABELS=[ 'Value' ]  $
      ,ROW_LABELS=[ 'Thr. Min.', 'Thr. Max.', 'Rad. (pix.)' ]  $
      ,XSIZE=1 ,YSIZE=3)

  
  WID_BUTTON_PickScaffoldFiducials_mSlab =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BUTTON_PickScaffoldFiducials_mSlab' ,XOFFSET=599  $
      ,YOFFSET=806 ,SCR_XSIZE=244 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Pick Scaffold Fiducials File')

  
  WID_TEXT_ScaffoldFiducialsFile_mSlabs =  $
      Widget_Text(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_TEXT_ScaffoldFiducialsFile_mSlabs' ,XOFFSET=599  $
      ,YOFFSET=846 ,SCR_XSIZE=296 ,SCR_YSIZE=55 ,/EDITABLE ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_BASE_PerformFiltering_mSlabs =  $
      Widget_Base(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BASE_PerformFiltering_mSlabs' ,XOFFSET=410  $
      ,YOFFSET=110 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_PerformFiltering_mSlabs =  $
      Widget_Button(WID_BASE_PerformFiltering_mSlabs,  $
      UNAME='WID_BUTTON_PerformFiltering_mSlabs' ,/ALIGN_LEFT  $
      ,VALUE='Perfrom Filtering')

  
  WID_BASE_PerformPurging_mSlabs =  $
      Widget_Base(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BASE_PerformPurging_mSlabs' ,XOFFSET=410  $
      ,YOFFSET=495 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_PerformPurging_mSlabs =  $
      Widget_Button(WID_BASE_PerformPurging_mSlabs,  $
      UNAME='WID_BUTTON_PerformPurging_mSlabs' ,/ALIGN_LEFT  $
      ,VALUE='Perfrom Purging')

  
  WID_TEXT_ZStep_mSlabs =  $
      Widget_Text(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_TEXT_ZStep_mSlabs' ,XOFFSET=231 ,YOFFSET=747  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=35 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_LABEL_Z_Step_mSlabs =  $
      Widget_Label(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_LABEL_Z_Step_mSlabs' ,XOFFSET=51 ,YOFFSET=757  $
      ,SCR_XSIZE=180 ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='Z Step (nm)'+ $
      ' between slabs')

  
  WID_Purge_Parameters_mSlabs =  $
      Widget_Table(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_Purge_Parameters_mSlabs' ,XOFFSET=580 ,YOFFSET=480  $
      ,SCR_XSIZE=360 ,SCR_YSIZE=157 ,/EDITABLE ,COLUMN_LABELS=[  $
      'Min', 'Max' ] ,ROW_LABELS=[ 'Amplitude', 'Sigma X Pos Full',  $
      'Sigma Y Pos Full', 'Z Position', 'Sigma Z' ] ,XSIZE=2  $
      ,YSIZE=10)

  Widget_Control, /REALIZE, WID_BASE_Process_Multiple_PALM_Slabs

  XManager, 'WID_BASE_Process_Multiple_PALM_Slabs', WID_BASE_Process_Multiple_PALM_Slabs, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Process_Multiple_Palm_Slabs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Process_Multiple_PALM_Slabs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
