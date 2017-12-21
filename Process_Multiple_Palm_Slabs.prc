HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	11/22/2017 10:31.30
VERSION 1
END

WID_BASE_Process_Multiple_PALM_Slabs BASE 5 5 964 979
REALIZE "Initialize_Process_Multiple_PALM_Slabs"
TLB
CAPTION "Process Multiple PALM Sbals"
XPAD = 3
YPAD = 3
SPACE = 3
SYSMENU = 1
BEGIN
  WID_BUTTON_Cancel_Macro_mSlabs PUSHBUTTON 193 879 130 40
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancel_Macro_mSlabs"
  END
  WID_BTTN_SelectDirectory_mSlabs PUSHBUTTON 415 10 100 30
  VALUE "Pick Directory"
  ALIGNCENTER
  ONACTIVATE "On_Select_Directory_mSlabs"
  END
  WID_TXT_mSlabs_Directory TEXT 10 5 400 50
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_Filter_Parameters_mSlabs TABLE 580 110 360 157
  N_ROWS = 10
  N_COLS = 2
  NUMCOLLABELS = 2
  COLLABEL "Min"
  COLLABEL "Max"
  NUMROWLABELS = 5
  ROWLABEL "Amplitude"
  ROWLABEL "Sigma X Pos Full"
  ROWLABEL "Sigma Y Pos Full"
  ROWLABEL "Z Position"
  ROWLABEL "Sigma Z"
  EDITABLE
  ONINSERTCHAR "Do_Change_Filter_Params_mSlabs"
  END
  WID_BUTTON_Start_Macro_mSlabs PUSHBUTTON 29 880 150 40
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "Start_Macro_mSlabs"
  END
  WID_BTTN_ReFind_Files_mSlabs PUSHBUTTON 24 120 150 30
  VALUE "Re-Find Files"
  ALIGNCENTER
  ONACTIVATE "On_ReFind_Files_mSlabs"
  END
  WID_LIST_Process_mSlabs LIST 12 182 370 542
  WIDTH = 11
  HEIGHT = 2
  END
  WID_BASE_Include_Subdirectories_mSlabs BASE 200 120 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Include_Subdirectories_mSlabs PUSHBUTTON -1 -1 0 0
    VALUE "Include Subdirectories"
    ALIGNLEFT
    END
  END
  WID_LABEL_nfiles_mSlabs LABEL 14 152 190 25
  VALUE ""
  ALIGNLEFT
  END
  WID_BTTN_Remove_Selected_mSlabs PUSHBUTTON 60 829 230 40
  VALUE "Remove Selected Files"
  ALIGNCENTER
  ONACTIVATE "On_Remove_Selected_mSlabs"
  END
  WID_TXT_mSlabs_FileMask TEXT 540 5 175 50
  NUMITEMS = 1
  ITEM "*_IDL.sav"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BASE_AutoFindFiducials_mSlabs BASE 410 300 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_AutoFindFiducials_mSlabs PUSHBUTTON -1 -1 0 0
    VALUE "Auto Detect Fiducials"
    ALIGNLEFT
    ONACTIVATE "OnSet_Button_AutoFindFiducials_mSlabs"
    END
  END
  WID_BASE_DriftCorrection_mSlabs BASE 409 417 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_DriftCorrect_mSlabs PUSHBUTTON -1 -1 0 0
    VALUE "Perform Drift Correction"
    ALIGNLEFT
    ONACTIVATE "OnSet_Button_DriftCorrect_mSlabs"
    END
  END
  WID_BASE_Group_mSlabs BASE 410 650 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Group_mSlabs PUSHBUTTON -1 -1 0 0
    VALUE "Perform Grouping"
    ALIGNLEFT
    ONACTIVATE "OnSet_Button_PerfromGrouping_mSlabs"
    END
  END
  WID_SLIDER_FramesPerNode_mSlabs SLIDER 579 719 142 48
  CAPTION "Frames per Node (Cluster)"
  VALUE = 500
  MINIMUM = 0
  MAXIMUM = 10000
  END
  WID_SLIDER_Group_Gap_mSlabs SLIDER 579 649 149 48
  CAPTION "Group Gap"
  VALUE = 3
  MAXIMUM = 256
  END
  WID_SLIDER_Grouping_Radius_mSlabs SLIDER 759 649 146 48
  CAPTION "Grouping Radius*100"
  VALUE = 25
  MAXIMUM = 200
  END
  WID_DROPLIST_GroupEngine_mSlabs DROPLIST 743 719 181 22
  CAPTION "Grouping Engine"
  NUMITEMS = 3
  ITEM "Local"
  ITEM "Cluster"
  ITEM "IDL Bridge"
  END
  WID_BASE_RegisterToScaffold_mSlabs BASE 410 806 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_RegisterToScaffold_mSlabs PUSHBUTTON -1 -1 0 0
    VALUE "Register to Scaffold"
    ALIGNLEFT
    ONACTIVATE "OnSet_Button_RegistertoScaffold_mSlabs"
    END
  END
  WID_AutoFindFiducials_Parameters_mSlabs TABLE 580 300 0 0
  N_ROWS = 3
  N_COLS = 1
  NUMCOLLABELS = 1
  COLLABEL "Value"
  NUMROWLABELS = 3
  ROWLABEL "Thr. Min."
  ROWLABEL "Thr. Max."
  ROWLABEL "Rad. (pix.)"
  EDITABLE
  ONINSERTCHAR "DoInsert_Autodetect_Param_mSlabs"
  END
  WID_BUTTON_PickScaffoldFiducials_mSlab PUSHBUTTON 599 806 244 30
  VALUE "Pick Scaffold Fiducials File"
  ALIGNCENTER
  ONACTIVATE "OnPick_ScaffoldFiducials_File"
  END
  WID_TEXT_ScaffoldFiducialsFile_mSlabs TEXT 599 846 296 55
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BASE_PerformFiltering_mSlabs BASE 410 110 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_PerformFiltering_mSlabs PUSHBUTTON -1 -1 0 0
    VALUE "Perfrom Filtering"
    ALIGNLEFT
    ONACTIVATE "OnSet_Button_PerformFiltering_mSlabs"
    END
  END
  WID_BASE_PerformPurging_mSlabs BASE 410 495 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_PerformPurging_mSlabs PUSHBUTTON -1 -1 0 0
    VALUE "Perfrom Purging"
    ALIGNLEFT
    ONACTIVATE "OnSet_Button_PerformPurging_mSlabs"
    END
  END
  WID_TEXT_ZStep_mSlabs TEXT 231 747 70 35
  EDITABLE
  WRAP
  ALLEVENTS
  ONINSERTCHAR "On_Change_ZStep_mSlabs"
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_Z_Step_mSlabs LABEL 51 757 180 25
  VALUE "Z Step (nm) between slabs"
  ALIGNLEFT
  END
  WID_Purge_Parameters_mSlabs TABLE 580 480 360 157
  N_ROWS = 10
  N_COLS = 2
  NUMCOLLABELS = 2
  COLLABEL "Min"
  COLLABEL "Max"
  NUMROWLABELS = 5
  ROWLABEL "Amplitude"
  ROWLABEL "Sigma X Pos Full"
  ROWLABEL "Sigma Y Pos Full"
  ROWLABEL "Z Position"
  ROWLABEL "Sigma Z"
  EDITABLE
  ONINSERTCHAR "Do_Change_Purge_Params_mSlabs"
  END
END
