HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	11/20/2017 10:35.09
VERSION 1
END

WID_BASE_Extract_Peaks_Multiple_TIFFs BASE 5 5 749 1054
REALIZE "Initialize_Extract_Peaks_mTIFFs"
TLB
CAPTION "Extract Peaks from multiple TIFF files in a single directory"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Cancel_Extract_mTIFFS PUSHBUTTON 194 915 130 40
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancel_Extract_mTIFFS"
  END
  WID_DROPLIST_SetSigmaFitSym_mTIFFS DROPLIST 415 690 290 30
  CAPTION "SetSigmaFitSymmetry"
  NUMITEMS = 3
  ITEM "R"
  ITEM "X Y unconstrained"
  ITEM "X Y constr: SigX(Z), SigY(Z)"
  ONSELECT "Set_SigmaFitSym_mTIFFS"
  END
  WID_BTTN_SelectDirectory PUSHBUTTON 415 10 100 30
  VALUE "Pick Directory"
  ALIGNCENTER
  ONACTIVATE "On_Select_Directory"
  END
  WID_TXT_mTIFFS_Directory TEXT 10 5 400 50
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_DROPLIST_TransformEngine_mTIFFS DROPLIST 480 660 225 30
  CAPTION "Transformation Engine"
  NUMITEMS = 3
  ITEM "Local"
  ITEM "Cluster"
  ITEM "IDL Bridge"
  ONSELECT "Set_TransformEngine_mTIFFS"
  END
  WID_BUTTON_Start_Extract_mTIFFS PUSHBUTTON 30 916 150 40
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "Start_mTIFFS_Extract"
  END
  WID_TABLE_InfoFile_mTIFFS TABLE 410 150 300 500
  FRAME = 1
  N_ROWS = 23
  N_COLS = 1
  NUMCOLLABELS = 1
  COLLABEL "Values"
  NUMROWLABELS = 23
  ROWLABEL "Zero Dark Cnt"
  ROWLABEL "X pixels Data"
  ROWLABEL "Y pixels Data"
  ROWLABEL "Peak Threshold Criteria"
  ROWLABEL "File type (0 - .dat, 1 - .tif)"
  ROWLABEL "Min Peak Ampl."
  ROWLABEL "Max Peak Ampl."
  ROWLABEL "Min Peak Width"
  ROWLABEL "Max Peak Width"
  ROWLABEL "Limit ChiSq"
  ROWLABEL "Counts Per Electron"
  ROWLABEL "Max # Peaks Iter1"
  ROWLABEL "Max # Peaks Iter2"
  ROWLABEL "Flip Horizontal"
  ROWLABEL "Flip Vertical"
  ROWLABEL "(Half) Gauss Size (d)"
  ROWLABEL "Appr. Gauss Sigma"
  ROWLABEL "Max Block Size"
  ROWLABEL "Sparse OverSampling"
  ROWLABEL "Sparse Lambda"
  ROWLABEL "Sparse Delta"
  ROWLABEL "Sp. Max Error"
  ROWLABEL "Sp. Max # of Iter."
  EDITABLE
  ONINSERTCHAR "Do_Change_TIFF_Info_mTIFFS"
  END
  WID_BTTN_ReFind_Files_mTIFFS PUSHBUTTON 24 120 150 30
  VALUE "Re-Find Files"
  ALIGNCENTER
  ONACTIVATE "On_ReFind_Files_mTIFFS"
  END
  WID_LIST_Extract_mTIFFS LIST 12 180 370 680
  WIDTH = 11
  HEIGHT = 2
  END
  WID_BASE_Include_Subdirectories_mTIFFs BASE 200 120 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Include_Subdirectories_mTIFFs PUSHBUTTON -1 -1 0 0
    VALUE "Include Subdirectories"
    ALIGNLEFT
    END
  END
  WID_LABEL_nfiles_mTIFFs LABEL 14 152 190 25
  VALUE ""
  ALIGNLEFT
  END
  WID_BTTN_Remove_Selected_mTIFFS PUSHBUTTON 61 865 230 40
  VALUE "Remove Selected Files"
  ALIGNCENTER
  ONACTIVATE "On_Remove_Selected_mTIFFS"
  END
  WID_BTTN_Read_TIFF_Info_mTIFFS PUSHBUTTON 410 117 120 30
  VALUE "Read TIFF Info"
  ALIGNCENTER
  ONACTIVATE "On_Read_TIFF_Info_mTIFFS"
  END
  WID_BASE_ExclPKS BASE 460 63 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Excl_PKS PUSHBUTTON -1 -1 0 0
    VALUE "Excl PKS."
    ALIGNLEFT
    END
  END
  WID_TXT_mTIFFS_FileMask TEXT 540 5 175 50
  NUMITEMS = 1
  ITEM "*.tif"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_ExclPKS_explanation TEXT 560 63 160 70
  TAB_MODE = 1
  SCROLL
  NUMITEMS = 4
  ITEM "Check this to skip"
  ITEM "processing files that had"
  ITEM "already been processed"
  ITEM "and have *.PKS files"
  WRAP
  WIDTH = 20
  HEIGHT = 1
  END
  WID_BASE_Include_UseGobIni_mTIFFs BASE 10 60 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_UseGlobIni_mTIFFs PUSHBUTTON -1 -1 0 0
    VALUE "Use Glob Ini"
    ALIGNLEFT
    ONACTIVATE "Set_UseGlobIni_mTIFFs"
    END
  END
  WID_TXT_mTIFFS_GlobINI_File TEXT 130 57 200 60
  NUMITEMS = 1
  ITEM "*.ini"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BTTN_PickGlobIni PUSHBUTTON 340 64 100 30
  VALUE "Pick Glob Ini"
  ALIGNCENTER
  ONACTIVATE "On_Select_GolbIni"
  END
  WID_LABEL_nfiles_Glob_mTIFFs LABEL 210 152 190 25
  VALUE ""
  ALIGNLEFT
  END
  WID_BUTTON_PickCalFile_MultiTIFF PUSHBUTTON 415 720 150 32
  VALUE "Pick CAL (WND) File"
  ALIGNCENTER
  ONACTIVATE "OnPickCalFile_Astig_MultiTIFF"
  END
  WID_TEXT_WindFilename_Astig_MultiTIFF TEXT 400 750 310 70
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_Filter_Parameters_mTIFFs TABLE 380 830 340 170
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
  ONINSERTCHAR "Do_Change_Filter_Params_mTIFFs"
  END
END
