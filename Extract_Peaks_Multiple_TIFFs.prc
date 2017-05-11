HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	05/09/2017 14:36.58
VERSION 1
END

WID_BASE_Extract_Peaks_Multiple_TIFFs BASE 5 5 749 952
REALIZE "Initialize_Extract_Peaks_mTIFFs"
TLB
CAPTION "Extract Peaks from multiple TIFF files in a single directory"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Cancel_Extract_mTIFFS PUSHBUTTON 579 847 130 40
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancel_Extract_mTIFFS"
  END
  WID_DROPLIST_SetSigmaFitSym_mTIFFS DROPLIST 499 653 200 30
  CAPTION "SetSigmaFitSymmetry"
  NUMITEMS = 2
  ITEM "R"
  ITEM "X Y"
  ONSELECT "Set_SigmaFitSym_mTIFFS"
  END
  WID_BTTN_SelectDirectory PUSHBUTTON 515 10 200 30
  VALUE "Select Directory"
  ALIGNCENTER
  ONACTIVATE "On_Select_Directory"
  END
  WID_TXT_mTIFFS_Directory TEXT 10 10 500 40
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_DROPLIST_TransformEngine_mTIFFS DROPLIST 484 618 225 30
  CAPTION "Transformation Engine"
  NUMITEMS = 3
  ITEM "Local"
  ITEM "Cluster"
  ITEM "IDL Bridge"
  ONSELECT "Set_TransformEngine_mTIFFS"
  END
  WID_Filter_Parameters_mTIFFS TABLE 409 697 300 120
  N_ROWS = 10
  N_COLS = 1
  NUMCOLLABELS = 1
  COLLABEL "Value"
  NUMROWLABELS = 3
  ROWLABEL "Nph. Min."
  ROWLABEL "Full Sigma X Max. (pix.)"
  ROWLABEL "Full Sigma Y Max. (pix.)"
  EDITABLE
  ONINSERTCHAR "Do_Change_Astig_Macroparams_mTIFFS"
  END
  WID_BUTTON_Start_Extract_mTIFFS PUSHBUTTON 414 847 150 40
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "Start_mTIFFS_Extract"
  END
  WID_TABLE_InfoFile_mTIFFS TABLE 409 104 300 500
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
  WID_BTTN_ReFind_Files_mTIFFS PUSHBUTTON 24 64 150 30
  VALUE "Re-Find Files"
  ALIGNCENTER
  ONACTIVATE "On_ReFind_Files_mTIFFS"
  END
  WID_LIST_Extract_mTIFFS LIST 12 134 370 700
  WIDTH = 11
  HEIGHT = 2
  END
  WID_BASE_Include_Subdirectories_mTIFFs BASE 209 64 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Include_Subdirectories_mTIFFs PUSHBUTTON -1 -1 0 0
    VALUE "Include Subdirectories"
    ALIGNLEFT
    END
  END
  WID_LABEL_nfiles_mTIFFs LABEL 14 99 300 25
  VALUE ""
  ALIGNLEFT
  END
  WID_BTTN_Remove_Selected_mTIFFS PUSHBUTTON 61 849 250 30
  VALUE "Remove Selected Files"
  ALIGNCENTER
  ONACTIVATE "On_Remove_Selected_mTIFFS"
  END
  WID_BTTN_Read_TIFF_Info_mTIFFS PUSHBUTTON 449 64 150 30
  VALUE "Read TIFF Info"
  ALIGNCENTER
  ONACTIVATE "On_Read_TIFF_Info_mTIFFS"
  END
END
