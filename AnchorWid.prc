HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	04/15/2013 12:03.54
VERSION 1
END

WID_BASE_AnchorPts BASE 5 5 1068 906
REALIZE "InitializeFidAnchWid"
TLB
CAPTION "Fiducial Anchor Points"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_Anchors_XY_Table TABLE 0 365 476 510
  N_ROWS = 500
  N_COLS = 6
  NUMCOLLABELS = 6
  COLLABEL "Red X"
  COLLABEL "Red Y"
  COLLABEL "Green X"
  COLLABEL "Green Y"
  COLLABEL "Blue X"
  COLLABEL "Blue Y"
  EDITABLE
  ONINSERTCHAR "DoInsertAnchor"
  END
  WID_BUTTON_0 PUSHBUTTON 490 100 90 30
  VALUE "Transform"
  ALIGNCENTER
  ONACTIVATE "Do_RGB_Transforms"
  END
  WID_BUTTON_AddRed PUSHBUTTON 130 58 95 30
  VALUE "Add Red Fid."
  ALIGNCENTER
  ONACTIVATE "OnButton_AddRedFiducial"
  END
  WID_BUTTON_AddGreen PUSHBUTTON 240 58 95 30
  VALUE "Add Green Fid."
  ALIGNCENTER
  ONACTIVATE "OnButton_AddGreenFiducial"
  END
  WID_BUTTON_AddBlue PUSHBUTTON 350 58 95 30
  VALUE "Add Blue Fid."
  ALIGNCENTER
  ONACTIVATE "OnButton_AddBlueFiducial"
  END
  WID_BUTTON_2 PUSHBUTTON 15 45 90 35
  VALUE "Clear All"
  ALIGNCENTER
  ONACTIVATE "ClearFiducials"
  END
  WID_LABEL_0 LABEL 115 93 368 19
  VALUE "To add new fiducial point, first select the area of the interest on the image."
  ALIGNLEFT
  END
  WID_BASE_0 BASE 130 130 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_RTG PUSHBUTTON -1 -1 0 0
    VALUE "Red to Green"
    ALIGNLEFT
    ONACTIVATE "RGB_check_buttons"
    END
  END
  WID_BASE_1 BASE 130 160 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_RTB PUSHBUTTON -1 -1 0 0
    VALUE "Red to Blue"
    ALIGNLEFT
    ONACTIVATE "RGB_check_buttons"
    END
  END
  WID_BASE_2 BASE 240 160 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_GTB PUSHBUTTON -1 -1 0 0
    VALUE "Green to Blue"
    ALIGNLEFT
    ONACTIVATE "RGB_check_buttons"
    END
  END
  WID_BASE_3 BASE 240 130 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_GTR PUSHBUTTON -1 -1 0 0
    VALUE "Green to Red"
    ALIGNLEFT
    ONACTIVATE "RGB_check_buttons"
    END
  END
  WID_BASE_4 BASE 350 130 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_BTR PUSHBUTTON -1 -1 0 0
    VALUE "Blue to Red"
    ALIGNLEFT
    ONACTIVATE "RGB_check_buttons"
    END
  END
  WID_BASE_5 BASE 350 160 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_BTG PUSHBUTTON -1 -1 0 0
    VALUE "Blue to Green"
    ALIGNLEFT
    ONACTIVATE "RGB_check_buttons"
    END
  END
  WID_BUTTON_PickANCFile PUSHBUTTON 330 205 100 30
  VALUE "Pick *.anc File"
  ALIGNCENTER
  ONACTIVATE "OnPickANCFile"
  END
  WID_TEXT_ANCFilename TEXT 12 247 420 32
  NUMITEMS = 1
  ITEM "Select *.anc File"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_Load_ANC PUSHBUTTON 10 205 140 30
  VALUE "Load Fiducials (*.anc)"
  ALIGNCENTER
  ONACTIVATE "Load_ANC_File"
  END
  WID_BUTTON_Save_ANC PUSHBUTTON 170 205 140 30
  VALUE "Save Fiducials (*.anc)"
  ALIGNCENTER
  ONACTIVATE "Save_ANC_File"
  END
  WID_LABEL_5 LABEL 13 279 420 15
  VALUE "Leave this field empty if you do not want to save fiducials into file"
  ALIGNLEFT
  END
  WID_DROPLIST_TRANSFORM_METHOD DROPLIST 208 8 226 26
  NUMITEMS = 3
  ITEM "Linear Regression"
  ITEM "POLYWARP"
  ITEM "Pivot and Average (3 pts only)"
  ONSELECT "Set_Transf_Method"
  END
  WID_LABEL_2 LABEL 19 4 193 18
  VALUE "Method for image transformation"
  ALIGNLEFT
  END
  WID_LABEL_1 LABEL 115 110 193 18
  VALUE "Then press one of the buttons above."
  ALIGNLEFT
  END
  WID_LABEL_3 LABEL 18 22 193 19
  VALUE "(when there are at least 3 fiducials)"
  ALIGNLEFT
  END
  WID_BASE_6 BASE 563 335 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Align_Z PUSHBUTTON -1 -1 0 0
    VALUE "Align Z"
    ALIGNLEFT
    END
  END
  WID_Anchors_Z_Table TABLE 481 365 276 510
  N_ROWS = 500
  N_COLS = 3
  NUMCOLLABELS = 3
  COLLABEL "Red Z"
  COLLABEL "Green Z"
  COLLABEL "Blue Z"
  EDITABLE
  ONINSERTCHAR "DoInsertZAnchor"
  END
  WID_BUTTON_DisplayFiducials PUSHBUTTON 23 300 150 35
  VALUE "Display Fiducials"
  ALIGNCENTER
  ONACTIVATE "Display_RGB_fiducials"
  END
  WID_BASE_AutoDisplayCompleteFiducialSet BASE 29 340 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_AutoDisplay_Selected_Fiducials PUSHBUTTON -1 -1 0 0
    VALUE "Display fiducials when selected"
    ALIGNLEFT
    ONACTIVATE "Set_AutoDisplay_Selected_Fiducials"
    END
  END
  WID_BUTTON_Set_Fid_outline_size PUSHBUTTON 208 300 250 35
  VALUE "Set Fiducial Outline Circle Rad. (pix)"
  ALIGNCENTER
  ONACTIVATE "SetFiducialOutlineSize"
  END
  WID_TEXT_FidOutlineSize TEXT 478 304 70 35
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_RemoveFiducial PUSHBUTTON 680 228 100 31
  VALUE "Remove Fid."
  ALIGNCENTER
  ONACTIVATE "OnButton_RemoveFiducial"
  END
  WID_TEXT_FidRemoveNumber TEXT 695 265 50 30
  NUMITEMS = 1
  ITEM "0"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_AutoDetect PUSHBUTTON 800 5 200 31
  VALUE "Autodetect Red Fiducials"
  ALIGNCENTER
  ONACTIVATE "OnButton_AutodetectRedFiducials"
  END
  WID_AutoDetect_Parameters TABLE 630 3 0 0
  N_ROWS = 3
  N_COLS = 1
  NUMCOLLABELS = 2
  COLLABEL "Value"
  COLLABEL ""
  NUMROWLABELS = 3
  ROWLABEL "Thr. Min."
  ROWLABEL "Thr. Max."
  ROWLABEL "Rad. (pix.)"
  EDITABLE
  ONINSERTCHAR "DoInsert_Autodetect_Param"
  END
  WID_BUTTON_AutoDetect_0 PUSHBUTTON 800 75 200 30
  VALUE "Autodetect Matching Fiducials"
  ALIGNCENTER
  ONACTIVATE "OnButton_AutodetectMatchingFiducials"
  END
  WID_AutoDetect_Match_Parameters TABLE 630 104 0 0
  N_ROWS = 3
  N_COLS = 1
  NUMCOLLABELS = 1
  COLLABEL "Value"
  NUMROWLABELS = 3
  ROWLABEL "Amp. Min."
  ROWLABEL "Amp. Max."
  ROWLABEL "Rad. (pix.)"
  EDITABLE
  ONINSERTCHAR "DoInsert_Autodetect_Matching_Param"
  END
  WID_Anchors_Transf_Dist_Test TABLE 770 365 276 510
  N_ROWS = 500
  N_COLS = 3
  NUMCOLLABELS = 3
  COLLABEL "Dist. R-G"
  COLLABEL "Dist. B-G"
  COLLABEL "Dist. R-B"
  EDITABLE
  END
  WID_BUTTON_Test_Transformation PUSHBUTTON 851 210 200 31
  VALUE "Test Transformation"
  ALIGNCENTER
  ONACTIVATE "TestFiducialTransformation"
  END
  WID_BASE_AutoDisplayFiducialIDs BASE 289 340 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Display_Fiducial_IDs PUSHBUTTON -1 -1 0 0
    VALUE "Display Fiducials IDs"
    ALIGNLEFT
    ONACTIVATE "Set_Display_Fiducial_IDs"
    END
  END
  WID_DROPLIST_Autodetect_Filter DROPLIST 800 42 150 25
  CAPTION "Use"
  NUMITEMS = 2
  ITEM "Frame Peaks"
  ITEM "Grouped Peaks"
  END
  WID_BUTTON_Remove_Unmatched PUSHBUTTON 800 110 200 30
  VALUE "Remove Unmatched Fiducials"
  ALIGNCENTER
  ONACTIVATE "OnButton_Remove_Unmatched"
  END
  WID_Anchors_Transf_AverageErrors TABLE 764 305 296 55
  N_ROWS = 1
  N_COLS = 3
  NUMCOLLABELS = 3
  COLLABEL "Dist. R-G"
  COLLABEL "Dist. B-G"
  COLLABEL "Dist. R-B"
  NUMROWLABELS = 1
  ROWLABEL "Average"
  EDITABLE
  END
  WID_Anchors_Transf_WorstErrors TABLE 764 245 296 55
  N_ROWS = 1
  N_COLS = 3
  NUMCOLLABELS = 3
  COLLABEL "Dist. R-G"
  COLLABEL "Dist. B-G"
  COLLABEL "Dist. R-B"
  NUMROWLABELS = 1
  ROWLABEL "Worst"
  EDITABLE
  END
  WID_SLIDER_POLYWARP_Degree SLIDER 460 1 140 48
  CAPTION "Polywarp Degree"
  VALUE = 1
  MINIMUM = 1
  MAXIMUM = 5
  ONVALUECHANGED "Set_PW_deg"
  END
  WID_BASE_RescaleSigma BASE 475 137 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Adj_Scl PUSHBUTTON -1 -1 0 0
    VALUE "Adj. Scale/Sigmas"
    ALIGNLEFT
    ONACTIVATE "Set_AdjustScale"
    END
  END
  WID_BUTTON_Remove_Bad_Fiducials PUSHBUTTON 800 153 180 30
  VALUE "Remove Bad Fiducials"
  ALIGNCENTER
  ONACTIVATE "OnButton_Remove_Bad_Fiducials"
  END
  WID_TEXT_FidRemove_Thr TEXT 995 153 56 30
  NUMITEMS = 1
  ITEM "20"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_DROPLIST_Fiducial_Source DROPLIST 460 62 125 25
  CAPTION "Use"
  NUMITEMS = 2
  ITEM "Frame Peaks"
  ITEM "Window Cntr"
  END
  WID_LABEL_andSAve_1 LABEL 795 190 260 15
  VALUE "Repeat Test+Remove till Worst is better then above"
  ALIGNLEFT
  END
  WID_BUTTON_Swap_RedGreen PUSHBUTTON 15 90 80 25
  VALUE "Swap R / G"
  ALIGNCENTER
  ONACTIVATE "OnButton_SwapRedGreen"
  END
  WID_BUTTON_Swap_RedBlue PUSHBUTTON 15 160 80 25
  VALUE "Swap R / B"
  ALIGNCENTER
  ONACTIVATE "OnButton_SwapRedBlue"
  END
  WID_BASE_XYlimits BASE 475 187 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_XYlimits PUSHBUTTON -1 -1 0 0
    VALUE "Limit X-Y coords"
    ALIGNLEFT
    ONACTIVATE "Set_LimitXY"
    END
  END
  WID_XYlimits_table TABLE 440 218 230 75
  N_ROWS = 2
  N_COLS = 2
  NUMCOLLABELS = 2
  COLLABEL "Min"
  COLLABEL "Max"
  NUMROWLABELS = 2
  ROWLABEL "X"
  ROWLABEL "Y"
  EDITABLE
  ONINSERTCHAR "Set_XY_limits"
  ONINSERTSTRING "Set_XY_limits"
  ONDELETE "Set_XY_limits"
  END
  WID_BASE_LeaveOrigTotalRaw BASE 475 162 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_LeaveOrigTotalRaw PUSHBUTTON -1 -1 0 0
    VALUE "Leave Orig. TotalRaw"
    ALIGNLEFT
    ONACTIVATE "Set_LeaveOrigTotRaw"
    END
  END
  WID_BUTTON_DisplayFiducials_with_overalys PUSHBUTTON 646 307 100 35
  VALUE "Disp Fid+Over"
  ALIGNCENTER
  ONACTIVATE "Display_RGB_fiducials_with_overlays"
  END
  WID_BUTTON_Refind PUSHBUTTON 980 40 70 30
  VALUE "Re-Find"
  ALIGNCENTER
  ONACTIVATE "OnButton_RefindFiducials"
  END
  WID_BUTTON_Copy_Red_to_Green PUSHBUTTON 16 122 80 25
  VALUE "Copy R->G"
  ALIGNCENTER
  ONACTIVATE "OnButton_Copy_Red_to_Green"
  END
END
