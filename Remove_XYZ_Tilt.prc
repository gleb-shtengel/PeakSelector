HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	04/30/2008 07:48.13
VERSION 1
END

WID_BASE_XYZ_Fid_Pts BASE 5 5 453 460
REALIZE "Initialize_XYZ_Fid_Wid"
TLB
CAPTION "XYZ Tilt Fiducial Points"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_TABLE_0 TABLE 50 40 279 115
  N_ROWS = 3
  N_COLS = 3
  NUMCOLLABELS = 3
  COLLABEL "X"
  COLLABEL "Y"
  COLLABEL "Z"
  EDITABLE
  ONINSERTCHAR "DoInsert_XYZ_Anchor"
  END
  WID_BUTTON_0 PUSHBUTTON 10 250 90 35
  VALUE "Transform"
  ALIGNCENTER
  ONACTIVATE "Do_XYZ_Tilt_Transforms"
  END
  WID_BUTTON_Add_XYZ_Fiducial PUSHBUTTON 120 184 115 31
  VALUE "Add XYZ Fiducial"
  ALIGNCENTER
  ONACTIVATE "OnButton_XYZ_AddFiducial"
  END
  WID_BUTTON_Clear_XYZ PUSHBUTTON 10 185 90 35
  VALUE "Clear All"
  ALIGNCENTER
  ONACTIVATE "Clear_XYZ_Fiducials"
  END
  WID_LABEL_0 LABEL 122 215 292 18
  VALUE "To add new fiducial point, first select the area of the interest"
  ALIGNLEFT
  END
  WID_LABEL_1 LABEL 124 230 280 18
  VALUE "on the image. Then press one of the buttons above."
  ALIGNLEFT
  END
  WID_BUTTON_Pick_XYZ_File PUSHBUTTON 215 268 180 35
  VALUE "Pick *.xyz File"
  ALIGNCENTER
  ONACTIVATE "OnPick_XYZ_File"
  END
  WID_TEXT_XYZ_Filename TEXT 4 377 432 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_Load_XYZ PUSHBUTTON 20 330 180 35
  VALUE "Load Fiducials (*.xyz)"
  ALIGNCENTER
  ONACTIVATE "Load_XYZ_File"
  END
  WID_BUTTON_Save_XYZ PUSHBUTTON 215 330 180 35
  VALUE "Save Fiducials (*.xyz)"
  ALIGNCENTER
  ONACTIVATE "Save_XYZ_File"
  END
  WID_LABEL_andSAve LABEL 5 290 120 15
  VALUE "(and save Fiducials"
  ALIGNLEFT
  END
  WID_LABEL_andSAve_0 LABEL 5 305 120 15
  VALUE "into ANC file)"
  ALIGNLEFT
  END
  WID_LABEL_5 LABEL 5 410 500 15
  VALUE "Leave this field empty if you do not want to save fiducials into file"
  ALIGNLEFT
  END
END
