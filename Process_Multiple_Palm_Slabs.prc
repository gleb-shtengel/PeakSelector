HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	01/10/2019 08:04.16
VERSION 1
END

WID_BASE_Process_Multiple_PALM_Slabs BASE 5 5 562 686
REALIZE "Initialize_Process_Multiple_PALM_Slabs"
TLB
CAPTION "Process Multiple PALM Sbals (Z Voltage States)"
XPAD = 3
YPAD = 3
SPACE = 3
SYSMENU = 1
BEGIN
  WID_BUTTON_Cancel_ZvsV_mSlabs PUSHBUTTON 217 576 130 40
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancel_ZvsV_mSlabs"
  END
  WID_BTTN_Select_RunDat_File_mSlabs PUSHBUTTON 100 7 300 30
  VALUE "Select Run Setup (.dat) File"
  ALIGNCENTER
  ONACTIVATE "On_Select_RunDat_File_mSlabs"
  END
  WID_TXT_RunDat_Filename_mSlabs TEXT 15 42 500 69
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_Parameters_mSlabs TABLE 10 163 529 157
  RESIZECOLUMNS
  N_ROWS = 10
  N_COLS = 4
  NUMCOLLABELS = 4
  COLLABEL "Z Voltage (V)"
  COLLABEL "# of Frames"
  COLLABEL "# of Trans. Frames"
  COLLABEL "Z offset (nm)"
  NUMROWLABELS = 2
  ROWLABEL "State 0"
  ROWLABEL "State 1"
  EDITABLE
  ONINSERTCHAR "Do_Change_Params_mSlabs"
  ONINSERTSTRING "Do_Change_Params_mSlabs"
  END
  WID_BUTTON_Shift_ZvsV_mSlabs PUSHBUTTON 90 460 350 35
  VALUE "Assign States + Shift Z according to State Table"
  ALIGNCENTER
  ONACTIVATE "Assign_zStates_and_Shift_ZvsV_mSlabs"
  END
  WID_BTTN_Load_RunDat_File PUSHBUTTON 100 118 300 30
  VALUE "Load Run Setup File"
  ALIGNCENTER
  ONACTIVATE "On_Load_RunDat_mSlabs"
  END
  WID_LABEL_ZvsV_Slope_mSlabs LABEL 30 355 180 25
  VALUE "Z(nm) vs Voltage(V) Slope "
  ALIGNLEFT
  END
  WID_TEXT_ZvsV_Slope_mSlabs TEXT 220 350 70 35
  EDITABLE
  WRAP
  ALLEVENTS
  ONINSERTCHAR "On_Change_ZvsV_Slope_mSlabs"
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_Assign_zStates_mSlabs PUSHBUTTON 90 415 350 35
  VALUE "Assign  Z States according to State Table"
  ALIGNCENTER
  ONACTIVATE "Assign_zStates_mSlabs"
  END
  WID_BUTTON_Assign_Transition_Frames_mSlabs PUSHBUTTON 89 507 350 35
  VALUE "Assign  Transition Frames to Z State -1"
  ALIGNCENTER
  ONACTIVATE "Assign_Transition_Frames_mSlabs"
  END
END
