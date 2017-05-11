HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	08/25/2016 11:43.30
VERSION 1
END

WID_BASE_Process_SRM BASE 5 5 674 162
REALIZE "Initialize_Process_SRM"
TLB
CAPTION "Process SRM File and create data set for PeakSelector"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Start_SRM_processing PUSHBUTTON 253 63 150 55
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "Start_SRM_Processing"
  END
  WID_BUTTON_Cancel_SRM_processing PUSHBUTTON 413 63 150 55
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancelReExtract"
  END
  WID_BUTTON_Pick_SRM_File PUSHBUTTON 509 10 150 30
  VALUE "Pick SRM File"
  ALIGNCENTER
  ONACTIVATE "OnPick_SRM_File"
  END
  WID_TEXT_SRM_Filename TEXT 4 9 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BASE_0 BASE 47 75 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_Make_DAT_Duplicates PUSHBUTTON -1 -1 0 0
    VALUE "Make Duplicate .DAT files"
    ALIGNLEFT
    END
  END
END
