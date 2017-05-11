HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	12/02/2010 15:25.30
VERSION 1
END

WID_BASE_PeakSelector_INI BASE 5 5 917 726
REALIZE "DoRealize_PeakSelector_INI"
TLB
CAPTION "PeakSelector Settings"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Save_INI PUSHBUTTON 450 600 200 35
  VALUE "Force Settings + Save (*.ini)"
  ALIGNCENTER
  ONACTIVATE "Save_INI_File"
  END
  WID_BUTTON_Load_ANC PUSHBUTTON 230 600 200 35
  VALUE "Load (*.ini) + Force Settings"
  ALIGNCENTER
  ONACTIVATE "Load_INI_File"
  END
  WID_TEXT_INI_Filename TEXT 25 646 868 40
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_Pick_INI PUSHBUTTON 30 600 150 35
  VALUE "Pick *.ini File"
  ALIGNCENTER
  ONACTIVATE "OnPickINIFile"
  END
  WID_TEXT_PeakSelector_INI TEXT 10 8 880 577
  SCROLL
  EDITABLE
  WIDTH = 20
  HEIGHT = 1
  END
  WID_BUTTON_Cancel_INI_Edit PUSHBUTTON 700 600 150 35
  VALUE "Close"
  ALIGNCENTER
  ONACTIVATE "OnCancel_INI_Edit"
  END
END
