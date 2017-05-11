HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	11/12/2007 10:29.28
VERSION 1
END

WID_BASE_Rot BASE 5 5 399 203
REALIZE "OnRealizeRotz"
TLB
CAPTION "Rotation"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_SLIDER_Rot SLIDER 50 40 300 50
  CAPTION "Angle (degrees)"
  VALUE = 0
  MINIMUM = -90
  MAXIMUM = 90
  END
  WID_BUTTON_Rotate PUSHBUTTON 47 113 112 36
  VALUE "Rotate"
  ALIGNCENTER
  ONACTIVATE "OnRotate"
  END
  WID_BUTTON_Cancel PUSHBUTTON 201 113 112 36
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancel"
  END
END
