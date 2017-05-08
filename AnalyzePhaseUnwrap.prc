HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	11/05/2010 13:26.34
VERSION 1
END

WID_BASE_AnalyzePhaseUnwrap BASE 5 5 348 259
REALIZE "Initialize_AnalyzePhaseUnwrap"
TLB
CAPTION "Analyze Phase Unwrapping and Localizations"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Analyze_PhaseUnwrap_OK PUSHBUTTON 115 117 120 30
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "On_Analyze_PhaseUnwrap_Start"
  END
  WID_LABEL_NumFramesPerStep LABEL 20 25 175 15
  VALUE "Number of Frames Per Step"
  ALIGNLEFT
  END
  WID_number_frames_per_step TEXT 220 15 100 30
  NUMITEMS = 1
  ITEM "100"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_nm_per_step TEXT 220 65 100 30
  NUMITEMS = 1
  ITEM "8"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_nm_per_step LABEL 25 75 175 15
  VALUE "NM per step"
  ALIGNLEFT
  END
  WID_TEXT_ResultsFilename TEXT 14 168 311 53
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
END
