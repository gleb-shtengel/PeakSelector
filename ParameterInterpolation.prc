HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	12/03/2010 15:30.54
VERSION 1
END

WID_BASE_ParameterInterpolation BASE 5 5 337 362
REALIZE "Initialize_XY_Interp_Menu"
TLB
CAPTION "Interpolate / Subtract Trend"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Write_XY_Interp PUSHBUTTON 165 240 130 32
  VALUE "Subtract Interpolation"
  ALIGNCENTER
  ONACTIVATE "OnWrite_XY_Interp"
  END
  WID_SLIDER_XY_Poly_Interp_Order SLIDER 15 110 300 48
  CAPTION "Polyn. Interp. Order"
  VALUE = 5
  MINIMUM = 1
  MAXIMUM = 10
  END
  WID_BUTTON_Test_Interp PUSHBUTTON 10 240 130 32
  VALUE "Test Interpolation"
  ALIGNCENTER
  ONACTIVATE "OnTest_XY_Interpolation"
  END
  WID_SLIDER_XY_Interp_Sm_Width SLIDER 15 170 300 48
  CAPTION "Smoothing Width"
  VALUE = 100
  MINIMUM = 1
  MAXIMUM = 1000
  END
  WID_DROPLIST_XY_Interp_Method DROPLIST 10 75 300 30
  CAPTION "Interp. Method"
  NUMITEMS = 2
  ITEM "Polynomial"
  ITEM "Weighted Smoothing"
  END
  WID_BUTTON_Close_XY_Interp PUSHBUTTON 83 287 145 35
  VALUE "Close"
  ALIGNCENTER
  ONACTIVATE "OnButtonClose_XY_Interp"
  END
  WID_DROPLIST_Y_Interp DROPLIST 50 40 250 25
  CAPTION "Y Axis"
  NUMITEMS = 27
  ITEM "Offset"
  ITEM "Amplitude"
  ITEM "X Position"
  ITEM "Y Position"
  ITEM "X Peak Width"
  ITEM "Y Peak Width"
  ITEM "6 N Photons"
  ITEM "ChiSquared"
  ITEM "FitOK"
  ITEM "Frame Number"
  ITEM "Peak Index of Frame"
  ITEM "Peak Global Index"
  ITEM "12 Sigma Offset"
  ITEM "Sigma Amplitude"
  ITEM "Sigma X Pos rtNph"
  ITEM "Sigma Y Pos rtNph"
  ITEM "Sigma X Pos Full"
  ITEM "Sigma Y Pos Full"
  ITEM "18 Grouped Index"
  ITEM "Group X Position"
  ITEM "Group Y Position"
  ITEM "Group Sigma X Pos"
  ITEM "Group Sigma Y Pos"
  ITEM "Group N Photons"
  ITEM "24 Group Size"
  ITEM "Frame Index in Grp"
  ITEM "Label Set"
  END
  WID_DROPLIST_X_Interp DROPLIST 50 5 250 25
  CAPTION "X Axis"
  NUMITEMS = 27
  ITEM "Offset"
  ITEM "Amplitude"
  ITEM "X Position"
  ITEM "Y Position"
  ITEM "X Peak Width"
  ITEM "Y Peak Width"
  ITEM "6 N Photons"
  ITEM "ChiSquared"
  ITEM "FitOK"
  ITEM "Frame Number"
  ITEM "Peak Index of Frame"
  ITEM "Peak Global Index"
  ITEM "12 Sigma Offset"
  ITEM "Sigma Amplitude"
  ITEM "Sigma X Pos rtNph"
  ITEM "Sigma Y Pos rtNph"
  ITEM "Sigma X Pos Full"
  ITEM "Sigma Y Pos Full"
  ITEM "18 Grouped Index"
  ITEM "Group X Position"
  ITEM "Group Y Position"
  ITEM "Group Sigma X Pos"
  ITEM "Group Sigma Y Pos"
  ITEM "Group N Photons"
  ITEM "24 Group Size"
  ITEM "Frame Index in Grp"
  ITEM "Label Set"
  END
END
