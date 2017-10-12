HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	10/05/2017 17:04.27
VERSION 1
END

WID_BASE_GroupPeaks BASE 5 5 357 248
REALIZE "Initialize_GroupPeaks"
TLB
CAPTION "Group Peaks"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_DROPLIST_GroupEngine DROPLIST 57 84 181 22
  CAPTION "Grouping Engine"
  NUMITEMS = 3
  ITEM "Local"
  ITEM "Cluster"
  ITEM "IDL Bridge"
  END
  WID_SLIDER_Grouping_Radius SLIDER 176 11 146 48
  CAPTION "Grouping Radius*100"
  VALUE = 25
  MAXIMUM = 200
  END
  WID_SLIDER_Group_Gap SLIDER 11 12 149 48
  CAPTION "Group Gap"
  VALUE = 3
  MAXIMUM = 256
  END
  WID_SLIDER_FramesPerNode SLIDER 10 130 142 48
  CAPTION "Frames per Node (Cluster)"
  VALUE = 500
  MINIMUM = 0
  MAXIMUM = 10000
  END
  WID_BUTTON_GroupingOK PUSHBUTTON 195 128 129 46
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "OnGroupingInfoOK"
  END
END
