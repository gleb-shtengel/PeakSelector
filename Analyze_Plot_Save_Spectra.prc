HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	04/20/2015 10:22.23
VERSION 1
END

WID_BASE_Analyze_Plot_Save_Spectra BASE 5 5 1100 1220
REALIZE "Initialize_Analyze_Plot_Save_Spectra"
KILLNOTIFY "Set_def_window"
TLB
CAPTION "Analyze, Plot, and Save Spectra"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_DROPLIST_Spectrum_Calc DROPLIST 280 70 225 30
  CAPTION "Calculate Spectrum"
  NUMITEMS = 4
  ITEM "Frame, with BG Subtr."
  ITEM "Frame, no BG Subtr."
  ITEM "Total, with BG Subtr."
  ITEM "Total, no BG Subtr."
  ONSELECT "Change_Spectrum_Calc"
  END
  WID_DRAW_Spectra DRAW 6 128 1024 1024
  END
  WID_BUTTON_SaveSpectrum PUSHBUTTON 10 70 125 25
  VALUE "Save Spectrum"
  ALIGNCENTER
  ONACTIVATE "OnSaveSpectrum"
  END
  WID_BUTTON_Cancel PUSHBUTTON 145 70 125 25
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancel"
  END
  WID_TABLE_BG_Subtr_Params TABLE 680 0 200 125
  RESIZECOLUMNS
  N_ROWS = 5
  N_COLS = 1
  NUMROWLABELS = 5
  ROWLABEL "BackGround Top"
  ROWLABEL "Spectrum Top"
  ROWLABEL "Spectrum Bot"
  ROWLABEL "BackGround Bot"
  ROWLABEL "Increment"
  EDITABLE
  ONINSERTCHAR "Do_Edit_BG_Subtraction_Params"
  END
  WID_LABEL_Sopectrum_Filename LABEL 8 13 0 0
  VALUE "Filename:"
  ALIGNLEFT
  END
  WID_TEXT_SpFilename TEXT 4 28 500 36
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_GB_Top_Up PUSHBUTTON 890 5 80 20
  VALUE "BG Top Up"
  ALIGNCENTER
  ONACTIVATE "BG_Top_Up"
  END
  WID_BUTTON_GB_Top_Down PUSHBUTTON 980 5 80 20
  VALUE "BG Top Down"
  ALIGNCENTER
  ONACTIVATE "BG_Top_Down"
  END
  WID_BUTTON_Sp_Top_Up PUSHBUTTON 890 28 80 20
  VALUE "Sp Top Up"
  ALIGNCENTER
  ONACTIVATE "Sp_Top_Up"
  END
  WID_BUTTON_Sp_Top_Down PUSHBUTTON 980 28 80 20
  VALUE "Sp Top Down"
  ALIGNCENTER
  ONACTIVATE "Sp_Top_Down"
  END
  WID_BUTTON_BG_Bot_Up PUSHBUTTON 890 74 80 20
  VALUE "BG Bot Up"
  ALIGNCENTER
  ONACTIVATE "BG_Bot_Up"
  END
  WID_BUTTON_BG_Bot_Down PUSHBUTTON 980 74 80 20
  VALUE "BG Bot Down"
  ALIGNCENTER
  ONACTIVATE "BG_Bot_Down"
  END
  WID_BUTTON_Sp_Bot_Up PUSHBUTTON 890 51 80 20
  VALUE "Sp Bot Up"
  ALIGNCENTER
  ONACTIVATE "Sp_Bot_Up"
  END
  WID_BUTTON_Sp_Bot_Down PUSHBUTTON 980 51 80 20
  VALUE "Sp Bot Down"
  ALIGNCENTER
  ONACTIVATE "Sp_Bot_Down"
  END
  WID_BUTTON_All_Up PUSHBUTTON 890 104 80 20
  VALUE "All Up"
  ALIGNCENTER
  ONACTIVATE "All_Up"
  END
  WID_BUTTON_All_Down PUSHBUTTON 980 104 80 20
  VALUE "All Down"
  ALIGNCENTER
  ONACTIVATE "All_Down"
  END
  WID_SLIDER_RawFrameNumber_Spectral SLIDER 515 10 150 46
  CAPTION "Raw Frame Number"
  ONVALUECHANGED "OnRawFrameNumber_Spectral"
  END
  WID_BUTTON_Save_All_Spectra PUSHBUTTON 10 100 125 25
  VALUE "Save All Spectra"
  ALIGNCENTER
  ONACTIVATE "On_Save_All_Spectra"
  END
  WID_BUTTON_Plot_All_Spectra PUSHBUTTON 280 100 225 25
  VALUE "Plot All Spectra (deected peaks)"
  ALIGNCENTER
  ONACTIVATE "On_Plot_All_Spectra"
  END
  WID_SLIDER_RawPeak_Index_Spectral SLIDER 515 69 150 46
  CAPTION "Detected Peak #"
  ONVALUECHANGED "OnRawPeakIndex_Spectral"
  END
END
