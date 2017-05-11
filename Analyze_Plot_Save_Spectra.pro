; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	04/20/2015 10:22.28
; 
pro WID_BASE_Analyze_Plot_Save_Spectra_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Analyze_Plot_Save_Spectra'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Spectrum_Calc'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        Change_Spectrum_Calc, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SaveSpectrum'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSaveSpectrum, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancel, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TABLE_BG_Subtr_Params'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Do_Edit_BG_Subtraction_Params, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GB_Top_Up'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        BG_Top_Up, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GB_Top_Down'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        BG_Top_Down, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Sp_Top_Up'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Sp_Top_Up, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Sp_Top_Down'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Sp_Top_Down, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_BG_Bot_Up'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        BG_Bot_Up, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_BG_Bot_Down'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        BG_Bot_Down, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Sp_Bot_Up'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Sp_Bot_Up, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Sp_Bot_Down'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Sp_Bot_Down, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_All_Up'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        All_Up, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_All_Down'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        All_Down, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_RawFrameNumber_Spectral'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        OnRawFrameNumber_Spectral, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_All_Spectra'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Save_All_Spectra, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_All_Spectra'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Plot_All_Spectra, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_RawPeak_Index_Spectral'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        OnRawPeakIndex_Spectral, Event
    end
    else:
  endcase

end

pro WID_BASE_Analyze_Plot_Save_Spectra, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Analyze_Plot_Save_Spectra_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Analyze_Plot_Save_Spectra = Widget_Base(  $
      GROUP_LEADER=wGroup, UNAME='WID_BASE_Analyze_Plot_Save_Spectra'  $
      ,XOFFSET=5 ,YOFFSET=5 ,SCR_XSIZE=1100 ,SCR_YSIZE=1220  $
      ,NOTIFY_REALIZE='Initialize_Analyze_Plot_Save_Spectra'  $
      ,KILL_NOTIFY='Set_def_window' ,TITLE='Analyze, Plot, and Save'+ $
      ' Spectra' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_DROPLIST_Spectrum_Calc =  $
      Widget_Droplist(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_DROPLIST_Spectrum_Calc' ,XOFFSET=280 ,YOFFSET=70  $
      ,SCR_XSIZE=225 ,SCR_YSIZE=30 ,TITLE='Calculate Spectrum'  $
      ,VALUE=[ 'Frame, with BG Subtr.', 'Frame, no BG Subtr.',  $
      'Total, with BG Subtr.', 'Total, no BG Subtr.' ])

  
  WID_DRAW_Spectra = Widget_Draw(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_DRAW_Spectra' ,XOFFSET=6 ,YOFFSET=128  $
      ,SCR_XSIZE=1024 ,SCR_YSIZE=1024)

  
  WID_BUTTON_SaveSpectrum =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_SaveSpectrum' ,XOFFSET=10 ,YOFFSET=70  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Save'+ $
      ' Spectrum')

  
  WID_BUTTON_Cancel =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_Cancel' ,XOFFSET=145 ,YOFFSET=70  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Cancel')

  
  WID_TABLE_BG_Subtr_Params =  $
      Widget_Table(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_TABLE_BG_Subtr_Params' ,XOFFSET=680 ,SCR_XSIZE=200  $
      ,SCR_YSIZE=125 ,/EDITABLE ,/RESIZEABLE_COLUMNS ,ROW_LABELS=[  $
      'BackGround Top', 'Spectrum Top', 'Spectrum Bot', 'BackGround'+ $
      ' Bot', 'Increment' ] ,XSIZE=1 ,YSIZE=5)

  
  WID_LABEL_Sopectrum_Filename =  $
      Widget_Label(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_LABEL_Sopectrum_Filename' ,XOFFSET=8 ,YOFFSET=13  $
      ,/ALIGN_LEFT ,VALUE='Filename:')

  
  WID_TEXT_SpFilename =  $
      Widget_Text(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_TEXT_SpFilename' ,XOFFSET=4 ,YOFFSET=28  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=36 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BUTTON_GB_Top_Up =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_GB_Top_Up' ,XOFFSET=890 ,YOFFSET=5  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=20 ,/ALIGN_CENTER ,VALUE='BG Top Up')

  
  WID_BUTTON_GB_Top_Down =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_GB_Top_Down' ,XOFFSET=980 ,YOFFSET=5  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=20 ,/ALIGN_CENTER ,VALUE='BG Top'+ $
      ' Down')

  
  WID_BUTTON_Sp_Top_Up =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_Sp_Top_Up' ,XOFFSET=890 ,YOFFSET=28  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=20 ,/ALIGN_CENTER ,VALUE='Sp Top Up')

  
  WID_BUTTON_Sp_Top_Down =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_Sp_Top_Down' ,XOFFSET=980 ,YOFFSET=28  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=20 ,/ALIGN_CENTER ,VALUE='Sp Top'+ $
      ' Down')

  
  WID_BUTTON_BG_Bot_Up =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_BG_Bot_Up' ,XOFFSET=890 ,YOFFSET=74  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=20 ,/ALIGN_CENTER ,VALUE='BG Bot Up')

  
  WID_BUTTON_BG_Bot_Down =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_BG_Bot_Down' ,XOFFSET=980 ,YOFFSET=74  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=20 ,/ALIGN_CENTER ,VALUE='BG Bot'+ $
      ' Down')

  
  WID_BUTTON_Sp_Bot_Up =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_Sp_Bot_Up' ,XOFFSET=890 ,YOFFSET=51  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=20 ,/ALIGN_CENTER ,VALUE='Sp Bot Up')

  
  WID_BUTTON_Sp_Bot_Down =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_Sp_Bot_Down' ,XOFFSET=980 ,YOFFSET=51  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=20 ,/ALIGN_CENTER ,VALUE='Sp Bot'+ $
      ' Down')

  
  WID_BUTTON_All_Up =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_All_Up' ,XOFFSET=890 ,YOFFSET=104  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=20 ,/ALIGN_CENTER ,VALUE='All Up')

  
  WID_BUTTON_All_Down =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_All_Down' ,XOFFSET=980 ,YOFFSET=104  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=20 ,/ALIGN_CENTER ,VALUE='All Down')

  
  WID_SLIDER_RawFrameNumber_Spectral =  $
      Widget_Slider(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_SLIDER_RawFrameNumber_Spectral' ,XOFFSET=515  $
      ,YOFFSET=10 ,SCR_XSIZE=150 ,SCR_YSIZE=46 ,TITLE='Raw Frame'+ $
      ' Number')

  
  WID_BUTTON_Save_All_Spectra =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_Save_All_Spectra' ,XOFFSET=10 ,YOFFSET=100  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Save All'+ $
      ' Spectra')

  
  WID_BUTTON_Plot_All_Spectra =  $
      Widget_Button(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_BUTTON_Plot_All_Spectra' ,XOFFSET=280 ,YOFFSET=100  $
      ,SCR_XSIZE=225 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Plot All'+ $
      ' Spectra (deected peaks)')

  
  WID_SLIDER_RawPeak_Index_Spectral =  $
      Widget_Slider(WID_BASE_Analyze_Plot_Save_Spectra,  $
      UNAME='WID_SLIDER_RawPeak_Index_Spectral' ,XOFFSET=515  $
      ,YOFFSET=69 ,SCR_XSIZE=150 ,SCR_YSIZE=46 ,TITLE='Detected Peak'+ $
      ' #')

  Widget_Control, /REALIZE, WID_BASE_Analyze_Plot_Save_Spectra

  XManager, 'WID_BASE_Analyze_Plot_Save_Spectra', WID_BASE_Analyze_Plot_Save_Spectra, /NO_BLOCK  ,CLEANUP='Set_def_window'  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Analyze_Plot_Save_Spectra, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Analyze_Plot_Save_Spectra, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
