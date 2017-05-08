; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	02/21/2012 20:01.12
; 
pro WID_BASE_Transform_2D_and_Spectrum_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Transform_2D_and_Spectrum'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_stop'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnStopSpectralProcessing, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Process'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Start_Process_2DSpectrum, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CreateSingleCal'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Create_Single_Sp_Cal, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickCalFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickSpCalFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SP_Cal_Table'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Do_Edit_Sp_Cal, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SaveCalFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSaveSpCalFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_RemoveSingleCal_0'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Remove_Single_Sp_Cal, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TABLE_WlCal_FrStart_FrStop'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Do_Edit_Cal_Frames, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_WlShiftSingleCal_1'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        WlShift_Single_Sp_Cal, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Transform'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        StartTransform, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CancelReExtract_0'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancelSave, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_Pick_XY_File'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickXYTxtFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_Pick_SP_File'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickSPTxtFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PlotDistributions'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Plot_Spectral_Weigths_Distributions, Event
    end
    else:
  endcase

end

pro WID_BASE_Transform_2D_and_Spectrum, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Transform_Save_Process_2D_and_Spectrum_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Transform_2D_and_Spectrum = Widget_Base(  $
      GROUP_LEADER=wGroup, UNAME='WID_BASE_Transform_2D_and_Spectrum'  $
      ,XOFFSET=5 ,YOFFSET=5 ,SCR_XSIZE=998 ,SCR_YSIZE=743  $
      ,NOTIFY_REALIZE='Initialize_Transform_2D_and_spectrum'  $
      ,TITLE='Transform Data and process 2D and Spectrum analysis'  $
      ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_TEXT_XY_Filename =  $
      Widget_Text(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_TEXT_XY_Filename' ,XOFFSET=5 ,YOFFSET=20  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_TEXT_TrnSpFilename =  $
      Widget_Text(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_TEXT_TrnSpFilename' ,XOFFSET=5 ,YOFFSET=70  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_LABEL_XY_File =  $
      Widget_Label(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_LABEL_XY_File' ,XOFFSET=5 ,YOFFSET=5 ,/ALIGN_LEFT  $
      ,VALUE='X-Y Localization File:')

  
  WID_LABEL_TrnSpFile =  $
      Widget_Label(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_LABEL_TrnSpFile' ,XOFFSET=5 ,YOFFSET=55 ,/ALIGN_LEFT  $
      ,VALUE='Transformed Spectrum File:')

  
  WID_BUTTON_stop = Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BUTTON_stop' ,XOFFSET=660 ,YOFFSET=133  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Stop')

  
  WID_BUTTON_Process =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BUTTON_Process' ,XOFFSET=820 ,YOFFSET=12  $
      ,SCR_XSIZE=160 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Process'+ $
      ' Spectra')

  
  WID_DROPLIST_FitDisplayType_Spectrum =  $
      Widget_Droplist(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_DROPLIST_FitDisplayType_Spectrum' ,XOFFSET=637  $
      ,YOFFSET=81 ,SCR_XSIZE=225 ,SCR_YSIZE=30 ,TITLE='Fit-Display'+ $
      ' Level' ,VALUE=[ 'No Display', 'Some Frames/Peaks', 'All'+ $
      ' Frames/Peaks ', 'Cluster - No Display' ])

  
  WID_BUTTON_CreateSingleCal =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BUTTON_CreateSingleCal' ,XOFFSET=245 ,YOFFSET=210  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Create'+ $
      ' Single Calibration')

  
  WID_TEXT_SpCalFilename =  $
      Widget_Text(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_TEXT_SpCalFilename' ,XOFFSET=5 ,YOFFSET=150  $
      ,SCR_XSIZE=600 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_LABEL_TrnSpFile_0 =  $
      Widget_Label(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_LABEL_TrnSpFile_0' ,XOFFSET=7 ,YOFFSET=130  $
      ,/ALIGN_LEFT ,VALUE='Calibration File:')

  
  WID_BUTTON_PickCalFile =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BUTTON_PickCalFile' ,XOFFSET=460 ,YOFFSET=115  $
      ,SCR_XSIZE=180 ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Pick'+ $
      ' Calibration File')

  
  WID_SP_Cal_Table = Widget_Table(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_SP_Cal_Table' ,XOFFSET=10 ,YOFFSET=280  $
      ,SCR_XSIZE=432 ,SCR_YSIZE=430 ,/EDITABLE ,COLUMN_LABELS=[ 'Sp'+ $
      ' #1', 'Sp #2', 'Sp #3', 'Sp #4', 'Sp #5' ] ,XSIZE=5  $
      ,YSIZE=200)

  
  WID_DRAW_Spectra = Widget_Draw(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_DRAW_Spectra' ,XOFFSET=460 ,YOFFSET=280  $
      ,SCR_XSIZE=440 ,SCR_YSIZE=430)

  
  WID_BUTTON_SaveCalFile =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BUTTON_SaveCalFile' ,XOFFSET=460 ,YOFFSET=185  $
      ,SCR_XSIZE=180 ,SCR_YSIZE=32 ,/ALIGN_CENTER ,VALUE='Save'+ $
      ' Calibration File')

  
  WID_BUTTON_RemoveSingleCal_0 =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BUTTON_RemoveSingleCal_0' ,XOFFSET=5 ,YOFFSET=210  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Remove'+ $
      ' Single Calibration')

  
  WID_TEXT_CalSpNum = Widget_Text(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_TEXT_CalSpNum' ,XOFFSET=210 ,YOFFSET=210  $
      ,SCR_XSIZE=30 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '0' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_TABLE_WlCal_FrStart_FrStop =  $
      Widget_Table(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_TABLE_WlCal_FrStart_FrStop' ,XOFFSET=660  $
      ,YOFFSET=199 ,SCR_XSIZE=154 ,SCR_YSIZE=77 ,/EDITABLE  $
      ,ROW_LABELS=[ 'Start Frame', 'Stop Frame' ] ,XSIZE=1 ,YSIZE=2)

  
  WID_BUTTON_WlShiftSingleCal_1 =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BUTTON_WlShiftSingleCal_1' ,XOFFSET=125 ,YOFFSET=245  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Shift'+ $
      ' Single Calibration (pix)')

  
  WID_TEXT_WlShift = Widget_Text(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_TEXT_WlShift' ,XOFFSET=330 ,YOFFSET=245  $
      ,SCR_XSIZE=30 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '0' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_BASE_0 = Widget_Base(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BASE_0' ,XOFFSET=454 ,YOFFSET=236 ,TITLE='IDL'  $
      ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_TransformFirstForCal = Widget_Button(WID_BASE_0,  $
      UNAME='WID_BUTTON_TransformFirstForCal' ,/ALIGN_LEFT  $
      ,VALUE='Transform First (For Single Cal)')

  
  WID_BUTTON_Transform =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BUTTON_Transform' ,XOFFSET=650 ,YOFFSET=12  $
      ,SCR_XSIZE=160 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Confirm +'+ $
      ' Transform')

  
  WID_BUTTON_CancelReExtract_0 =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BUTTON_CancelReExtract_0' ,XOFFSET=825 ,YOFFSET=132  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Cancel')

  
  WID_BTTN_Pick_XY_File =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BTTN_Pick_XY_File' ,XOFFSET=520 ,YOFFSET=20  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick XY'+ $
      ' File')

  
  WID_BTTN_Pick_SP_File =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BTTN_Pick_SP_File' ,XOFFSET=520 ,YOFFSET=70  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick SP'+ $
      ' File')

  
  WID_BUTTON_PlotDistributions =  $
      Widget_Button(WID_BASE_Transform_2D_and_Spectrum,  $
      UNAME='WID_BUTTON_PlotDistributions' ,XOFFSET=825 ,YOFFSET=207  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=55 ,/ALIGN_CENTER ,VALUE='Plot'+ $
      ' Distributions')

  Widget_Control, /REALIZE, WID_BASE_Transform_2D_and_Spectrum

  XManager, 'WID_BASE_Transform_2D_and_Spectrum', WID_BASE_Transform_2D_and_Spectrum, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Transform_Save_Process_2D_and_Spectrum, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Transform_2D_and_Spectrum, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
