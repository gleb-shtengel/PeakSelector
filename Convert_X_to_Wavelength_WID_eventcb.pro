;
;-----------------------------------------------------------------
;
pro Initialize_Convert_X_to_Wavelength, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	OnCancel      ; if data not loaded return
endif

WidOffsetValueID = Widget_Info(wWidget, find_by_uname='WID_TEXT_offset_value')
if (size(sp_offset))[2] eq 0 then sp_offset=500.00
Offset_value_txt=string(sp_offset,FORMAT='(F6.2)')
widget_control,WidOffsetValueID,SET_VALUE = Offset_value_txt

WidDispersionValueID = Widget_Info(wWidget, find_by_uname='WID_TEXT_dispersion_value')
if (size(sp_dispersion))[2] eq 0 then sp_dispersion=1.024
Dispersion_value_txt=string(sp_dispersion,FORMAT='(F6.3)')
widget_control,WidDispersionValueID,SET_VALUE = Dispersion_value_txt

end
;
;-----------------------------------------------------------------
;
pro OnRecalculate_Wavelength, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	OnCancel      ; if data not loaded return
endif
CGroupParams[12,*]=CGroupParams[2,*]*sp_dispersion+sp_offset
ReloadParamlists, Event, [12]
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnCancel, Event
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro Change_offset_text, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
WidOffsetValueID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_offset_value')
widget_control,WidOffsetValueID,GET_VALUE = Offset_value_txt
sp_offset=float(Offset_value_txt[0])
end
;
;-----------------------------------------------------------------
;
pro Change_dispersion_text, Event
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
WidDispersionValueID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_dispersion_value')
widget_control,WidDispersionValueID,GET_VALUE = Dispersion_value_txt
sp_dispersion=float(Dispersion_value_txt[0])
end
;
;-----------------------------------------------------------------
;
;
; Empty stub procedure used for autoloading.
;
pro Convert_X_to_Wavelength_WID_eventcb
end
;-----------------------------------------------------------------
