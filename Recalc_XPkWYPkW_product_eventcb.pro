;
; Empty stub procedure used for autoloading.
;
pro Recalc_XpkWYPkw_product_eventcb
end
;
;-----------------------------------------------------------------
;
pro OnCancel, Event; cancels and closes the menu widget
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnRecalculate, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Offset, PkWidth_offset
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

Xwid_ind = min(where(RowNames eq 'X Peak Width'))                        ; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))                        ; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))                    ; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)

WidOffsetValueID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_offset_value')
widget_control,WidOffsetValueID,GET_VALUE = Offset_value_txt
print,WidOffsetValueID, Offset_value_txt
PkWidth_offset=float(Offset_value_txt[0])

CGroupParams[Par12_ind,*]=((CGroupParams[Xwid_ind,*]-PkWidth_offset) > 0.0)*((CGroupParams[Ywid_ind,*]-PkWidth_offset)>0.0)

if bridge_exists then begin
	print,'Reloading the Bridge Array'
	CATCH, Error_status
	CGroupParams_bridge = SHMVAR(shmName_data)
	CGroupParams_bridge[Par12_ind,*] = CGroupParams[Par12_ind,*]
	IF Error_status NE 0 THEN BEGIN
		PRINT, 'Bridge Refresh Error: Drift Correction:',!ERROR_STATE.MSG
		PRINT, 'System Error Message:',!ERROR_STATE.SYS_MSG
		bridge_exists = 0
		SHMUnmap, shmName_data
		SHMUnmap, shmName_filter
		for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
		CATCH,/CANCEL
	ENDIF
	CATCH,/CANCEL
	print,'Finished Reloading the Bridge Array'
endif


widget_control,event.top,/destroy

end
;
;-----------------------------------------------------------------
;
pro Initialize_Recalculate_Menu, wWidget
common Offset, PkWidth_offset
if (size(PkWidth_offset))[2] eq 0 then PkWidth_offset=0.00
WidOffsetValueID = Widget_Info(wWidget, find_by_uname='WID_TEXT_offset_value')
Offset_value_txt=string(PkWidth_offset,FORMAT='(F6.2)')
widget_control,WidOffsetValueID,SET_VALUE = Offset_value_txt
end
