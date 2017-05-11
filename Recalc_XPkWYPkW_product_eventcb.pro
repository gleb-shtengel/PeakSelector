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
if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

WidOffsetValueID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_offset_value')
widget_control,WidOffsetValueID,GET_VALUE = Offset_value_txt
PkWidth_offset=float(Offset_value_txt[0])

CGroupParams[12,*]=((CGroupParams[4,*]-PkWidth_offset) > 0.0)*((CGroupParams[5,*]-PkWidth_offset)>0.0)

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
