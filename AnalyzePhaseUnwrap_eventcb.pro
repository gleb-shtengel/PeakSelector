;
; Empty stub procedure used for autoloading.
;
pro AnalyzePhaseUnwrap_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_AnalyzePhaseUnwrap, wWidget
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
SAV_filename_label_ID = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,SAV_filename_label_ID,get_value=SAV_filename
PhaseUnwrapAnal_File=AddExtension(SAV_filename,'_phase_unwr_anal.txt')

Results_filename_label_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_ResultsFilename')
widget_control,Results_filename_label_ID,set_value=PhaseUnwrapAnal_File

WID_number_frames_per_step_ID = Widget_Info(wWidget, find_by_uname='WID_number_frames_per_step')
widget_control,WID_number_frames_per_step_ID,SET_VALUE = '99'

WID_nm_per_step_ID = Widget_Info(wWidget, find_by_uname='WID_nm_per_step')
widget_control,WID_nm_per_step_ID,SET_VALUE = '8.9'

end
;
;-----------------------------------------------------------------
;
pro On_Analyze_PhaseUnwrap_Start, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WID_Base0_Peakselector, ID:TopID, TOP:TopID, HANDLER:TopID }

X_ind=min(where(RowNames eq 'X Position'))
Y_ind=min(where(RowNames eq 'Y Position'))
GrX_ind=min(where(RowNames eq 'Group X Position'))
GrY_ind=min(where(RowNames eq 'Group Y Position'))
Z_ind=min(where(RowNames eq 'Z Position'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
UnwZ_ind=min(where(RowNames eq 'Unwrapped Z'))
UnwGrZ_ind=min(where(RowNames eq 'Unwrapped Group Z'))

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

WID_number_frames_per_step_ID = Widget_Info(Event.top, find_by_uname='WID_number_frames_per_step')
widget_control,WID_number_frames_per_step_ID,GET_VALUE = number_frames_per_step_txt
number_frames_per_step=float(number_frames_per_step_txt[0])

WID_nm_per_step_ID = Widget_Info(Event.top, find_by_uname='WID_nm_per_step')
widget_control,WID_nm_per_step_ID,GET_VALUE = nm_per_step_txt
nm_per_step=float(nm_per_step_txt[0])

FirstFrame=ParamLimits[9,0]
LastFrame=ParamLimits[9,1]
StepNum=(LastFrame-FirstFrame+1)/number_frames_per_step
center_lobe_delta=100.0  ;	 nm

Results=fltarr(10,stepnum)

for ip=0,(StepNum-1) do begin

	Fr_start=FirstFrame+ip*number_frames_per_step
	Fr_stop=Fr_start+number_frames_per_step-1
	indecis=where((CGroupParams[9,*] ge Fr_start) and (CGroupParams[9,*] le Fr_stop) and filter, cnt)
	if cnt ge 1 then begin
		Results[0,ip]=ip*nm_per_step
		Results[1,ip]=cnt

		coarse_mean_Z=mean(CGroupParams[UnwZ_ind,indecis])

		center_lobe_indecis = where((CGroupParams[UnwZ_ind,indecis] ge (coarse_mean_Z-center_lobe_delta)) $
			and (CGroupParams[UnwZ_ind,indecis] le (coarse_mean_Z+center_lobe_delta)),center_cnt)

		if (center_cnt gt 1) then begin
			low_ghost_indecis = where(CGroupParams[UnwZ_ind,indecis] lt (coarse_mean_Z-center_lobe_delta),low_ghost_cnt)
			high_ghost_indecis = where(CGroupParams[UnwZ_ind,indecis] gt (coarse_mean_Z+center_lobe_delta),high_ghost_cnt)
			Results[2,ip] = low_ghost_cnt
			Results[3,ip] = center_cnt
			Results[4,ip] = high_ghost_cnt
			Results[5,ip] = 100.0 * (low_ghost_cnt + high_ghost_cnt) / cnt
			Results[6,ip] = mean(CGroupParams[UnwZ_ind,indecis[center_lobe_indecis]])
			Results[7,ip] = stdev(CGroupParams[UnwZ_ind,indecis[center_lobe_indecis]])
		endif
		Results[8,ip] = mean(CGroupParams[16,indecis]) * nm_per_pixel
		Results[9,ip] = mean(CGroupParams[17,indecis]) * nm_per_pixel
		print,Results[*,ip]

	endif
endfor

Results_filename_label_ID = Widget_Info(event.top, find_by_uname='WID_TEXT_ResultsFilename')
widget_control,Results_filename_label_ID,get_value=PhaseUnwrapAnal_File
PhaseUnwrapAnal_File_BMP=AddExtension(PhaseUnwrapAnal_File,'.bmp')

Title_String='Z (nm)	TotalCount	Left Ghost Count	Center Lobe Count	Right Ghost Count	Error Rate (%)	Measured Z (nm)	Zstd (nm)	Xstd (nm)	Ystd (nm)'
openw,1,PhaseUnwrapAnal_File,width=1024
printf,1,Title_String
printf,1,Results,FORMAT='(8(E13.5,"'+string(9B)+'"),E13.5)'
close,1
widget_control,event.top,/destroy

!p.background=0
!p.multi=[0,1,4,0,0]

zz = Results[0,*]
zm = Results[6,*]
Err_rate = Results[5,*]
SigX = Results[8,*]
SigY = Results[9,*]
SigZ = Results[7,*]
Tot_ER=100.0*Total(Results[2,*]+Results[4,*])/Total(Results[1,*])

dx=0.02
ch_sz=2.5
plot,zz,zm,xtitle='Sample position (nm)',ytitle='Measured Z (nm)',position=[0.07,(0.755+dx),(1.0-dx),(1.0-dx)],charsize=ch_sz,ystyle=1
zm_fit=poly_fit(zz,zm,1)
str_out='Linear Fit Slope:'+strtrim(zm_fit[1],2)+'     Offset:'+strtrim(zm_fit[0],2)
xyouts,0.15,0.96,str_out,CHARSIZE=1.5,/NORMAL
zm_lin=poly(zz,zm_fit)
oplot,zz,zm_lin,col=200,THICK=2.0
oplot,zz,zm
oplot,zz,zm,psym=4

zm_lin_delta=zm_lin-zm
plot,zz,zm_lin_delta,xtitle='Sample position (nm)',ytitle='Deviation from Linear (nm)',position=[0.07,(0.51+dx),(1.0-dx),(0.755-dx)],charsize=ch_sz,ystyle=1
oplot,zz,zm_lin_delta,psym=4
str_out='St.Deviation from Linear Fit (nm):   '+strtrim(stdev(zm_lin_delta),2)
xyouts,0.15,0.71,str_out,CHARSIZE=1.5,/NORMAL

plot,zz,Err_rate,xtitle='Sample position (nm)',ytitle='Error Rate (%)',position=[0.07,(0.265+dx),(1.0-dx),(0.51-dx)],charsize=ch_sz,yrange=[0,30],ystyle=1
oplot,zz,Err_rate,psym=4
str_out='Total Error Rate (%):   '+strtrim(Tot_ER,2)
xyouts,0.15,0.465,str_out,CHARSIZE=1.5,/NORMAL

plot,zz,SigX,xtitle='Sample position (nm)',ytitle='Localization Sigma (nm)',position=[0.07,0.02+dx,(1.0-dx),(0.265-dx)],charsize=ch_sz,ystyle=1
oplot,zz,SigX,psym=4
oplot,zz,SigY,col=100
oplot,zz,SigY,psym=5,col=100
oplot,zz,SigZ,col=200
oplot,zz,SigZ,psym=6,col=200

!p.multi=[0,0,0,0,0]

presentimage=tvrd(true=1)
write_bmp,PhaseUnwrapAnal_File_BMP,presentimage,/rgb

end
