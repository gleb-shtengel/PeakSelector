;
; IDL Event Callback Procedures
; Remove_XYZ_Tilt_eventcb
;
; Generated on:	04/29/2008 15:44.56
;
;-----------------------------------------------------------------
; Notify Realize Callback Procedure.
; Argument:
;   wWidget - ID number of specific widget.
;
;
;   Retrieve the IDs of other widgets in the widget hierarchy using
;       id=widget_info(Event.top, FIND_BY_UNAME=name)

;-----------------------------------------------------------------
pro Initialize_XYZ_Fid_Wid, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  XYZ_Fid_Params,  XYZ_Fid_Pnts,  XYZ_Fid_File
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

if n_elements(XYZ_Fid_Pnts) eq 0 then XYZ_Fid_Pnts=dblarr(3,3)

table_id=widget_info(wWidget,FIND_BY_UNAME='WID_TABLE_0')
widget_control,table_id,set_value=(XYZ_Fid_Pnts), use_table_select=[0,0,2,2]

existing_ind=where((RawFilenames ne ''),nlabels)
if (existing_ind[0] eq -1) then return

data_file = RawFilenames[max(existing_ind)]
if data_file eq '' then return

SAV_filename_label_ID = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,SAV_filename_label_ID,get_value=SAV_filename

;XYZ_Fid_File=AddExtension(data_file,'.xyz')
XYZ_Fid_File=AddExtension(SAV_filename,'.xyz')

XYZ_File_WidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_XYZ_Filename')
widget_control,XYZ_File_WidID,SET_VALUE = XYZ_Fid_File

end
;
;-----------------------------------------------------------------
;
;
;-----------------------------------------------------------------
;
pro DoInsert_XYZ_Anchor, Event
common  XYZ_Fid_Params,  XYZ_Fid_Pnts,  XYZ_Fid_File
widget_control,event.id,get_value=thevalue
XYZ_Fid_Pnts[event.x,event.y]=thevalue[event.x,event.y]
widget_control,event.id,set_value=(XYZ_Fid_Pnts[0:2,0:2]), use_table_select=[0,0,2,2]
end
;
;-----------------------------------------------------------------
;
pro Do_XYZ_Tilt_Transforms, Event

common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common  XYZ_Fid_Params, XYZ_Fid_Pnts,  XYZ_Fid_File
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

catch,error_status

xvector=XYZ_Fid_Pnts[0,0:2]
x_zero_ind = where(xvector eq 0, cnt0)
if cnt0 gt 0 then xvector[x_zero_ind] = 1e-6		; turn zero elements into slightly non-zero to avoid singularities
yvector=XYZ_Fid_Pnts[1,0:2]
y_zero_ind = where(yvector eq 0, cnt1)
if cnt1 gt 0 then yvector[y_zero_ind] = 1e-6
zvector=XYZ_Fid_Pnts[2,0:2]
z_zero_ind = where(zvector eq 0, cnt2)
if cnt2 gt 0 then zvector[z_zero_ind] = 1e-6
zmean=mean(zvector)

if (cnt0 gt 0) or (cnt0 gt 0) or (cnt0 gt 0) then begin
	print,'changed 0-elemets to be 1e-6'
	print,'using these fiducials:',[xvector,yvector,zvector]
endif

det_XYZ=DETERM([xvector,yvector,zvector],/DOUBLE)
uvector=transpose([1,1,1])
a_XYZ=DETERM([uvector,yvector,zvector])/det_XYZ
b_XYZ=DETERM([xvector,uvector,zvector])/det_XYZ
c_XYZ=DETERM([xvector,yvector,uvector])/det_XYZ
n_wr=n_elements(wind_range)

print,'calculated discriminant'

if error_status NE 0 then begin
	print,'Error in Do_XYZ_Tilt_Transforms: ',!ERROR_STATE.MSG
	print,'Error code: ',!ERROR_STATE.CODE
	catch,/cancel
endif else begin
catch,/cancel
print,'n_wr=',n_wr

	if n_wr le 1 then begin
		if n_wr eq 1 then begin
			if UnwZ_ind ge 0 then begin
				CGroupParams[Z_ind,*]=(zmean+Cgroupparams[Z_ind,*]-1/c_XYZ+Cgroupparams[X_ind,*]*a_XYZ/c_XYZ+Cgroupparams[Y_ind,*]*b_XYZ/c_XYZ+10.0*wind_range) mod wind_range
				CGroupParams[GrZ_ind,*]=(zmean+Cgroupparams[GrZ_ind,*]-1/c_XYZ+Cgroupparams[GrX_ind,*]*a_XYZ/c_XYZ+Cgroupparams[GrY_ind,*]*b_XYZ/c_XYZ+10.0*wind_range) mod wind_range
				CGroupParams[UnwZ_ind,*]=zmean+Cgroupparams[UnwZ_ind,*]-1/c_XYZ+Cgroupparams[X_ind,*]*a_XYZ/c_XYZ+Cgroupparams[Y_ind,*]*b_XYZ/c_XYZ
				CGroupParams[UnwGrZ_ind,*]=zmean+Cgroupparams[UnwGrZ_ind,*]-1/c_XYZ+Cgroupparams[GrX_ind,*]*a_XYZ/c_XYZ+Cgroupparams[GrY_ind,*]*b_XYZ/c_XYZ
			endif else begin
				CGroupParams[Z_ind,*]=(zmean+Cgroupparams[Z_ind,*]-1/c_XYZ+Cgroupparams[X_ind,*]*a_XYZ/c_XYZ+Cgroupparams[Y_ind,*]*b_XYZ/c_XYZ)
				CGroupParams[GrZ_ind,*]=(zmean+Cgroupparams[GrZ_ind,*]-1/c_XYZ+Cgroupparams[GrX_ind,*]*a_XYZ/c_XYZ+Cgroupparams[GrY_ind,*]*b_XYZ/c_XYZ)
			endelse
		endif
	endif else begin
		cgr26_min=min(CGroupParams[LabelSet_ind,*])
		for i_wr=0,(n_wr-1) do begin
			ind_this_lbl=where(CGroupParams[LabelSet_ind,*] eq (cgr26_min+i_wr))
			if UnwZ_ind ge 0 then begin
				CGroupParams[Z_ind,ind_this_lbl]=(zmean+Cgroupparams[Z_ind,ind_this_lbl]-1/c_XYZ+Cgroupparams[X_ind,ind_this_lbl]*a_XYZ/c_XYZ+Cgroupparams[Y_ind,ind_this_lbl]*b_XYZ/c_XYZ+10.0*wind_range[i_wr]) mod wind_range[i_wr]
				CGroupParams[GrZ_ind,ind_this_lbl]=(zmean+Cgroupparams[GrZ_ind,ind_this_lbl]-1/c_XYZ+Cgroupparams[GrX_ind,ind_this_lbl]*a_XYZ/c_XYZ+Cgroupparams[GrY_ind,ind_this_lbl]*b_XYZ/c_XYZ+10.0*wind_range[i_wr]) mod wind_range[i_wr]
				CGroupParams[UnwZ_ind,ind_this_lbl]=zmean+Cgroupparams[UnwZ_ind,ind_this_lbl]-1/c_XYZ+Cgroupparams[X_ind,ind_this_lbl]*a_XYZ/c_XYZ+Cgroupparams[Y_ind,ind_this_lbl]*b_XYZ/c_XYZ
				CGroupParams[UnwGrZ_ind,ind_this_lbl]=zmean+Cgroupparams[UnwGrZ_ind,ind_this_lbl]-1/c_XYZ+Cgroupparams[GrX_ind,ind_this_lbl]*a_XYZ/c_XYZ+Cgroupparams[GrY_ind,ind_this_lbl]*b_XYZ/c_XYZ
			endif else begin
				CGroupParams[Z_ind,ind_this_lbl]=(zmean+Cgroupparams[Z_ind,ind_this_lbl]-1/c_XYZ+Cgroupparams[X_ind,ind_this_lbl]*a_XYZ/c_XYZ+Cgroupparams[Y_ind,ind_this_lbl]*b_XYZ/c_XYZ)
				CGroupParams[GrZ_ind,ind_this_lbl]=(zmean+Cgroupparams[GrZ_ind,ind_this_lbl]-1/c_XYZ+Cgroupparams[GrX_ind,ind_this_lbl]*a_XYZ/c_XYZ+Cgroupparams[GrY_ind,ind_this_lbl]*b_XYZ/c_XYZ)
			endelse
		endfor
	endelse
	Save_XYZ_File, Event
	Clear_XYZ_Fiducials, Event
	widget_control,event.top,/destroy
	ReloadParamlists, Event, [Z_ind,GrZ_ind,UnwZ_ind,UnwZErr_ind,UnwGrZ_ind,UnwGrZErr_ind]

	if bridge_exists then begin
		print,'Reloading the Bridge Array'
		CATCH, Error_status
		CGroupParams_bridge = SHMVAR(shmName_data)
		CGroupParams_bridge[UnwZ_ind,*] = CGroupParams[UnwZ_ind,*]
		CGroupParams_bridge[Z_ind,*] = CGroupParams[Z_ind,*]
		CGroupParams_bridge[UnwGrZ_ind,*] = CGroupParams[UnwGrZ_ind,*]
		CGroupParams_bridge[UnwZ_ind,*] = CGroupParams[UnwZ_ind,*]
		IF Error_status NE 0 THEN BEGIN
			PRINT, 'Bridge Refresh Error: Drift Correction:',!ERROR_STATE.MSG
			bridge_exists = 0
			SHMUnmap, shmName_data
			SHMUnmap, shmName_filter
			for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
			PRINT, 'Starting: Error:',!ERROR_STATE.MSG
			CATCH,/CANCEL
		ENDIF
		CATCH,/CANCEL
		print,'Finished Reloading the Bridge Array'
	endif
endelse

end
;
;-----------------------------------------------------------------
;
pro OnButton_XYZ_AddFiducial, Event
common  XYZ_Fid_Params,  XYZ_Fid_Pnts,  XYZ_Fid_File
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

filter0=filter
filter = filter and (CGroupParams[LabelSet_ind,*] le 1)
indecis=where(filter eq 1)
if (size(indecis))[0] eq 0 then return
xposition=total(CGroupParams[X_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
yposition=total(CGroupParams[Y_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
zposition=total(CGroupParams[Z_ind,indecis]*CGroupParams[Nph_ind,indecis])/total(CGroupParams[Nph_ind,indecis])
XYZ_Fid_Ind=min(where(XYZ_Fid_Pnts[0,*] eq 0))

if XYZ_Fid_Ind eq -1 then XYZ_Fid_Ind=2
xpr=XYZ_Fid_Pnts[0,((XYZ_Fid_Ind-1)>0)]
ypr=XYZ_Fid_Pnts[1,((XYZ_Fid_Ind-1)>0)]
zpr=XYZ_Fid_Pnts[2,((XYZ_Fid_Ind-1)>0)]
fdist=sqrt((xpr-xposition)^2+(ypr-yposition)^2+(zpr-zposition)^2/(140)^2)
if fdist gt 0.1 then begin
	XYZ_Fid_Pnts[0:2,XYZ_Fid_Ind]=[xposition,yposition,zposition]
	table_id=widget_info(event.top,FIND_BY_UNAME='WID_TABLE_0')
	widget_control,table_id,set_value=([xposition,yposition,zposition]), use_table_select=[0,XYZ_Fid_Ind,2,XYZ_Fid_Ind]
endif
filter=filter0
end
;
;-----------------------------------------------------------------
;
pro Clear_XYZ_Fiducials, Event
common  XYZ_Fid_Params,  XYZ_Fid_Pnts,  XYZ_Fid_File
XYZ_Fid_Pnts=fltarr(3,3)
table_id=widget_info(event.top,FIND_BY_UNAME='WID_TABLE_0')
widget_control,table_id,set_value=(XYZ_Fid_Pnts), use_table_select=[0,0,2,2]
end
;
;-----------------------------------------------------------------
;
pro OnPick_XYZ_File, Event
common  XYZ_Fid_Params,  XYZ_Fid_Pnts,  XYZ_Fid_File
XYZ_Fid_File = Dialog_Pickfile(/read,get_path=fpath,filter=['*.xyz'],title='Select *.xyz file to open')
if XYZ_Fid_File ne '' then cd,fpath
XYZ_File_WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_XYZ_Filename')
widget_control,XYZ_File_WidID,SET_VALUE = XYZ_Fid_File
end
;
;-----------------------------------------------------------------
;
pro Load_XYZ_File, Event
common  XYZ_Fid_Params,  XYZ_Fid_Pnts,  XYZ_Fid_File
XYZ_File_WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_XYZ_Filename')
widget_control,XYZ_File_WidID,GET_VALUE = XYZ_Fid_File
XYZ_File_Info=file_info(XYZ_Fid_File)
if ~XYZ_File_Info.exists then return
openr,1,XYZ_Fid_File
readf,1,XYZ_Fid_Pnts
close,1
table_id=widget_info(event.top,FIND_BY_UNAME='WID_TABLE_0')
widget_control,table_id,set_value=(XYZ_Fid_Pnts), use_table_select=[0,0,2,2]
end
;
;-----------------------------------------------------------------
;
pro Save_XYZ_File, Event
common  XYZ_Fid_Params,  XYZ_Fid_Pnts,  XYZ_Fid_File
XYZ_File_WidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_XYZ_Filename')
widget_control,XYZ_File_WidID,GET_VALUE = XYZ_Fid_File
if XYZ_Fid_File eq '' then return
openw,1,XYZ_Fid_File,width=512
printf,1,XYZ_Fid_Pnts
close,1
end
;
; Empty stub procedure used for autoloading.
;
pro Remove_XYZ_Tilt_eventcb
end
