;-----------------------------------------------------------------
;
; IDL Event Callback Procedures
; _eventcb
;
; Generated on:	12/09/2005 13:11.16
;
;-----------------------------------------------------------------
;
;checks if the filename has the extension (caracters after dot in "extension" variable). If it does not, adds the extension.
function AddExtension, filename, extension		;
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
dot_pos=strpos(extension,'.',/REVERSE_OFFSET,/REVERSE_SEARCH)
short_ext=strmid(extension,dot_pos)
short_ext_pos=strpos(filename,short_ext,/REVERSE_OFFSET,/REVERSE_SEARCH)
ext_pos=strpos(filename,extension,/REVERSE_OFFSET,/REVERSE_SEARCH)
add_ext=(ext_pos lt 0)	?	extension	:	''
file_sep_pos=strpos(filename,sep,/REVERSE_OFFSET,/REVERSE_SEARCH)
file_dot_pos=strpos(filename,'.',/REVERSE_OFFSET,/REVERSE_SEARCH)
filename_without_ext = ((strlen(filename) - file_dot_pos) le 4) ? strmid(filename,0,file_dot_pos) :  filename
filename_with_ext = (ext_pos gt 0) ? filename : (filename_without_ext + add_ext)
return,filename_with_ext
end
;
;-----------------------------------------------------------------
;
function interpol_gs, V, X, U		; this function is the same as IDL's own interpol except
                                    ; that returns the edge values for the elements with U outside th range of X
F = interpol(V, X, U)
Xi =  min(X, Indi)
ind_low = where(U lt Xi, cnt_low)
if cnt_low gt 0 then F[ind_low] = V[Indi]
Xa =  max(X, Inda)
ind_high = where(U gt Xa, cnt_high)
if cnt_high gt 0 then F[ind_high] = V[Inda]
return,F
end
;
;-----------------------------------------------------------------
;
FUNCTION Phase_Unwrap, Y, T
	; Y is array to be phase-unwrapped for period T
	; That is if for any point Y[i] in array Y, the next point Y[i+1] differs by more than 0.5*T, that difference is reduced by integer number of T's
	;dY = Y - shift(Y,1)
	;dY[0]=0.0
	;dYunw = dY - round(dY/T)*T
	;Yunw = TOTAL(dYunw, /CUMULATIVE) + Y[0]
	Ym = median(Y)
	dY = Y - Ym
	Yunw = Y - round(dY/T)*T
	return, Yunw
end
;
;-----------------------------------------------------------------
;
function StripExtension, filename		; checks if the filename has the extension (caracters after dot in "extension" variable. If it does not, adds the extension.
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
file_sep_pos=strpos(filename,sep,/REVERSE_OFFSET,/REVERSE_SEARCH)
file_dot_pos=strpos(filename,'.',/REVERSE_OFFSET,/REVERSE_SEARCH)
filename_without_ext = (file_dot_pos gt file_sep_pos) ? strmid(filename,0,file_dot_pos) :  filename
return,filename_without_ext
end
;
;-----------------------------------------------------------------
;
FUNCTION UnwrapError, X, A
	y1 = exp(-0.5*(((X-A[1])/A[2])^2)) + exp(-0.5*(((X-A[1]-A[3])/A[2])^2)) + exp(-0.5*(((X-A[1]+A[3])/A[2])^2))
	y = A[0]*y1
	y2 = A[0] * exp(-0.5*(((X-A[1])/A[2])^2))*(X-A[1])/(A[2]^2) + $
		A[0] * exp(-0.5*(((X-A[1]-A[3])/A[2])^2))*(X-A[1]-A[3])/(A[2]^2) + $
		A[0] * exp(-0.5*(((X-A[1]+A[3])/A[2])^2))*(X-A[1]+A[3])/(A[2]^2)
	y3 = A[0] * exp(-0.5*(((X-A[1])/A[2])^2)) * 1.5 * (X-A[1])^2/(A[2]^3) + $
		A[0] * exp(-0.5*(((X-A[1]-A[3])/A[2])^2)) * 1.5 * (X-A[1]-A[3])^2/(A[2]^3) + $
		A[0] * exp(-0.5*(((X-A[1]+A[3])/A[2])^2)) * 1.5 * (X-A[1]+A[3])^2/(A[2]^3)
	y4 = A[0] * exp(-0.5*(((X-A[1]-A[3])/A[2])^2)) * (X-A[1]-A[3])/(A[2]^2) - $
		A[0] * exp(-0.5*(((X-A[1]+A[3])/A[2])^2)) * (X-A[1]+A[3])/(A[2]^2)
	return, [y,y1,y2,y3,y4]
end
;
;-----------------------------------------------------------------
;
;sudo cp /media/hesslab/PalmClusterTest/PeakSelector_V8.8/3D_Viewer/src/gpu_device_query/gpu_device_query.so /usr/local/itt/idl64/bin/bin.linux.x86_64/gpu_device_query.so
;sudo cp /media/hesslab/PalmClusterTest/PeakSelector_V8.8/3D_Viewer/src/gpu_device_query/gpu_device_query.dlm /usr/local/itt/idl64/bin/bin.linux.x86_64/gpu_device_query.dlm
;sudo cp /media/hesslab/PalmClusterTest/PeakSelector_V8.8/3D_Viewer/src/cu_poly_3d/cu_poly_3d.dlm /usr/local/itt/idl64/bin/bin.linux.x86_64/cu_poly_3d.dlm
;sudo cp /media/hesslab/PalmClusterTest/PeakSelector_V8.8/3D_Viewer/src/cu_poly_3d/cu_poly_3d.so /usr/local/itt/idl64/bin/bin.linux.x86_64/cu_poly_3d.so

;sudo cp /media/hesslab/PalmClusterTest/PeakSelector_V8.8/3D_Viewer/src/cu_gauss2d/particle_swarm.so /usr/local/itt/idl64/bin/bin.linux.x86_64/particle_swarm.so
;sudo cp /media/hesslab/PalmClusterTest/PeakSelector_V8.8/3D_Viewer/src/cu_gauss2d/particle_swarm.dlm /usr/local/itt/idl64/bin/bin.linux.x86_64/particle_swarm.dlm
;sudo cp /media/hesslab/PalmClusterTest/PeakSelector_V8.8/3D_Viewer/src/cu_gauss2d/cu_gauss2d.so /usr/local/itt/idl64/bin/bin.linux.x86_64/cu_gauss2d.so
;sudo cp /media/hesslab/PalmClusterTest/PeakSelector_V8.8/3D_Viewer/src/cu_gauss2d/cu_gauss2d.dlm /usr/local/itt/idl64/bin/bin.linux.x86_64/cu_gauss2d.dlm
;sudo cp /media/hesslab/PalmClusterTest/PeakSelector_V8.8/3D_Viewer/src/cu_gauss2d/psfit/particle_swarm_fit.dlm /usr/local/itt/idl64/bin/bin.linux.x86_64/particle_swarm_fit.dlm
;sudo cp /media/hesslab/PalmClusterTest/PeakSelector_V8.8/3D_Viewer/src/cu_gauss2d/psfit/particle_swarm_fit.so /usr/local/itt/idl64/bin/bin.linux.x86_64/particle_swarm_fit.so

;--------------------------------------------------------------
;+
;  Filter based on the given set of parameters.
;-
function FilterByParameter, CGroupParams, ParamLimits, indices, params
    compile_opt idl2, logical_predicate
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
sz = size(CGroupParams)
CGrpSize = sz[1]
npk_tot = sz[2]

	if (NOT LMGR(/VM)) and (NOT LMGR(/DEMO)) and (NOT LMGR(/TRIAL)) and allow_bridge then begin
		print,'Starting Bridge filtering'
		if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else	idl_pwd=pref_get('IDL_WDE_START_DIR')
		cd,current=curr_pwd

		; ************** check if the bridge data is of a proper size. If not - unmap it and start anew
		if bridge_exists then begin
			print,'comparing structures',n_elements(CGroupParams),n_elem_CGP
			if (n_elements(CGroupParams) ne n_elem_CGP) then begin
				print,'Existing Bridge data structure does not agree with the data. Resetting'
				CATCH, Error_status
				SHMUnmap, shmName_data
				SHMUnmap, shmName_filter
				for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
				IF Error_status NE 0 THEN BEGIN
					PRINT, 'Filtering: Error :',!ERROR_STATE.MSG
					CATCH,/CANCEL
				ENDIF
				bridge_exists = 0
			endif
		endif

		;***************** If bridge had not been started, start IDL bridge workers
		if bridge_exists eq 0 then begin
			;common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

			ncores_cluster = fix(strtrim(GETENV('LSB_DJOB_NUMPROC'),2))
			n_br_loops = ncores_cluster gt 0 ? ncores_cluster : !CPU.HW_NCPU
			print, 'Bridge structure not set.    ', n_br_loops, '  CPU cores are present'
			n_br_loops = n_br_loops < n_br_max
			print, 'Will set up',n_br_loops,' bridge child processes'
			CATCH, Error_status
			IF Error_status NE 0 THEN BEGIN
				SHMUnmap, shmName_data
				SHMUnmap, shmName_filter
				for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
				PRINT, 'Starting: Error:',!ERROR_STATE.MSG
				CATCH,/CANCEL
			ENDIF
			i=0
			;fbr_arr=obj_new("IDL_IDLBridge", output=('Z:\IDL\PeakSelector_V9.6\debug\Bridge_output'+strtrim(i,2)+'.txt'))
			;for i=1, n_br_loops-1 do fbr_arr=[fbr_arr, obj_new("IDL_IDLBridge", output=('Z:\IDL\PeakSelector_V9.6\debug\Bridge_output'+strtrim(i,2)+'.txt'))]
			fbr_arr=obj_new("IDL_IDLBridge", output='')
			for i=1, n_br_loops-1 do fbr_arr=[fbr_arr, obj_new("IDL_IDLBridge", output='')]
			npk_sub = ceil(npk_tot/n_br_loops)
			sec_st = strtrim(ulong(SYSTIME(/seconds)),2)
			shmName_data='PALM_data_' + sec_st
			SHMMAP , shmName_data, /FLOAT, /DESTROY_SEGMENT, Dimension=[CGrpSize,npk_tot], GET_OS_HANDLE=OS_handle_val1
			CGroupParams_bridge = SHMVAR(shmName_data)
			CGroupParams_bridge[0,0] = CGroupParams
			n_elem_CGP = n_elements(CGroupParams_bridge)
			shmName_filter='PALM_filter_' + sec_st
			SHMMAP,shmName_filter, /LONG, /DESTROY_SEGMENT, Dimension=[npk_tot], GET_OS_HANDLE=OS_handle_val2
			n_elem_fbr = npk_tot
			imin = ulindgen(n_br_loops) * npk_sub
			imax = ((imin + npk_sub)  < npk_tot)-1
 			for i=0, n_br_loops-1 do begin
				fbr_arr[i]->setvar, 'nlps',i
				fbr_arr[i]->setvar, 'CGrpSize',	CGrpSize
				fbr_arr[i]->setvar, 'npk_sub',npk_sub
				fbr_arr[i]->setvar, 'npk_tot',npk_tot
				fbr_arr[i]->setvar, 'OS_handle_val',OS_handle_val1
				fbr_arr[i]->setvar, 'OS_handle_val',OS_handle_val2
				fbr_arr[i]->setvar, 'shmName_data',shmName_data
				fbr_arr[i]->setvar, 'shmName_filter',shmName_filter
				fbr_arr[i]->setvar, 'istart',imin[i]
				fbr_arr[i]->setvar, 'istop',imax[i]
				fbr_arr[i]->execute, 'CPU,TPOOL_NTHREADS=1'
            	fbr_arr[i]->execute,'di = istop - istart +1'
            	fbr_arr[i]->execute,'SHMMAP,shmName_data,/FLOAT, Dimension=[CGrpSize,npk_tot],OS_Handle=OS_handle_val1'
            	fbr_arr[i]->execute,'SHMMAP,shmName_filter,/LONG, Dimension=[npk_tot],OS_Handle=OS_handle_val2
            	fbr_arr[i]->setvar, 'IDL_dir', IDL_pwd
			    fbr_arr[i]->execute,'cd, IDL_dir'
            	fbr_arr[i]->execute,"restore,'Peakselector.sav'"
			endfor
			bridge_exists = 1
		endif

		filter_bridge=SHMVAR(shmName_filter)
		filter_bridge[0]=LONG(npk_tot)

		for i=0, n_br_loops-1 do begin
			fbr_arr[i]->execute,'CGroupParams_bridge=SHMVAR(shmName_data)'
			fbr_arr[i]->execute,'filter_bridge=SHMVAR(shmName_filter)'
			fbr_arr[i]->setvar, 'params',params
			fbr_arr[i]->setvar, 'ParamLimits',ParamLimits
          	fbr_arr[i]->execute,'low  = ParamLimits[params,0]#replicate(1,di)'
        	fbr_arr[i]->execute,'high = ParamLimits[params,1]#replicate(1,di)'
            fbr_arr[i]->execute,'filter_subset = (CGroupParams_bridge[params,istart:istop] ge low) and (CGroupParams_bridge[params,istart:istop] le high)',/NOWAIT
		endfor

		Alldone = 0
		while alldone EQ 0 do begin
			wait,0.5
			Alldone = 1
			for i=0, n_br_loops-1 do begin
				bridge_done=fbr_arr[i]->Status()
				print,'Bridge',i,'  status0:',bridge_done
				Alldone = Alldone * (bridge_done ne 1)
			endfor
		endwhile

		for i=0, n_br_loops-1 do begin
			fbr_arr[i]->setvar, 'nlps',i
		    fbr_arr[i]->execute,'istart = nlps * npk_sub'
          	fbr_arr[i]->execute,'istop = ((istart + npk_sub) < npk_tot)-1'
		    fbr_arr[i]->execute, 'filter_bridge[istart] = floor(total(temporary(filter_subset), 1) / n_elements(params))',/NOWAIT
		endfor

		Alldone = 0
		while alldone EQ 0 do begin
			wait,0.5
			Alldone = 1
			for i=0, n_br_loops-1 do begin
				bridge_done=fbr_arr[i]->Status()
				print,'Bridge',i,'  status1:',bridge_done
				Alldone = Alldone * (bridge_done ne 1)
			endfor
		endwhile

		filter = filter_bridge

	endif else begin
		;Use plain IDL
		CGPsz = size(CGroupParams)
		low  = ParamLimits[params,0]#replicate(1,CGPsz[2])
		high = ParamLimits[params,1]#replicate(1,CGPsz[2])
		filter = (CGroupParams[params,*] ge low) and (CGroupParams[params,*] le high)
		filter = floor(total(temporary(filter), 1) / n_elements(params))
	endelse

    CATCH,/CANCEL
    return, filter
end
;
;--------------------------------------------------------------
;
pro On_Change_Filter_Select, Event
	current_button=widget_info(Event.ID,/button_set)
	widget_control,Event.ID,set_button=(1-current_button)
end
;
;--------------------------------------------------------------
;
pro Set_Allow_Bridge, Event
	common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
	allow_bridge = widget_info(Event.ID,/button_set)
	if (allow_bridge eq 0) and (bridge_exists eq 1) then begin
		CATCH, Error_status
		SHMUnmap, shmName_data
		SHMUnmap, shmName_filter
		for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
		bridge_exists = 0
		IF Error_status NE 0 THEN BEGIN
			PRINT, 'Error index: ', Error_status
      		PRINT, 'Error message: ', !ERROR_STATE.MSG
			CATCH,/CANCEL
		ENDIF
	endif
end
;--------------------------------------------------------------
;+
;  Filter the peaks using the parameters for single peaks.
;-
pro Filterit
    compile_opt idl2, logical_predicate
	common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
	common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
    common managed, ids, names, modalList

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
Chi_ind = min(where(RowNames eq 'ChiSquared'))							; CGroupParametersGP[7,*] - Chi Squared
FitOK_ind = min(where(RowNames eq 'FitOK'))								; CGroupParametersGP[8,*] - Original FitOK
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))				; CGroupParametersGP[10,*]
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))				; CGroupParametersGP[11,*]
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))					; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigAmp_ind = min(where(RowNames eq 'Sigma Amplitude'))					; CGroupParametersGP[13,*]
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))				; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))				; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
AmpL1_ind = min(where(RowNames eq 'Amplitude L1'))						; CGroupParametersGP[27,*] - Label 1 Amplitude
AmpL2_ind = min(where(RowNames eq 'Amplitude L2'))						; CGroupParametersGP[28,*] - Label 2 Amplitude
AmpL3_ind = min(where(RowNames eq 'Amplitude L3'))						; CGroupParametersGP[29,*] - Label 3 Amplitude
ZState_ind = min(where(RowNames eq 'Z State'))							; CGroupParametersGP[32,*] - Z State
SigL1_ind = min(where(RowNames eq 'Zeta0'))								; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))						; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))						; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))							; CGroupParametersGP[35,*] - Sigma Z
Coh_ind = min(where(RowNames eq '36 Coherence'))						; CGroupParametersGP[36,*] - Coherence
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

allind1 = [Off_ind, $
		Amp_ind, $
		X_ind, $
		Y_ind, $
		Xwid_ind, $
		Ywid_ind, $
		Nph_ind, $
		Chi_ind, $
		FitOK_ind, $
		FrNum_ind, $
		PkInd_ind, $
		PkGlInd_ind, $
		Par12_ind, $
		SigAmp_ind, $
		SigNphX_ind, $
		SigNphY_ind, $
		SigX_ind, $
		SigY_ind, $
		GrInd_ind, $
		LabelSet_ind, $
		AmpL1_ind, $
		AmpL2_ind, $
		AmpL3_ind, $
		SigL1_ind, $
		ZState_ind, $
		SigL2_ind, $
		SigL3_ind, $
		Z_ind, $
		SigZ_ind, $
		Coh_ind, $
		Ell_ind, $
		UnwZ_ind, $
		UnwZErr_ind]
	allind_valid = allind1[where(allind1 ge 0)]

    tstart = systime(/SECONDS)
    topID = ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
    WidID_MENU_ApplyFilterSelectively = Widget_Info(topID, find_by_uname='W_MENU_ApplyFilterSelectively')
    filter_select=widget_info(WidID_MENU_ApplyFilterSelectively, /BUTTON_SET)
    if filter_select then begin
    	filter0=filter
    	WidDL_LabelID = Widget_Info(topID, find_by_uname='WID_DROPLIST_Label')
		selectedlabel = widget_info(WidDL_LabelID,/DropList_Select)
	endif

    if (n_elements(CGroupParams) le 1) then begin
        void = dialog_message('Please load a data file')
        return      ; if data not loaded return
    endif

	if GrX_ind gt 0 then PALM_with_groups = (ParamLimits[GrSigX_ind,1] gt 0) else PALM_with_groups = 0

    ;allind = indgen(CGrpSize)
    ;params = allind
    ;if PALM_with_groups then params = (CGrpSize ge 49) ? [allind[0:17],allind[25:36],allind[43:45]]  : [allind[0:17],allind[25:(36<(CGrpSize-1))]]
    params = allind_valid
    params = params[where((ParamLimits[params,0] ne 0) or (ParamLimits[params,1] ne 0))]
    indices = intarr(CGrpSize)
    indices[params]=1
    ;print,'Starting FilterByParameter'
    filter = FilterByParameter(CGroupParams, ParamLimits, indices, params)

    if filter_select then begin
    	filter1=filter
		if LabelSet_ind ge 0 then filter1 = filter1 or (CGroupParams[LabelSet_ind,*] ne selectedlabel)
    	filter=filter0*filter1
	endif

    peakcount=total(filter)

    if peakcount ge 1 then begin
        vp=finite(CGroupParams[*,where(filter)])
        vpcnt=round(total(vp)/CGrpSize)
        vpcnt1=round(total(vp)/50)
        if (peakcount ne vpcnt) and (peakcount ne vpcnt1) then begin
            print,'invalid peak parameters encountered'
            print,'total number of peaks=',peakcount,'   valid=',vpcnt
        endif
    endif
    wlabel = widget_info(topID, find_by_uname='WID_LABEL_NumberSelected')
    widget_control,wlabel,set_value='Peak Count = '+string(peakcount)
    print,'total filtering time (sec):  ',(systime(/seconds)-tstart),'     total filtered peaks:  ',peakcount
end


;--------------------------------------------------------------
;+
;  Filter the peaks using the parameters for grouped peaks.
;-
pro GroupFilterit
compile_opt idl2, logical_predicate
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common managed, ids, names, modalList
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

print,'Running Group FilterIt'
Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
Chi_ind = min(where(RowNames eq 'ChiSquared'))							; CGroupParametersGP[7,*] - Chi Squared
FitOK_ind = min(where(RowNames eq 'FitOK'))								; CGroupParametersGP[8,*] - Original FitOK
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))				; CGroupParametersGP[10,*]
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))				; CGroupParametersGP[11,*]
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))					; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigAmp_ind = min(where(RowNames eq 'Sigma Amplitude'))					; CGroupParametersGP[13,*]
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))				; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))				; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
AmpL1_ind = min(where(RowNames eq 'Amplitude L1'))						; CGroupParametersGP[27,*] - Label 1 Amplitude
AmpL2_ind = min(where(RowNames eq 'Amplitude L2'))						; CGroupParametersGP[28,*] - Label 2 Amplitude
AmpL3_ind = min(where(RowNames eq 'Amplitude L3'))						; CGroupParametersGP[29,*] - Label 3 Amplitude
SigL1_ind = min(where(RowNames eq 'Zeta0'))								; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))						; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))						; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))							; CGroupParametersGP[35,*] - Sigma Z
Coh_ind = min(where(RowNames eq '36 Coherence'))						; CGroupParametersGP[36,*] - Coherence
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

allind1 = [FrNum_ind, $
		Gr_ind, $
		GrX_ind, $
		GrY_ind, $
		GrSigX_ind, $
		GrSigY_ind, $
		GrNph_ind, $
		Gr_size_ind, $
		GrInd_ind, $
		LabelSet_ind, $
		GrAmpL1_ind, $
		GrAmpL2_ind, $
		GrAmpL3_ind, $
		GrZ_ind, $
		GrSigZ_ind, $
		GrCoh_ind, $
		Gr_Ell_ind, $
		UnwGrZ_ind, $
		UnwGrZErr_ind]
	allind_valid = allind1[where(allind1 ge 0)]

    tstart = systime(/SECONDS)
    topID = ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

    if (n_elements(CGroupParams) le 1) then begin
        void = dialog_message('Please load a data file')
        return      ; if data not loaded return
    endif

    WidID_MENU_ApplyFilterSelectively = Widget_Info(topID, find_by_uname='W_MENU_ApplyFilterSelectively')
    filter_select=widget_info(WidID_MENU_ApplyFilterSelectively, /BUTTON_SET)
    if filter_select then begin
    	filter0=filter
    	WidDL_LabelID = Widget_Info(topID, find_by_uname='WID_DROPLIST_Label')
		selectedlabel = widget_info(WidDL_LabelID,/DropList_Select)
	endif

   params = allind_valid

    params = params[where((ParamLimits[params,0] ne 0) or (ParamLimits[params,1] ne 0))]
    indices = intarr(CGrpSize)
    indices[params]=1
	filter = FilterByParameter(CGroupParams, ParamLimits, indices, params)

    peakcount=total(filter)
    if peakcount ge 1 then begin
        vp=finite(CGroupParams[*,where(filter)])
        vpcnt=round(total(vp)/CGrpSize)
        vpcnt1=round(total(vp)/50)
        if (peakcount ne vpcnt) and (peakcount ne vpcnt1) then begin
            print,'invalid peak parameters encountered'
            print,'total number of peaks=',peakcount,'   valid=',vpcnt
        endif
    endif
    wlabel = Widget_Info(TopID, find_by_uname='WID_LABEL_NumberSelected')

    if filter_select then begin
    	filter1=filter or (CGroupParams[LabelSet_ind,*] ne selectedlabel)
    	filter=filter0*filter1
	endif

    peakcount=long(total(filter))
	filter=temporary(filter)*(CGroupParams[GrInd_ind,*] eq 1)				;reference by first peak in local group (they are indexed from 1, not from 0)
    peakcountGroups=long(total(filter))
    print,'Grp Cnt/Pk Cnt = '+strtrim(string(peakcountGroups),2)+' / '+strtrim(string(peakcount),2)
    widget_control,wlabel,set_value='Grp Cnt/Pk Cnt = '+strtrim(string(peakcountGroups),2)+' / '+strtrim(string(peakcount),2)
    wait,0.1
end
;
;-----------------------------------------------------------------
;

function GetScaleFactor, ParamLimits, wxsz, wysz		;Returns mag factor of CCD pixels to display pixels
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

FunctionId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Function')
FunctionItem=widget_info(FunctionId,/DropList_Select)						;	(0: Center Locations,  1: Gaussian Normalized,   2: Gaussian Amplitude)
FilterId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Filter')
FilterItem=widget_info(FilterId,/DropList_Select)							;	(0: Frame Peaks,   1: Group Peaks)
AccumId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Accumulate')
AccumItem=widget_info(AccumId,/DropList_Select)								;	(0: Envelope,   1: Sum)
X_ind = FilterItem ? min(where(RowNames eq 'Group X Position')) : min(where(RowNames eq 'X Position'))
Y_ind = FilterItem ? min(where(RowNames eq 'Group Y Position')) : min(where(RowNames eq 'Y Position'))
Z_ind = FilterItem ? min(where(RowNames eq 'Group Z Position')) : min(where(RowNames eq 'Z Position'))
Z_unwr_ind = FilterItem ? min(where(RowNames eq 'Unwrapped Group Z')) : min(where(RowNames eq 'Unwrapped Z'))

ZUnwrZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_65')
ZUnwrZ_swap_state=widget_info(ZUnwrZ_swap_menue_ID,/button_set)
if ZUnwrZ_swap_state then begin
	place_holder=Z_ind
	Z_ind=Z_unwr_ind
	Z_unwr_ind=place_holder
endif

XZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapXZ')
XZ_swap_state = widget_info(XZ_swap_menue_ID,/button_set)
if XZ_swap_state then begin
	place_holder=X_ind
	X_ind=Z_ind
	Z_ind=place_holder
endif

YZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapYZ')
YZ_swap_state = widget_info(YZ_swap_menue_ID,/button_set)
if YZ_swap_state then begin
	place_holder=Y_ind
	Y_ind=Z_ind
	Z_ind=place_holder
endif

dxmn = paramlimits[X_ind,0]
dymn = paramlimits[Y_ind,0]
dxmx = paramlimits[X_ind,1]
dymx = paramlimits[Y_ind,1]
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))
return,mgw
end
;
;----------------------------------------------------------------------------
;
pro LimitUnwrapZ
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if CGrpSize ge 47 then begin
	for iz = 44,47,3 do begin
		ParamLimits[iz,0] = ParamLimits[iz,0] > (-1000.00)
		ParamLimits[iz,1] = ParamLimits[iz,1] < 1000.00
	endfor
endif
end
;
;-----------------------------------------------------------------
;
Pro ReloadParamlists, Event, param_indecis	;  refreshes the list of parameters in ParamLimits array and table
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

	if n_elements(CGroupParams) le 1 then begin
		z=dialog_message('Please load a data file')
		return
	endif

	Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
	Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
	X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
	Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
	Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
	Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
	Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
	Chi_ind = min(where(RowNames eq 'ChiSquared'))							; CGroupParametersGP[7,*] - Chi Squared
	FitOK_ind = min(where(RowNames eq 'FitOK'))								; CGroupParametersGP[8,*] - Original FitOK
	FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
	Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))					; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
	SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))				; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
	SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))				; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
	SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
	SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
	Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
	GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
	GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
	GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
	GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
	GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
	Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
	GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group
	LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
	AmpL1_ind = min(where(RowNames eq 'Amplitude L1'))						; CGroupParametersGP[27,*] - Label 1 Amplitude
	AmpL2_ind = min(where(RowNames eq 'Amplitude L2'))						; CGroupParametersGP[28,*] - Label 2 Amplitude
	AmpL3_ind = min(where(RowNames eq 'Amplitude L3'))						; CGroupParametersGP[29,*] - Label 3 Amplitude
	SigL1_ind = min(where(RowNames eq 'Zeta0'))								; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
	SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))						; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
	ZState_ind = max(where(RowNames eq 'Z State'))
	SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))						; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
	Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
	SigZ_ind = min(where(RowNames eq 'Sigma Z'))							; CGroupParametersGP[35,*] - Sigma Z
	GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
	GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
	GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
	GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
	GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
	Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
	Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - group Ellipticity

	if GrX_ind gt 0 then PALM_with_groups = (ParamLimits[GrSigX_ind,1] gt 0) else PALM_with_groups = 0

	TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
	bot=fltarr(CGrpSize)
	bot_orig = [0.001, 1., 0., 0., 0., 0., 1., 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
	xm = min([n_elements(bot),n_elements(bot_orig)])-1
	bot[0:xm] = bot_orig[0:xm]
	Par_Size = (CGrpSize < (size(paramlimits))[1])
	; param_indecis is optional parameter, if not supplied in the call, ParamLimits are reloaded for all parameters
	if n_elements(param_indecis) eq 0 then param_indecis=indgen(Par_Size) else begin
		Par_Size = Par_Size < max(param_indecis)
		param_indecis = param_indecis(where((param_indecis le Par_Size) and (param_indecis ge 0)))
	endelse
	valid_cgp=WHERE(FINITE(CGroupParams[param_indecis,*]),cnt)
	if cnt le 1 then return
	for k=0,(n_elements(param_indecis)-1) do begin
		i=param_indecis[k]
		valid_cgp=WHERE(FINITE(CGroupParams[i,*]),cnt)
		Subset=CGroupParams[i,valid_cgp]
		if cnt gt 1 then begin
			ParamLimits[i,2]=Median(Subset)
			if i eq GrSigX_ind then ParamLimits[GrSigX_ind,2]=(ParamLimits[GrSigX_ind,2]>ParamLimits[SigX_ind,2])
			if i eq GrSigY_ind then ParamLimits[GrSigY_ind,2]=(ParamLimits[GrSigY_ind,2]>ParamLimits[SigY_ind,2])
			if i eq GrNph_ind then ParamLimits[GrNph_ind,2]=(ParamLimits[GrNph_ind,2]>ParamLimits[Nph_ind,2])
			if Gr_size_ind ge 0 then if i eq Gr_size_ind then ParamLimits[Gr_size_ind,2]=(ParamLimits[Gr_size_ind,2]>10)
			if GrInd_ind ge 0 then if i eq GrInd_ind then ParamLimits[GrInd_ind,2]=(ParamLimits[GrInd_ind,2]>10)
			if GrSigZ_ind ge 0 then if i eq GrSigZ_ind then ParamLimits[GrSigZ_ind,2]=(ParamLimits[GrSigZ_ind,2]>ParamLimits[SigZ_ind,2])
			Min_ParamLimits=min(Subset)
			if i lt 11 then ParamLimits[i,0]=bot[i] > 0.9*Min_ParamLimits else ParamLimits[i,0]=0.9*Min_ParamLimits
			if Min_ParamLimits lt 0 then ParamLimits[i,0]=1.1*Min_ParamLimits < 0.9*Min_ParamLimits
			;ParamLimits[i,1]=20.*ParamLimits[i,2] < (1.1*max(Subset) > 0.9*max(Subset))
			ParamLimits[i,1]=max(Subset)
			if i ge 43 and i le 48 then begin
				ParamLimits[i,0]= 1.1*Min_ParamLimits < 0.9*Min_ParamLimits
				ParamLimits[i,1]= 1.1*max(Subset) > 0.9*max(Subset)
			endif
			ParamLimits[i,3]=ParamLimits[i,1] - ParamLimits[i,0]
			ParamLimits[i,2]=(ParamLimits[i,1] + ParamLimits[i,0])/2.
		endif
	endfor
	LimitUnwrapZ
	if FitOK_ind ge 0 then ParamLimits[FitOK_ind,0:3]=[1.,2.,1.5,1]
	if GrInd_ind ge 0 then begin
		mx = max(CgroupParams[GrInd_ind,*])
		ParamLimits[GrInd_ind,0:3]=[0,mx,mx/2.,mx-1]
	endif

	fri1 = [Off_ind, Amp_ind, Nph_ind, Chi_ind, FrNum_ind, Par12_ind, GrInd_ind,  $
			AmpL1_ind, AmpL2_ind, AmpL3_ind, SigL1_ind, SigL2_ind, SigL3_ind,     $
			ZState_ind, Z_ind, GrAmpL1_ind, GrAmpL2_ind, GrAmpL3_ind, GrZ_ind]
	fri1 = fri1[where(fri1 ge 0)]

	fri2 = [Off_ind, Amp_ind, Nph_ind, Chi_ind, FrNum_ind, Par12_ind, Gr_ind,  $
			GrX_ind, GrY_ind, GrSigX_ind, GrSigY_ind, GrNph_ind, Gr_size_ind,  $
			GrInd_ind, LabelSet_ind]
	fri2 = fri2[where(fri2 ge 0)]

	if PALM_with_groups then full_range_indecis = fri1 else full_range_indecis = fri2

	for ii=0,(n_elements(full_range_indecis)-1) do begin
		ip=full_range_indecis[ii]
		trash = where(param_indecis eq ip,cnt_ip)
		if cnt_ip then begin
			ParamLimits[ip,0]=min(CgroupParams[ip,*])
			ParamLimits[ip,1]=max(CgroupParams[ip,*])
			ParamLimits[ip,2]=(ParamLimits[ip,0]+ParamLimits[ip,1])/2.
			ParamLimits[ip,3]=(ParamLimits[ip,1]-ParamLimits[ip,0]-1)
		endif
	endfor

	xmin= 0 > ParamLimits[X_ind,0]
	xmax= (xydsz[0]-1) < ParamLimits[X_ind,1]
	ymin= 0 > ParamLimits[Y_ind,0]
	ymax= (xydsz[1]-1) < ParamLimits[Y_ind,1]

	XZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapXZ')
	XZ_swapped=Widget_Info(XZ_swap_menue_ID,/button_set)

	YZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapYZ')
	YZ_swapped=Widget_Info(YZ_swap_menue_ID,/button_set)

	if XZ_swapped then begin
		xmin= ParamLimits[X_ind,0]
		xmax= ParamLimits[X_ind,1]
	endif
	if YZ_swapped then begin
		ymin= ParamLimits[Y_ind,0]
		ymax= ParamLimits[Y_ind,1]
	endif

	ParamLimits[X_ind,0]=xmin
	ParamLimits[X_ind,1]=xmax
	ParamLimits[X_ind,2]=(xmin+xmax)/2.
	ParamLimits[X_ind,3]=(xmax-xmin-1)
	ParamLimits[Y_ind,0]=ymin
	ParamLimits[Y_ind,1]=ymax
	ParamLimits[Y_ind,2]=(ymin+ymax)/2.
	ParamLimits[Y_ind,3]=(ymax-ymin-1)
	if PALM_with_groups then begin
		ParamLimits[GrX_ind,*]=ParamLimits[X_ind,*]
		ParamLimits[GrY_ind,*]=ParamLimits[Y_ind,*]
	endif

	wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
	widget_control,wtable,set_value=transpose(ParamLimits[0:(CGrpSize-1),0:3]), use_table_select=[0,0,3,(CGrpSize-1)]
	widget_control, wtable, /editable,/sensitive
end
;
;+
; NAME:
;
;	DIALOG
;
; PURPOSE:
;
;       A popup widget dialog box to get user input. Like
;       WIDGET_MESSAGE, which is better in some cases, but this widget
;       also includes fields and lists.
;
; CATEGORY:
;
;	Widgets.
;
; CALLING SEQUENCE:
;
;	Result = DIALOG([TEXT])
;
; OPTIONAL INPUTS:
;
;	TEXT - The label seen by the user.
;
; KEYWORD PARAMETERS:
;
;   There are 6 types of dialogs, each with unique behavior.  With
;   each default dialog type are associated buttons; these buttons can
;   be overridden with the BUTTONS keyword, except in the case of the
;   LIST and FIELD dialogs.
;
;       One of the following six keywords MUST be set:
;
;       ERROR - Display an error message; default BUTTONS =
;               ['Abort','Continue']
;
;       WARNING - Display a warning message.  default BUTTONS = ['OK']
;
;       INFO - Display an informational message;
;              default BUTTONS = ['Cancel','OK']
;
;       QUESTION - Ask a question.  default BUTTONS =
;                  ['Cancel','No','Yes']
;
;       LIST - Get a selection from a list of choices.  default
;              BUTTONS = ['Cancel','OK'] Must specify CHOICES = string
;              array of list choices.
;
;              Set the RETURN_INDEX keyword to cause the returned
;	       value to be the zero-based index of the selected
;	       list item.
;
;       FIELD - Get user input, using CW_FIELD.  default BUTTONS =
;               ['Cancel','OK']. FLOAT, INTEGER, LONG, and STRING
;               keywords apply here, as does the VALUE keyword to set
;               an initial value.
;
;   GROUP - Group leader keyword.
;
;   TITLE - title of popup widget.
;
; COMMON BLOCKS:
;
;       DIALOG
;
; OUTPUTS:
;
;       In the case of LIST or FIELD dialogs, this function returns
;       the selected list element or the user input, respectively.
;       Otherwise, this function returns the name of the pressed
;       button.
;
; EXAMPLE:
;
;	1. Create a QUESTION DIALOG widget.
;
;       D = DIALOG(/QUESTION,'Do you want to continue?')
;
;       2. Get the user to enter a number.
;
;       D = DIALOG(/FLOAT,VALUE=3.14159,'Enter a new value for pi.')
;
;       3. Get the user to choose from a list of options.
;
;       D = DIALOG(/LIST,CHOICES=['Snoop','Doggy','Dog'])
;
; MODIFICATION HISTORY:
;
;       David L. Windt, Bell Labs, March 1997
;
;       May 1997 - Added GROUP keyword, and modified use of MODAL
;                  keyword to work with changes in IDL V5.0
;
;       windt@bell-labs.com
;-

pro dialog_event,event
common dialog,selection,fieldid,cho,listid,listindex
on_error,0

; get uvalue of this event
widget_control,event.id,get_uvalue=uvalue

case uvalue of

    'list': begin
        if listindex then selection=event.index else selection=cho(event.index)
        ;; if user double-clicks, then we're done:
        if event.clicks ne 2 then return
    end

    'field': begin
        widget_control,fieldid,get_value=selection
        return
    end

    'buttons': begin
        case 1 of

            ;; field widget?
            widget_info(fieldid,/valid): if (event.value eq 'Cancel') then  $
              selection=event.value else  $
              widget_control,fieldid,get_value=selection

            ;; list widget?
            widget_info(listid,/valid): begin
                if (event.value eq 'Cancel') then begin
                    if listindex then selection=-1 else selection=event.value
                endif else begin
                    id=widget_info(listid,/list_select)
                    if listindex then selection=id else begin
                        if id ge 0 then selection=cho(id) else selection='Cancel'
                    endelse
                endelse
            end

            else: selection=event.value

        endcase
    end

endcase
widget_control,event.top,/destroy
return
end

function dialog,text,buttons=buttons, $
                error=error,warning=warning,info=info,question=question, $
                field=field,float=float,integer=integer, $
                long=long,string=string,value=value, $
                list=list,choices=choices,return_index=return_index, $
                title=title,group=group
common dialog
on_error,2

; set the list and field widget id's to zero, in case
; they've already been defined from a previous instance of dialog.
fieldid=0L
listid=0L
listindex=keyword_set(return_index)

if keyword_set(title) eq 0 then title=' '

; make widget base:
if keyword_set(group) eq 0 then group=0L
if (strmid(!version.release,0,1) eq '5') and $
  widget_info(long(group),/valid) then $
  base=widget_base(title=title,/column,/base_align_center,/modal,group=group) $
else base=widget_base(title=title,/column,/base_align_center)

if keyword_set(float) then field=1
if keyword_set(integer) then field=1
if keyword_set(long) then field=1
if keyword_set(string) then field=1
if n_elements(value) eq 0 then value=0
if n_elements(choices) gt 0 then list=1

; widget configuration depends on type of dialog:
case 1 of
    keyword_set(error):begin
        if n_params() eq 0 then text='Error' else $
          text='Error: '+text
        label=widget_label(base,value=text,frame=0)
        if keyword_set(buttons) eq 0 then buttons=['Abort','Continue']
    end
    keyword_set(warning):begin
        if n_params() eq 0 then text='Warning' else $
          text='Warning: '+text
        label=widget_label(base,value=text,frame=0)
        if keyword_set(buttons) eq 0 then buttons=['OK']
    end
    keyword_set(info):begin
        if n_params() eq 0 then text=' '
        label=widget_label(base,value=text,frame=0)
        if keyword_set(buttons) eq 0 then buttons=['Cancel','OK']
    end
    keyword_set(question):begin
        if n_params() eq 0 then text='Question?'
        label=widget_label(base,value=text,frame=0)
        if keyword_set(buttons) eq 0 then buttons=['Cancel','No','Yes']
    end
    keyword_set(field):begin
        isfield=1
        if n_params() eq 0 then text='Input: '
        sz=size(value)
        if keyword_set(string) and (sz(1) ne 7) then value=strtrim(value)
        fieldid=cw_field(base,title=text, $
                       uvalue='field', $
                       value=value, $
                       /return_events, $
                       float=keyword_set(float), $
                       integer=keyword_set(integer), $
                       long=keyword_set(long), $
                       string=keyword_set(string))
        buttons=['Cancel','OK']
    end
    keyword_set(list):begin
        if keyword_set(choices) eq 0 then $
          message,'Must supply an array of choices for the list.'
        cho=choices
        if n_params() eq 0 then text='Choose: '
        label=widget_label(base,value=text,frame=0)
        listid=widget_list(base,value=choices, $
                           ysize=n_elements(choices) < 10, $
                           uvalue='list')
        buttons=['Cancel','OK']
        cho=choices             ; set common variable for event handler.
    end
endcase
; make widget buttons:
bgroup=cw_bgroup(base,/row,buttons,uvalue='buttons',/return_name)

; realize widget:
widget_control,base,/realize

; manage widget:
if (strmid(!version.release,0,1) eq '5') and  $
  widget_info(long(group),/valid) then $
  xmanager,'dialog',base,group=group  $
else xmanager,'dialog',base,/modal

; return selection:
return,selection
end
;
;-----------------------------------------------------------------
;
Pro ReloadPeakColumn,peak_index		; reloads the "Peak" column in the main table, according to modified 'Frame #', 'Peak Index #', "Global Peak Index #', or 'Label' values
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))
if PkGlInd_ind lt 0 then return
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
if n_elements(CGroupParams) le 1 then return

if (peak_index lt min(CGroupParams[PkGlInd_ind,*])) or (peak_index gt max(CGroupParams[PkGlInd_ind,*])) then begin
	print,'cannot find data for the peak',peak_index
	return
endif else print,'Updating Peak Column data for peak #',peak_index
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
peak_settings=CGroupParams[0:(CGrpSize-1),peak_index]
sz=size(CGroupParams)
wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(peak_settings), use_table_select=[4,0,4,(CGrpSize-1)]
widget_control, wtable, /editable,/sensitive
end
;
;-----------------------------------------------------------------
;
Pro SetRawSliders,Event				; Resets the WID_SLIDER_RawFrameNumber and WID_SLIDER_RawPeakIndex
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
if (size(RawFilenames))[2] eq 0 then return
raw_files=where(RawFilenames ne '',raw_cnt)
if raw_cnt eq 0 then return
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
RawFileNameWidID = Widget_Info(TopID, find_by_uname='WID_DROPLIST_RawFileName')
WidFrameNumber = Widget_Info(TopID, find_by_uname='WID_SLIDER_RawFrameNumber')
WidPeakNumber = Widget_Info(TopID, find_by_uname='WID_SLIDER_RawPeakIndex')
filenames_no_path=strarr(raw_cnt)

FrNum_ind = min(where(RowNames eq 'Frame Number'))
LabelSet_ind = min(where(RowNames eq 'Label Set'))
X_ind = min(where(RowNames eq 'X Position'))

for i=0,(raw_cnt-1) do begin
	pos_rawfilename_wind=strpos(RawFileNames[raw_files[i]],'\',/reverse_search,/reverse_offset)
	pos_rawfilename_unix=strpos(RawFileNames[raw_files[i]],'/',/reverse_search,/reverse_offset)
	pos_rawfilename=max([pos_rawfilename_wind,pos_rawfilename_unix])
	filenames_no_path[i]=AddExtension(strmid(RawFileNames[raw_files[i]],pos_rawfilename),'.dat')
endfor
widget_control,RawFileNameWidID, GET_VALUE=existing_filenames_no_path
if ~array_equal(existing_filenames_no_path,filenames_no_path) then widget_control,RawFileNameWidID, SET_VALUE=filenames_no_path

widget_control,WidFrameNumber,get_value=RawFrameNumber
Raw_File_Index=(widget_info(RawFileNameWidID,/DropList_Select))[0]
if n_elements(CGroupParams) le 1 then return
if n_elements(CGroupParams[FrNum_ind,*]) le 1 then return

if LabelSet_ind ge 0 then begin
	label_ind = (raw_cnt gt 1)	?	where(CGroupParams[LabelSet_ind,*] eq (Raw_File_Index+1),lbl_cnt)	:	where(CGroupParams[LabelSet_ind,*] eq 0,lbl_cnt)
endif else label_ind = where(CGroupParams[X_ind,*] gt -500, lbl_cnt)

if lbl_cnt lt 1 then begin
	print,'No peaks in this frame'
	widget_control,WidPeakNumber, SET_SLIDER_MAX=0, Sensitive=0
	black_region=intarr(950,50)
	tv,black_region,35,490
	msg_str='No peaks in this frame:  Frame:'+strtrim(RawFrameNumber,2)
	xyouts,65,500,msg_str, CHARSIZE=3,/device
	return
endif

widget_control,WidFrameNumber,SET_SLIDER_MIN = (raw_cnt eq 1)	?	min(CGroupParams[FrNum_ind,*])	:	min(CGroupParams[FrNum_ind,label_ind])
widget_control,WidFrameNumber,SET_SLIDER_MAX = (raw_cnt eq 1)	?	max(CGroupParams[FrNum_ind,*])	:	max(CGroupParams[FrNum_ind,label_ind])
widget_control,WidFrameNumber,get_value=RawFrameNumber

frame_indecis = (raw_cnt eq 1)	?	where(CGroupParams[FrNum_ind,*] eq RawFrameNumber,cnt)	:	where(CGroupParams[FrNum_ind,label_ind] eq RawFrameNumber,cnt)
if cnt lt 1 then begin
	print,'No peaks in this frame'
	widget_control,WidPeakNumber, SET_SLIDER_MAX=0, Sensitive=0
	black_region=intarr(950,50)
	tv,black_region,35,490
	msg_str='No peaks in this frame:  Frame:'+strtrim(RawFrameNumber,2)
	xyouts,65,500,msg_str, CHARSIZE=3,/device
	return
endif

widget_control,WidPeakNumber,SET_SLIDER_MIN = 0, Sensitive=1
widget_control,WidPeakNumber,SET_SLIDER_MAX = max(CGroupParams[10,frame_indecis])>1

print,'Frame:'+strtrim(RawFrameNumber,2)+', total peaks found:  ' + strtrim(n_elements(frame_indecis),2);,' indices:  ',transpose(CGroupParams[10,frame_indecis])

end
;
;-----------------------------------------------------------------
;
pro TVscales,wxsz,wysz,mgw,nm_per_pixel			;Draws micron scale bar
displaypix_per_micron=1000.*mgw/nm_per_pixel
if (displaypix_per_micron lt 20) and (displaypix_per_micron ge 1) then scl=0.1
if (displaypix_per_micron lt 40) and (displaypix_per_micron ge 20) then scl=0.2
if (displaypix_per_micron lt 50) and (displaypix_per_micron ge 40) then scl=0.4
if (displaypix_per_micron lt 100) and (displaypix_per_micron ge 50) then scl=0.5
if (displaypix_per_micron lt 200) and (displaypix_per_micron ge 100) then scl=1.
if (displaypix_per_micron lt 400) and (displaypix_per_micron ge 200) then scl=2.
if (displaypix_per_micron lt 500) and (displaypix_per_micron ge 400) then scl=4.
if (displaypix_per_micron lt 1000) and (displaypix_per_micron ge 500) then scl=5.
if (displaypix_per_micron lt 2000) and (displaypix_per_micron ge 1000) then scl=10.
if (displaypix_per_micron lt 4000) and (displaypix_per_micron ge 2000) then scl=20.
if (displaypix_per_micron lt 5000) and (displaypix_per_micron ge 4000) then scl=40.
if (displaypix_per_micron lt 10000) and (displaypix_per_micron ge 5000) then scl=50.
if (displaypix_per_micron lt 20000) and (displaypix_per_micron ge 10000) then scl=100.
if (displaypix_per_micron lt 1) or (displaypix_per_micron ge 20000) then return
bar=replicate(255,displaypix_per_micron/scl,5)
tv,bar,wxsz*0.8,wysz*0.04
if scl le 5 then xyouts,0.85,0.015,string(1.0/scl,format='(G5.3)')+' micron',/normal,align=0.5,col=255l+256l*255l
if scl gt 5 then xyouts,0.85,0.015,string(1000.0/scl,format='(G6.3)')+' nm',/normal,align=0.5,col=255l+256l*255l
return
end
;-----------------------------------------------------------------
pro TVscales2,wxsz,wysz,mgw,nm_per_pixel			;Draws pixel based scale bars and boxes
;common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier

box4=replicate(255,4,4)
box8=replicate(255,8,8)
box12=replicate(255,12,12)
box16=replicate(255,16,16)
bar160=replicate(255,160,2)

tv,box4,wxsz-200,50
tv,box8,wxsz-150,50
tv,box12,wxsz-100,50
tv,box16,wxsz-50,50
tv,bar160,wxsz-200,22
for j=0,3 do xyouts,wxsz-200+50*j,35,$
	string(nm_per_pixel*4*(j+1)/mgw,format='(I4)')+' nm',/device,align=0.5,col=255l+256l*255l
j=3
xyouts,(wxsz -100),9,string(nm_per_pixel*40*(j+1)/mgw,format='(I5)')+' nm',/device,align=0.5,col=255l+256l*255l
return
end

;-----------------------------------------------------------------
pro ShowColorScaleBarLabel,Event,xydsz,mgw,min_mid_max,label	;Draws color scale bar of rendered image in probablility per nm^2
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

if label eq 0 then begin
	bar=replicate(255,258,12)
	bar[1:256,1:10]=findgen(256)#replicate(1,10)
	tv,bar,40,45
	ypos=0
endif
if label ge 1 then begin
	bar=intarr(258,12,3)
	bar[*,*,label-1]=replicate(255,258,12)
	bar[1:256,1:10,*]=0
	bar[1:256,1:10,Label-1]=findgen(256)#replicate(1,10)
	ypos=30*(Label-1)
	tv,bar,40,45+ypos,true=3
endif

values=min_mid_max/(nm_per_pixel/mgw)^2
strminval=string(values[0],format='(f8.5)')
strmidval=string(values[1],format='(f8.5)')
strmaxval=string(values[2],format='(f8.5)')
xyouts,40,26+ypos,strminval,charsize=1.5,/device,align=0.5
xyouts,40+128,26+ypos,strmidval,charsize=1.5,/device,align=0.5
xyouts,40+256,26+ypos,strmaxval,charsize=1.5,/device,align=0.5

FunctionId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Function')
FunctionItem=widget_info(FunctionId,/DropList_Select)
FilterId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Filter')
FilterItem=widget_info(FilterId,/DropList_Select)
AccumId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Accumulate')
AccumItem=widget_info(AccumId,/DropList_Select)

if FunctionItem eq 1 then funcString='Fluor Molecule'
if FunctionItem eq 2 then funcString='Localized Flour Photon'
if AccumItem eq 0 then accumString='Envelope '
if AccumItem eq 1 then accumString='Total '

if FunctionItem ne 0 then xyouts,40+128,4,accumString+funcString+' Probability /nm^2',charsize=1.5,/device,align=0.5

return
end
;
;-----------------------------------------------------------------
;
; Empty stub procedure used for autoloading.
;
PRO PeakSelector_eventcb
	Print,'PeakSelector call back'
end
;
;-----------------------------------------------------------------
;
pro OnExtractPeaks, Event			;Fit data & converts into PKS files, display fitting steps std x,y gaussian
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
Initialization_PeakSelector, Event.top
filen=''
ReadInfoFile		;Dialog read Txt file
if (size(filen))[2] eq 0 then return
if strlen(filen) eq 0 then return
FittingInfoWid,Group_Leader=Event.top					;show fit params, set display, fit params, & launch fitting

if n_elements(saved_pks_filename) eq 0 then return

if n_elements(CGroupParams) gt 0 then begin
	ReloadParamlists, Event

	wlabel = Widget_Info(Event.Top, find_by_uname='WID_LABEL_0')
	widget_control,wlabel,set_value=saved_pks_filename

	WidFrameNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawFrameNumber')
	WidPeakNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawPeakIndex')
	widget_control,WidFrameNumber,set_value=0
	widget_control,WidPeakNumber,set_value=0
	SetRawSliders,Event
	OnUnZoomButton, Event
	peak_index=0L
	ReloadPeakColumn,peak_index
endif

end
;
;-----------------------------------------------------------------
;
pro OnExtractPeaks_Multiple_TIFFs, Event
	Extract_Peaks_Multiple_TIFFs, GROUP_LEADER=Event.Top
end
;
;-----------------------------------------------------------------
;
pro OpenTheFile, Event				;Read in the the .pks files of fitted parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
Chi_ind = min(where(RowNames eq 'ChiSquared'))							; CGroupParametersGP[7,*] - Chi Squared
FitOK_ind = min(where(RowNames eq 'FitOK'))								; CGroupParametersGP[8,*] - Original FitOK
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))				; CGroupParametersGP[10,*]
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))				; CGroupParametersGP[11,*]
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))					; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigAmp_ind = min(where(RowNames eq 'Sigma Amplitude'))					; CGroupParametersGP[13,*]
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))				; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))				; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
AmpL1_ind = min(where(RowNames eq 'Amplitude L1'))						; CGroupParametersGP[27,*] - Label 1 Amplitude
AmpL2_ind = min(where(RowNames eq 'Amplitude L2'))						; CGroupParametersGP[28,*] - Label 2 Amplitude
AmpL3_ind = min(where(RowNames eq 'Amplitude L3'))						; CGroupParametersGP[29,*] - Label 3 Amplitude
SigL1_ind = min(where(RowNames eq 'Zeta0'))								; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))						; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))						; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))							; CGroupParametersGP[35,*] - Sigma Z
Coh_ind = min(where(RowNames eq '36 Coherence'))						; CGroupParametersGP[36,*] - Coherence
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

xydsz=[256,256]
Print,'Open the file'
dataFile = Dialog_Pickfile(/read,get_path=fpath,filter=['*.pks'],title='Select *.pks file to open')
if dataFile eq '' then return

cd,fpath
widget_control,/hourglass
restore,dataFile
xydsz=[xsz,ysz]


ind_good = where((ApeakParams.NPhot gt 100) and (ApeakParams.FitOK eq 1), sz)
CGroupParams=fltarr(CGrpSize,sz)

CGroupParams[Off_ind:Amp_ind,*]=ApeakParams[ind_good].A[0:1]
CGroupParams[X_ind,*]=ApeakParams[ind_good].peakx
CGroupParams[Y_ind,*]=ApeakParams[ind_good].peaky
CGroupParams[Xwid_ind:Ywid_ind,*]=ApeakParams[ind_good].A[2:3]
CGroupParams[Nph_ind,*]=ApeakParams[ind_good].NPhot
if Chi_ind ge 0 then CGroupParams[Chi_ind,*]=ApeakParams[ind_good].ChiSq
if FitOK_ind ge 0 then CGroupParams[FitOK_ind,*]=ApeakParams[ind_good].FitOK
if FrNum_ind ge 0 then CGroupParams[FrNum_ind,*]=ApeakParams[ind_good].FrameIndex
if PkInd_ind ge 0 then CGroupParams[PkInd_ind,*]=ApeakParams[ind_good].PeakIndex
if PkGlInd_ind ge 0 then CGroupParams[PkGlInd_ind,*]=dindgen(sz)

if tag_names(Apeakparams[0],/structure_name) eq 'TWINKLE' then begin
	CGroupParams[Xwid_ind:Ywid_ind,*]=ApeakParams[ind_good].A[2:3]
	if SigNphX_ind gt 0 then CGroupParams[SigNphX_ind:SigNphY_ind,*]=ApeakParams[ind_good].Sigma2[4:5]
	if Par12_ind gt 0 then CGroupParams[Par12_ind,*]=ApeakParams[ind_good].A[2]*ApeakParams[ind_good].A[3]
	if SigAmp_ind gt 0 then CGroupParams[SigAmp_ind,*]=ApeakParams[ind_good].Sigma2[1]
	if SigX_ind gt 0 then CGroupParams[SigX_ind:SigY_ind,*]=ApeakParams[ind_good].Sigma2[2:3]
	if Ell_ind gt 0 then CGroupParams[Ell_ind,*]=(ApeakParams[ind_good].A[2]-ApeakParams[ind_good].A[3])/(ApeakParams[ind_good].A[2]+ApeakParams[ind_good].A[3])
endif
if tag_names(Apeakparams[0],/structure_name) eq 'TWINKLE_Z' then begin
	CGroupParams[Xwid_ind,*] = ApeakParams[ind_good].peak_widx
	CGroupParams[Ywid_ind,*] = ApeakParams[ind_good].peak_widy
	if Z_ind gt 0 then CGroupParams[Z_ind,*]=ApeakParams[ind_good].A[4]
	if SigX_ind gt 0 then CGroupParams[SigX_ind:SigY_ind,*]=ApeakParams[ind_good].Sigma[2:3]
	if SigZ_ind gt 0 then CGroupParams[SigZ_ind,*]=ApeakParams[ind_good].Sigma[4]
	if Ell_ind gt 0 then CGroupParams[Ell_ind,*]=(ApeakParams[ind_good].peak_widx-ApeakParams[ind_good].peak_widy)/(ApeakParams[ind_good].peak_widx+ApeakParams[ind_good].peak_widy)
	if Par12_ind gt 0 then CGroupParams[Par12_ind,*]=ApeakParams[ind_good].A[5]
endif

wlabel = Widget_Info(Event.Top, find_by_uname='WID_LABEL_0')
widget_control,wlabel,set_value=filename

TotalRawData = totdat

RawFilenames=strarr(3)
pos=strpos(strmid(dataFile,0,strlen(dataFile)-8),'_',/reverse_search,/reverse_offset)
RawFilenames[0]=strmid(dataFile,0,pos)
FlipRotate={frt,present:0B,transp:0B,flip_h:0B,flip_v:0B}
NFrames=long64(max(CGroupParams[FrNum_ind,*]))+1
;NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[FrNum_ind,*])+1))	: long64(max(CGroupParams[FrNum_ind,*])+1)
GuideStarDrift={present:0B,xdrift:fltarr(Nframes),ydrift:fltarr(Nframes),zdrift:fltarr(Nframes)}
FiducialCoeff={fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)}

OnUnZoomButton, Event
WidFrameNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawFrameNumber')
WidPeakNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawPeakIndex')
widget_control,WidFrameNumber,set_value=0
widget_control,WidPeakNumber,set_value=0
SetRawSliders,Event

peak_index=0L
ReloadPeakColumn,peak_index
ReloadParamlists, Event

end
;
;-----------------------------------------------------------------
;
pro AddtotheFile, Event				;Append more fitted parameters of new .pks file to presently loaded fitted parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
Print,'Add to the file'
dataFile = Dialog_Pickfile(/read,get_path=fpath,filter=['*.pks'],title='Select .pks file to append')
if dataFile eq '' then return
cd,fpath
restore,dataFile
if (xsz ne xydsz[0]) or (ysz ne xydsz[1]) then print,'Added file size does not match original file size'
if (xsz ne xydsz[0]) or (ysz ne xydsz[1]) then return
szCGP=size(CGroupParams)
szAPP=size(Apeakparams)
toppeaknumber=CGroupParams[11,szCGP[2]-1]
topframes = CGroupParams[FrNum_ind,szCGP[2]-1]
AGroupParams=dblarr(CGrpSize,szAPP[1])
AGroupParams[0:1,*]=ApeakParams.A[0:1]
AGroupParams[2,*]=ApeakParams.peakx
AGroupParams[3,*]=ApeakParams.peaky
AGroupParams[4:5,*]=ApeakParams.A[2:3]
AGroupParams[6,*]=ApeakParams.NPhot
AGroupParams[7,*]=ApeakParams.ChiSq
AGroupParams[8,*]=ApeakParams.FitOK
AGroupParams[9,*]=ApeakParams.FrameIndex + topframes + 1
AGroupParams[10,*]=ApeakParams.PeakIndex
AGroupParams[11,*]=dindgen(szAPP[1]) + toppeaknumber + 1
AGroupParams[12:13,*]=ApeakParams.Sigma2[0:1]
AGroupParams[14:15,*]=ApeakParams.Sigma2[4:5]
AGroupParams[16:17,*]=ApeakParams.Sigma2[2:3]

CGroupParams=[[CGroupParams],[AGroupParams]]

ReloadParamlists, Event

wlabel = Widget_Info(Event.Top, find_by_uname='WID_LABEL_0')
widget_control,wlabel,set_value=filename

TotalRawData=TotalRawData+totdat
OnUnZoomButton, Event
end
;
;-----------------------------------------------------------------
;
pro On_Save_Data_ASCII_txt, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
	Save_data_ASCII,Group_Leader=Event.top
end
;
;-----------------------------------------------------------------
;
pro SaveASCIIPeakFiltered, Event		;Save the presently loaded & modified peak parameters into an ascii file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.prm'],title='Writes peak filtered params into *.prm ascii file')

if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
ParamsFile=AddExtension(filename,'_PKF.prm')

FilterIt

FilteredPeakIndex=where(filter eq 1)
if FilteredPeakIndex[0] eq -1 then return					;If no peaks in filter then return
FCGroupParams=CGroupParams[*,FilteredPeakIndex]

Title_String=RowNames[0]
for i=1,(CGrpSize-1) do Title_String=Title_String+'	'+RowNames[i]
	openw,1,ParamsFile,width=1024
	printf,1,Title_String
	printf,1,FCGroupParams,FORMAT='('+strtrim((CGrpSize-1),2)+'(E13.5,"'+string(9B)+'"),E13.5)'
close,1
end
;
;-----------------------------------------------------------------
;
pro SaveASCIIPeakFiltered_XYconverted, Event		;Save the presently loaded & modified peak parameters into an ascii file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.prm'],title='Writes peak filtered params into *.prm ascii file')
if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
ParamsFile=AddExtension(filename,'_PKF.prm')

FilterIt

FilteredPeakIndex=where(filter eq 1)
if FilteredPeakIndex[0] eq -1 then return					;If no peaks in filter then return
FCGroupParams=CGroupParams[*,FilteredPeakIndex]


FCGroupParams[2:5,*]*=nm_per_pixel
FCGroupParams[14:17,*]*=nm_per_pixel
FCGroupParams[19:22,*]*=nm_per_pixel

Title_String=RowNames[0]
for i=1,(CGrpSize-1) do Title_String=Title_String+'	'+RowNames[i]
openw,1,ParamsFile,width=1024
printf,1,Title_String
printf,1,FCGroupParams,FORMAT='('+strtrim((CGrpSize-1),2)+'(E13.5,"'+string(9B)+'"),E13.5)'
close,1
end
;
;-----------------------------------------------------------------
;
pro SaveASCIIGroupFiltered, Event		;Save the presently loaded & modified group parameters into an ascii file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.prm'],title='Writes Group filtered params into *.prm ascii file')
if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
ParamsFile=AddExtension(filename,'_GPF.prm')

GroupFilterIt
FilteredPeakIndex=where(filter eq 1)
if FilteredPeakIndex[0] eq -1 then return					;If no peaks in filter then return
FCGroupParams=CGroupParams[*,FilteredPeakIndex]

Title_String=RowNames[0]
for i=1,(CGrpSize-1) do Title_String=Title_String+'	'+RowNames[i]
openw,1,ParamsFile,width=1024
printf,1,Title_String
printf,1,FCGroupParams,FORMAT='('+strtrim((CGrpSize-1),2)+'(E13.5,"'+string(9B)+'"),E13.5)'
close,1
end
;
;-----------------------------------------------------------------
;
pro SaveASCIIGroupFiltered_XYconverted, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.prm'],title='Writes Group filtered params into *.prm ascii file')
if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
ParamsFile=AddExtension(filename,'_GPF.prm')

GroupFilterIt
FilteredPeakIndex=where(filter eq 1)
if FilteredPeakIndex[0] eq -1 then return					;If no peaks in filter then return
FCGroupParams=CGroupParams[*,FilteredPeakIndex]

FCGroupParams[2:5,*]*=nm_per_pixel
FCGroupParams[14:17,*]*=nm_per_pixel
FCGroupParams[19:22,*]*=nm_per_pixel

Title_String=RowNames[0]
for i=1,(CGrpSize-1) do Title_String=Title_String+'	'+RowNames[i]
openw,1,ParamsFile,width=1024
printf,1,Title_String
printf,1,FCGroupParams,FORMAT='('+strtrim((CGrpSize-1),2)+'(E13.5,"'+string(9B)+'"),E13.5)'
close,1
end
;
;-----------------------------------------------------------------
;
pro SavetheCommon, Event			;Save the presently loaded & modified parameters into an idl .sav file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common Zdisplay, Z_scale_multiplier, vbar_top
common Offset, PkWidth_offset
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope

if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
LastRaw=max(where(RawFilenames ne ''))
ref_file=(((file_info(AddExtension(RawFilenames[LastRaw],'.dat'))).exists) or ((file_info(AddExtension(RawFilenames[LastRaw],'.tif'))).exists))? (RawFilenames[LastRaw]+'_IDL.sav') : ''
filename = Dialog_Pickfile(/write,get_path=fpath,file=ref_file, filter=['*IDL.sav'],title='Writes peak params into *IDL.sav file')
if filename eq '' then return

cd,fpath
filename=AddExtension(filename,'_IDL.sav')
if n_elements(SavFilenames) eq 0 then SavFilenames = filename

save, CGroupParams, CGrpSize, ParamLimits, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames, MLRawFilenames,$
		GuideStarDrift, FiducialCoeff, FlipRotate, thisfitcond, RowNames, $
		lambda_vac,nd_water, nd_oil, nmperframe, wind_range, aa, z_unwrap_coeff, cal_lookup_data, ellipticity_slopes, $
		nm_per_pixel, wfilename, PkWidth_offset, Z_scale_multiplier, grouping_gap, grouping_radius100, $
		Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope, $
		lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames, sp_dispersion,  sp_offset, filename=filename

wlabel = Widget_Info(Event.Top, find_by_uname='WID_LABEL_0')
widget_control,wlabel,set_value=filename

end
;
;-----------------------------------------------------------------
;
pro ImportUserASCII, Event
	Import_data_ASCII,Group_Leader=Event.top
end
;
;-----------------------------------------------------------------
;
pro ImportZeissTXt, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common Zdisplay, Z_scale_multiplier, vbar_top
common Offset, PkWidth_offset

Initialization_PeakSelector,Event.top
xydsz=[512,512]

reffilename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.txt'],title='Select *.txt file with Zeiss peak data')
if reffilename eq '' then return
RawFilenames = reffilename
cd,fpath
title=strarr(1)						;nnnn
fname=strmid(reffilename,strlen(fpath))
params_single=transpose(dblarr(12))
cnt=0
openr,1,reffilename
	readf,1,title
	while (~ EOF(1)) do begin
		readf,1,params_single
		if cnt eq 0 then params=params_single else params = [params, params_single]
		cnt+=1
	endwhile
close,1
params=transpose(params)

CGroupParams=MAKE_ARRAY(CGrpSize,cnt,/FLOAT,VALUE=1.0)

nm_per_pixel=100;

CGroupParams[2,*]=params[4,*] / nm_per_pixel					; X
CGroupParams[3,*]=512.0 - params[5,*] / nm_per_pixel			; Y
CGroupParams[4,*]=params[10,*]/nm_per_pixel					; gauss X width
CGroupParams[5,*]=CGroupParams[4,*]							; gauss X width
CGroupParams[6,*]=params[7,*]								; Number of photons
CGroupParams[7,*]=params[9,*]								; Number of photons
CGroupParams[11,*]=dindgen(cnt)								; Absolute PeakIndex
CGroupParams[16,*]=params[6,*]/nm_per_pixel					; localization accuracy (X)
CGroupParams[17,*]=params[6,*]/nm_per_pixel					; localization accuracy (Y)

ReloadParamlists

end
;
;-----------------------------------------------------------------
;
pro RecalltheCommon, Event			;Recall fitted parameters from either an ascii file or an idl .sav file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common Zdisplay, Z_scale_multiplier, vbar_top
common Offset, PkWidth_offset
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope

filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.sav','*.prm'],title='Select *IDL.sav file to open')
if filename eq '' then begin
	print,'filename not recognized', filename
	return
endif

; if there was a previous bridge structure - remove it
if bridge_exists then begin
	print,'Loading New Data Set. Resetting Bridge'
	CATCH, Error_status
	SHMUnmap, shmName_data
	SHMUnmap, shmName_filter
	for nlps=0L,n_br_loops-1 do	obj_destroy, fbr_arr[nlps]
	IF Error_status NE 0 THEN BEGIN
		PRINT, 'Loading New File: Error while unmapping',!ERROR_STATE.MSG
		CATCH,/CANCEL
	ENDIF
	bridge_exists = 0
endif

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number

Initialization_PeakSelector, Event.top
RowNames0 = RowNames
RowNames = ['']
FlipRotate={frt,present:0B,transp:0B,flip_h:0B,flip_v:0B}
FiducialCoeff={fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)}
cd,fpath

if strpos(filename,'IDL.sav') ne -1 then begin
	restore,filename=filename
endif
;print,'after   ',nm_per_pixel
if strpos(filename,'CGP.prm') ne -1 then begin
	openr,1,filename
	readu,1,CGroupParams
	close,1
endif

; if CGroupParams is NOT float type - convert it into float
if (size(CGroupParams))[3] ne 4 then CGroupParams = float(temporary(CGroupParams))

if n_elements(RowNames) le 1 then LoadRowNames

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
Chi_ind = min(where(RowNames eq 'ChiSquared'))							; CGroupParametersGP[7,*] - Chi Squared
FitOK_ind = min(where(RowNames eq 'FitOK'))								; CGroupParametersGP[8,*] - Original FitOK
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))				; CGroupParametersGP[10,*]
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))				; CGroupParametersGP[11,*]
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))					; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigAmp_ind = min(where(RowNames eq 'Sigma Amplitude'))					; CGroupParametersGP[13,*]
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))				; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))				; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
AmpL1_ind = min(where(RowNames eq 'Amplitude L1'))						; CGroupParametersGP[27,*] - Label 1 Amplitude
AmpL2_ind = min(where(RowNames eq 'Amplitude L2'))						; CGroupParametersGP[28,*] - Label 2 Amplitude
AmpL3_ind = min(where(RowNames eq 'Amplitude L3'))						; CGroupParametersGP[29,*] - Label 3 Amplitude
SigL1_ind = min(where(RowNames eq 'Zeta0'))								; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))						; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))						; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))							; CGroupParametersGP[35,*] - Sigma Z
Coh_ind = min(where(RowNames eq '36 Coherence'))						; CGroupParametersGP[36,*] - Coherence
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[FrNum_ind,*])+1))	: long64(max(CGroupParams[FrNum_ind,*])+1)
if (size(GuideStarDrift))[0] eq 0 then GuideStarDrift={present:0B,xdrift:fltarr(Nframes),ydrift:fltarr(Nframes),zdrift:fltarr(Nframes)}
sz=size(CGroupParams)
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Invalid data file')
	return      ; if data not loaded return
endif

;if sz[1] lt 47 then begin
;	add=dblarr((47-sz[1]),sz[2])
;	CGroupParams=[CGroupParams,add]
;endif
size_CGP = size(CgroupParams)
CGrpSize = size_CGP[1]

im_sz=size(image)
WidDL_LabelID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Label')
selectedlabel=0
if im_sz[0] eq 3 then selectedlabel=1	;set to red
if im_sz[0] eq 2 then selectedlabel=0	;set to null
widget_control,WidDL_LabelID,set_droplist_select=selectedlabel
WidSldTopID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Top')
widget_control,WidSldTopID,set_value=labelContrast[0,selectedlabel]
WidSldGammaID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Gamma')
widget_control,WidSldGammaID,set_value=labelContrast[1,selectedlabel]
WidSldBotID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Bot')
widget_control,WidSldBotID,set_value=labelContrast[2,selectedlabel]

wlabel = Widget_Info(Event.Top, find_by_uname='WID_LABEL_0')
widget_control,wlabel,set_value=filename
if (size(SavFilenames))[0] eq 0 then SavFilenames=filename

WidFrameNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawFrameNumber')
WidPeakNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawPeakIndex')
widget_control,WidFrameNumber,set_value=0
widget_control,WidPeakNumber,set_value=0
SetRawSliders,Event

; check if the "RawFileName" points to a non-local file
; if the local file with the same name exists, change RawFileName to point to it
; same with other files and pth

pth=fpath

pos_filename_wind=strpos(filename,'\',/reverse_search,/reverse_offset)
pos_filename_unix=strpos(filename,'/',/reverse_search,/reverse_offset)
pos_filename=max([pos_filename_wind,pos_filename_unix])

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
raw_file_extension = thisfitcond.filetype ? '.tif' : '.dat'
for i=0,n_elements(RawFileNames)-1 do begin
	pos_rawfilename_wind=strpos(RawFileNames[i],'\',/reverse_search,/reverse_offset)
	pos_rawfilename_unix=strpos(RawFileNames[i],'/',/reverse_search,/reverse_offset)
	pos_rawfilename=max([pos_rawfilename_wind,pos_rawfilename_unix])
	if (pos_rawfilename gt 0) and (pos_filename gt 0) then begin
		local_rawfilename=strmid(filename,0,pos_filename)+sep+strmid(RawFileNames[i],pos_rawfilename+1,strlen(RawFileNames[i])-pos_rawfilename-1)
		;conf_info=file_info(local_rawfilename+raw_file_extension)
		conf_info=file_info(AddExtension(local_rawfilename,raw_file_extension))
		if conf_info.exists then RawFileNames[i]=local_rawfilename
	endif
endfor
for i=0,n_elements(MLRawFilenames)-1 do begin
	pos_MLRawFilename_wind=strpos(MLRawFilenames[i],'\',/reverse_search,/reverse_offset)
	pos_MLRawFilename_unix=strpos(MLRawFilenames[i],'/',/reverse_search,/reverse_offset)
	pos_MLRawFilename=max([pos_MLRawFilename_wind,pos_MLRawFilename_unix])
	if (pos_MLRawFilename gt 0) and (pos_filename gt 0) then begin
		local_MLRawFilename=strmid(filename,0,pos_filename)+sep+strmid(MLRawFilenames[i],pos_MLRawFilename+1,strlen(MLRawFilenames[i])-pos_MLRawFilename-1)
		conf_info=file_info(AddExtension(local_MLRawFilename,'.dat'))
		if conf_info.exists then MLRawFilenames[i]=local_MLRawFilename
	endif
endfor
for i=0,n_elements(SavFilenames)-1 do begin
	pos_SavFilename_wind=strpos(SavFilenames[i],'\',/reverse_search,/reverse_offset)
	pos_SavFilename_unix=strpos(SavFilenames[i],'/',/reverse_search,/reverse_offset)
	pos_SavFilename=max([pos_SavFilename_wind,pos_SavFilename_unix])
	if (pos_SavFilename gt 0) and (pos_filename gt 0) then begin
		local_SavFilename=strmid(filename,0,pos_filename)+sep+strmid(SavFilenames[i],pos_SavFilename+1,strlen(SavFilenames[i])-pos_SavFilename-1)
		conf_info=file_info(local_SavFilename)
		if conf_info.exists then SavFilenames[i]=local_SavFilename
	endif
endfor
pos_wfilename_wind = strpos(wfilename,'\',/reverse_search,/reverse_offset)
pos_wfilename_unix = strpos(wfilename,'/',/reverse_search,/reverse_offset)
pos_wfilename = max([pos_wfilename_wind, pos_wfilename_unix])
if (pos_wfilename gt 0) and (pos_filename gt 0) then begin
	local_wfilename = strmid(filename,0,pos_filename)+sep+strmid(wfilename,pos_wfilename+1,strlen(wfilename)-pos_wfilename-1)
	conf_info=file_info(AddExtension(local_wfilename, '_WND.sav'))
	if conf_info.exists then wfilename =local_wfilename
endif

ReloadMainTableColumns, Event.top
peak_index=0L
ReloadPeakColumn,peak_index
ReloadParamlists, Event
OnUnZoomButton, Event

end
;
;-----------------------------------------------------------------
;
pro AddNextLabelData, Event			;Append more fit parameters from a idl .sav file but assign new label number/color
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common Offset, PkWidth_offset
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group

filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.sav','*.prm','*.pks'],title='Select *IDL.sav file to open')
if filename eq '' then return
cd,fpath

existing_ind=where(RawFilenames ne '')  ; normally would not need this, but in case of loading older files with extra empty filenames - get rid of empty overheads
TotalRawData0=TotalRawData
RawFilenames0=RawFilenames[existing_ind]

wlabel = Widget_Info(Event.Top, find_by_uname='WID_LABEL_0')
widget_control,wlabel,get_value=filename0
if n_elements(SavFilenames) eq 0 then SavFilenames0=filename0	else SavFilenames0=SavFilenames[existing_ind]

if ((size(thisfitcond))[2] eq 8)	then	Thisfitcond1=thisfitcond
n_frames_c1=((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[FrNum_ind,*])+1))	: long64(max(CGroupParams[FrNum_ind,*])+1)

P1=ParamLimits
C1=CGroupParams
C1sz=size(C1)
xydsz1=xydsz
TotalRawData1=TotalRawData
WR1=wind_range
C1[LabelSet_ind,*]=C1[LabelSet_ind,*] > 1
NLabels=C1[LabelSet_ind,C1sz[2]-1]			; label of last data point

;if (size(FlipRotate))[2] ne 0 then FlipRotate0=FlipRotate[existing_ind]
;if (size(GuideStarDrift))[2] ne 0 then GuideStarDrift0=GuideStarDrift[existing_ind]
;if (size(FiducialCoeff))[2] ne 0 then FiducialCoeff0=FiducialCoeff[existing_ind]

if strpos(filename,'IDL.sav') ne -1 then begin
	restore,filename=filename
	if n_elements(CGroupParams) le 2 then begin
		z=dialog_message('Invalid data file')
		return      ; if data not loaded return
	endif
	CGPsz=size(CGroupParams)
	Nframes = (CGPsz[2]+C1sz[2])
	existing_ind=where(RawFilenames ne '')  ; normally would not need this, but in case of loading older files with extra empty filenames - get rid of empty overheads
	SavFilenames = [SavFilenames0, filename]
	sz=size(CGroupParams)
	;TotalRawData = TotalRawData0

	; check if the "RawFileName" points to a non-local file
	; if the local file with the same name exists, change RawFileName to point to it
	sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
	for i=0,n_elements(RawFileNames)-1 do begin
		pos_rawfilename_wind=strpos(RawFileNames[i],'\',/reverse_search,/reverse_offset)
		pos_rawfilename_unix=strpos(RawFileNames[i],'/',/reverse_search,/reverse_offset)
		pos_rawfilename=max([pos_rawfilename_wind,pos_rawfilename_unix])
		pos_filename_wind=strpos(filename,'\',/reverse_search,/reverse_offset)
		pos_filename_unix=strpos(filename,'/',/reverse_search,/reverse_offset)
		pos_filename=max([pos_filename_wind,pos_filename_unix])
		if (pos_rawfilename gt 0) and (pos_filename gt 0) then begin
			local_rawfilename=strmid(filename,0,pos_filename)+sep+strmid(RawFileNames[i],pos_rawfilename+1,strlen(RawFileNames[i])-pos_rawfilename-1)
			conf_info=file_info(AddExtension(local_rawfilename,'.dat'))
			if conf_info.exists then RawFileNames[i]=local_rawfilename
		endif
	endfor

	n_frames_cgr = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[FrNum_ind,*])+1))	: long64(max(CGroupParams[FrNum_ind,*])+1)

	RawFilenames=[RawFilenames0,RawFilenames[existing_ind]]
	wind_range=[WR1,wind_range]

	;if (size(FlipRotate0))[2] ne 0 then begin
	;	if (size(FlipRotate))[2] ne 0 then FlipRotate=[FlipRotate0,FlipRotate[existing_ind]] else FlipRotate=FlipRotate0
	;endif
	;if ((size(GuideStarDrift0))[2] ne 0) and (n_frames_c1 eq n_frames_cgr) then begin
	;	if ((size(GuideStarDrift))[2] ne 0) and (n_elements(GuideStarDrift0.xdrift) eq n_elements(GuideStarDrift.xdrift)) then GuideStarDrift=[GuideStarDrift0,GuideStarDrift[existing_ind]] else GuideStarDrift=GuideStarDrift0
	;endif
	;if (size(FiducialCoeff0))[2] ne 0 then begin
	;	if (size(FiducialCoeff))[2] ne 0 then FiducialCoeff=[FiducialCoeff0,FiducialCoeff[existing_ind]] else FiducialCoeff=FiducialCoeff0
	;endif
	FlipRotate=replicate({frt,present:0B,transp:0B,flip_h:0B,flip_v:0B},3)
	GuideStarDrift=replicate({present:0B,xdrift:fltarr(Nframes),ydrift:fltarr(Nframes),zdrift:fltarr(Nframes)},3)
	FiducialCoeff=replicate({fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)},3)

endif else begin
			if strpos(filename,'CGP.prm') ne -1 then begin
				openr,1,filename
				readu,1,CGroupParams
				close,1
			endif else return
endelse

Wid_Check_Recenter_ID = Widget_Info(event.top, find_by_uname='W_MENU_Check_Recenter')
Recenter_XY = widget_info(Wid_Check_Recenter_ID,/BUTTON_SET)
if Recenter_XY and (total(xydsz1 ne xydsz) ne 0) then begin
	print,'recentering the arrays'
	xydsz_new = xydsz1 > xydsz
	TotalRawData_new = fltarr(xydsz_new)
	dxy = (xydsz1 - xydsz)/2.0
	xi = (xydsz_new[0] - xydsz1[0])/2
	xa = xi + xydsz1[0]-1
	yi = (xydsz_new[1] - xydsz1[1])/2
	ya = yi + xydsz1[1]-1
	TotalRawData_new[xi:xa,yi:ya] = TotalRawData1
	xi = (xydsz_new[0] - xydsz[0])/2
	xa = xi + xydsz[0]-1
	yi = (xydsz_new[1] - xydsz[1])/2
	ya = yi + xydsz[1]-1
	TotalRawData_new[xi:xa,yi:ya] = TotalRawData
	if 	dxy[0] lt 0 then begin
		C1[X_ind,*]-=dxy[0]
		C1[GrX_ind,*]-=dxy[0]
		;if dxy[1] lt 0 then TotalRawData_new = TotalRawData
	endif else begin
		if 	dxy[0] gt 0 then begin
			CGroupParams[X_ind,*]+=dxy[0]
			CGroupParams[GrX_ind,*]+=dxy[0]
			;if dxy[1] gt 0 then TotalRawData_new = TotalRawData1
		endif
	endelse
	if 	dxy[1] lt 0 then begin
		C1[Y_ind,*]-=dxy[1]
		C1[GrY_ind,*]-=dxy[1]
	endif else begin
		if 	dxy[1] gt 0 then begin
			CGroupParams[Y_ind,*]+=dxy[1]
			CGroupParams[GrY_ind,*]+=dxy[1]
		endif
	endelse
	xydsz = xydsz_new
	TotalRawData = TotalRawData_new
endif

CGroupParams[LabelSet_ind,*]=NLabels+1						;write in next label index
CGroupParams = [[C1],[CGroupParams]]
C1=0
sz=size(CGroupParams)

WidFrameNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawFrameNumber')
WidPeakNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawPeakIndex')
widget_control,WidFrameNumber,set_value=0
widget_control,WidPeakNumber,set_value=0
SetRawSliders,Event

wlabel = Widget_Info(Event.Top, find_by_uname='WID_LABEL_0')
widget_control,wlabel,set_value=filename

ReloadParamlists, Event
OnUnZoomButton, Event

end
;
;-----------------------------------------------------------------
;
pro MergeLabelsConsecutively, Event			; Merges Multipole Label Sets into a Single Set, the frames are incremented consequtively
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

;CGroupParametersGP[9,*] - frame number
;CGroupParametersGP[10,*] - peak index in the frame
;CGroupParametersGP[11,*] - peak global index
;CGroupParametersGP[26,*] - label number
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number

max_label=max(CGroupParams[LabelSet_ind,*])
min_label=min(CGroupParams[LabelSet_ind,*])
if max_label eq min_label then begin
	CGroupParams[LabelSet_ind,*]=0
	return
endif

prior_peak_ind=where(CGroupParams[LabelSet_ind,*] eq min_label,prior_cnt)
prior_frms = max(CGroupParams[FrNum_ind,prior_peak_ind])
prior_pks = prior_cnt - 1

for i=(min_label+1),max_label do begin
	curr_peak_ind=where((CGroupParams[LabelSet_ind,*] eq i),curr_peaks)
	min_fr=min(CGroupParams[FrNum_ind,curr_peak_ind])
 	curr_frames=max(CGroupParams[FrNum_ind,curr_peak_ind]) - min_fr +1
 	min_pk=min(CGroupParams[11,curr_peak_ind])
 	CGroupParams[FrNum_ind,curr_peak_ind] = CGroupParams[FrNum_ind,curr_peak_ind] - min_fr + prior_frms +1
 	CGroupParams[11,curr_peak_ind] = CGroupParams[11,curr_peak_ind] - min_pk + prior_pks +1
	prior_frms = prior_frms + curr_frames
	prior_pks = prior_pks + curr_peaks
endfor

if n_elements(wind_range) ge 1 then wind_range=wind_range[0]
CGroupParams[LabelSet_ind,*]=0
ReloadParamlists, Event

GuideStarDrift={present:0B,xdrift:fltarr(prior_frms+1),ydrift:fltarr(prior_frms+1),zdrift:fltarr(prior_frms+1)}
OnUnZoomButton, Event

end
;
;-----------------------------------------------------------------
;
pro LoadDIC, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
ReturnDIC_image,DIC_loaded
if n_elements(DIC_loaded) eq 0 then return
DICx = (size(DIC_loaded))[1]				&	DICy = (size(DIC_loaded))[2]
xa = (DICx < xydsz[0])					&	ya = (DICy < xydsz[1])
DIC = DIC_loaded[0:(xa-1),0:(ya-1)]
end
;
;-----------------------------------------------------------------
;
pro ReturnDIC_image,DIC_loaded
filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.sav'],title='Select *IDL.sav file to open')
if filename eq '' then begin
	print,'filename not recognized', filename
	return
endif
cd,fpath
print,'opening file: ', filename
if strpos(filename,'IDL.sav') ne -1 then begin
	restore,filename=filename
	DIC_loaded=TotalRawData
endif else DIC_loaded=-1
end
;
;-----------------------------------------------------------------
;
pro OnPurgeButton, Event			;Purges peaks with poor fit OK values 0 or > 3 and prunes zeroed params
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
          z=dialog_message('Please load a data file')
          return      ; if data not loaded return
endif
OKindex=where(CGroupParams[8,*] eq 1 or CGroupParams[8,*] eq 2,OKcnt)
CGroupParams=CGroupParams[*,OKindex]
sz=size(CGroupParams)
CGroupParams[11,*]=lindgen(sz[2])
ReloadParamlists, Event
end
;
;-----------------------------------------------------------------
;
pro Reassign_Peak_Indecis_within_Frames, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
if n_elements(CGroupParams) le 2 then begin
          z=dialog_message('Please load a data file')
          return      ; if data not loaded return
endif
oStatusBar = obj_new('PALM_StatusBar', $
        COLOR=[0,0,255], $
        TEXT_COLOR=[255,255,255], $
        TITLE='Re-numbering Peaks...', $
        TOP_LEVEL_BASE=tlb)
fraction_complete_last=0.0D
pr_bar_inc=0.01D
fr_max=max(CGroupParams[FrNum_ind,*])
for frame=0L,fr_max do begin
	indecis=where((CGroupParams[FrNum_ind,*]) eq frame,cnt)
	if cnt gt 0 then CGroupParams[10,indecis]=lindgen(cnt)

	fraction_complete=float(frame)/float((fr_max))
	if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
		fraction_complete_last=fraction_complete
		oStatusBar -> UpdateStatus, fraction_complete
	endif
endfor
obj_destroy, oStatusBar
ReloadParamlists, Event
end
;
;-----------------------------------------------------------------
;
pro Purge_Filtered, Event			;Purges peaks with filter=0 and collapses CGroupParams
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
FilterIt
pk_indecis=where(filter,cnt)
if cnt lt 1 then begin
	z=dialog_message('Filter returned no valid peaks')
	return      ; if data not loaded return
endif
CGroupParams = temporary(CGroupParams[*,pk_indecis])
ReloadParamlists, Event
end
;
;-----------------------------------------------------------------
;
pro Purge_selected_peaks, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
FilterIt
pk_indecis=where(filter eq 0,cnt)
if cnt lt 1 then begin
	z=dialog_message('Filter returned no valid peaks')
	return      ; if data not loaded return
endif
CGroupParams = temporary(CGroupParams[*,pk_indecis])
ReloadParamlists, Event
end
;
;-----------------------------------------------------------------
;
pro Purge_Group_Filtered, Event		;Purges peaks with group filter=0 and collapses CGroupParams
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))                ; CGroupParametersGP[25,*] - Frame Index in the Group

GroupFilterIt
pk_indecis=where(filter and CGroupParams[GrInd_ind,*] eq 1,cnt)

if cnt lt 1 then begin
	z=dialog_message('Group filter returned no valid peaks')
	return      ; if data not loaded return
endif
CGroupParams=temporary(CGroupParams[*,pk_indecis])

ReloadParamlists, Event
end
;
;-----------------------------------------------------------------
;
pro Purge_current_filter, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
pk_indecis=where(filter,cnt)
if cnt lt 1 then begin
	z=dialog_message('Filter returned no valid peaks')
	return      ; if data not loaded return
endif
CGroupParams = temporary(CGroupParams[*,pk_indecis])
ReloadParamlists, Event
end
;
;-----------------------------------------------------------------
;
pro On_ApplyFilter1_Button, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
	SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
	SigZ_ind = min(where(RowNames eq 'Sigma Z'))                            ; CGroupParametersGP[35,*] - Sigma Z
	ParamLimits[SigX_ind,1]=0.2
	ParamLimits[SigY_ind,1]=0.2
	ParamLimits[SigZ_ind,1]=15
	TopIndex = (CGrpSize-1)
	wtable = Widget_Info(Event.top, find_by_uname='WID_TABLE_0')
	widget_control,wtable,set_value=transpose(ParamLimits[0:TopIndex,0:3]), use_table_select=[0,0,3,TopIndex]
	OnPeakCentersButton, Event
end
;
;-----------------------------------------------------------------
;
pro On_ApplyFilter2_Button, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
	SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
	SigZ_ind = min(where(RowNames eq 'Sigma Z'))                            ; CGroupParametersGP[35,*] - Sigma Z
	ParamLimits[SigX_ind,1]=0.15
	ParamLimits[SigY_ind,1]=0.15
	ParamLimits[SigZ_ind,1]=10
	TopIndex = (CGrpSize-1)
	wtable = Widget_Info(Event.top, find_by_uname='WID_TABLE_0')
	widget_control,wtable,set_value=transpose(ParamLimits[0:TopIndex,0:3]), use_table_select=[0,0,3,TopIndex]
	OnPeakCentersButton, Event
end
;
;-----------------------------------------------------------------
;
pro Purge_by_Size, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common managed, ids, names, modalList

tstart = systime(/SECONDS)
topID = ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

if (n_elements(CGroupParams) le 2) then begin
	void = dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

params = [2,3]
indices = intarr(CGrpSize)
indices[params]=1
ParamLimits_init=ParamLimits
Paramlimits[2,0]=0.01
Paramlimits[3,0]=0.01
filter = FilterByParameter(CGroupParams, ParamLimits, indices, params)
peakcount=long(total(filter))
ParamLimits=ParamLimits_init

wlabel = widget_info(topID, find_by_uname='WID_LABEL_NumberSelected')
widget_control,wlabel,set_value='Peak Count = '+string(peakcount)
print,'total filtering time (sec):  ',(systime(/seconds)-tstart),'     total filtered peaks:  ',peakcount

xsz=ParamLimits[2,1]
ysz=ParamLimits[3,1]
xydsz=[xsz,ysz]

pk_indecis=where(filter,cnt)
if cnt lt 1 then begin
	z=dialog_message('Filter returned no valid peaks')
	return      ; if data not loaded return
endif
CGroupParams = temporary(CGroupParams[*,pk_indecis])
ReloadParamlists, Event
end
;
;-----------------------------------------------------------------
;
pro Purge_by_XY_coords, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common managed, ids, names, modalList
X_ind = min(where(RowNames eq 'X Position'))                            ; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))                            ; CGroupParametersGP[3,*] - Peak Y  Position
GrX_ind = min(where(RowNames eq 'Group X Position'))                    ; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))                    ; CGroupParametersGP[20,*] - average y - position in the group

tstart = systime(/SECONDS)
topID = ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

if (n_elements(CGroupParams) le 2) then begin
	void = dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

params = [X_ind,Y_ind]
indices = intarr(CGrpSize)
indices[params]=1
;ParamLimits_init=ParamLimits
;Paramlimits[2,0]=0.01
;Paramlimits[3,0]=0.01
filter = FilterByParameter(CGroupParams, ParamLimits, indices, params)
peakcount=long(total(filter))
;ParamLimits=ParamLimits_init

wlabel = widget_info(topID, find_by_uname='WID_LABEL_NumberSelected')
widget_control,wlabel,set_value='Peak Count = '+string(peakcount)
print,'total filtering time (sec):  ',(systime(/seconds)-tstart),'     total filtered peaks:  ',peakcount

xsz=ParamLimits[X_ind,1]-ParamLimits[X_ind,0]
ysz=ParamLimits[Y_ind,1]-ParamLimits[Y_ind,0]
xydsz=[xsz,ysz]

if n_elements(TotalRawData) gt 1 then TotalRawData=TotalRawData(ParamLimits[X_ind,0]:ParamLimits[X_ind,1],ParamLimits[Y_ind,0]:ParamLimits[Y_ind,1])
if n_elements(DIC) gt 1 then DIC=DIC(ParamLimits[X_ind,0]:ParamLimits[X_ind,1],ParamLimits[Y_ind,0]:ParamLimits[Y_ind,1])

pk_indecis=where(filter,cnt)
if cnt lt 1 then begin
	z=dialog_message('Filter returned no valid peaks')
	return      ; if data not loaded return
endif
CGroupParamsGP=CGroupParams[*,pk_indecis]
CGroupParamsGP[X_ind,*]=CGroupParamsGP[X_ind,*]-ParamLimits[X_ind,0]
CGroupParamsGP[GrX_ind,*]=CGroupParamsGP[GrX_ind,*]-ParamLimits[X_ind,0]
CGroupParamsGP[Y_ind,*]=CGroupParamsGP[Y_ind,*]-ParamLimits[Y_ind,0]
CGroupParamsGP[GrY_ind,*]=CGroupParamsGP[GrY_ind,*]-ParamLimits[Y_ind,0]
CGroupParams=CGroupParamsGP
delvar,CGroupParamsGP
end
;
;-----------------------------------------------------------------
;
pro SaveImageJPEG, Event			;Save current image into a jpg file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
filename = Dialog_Pickfile(/write,get_path=fpath)
if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
presentimage=tvrd(true=1)
filename=AddExtension(filename,'.jpg')
write_jpeg,filename,presentimage,true=1
end
;
;-----------------------------------------------------------------
;
pro SaveImageBMP, Event				;Save current image into BMP file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
filename = Dialog_Pickfile(/write,get_path=fpath)
if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
presentimage=tvrd(true=1)
filename=AddExtension(filename,'.bmp')
write_bmp,filename,presentimage,/rgb
end
;
;-----------------------------------------------------------------
;
pro SaveImageTIFF, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
filename = Dialog_Pickfile(/write,get_path=fpath)
if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
presentimage=reverse(tvrd(true=1),3)
filename=AddExtension(filename,'.tiff')
write_tiff,filename,presentimage,orientation=1
end
;
;-----------------------------------------------------------------
;
pro SaveLabelsTIFF, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

FunctionId=widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Function')
FunctionItem=widget_info(FunctionId,/DropList_Select)
FilterId=widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Filter')
FilterItem=widget_info(FilterId,/DropList_Select)
AccumId=widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Accumulate')
AccumItem=widget_info(AccumId,/DropList_Select)
wxsz=1024 & wysz=1024
mgw=GetScaleFactor(ParamLimits, wxsz, wysz)

WARN_on_NOGROUPS=1
RenderWithoutAutoscale, Event, wxsz, wysz, WARN_on_NOGROUPS
img_mn=min(image)
img_mx=max(image)
img_scl=(!D.TABLE_SIZE-1)/(img_mx-img_mn)
scaled_image=(image-img_mn)*img_scl
tv,scaled_image
WidSldTopID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Top')
widget_control,WidSldTopID,get_value=topV
WidSldBotID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Bot')
widget_control,WidSldBotID,get_value=botV
WidSldGammaID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Gamma')
widget_control,WidSldGammaID,get_value=gamma
if topV le botV then begin
	topV = botV+1
	widget_control,WidSldTopID,set_value=TopV
endif

sz_im=size(image)
if sz_im[0] eq 2 then begin
	Timage=image^(gamma/1000.)
	rng=Max(Timage)-Min(Timage)
	tv,bytscl(Timage,min=(botV/1000.)*rng+Min(Timage),max=Max(Timage)-(1.-topV/1000.)*rng)
endif

lmn=min(CGroupParams[LabelSet_ind,*])
lmx=max(CGroupParams[LabelSet_ind,*])
Param26=ParamLimits[LabelSet_ind,*]
filter0=filter
for lbl=lmn,lmx do begin
	filter=filter0
	ParamLimits[LabelSet_ind,0]=lbl
	ParamLimits[LabelSet_ind,1]=lbl
	ParamLimits[LabelSet_ind,2]=lbl
	ParamLimits[LabelSet_ind,4]=lbl
	ParamLimits[LabelSet_ind,3]=0
	RenderWithoutAutoscale, Event, wxsz, wysz, WARN_on_NOGROUPS
	scaled_image=(image-img_mn)*img_scl
	tv,scaled_image
	sz_im=size(image)
	if sz_im[0] eq 2 then begin
		Timage=image^(gamma/1000.)
		rng=Max(Timage)-Min(Timage)
		tv,bytscl(Timage,min=(botV/1000.)*rng+Min(Timage),max=Max(Timage)-(1.-topV/1000.)*rng)
	endif
	presentimage=reverse(tvrd(true=1),3)
	filename=AddExtension(RawFilenames[lbl-lmn],'.tiff')
	write_tiff,filename,presentimage,orientation=1
endfor
ParamLimits[LabelSet_ind,*]=Param26
filter=filter0
return
end
;
;-----------------------------------------------------------------
;
pro SaveCustomTIFF_menu, Event
if !VERSION.OS_family ne 'Windows' then DEVICE, RETAIN=2
SaveCustomTIFF
;SaveCustomTIFF,Group_Leader=Event.top
end
;
;-----------------------------------------------------------------
;
pro SaveImageIDLData, Event			;Save numeric data array of current image into a recallable idl .sav file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
filename = Dialog_Pickfile(/read,get_path=fpath)
if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
filename=AddExtension(filename,'IDLdat.sav')
save,image,filename=filename
end
;
;-----------------------------------------------------------------
;
pro PrintImage, Event
def_dev=!D
presentimage=tvrd(true=1)
set_plot, 'printer'
p= DIALOG_PRINTERSETUP()
;j=DIALOG_PRINTJOB()
TVLCT, R, G, B, /GET
TVLCT, R, G, B
tv,presentimage
device, /CLOSE_DOCUMENT
set_plot,def_dev.name
TVLCT, R, G, B
end
;
;-----------------------------------------------------------------
;
pro StopthePeakSelect, Event		;Stop program for debugging and accessing parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common Offset, PkWidth_offset
common Zdisplay, Z_scale_multiplier, vbar_top
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
Common Multiple_PALM_TIFFs, DoPurge_mTIFFs, Purge_RowNames_mTIFFs, Purge_Params_mTIFFs
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
	Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
common Glob, UseGlobIni_mTIFFs, GlobINI_FileName, Glob_lines
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
stop
end
;
;-----------------------------------------------------------------
;
pro ExittheFile, Event				;Close and exit everything
widget_control,Event.top,/destroy
return
end
;
;-----------------------------------------------------------------
;
pro OnTranspose, Event				;Transpose x and y pixels in coordinate of fitted parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
TransposeImage
OnUnZoomButton, Event
return
end
;
;-----------------------------------------------------------------
;
pro TransposeImage				;Transpose x and y pixels in coordinate of fitted parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
if  (size(FlipRotate))[0] eq 0 then FlipRotate={frt,present:0B,transp:0B,flip_h:0B,flip_v:0B}
FlipRotate[0].present=1
FlipRotate[0].transp=1

newy=CGroupParams[2,*]
CGroupParams[2,*]=CGroupParams[3,*]
CGroupParams[3,*]=newy

newypkwidth=CGroupParams[4,*]
CGroupParams[4,*]=CGroupParams[5,*]
CGroupParams[5,*]=newypkwidth

newsigmaypos=CGroupParams[14,*]
CGroupParams[14,*]=CGroupParams[15,*]
CGroupParams[15,*]=newsigmaypos

newsigmaypkwidth=CGroupParams[16,*]
CGroupParams[16,*]=CGroupParams[17,*]
CGroupParams[17,*]=newsigmaypkwidth

newGy=CGroupParams[19,*]
CGroupParams[19,*]=CGroupParams[20,*]
CGroupParams[20,*]=newGy

newgroupsigmaypos=CGroupParams[21,*]
CGroupParams[21,*]=CGroupParams[22,*]
CGroupParams[22,*]=newgroupsigmaypos

xydsz = reverse(xydsz)

TotalRawData=transpose(temporary(TotalRawData))
if (size(DIC))[0] ne 0 then DIC=transpose(temporary(DIC))
return
end
;
;-----------------------------------------------------------------
;
pro OnFlipHorizontal, Event			;Flip x pixels in coordinate of fitted parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
if  (size(FlipRotate))[0] eq 0 then FlipRotate={frt,present:0B,transp:0B,flip_h:0B,flip_v:0B}
FlipRotate[0].present=1
FlipRotate[0].flip_h=1
newx=xydsz[0]-CGroupParams[2,*]
CGroupParams[2,*]=(newx>2)<(xydsz[0]-2)
newGx=xydsz[0]-CGroupParams[19,*]
CGroupParams[19,*]=(newGx>2)<(xydsz[0]-2)
TotalRawData=reverse(temporary(TotalRawData),1)
if n_elements(DIC) gt 1 then DIC=reverse(temporary(DIC),1)
OnUnZoomButton, Event
return
end
;
;-----------------------------------------------------------------
;
pro OnFlipVertical, Event			;Flip y pixels in coordinate of fitted parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
if  (size(FlipRotate))[0] eq 0 then FlipRotate={frt,present:0B,transp:0B,flip_h:0B,flip_v:0B}
FlipRotate[0].present=1
FlipRotate[0].flip_v=1
newy=xydsz[1]-CGroupParams[3,*]
CGroupParams[3,*]=(newy>2)<(xydsz[1]-2)
newGy=xydsz[1]-CGroupParams[20,*]
CGroupParams[20,*]=(newGy>2)<(xydsz[1]-2)
TotalRawData=reverse(temporary(TotalRawData),2)
if n_elements(DIC) gt 1 then DIC=reverse(temporary(DIC),2)
OnUnZoomButton, Event
return
end
;
;-----------------------------------------------------------------
;
pro OnRotate90CW, Event				;Rotate 90 degree CW in coordinate of fitted parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
TransposeImage
OnFlipVertical,Event
;OnUnZoomButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnRotate90CCW, Event			;Rotate 90 degree CCW in coordinate of fitted parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
TransposeImage
OnFlipHorizontal,Event
;OnUnZoomButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnTwist, Event					;Open slider to set angle of rotating data
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
RotWid
OnUnZoomButton, Event
end
;
;-----------------------------------------------------------------
;
pro TestWriteGuideStar, Event		; Starts GuideStar Menu Widget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
GuideStarWid,Group_Leader=Event.top
end
;
;-----------------------------------------------------------------
;
pro XY_Interpolation, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
ParameterInterpolation,Group_Leader=Event.top
end
;
;-----------------------------------------------------------------
;
pro OnAnchorFiducialMenu, Event     ; Starts Anchor Fiducials Menu Widget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
AnchorWid
end
;
;-----------------------------------------------------------------
;
pro Set_nm_per_pixel_scale, Event
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
Result = DIALOG(/FLOAT,VALUE=nm_per_pixel,TITLE='Set Lateral Scale','Enter a new value for nm/pixel scale')
if (size(Result))[1] eq 4 then nm_per_pixel=Result
end
;
;-----------------------------------------------------------------
;
pro Check_Recenter, Event
	wID = Widget_Info(event.top, find_by_uname='W_MENU_Check_Recenter')
	state=widget_info(wID,/BUTTON_SET)
	switched_state=1-state
	widget_control,wID,set_button=switched_state
end
;
;-----------------------------------------------------------------
;
pro DisplayThisFitCond, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
print,'Thisfitcond:   ',thisfitcond
if n_elements(RawFilenames) gt 0 then print,'RawFilenames:  ',RawFilenames[*]
end
;
;-----------------------------------------------------------------
;
pro Set_Z_scale_multiplier, Event
common Zdisplay, Z_scale_multiplier, vbar_top
Result = DIALOG(/FLOAT,VALUE=Z_scale_multiplier,TITLE='Set Z-scale multiplier','Enter a new value for Z-scale multiplier for (X-Y swapped displays)')
if (size(Result))[1] eq 4 then Z_scale_multiplier=Result
end
;
;-----------------------------------------------------------------
;
pro Set_vbar_top, Event
common Zdisplay, Z_scale_multiplier, vbar_top
Result = DIALOG(/FLOAT,VALUE=vbar_top,TITLE='Set Color Bar Scale','Enter the molecular probability corresponding to v=0.5 of HSV scale (default 0.003 nm^-2)')
if (size(Result))[1] eq 4 then vbar_top=Result
;default vbar_top= molecular probability of 0.003 molecule per nm^2, then vbar_top/(mgW/nm_per_pixel)^2 is molecular probabilty per pixel
; Calculate scale factor so that scale_factor * vbar_top / (mgW/nm_per_pixel)^2 = 0.5
end
;
;-----------------------------------------------------------------
;
pro SetMaxProbability_2DPALM, Event
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
Result = DIALOG(/FLOAT,VALUE=Max_Prob_2DPALM,TITLE='Set Max probability for 2D PALM images','Enter a new value for probability')
if (size(Result))[1] eq 4 then Max_Prob_2DPALM=Result
end
;
;-----------------------------------------------------------------
;
pro OnForce_MaxProbability_2DPALM, Event
wID = Widget_Info(event.top, find_by_uname='W_MENU_Force_MaxProb_2DPALM')
state=widget_info(wID,/BUTTON_SET)
switched_state=1-state
widget_control,wID,set_button=switched_state
end
;
;-----------------------------------------------------------------
;
pro OnIgnoreLblsHist, Event
wID = Widget_Info(event.top, find_by_uname='W_MENU_IgnoreLblsHist')
state=widget_info(wID,/BUTTON_SET)
switched_state=1-state
widget_control,wID,set_button=switched_state
end
;
;-----------------------------------------------------------------
;
pro Adjust_Top_Slider_for_Max_2D_Probability
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
COMMON managed,	ids, names, modalList
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
WidSldTopID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Top')
WidSldGammaID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Gamma')
widget_control,WidSldGammaID,get_value=gamma
sz_im=size(image)
if sz_im[0] eq 2 then begin
	wxsz=1024 & wysz=1024
	mgw=GetScaleFactor(ParamLimits, wxsz, wysz)
	TopV=((round(1000.*(Max_Prob_2DPALM*(nm_per_pixel/mgw)^2/max(image))^(gamma/1000.))) < 1000 ) > 0
	widget_control,WidSldTopID,set_value=TopV
endif


end
;
;-----------------------------------------------------------------
;
pro OnGroupPeaks, Event					; Starts Grouping Menu Widget, perfromes grouping, then refreshes the ParamLimits and displays the grouped peaks
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

GroupWid,Group_Leader=Event.top

end
;
;-----------------------------------------------------------------
;
pro Recalculate_XpkwYpkw, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))					; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
	Recalc_XPkWYPkW_product,Group_Leader=Event.top
	ReloadParamlists, Event, [Par12_ind]
end
;
;-----------------------------------------------------------------
;
pro Reprocess_Palm_Set, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Offset, PkWidth_offset
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

Xwid_ind = min(where(RowNames eq 'X Peak Width'))                        ; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))                        ; CGroupParametersGP[5,*] - Peak Y Gaussian Width
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))                    ; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group
SigZ_ind = min(where(RowNames eq 'Sigma Z'))                            ; CGroupParametersGP[35,*] - Sigma Z
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error



	PkWidth_offset = 1.5
	Par12_limit = 0.05
	SigZ_max = 1000

	; step 1: recalculate XpkwYpkw

	print,'step 1: recalculate XpkwYpkw with offset:',PkWidth_offset
	CGroupParams[Par12_ind,*]=((CGroupParams[Xwid_ind,*]-PkWidth_offset) > 0.0)*((CGroupParams[Ywid_ind,*]-PkWidth_offset)>0.0)

	; step 2: purge Zsigma<750 and XpkwYpkw<0.05
	print,'step 2: purge Zsigma <',SigZ_max,',   and XpkwYpkw <',Par12_limit
	pk_indecis=where(filter and (CGroupParams[SigZ_ind,*] le SigZ_max) and ((CGroupParams[Par12_ind,*] le Par12_limit)),cnt)
	if cnt lt 1 then begin
		z=dialog_message('Filter returned no valid peaks')
		return      ; if data not loaded return
	endif
	CGroupParams = temporary(CGroupParams[*,pk_indecis])

	; step 3: group: cluster
	print,'step 3: regroup'
		CGroupParams[Gr_ind,*]=0
		interrupt_load = 0
		increment = 2500
		grouping_gap = 30
		grouping_radius100 = 20

		framefirst=long(ParamLimits[FrNum_ind,0])
		framelast=long(ParamLimits[FrNum_ind,1])
		nloops=long(ceil((framelast-framefirst+1.0)/increment))
		grouping_radius=FLOAT(grouping_radius100)/100	; in CCD pixel units
		spacer = grouping_gap+2
		maxgrsize = 10						; not absolute max group size: max group size for arrayed processing (groups with elements>maxgroupsize are later analyzed separately)
		disp_increment = 100				; frame interval for progress display
		GroupDisplay = 0					; 0 for cluster, 1 for local
		if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR')	else	idl_pwd=pref_get('IDL_WDE_START_DIR')
		cd,current=curr_pwd

		td = 'temp' + strtrim(ulong(SYSTIME(/seconds)),2)
		temp_dir=curr_pwd + sep + td
		FILE_MKDIR,temp_dir

		save, curr_pwd,idl_pwd, CGrpSize, ParamLimits, increment, nloops, spacer, grouping_radius, maxgrsize, disp_increment, GroupDisplay, RowNames, filename=td + sep + 'temp.sav'		;save variables for cluster cpu access
		spawn,'sh '+idl_pwd+'/group_initialize_jobs.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd+' '+temp_dir			;Spawn grouping workers in cluster

		oStatusBar = obj_new('PALM_StatusBar', $
        	COLOR=[0,0,255], $
        	TEXT_COLOR=[255,255,255], $
        	CANCEL_BUTTON_PRESENT = 1, $
       	 	TITLE='Starting grouped data processing...', $
      		TOP_LEVEL_BASE=tlb)
		fraction_complete_last=0.0D
		pr_bar_inc=0.01D

		nlps = 0L

		while (nlps lt nloops) and (interrupt_load eq 0) do begin
			framestart=	framefirst + (nlps)*increment						;first frame in batch
			framestop=(framefirst + (nlps+1L)*increment-1)<framelast
			GoodPeaks=where((CGroupParams[FrNum_ind,*] ge framestart) and (CGroupParams[FrNum_ind,*] le framestop),OKpkcnt)
			GPmin = GoodPeaks[0]
			GPmax = GoodPeaks[n_elements(GoodPeaks)-1]
			fname_nlps=temp_dir+'/temp'+strtrim(Nlps,2)+'.sav'
			CGroupParamsGP = CGroupParams[*,GPmin:GPmax]	; faster
			save, curr_pwd,idl_pwd, CGroupParamsGP, CGrpSize, ParamLimits, increment, nloops, spacer, grouping_radius,maxgrsize,disp_increment,GroupDisplay,RowNames, filename=fname_nlps		;save variables for cluster cpu access
			wait,0.1
			fraction_complete=FLOAT(nlps)/FLOAT((nloops-1.0))
			if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
				fraction_complete_last=fraction_complete
				oStatusBar -> UpdateStatus, fraction_complete
			endif
			spawn,'sh '+idl_pwd+'/group_start_single_job.sh '+strtrim(nlps,2)+' '+curr_pwd+' '+idl_pwd+' '+temp_dir			;Spawn grouping workers in cluster
			nlps++
			interrupt_load = oStatusBar -> CheckCancel()
		endwhile

		obj_destroy, oStatusBar

		if interrupt_load eq 1 then print,'Grouping aborted, cleaning up...'

		if interrupt_load eq 0 then begin
			GroupPeaksCluster_ReadBack, interrupt_load
		endif
		print,'removing temp directory'
		CATCH, Error_status
		file_delete,td + sep + 'temp.sav'
		file_delete,td
		IF Error_status NE 0 THEN BEGIN
		    PRINT, 'Error index: ', Error_status
			PRINT, 'Error message: ', !ERROR_STATE.MSG
			CATCH,/CANCEL
		ENDIF
		CATCH,/CANCEL
		if interrupt_load eq 1 then print,'Finished cleaning up...'
		cd,curr_pwd

ReloadParamlists, Event

if bridge_exists then begin
	print,'Reloading the Bridge Array'
	CATCH, Error_status
	CGroupParams_bridge = SHMVAR(shmName_data)
	CGroupParams_bridge = CGroupParams
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

end
;
;-----------------------------------------------------------------
;
pro OnAutoCorrelPeak, Event			;AutoCorrelates all found peaks in an interval
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
type='peak'
DoAutoCorrelate,type
end
;
;-----------------------------------------------------------------
;
pro OnAutoCorrelateGroups, Event	;AutoCorrelates all 1st grouped elements
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
type='group'
DoAutoCorrelate,type
end
;
;-----------------------------------------------------------------
;
pro DoAutoCorrelate,type				;Plots autocorrelation using 2nd 1/8th frame interval with autocorr range of 1/32 of Nframes
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
FilterIt
peak_indecis=where(filter,cnt)
if cnt le 1 then return
CGroupParamsGP=CGroupParams[*,peak_indecis]
nframes=long64(max(CGroupParamsGP[FrNum_ind,*])-min(CGroupParamsGP[FrNum_ind,*])+1)
;nframes=long64(max(CGroupParams[FrNum_ind,*]))
frame_start=nframes/8.+min(CGroupParamsGP[FrNum_ind,*])
;frame_start=nframes/8.								;start of autocorrelation
frame_interval=nframes/8							;interval over which autocorr is tested
if frame_interval lt 128 then result=dialog_message('Not enough data points for decent autocorrel plot (N<1100)')
if frame_interval lt 128 then return				; not enough points for decent autocorrelation
frame_stop=frame_start+frame_interval				;end of autocorrelation
loc=bytarr(xydsz[0],xydsz[1],frame_interval+1)		;byte (binary array to indicate peak
framefilter=where((CGroupParamsGP[FrNum_ind,*] ge frame_start) and (CGroupParamsGP[FrNum_ind,*] lt frame_stop),cnt)		;chose peaks in frame interval
;framefilter=where((CGroupParams[FrNum_ind,*] ge frame_start) and (CGroupParams[FrNum_ind,*] lt frame_stop),cnt)		;chose peaks in frame interval
if type eq 'group' then framefilter=framefilter*(CGroupParamsGP[25,*] eq 1)	;chose only first of grouped peaks if flagged
;if type eq 'group' then framefilter=framefilter*(CGroupParams[25,*] eq 1)	;chose only first of grouped peaks if flagged
if (cnt ge 1) and (nframes gt 100) then begin
	loc[CgroupParamsGP[2,framefilter],CGroupParamsGP[3,framefilter],CGroupParamsGP[FrNum_ind,framefilter]-frame_start] = 1	;write pnts
	;loc[CgroupParams[2,framefilter],CGroupParams[3,framefilter],CGroupParams[FrNum_ind,framefilter]-frame_start] = 1	;write pnts
	s=replicate(1,3,3)							;dilation kernal to make slightly bigger target
	dil=dilate(loc,s)							;dilate
	maxmemorysize=5e8							;maximum allowable memory used for the overlap calculation
	maxframes=maxmemorysize/xydsz[0]/xydsz[1]	;corresponding maximum number of frames
	frameshiftrange=(frame_interval/4) < maxframes		;range over which auto correlation works

	overlap=[1.0]								;start array to store auto correlation probablity of data cube
	findj=128									;number of shift intervals used
	arrshiftj=[1.0]								;start array of shift values
	pkcnt=total(loc)							;total number of peaks in interval
	x=(frameshiftrange+1)^(1./findj)			;interval multiplicative factor for each step to get log type ranging spacing
	for i=0,25 do x=(frameshiftrange*(x-1)+1)^(1./findj)			;iterate to converge on right value
	for j=0,findj-1 do begin
		if j eq 0 then arrshiftj = [ fix(x^j+0.4)]						;enter new increment value
		if j gt 0 then arrshiftj = [arrshiftj, fix(x^j+0.4)+arrshiftj[(j-1)>0]]			;enter new increment value
		if j eq 0 then overlap = [total(dil*shift(loc,0,0,-arrshiftj[j]))/pkcnt]		;probability of overlapping pnts vs frame shift
		if j gt 0 then overlap = [overlap, total(dil*shift(loc,0,0,-arrshiftj[j]))/pkcnt]		;probability of overlappng pnts vs frame shift
		if ((j mod 10) eq 5) or (j eq findj-1) then plot,arrshiftj,overlap+0.001,/xlog,/ylog,xrange=[1,frameshiftrange],yrange=[0.01,1],$
								title='Probablity of '+type+' overlap vs frame shift'
		if ((j mod 10) eq 5) or (j eq findj-1) then xyouts,700,930,string(pkcnt/(frame_interval))+' ave '+type+' per frame', /device
		if (j mod 10) eq 5 then wait,0.1
	endfor
endif
end
;
;-----------------------------------------------------------------
;
pro OnSmooth3, Event				;3 point smoothing of displayed data
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
image=smooth(image,3)
onstretchbottom, event
end
;
;-----------------------------------------------------------------
;
pro OnSmooth5, Event				;5 point smoothing of displayed data
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
image=smooth(image,5)
onstretchbottom, event
end
;
;-----------------------------------------------------------------
;
pro OnSmooth9, Event				;9 point smoothing of displayed data
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
image=smooth(image,9)
onstretchbottom, event
end
;
;-----------------------------------------------------------------
;
pro ColorTheTable, Event			;Chose other color tables
tempimage=tvrd()
xloadct,updatecallback='Updateimage',updatecbdata=tempimage		;,/use_current
return
end

PRO Updateimage,data=tempimage
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
tvscl,tempimage
return
end
;
;-----------------------------------------------------------------
;
pro DoToggleZtoHue, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate

wID = Widget_Info(event.top, find_by_uname='W_MENU_54')
original='Z using Color Table -> Hue'
HueName = 'Z using Hue Scale'
widget_control,wID,get_value=test
if test eq original then begin
	widget_control,wID,set_button=1
	widget_control,wID,set_value=HueName
	;ParamLimits[26,0:3]=[0,4,2,4]
	;CGroupParams[26,*]=4
endif
if test eq HueName then begin
	widget_control,wID,set_button=0
	widget_control,wID,set_value=original
	;CGroupParams[26,*]=0
endif
return
end
;
;-----------------------------------------------------------------
;
pro Set_Hist_Log_X, Event
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	hist_log_x = widget_info(Event.ID,/button_set)
end
;
;-----------------------------------------------------------------
;
pro Set_Hist_Log_Y, Event
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	hist_log_y = widget_info(Event.ID,/button_set)
end

;
;-----------------------------------------------------------------
;
pro InsertChar, Event				;Udate paramlimits table and show histogram of edited parameter
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common PreviousStep, PrevParamLimits, PrevGRP
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius

;CATCH, Error_status

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
Chi_ind = min(where(RowNames eq 'ChiSquared'))							; CGroupParametersGP[7,*] - Chi Squared
FitOK_ind = min(where(RowNames eq 'FitOK'))								; CGroupParametersGP[8,*] - Original FitOK
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))					; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigAmp_ind = min(where(RowNames eq 'Sigma Amplitude'))
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))				; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))				; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma
Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number
AmpL1_ind = min(where(RowNames eq 'Amplitude L1'))						; CGroupParametersGP[27,*] - Label 1 Amplitude
AmpL2_ind = min(where(RowNames eq 'Amplitude L2'))						; CGroupParametersGP[28,*] - Label 2 Amplitude
AmpL3_ind = min(where(RowNames eq 'Amplitude L3'))						; CGroupParametersGP[29,*] - Label 3 Amplitude
SigL1_ind = min(where(RowNames eq 'Zeta0'))								; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))						; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))						; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))							; CGroupParametersGP[35,*] - Sigma Z
Coh_ind = min(where(RowNames eq '36 Coherence'))						; CGroupParametersGP[36,*] - Coherence
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

if GrX_ind gt 0 then PALM_with_groups = (ParamLimits[GrSigX_ind,1] gt 0) else PALM_with_groups = 0
allind1 = [Gr_ind, $
		GrX_ind, $
		GrY_ind, $
		GrSigX_ind, $
		GrSigY_ind, $
		GrNph_ind, $
		Gr_size_ind, $
		;GrInd_ind, $
		;LabelSet_ind, $
		GrAmpL1_ind, $
		GrAmpL2_ind, $
		GrAmpL3_ind, $
		GrZ_ind, $
		GrSigZ_ind, $
		GrCoh_ind, $
		Gr_Ell_ind, $
		UnwGrZ_ind, $
		UnwGrZErr_ind]
GroupParInd = allind1[where(allind1 ge 0)]
Pam_is_Grp = (max(where(GroupParInd eq event.y)) ge 0)

widget_control,event.id,get_value=thevalue
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

Recalculate_Histograms_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Redraw')
Recalculate_Histograms=widget_info(Recalculate_Histograms_id,/button_set)

wID = Widget_Info(event.top, find_by_uname='W_MENU_IgnoreLblsHist')
IgnoreLblsHist=widget_info(wID,/BUTTON_SET)

if Recalculate_Histograms then begin
	lbl_min=(LabelSet_ind lt 0) ? 0 : min(CGroupParams[LabelSet_ind,*])
	lbl_max=(LabelSet_ind lt 0) ? 0 : max(CGroupParams[LabelSet_ind,*])
	lbl_cnt=lbl_max-lbl_min+1

	mult_colors_hist = ~ (IgnoreLblsHist or (lbl_cnt eq 1))
	if IgnoreLblsHist then begin
		lbl_cnt = 1
		lbl_max = lbl_min
	endif

	if mult_colors_hist then begin
		device,decompose=0
		loadct,12
	endif
endif


xtitle=RowNames[event.y]
tk=1.5

PrevParamLimits=ParamLimits

ParamLimits[event.y,event.x]=thevalue[event.x,event.y]
CASE event.x OF
	0:	begin
		ParamLimits[event.y,2]=(ParamLimits[event.y,0] + ParamLimits[event.y,1])/2.
		ParamLimits[event.y,3]=(ParamLimits[event.y,1] - ParamLimits[event.y,0])
		end
	1:	begin
		ParamLimits[event.y,2]=(ParamLimits[event.y,0] + ParamLimits[event.y,1])/2.
		ParamLimits[event.y,3]=(ParamLimits[event.y,1] - ParamLimits[event.y,0])
		end
	2:	begin
		ParamLimits[event.y,0]=(ParamLimits[event.y,2] - ParamLimits[event.y,3]/2.)
		ParamLimits[event.y,1]=(ParamLimits[event.y,2] + ParamLimits[event.y,3]/2.)
		end
	3:	begin
		ParamLimits[event.y,0]=(ParamLimits[event.y,2] - ParamLimits[event.y,3]/2.)
		ParamLimits[event.y,1]=(ParamLimits[event.y,2] + ParamLimits[event.y,3]/2.)
		end
	4:	begin
			widget_control,event.id,get_value=whole_table
			peak_settings=whole_table[4,*]
			case event.y of
				FrNum_ind:	begin
					peak_settings[LabelSet_ind]=peak_settings[LabelSet_ind]<max(CGroupParams[LabelSet_ind,*])
					label_ind=where(CGroupParams[LabelSet_ind,*] eq peak_settings[LabelSet_ind],lbl_cnt)
					peak_settings[FrNum_ind]=peak_settings[FrNum_ind]<max(CGroupParams[FrNum_ind,label_ind])
					frame_ind=label_ind[where(CGroupParams[FrNum_ind,label_ind] eq peak_settings[FrNum_ind],frm_cnt)]
					peak_settings[PkInd_ind]=peak_settings[PkInd_ind]<max(CGroupParams[PkInd_ind,frame_ind])
					peak_index=frame_ind[min(where(CGroupParams[PkInd_ind,frame_ind] eq peak_settings[PkInd_ind]))]
					peak_settings[PkGlInd_ind]=CGroupParams[PkGlInd_ind,peak_index]
					end
				PkInd_ind:	begin
					peak_settings[LabelSet_ind]=peak_settings[LabelSet_ind]<max(CGroupParams[LabelSet_ind,*])
					label_ind=where(CGroupParams[LabelSet_ind,*] eq peak_settings[LabelSet_ind],lbl_cnt)
					peak_settings[FrNum_ind]=peak_settings[FrNum_ind]<max(CGroupParams[FrNum_ind,label_ind])
					frame_ind=label_ind[where(CGroupParams[FrNum_ind,label_ind] eq peak_settings[FrNum_ind],frm_cnt)]
					peak_settings[PkInd_ind]=peak_settings[PkInd_ind]<max(CGroupParams[PkInd_ind,frame_ind])
					peak_index=frame_ind[min(where(CGroupParams[PkInd_ind,frame_ind] eq peak_settings[PkInd_ind]))]
					peak_settings[PkGlInd_ind]=CGroupParams[PkGlInd_ind,peak_index]
					end
				PkGlInd_ind:	begin
					peak_settings[LabelSet_ind]=peak_settings[LabelSet_ind]<max(CGroupParams[LabelSet_ind,*])
					label_ind=where(CGroupParams[LabelSet_ind,*] eq peak_settings[LabelSet_ind],lbl_cnt)
					peak_settings[PkGlInd_ind]=peak_settings[PkGlInd_ind]<max(CGroupParams[PkGlInd_ind,label_ind])
					peak_index=label_ind[min(where(CGroupParams[PkGlInd_ind,label_ind] eq peak_settings[PkGlInd_ind]))]
					peak_settings[FrNum_ind]=CGroupParams[FrNum_ind,peak_index]
					peak_settings[PkInd_ind]=CGroupParams[PkInd_ind,peak_index]
					end
				LabelSet_ind:	begin
					peak_settings[LabelSet_ind]=peak_settings[LabelSet_ind]<max(CGroupParams[LabelSet_ind,*])
					label_ind=where(CGroupParams[LabelSet_ind,*] eq peak_settings[LabelSet_ind],lbl_cnt)
					peak_settings[FrNum_ind]=peak_settings[FrNum_ind]<max(CGroupParams[FrNum_ind,label_ind])
					frame_ind=label_ind[where(CGroupParams[FrNum_ind,label_ind] eq peak_settings[FrNum_ind],frm_cnt)]
					peak_settings[PkInd_ind]=peak_settings[PkInd_ind]<max(CGroupParams[PkInd_ind,frame_ind])
					peak_index=frame_ind[min(where(CGroupParams[PkInd_ind,frame_ind] eq peak_settings[PkInd_ind]))]
					peak_settings[PkGlInd_ind]=CGroupParams[PkGlInd_ind,peak_index]
					end
			endcase
			if (size(peak_index))[2] ge 1 then begin
				ReloadPeakColumn,peak_index
				device,decompose=0
				loadct,3
				return
			endif
		end
ENDCASE
if PALM_with_groups then begin
	if event.y eq X_ind then ParamLimits[GrX_ind,*]=ParamLimits[X_ind,*]					;update x,y & group x,y together
	if event.y eq Y_ind then ParamLimits[GrY_ind,*]=ParamLimits[Y_ind,*]
	if event.y eq GrX_ind then ParamLimits[X_ind,*]=ParamLimits[GrX_ind,*]
	if event.y eq GrY_ind then ParamLimits[Y_ind,*]=ParamLimits[GrY_ind,*]
endif
newdisplaypeak = event.x eq 4 and event.y eq PkGlInd_ind
newdisplaygroup= event.x eq 4 and event.y eq Gr_ind


if event.y eq Z_ind then ParamLimits[GrZ_ind,*]=ParamLimits[Z_ind,*]					;update Z, UnwrapZ & Gr. Z, Gr.Unwrap Z together
if event.y eq GrZ_ind then ParamLimits[Z_ind,*]=ParamLimits[GrZ_ind,*]
if event.y eq UnwZ_ind then ParamLimits[UnwGrZ_ind,*]=ParamLimits[UnwZ_ind,*]
if event.y eq UnwGrZ_ind then ParamLimits[UnwZ_ind,*]=ParamLimits[UnwGrZ_ind,*]
if event.y eq Ell_ind then ParamLimits[Gr_Ell_ind,*]=ParamLimits[Ell_ind,*]
if event.y eq Gr_Ell_ind then ParamLimits[Ell_ind,*]=ParamLimits[Gr_Ell_ind,*]

widget_control,event.id,set_value=transpose(ParamLimits[event.y,0:3]), use_table_select=[0,event.y,3,event.y]
if PALM_with_groups then begin
	if event.y eq X_ind then widget_control,event.id,set_value=transpose(ParamLimits[GrX_ind,0:3]), use_table_select=[0,GrX_ind,3,GrX_ind]
	if event.y eq Y_ind then widget_control,event.id,set_value=transpose(ParamLimits[GrY_ind,0:3]), use_table_select=[0,GrY_ind,3,GrY_ind]
	if event.y eq GrX_ind then widget_control,event.id,set_value=transpose(ParamLimits[X_ind,0:3]), use_table_select=[0,X_ind,3,X_ind]
	if event.y eq GrY_ind then widget_control,event.id,set_value=transpose(ParamLimits[Y_ind,0:3]), use_table_select=[0,Y_ind,3,Y_ind]
endif

if event.y eq Z_ind then widget_control,event.id,set_value=transpose(ParamLimits[GrZ_ind,0:3]), use_table_select=[0,GrZ_ind,3,GrZ_ind]
if event.y eq GrZ_ind then widget_control,event.id,set_value=transpose(ParamLimits[Z_ind,0:3]), use_table_select=[0,Z_ind,3,Z_ind]
if event.y eq UnwZ_ind then widget_control,event.id,set_value=transpose(ParamLimits[UnwGrZ_ind,0:3]), use_table_select=[0,UnwGrZ_ind,3,UnwGrZ_ind]
if event.y eq UnwGrZ_ind then widget_control,event.id,set_value=transpose(ParamLimits[UnwZ_ind,0:3]), use_table_select=[0,UnwZ_ind,3,UnwZ_ind]

Recalculate_Histograms_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Redraw')
Recalculate_Histograms=widget_info(Recalculate_Histograms_id,/button_set)
if ~Recalculate_Histograms then return

if PALM_with_groups and Pam_is_Grp then GroupFilterIt else  FilterIt

if newdisplaypeak then 	widget_control,event.id,set_value=transpose(CGroupParams[0:(CGrpSize-1),thevalue[event.x,event.y]]), use_table_select=[4,0,4,(CGrpSize-1)]
if newdisplaygroup then begin
	GroupLoc=where(thevalue[event.x,event.y] eq CGroupParams[Gr_ind,*],cnt)
	if cnt ne -1 then widget_control,event.id,set_value=transpose(CGroupParams[0:(CGrpSize-1),GroupLoc[0]]), use_table_select=[4,0,4,(CGrpSize-1)]
endif

if total(filter) eq 0 then begin
	device,decompose=0
	loadct,3
	return
endif

;nbns=50
nbns = hist_nbins

x=findgen(nbns)/nbns*ParamLimits[event.y,3]+ParamLimits[event.y,0]
dx=ParamLimits[event.y,3]/nbns/2

if PALM_with_groups then xory = event.y eq X_ind or event.y eq Y_ind or event.y eq GrX_ind or event.y eq GrY_ind else xory = event.y eq X_ind or event.y eq Y_ind
zhist = event.y eq Z_ind or event.y eq GrZ_ind or event.y eq UnwZ_ind or event.y eq UnwGrZ_ind

lbl_color = (xory or zhist) ?  [0,200,40,105,0] : [250,200,40,105,250]
hist_count=lonarr(lbl_cnt)
color_cnt = lonarr(lbl_cnt)
histhist_multilable=lonarr(lbl_cnt,2*nbns)
mean_val=fltarr(lbl_cnt)
median_val=fltarr(lbl_cnt)
hist_peak_val=fltarr(lbl_cnt)

if xory or zhist then begin
	xl90_multilable = dblarr(lbl_cnt)
	xl_multilable = dblarr(lbl_cnt)
	xr90_multilable = dblarr(lbl_cnt)
	xr_multilable = dblarr(lbl_cnt)
	histhalf_multilable = dblarr(lbl_cnt)
	search_cnt_multilable= lonarr(lbl_cnt)
	xfwhm_multilable = dblarr(lbl_cnt,2)
	yfwhm_multilable = dblarr(lbl_cnt,2)
endif

DoLogX = (hist_log_x eq 1) and (min(ParamLimits[event.y,0]) gt 0) and ~xory and ~zhist and (ParamLimits[event.y,1] gt ParamLimits[event.y,0])

for lbl_i=lbl_min,lbl_max do begin
	lbl_i0=lbl_i-lbl_min
	if mult_colors_hist then filter_lbl=filter*(CGroupParams[LabelSet_ind,*] eq lbl_i) else filter_lbl=filter
	hist_set=where(filter_lbl,cnt)
	if PALM_with_groups AND Pam_is_Grp then begin
		GrPkIndMin = fix(min(CGroupParams[GrInd_ind,filter_lbl]))
		hist_set=where(filter_lbl*(CGroupParams[GrInd_ind,*] eq GrPkIndMin),cnt)
	endif
		;(((event.y ge Gr_ind) and (event.y lt Gr_size_ind)) or ((event.y ge GrAmpL1_ind) and (event.y le GrCoh_ind)) or ((event.y ge Gr_Ell_ind) and (event.y le UnwGrZErr_ind))) then hist_set=where(filter_lbl*(CGroupParams[GrInd_ind,*] eq 1),cnt)
	print,'label=',lbl_i,'    count=',cnt
	mean_val[lbl_i0] = (cnt gt 0) ? mean(CGroupParams[event.y,hist_set]) : 0
	median_val[lbl_i0] = (cnt gt 0) ? median(CGroupParams[event.y,hist_set]) : 0
	color_cnt[lbl_i0] = cnt
	WidSldFractionHistAnalID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_FractionHistAnal')
	widget_control,WidSldFractionHistAnalID,get_value=fr_search

	if cnt ge 1 then begin
		mn = ParamLimits[event.y,0]
		mx = ParamLimits[event.y,1]
		if mx lt mn then begin
			device,decompose=0
			loadct,3
			return
		endif


		if DoLogX then begin
			;print,'Log X scale'
			mn = alog10((ParamLimits[event.y,0]) >1e-6)
			mx = alog10(ParamLimits[event.y,1])
			binsize=(mx-mn)/(nbns-1.0)
			x=findgen(nbns)/nbns*(mx-mn)+mn
			dx=(mx-mn)/nbns/2
			hist=histogram(alog10(CGroupParams[event.y,hist_set]),min=mn,max=mx,nbins=nbns)
		endif else begin
			;print,'Linear X scale'
			binsize=(mx-mn)/(nbns-1.0)
			hist=histogram(CGroupParams[event.y,hist_set],min=mn,max=mx,nbins=nbns)
		endelse

		hist_max=max(hist,hist_max_bin)
		hist_peak_val[lbl_i0] = x[hist_max_bin]
		xx=fltarr(2*nbns)
		histhist=fltarr(2*nbns)
		evens=2*indgen(nbns)
		odds=evens+1
		xx[evens]=x-dx
		xx[odds]=x+dx
		histhist[evens]=hist
		histhist[odds]=hist
		histhist_multilable[lbl_i0,*]=histhist
		hist_count[lbl_i0]=max(histhist)
		xcoord=xx
		if xory then begin
			xcoord=nm_per_pixel*(xx-xx[0])
			histmax=max(histhist,xmax)
			histhalf=mean(histhist[((xmax-2)>0):((xmax+2)<(n_elements(histhist)-1))])*0.5
			nx=n_elements(xcoord)
			xl=xcoord[0]
			xr=xcoord[nx-1]
			rightsearch=1
			leftsearch=1
			for xi=2,(nx-2) do begin
				if  rightsearch and histhist[nx-xi] lt histhalf and histhist[nx-1-xi] ge histhalf then begin
					rightsearch=0
					xr=xcoord[nx-xi]+(xcoord[nx-1-xi]-xcoord[nx-xi])*$
					(histhalf-histhist[nx-xi])/(histhist[nx-1-xi]-histhist[nx-xi])
				endif
				if  leftsearch and histhist[xi-1] lt histhalf and histhist[xi] ge histhalf then begin
					leftsearch=0
					xl=xcoord[xi-1]+(xcoord[xi]-xcoord[xi-1])*(histhalf-histhist[xi-1])/(histhist[xi]-histhist[xi-1])
				endif
			endfor
			; Search for width at population fraction >= fr_search
			gr_fr=FLOAT(cnt)/100.0*fr_search
			xl90=xl
			xr90=xr
			search90=1
			iter=0
			while search90 do begin
				iter+=1
				xl90 = (xl90 - binsize*nm_per_pixel/10.0) > (mn-xx[0])*nm_per_pixel
				xr90 = (xr90 + binsize*nm_per_pixel/10.0) < (mx-xx[0])*nm_per_pixel
				trash  = where(((CGroupParams[event.y,hist_set]-xx[0])*nm_per_pixel ge xl90) and ((CGroupParams[event.y,hist_set]-xx[0])*nm_per_pixel le xr90),search_cnt)
				if  (search_cnt ge gr_fr) or ((xl90 le (mn-xx[0])*nm_per_pixel) and (xr90 ge (mx-xx[0])*nm_per_pixel)) or (iter ge 300) then search90=0
			endwhile
			x90=[xl90,xr90]
			hist_count[lbl_i0]=max(histhist)
			xfwhm=[xl,xr]
			yfwhm=[histhalf,histhalf]
			hist_count[lbl_i0]=max(histhist)
			xl90_multilable[lbl_i0] = xl90
			xl_multilable[lbl_i0] = xl
			xr90_multilable[lbl_i0] = xr90
			xr_multilable[lbl_i0] = xr
			histhalf_multilable[lbl_i0] = histhalf
			xfwhm_multilable[lbl_i0,*] = transpose(xfwhm)
			yfwhm_multilable[lbl_i0,*] = transpose(yfwhm)
			search_cnt_multilable[lbl_i0] = search_cnt
		endif

		if zhist then begin
			histmax=max(histhist,xmax)
			xcoord=xx
			histhalf=mean(histhist[((xmax-2)>0):((xmax+2)<(n_elements(histhist)-1))])*0.5
			nx=n_elements(xcoord)
			xl=xcoord[0]
			xr=xcoord[nx-1]
			rightsearch=1
			leftsearch=1
			for xi=2,(nx-2) do begin
				if  rightsearch and histhist[nx-xi] lt histhalf and histhist[nx-1-xi] ge histhalf then begin
					rightsearch=0
					xr=xcoord[nx-xi]+(xcoord[nx-1-xi]-xcoord[nx-xi])*$
					(histhalf-histhist[nx-xi])/(histhist[nx-1-xi]-histhist[nx-xi])
				endif
				if  leftsearch and histhist[xi-1] lt histhalf and histhist[xi] ge histhalf then begin
					leftsearch=0
					xl=xcoord[xi-1]+(xcoord[xi]-xcoord[xi-1])*(histhalf-histhist[xi-1])/(histhist[xi]-histhist[xi-1])
				endif
			endfor
			; Search for width at 90% (or different fraction = fr_search)
			gr_fr=cnt/100.0*fr_search
			xl90=xl
			xr90=xr
			search90=1
			iter=0
			while search90 do begin
				iter+=1
				xl90 = (xl90 - binsize/10.0) > mn
				xr90 = (xr90 + binsize/10.0) < mx
				trash = where((CGroupParams[event.y,hist_set] ge xl90) and (CGroupParams[event.y,hist_set] le xr90),search_cnt)
				if  (search_cnt ge gr_fr) or ((xl90 le mn) and (xr90 ge mx)) or (iter ge 300) then search90=0
			endwhile
			x90=[xl90,xr90]
			xfwhm=[xl,xr]
			yfwhm=[histhalf,histhalf]
			hist_count[lbl_i0]=max(histhist)
			xl90_multilable[lbl_i0] = xl90
			xl_multilable[lbl_i0] = xl
			xr90_multilable[lbl_i0] = xr90
			xr_multilable[lbl_i0] = xr
			histhalf_multilable[lbl_i0] = histhalf
			xfwhm_multilable[lbl_i0,*] = transpose(xfwhm)
			yfwhm_multilable[lbl_i0,*] = transpose(yfwhm)
			search_cnt_multilable[lbl_i0] = search_cnt
		endif
	endif
endfor
 if xory then begin
 			mean_val =	mean_val * nm_per_pixel
			median_val = median_val * nm_per_pixel
			hist_peak_val = hist_peak_val * nm_per_pixel
endif
yrange_hist=[0, max(hist_count)*1.1]
xrange_hist = xory ? nm_per_pixel*([ParamLimits[event.y,0],ParamLimits[event.y,1]]-xx[0]) : [ParamLimits[event.y,0],ParamLimits[event.y,1]]

if xory or zhist then begin
units_lb1 = (strpos(xtitle,'Angle') ge 0) ? '' : '  (nm)'
units_txt1 = (strpos(xtitle,'Angle') ge 0) ? '' : ' nm'
	for lbl_i=lbl_min,lbl_max do begin
		lbl_i0=lbl_i-lbl_min
		if (lbl_i eq lbl_min) then begin
			if hist_log_y eq 1 then begin
				nonzero_ind = where(histhist_multilable gt 0)
				yrange_hist = yrange_hist > min(histhist_multilable[nonzero_ind])
				histhist_multilable = histhist_multilable > 0.1
				plot, xcoord, histhist_multilable[lbl_i0,*], background=255, col=0, thick=1.0, xrange=xrange_hist, yrange = yrange_hist, xtitle=RowNames[event.y]+units_lb1, ytitle='Molecule Count', $
							xthick=1.0, ythick=1.0, charsize=tk, charthick=tk, /YLOG
			endif else begin
				plot, xcoord, histhist_multilable[lbl_i0,*], background=255, col=0, thick=1.0, xrange=xrange_hist, yrange = yrange_hist, xtitle=RowNames[event.y]+units_lb1, ytitle='Molecule Count', $
							xthick=1.0, ythick=1.0, charsize=tk, charthick=tk
			endelse
		endif
		if lbl_cnt gt 1 then oplot,xcoord,histhist_multilable[lbl_i0,*], color=lbl_color[lbl_i]
		oplot,xfwhm_multilable[lbl_i0,*],yfwhm_multilable[lbl_i0,*],psym=6,color=lbl_color[lbl_i]
		oplot,xfwhm_multilable[lbl_i0,*],yfwhm_multilable[lbl_i0,*],color=lbl_color[lbl_i]
		x90=[xl90_multilable[lbl_i0],xr90_multilable[lbl_i0]]
		oplot,x90,yfwhm_multilable[lbl_i0,*]*0.7,psym=6,color=lbl_color[lbl_i]
		oplot,x90,yfwhm_multilable[lbl_i0,*]*0.7,color=lbl_color[lbl_i]
		xyouts,0.12,0.95-0.05*lbl_i0,'HalfMax = '+strtrim(histhalf_multilable[lbl_i0],2)+'    FWHM = ' + $
			strtrim(xr_multilable[lbl_i0]-xl_multilable[lbl_i0],2) + units_txt1,	color=lbl_color[lbl_i],charsize=1.5,CHARTHICK=1.5,/NORMAL
		xyouts,0.12,0.93-0.05*lbl_i0,strtrim(fix(fr_search),2)+'% population = '+ strtrim(ulong(color_cnt[lbl_i0]),2) + $
				',  '+ strtrim(ulong(search_cnt_multilable[lbl_i0]),2) + ' are within range = ' + $
				strtrim((xr90_multilable[lbl_i0]-xl90_multilable[lbl_i0]),2)+units_txt1, $
				color=lbl_color[lbl_i], charsize=1.5,CHARTHICK=1.5,/NORMAL
		xyouts,0.12,0.91-0.05*lbl_i0,'Mean = '+ strtrim(mean_val[lbl_i0],2) + $
				'       Median = '+ strtrim(median_val[lbl_i0],2) + $
				'       Hist. Peak = '+ strtrim(hist_peak_val[lbl_i0],2),$
				color=lbl_color[lbl_i], charsize=1.5,CHARTHICK=1.5,/NORMAL
	endfor
endif else begin
	for lbl_i=lbl_min,lbl_max do begin
		lbl_i0=lbl_i-lbl_min
		if lbl_i eq lbl_min then begin
			if DoLogX then begin
				;print,'Plot Log X Scale'
				mnl = floor(mn)
				mxl = ceil(mx)
				xrange_hist = [mnl,mxl]
				for il = mnl,(mxl-1) do begin
					if il eq mnl then begin
						xtickv = alog10(10.0^il*(1+ findgen(9)))
					endif else begin
						xtickv = [xtickv, alog10(10.0^il*(1+ findgen(9)))]
					endelse
				endfor
				xtickv = [xtickv, mxl]
				xticknames = strtrim(round(10^xtickv),2)
				xticks = n_elements(xtickv)
				xticks_int = xticks-1
				for i = 0, xticks_int do if ((i mod 9) ne 0) then xticknames[i] = ' '
				print,xticknames
				if (hist_log_y eq 1) then begin
					nonzero_ind = where(histhist_multilable gt 0)
					yrange_hist = yrange_hist > min(histhist_multilable[nonzero_ind])
					histhist_multilable = histhist_multilable > 0.1
					;print,'y-range: ',yrange_hist
					plot,xx,histhist_multilable[lbl_i0,*],xstyle=1, xtitle=RowNames[event.y], ytitle='Molecule Count', $
								thick=1.0, xthick=1.0, ythick=1.0, charsize=tk, charthick=tk, xrange = xrange_hist, $
								xticks = xticks_int, xtickv = xtickv, xtickname = xticknames, yrange = yrange_hist, /YLOG
				endif else begin
					plot,xx,histhist_multilable[lbl_i0,*],xstyle=1, xtitle=RowNames[event.y], ytitle='Molecule Count', $
								thick=1.0, xthick=1.0, ythick=1.0, charsize=tk, charthick=tk, xrange = xrange_hist, $
								xticks = xticks_int, xtickv = xtickv, xtickname = xticknames, yrange = yrange_hist
				endelse
			endif else begin
				if (hist_log_y eq 1) then begin
					nonzero_ind = where(histhist_multilable gt 0)
					yrange_hist = yrange_hist > min(histhist_multilable[nonzero_ind])
					;print,'y-range: ',yrange_hist
					histhist_multilable = histhist_multilable > 0.1
					plot,xx,histhist_multilable[lbl_i0,*],xstyle=1, xtitle=RowNames[event.y], ytitle='Molecule Count', $
								thick=1.0, xthick=1.0, ythick=1.0, charsize=tk, charthick=tk, xrange = xrange_hist, yrange = yrange_hist, /YLOG
				endif else begin
					plot,xx,histhist_multilable[lbl_i0,*],xstyle=1, xtitle=RowNames[event.y], ytitle='Molecule Count', $
								thick=1.0, xthick=1.0, ythick=1.0, charsize=tk, charthick=tk, xrange = xrange_hist, yrange = yrange_hist
				endelse
			endelse
		endif
		if lbl_cnt gt 1 then oplot,xx,histhist_multilable[lbl_i0,*], color=lbl_color[lbl_i]
		hpv = hist_peak_val[lbl_i0]
		if (total(hist_set) ge 0) then if (hist_log_x eq 1) and (min(CGroupParams[event.y,hist_set]) ge 0) and ~xory and ~zhist then hpv = 10^hpv
		xyouts,0.12,0.95-0.05*lbl_i0,'Mean = '+ strtrim(mean_val[lbl_i0],2) + $
				'       Median = '+ strtrim(median_val[lbl_i0],2) + $
				'       Hist. Peak = '+ strtrim(hpv,2),$
				color=lbl_color[lbl_i], charsize=1.5,CHARTHICK=1.5,/NORMAL
	endfor
endelse

if mult_colors_hist then begin
	device,decompose=0
	loadct,3
endif

if event.y eq UnwGrZErr_ind or event.y eq UnwZErr_ind then begin	; if unwrap Z error, fit it to Gaussian
	UnwrZ_Err_index=event.y
	ZMin=mn
	ZMax=mx
	redrawhist=0
	Estimate_Unwrap_Ghost, UnwrZ_Err_index, hist_set, ZMin, ZMax, nbns, gres, redrawhist,1
endif

Save_Histograms_BMP_ID = Widget_Info(event.top, find_by_uname='W_MENU_Save_Histograms_BMP')
Save_Histograms_BMP_state=widget_info(Save_Histograms_BMP_ID,/BUTTON_SET)

if Save_Histograms_BMP_state then begin
	Histograms_BMP_File=AddExtension(RawFilenames[0]+'_'+xtitle,'.bmp')
	presentimage=tvrd(true=1)
	write_bmp,Histograms_BMP_File,presentimage,/rgb
endif

;IF Error_status NE 0 THEN BEGIN
;	PRINT, 'InsertChar: Error :',!ERROR_STATE.MSG
;ENDIF
;CATCH,/CANCEL

end
;
;-----------------------------------------------------------------
;
pro Estimate_Unwrap_Ghost, UnwrZ_Err_index, hist_set, ZMin, ZMax, nbns, gres, redrawhist, disp; disp=0 - no reporting, disp=1 - full reporting and plotting
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
	tk=1.5
	Zrange=Zmax-Zmin
	x=findgen(nbns)/nbns*Zrange+Zmin
	dx=Zrange/nbns/2
	binsize=(Zmax-Zmin)/(nbns-1.0)
	hist=histogram(CGroupParams[UnwrZ_Err_index,hist_set],min=Zmin,max=Zmax,nbins=nbns)
	hist_max=max(hist,hist_max_bin)
	;hist_peak_val[lbl_i0] = x[hist_max_bin]
	xx=fltarr(2*nbns)
	histhist=fltarr(2*nbns)
	evens=2*indgen(nbns)
	odds=evens+1
	xx[evens]=x-dx
	xx[odds]=x+dx
	histhist[evens]=hist
	histhist[odds]=hist
	xcoord=xx
	xx1=xx[1:(n_elements(xx)-2)]
	YY=histhist[1:(n_elements(xx)-2)]
	CATCH, Error_status
	IF Error_status NE 0 THEN BEGIN
		z=dialog_message('Could not perfrom Gaussian Fitting')
		CATCH, /CANCEL
	ENDIF else begin
		maxYY=max(YY,max_ind)
		gest=[maxYY,xx1[max_ind],wind_range[0]/4.0,wind_range[0]]
		a=gest
		fita=[1,1,1,0]
		result=lmfit(xx1,YY,a,fita=fita,FUNCTION_NAME='UnwrapError',/DOUBLE)
		gres=a
		if disp then print,'Gauss Est.:  ',gest
		if disp then print,'Gauss Fit :  ',gres
		xxfit =	findgen(2*fix(wind_range[0]))-fix(wind_range[0])
		YYfit0 = UnwrapError(xxfit,gres)
		xxfit_neg = xxfit-wind_range[0]
		xxfit_pos = xxfit+wind_range[0]
		yyfit = gres[0]*exp(-0.5*((xxfit-gres[1])/gres[2])^2)
		truepeaks = total(yyfit[where((xxfit le Zmax) and (xxfit ge Zmin))])
		leftghosts = total(yyfit[where((xxfit_neg le Zmax) and (xxfit_neg ge Zmin))])
		rightghosts = total(yyfit[where((xxfit_pos le Zmax) and (xxfit_pos ge Zmin))])
		GhostRatio = (leftghosts+rightghosts) / (truepeaks+leftghosts+rightghosts)

		if disp then begin
			if redrawhist then begin
				yrange_hist=[0, max(hist)*1.1]
				plot, xcoord, histhist, thick=1.0, xrange=[Zmin,Zmax], yrange = yrange_hist, xtitle=RowNames[UnwrZ_Err_index]+'  (nm)', ytitle='Molecule Count', $
							xthick=1.0, ythick=1.0, charsize=tk, charthick=tk
			endif

			oplot,xxfit,yyfit0, color=200
			oplot,xxfit,yyfit, color=100
			oplot,xxfit_neg,yyfit, color=200
			oplot,xxfit_pos,yyfit, color=200
			xyouts,0.12,0.92,'Gauss. width = '+ strtrim(gres[2],2) + '       Gauss.Ampl = '+ strtrim(gres[0],2)+'       Center = '+ strtrim(gres[1],2), $
				color=100, charsize=1.5,CHARTHICK=1.5,/NORMAL
			xyouts,0.12,0.89,'Estimated Ghost Ratio = '+ strtrim(GhostRatio,2), color=200, charsize=1.5,CHARTHICK=1.5,/NORMAL
		endif
	endelse
end
;
;-----------------------------------------------------------------
;
pro Export_Hist_ASCII, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common PreviousStep, PrevParamLimits, PrevGRP
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

LabelSet_ind = min(where(RowNames eq 'Label Set'))                        ; CGroupParametersGP[26,*] - Label Number

filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.txt'],title='Writes histogram data into *.txt ascii file')

if strlen(fpath) ne 0 then cd,fpath
if filename eq '' then return
HistFile=AddExtension(filename,'.txt')

lbl_cnt=max(CGroupParams[LabelSet_ind,*])-min(CGroupParams[LabelSet_ind,*])+1

openw,1,HistFile,width=1024
header=xtitle
for i=1,lbl_cnt do header=header+string(9B)+'Molecule Count Label #'+strtrim(fix(i),2)
printf,1,header

xc = hist_log_x ?	10^(xcoord)	: xcoord
if hist_log_y then begin
	if lbl_cnt gt 1 then begin
		zero_ind = where(histhist_multilable le 0.5, cnt_zeros)
		if cnt_zeros ge 1 then histhist_multilable[zero_ind]=0
	endif else begin
		zero_ind = where(histhist le 0.5, cnt_zeros)
		if cnt_zeros ge 1 then histhist[zero_ind]=0
	endelse
endif

if lbl_cnt gt 1 then printf,1, [transpose(xc), (histhist_multilable)],FORMAT='('+strtrim(fix(lbl_cnt),2)+'(E12.4,"'+string(9B)+'"),E12.4)' else $
printf,1, [transpose(xc), transpose(histhist)],FORMAT='(E12.4,"'+string(9B)+'",E12.4)'

close,1
end
;
;-----------------------------------------------------------------
pro OnSaveHistBMP, Event
wID = Widget_Info(event.top, find_by_uname='W_MENU_Save_Histograms_BMP')
state=widget_info(wID,/BUTTON_SET)
switched_state=1-state
widget_control,wID,set_button=switched_state
end

;
;-----------------------------------------------------------------
;
pro OnPeakCentersButton, Event		;Display peak center points for present xpixel, ypixel range
common PreviousStep, PrevParamLimits, PrevGRP
GRP = 0
PrevGRP=GRP
DrawCenters, Event,GRP
end
;
;-----------------------------------------------------------------
;
pro OnGroupCentersButton, Event		;Display group center points for present xpixel, ypixel range
common PreviousStep, PrevParamLimits, PrevGRP
GRP = 1
PrevGRP=GRP
DrawCenters, Event,GRP
end
;
;-----------------------------------------------------------------
;
pro DrawCenters, Event, GRP			;Does the display work for On(Peak/Group)CentersButton
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number

XZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapXZ')
XZ_swapped=Widget_Info(XZ_swap_menue_ID,/button_set)

YZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapYZ')
YZ_swapped=Widget_Info(YZ_swap_menue_ID,/button_set)

if XZ_swapped then begin
	X_ind = min(where(RowNames eq 'Z Position'))
	GrX_ind = min(where(RowNames eq 'Group Z Position'))
endif
if YZ_swapped then begin
	Y_ind = min(where(RowNames eq 'Z Position'))
	GrY_ind = min(where(RowNames eq 'Group Z Position'))
endif


if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

if Event.top eq TopID then begin						; if called from the main menu, the image size is 1024x1024, otherwise Cust_TIFF_Pix_X x Cust_TIFF_Pix_Y
	wxsz=1024 & wysz=1024								; size of the display window
endif else begin
	wxsz=Cust_TIFF_Pix_X
	wysz=Cust_TIFF_Pix_Y
endelse

dxmn = paramlimits[X_ind,0]
dymn = paramlimits[Y_ind,0]
dxmx = paramlimits[X_ind,1]
dymx = paramlimits[Y_ind,1]
loc=fltarr(wxsz,wysz)
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))


lbl_color = [250,200,40,105,250]; label colors, color table 12 (16 level)
								; zero label - white (250)
								; first label - red (200)
								; second label - green (40)
								; third label - blue (105)
								; fourth label - white (250)

;print,'using colors', lbl_color

wID = Widget_Info(TopID, find_by_uname='W_MENU_59')
original='Render Without Legends'
modified = 'Do not plot Bars/Legends'
widget_control,wID,get_value=DrawLegends

FilterIt
FilteredPeakIndex=where(filter,cnt0)

if GRP eq 0 then begin
	if (cnt0 eq 0) then begin
		print,'no peaks inside the window '
		;OnUnZoomButton, Event					;If no peaks inside dragged window then unzoom
		return									;If no peaks inside dragged window then return
	endif
	loc[[mgw*(CGroupParams[X_ind,FilteredPeakIndex]-dxmn)],[mgw*(CGroupParams[Y_ind,FilteredPeakIndex]-dymn)]]=255.
	mx = (LabelSet_ind lt 0) ? 0 : max(CGroupParams[LabelSet_ind,FilteredPeakIndex])								; are there more than one label set of data,... then show in R,G,B
endif
if GRP eq 1 then begin
	filter0 = filter
	GroupFilterIt
	filterGRP = filter
	FilteredGroupedPeakIndex=where(filterGRP,cnt)
	if (cnt eq 0) then begin
		print,'no groups inside the dragged window'
		;OnUnZoomButton, Event				;If no groups inside dragged window then unzoom
		return
	endif
	if cnt0 ge 1 then loc[[mgw*(CGroupParams[X_ind,FilteredPeakIndex]-dxmn)],[mgw*(CGroupParams[Y_ind,FilteredPeakIndex]-dymn)]]=150
	if cnt ge 1 then loc[[mgw*(CGroupParams[GrX_ind,FilteredGroupedPeakIndex]-dxmn)],[mgw*(CGroupParams[GrY_ind,FilteredGroupedPeakIndex]-dymn)]]=255
	if cnt ge 1 then mx=max(CGroupParams[LabelSet_ind,FilteredGroupedPeakIndex]) else mx=-1					; are there more than one label set of data,... then show in R,G,B
endif

;print,'mx=',mx
if (mx lt 1)  then begin
	tv,loc
	if DrawLegends eq original then TVscales,wxsz,wysz,mgw,nm_per_pixel
endif else begin
	loc=fltarr(wxsz,wysz)
	loadct,12		; load color table of discrete colors
	loc=intarr(wxsz,wysz)
	for i=1,mx do begin
		LabelIndices=where((filter eq 1) and (CGroupParams[LabelSet_ind,*] eq i),lbl_cnt)
		if lbl_cnt ge 1 then begin
			print,'label:',i,',    number of peaks:',n_elements(LabelIndices),',       color:',lbl_color[i]
			xpk = mgw * (reform(CGroupParams[X_ind,LabelIndices])-dxmn)
			ypk = mgw * (reform(CGroupParams[Y_ind,LabelIndices])-dymn)
			loc[xpk,ypk]=lbl_color[i]
		endif
	endfor
	tv,loc
	loadct,3
	if DrawLegends eq original then TVscales,wxsz,wysz,mgw,nm_per_pixel
endelse
end
;
;-----------------------------------------------------------------
;
pro OnTotalRawDataButton, Event		;Display raw data for present xpixel, ypixel range
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

if n_elements(TotalRawData) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	PRINT, 'Error during TotalRaw rendering:',!ERROR_STATE.MSG
	CATCH,/CANCEL
	return
ENDIF
if Event.top eq TopID then begin						; if called from the main menu, the image size is 1024x1024, otherwise Cust_TIFF_Pix_X x Cust_TIFF_Pix_Y
	wxsz=1024 & wysz=1024								; size of the display window
endif else begin
	wxsz=Cust_TIFF_Pix_X
	wysz=Cust_TIFF_Pix_Y
endelse

dxmn = paramlimits[2,0]
dymn = paramlimits[3,0]
dxmx = paramlimits[2,1]
dymx = paramlimits[3,1]
loc=fltarr(wxsz,wysz)
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))

newx=(dxmx-dxmn)*mgw
newy=(dymx-dymn)*mgw

dxmn_f=floor(dxmn)>0
dymn_f=floor(dymn)>0
dxmx_f=(ceil(dxmx)+1)<(xydsz[0]-1)
dymx_f=(ceil(dymx)+1)<(xydsz[1]-1)

dx=round((dxmx_f-dxmn_f+1)*mgw)
dy=round((dymx_f-dymn_f+1)*mgw)

FnewImage=Congrid(TotalRawData[dxmn_f:dxmx_f,dymn_f:dymx_f],dx,dy)
lim=size(FnewImage)

dxmn_l=(dxmn-dxmn_f)*mgw>0
dymn_l=(dymn-dymn_f)*mgw>0
dxmx_l=(dxmn_l+newx-1)<(lim[1]-1)
dymx_l=(dymn_l+newy-1)<(lim[2]-1)

Fimage=FnewImage[dxmn_l:dxmx_l,dymn_l:dymx_l]
sizef=size(fimage)

xp=round(-1*dxmn*mgw>0)
yp=round(-1*dymn*mgw>0)
if xp gt 0 then begin
	Fimage1=intarr((sizef[1]+xp),sizef[2])
	Fimage1[xp:(sizef[1]+xp-1),*]=Fimage
	Fimage=Fimage1
	sizef=size(fimage)
endif
if yp gt 0 then begin
	Fimage1=intarr(sizef[1],(sizef[2]+yp))
	Fimage1[*,yp:(sizef[2]+yp-1)]=Fimage
	Fimage=Fimage1
endif

tv,bytarr(wxsz,wysz)
tvscl,Fimage
image=fimage				;tvrd(true=1)

AdjustContrastnDisplay, Event

wID = Widget_Info(event.top, find_by_uname='W_MENU_59')
original='Render Without Legends'
modified = 'Do not plot Bars/Legends'
widget_control,wID,get_value=DrawLegends
if DrawLegends eq original then TVscales,wxsz,wysz,mgw,nm_per_pixel

CATCH,/CANCEL

end
;
;-----------------------------------------------------------------
;
pro Replace_TotalRaw_with_Rendered, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

if n_elements(TotalRawData) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

CATCH, Error_status
IF Error_status NE 0 THEN BEGIN
	PRINT, 'Error during replacing TotalRaw:',!ERROR_STATE.MSG
	CATCH,/CANCEL
	return
ENDIF

presentimage=tvrd()

TR_size=size(TotalRawData)

TotalRawData = congrid(presentimage,TR_size[1],TR_size[2])

CATCH,/CANCEL
end
;
;-----------------------------------------------------------------
;
pro OverlayDIC, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w

if n_elements(DIC) le 1 then begin
	z=dialog_message('DIC/EM image not loaded')
	return
endif
present_image=tvrd(true=3)

DICx=(size(DIC))[1] & DICy=(size(DIC))[2]
wxsz=1024 & wysz=1024
dxmn = paramlimits[2,0]
dymn = paramlimits[3,0]
dxmx = paramlimits[2,1]
dymx = paramlimits[3,1]
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))
rmg=FLOAT(max((size(DIC))[1:2])) / (DICx > DICy)		; real magnification

XI=fix(floor(rmg*dxmn))>0
XA=fix(floor(rmg*(dxmx))) < (DICx - 1)
YI=fix(floor(rmg*dymn)) > 0
YA=fix(floor(rmg*(dymx))) < (DICy - 1)

botV=labelContrast[2,4]/1000.
gamma=labelContrast[1,4]/1000.
topV=labelContrast[1,4]/1000.

Fimage=DIC[XI : XA, YI : YA] ^gamma
;rng=(Max(Fimage)-Min(Fimage))
Fimage= ((Fimage - botV*Max(Fimage))>0) < topV*Max(Fimage)
Fimage=Fimage/max(Fimage)*255
fimagex=fix(float(XA-XI+1)*mgw/rmg)
fimagey=fix(float(YA-YI+1)*mgw/rmg)
Fimage=Congrid(Fimage,fimagex,fimagey)
xs=(fimagex-1) < ((size(present_image))[1]-1)
ys=(fimagey-1) < ((size(present_image))[2]-1)

tv,bytarr(wxsz,wysz)

Overl_Image=present_image
Overl_Image[0:xs,0:ys,0]=Overl_Image[0:xs,0:ys,0]/2.0+Fimage[0:xs,0:ys]/2.0
Overl_Image[0:xs,0:ys,1]=Overl_Image[0:xs,0:ys,1]/2.0+Fimage[0:xs,0:ys]/2.0
Overl_Image[0:xs,0:ys,2]=Overl_Image[0:xs,0:ys,2]/2.0+Fimage[0:xs,0:ys]/2.0

tvscl,Overl_Image,true=3

end
;
;-----------------------------------------------------------------
;
pro OnUnZoomButton, Event			;Zoom out to show full range of data
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common PreviousStep, PrevParamLimits, PrevGRP
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets

if n_elements(CGroupParams) le 2 then begin
          z=dialog_message('Please load a data file')
          return      ; if data not loaded return
endif

X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma


TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
XZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapXZ')
XZ_swapped=Widget_Info(XZ_swap_menue_ID,/button_set)
YZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapYZ')
YZ_swapped=Widget_Info(YZ_swap_menue_ID,/button_set)

if XZ_swapped then begin
	X_ind = min(where(RowNames eq 'Z Position'))
	GrX_ind = min(where(RowNames eq 'Group Z Position'))
endif
if YZ_swapped then begin
	Y_ind = min(where(RowNames eq 'Z Position'))
	GrY_ind = min(where(RowNames eq 'Group Z Position'))
endif

if GrX_ind gt 0 then PALM_with_groups = (ParamLimits[GrSigX_ind,1] gt 0) else PALM_with_groups = 0

if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

if (size(ParamLimits))[X_ind] eq 0 then return
PrevParamLimits=ParamLimits
;xmin = min(CGroupParams[X_ind,*])>0.0
xmin=0.0 & ymin=0.0
xmax=xydsz[0] & ymax=xydsz[1]
;xmax = xmin + xydsz[0] - 1
;ymin = min(CGroupParams[Y_ind,*])>0.0
;ymax = ymin + xydsz[1] - 1
ParamLimits[X_ind,0]=xmin
ParamLimits[X_ind,1]=xmax
ParamLimits[X_ind,2]=(xmin+xmax)/2.
ParamLimits[X_ind,3]=(xmax-xmin-1)
ParamLimits[Y_ind,0]=ymin
ParamLimits[Y_ind,1]=ymax
ParamLimits[Y_ind,2]=(ymin+ymax)/2.
ParamLimits[Y_ind,3]=(ymax-ymin-1)
if PALM_with_groups then begin
	ParamLimits[GrX_ind,*]=ParamLimits[X_ind,*]
	ParamLimits[GrY_ind,*]=ParamLimits[Y_ind,*]
endif
wtable = Widget_Info(Event.Top, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(ParamLimits[X_ind:Y_ind,0:3]), use_table_select=[0,X_ind,3,Y_ind]
if PALM_with_groups then begin
	widget_control,wtable,set_value=transpose(ParamLimits[GrX_ind:GrY_ind,0:3]), use_table_select=[0,GrX_ind,3,GrY_ind]
endif
OnPeakCentersButton, Event

end
;
;-----------------------------------------------------------------
;
pro OnUnZoom2X, Event			;  Increase the displayed field size by 2X in X- and Y- directions
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common PreviousStep, PrevParamLimits, PrevGRP
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
if (size(ParamLimits))[2] le 0 then return
PrevParamLimits = ParamLimits
XI = ParamLimits[X_ind,0]
XA = ParamLimits[X_ind,1]
YI = ParamLimits[Y_ind,0]
YA = ParamLimits[Y_ind,1]
DX=XA-XI
DY=YA-YI
ParamLimits[X_ind,0] = XI - DX/2.0
ParamLimits[X_ind,1] = XA + DX/2.0
ParamLimits[X_ind,3] = ParamLimits[X_ind,1] - ParamLimits[X_ind,0]
ParamLimits[Y_ind,0] = YI - DY/2.0
ParamLimits[Y_ind,1] = YA + DY/2.0
ParamLimits[Y_ind,3] = ParamLimits[Y_ind,1] - ParamLimits[Y_ind,0]

wtable = Widget_Info(Event.Top, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(ParamLimits[X_ind:Y_ind,0:3]), use_table_select=[0,X_ind,3,Y_ind]
if CGrpSize gt 27 then begin
	ParamLimits[GrX_ind,*] = ParamLimits[X_ind,*]
	ParamLimits[GrY_ind,*] = ParamLimits[Y_ind,*]
	widget_control,wtable,set_value=transpose(ParamLimits[GrX_ind:GrY_ind,0:3]), use_table_select=[0,GrX_ind,3,GrY_ind]
endif
if (size(PrevGRP))[2] le 0 then GRP=0 else GRP=PrevGRP
print,GRP
DrawCenters, Event,GRP
end
;
;-----------------------------------------------------------------
;
pro On1StepBack, Event			;  Re-loads pervious ParamLimits and re-draws frame or group peaks
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common PreviousStep, PrevParamLimits, PrevGRP
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
if (size(PrevParamLimits))[2] le 0 then return
ParamLimits=PrevParamLimits
if (size(PrevGRP))[2] le 0 then GRP=0 else GRP=PrevGRP
sz=size(CGroupParams)
wtable = Widget_Info(Event.Top, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(ParamLimits[0:(CGrpSize-1),0:3]), use_table_select=[0,0,3,(CGrpSize-1)]
if sz[0] ge 2 then widget_control,wtable,set_value=transpose(CGroupParams[0:(CGrpSize-1),sz[2]/2]), use_table_select=[4,0,4,(CGrpSize-1)]
widget_control, wtable, /editable,/sensitive
DrawCenters, Event,GRP
end
;
;-----------------------------------------------------------------
;
pro OnButtonDraw0, Event			;Resets x,y pixel range for click drag operation in data window
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common PreviousStep, PrevParamLimits, PrevGRP
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma

if GrX_ind gt 0 then PALM_with_groups = (ParamLimits[GrSigX_ind,1] gt 0) else PALM_with_groups = 0

if event.type eq 0 then begin		; event.type=0: Button pressed - enter first coordinates of the selected box
	b_set=float([event.x,event.y])
	return
endif
if (size(b_set))[0] le 0 then return

wxsz=1024 & wysz=1024
dxmn = paramlimits[X_ind,0]
dymn = paramlimits[Y_ind,0]
dxmx = paramlimits[X_ind,1]
dymx = paramlimits[Y_ind,1]
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))

if event.type eq 1 then begin		; event.type=1: Button released - enter second coordinates of the selected box
	PrevParamLimits = ParamLimits
	b_rel=float([event.x,event.y])
	xmin= b_set[0] < b_rel[0]
	xmax= (b_set[0] > b_rel[0]) > (xmin + 2.)
	ymin= b_set[1] < b_rel[1]
	ymax= (b_set[1] > b_rel[1]) > (ymin + 2.)
	oldx=[paramlimits[X_ind,0],paramlimits[X_ind,1]]
	oldy=[paramlimits[Y_ind,0],paramlimits[Y_ind,1]]

	ParamLimits[X_ind,0]=oldx[0]+float(xmin)/mgw
	ParamLimits[X_ind,1]=oldx[0]+float(xmax)/mgw
	ParamLimits[X_ind,2]=(ParamLimits[X_ind,0]+ParamLimits[X_ind,1])/2.
	ParamLimits[X_ind,3]=(ParamLimits[X_ind,1]-ParamLimits[X_ind,0])
	ParamLimits[Y_ind,0]=oldy[0]+float(ymin)/mgw
	ParamLimits[Y_ind,1]=oldy[0]+float(ymax)/mgw
	ParamLimits[Y_ind,2]=(ParamLimits[Y_ind,0]+ParamLimits[Y_ind,1])/2.
	ParamLimits[Y_ind,3]=(ParamLimits[Y_ind,1]-ParamLimits[Y_ind,0])
	if PALM_with_groups then begin
		ParamLimits[GrX_ind,*]=ParamLimits[X_ind,*]
		ParamLimits[GrY_ind,*]=ParamLimits[Y_ind,*]
	endif

	wtable = Widget_Info(Event.Top, find_by_uname='WID_TABLE_0')
	widget_control,wtable,set_value=transpose(ParamLimits[X_ind:Y_ind,0:3]), use_table_select=[0,X_ind,3,Y_ind]
	if PALM_with_groups then widget_control,wtable,set_value=transpose(ParamLimits[GrX_ind:GrY_ind,0:3]), use_table_select=[0,GrX_ind,3,GrY_ind]
	OnPeakCentersButton, Event
endif
end
;
;-----------------------------------------------------------------
;
pro OnDraw0Realize, wWidget			;Set initial color scale on creating viewing window
device,decompose=0
loadct,3
end
;
;-----------------------------------------------------------------
;
pro OnPlotXYButton, Event			;On button press plots x,y plot (z color) with axis slected by x,y (z) droplists settings
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
tk=1.5
Xid=widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_X')
Xitem=widget_info(Xid,/DropList_Select)
Yid=widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Y')
Yitem=widget_info(Yid,/DropList_Select)
Zid=widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Z')
Zitem=widget_info(Zid,/DropList_Select)

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position
Xwid_ind = min(where(RowNames eq 'X Peak Width'))						; CGroupParametersGP[4,*] - Peak X Gaussian Width
Ywid_ind = min(where(RowNames eq 'Y Peak Width'))						; CGroupParametersGP[5,*] - Peak Y Gaussian Width
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
Chi_ind = min(where(RowNames eq 'ChiSquared'))							; CGroupParametersGP[7,*] - Chi Squared
FitOK_ind = min(where(RowNames eq 'FitOK'))								; CGroupParametersGP[8,*] - Original FitOK
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))
PkGlInd_ind = min(where(RowNames eq 'Peak Global Index'))
Par12_ind = min(where(RowNames eq '12 X PkW * Y PkW'))					; CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
SigAmp_ind = min(where(RowNames eq 'Sigma Amplitude'))
SigNphX_ind = min(where(RowNames eq 'Sigma X Pos rtNph'))				; CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
SigNphY_ind = min(where(RowNames eq 'Sigma Y Pos rtNph'))				; CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma

Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))				; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))				; CGroupParametersGP[22,*] - new y - position sigma
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
Gr_size_ind = min(where(RowNames eq '24 Group Size'))					; CGroupParametersGP[24,*] - total peaks in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group

LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number

AmpL1_ind = min(where(RowNames eq 'Amplitude L1'))						; CGroupParametersGP[27,*] - Label 1 Amplitude
AmpL2_ind = min(where(RowNames eq 'Amplitude L2'))						; CGroupParametersGP[28,*] - Label 2 Amplitude
AmpL3_ind = min(where(RowNames eq 'Amplitude L3'))						; CGroupParametersGP[29,*] - Label 3 Amplitude
SigL1_ind = min(where(RowNames eq 'Zeta0'))								; CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
SigL2_ind = min(where(RowNames eq 'Sigma Amp L2'))						; CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
SigL3_ind = min(where(RowNames eq 'Sigma Amp L3'))						; CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))							; CGroupParametersGP[35,*] - Sigma Z
Coh_ind = min(where(RowNames eq '36 Coherence'))						; CGroupParametersGP[36,*] - Coherence
GrAmpL1_ind = min(where(RowNames eq 'Group A1'))						; CGroupParametersGP[37,*] - Group L1 Amplitude
GrAmpL2_ind = min(where(RowNames eq 'Group A2'))						; CGroupParametersGP[38,*] - Group L2 Amplitude
GrAmpL3_ind = min(where(RowNames eq 'Group A3'))						; CGroupParametersGP[39,*] - Group L3 Amplitude
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Ell_ind = min(where(RowNames eq 'XY Ellipticity'))						; CGroupParametersGP[43,*] - XY Ellipticity
UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
UnwZErr_ind = min(where(RowNames eq 'Unwrapped Z Error'))				; CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error

allind1 = [Gr_ind, $
		GrX_ind, $
		GrY_ind, $
		GrSigX_ind, $
		GrSigY_ind, $
		GrNph_ind, $
		Gr_size_ind, $
		;LabelSet_ind, $
		GrAmpL1_ind, $
		GrAmpL2_ind, $
		GrAmpL3_ind, $
		GrZ_ind, $
		GrSigZ_ind, $
		GrCoh_ind, $
		Gr_Ell_ind, $
		UnwGrZ_ind, $
		UnwGrZErr_ind]
GroupParInd = allind1[where(allind1 ge 0)]
FromGroup = (max(where(GroupParInd eq Xitem)) ge 0) or (max(where(GroupParInd eq Yitem)) ge 0) or (max(where(GroupParInd eq Zitem)) ge 0)
if FromGroup then GroupFilterIt else FilterIt

Wid_XYZ_diamonds = Widget_Info(Event.top, find_by_uname='W_MENU_XYZ_diamonds')
use_diamonds = WIDGET_INFO(Wid_XYZ_diamonds,/button_set)
sym = use_diamonds ? 4 : 3
symsize=0

ThisGroup=where(filter eq 1,cnt)
if cnt gt 1 then begin
	if Zitem eq LabelSet_ind then begin
		lmin=min(CGroupParams[LabelSet_ind,ThisGroup])
		lmax=max(CGroupParams[LabelSet_ind,ThisGroup])
		ThisGroup_l=where(filter and (CGroupParams[LabelSet_ind,*] eq lmin))
			plot,CGroupParams[Xitem,ThisGroup_l],CGroupParams[Yitem,ThisGroup_l],psym=sym, SYMSIZE=symsize, $
			xrange=[ParamLimits[Xitem,0],ParamLimits[Xitem,1]],xstyle=1,$
			yrange=[ParamLimits[Yitem,0],ParamLimits[Yitem,1]],ystyle=1, xtitle=RowNames[Xitem], ytitle=RowNames[Yitem],$
			title='                                                   Color: '+RowNames[Zitem],$
			thick=tk, xthick=1.0, ythick=1.0, charsize=tk, charthick=tk
		if lmax gt lmin then begin
			lbl_color = [200, 40, 105, 135, 235]				; [red, green, blue, purple, gray]
			device,decompose=0
			loadct,12
			oplot,CGroupParams[Xitem,ThisGroup_l],CGroupParams[Yitem,ThisGroup_l], psym=sym, SYMSIZE=symsize, col=lbl_color[0]
			for ic=lmin+1,lmax do begin
				ThisGroup_l=where(filter and (CGroupParams[LabelSet_ind,*] eq ic))
				oplot,CGroupParams[Xitem,ThisGroup_l],CGroupParams[Yitem,ThisGroup_l],psym=sym, SYMSIZE=symsize, col=lbl_color[ic-lmin]
			endfor
			device,decompose=0
			loadct,3
		endif
	endif else begin
		if ((Xitem eq X_ind) or (Xitem eq GrX_ind)) and ((Yitem eq Y_ind) or (Yitem eq GrY_ind)) then begin
			plot,CGroupParams[Xitem,Thisgroup],CGroupParams[Yitem,Thisgroup], psym=sym, SYMSIZE=symsize,$
			xrange=[ParamLimits[Xitem,0],ParamLimits[Xitem,1]],xstyle=1,$
			yrange=[ParamLimits[Yitem,0],ParamLimits[Yitem,1]],ystyle=1,$
			position=[0,0,1,1], xtitle=RowNames[Xitem], ytitle=RowNames[Yitem], $
			title='                                                   Color: '+RowNames[Zitem],$
			thick=tk, xthick=1.0, ythick=1.0, charsize=tk, charthick=tk
		endif else begin
			plot,CGroupParams[Xitem,Thisgroup],CGroupParams[Yitem,Thisgroup], psym=sym, SYMSIZE=symsize,$
			xrange=[ParamLimits[Xitem,0],ParamLimits[Xitem,1]],xstyle=1,$
			yrange=[ParamLimits[Yitem,0],ParamLimits[Yitem,1]],ystyle=1, xtitle=RowNames[Xitem], ytitle=RowNames[Yitem],$
			title='                                                   Color: '+RowNames[Zitem],$
			thick=tk, xthick=1.0, ythick=1.0, charsize=tk, charthick=tk
		endelse
		col=bytscl(CGroupParams[Zitem,Thisgroup])
		plots,CGroupParams[Xitem,Thisgroup],CGroupParams[Yitem,Thisgroup],psym=3,$
			color=col*0.75+64,/data
	endelse
endif
end
;
;-----------------------------------------------------------------
;
pro OnPlotXgrYgrZgr, Event
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	GrXindex = max(where(RowNames[*] eq 'Group X Position'))
	GrYindex = max(where(RowNames[*] eq 'Group Y Position'))
	GrZindex = max(where(RowNames[*] eq 'Group Z Position'))
	WidDrXID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select = GrXindex	; 	19 = GroupX
	WidDrYID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select = GrYindex ; 	20 = GroupY
	WidDrZID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select = GrZindex ; 	40 = GroupZ
	OnPlotXYButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnPlotXgrZgrXgr, Event
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	GrXindex = max(where(RowNames[*] eq 'Group X Position'))
	GrYindex = max(where(RowNames[*] eq 'Group Y Position'))
	GrZindex = max(where(RowNames[*] eq 'Group Z Position'))
	WidDrXID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select = GrXindex	; 	19 = GroupX
	WidDrYID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select = GrZindex ; 	40 = GroupZ
	WidDrZID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select = GrYindex ; 	20 = GroupY
	OnPlotXYButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnPlotYgrZgrXgr, Event
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	GrXindex = max(where(RowNames[*] eq 'Group X Position'))
	GrYindex = max(where(RowNames[*] eq 'Group Y Position'))
	GrZindex = max(where(RowNames[*] eq 'Group Z Position'))
	WidDrXID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select = GrYindex ; 	20 = GroupY
	WidDrYID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select = GrZindex ; 	40 = GroupZ
	WidDrZID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select = GrXindex	; 	19 = GroupX
	OnPlotXYButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnPlotXgrUnwZgrXgr, Event
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	GrXindex = max(where(RowNames[*] eq 'Group X Position'))
	GrYindex = max(where(RowNames[*] eq 'Group Y Position'))
	UnwGrZindex = max(where(RowNames[*] eq 'Unwrapped Group Z'))
	WidDrXID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select = GrXindex	; 	19 = GroupX
	WidDrYID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select = UnwGrZindex ; 	47 = UwrGroupZ
	WidDrZID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select = GrYindex ; 	20 = GroupY
	OnPlotXYButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnPlotYgrUnwZgrXgr, Event
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	GrXindex = max(where(RowNames[*] eq 'Group X Position'))
	GrYindex = max(where(RowNames[*] eq 'Group Y Position'))
	UnwGrZindex = max(where(RowNames[*] eq 'Unwrapped Group Z'))
	WidDrXID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select = GrYindex ; 	20 = GroupY
	WidDrYID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select = UnwGrZindex ; 	47 = UwrGroupZ
	WidDrZID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select = GrXindex	; 	19 = GroupX
	OnPlotXYButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnPlotFrameZX, Event
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	Xindex = max(where(RowNames[*] eq 'X Position'))
	Zindex = max(where(RowNames[*] eq 'Z Position'))
	Frindex = max(where(RowNames[*] eq 'Frame Number'))
	WidDrXID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select = Frindex	;	9 = Frame
	WidDrYID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select = Zindex	;	34 = Z
	WidDrZID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select = Xindex	; 	2 = X
	OnPlotXYButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnPlotFrameUnwrZX, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

	X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
	FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
	UnwZ_ind = min(where(RowNames eq 'Unwrapped Z'))						; CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
	SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))					; CGroupParametersGP[16,*] - x - sigma
	SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))					; CGroupParametersGP[17,*] - y - sigma

	ParamLimits[FrNum_ind,0]=15
	ParamLimits[FrNum_ind,1]=85
	ParamLimits[SigX_ind,1]=0.05
	ParamLimits[SigY_ind,1]=0.05
	ParamLimits[UnwZ_ind,0]=-400
	ParamLimits[UnwZ_ind,1]=400
	TopIndex = (CGrpSize-1)
	wtable = Widget_Info(Event.top, find_by_uname='WID_TABLE_0')
	widget_control,wtable,set_value=transpose(ParamLimits[0:TopIndex,0:3]), use_table_select=[0,0,3,TopIndex]
	WidDrXID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select = FrNum_ind	;	9 = Frame
	WidDrYID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select = UnwZ_ind	;	44 = Z
	WidDrZID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select = X_ind	; 	2 = X
	OnPlotXYButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnPlotXZLbl, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
	Xindex = max(where(RowNames[*] eq 'X Position'))
	Zindex = max(where(RowNames[*] eq 'Z Position'))
	Lblindex = max(where(RowNames[*] eq 'Label Set'))
	WidDrXID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select= Xindex	;	2 = X
	WidDrYID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select = Zindex	;	34 = Z
	WidDrZID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select = Lblindex	; 	26 = Label Number
	OnPlotXYButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnPlotXgrUnwZgrLbl, Event
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
GrX_ind = min(where(RowNames eq 'Group X Position'))                    ; CGroupParametersGP[19,*] - average x - position in the group
LabelSet_ind = min(where(RowNames eq 'Label Set'))                       ; CGroupParametersGP[26,*] - Label Number
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))                ; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
GrZ_ind = min(where(RowNames eq 'Group Z Position'))                    ; CGroupParametersGP[40,*] - Group Z Position

Z_ind = (UnwGrZ_ind gt 0) ?	UnwGrZ_ind	: GrZ_ind

	WidDrXID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select = GrX_ind
	WidDrYID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select = Z_ind
	WidDrZID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select = LabelSet_ind
	OnPlotXYButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnStretchTop, Event				;Set max of color range
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
WidDL_LabelID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Label')
selectedlabel = widget_info(WidDL_LabelID,/DropList_Select)
WID_BUTTON_Tie_RGB_ID = Widget_Info(Event.Top, find_by_uname='WID_BUTTON_Tie_RGB')
Tie_RGB = Widget_Info(WID_BUTTON_Tie_RGB_ID, /BUTTON_SET)
if Tie_RGB then labelContrast[0,*]=Event.value else labelContrast[0,selectedlabel]=Event.value
;labelContrast[0,selectedlabel]=Event.value
AdjustContrastnDisplay,Event
end
;
;-----------------------------------------------------------------
;
pro OnGamma, Event					;Set gamma of color range
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
WidDL_LabelID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Label')
selectedlabel = widget_info(WidDL_LabelID,/DropList_Select)
WID_BUTTON_Tie_RGB_ID = Widget_Info(Event.Top, find_by_uname='WID_BUTTON_Tie_RGB')
Tie_RGB = Widget_Info(WID_BUTTON_Tie_RGB_ID, /BUTTON_SET)
if Tie_RGB then labelContrast[1,*]=Event.value else labelContrast[1,selectedlabel]=Event.value
;labelContrast[1,*]=Event.value
AdjustContrastnDisplay, Event
end
;
;-----------------------------------------------------------------
;
pro OnStretchBottom, Event			;Set min of color range
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
WidDL_LabelID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Label')
selectedlabel = widget_info(WidDL_LabelID,/DropList_Select)
WID_BUTTON_Tie_RGB_ID = Widget_Info(Event.Top, find_by_uname='WID_BUTTON_Tie_RGB')
Tie_RGB = Widget_Info(WID_BUTTON_Tie_RGB_ID, /BUTTON_SET)
if Tie_RGB then labelContrast[2,*]=Event.value else labelContrast[2,selectedlabel]=Event.value
;labelContrast[2,selectedlabel]=Event.value
AdjustContrastnDisplay, Event
end
;
;-----------------------------------------------------------------
;
pro Set_Tie_RGB, Event
	; right now - do nothing.
end
;
;-----------------------------------------------------------------
;
pro AdjustContrastnDisplay, Event	;Adjust conrast according to top gamma and bottom and display it
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number

WidSldTopID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Top')
;widget_control,WidSldTopID,get_value=topV
WidSldBotID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Bot')
;widget_control,WidSldBotID,get_value=botV
WidSldGammaID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Gamma')
;widget_control,WidSldGammaID,get_value=gamma

WidDL_LabelID = Widget_Info(TopID, find_by_uname='WID_DROPLIST_Label')
Label=widget_info(WidDL_LabelID,/DropList_Select)
botV=labelContrast[2,Label]
gamma=labelContrast[1,Label]
topV=labelContrast[0,Label]
if topV le botV then begin
	topV = botV+1
	widget_control,WidSldTopID,set_value=TopV
endif
sz_im=size(image)
if (sz_im[0] eq 2) and (label eq 0) then begin
	Timage=image ^(gamma/1000.)
	rng=Max(Timage)-Min(Timage)
	tv,bytscl(Timage,min=((botV/1000.)*rng+Min(Timage)),max=(Max(Timage)-(1.-topV/1000.)*rng))
endif

mx = (LabelSet_ind lt 0) ? 0 : max(CGroupParams[LabelSet_ind,*])
wID = Widget_Info(TopID, find_by_uname='W_MENU_54')
widget_control,wID,get_value=z_color
if z_color eq 'Z using Hue Scale' then mx=4
;	magenta=1
;	if magenta then mx>=3
if (sz_im[0] eq 3) and (mx gt 1) then begin
	Labelcontrast[*,label]=[topV,gamma,botV]
	Timage=image
	mx<=3
	for i=0,mx-1 do begin
		gamma=labelcontrast[1,i+1]
		topV=labelcontrast[0,i+1]
		botV=labelcontrast[2,i+1]
		Timage[*,*,i]=image[*,*,i]^(gamma/1000.)
		rng=Max(Timage[*,*,i])-Min(Timage[*,*,i])
		Timage[*,*,i]=bytscl(Timage[*,*,i],min=(botV/1000.)*rng+Min(Timage[*,*,i]),max=Max(Timage[*,*,i])-(1.-topV/1000.)*rng)
	endfor
	tv,Timage,true=3
endif

end
;-----------------------------------------------------------------
; Activate Button Callback Procedure.
;-----------------------------------------------------------------
;Paints Image of Gaussians in zoomed range of
;		FuntionItem = 0		Centers of Fits
;		FunctionItem= 1		Normalized Gaussians
;		FunctionItem= 2		Photon Intensity Amplitude Fitted Gaussians
;		FilterItem	= 0		Use Peaks data 0:17
;		FilterItem	= 1		Used Group data 18:25
;		AccumItem	= 0		Use envelope accumualtion
;		AccumItme	= 1		Use Summed accumulation
;-----------------------------------------------------------------
pro OnRenderButton, Event			;Render the display according to function filter & accum settings (maybe slow)
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
if n_elements(CGroupParams) le 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

LabelSet_ind = min(where(RowNames eq 'Label Set'))						; CGroupParametersGP[26,*] - Label Number

FunctionId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Function')
FunctionItem=widget_info(FunctionId,/DropList_Select)						;	(0: Center Locations,  1: Gaussian Normalized,   2: Gaussian Amplitude)

;In the case of 2DPALM image, check if the "Force max probability" button is checked.
; If checked, adjust the Gamma and Top sliders accordingly.
wID = Widget_Info(TopID, find_by_uname='W_MENU_54')
widget_control,wID,get_value=z_color
wID = Widget_Info(TopID, find_by_uname='W_MENU_Force_MaxProb_2DPALM')
Force_MaxProb_button_state=widget_info(wID,/BUTTON_SET)
if (z_color ne 'Z using Hue Scale') and Force_MaxProb_button_state then Adjust_Top_Slider_for_Max_2D_Probability

mx = (LabelSet_ind lt 0) ? 0 : max(CGroupParams[LabelSet_ind,*])									; Set the droplist Label to red if more than one Label (ie. 3D image array)
wID = Widget_Info(TopID, find_by_uname='W_MENU_54')
widget_control,wID,get_value=z_color
if z_color eq 'Z using Hue Scale' then mx=4

if mx ge 1 then begin
	wDROPLISTLabel = Widget_Info(TopID, find_by_uname='WID_DROPLIST_Label')
	widget_control,wDROPLISTlabel,set_droplist_select=1				;set to red
	selectedlabel=1
endif
if mx eq 0 then begin
	wDROPLISTLabel = Widget_Info(TopID, find_by_uname='WID_DROPLIST_Label')
	widget_control,wDROPLISTlabel,set_droplist_select=0				;set to null
	selectedlabel=0
endif

WidSldTopID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Top')
widget_control,WidSldTopID,set_value=labelContrast[0,selectedlabel]
WidSldGammaID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Gamma')
widget_control,WidSldGammaID,set_value=labelContrast[1,selectedlabel]
WidSldBotID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Bot')
widget_control,WidSldBotID,set_value=labelContrast[2,selectedlabel]

if Event.top eq TopID then begin						; if called from the main menu, the image size is 1024x1024, otherwise Cust_TIFF_Pix_X x Cust_TIFF_Pix_Y
	wxsz=1024 & wysz=1024								; size of the display window
endif else begin
	wxsz=Cust_TIFF_Pix_X
	wysz=Cust_TIFF_Pix_Y
endelse

;loc=fltarr(wxsz,wysz)
;mg=(wxsz<wysz)/((dxmx-dxmn)>(dymx-dymn))			; # of display pixels per CCD pixel = ratio of the display window size to th ethe image size (magnification)
;mgw=fix(mg)											; rounded magnification number
mgw=GetScaleFactor(ParamLimits, wxsz, wysz)

WARN_on_NOGROUPS = 1
RenderWithoutAutoscale, Event, wxsz, wysz, WARN_on_NOGROUPS
filterlist=where(filter eq 1,cnt)
if cnt le 1 then return
;if FunctionItem ne 0 then AdjustContrastnDisplay, Event
AdjustContrastnDisplay, Event

wID = Widget_Info(TopID, find_by_uname='W_MENU_59')
original='Render Without Legends'
modified = 'Do not plot Bars/Legends'
widget_control,wID,get_value=test
;if (test eq original) and (Event.top eq TopID) then begin
if (test eq original) then begin
	if FunctionItem ne 0 then OnAddColorBarButton, Event
	TVscales,wxsz,wysz,mgw,nm_per_pixel
	OnAddSigmaFilterButton, Event
endif

end
;
;-----------------------------------------------------------------
;
function build_fimage, CGroupParams, filter, ParamLimits, render_ind, render_params, render_win, nm_per_pixel, liveupdate

	filterlist = where(filter eq 1,cnt)

	X_ind = render_ind[0]
	Y_ind = render_ind[1]
	Z_ind = render_ind[2]
	Xs_ind = render_ind[3]
	Ys_ind = render_ind[4]
	Nph_ind = render_ind[5]
	GrNph_ind = render_ind[6]
	Frame_Number_ind = render_ind[7]
	Label_ind = render_ind[8]

	FilterItem = render_params[0]
	FunctionItem = render_params[1]
	AccumItem = render_params[2]
	rend_z_color = render_params[3]
	lbl_mx = render_params[4]
	testXZ = render_params[5]
	testYZ = render_params[6]

	cur_win = render_win[0]
	dxmn = render_win[1]
	dymn = render_win[2]
	dxmx = render_win[3]
	dymx = render_win[4]
	hue_scale = render_win[5]
	wxsz = render_win[6]
	wysz = render_win[7]
	vbar_top = render_win[8]

	mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))			; # of display pixels per CCD pixel = ratio of the display window size to th ethe image size (magnification)
	wd=fix(mgw*1.5*133./nm_per_pixel) > 4.0									; scale "radius" of gaussian sub-window to ~ 150 nm

	lbl_pos = CgroupParams[Label_ind,filterlist]-1
	wxpkpos	= mgw * (CGroupParams[X_ind,filterlist]-dxmn)		;x peak positions in units of display window pixels - vector for filtered peaks
	wypkpos	= mgw * (CGroupparams[Y_ind,filterlist]-dymn)
	;wxofs_v	=(fix(wxpkpos)>0)
	;wyofs_v	=(fix(wypkpos)>0)
	wxofs_v	=floor(wxpkpos)
	wyofs_v	=floor(wypkpos)
	if FunctionItem gt 0 then begin
		wdd=2*wd+1
		wx = findgen(wdd)-wd									;wdd x vector of window pixels (zero is in middle of array)
		wy = findgen(wdd)-wd									;wdd y vector
		wxsig	= mgw * CGroupParams[xs_ind,filterlist]				;x sigma  in units of display window pixels - vector for filtered peaks
		wysig	= mgw * CGroupParams[ys_ind,filterlist]
		A1 = 1.0D/(2.*!pi*wxsig*wysig)*(FunctionItem eq 1) + CGroupParams[Nph_ind,filterlist]*(FilterItem eq 0)/mgw/mgw*(FunctionItem eq 2);Gaussian amplitude - vector for filtered peaks
		if 	GrNph_ind gt 0 then A1 = A1	+ CGroupParams[GrNph_ind,filterlist]*(FilterItem eq 1)/mgw/mgw*(FunctionItem eq 2)
		;FunctionItem	=	(0: Center Locations,  1: Gaussian Normalized,   2: Gaussian Amplitude)
		;FilterItem		=	(0: Frame Peaks,   1: Group Peaks)
		;AccumItem		=	(0: Envelope,   1: Sum)
	endif
		;
		; default vbar_top= molecular probability of 0.003 molecule per nm^2, then vbar_top/(mgW/nm_per_pixel)^2 is molecular probabilty per pixel
		; Calculate scale factor so that scale_factor * vbar_top / (mgW/nm_per_pixel)^2 = 0.5
	scale_factor = 0.5 / vbar_top * float(mgw/nm_per_pixel)^2
	;
	;---------------------------------						setup info in case Hue Scale is set

	if rend_z_color then begin
		if ~testXZ and ~testYZ then normZval=(CgroupParams[z_ind,filterlist]-ParamLimits[z_ind,0])/ParamLimits[z_ind,3]
		if testXZ and ~testYZ  then normZval=(CgroupParams[x_ind,filterlist]-ParamLimits[x_ind,0])/ParamLimits[x_ind,3]
		if ~testXZ and testYZ  then normZval=(CgroupParams[y_ind,filterlist]-ParamLimits[y_ind,0])/ParamLimits[y_ind,3]
	endif
	;[wxpkpos, wypkpos, wxsig, wysig, wxofs_v, wyofs_v, A1, frame_index, label_index, normZval]

	if liveupdate gt 0 then begin
		;cancel_button_present = (liveupdate eq 1) ? 1 : 0	; do not allow for cancel button in Bridge
		cancel_button_present = 1
		oStatusBar = obj_new('PALM_StatusBar', $
        	COLOR=[0,0,255], $
        	TEXT_COLOR=[255,255,255], $
        	CANCEL_BUTTON_PRESENT = cancel_button_present, $
       	 	TITLE='Rendering Image...', $
       		TOP_LEVEL_BASE=tlb)
		fraction_complete_last=0.0D
		pr_bar_inc=0.01D
	endif

	modval=((cnt/10)<10000)								; increment for progress diplay

	if (lbl_mx gt 0) or rend_z_color then fimage=dblarr(wxsz,wysz,3) else	fimage=dblarr(wxsz,wysz)
	for j=0l,cnt-1 do begin
		wxofs=wxofs_v[j]
		wyofs=wyofs_v[j]

		if FunctionItem eq 0 then begin			; centers only (hystogram)
			if NOT rend_z_color then begin	; DO NOT use Hue for z-coordinate
				if (lbl_mx eq 0)	then begin	; single label
					if (AccumItem eq 1) then fimage[wxofs<(wxsz-1),wyofs<(wysz-1)] += 1		;Sum
					if (AccumItem eq 0) then fimage[wxofs<(wxsz-1),wyofs<(wysz-1)] >= 1		;Envelope
				endif else	begin					; multiple labels
					;if (AccumItem eq 1) then fimage[wxofs<(wxsz-1),wyofs<(wysz-1),CgroupParams[Label_ind,j]-1] += 1		;Sum
					;if (AccumItem eq 0) then fimage[wxofs<(wxsz-1),wyofs<(wysz-1),CgroupParams[Label_ind,j]-1] >= 1		;Envelope
					if (AccumItem eq 1) then fimage[wxofs<(wxsz-1),wyofs<(wysz-1),lbl_pos[j]] += 1		;Sum
					if (AccumItem eq 0) then fimage[wxofs<(wxsz-1),wyofs<(wysz-1),lbl_pos[j]] >= 1		;Envelope
				endelse
			endif else begin
				hue=hue_scale * (normZval[j]-fix(normZval[j]))
				h = hue
				s = 1.0
				v = scale_factor <1.0
				;hsv_gauss=[h,s,v]
				hsv_gauss=[[[h]],[[s]],[[v]]]
				color_convert,hsv_gauss,rgb_gauss,Interleave=2,/hsv_rgb
				if (AccumItem eq 1) then fimage[wxofs<(wxsz-1),wyofs<(wysz-1),*] += rgb_gauss	;Sum
				if (AccumItem eq 0) then fimage[wxofs<(wxsz-1),wyofs<(wysz-1),*] >= rgb_gauss	;Envelope
			endelse
		endif else begin
			dwx=FLOAT(wx-(wxpkpos[j]-wxofs))/wxsig[j]
			dwy=FLOAT(wy-(wypkpos[j]-wyofs))/wysig[j]
			gausscenter=A1[j]*exp(-(((dwx^2)/2.0D)<1000.0D))#exp(-(((dwy^2)/2.0D)<1000.0D))
			gausscenter=gausscenter[(wd-wxofs)>0:(2*wd+((wxsz-1-wxofs-wd)<0)),(wd-wyofs)>0:(2*wd+((wysz-1-wyofs-wd)<0))]
			if NOT rend_z_color then begin	; DO NOT use Hue for z-coordinate
				if (lbl_mx eq 0)	then begin	; single label
					if (AccumItem eq 1) then fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1)] += gausscenter; + q		;Sum
					if (AccumItem eq 0) then fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1)] >= gausscenter; > q		;Envelope
				endif else	begin					; multiple labels
					;if (AccumItem eq 1) then fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),CgroupParams[Label_ind,j]-1] += gausscenter; + q		;Sum
					;if (AccumItem eq 0) then fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),CgroupParams[Label_ind,j]-1] >= gausscenter; > q		;Envelope
					if (AccumItem eq 1) then fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),lbl_pos[j]] += gausscenter; + q		;Sum
					if (AccumItem eq 0) then fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),lbl_pos[j]] >= gausscenter; > q		;Envelope
				endelse
			endif else begin								; use Hue for z-coordinate ;multiply hue_scale * normalized z to range
				hue=hue_scale * (normZval[j]-fix(normZval[j]))
				h=(gausscenter*0.+1.0)*hue
				s=gausscenter*0.+1.0
				v= scale_factor * gausscenter<1.0
				hsv_gauss=[[[h]],[[s]],[[v]]]
				color_convert,hsv_gauss,rgb_gauss,Interleave=2,/hsv_rgb
				if (AccumItem eq 1) then fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),*] += rgb_gauss; + q		;Sum
				if (AccumItem eq 0) then fimage[(wxofs-wd)>0:(wxofs+wd)<(wxsz-1),(wyofs-wd)>0:(wyofs+wd)<(wysz-1),*] >= rgb_gauss; > q		;Envelope
			endelse
		endelse
		; liveupdate = 1 - all updates, liveupdate = 2 - only progress bar
		if liveupdate gt 0 then begin
			fraction_complete=FLOAT(j)/FLOAT((cnt-1.0))
			if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
				if oStatusBar -> CheckCancel() then begin
					obj_destroy, oStatusBar
					fimage=0
					return, fimage
				endif
				fraction_complete_last=fraction_complete
				oStatusBar -> UpdateStatus, fraction_complete
			endif
			if (j mod modval eq 1) and (liveupdate eq 1) then begin
				wset,cur_win
				if (lbl_mx gt 0) or rend_z_color then tvscl,fimage,true=3 else tvscl,fimage
				xyouts,0.01,0.01,'Frame '+string(CgroupParams[Frame_Number_ind,j],format='(2i9)')+'/',/normal
				wait, 0.02
			endif
		endif
	endfor
if liveupdate gt 0 then obj_destroy, oStatusBar
return, fimage
end
;
;-----------------------------------------------------------------
;
pro RenderWithoutAutoscale, Event, wxsz, wysz, WARN_on_NOGROUPS			;Render the display according to function filter & accum settings (maybe slow)
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common Zdisplay, Z_scale_multiplier, vbar_top
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

Off_ind = min(where(RowNames eq 'Offset'))								; CGroupParametersGP[0,*] - Peak Base Level (Offset)
Amp_ind = min(where(RowNames eq 'Amplitude'))							; CGroupParametersGP[1,*] - Peak Amplitude
X_ind = min(where(RowNames eq 'X Position'))							; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))							; CGroupParametersGP[3,*] - Peak Y  Position)
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
GrX_ind = min(where(RowNames eq 'Group X Position'))					; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))					; CGroupParametersGP[20,*] - average y - position in the group
GrNph_ind = min(where(RowNames eq 'Group N Photons'))					; CGroupParametersGP[23,*] - total Photons in the group
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group
SigX_ind=min(where(RowNames eq 'Sigma X Pos Full'))
SigY_ind=min(where(RowNames eq 'Sigma Y Pos Full'))
GrSigX_ind=min(where(RowNames eq 'Group Sigma X Pos'))
GrSigY_ind=min(where(RowNames eq 'Group Sigma Y Pos'))
Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
UnwZ_ind=min(where(RowNames eq 'Unwrapped Z'))
UnwZ_Err_ind=min(where(RowNames eq 'Unwrapped Z Error'))
UnwGrZ_ind=min(where(RowNames eq 'Unwrapped Group Z'))
UnwGrZ_Err_ind=min(where(RowNames eq 'Unwrapped Group Z Error'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size_ind = min(where(RowNames eq '24 Group Size'))
Frame_Number_ind = min(where(RowNames eq 'Frame Number'))
Label_ind = min(where(RowNames eq 'Label Set'))

if n_elements(CGroupParams) le 2 then begin
	image=fltarr(wxsz,wysz)
	print,'no points to display'
	return      ; if data not loaded return
endif

cur_win=!D.window

FunctionId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Function')
FunctionItem=widget_info(FunctionId,/DropList_Select)						;	(0: Center Locations,  1: Gaussian Normalized,   2: Gaussian Amplitude)
FilterId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Filter')
FilterItem=widget_info(FilterId,/DropList_Select)							;	(0: Frame Peaks,   1: Group Peaks)
AccumId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Accumulate')
AccumItem=widget_info(AccumId,/DropList_Select)								;	(0: Envelope,   1: Sum)

X_ind = FilterItem ? GrX_ind : X_ind
Y_ind = FilterItem ? GrY_ind : Y_ind
xs_ind = FilterItem ? GrSigX_ind : SigX_ind
ys_ind = FilterItem ? GrSigY_ind : SigY_ind

Z_ind = FilterItem ? GrZ_ind : Z_ind
Zs_ind = FilterItem ? GrSigZ_ind : SigZ_ind
UnwZ_Ind = FilterItem ? UnwGrZ_ind : UnwZ_ind
Z_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_65')
Z_UnwZ_swap=Widget_Info(Z_swap_menue_ID,/button_set)
if Z_UnwZ_swap and UnwZ_Ind ge 0 then Z_ind = UnwZ_Ind


XZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapXZ')
XZ_swapped = Widget_Info(XZ_swap_menue_ID,/button_set)
YZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapYZ')
YZ_swapped = Widget_Info(YZ_swap_menue_ID,/button_set)

dxmn = paramlimits[x_ind,0]							; image size
dymn = paramlimits[y_ind,0]
dxmx = paramlimits[x_ind,1]
dymx = paramlimits[y_ind,1]

Z_ind0 = Z_ind
Zs_ind0 = Zs_ind
if XZ_swapped then begin
	dxmn = paramlimits[Z_ind,0]							; image size
	dxmx = paramlimits[Z_ind,1]
	Z_ind = X_ind
	X_ind = Z_ind0
	Zs_ind = Xs_ind
	Xs_ind = Zs_ind0
endif
if YZ_swapped then begin
	dymn = paramlimits[Z_ind,0]							; image size
	dymx = paramlimits[Z_ind,1]
	Z_ind = Y_ind
	Y_ind = Z_ind0
	Zs_ind = Ys_ind
	Ys_ind = Zs_ind0
endif

if FilterItem eq 0 then begin
	FilterIt
	npk_tot = total(filter)
	if npk_tot le 1 then begin
		z=dialog_message('No valid Peaks')
		return      ; if data not loaded return
	endif
	;if FunctionItem eq 0 then begin
	;	OnPeakCentersButton, Event
	;	return
	;endif
endif

if FilterItem eq 1 then begin
	GroupFilterIt
	npk_tot = total(filter)
	if ((NOT LMGR(/VM)) and (NOT LMGR(/DEMO)) and (NOT LMGR(/TRIAL)) and allow_bridge) then begin
		filter_bridge=SHMVAR(shmName_filter)
		filter_bridge[0]=filter
	endif
	if npk_tot le 1 then begin
		z=dialog_message('No valid Groups')
		return      ; if data not loaded return
	endif
	;if (FunctionItem eq 0) then begin
	;	OnGroupCentersButton, Event
	;	return
	;endif
endif

wID = Widget_Info(TopID, find_by_uname='W_MENU_54')
widget_control,wID,get_value=z_color
rend_z_color = z_color eq 'Z using Hue Scale'
lbl_mx = 0
if Label_ind ge 0 then begin
	filterlist = where(filter eq 1)
	lbl_mx = max(CGroupParams[Label_ind,filterlist])
endif

;YZ_swapped = 0
;YZ_swapped = 0
;if rend_z_color then begin
;	XZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapXZ')
;	YZ_swapped=Widget_Info(XZ_swap_menue_ID,/button_set)
;	YZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapYZ')
;	YZ_swapped=Widget_Info(YZ_swap_menue_ID,/button_set)
;endif
;print,'XZ_swapped, YZ_swapped', XZ_swapped, YZ_swapped
;print,X_ind, Y_ind, Z_ind, Xs_ind, Ys_ind
render_ind = [X_ind, Y_ind, Z_ind, Xs_ind, Ys_ind, Nph_ind, GrNph_ind, Frame_Number_ind, Label_ind]
render_params = [FilterItem, FunctionItem, AccumItem, rend_z_color, lbl_mx, XZ_swapped, YZ_swapped]
render_win = [cur_win, dxmn, dymn, dxmx, dymx, hue_scale, wxsz, wysz, vbar_top]

start=DOUBLE(systime(2))

if (NOT LMGR(/VM)) and (NOT LMGR(/DEMO)) and (NOT LMGR(/TRIAL)) and allow_bridge then begin
; ***** IDL Bridge Version ******************************
		print,'Starting Bridge rendering, no intermediate display updates'
		;common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
		npk_sub = ceil(npk_tot/n_br_loops)
		for i=0, n_br_loops-1 do begin
			if i eq 0 then begin
				fbr_arr[i]->setvar, 'liveupdate',2
			endif else fbr_arr[i]->setvar, 'liveupdate',0
			fbr_arr[i]->setvar, 'nlps', i
			fbr_arr[i]->setvar, 'render_ind', render_ind
			fbr_arr[i]->setvar, 'render_params', render_params
			fbr_arr[i]->setvar, 'render_win', render_win
			fbr_arr[i]->setvar, 'nm_per_pixel', nm_per_pixel
			fbr_arr[i]->setvar, 'ParamLimits', ParamLimits
			fbr_arr[i]->execute,'fimage_i = build_fimage (CGroupParams_bridge[*,istart:istop], filter_bridge[istart:istop], ParamLimits, render_ind, render_params, render_win, nm_per_pixel, liveupdate)', /NOWAIT
		endfor

		Alldone = 0
		aborted = 0
		while (alldone EQ 0) and (aborted EQ 0) do begin
			wait,0.5
			Alldone = 1
			if (fbr_arr[0]->Status() ne 1) then begin
				fbr_arr[0]->execute,'fimg_sz = n_elements(fimage_i)'
				fimg_sz = fbr_arr[0]->getvar('fimg_sz')
				aborted = fimg_sz gt 1 ? 0 : 1
			endif
			for i=0, n_br_loops-1 do begin
				bridge_done=fbr_arr[i]->Status()
				print,'Bridge',i,'  status0:',bridge_done
 				Alldone = Alldone * (bridge_done ne 1)
			endfor
		endwhile

		if aborted EQ 0 then begin
			fimage = fbr_arr[0]->getvar('fimage_i')
			if n_br_loops gt 1 then begin
				for i=1, n_br_loops-1 do begin
					fimage_i = fbr_arr[i]->getvar('fimage_i')
					if (AccumItem eq 1) then fimage += fimage_i; Sum
					if (AccumItem eq 0) then fimage >= fimage_i; Envelope
				endfor
			endif
		endif else begin
			if (lbl_mx gt 0) or rend_z_color then fimage=dblarr(wxsz,wysz,3) else	fimage=dblarr(wxsz,wysz)
		endelse

endif else begin
;***** Non Bridge Version **************** loop through all peaks
	liveupdate = 1
	fimage = build_fimage (CGroupParams, filter, ParamLimits, render_ind, render_params, render_win, nm_per_pixel, liveupdate)
endelse

wset,cur_win
mx=(Label_ind lt 0) ? 0 : max(CGroupParams[Label_ind,*])
image=fimage
print,'finshed rendering without autoscale',DOUBLE(systime(2))-start,'  seconds render time'
end
;
;-----------------------------------------------------------------
;
pro ToggleLegends, Event
wID = Widget_Info(event.top, find_by_uname='W_MENU_59')
original='Render Without Legends'
modified = 'Do not plot Bars/Legends'
widget_control,wID,get_value=test
if test eq original then begin
	widget_control,wID,set_button=1
	widget_control,wID,set_value = modified
endif
if test eq modified then begin
	widget_control,wID,set_button=0
	widget_control,wID,set_value=original
endif
end
;
;-----------------------------------------------------------------
; Activate Button Callback Procedure.
;-----------------------------------------------------------------
pro OnAddScaleBarButton, Event		;Get scaling and display micron bar
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
if n_elements(CGroupParams) le 2 then begin
          z=dialog_message('Please load a data file')
          return      ; if data not loaded return
endif
if Event.top eq TopID then begin						; if called from the main menu, the image size is 1024x1024, otherwise Cust_TIFF_Pix_X x Cust_TIFF_Pix_Y
	wxsz=1024 & wysz=1024								; size of the display window
	;nm_pix=nm_per_pixel
endif else begin
	wxsz=Cust_TIFF_Pix_X
	wysz=Cust_TIFF_Pix_Y
	;nm_pix=cust_nm_per_pix
endelse
mgw=GetScaleFactor(ParamLimits, wxsz, wysz)
print,'wxsz=',wxsz,',      mgw=',mgw,'       nm per pixel=',nm_per_pixel
TVscales,wxsz,wysz,mgw,nm_per_pixel
end
;-----------------------------------------------------------------
; Activate Button Callback Procedure.
;-----------------------------------------------------------------
pro OnAddScaleBarButton2, Event		;Get scaling and display pix box scales
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
if n_elements(CGroupParams) le 2 then begin
          z=dialog_message('Please load a data file')
          return      ; if data not loaded return
endif
if Event.top eq TopID then begin						; if called from the main menu, the image size is 1024x1024, otherwise Cust_TIFF_Pix_X x Cust_TIFF_Pix_Y
	wxsz=1024 & wysz=1024								; size of the display window
	;nm_pix=nm_per_pixel
endif else begin
	wxsz=Cust_TIFF_Pix_X
	wysz=Cust_TIFF_Pix_Y
	;nm_pix=cust_nm_per_pix
endelse
mgw=GetScaleFactor(ParamLimits, wxsz, wysz)
TVscales2,wxsz,wysz,mgw,nm_per_pixel
end
;-----------------------------------------------------------------
; Activate Button Callback Procedure.
;-----------------------------------------------------------------
pro OnAddColorBarButton, Event		;Get color scaling of image and display density in nm^2
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common Zdisplay, Z_scale_multiplier, vbar_top
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
FilterId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Filter')
FilterItem=widget_info(FilterId,/DropList_Select)							;	(0: Frame Peaks,   1: Group Peaks)

X_ind=min(where(RowNames eq 'X Position'))
Y_ind=min(where(RowNames eq 'Y Position'))
GrX_ind=min(where(RowNames eq 'Group X Position'))
GrY_ind=min(where(RowNames eq 'Group Y Position'))
SigX_ind=min(where(RowNames eq 'Sigma X Pos Full'))
SigY_ind=min(where(RowNames eq 'Sigma Y Pos Full'))
GrSigX_ind=min(where(RowNames eq 'Group Sigma X Pos'))
GrSigY_ind=min(where(RowNames eq 'Group Sigma Y Pos'))
Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
UnwZ_ind=min(where(RowNames eq 'Unwrapped Z'))
UnwZ_Err_ind=min(where(RowNames eq 'Unwrapped Z Error'))
UnwGrZ_ind=min(where(RowNames eq 'Unwrapped Group Z'))
UnwGrZ_Err_ind=min(where(RowNames eq 'Unwrapped Group Z Error'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size_ind = min(where(RowNames eq '24 Group Size'))
Frame_Number_ind = min(where(RowNames eq 'Frame Number'))
Label_ind = min(where(RowNames eq 'Label Set'))

;Z_ind = FilterItem ? GrZ_ind : Z_ind
;UnwZ_Ind = FilterItem ? UnwGrZ_ind : UnwZ_ind
Z_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_65')
Z_UnwZ_swap=Widget_Info(Z_swap_menue_ID,/button_set)
if Z_UnwZ_swap and UnwZ_Ind ge 0 then Z_ind = UnwZ_Ind
if Z_UnwZ_swap and UnwGrZ_ind ge 0 then GrZ_ind = UnwGrZ_ind

if n_elements(CGroupParams) le 2 then begin
          z=dialog_message('Please load a data file')
          return      ; if data not loaded return
endif
wID = Widget_Info(TopID, find_by_uname='W_MENU_54')
widget_control,wID,get_value=z_color
sz_im=size(image)
if Event.top eq TopID then begin						; if called from the main menu, the image size is 1024x1024, otherwise Cust_TIFF_Pix_X x Cust_TIFF_Pix_Y
	wxsz=1024 & wysz=1024								; size of the display window
endif else begin
	wxsz=Cust_TIFF_Pix_X
	wysz=Cust_TIFF_Pix_Y
endelse
WidColorBarMove = Widget_Info(TopID, find_by_uname='W_MENU_MoveColorBar')
original='Move Color Bar to the Top'
modified = 'Move Color Bar to the Bottom'
widget_control,WidColorBarMove,get_value=test
Bar_Vpos = (test eq original) ? 40 : 950

WidSldTopID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Top')
widget_control,WidSldTopID,get_value=topV
WidSldBotID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Bot')
widget_control,WidSldBotID,get_value=botV
WidSldGammaID = Widget_Info(TopID, find_by_uname='WID_SLIDER_Gamma')
widget_control,WidSldGammaID,get_value=gamma

if sz_im[0] eq 2 then begin
	Timage=image^(gamma/1000.)
	rng=Max(Timage)-Min(Timage)
	min_mid_max=[(Min(Timage)+(botV/1000.)*rng)^(1000./gamma),$
				(Min(Timage)+((botV/1000.)+(topV-botV)/1000./2)*rng)^(1000./gamma),$
				(Max(Timage)-(1.-topV/1000.)*rng)^(1000./gamma)]
	mgw=GetScaleFactor(ParamLimits, wxsz, wysz)
	label=0
	ShowColorScaleBarLabel,Event,xydsz,mgw,min_mid_max,label
endif

if sz_im[0] eq 3 then begin
	Timage=image
	mx=max(CGroupParams[Label_ind,*])
	if z_color eq 'Z using Hue Scale' then mx=4

	XZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapXZ')
	XZ_swap=widget_info(XZ_swap_menue_ID,/BUTTON_SET)
	YZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapYZ')
	YZ_swap=widget_info(YZ_swap_menue_ID,/BUTTON_SET)

	;bar_row_peaks = XZ_swap ?	X_ind : (YZ_swap ? Y_ind : Z_ind)
	;bar_row_groups = XZ_swap ?	GrX_ind : (YZ_swap ?	GrY_ind : GrZ_ind)
	;bar_row = 	FilterItem ? bar_row_groups : bar_row_peaks
	bar_row = 	FilterItem ? GrZ_ind : Z_ind

	hbar_min=ParamLimits[bar_row,0]
	hbar_max=ParamLimits[bar_row,1]
	if XZ_swap or YZ_swap then begin
		hbar_min*=nm_per_pixel / Z_scale_multiplier
		hbar_max*=nm_per_pixel / Z_scale_multiplier
	endif

	if mx eq 4 then begin
		mgw=GetScaleFactor(ParamLimits, wxsz, wysz)
		image_max=intarr(3)
		for i=0,2 do image_max[i]=max(image[*,*,i])
		; scale_factor=0.010 * mgw^2 above is done so that the molecular probability of 0.003 molecule per nm^2 will have v=0.5

		im_sz = size(image)
		xa_bar = 347 < im_sz[1]*0.95
		xi_bar = (xa_bar - 250) >0
		dx =  float(xa_bar-xi_bar+1)
		h=hue_scale/dx*findgen(dx)#replicate(1.0,29)							; generate the data for the color bar
		s=h*0+1.0
		v=(1.0*replicate(1.,dx)#((findgen(29)+1)/29.0)^2)<1
		color_convert,h,s,v,rbar,gbar,bbar,/hsv_rgb
		cbar=[[[rbar]],[[gbar]],[[bbar]]]

		image[xi_bar:xa_bar,Bar_Vpos:(Bar_Vpos+28),*]=cbar

		Wid_ColorBarExtend = Widget_Info(TopID, find_by_uname='W_MENU_ColorBarExtend')
		original='Extend Color Bar'
		modified = 'Restore Original Color Bar'
		widget_control,Wid_ColorBarExtend,get_value=test
		extend_color_bar = test eq modified

		if extend_color_bar  then begin
			h_add=hue_scale/dx*findgen(dx)#replicate(1.0,58)							; generate the data for the color bar
			s_add=h_add*0+1.0
			v_add=(1.0*replicate(1.,251)#((findgen(58)+1)/58.0)^2)<1
			color_convert,h_add,s_add,v_add,rbar_add,gbar_add,bbar_add,/hsv_rgb
			cbar_add=[[[rbar_add]],[[gbar_add]],[[bbar_add]]]
			image[xi_bar:xa_bar,(Bar_Vpos+29):(Bar_Vpos+57),*]=cbar_add[*,29:57,*]*4
		endif

		AdjustContrastnDisplay,event										; refresh the image with the color bar
		xyouts,xi_bar,(Bar_Vpos-20),strtrim(fix(hbar_min),2),/device,col=255,align=0.5		;375		; print color grade title
		xyouts,xi_bar+dx/3,(Bar_Vpos-20),'Z position  (nm)',/device,col=255,align=0.45			;500
		xyouts,xa_bar-3,(Bar_Vpos-20),strtrim(fix(hbar_max),2),/device,col=255,align=0.5		;625
		hbar=(findgen(dx) mod 25)#replicate(1,5) eq 0									;  generate "ticks" array
		tvscl,hbar,xi_bar,(Bar_Vpos-5)													;375		;  plot ticks

;
;default vbar_top= molecular probability of 0.003 molecule per nm^2, then vbar_top/(mgW/nm_per_pixel)^2 is molecular probabilty per pixel
; Calculate scale factor so that scale_factor * vbar_top / (mgW/nm_per_pixel)^2 = 0.5
;
		vbar_0=4.0 * vbar_top/(29)^2
		vbar_1=vbar_top/1.0
		vbar_2=vbar_top*16.0

		xyouts,88,(Bar_Vpos-3),string(vbar_0,format='(E8.1)'),/device,col=255,align=1.0
		xyouts,88,(Bar_Vpos+11),string(vbar_1,format='(E8.1)'),/device,col=255,align=1.0
		xyouts,88,((Bar_Vpos+Label_ind)),string(4*vbar_top,format='(E8.1)'),/device,col=255,align=1.0
		xyouts,90,(Bar_Vpos+40+extend_color_bar*30),'per nm^2',/device,col=255,align=1.0
		xyouts,90,(Bar_Vpos+55+extend_color_bar*30),'Prob.    ',/device,col=255,align=1.0
		if extend_color_bar then	xyouts,88,((Bar_Vpos+54)),string(vbar_2,format='(E8.1)'),/device,col=255,align=1.0
		vbar=replicate(1,5)#(findgen(29) mod 14) eq 0									;  generate "ticks" array
		if extend_color_bar then	vbar=replicate(1,5)#(findgen(60) mod 14) eq 0
		tvscl,vbar,90,(Bar_Vpos+1)													;375		;  plot ticks
	endif else begin

		mx<=3
		for i=0,mx-1 do begin
			gamma=labelcontrast[1,i+1]
			topV=labelcontrast[0,i+1]
			botV=labelcontrast[2,i+1]
			Timage[*,*,i]=image[*,*,i]^(gamma/1000.)
			rng=Max(Timage[*,*,i])-Min(Timage[*,*,i])
			min_mid_max=[(Min(Timage[*,*,i])+(botV/1000.)*rng)^(1000./gamma),$
				(Min(Timage[*,*,i])+((botV/1000.)+(topV-botV)/1000./2)*rng)^(1000./gamma),$
				(Max(Timage[*,*,i])-(1.-topV/1000.)*rng)^(1000./gamma)]
			Label=i+1
			mgw=GetScaleFactor(ParamLimits, wxsz, wysz)
			ShowColorScaleBarLabel,Event,xydsz,mgw,min_mid_max,Label
		endfor
	endelse
endif

end
;
;-----------------------------------------------------------------
;
pro OnColorBar_Move, Event
wID = Widget_Info(event.top, find_by_uname='W_MENU_MoveColorBar')
original='Move Color Bar to the Top'
modified = 'Move Color Bar to the Bottom'
widget_control,wID,get_value=test
if test eq original then begin
	widget_control,wID,set_button=1
	widget_control,wID,set_value = modified
endif
if test eq modified then begin
	widget_control,wID,set_button=0
	widget_control,wID,set_value=original
endif
end
;
;-----------------------------------------------------------------
;
pro OnColorBar_Extend, Event
wID = Widget_Info(event.top, find_by_uname='W_MENU_ColorBarExtend')
original='Extend Color Bar'
modified = 'Restore Original Color Bar'
widget_control,wID,get_value=test
if test eq original then begin
	widget_control,wID,set_button=1
	widget_control,wID,set_value = modified
endif
if test eq modified then begin
	widget_control,wID,set_button=0
	widget_control,wID,set_value=original
endif
end
;
;-----------------------------------------------------------------
;
pro On_XYZ_use_diamonds, Event
	set = WIDGET_INFO(Event.id,/button_set)
	widget_control,Event.id,set_button=1-set
end
;
;-----------------------------------------------------------------
;
pro OnAddLabelButton, Event			;Write file name at top
WidLabel0 = Widget_Info(Event.Top, find_by_uname='WID_LABEL_0')
widget_control,WidLabel0,get_value=label
BKGND=fix(mean(tvrd(0,1000,1000,20)))
FCOL = (BKGND gt 128) ? 0 : 255
xyouts,512,1010,label,col=FCOL,align=0.5,/device
end
;
;-----------------------------------------------------------------
;
pro OnAddSigmaFilterButton, Event			;Write file name at top
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common managed, ids, names, modalList
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

XZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapXZ')
XZ_swapped = Widget_Info(XZ_swap_menue_ID,/button_set)
YZ_swap_menue_ID = Widget_Info(TopID, find_by_uname='W_MENU_SwapYZ')
YZ_swapped = Widget_Info(YZ_swap_menue_ID,/button_set)


if Event.top eq TopID then begin						; if called from the main menu, the image size is 1024x1024, otherwise Cust_TIFF_Pix_X x Cust_TIFF_Pix_Y
	wxsz=1024 & wysz=1024								; size of the display window
endif else begin
	wxsz=Cust_TIFF_Pix_X
	wysz=Cust_TIFF_Pix_Y
endelse

FilterId=widget_info(TopID, FIND_BY_UNAME='WID_DROPLIST_Filter')
FilterItem=widget_info(FilterId,/DropList_Select)

X_ind=min(where(RowNames eq 'X Position'))
Y_ind=min(where(RowNames eq 'Y Position'))
GrX_ind=min(where(RowNames eq 'Group X Position'))
GrY_ind=min(where(RowNames eq 'Group Y Position'))
SigX_ind=min(where(RowNames eq 'Sigma X Pos Full'))
SigY_ind=min(where(RowNames eq 'Sigma Y Pos Full'))
GrSigX_ind=min(where(RowNames eq 'Group Sigma X Pos'))
GrSigY_ind=min(where(RowNames eq 'Group Sigma Y Pos'))
Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
UnwZ_ind=min(where(RowNames eq 'Unwrapped Z'))
UnwZ_Err_ind=min(where(RowNames eq 'Unwrapped Z Error'))
UnwGrZ_ind=min(where(RowNames eq 'Unwrapped Group Z'))
UnwGrZ_Err_ind=min(where(RowNames eq 'Unwrapped Group Z Error'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size_ind = min(where(RowNames eq '24 Group Size'))
Frame_Number_ind = min(where(RowNames eq 'Frame Number'))
Label_ind = min(where(RowNames eq 'Label Set'))

if FilterItem then begin		; grouped parameters
	SigX = XZ_swapped eq 1 ? ParamLimits[GrSigX_ind,1]/nm_per_pixel  :  ParamLimits[GrSigX_ind,1]
	SigY = YZ_swapped eq 1 ? ParamLimits[GrSigY_ind,1]/nm_per_pixel  :  ParamLimits[GrSigY_ind,1]
	xtext='Gr. !4'+ STRING("162B)+'!3!IX!N < ' + strtrim(string(SigX,FORMAT='(F10.3)'),2) + ' pix'
	ytext='Gr. !4'+ STRING("162B)+'!3!IY!N < ' + strtrim(string(SigY,FORMAT='(F10.3)'),2) + ' pix'
	if SigZ_ind ge 0 then begin
		SigZ = ((XZ_swapped) or (YZ_swapped)) ? ParamLimits[GrSigZ_ind,1]*nm_per_pixel  :  ParamLimits[GrSigZ_ind,1]
		ztext0='Gr. !4'+ STRING("162B)+'!3!IZ!N < ' + strtrim(string(SigZ,FORMAT='(F10.3)'),2) + ' nm'
		if UnwGrZ_Err_ind ge 0 then ztext1= strtrim(string(ParamLimits[UnwGrZ_Err_ind,0],FORMAT='(F10.1)'),2) + ' nm < Unwr.Gr.Z.Err < '+ strtrim(string(ParamLimits[UnwGrZ_Err_ind,1],FORMAT='(F10.1)'),2)
	endif
endif else begin
	SigX = XZ_swapped eq 1 ? ParamLimits[SigX_ind,1]/nm_per_pixel  :  ParamLimits[SigX_ind,1]
	SigY = YZ_swapped eq 1 ? ParamLimits[SigY_ind,1]/nm_per_pixel  :  ParamLimits[SigY_ind,1]
	xtext='!4' + STRING("162B)+'!3!IX!N < ' + strtrim(string(SigX,FORMAT='(F10.3)'),2) + ' pix'
	ytext='!4' + STRING("162B)+'!3!IY!N < ' + strtrim(string(SigY,FORMAT='(F10.3)'),2) + ' pix'
	if SigZ_ind ge 0 then begin
		SigZ = ((XZ_swapped eq 1) or (XZ_swapped eq 1)) ? ParamLimits[SigZ_ind,1]*nm_per_pixel  :  ParamLimits[SigZ_ind,1]
		ztext0='!4' + STRING("162B)+'!3!IZ!N < ' + strtrim(string(SigZ,FORMAT='(F10.3)'),2) + ' nm'
		if UnwZ_Err_ind ge 0 then ztext1= strtrim(string(ParamLimits[UnwZ_Err_ind,0],FORMAT='(F10.1)'),2) + ' nm < Unwr.Z.Err < '+ strtrim(string(ParamLimits[UnwZ_Err_ind,1],FORMAT='(F10.1)'),2)
	endif
endelse

xi = round(wxsz*0.1172)
xa = round(wxsz*0.8789)
yi = round(wysz*0.1172)
ya = round(wysz*0.8789)
BKGND=fix(mean(tvrd(xa,ya,xi,yi)))
FCOL = (BKGND gt 128) ? 0 : 255
;WidLabel0 = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
;widget_control,WidLabel0,get_value=label
x1 = round(wxsz*0.9277)
x2 = round(wxsz*0.8789)
y1 = round(wysz*0.9863)
y2 = round(wysz*0.9717)
y3 = round(wysz*0.9570)
y4 = round(wysz*0.9424)
xyouts,x1,y1,xtext,col=FCOL,align=0.5,/device
xyouts,x1,y2,ytext,col=FCOL,align=0.5,/device
if SigZ_ind ge 0 then xyouts,x1,y3,ztext0,col=FCOL,align=0.5,/device
if UnwZ_Err_ind ge 0 then xyouts,x2,y4,ztext1,col=FCOL,align=0.5,/device


end
;
;-----------------------------------------------------------------
;
pro OnAddAllLabels, Event
OnAddColorBarButton, Event
OnAddScaleBarButton, Event
OnAddLabelButton, Event
OnAddSigmaFilterButton, Event
end
;
;-----------------------------------------------------------------
;
pro OnRawFrameNumber, Event			;Displays a frame (selected by WID_SLIDER_RawFrameNumber) of Raw Data
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
if n_elements(CGroupParams) le 2 then begin
          z=dialog_message('Please load a data file')
          return      ; if data not loaded return
endif

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
X_ind=min(where(RowNames eq 'X Position'))
Y_ind=min(where(RowNames eq 'Y Position'))
GrX_ind=min(where(RowNames eq 'Group X Position'))
GrY_ind=min(where(RowNames eq 'Group Y Position'))
SigX_ind=min(where(RowNames eq 'Sigma X Pos Full'))
SigY_ind=min(where(RowNames eq 'Sigma Y Pos Full'))
GrSigX_ind=min(where(RowNames eq 'Group Sigma X Pos'))
GrSigY_ind=min(where(RowNames eq 'Group Sigma Y Pos'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
UnwZ_ind=min(where(RowNames eq 'Unwrapped Z'))
UnwZ_Err_ind=min(where(RowNames eq 'Unwrapped Z Error'))
UnwGrZ_ind=min(where(RowNames eq 'Unwrapped Group Z'))
UnwGrZ_Err_ind=min(where(RowNames eq 'Unwrapped Group Z Error'))
Gr_Size_ind = min(where(RowNames eq '24 Group Size'))
Frame_Number_ind = min(where(RowNames eq 'Frame Number'))
Label_ind = min(where(RowNames eq 'Label Set'))

dxmn = paramlimits[2,0]
dymn = paramlimits[3,0]
dxmx = paramlimits[2,1]
dymx = paramlimits[3,1]
wxsz=1024 & wysz=1024
loc=fltarr(wxsz,wysz)
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))
rmg=FLOAT(max((size(TotalRawData))[1:2])) / (max(xydsz))		; real magnification

XI=fix(floor(rmg*dxmn))>0
XA=fix(floor(rmg*(dxmx))) < ((size(TotalRawData))[1]-1)
YI=fix(floor(rmg*dymn)) > 0
YA=fix(floor(rmg*(dymx))) < ((size(TotalRawData))[2]-1)


	WidFrameNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawFrameNumber')
	widget_control,WidFrameNumber,get_value=RawFrameNumber
	WidPeakNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawPeakIndex')
	RawFileNameWidID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_RawFileName')
	Raw_File_Index=(widget_info(RawFileNameWidID,/DropList_Select))[0]
	raw_file_extension = thisfitcond.filetype ? '.tif' : '.dat'
	if Raw_File_Index eq -1 then begin
		z=dialog_message('Raw data file not found  '+RawFilenames)
		return
	endif
	file_conf=file_info(AddExtension(RawFilenames[Raw_File_Index],raw_file_extension))
	if ~file_conf.exists then begin
		z=dialog_message('Raw data file not found:  '+(RawFilenames[Raw_File_Index]+raw_file_extension))
		return
	endif

    reffilename=AddExtension(RawFilenames[Raw_File_Index],'.txt')
	;ReadThisFitCond, reffilename, pth, filen, ini_filename, thisfitcond
	clip=ReadData(RawFilenames[Raw_File_Index],thisfitcond,RawFrameNumber,1)
	Fimage=clip[XI : XA, YI : YA]
	fimagex=fix(float(XA-XI+1)*mgw/rmg)
	fimagey=fix(float(YA-YI+1)*mgw/rmg)
	Fimage=Congrid(Fimage,fimagex,fimagey)
	tv,bytarr(wxsz,wysz)
	tvscl,Fimage
	image=fimage				;tvrd(true=1)
	AdjustContrastnDisplay, Event
	TVscales,wxsz,wysz,mgw,nm_per_pixel

	xyouts,0.65,0.02,string(RawFrameNumber)+'#',/normal

	SetRawSliders,Event
	;if (widget_info(WidPeakNumber,/SENSITIVE)) eq 0 then return

	widget_control,WidFrameNumber,get_value=RawFrameNumber
	widget_control,WidPeakNumber,get_value=PeakNumber
	Raw_File_Index=(widget_info(RawFileNameWidID,/DropList_Select))[0]
	label=max(CgroupParams[Label_ind,*] eq 0)	?	0	:	(Raw_File_Index+1)
	peak_index=min(where((CgroupParams[Label_ind,*] eq label) and (CgroupParams[FrNum_ind,*] eq RawFrameNumber) and (CgroupParams[10,*] eq PeakNumber),cnt))
	if cnt eq 1 then ReloadPeakColumn,peak_index

end
;
;-----------------------------------------------------------------
;
pro OnRawPeakIndex, Event			;Shows the peak location and runs FindnWackaPeak (including display of fit parameters) for a peak selected by WID_SLIDER_RawPeakIndex
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius

if n_elements(CGroupParams) le 2 then begin
          z=dialog_message('Please load a data file')
          return      ; if data not loaded return
endif

FitOK_ind = min(where(RowNames eq 'FitOK'))                                ; CGroupParametersGP[8,*] - Original FitOK

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
PkInd_ind = min(where(RowNames eq 'Peak Index of Frame'))

X_ind=min(where(RowNames eq 'X Position'))
Y_ind=min(where(RowNames eq 'Y Position'))
GrX_ind=min(where(RowNames eq 'Group X Position'))
GrY_ind=min(where(RowNames eq 'Group Y Position'))
SigX_ind=min(where(RowNames eq 'Sigma X Pos Full'))
SigY_ind=min(where(RowNames eq 'Sigma Y Pos Full'))
GrSigX_ind=min(where(RowNames eq 'Group Sigma X Pos'))
GrSigY_ind=min(where(RowNames eq 'Group Sigma Y Pos'))
Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
UnwZ_ind=min(where(RowNames eq 'Unwrapped Z'))
UnwZ_Err_ind=min(where(RowNames eq 'Unwrapped Z Error'))
UnwGrZ_ind=min(where(RowNames eq 'Unwrapped Group Z'))
UnwGrZ_Err_ind=min(where(RowNames eq 'Unwrapped Group Z Error'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size_ind = min(where(RowNames eq '24 Group Size'))
Frame_Number_ind = min(where(RowNames eq 'Frame Number'))
LabelSet_ind = min(where(RowNames eq 'Label Set'))                        ; CGroupParametersGP[26,*] - Label Number

Wl_ind = min(where(RowNames eq 'Wavelength (nm)'))

	WidFrameNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawFrameNumber')
	widget_control,WidFrameNumber,get_value=RawFrameNumber
	WidPeakNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawPeakIndex')
	widget_control,WidPeakNumber,get_value=PeakNumber
	RawFileNameWidID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_RawFileName')
	Raw_File_Index=(widget_info(RawFileNameWidID,/DropList_Select))[0]
	raw_files=where(RawFilenames ne '',raw_cnt)
	raw_file_extension = thisfitcond.filetype ? '.tif' : '.dat'


	if Raw_File_Index eq -1 then begin
		z=dialog_message('Raw data file not found  '+RawFilenames)
		return
	endif
	file_conf=file_info(AddExtension(RawFilenames[Raw_File_Index],raw_file_extension))
	if ~file_conf.exists then begin
		z=dialog_message('Raw data file not found:  '+(AddExtension(RawFilenames[Raw_File_Index],raw_file_extension)))
		return
	endif
	OnRawFrameNumber, Event
	raw_ind = (raw_cnt eq 1)	?	0	:	Raw_File_Index+1
	cgrp_index=where(((CGroupParams[FrNum_ind,*] eq RawFrameNumber) and (CGroupParams[PkInd_ind,*] eq PeakNumber) and (CGroupParams[LabelSet_ind,*] eq raw_ind)),cnt)
	if cnt eq 0 then begin
		print,'Peak not found:  file',(RawFilenames[Raw_File_Index]+raw_file_extension),',  Frame:',RawFrameNumber,'   Peak:',PeakNumber
		black_region=intarr(950,50)
		tv,black_region,35,490
		msg_str='Peak Not Found:  Frame:'+strtrim(RawFrameNumber,2)+',  Peak:'+strtrim(PeakNumber,2)
		xyouts,65,500,msg_str, CHARSIZE=3,/device
		return
	endif
	fitOK = CGroupParams[FitOK_ind,cgrp_index[0]]
	if (fitOK ne 1) and (fitOK ne 2) then begin
		print,'Bad Fit,    FitOK = ',fitOK
		black_region=intarr(950,50)
		tv,black_region,35,490
		msg_str='Bad Fit:  Frame:'+strtrim(RawFrameNumber,2)+',  Peak:'+strtrim(PeakNumber,2)+',  FitOK = '+strtrim(fitOK,2)
		xyouts,65,500,msg_str, CHARSIZE=3,/device
		return
	endif
	clip=ReadData(RawFilenames[Raw_File_Index], thisfitcond, RawFrameNumber, 1)
	d=thisfitcond.MaskSize & dd=2*d+1
	SigmaSym = thisfitcond.SigmaSym
	PeakX=CGroupParams[X_ind,cgrp_index[0]]
	PeakY=CGroupParams[Y_ind,cgrp_index[0]]
	clipsize=size(clip)
	peakx=(peakx<(clipsize[1]-d-1))>d
	peaky=(peaky<(clipsize[2]-d-1))>d
	region=clip[peakx-d:peakx+d,peaky-d:peaky+d]
	Dispxy=[0,0]
	DisplaySet=0

	if SigmaSym le 1 then begin
		peakparams = {twinkle,frameindex:0l,peakindex:0,fitOK:0,peakx:0.0,peaky:0.0,A:fltarr(7),sigma1:fltarr(7),sigma2:fltarr(7),chisq:0.0,Nphot:0l}
		peakparams.A=[0.0,1.0,1.2,1.2,d,d,0.]
		fita = [1,1,1,1,1,1]
		FindnWackaPeak, clip, d, peakparams, fita, result, thisfitcond, DisplaySet, peakx, peaky, criteria, Dispxy, 1
		A=peakparams.a
		cx = a[4]
		cy = a[5]
		widx = A[2]
		widy = A[3]
	endif else begin
		peakparams = {twinkle_z, frameindex:0l, peakindex:0l, fitOK:1, peakx:0.0, peaky:0.0, peak_widx:0.0, peak_widy:0.0, A:fltarr(6), sigma:fltarr(6), chisq:0.0, Nphot:0l}
		peakparams.A=[0.0,1.0,d,d,0.,1.0]
		fita = [1,1,1,1,1,1]
		FindnWackaPeak_AstigZ, clip, d, peakparams, fita, result, thisfitcond, aa, DisplaySet, peakx, peaky, criteria, Dispxy, 0		;Find, fit, and remove the peak with biggest criteria
		A=peakparams.a
		cx = a[2]
		cy = a[3]
		nb = n_elements(aa)/2
		b = aa[0:nb-1]
		c = aa[nb:*]
		widx = poly(a[4],b)
		widy = poly(a[4],c)
	endelse

	scl=250.0/(max((region-min(region)))>max((result-min(result))));
	gscl=15.
	xtvpeak=(dd*gscl*dispxy[0] mod (fix(1024.0/dd/gscl)*dd*gscl))
	ytvpeak=1024-dd*gscl
	tv,50+scl*rebin(region-min(region),dd*gscl,dd*gscl,/sample)<255,xtvpeak,ytvpeak				;tv slected peak region
	tv,50+scl*rebin(result-min(result),dd*gscl,dd*gscl,/sample)<255,xtvpeak+dd*gscl+2,ytvpeak		;tv resulting fit

	plots,gscl*(cx+0.5)+xtvpeak,gscl*(cy+0.5)+ytvpeak,psym=1,/device,col=0	;mark the center of data peak with plus
	plots,gscl*(cx+0.5)+xtvpeak,gscl*(cy+0.5)+ytvpeak,psym=3,/device		;mark center of data peak, put dot in middle
	plots,gscl*(cx+0.5)+xtvpeak,gscl*(cy+0.5)+ytvpeak+dd*gscl,psym=1,/device,col=0	;mark center of peak fit
	plots,gscl*(cx+0.5)+xtvpeak,gscl*(cy+0.5)+ytvpeak+dd*gscl,psym=3,/device		;mark center of peak fit
	xpos=(findgen(dd*gscl)/gscl-cx-0.5)#replicate(1,dd*gscl)
	ypos=replicate(1.,dd*gscl)#(findgen(dd*gscl)/gscl-cy-0.5)
	minclp=min(clip[PeakX-d:PeakX+d,PeakY-d:PeakY+d])
	tv,50+scl*rebin(clip[PeakX-d:PeakX+d,PeakY-d:PeakY+d]-minclp,dd*gscl,dd*gscl,/sample)<255,xtvpeak+2*dd*gscl+4,ytvpeak		;tv slected residual after peak removal
	str1='WidX = '+strtrim(widx,2)+'     WidY = '+strtrim(widy,2)
	if SigmaSym eq 2 then str1 = str1	+'     Z = '+strtrim(A[4],2)
	str2='Amp = '+strtrim(A[1],2)+'     Offset = '+strtrim(A[0],2)
	xyouts,xtvpeak+dd*gscl+2,ytvpeak+(dd-1)*gscl,str2,/device
	xyouts,xtvpeak+dd*gscl+2,ytvpeak+(dd-2)*gscl,str1,/device

	Fid_Outl_Sz=5
	Fid_Outline_color=1
	Display_single_fiducial_outline, Event, peakx, peaky, Fid_Outl_Sz, Fid_Outline_color

	Raw_File_Index = (widget_info(RawFileNameWidID,/DropList_Select))[0]
	label = max(CgroupParams[LabelSet_ind,*] eq 0)	?	0	:	(Raw_File_Index+1)
	peak_index = min(where((CgroupParams[LabelSet_ind,*] eq label) and $
							(CgroupParams[FrNum_ind,*] eq RawFrameNumber) and $
							(CgroupParams[PkInd_ind,*] eq PeakNumber),cnt))
	if cnt eq 1 then ReloadPeakColumn,peak_index

;	common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
	if Wl_ind ge 0 then begin		; if the data had been processed for XY spectral decomposition, show the spectral decomposition results
		SVDC, cal_spectra, Wsp, Usp, Vsp
		N = N_ELEMENTS(Wsp)
		WPsp = FLTARR(N, N)
		sc_mag=2.0
		FOR K = 0, N-1 DO   IF ABS(Wsp(K)) GE 1.0e-5 THEN WPsp(K, K) = 1.0/Wsp(K)
		ReadThisFitCond, (lab_filenames[1]+'.txt'), pth, filen, ini_filename, thisfitcond1
		clip=ReadData(lab_filenames[1],thisfitcond1,RawFrameNumber,1)	;Reads thefile and returns data (bunch of frames) in (units of photons)
		Dispxy=[1,-2]
		process_display=1
		SubtractBacground,clip,peakx,peaky,1,sp_d
		ExtractSpectrum, clip, sp_d, peakx, peaky, spectrum, process_display, Dispxy, sc_mag				;
		AnalyzeSpectrum, spectrum, Vsp,WPsp,Usp, process_display, Dispxy, sc_mag, coeffs, resid
		print, resid, transpose(coeffs)
	endif

end
;
;-----------------------------------------------------------------
;
pro OnPeakOverlayAllCentersButton, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
COMMON COLORS, R_orig, G_orig, B_orig, R_curr, G_curr, B_curr
if n_elements(CGroupParams) le 2 then begin
          z=dialog_message('Please load a data file')
          return      ; if data not loaded return
endif

wxsz=1024 & wysz=1024
dxmn = paramlimits[2,0]
dymn = paramlimits[3,0]
dxmx = paramlimits[2,1]
dymx = paramlimits[3,1]
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))

FilterIt
FilteredPeakIndex=where(filter eq 1,cnt)
if cnt le 0 then return

TVLCT,R0,G0,B0,/GET
TVLCT, [[255], [0], [0]], 0
TVLCT, [[0], [255], [0]], 1
TVLCT, [[0], [0], [255]], 2
x_centers = mgw*(CGroupParams[2,FilteredPeakIndex]-dxmn)
y_centers = mgw*(CGroupParams[3,FilteredPeakIndex]-dymn)
plots,x_centers,y_centers,/device,psym=1,color=2

TVLCT,R0,G0,B0
end
;
;-----------------------------------------------------------------
;
pro OnPeakOverlayFrameCentersButton, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max; TransformEngine : 0=Local, 1=Cluster
COMMON COLORS, R_orig, G_orig, B_orig, R_curr, G_curr, B_curr
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number

if n_elements(CGroupParams) le 2 then begin
          z=dialog_message('Please load a data file')
          return      ; if data not loaded return
endif
WidFrameNumber = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_RawFrameNumber')
widget_control,WidFrameNumber,get_value=RawFrameNumber

wxsz=1024 & wysz=1024
dxmn = paramlimits[2,0]
dymn = paramlimits[3,0]
dxmx = paramlimits[2,1]
dymx = paramlimits[3,1]
mgw=(wxsz /(dxmx-dxmn))<(wysz /(dymx-dymn))

FilterIt
filter_fr = filter and (CGroupParams[FrNum_ind,*] eq RawFrameNumber)
FilteredPeakIndex=where(filter_fr eq 1,cnt_fr)
if cnt_fr le 0 then return

TVLCT,R0,G0,B0,/GET
TVLCT, [[255], [0], [0]], 0
TVLCT, [[0], [255], [0]], 1
TVLCT, [[0], [0], [255]], 2
x_centers = mgw*(CGroupParams[2,FilteredPeakIndex]-dxmn)
y_centers = mgw*(CGroupParams[3,FilteredPeakIndex]-dymn)
plots,x_centers,y_centers,/device,psym=1,color=2

TVLCT,R0,G0,B0
end
;
;-----------------------------------------------------------------
;
pro OnLabelDropList, Event			;Change color scale sliders to match label
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w

SelectedLabel=event.index
WidSldTopID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Top')
widget_control,WidSldTopID,set_value=labelContrast[0,selectedlabel]
WidSldGammaID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Gamma')
widget_control,WidSldGammaID,set_value=labelContrast[1,selectedlabel]
WidSldBotID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Bot')
widget_control,WidSldBotID,set_value=labelContrast[2,selectedlabel]

end
;
;-----------------------------------------------------------------
;
pro Initialization_PeakSelector_Main, wWidget, _EXTRA=_VWBExtra_	; initialises menue tables, droplists, IDL starting directory, material and other parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common Offset, PkWidth_offset
common Zdisplay, Z_scale_multiplier, vbar_top
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max;
;TransformEngine : 0=Local, 1=Cluster, TransformEngine, grouping_gap,grouping_radius100; TransformEngine : 0=Local, 1=Cluster
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
Common Multiple_PALM_TIFFs, DoPurge_mTIFFs, Purge_RowNames_mTIFFs, Purge_Params_mTIFFs
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
	Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope

CPU,TPOOL_NTHREADS=1
help, /structure, !CPU
;WID_BASE_0_PeakSelector resizing to fit computer screen
disp_xy=GET_SCREEN_SIZE()
;disp_xy[1]=600
peakselector_x=1622		; x-size, pixels
peakselector_y=1115		; y-size, pixels
dx = peakselector_x < (disp_xy[0]-20)
dy = peakselector_y < (disp_xy[1]-20)
if (dx lt peakselector_x) or (dy lt peakselector_y) then $
	widget_control,wWidget,Xsize=dx, SCR_XSIZE=peakselector_x, Ysize=dy, SCR_YSIZE=peakselector_y

def_w=!D.WINDOW
cd, current=cur_dir
if strpos(cur_dir,'PeakSelector') ge 0 then begin
	if !VERSION.OS_family eq 'unix' then pref_set,'IDL_MDE_START_DIR',cur_dir,/commit	else	pref_set,'IDL_WDE_START_DIR',cur_dir,/commit
endif
idl_pwd=cur_dir
!PATH = idl_pwd + ';' + !PATH

ini_filename = !VERSION.OS_family eq 'unix' ? pref_get('IDL_MDE_START_DIR')+'/PeakSelector.ini' : pref_get('IDL_WDE_START_DIR')+'\PeakSelector.ini'
Initialization_PeakSelector, wWidget, _EXTRA=_VWBExtra_

end
;
;-----------------------------------------------------------------
;
pro Initialization_PeakSelector, wWidget, _EXTRA=_VWBExtra_	; initialises menue tables, droplists, IDL starting directory, material and other parameters
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common Offset, PkWidth_offset
common Zdisplay, Z_scale_multiplier, vbar_top
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max
; TransformEngine : 0=Local, 1=Cluster, TransformEngine, grouping_gap,grouping_radius100; TransformEngine : 0=Local, 1=Cluster
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
Common Multiple_PALM_TIFFs, DoPurge_mTIFFs, Purge_RowNames_mTIFFs, Purge_Params_mTIFFs
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
	Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope


Rundat_Filename = ''

Wid_ID_allow_bridge = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Allow_Bridge')
allow_bridge = widget_info(Wid_ID_allow_bridge,/button_set)
bridge_exists = 0
n_elem_CGP = 0

Wid_ID_hist_log_X = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Hist_Log_X')
hist_log_x = widget_info(Wid_ID_hist_log_X,/button_set)
Wid_ID_hist_log_Y = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Hist_Log_Y')
hist_log_y = widget_info(Wid_ID_hist_log_Y,/button_set)
Recalculate_Histograms_id=widget_info(wWidget,FIND_BY_UNAME='WID_BUTTON_Redraw')
widget_control,Recalculate_Histograms_id,set_button=1

;if !VERSION.OS_family eq 'unix' then	ConfigureEnvironment			;loads dlms for Cuda

z_media_multiplier=1.00			; depends on objective NA and media index. This is ratio which determines by how much the focal plane of the (air) objective shifts in the media for a unit shift of the objective along the axis.
lambda_vac=590.0
nd_water=1.33
nd_oil=1.515
wfilename=''
wind_range=220.0
nmperframe = 20.0			; nm per frame. calibration using piezo parameters

z_unwrap_coeff = [0.0,0.0,0.0]
ellipticity_slopes = [0.0,0.0,0.0,0.0]
AnchorFile=''
Fid_Outl_Sz = 0.5
AutoDisp_Sel_Fids = 1
Disp_Fid_IDs = 1
AnchPnts_MaxNum=500
AnchorPnts=dblarr(6,AnchPnts_MaxNum)
ZPnts=dblarr(3,AnchPnts_MaxNum)

PkWidth_offset=1.0
hue_scale=320.0									;hue scale in degrees for full range
nm_per_pixel=133.3		; opt_mag=120, pixel=16 nm, nm_per_pixel=16/120=133.3
Z_scale_multiplier=1.0	; Z-scale multiplier for X-Z and Y-Z swapped displays
vbar_top=0.003			; molecular probability (for the color bar top value)
Max_Prob_2DPALM=0.05	; molecular probability (for the color bar top value)

n_br_max = 1024;
hist_nbins = 128;  number of histogram bins
n_cluster_nodes_max = 1024

Cust_TIFF_Pix_X = 1024	; image window size for "Custom TIFF"
Cust_TIFF_Pix_Y = 1024	; image window size for "Custom TIFF"
Cust_TIFF_XY_subvol_nm = 100.0  ; Z-size of the Gaussian cloud for custom TIFF render
Cust_TIFF_Z_subvol_nm = 100.0  	; X-Y-size of the Gaussian cloud for custom TIFF render

SaveASCII_Filter = 0	; 0 for Peak Filtered; 1 for Group Filtered
SaveASCII_units  = 0	; 0 for pixels, 1 for nm
SaveASCII_ParamChoice = 0; 0 for all peaks, 1 for selected list
SaveASCII_ParamList = [0,1,2,3,4,5,6,9,13,14,16]	; indecis of the paraments from the the RowNames list to be saved

ImportASCII_units  = 0	; 0 for pixels, 1 for nm. The parameters related to lateral coordinates (X, Y position, X, Y Sigmas etc)
ImportASCII_nm_per_pixel = 133.33	; nm per pixel for imported ASCII data
ImportASCII_ParamList = [0,1,2,3,4,6,9,13,14,16]	; default list of indices of the parameters from the RowLbls list corresponding to the columns of the data to be imported.

iPALM_MacroParameters_XY = [100.0,	0.4,	0.4,	40.0,	0.1,	1.3,	1.4,	0.01,	0.6,	0.01]
iPALM_MacroParameters_R = [100.0,	0.3,	0.3,	30.0,	0.1,	5.0,	5.0,	0.01,	0.2,	0.01]

CGrpSize = 49	; =27 for 2D PALM, = 49 for 3D PALM
ParamLimits=fltarr(CGrpSize,5)

grouping_gap=6
grouping_radius100=40		; in % of pixel number 40 here means that the grouping radius is 0.40*pixel
TransformEngine = (!VERSION.OS_family eq 'unix')? 1 : 0			;Set the default value to 0=Local for Windows, and 1=Cluster for UNIX

thisfitcond={			$
f_info:'',				$
zerodark:100.,			$
xsz:256L,				$
ysz:256L,				$
Nframesmax:1500ul,		$
Frm0:0ul,				$
FrmN:1499ul,			$
Thresholdcriteria:37.8,	$
filetype:0,				$
LimBotA1:12.6,			$
LimTopA1:10000.,		$
LimBotSig:0.5,			$
LimTopSig:3.5,			$
LimChiSq:1500.,			$
Cntpere:13.878,			$
maxcnt1:125,			$
maxcnt2:0,				$
fliphor:0,				$
flipvert:0,				$
SigmaSym:0,				$
MaskSize:5,				$
GaussSig:1.0,			$
MaxBlck:512,			$
LocalizationMethod:0,	$
SparseOversampling:9,	$
SparseLambda:1e11,		$
SparseDelta:1e-5,		$
SpError:0.3,			$
SpMaxIter:1e3			}

ini_file_info=FILE_INFO(ini_filename)
if ~(ini_file_info.exists) then return
print, 'Loading preferences from the file:    ',ini_filename

	ini_file = ini_filename

	Initialize_Common_parameters, ini_file

	ReloadMainTableColumns, wWidget

CGroupParams=[0]
TotalRawData=[0]
DIC=[0]

labelcontrast=intarr(3,5)			;stretch top, gamma, stretch bottom	rows x blank, red, green, blue, DIC columns
for i =0, 3 do labelcontrast[*,i]=[500,500,0]
labelcontrast[*,4]=[1000,1000,0]

WidDrFunct = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Function')
widget_control,WidDrFunct, Set_Droplist_Select=1
WidDrFilter = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Filter')
widget_control,WidDrFilter, Set_Droplist_Select=1
WidDrAccum = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Accumulate')
widget_control,WidDrAccum, Set_Droplist_Select=1

WidDL_LabelID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Label')
selectedlabel = widget_info(WidDL_LabelID,/DropList_Select)
WidSldTopID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Top')
widget_control,WidSldTopID,set_value=labelContrast[0,selectedlabel]
WidSldGammaID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Gamma')
widget_control,WidSldGammaID,set_value=labelContrast[1,selectedlabel]
WidSldBotID = Widget_Info(wWidget, find_by_uname='WID_SLIDER_Bot')
widget_control,WidSldBotID,set_value=labelContrast[2,selectedlabel]

XZ_swap_menue_ID = Widget_Info(wWidget, find_by_uname='W_MENU_SwapXZ')
widget_control,XZ_swap_menue_ID,set_button=0

YZ_swap_menue_ID = Widget_Info(wWidget, find_by_uname='W_MENU_SwapYZ')
widget_control,YZ_swap_menue_ID,set_button=0

Z_unwrap_swap_menue_ID = Widget_Info(wWidget, find_by_uname='W_MENU_65')
widget_control,Z_unwrap_swap_menue_ID,set_button=0,set_value='Swap Z with Unwrapped Z'
ZvsV_Slope = nmperframe/0.040/2.0  ; nm per V sample movement sensitivity, need to divide by 2 because in calibration we measure the phase difference which is double the distance.

end
;
;-----------------------------------------------------------------
;
pro Initialize_Common_parameters, ini_file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
common XY_spectral, lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames
common spectra_data, sp_win, sp_2D_data, sp_2D_image, spectra,  sp_dispersion,  sp_offset, sp_calc_method, BG_subtr_params,  RawFrameNumber, Peak_Indecis, RawPeakIndex, sp_filename
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common Offset, PkWidth_offset
common Zdisplay, Z_scale_multiplier, vbar_top
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max
; TransformEngine : 0=Local, 1=Cluster, TransformEngine, grouping_gap,grouping_radius100; TransformEngine : 0=Local, 1=Cluster
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
common Custom_TIFF, Cust_TIFF_window,  Cust_TIFF_3D, Cust_TIFF_Accumulation, Cust_TIFF_Filter, Cust_TIFF_Function, cust_nm_per_pix, Cust_TIFF_Pix_X, Cust_TIFF_Pix_Y,$
		Cust_TIFF_volume_image, Cust_TIFF_max,Cust_TIFF_Z_multiplier, Cust_TIFF_Z_start, Cust_TIFF_Z_stop, Cust_TIFF_XY_subvol_nm, Cust_TIFF_Z_subvol_nm
common ImportASCII, ImportASCII_Filename, ImportASCII_nm_per_pixel, ImportASCII_units, ImportASCII_ParamList
common iPALM_macro_parameters, iPALM_MacroParameters_XY, iPALM_MacroParameters_R,  Astig_MacroParameters
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
Common Multiple_PALM_TIFFs, DoPurge_mTIFFs, Purge_RowNames_mTIFFs, Purge_Params_mTIFFs
Common Multiple_PALM_Slabs, mSlab_Filenames, DoFilter, DoAutoFindFiducials, DoDriftCottect, DoGrouping, DoPurge, DoScaffoldRegister, $
	Filter_RowNames, Filter_Params, Purge_RowNames_mSlabs, Purge_Params_mSlabs, AutoFindFiducial_Params, Scaffold_Fid_FName, Scaffold_Fid, ZStep_mSlabs
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope

ini_file_info=FILE_INFO(ini_file)
if ~(ini_file_info.exists) then return
ini_filename=ini_file
;print, 'ICP: Loading thisfitcond from the file:    ',ini_filename
close,1
openr, 1, ini_filename
thisfitcond={			$
		f_info:'',				$
		zerodark:100.,			$
		xsz:256L,				$
		ysz:256L,				$
		Nframesmax:1500ul,		$
		Frm0:0ul,				$
		FrmN:1499ul,			$
		Thresholdcriteria:10.0,	$
		filetype:0,				$
		LimBotA1:12.6,			$
		LimTopA1:10000.,		$
		LimBotSig:0.5,			$
		LimTopSig:3.5,			$
		LimChiSq:1500.,			$
		Cntpere:15.1,			$
		maxcnt1:126,			$
		maxcnt2:0,				$
		fliphor:0,				$
		flipvert:0,				$
		SigmaSym:0,				$
		MaskSize:5,				$
		GaussSig:1.0,			$
		MaxBlck:512,			$
		LocalizationMethod:0,	$
		SparseOversampling:9,	$
		SparseLambda:1e11,		$
		SparseDelta:1e-5,		$
		SpError:0.3,			$
		SpMaxIter:1e3			}

CATCH, Error_status
line_i=''
i=0
IF Error_status NE 0 THEN BEGIN
      PRINT, 'ICP: Incorrect INI format. Could not parse the settings from the line:   ',line_i
      PRINT, byte(line_i);
      window,12,xsize=1000,ysize=100,xpos=50,ypos=250,Title='LoadThiFitCond: Incorrect INI format'
	  msg0= string(ini_filename)
      msg1='ICP: Cant parse from the line:   ' + line_i + '    i=' + strtrim(i)
      xyouts,0.05,0.8,!ERROR_STATE.MSG,CHARSIZE=1.0,/NORMAL
      xyouts,0.05,0.6,!ERROR_STATE.SYS_MSG,CHARSIZE=1.0,/NORMAL
      xyouts,0.05,0.4,msg0,CHARSIZE=1.0,/NORMAL
	  xyouts,0.05,0.2,msg1,CHARSIZE=1.0,/NORMAL
	  wset,def_w	; set back to main (default) window
	  CATCH, /CANCEL
	  close,1
	  return
ENDIF
;print, 'ICP: Loading INI settings:    ',ini_filename
while (~ EOF(1)) and (strmid(line_i,0,7) ne 'RowLbls') do begin
	c=0b
	cc=32b
	while c ne 10b do begin
		readu,1,c
		cc=[[cc],c]
	endwhile
	line_i=strtrim(cc,2)
	;print,line_i
	IF LMGR(/VM) eq 0 then begin 		; if IDL is not in Virtual Machine mode, then command EXECUTE can be performed
		;print,i,line_i
		if (strmid(line_i,0,7) ne 'RowLbls') then x=execute(line_i)
	endif else begin					; otherwise do the following lengthy and not so elegant checking
		var_name = strtrim(strmid(line_i,0,strpos(line_i,'=')),2)
		var_value_str0 = strtrim(strmid(line_i,strpos(line_i,'=')+1),2)
	    space_pos=strpos(var_value_str0,' ')
	    var_value_str = (space_pos gt 0) ? strtrim(strmid(var_value_str0,0,space_pos),2) : var_value_str0
	    semicolon_pos=strpos(var_value_str0,';')
	    var_value_str = (semicolon_pos gt 0) ? strtrim(strmid(var_value_str0,0,semicolon_pos),2) : var_value_str0
	    semicolon_pos=strpos(var_value_str0,'	')
	    var_value_str = (semicolon_pos gt 0) ? strtrim(strmid(var_value_str0,0,semicolon_pos),2) : var_value_str0
	    semicolon_pos=strpos(var_value_str0,string(13B))
	    var_value_str = (semicolon_pos gt 0) ? strtrim(strmid(var_value_str0,0,semicolon_pos),2) : var_value_str0
	    ;print, var_name, var_value_str

	    if var_name eq 'CGrpSize' then begin
	    	reads, var_value_str, var_value, FORMAT='(I)'
	    	CGrpSize = fix(var_value)
	    endif
		if var_name eq 'thisfitcond.f_info' then thisfitcond.f_info = var_value_str
		if var_name eq 'thisfitcond.zerodark' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.zerodark = float(var_value)
		endif
		if var_name eq 'thisfitcond.xsz' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.xsz = fix(var_value)
		endif
		if var_name eq 'thisfitcond.ysz' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.ysz = fix(var_value)
		endif
		if var_name eq 'thisfitcond.Nframesmax' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.Nframesmax = ulong(var_value)
		endif
		if var_name eq 'thisfitcond.Frm0' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.Nframesmax = ulong(var_value)
		endif
		if var_name eq 'thisfitcond.FrmN' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.FrmN = ulong(var_value)
		endif
		if var_name eq 'thisfitcond.Thresholdcriteria' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.Thresholdcriteria = float(var_value)
		endif
		if var_name eq 'thisfitcond.filetype' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.filetype = fix(var_value)
		endif
		if var_name eq 'thisfitcond.Thresholdcriteria' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.Thresholdcriteria = float(var_value)
		endif
		if var_name eq 'thisfitcond.LimBotA1' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.LimBotA1 = float(var_value)
		endif
		if var_name eq 'thisfitcond.LimTopA1' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.LimTopA1 = float(var_value)
		endif
		if var_name eq 'thisfitcond.LimBotSig' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.LimBotSig = float(var_value)
		endif
		if var_name eq 'thisfitcond.LimTopSig' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.LimTopSig = float(var_value)
		endif
		if var_name eq 'thisfitcond.Cntpere' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.Cntpere = float(var_value)
		endif
		if var_name eq 'thisfitcond.maxcnt1' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.maxcnt1 = fix(var_value)
		endif
		if var_name eq 'thisfitcond.maxcnt2' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.maxcnt2 = fix(var_value)
		endif
		if var_name eq 'thisfitcond.fliphor' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.fliphor = fix(var_value)
		endif
		if var_name eq 'thisfitcond.flipvert' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.flipvert = fix(var_value)
		endif
		if var_name eq 'thisfitcond.SigmaSym' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.SigmaSym = fix(var_value)
		endif
		if var_name eq 'thisfitcond.MaskSize' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.MaskSize = fix(var_value)
		endif
		if var_name eq 'thisfitcond.GaussSig' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.GaussSig = float(var_value)
		endif
		if var_name eq 'thisfitcond.MaxBlck' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.MaxBlck = fix(var_value)
		endif
		if var_name eq 'thisfitcond.LocalizationMethod' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.LocalizationMethod = fix(var_value)
		endif
		if var_name eq 'thisfitcond.SparseOversampling' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.SparseOversampling = fix(var_value)
		endif
		if var_name eq 'thisfitcond.SparseLambda' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.SparseLambda = float(var_value)
		endif
		if var_name eq 'thisfitcond.SparseDelta' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.SparseDelta = float(var_value)
		endif
		if var_name eq 'thisfitcond.SpError' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.SpError = float(var_value)
		endif
		if var_name eq 'thisfitcond.SpMaxIter' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.SpMaxIter = float(var_value)
		endif

		if var_name eq 'TransformEngine' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			TransformEngine = fix(var_value)
		endif

		if var_name eq 'nm_per_pixel' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			nm_per_pixel = float(var_value)
		endif
		if var_name eq 'grouping_gap' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			grouping_gap = fix(var_value)
		endif
		if var_name eq 'grouping_radius100' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			grouping_radius100 = fix(var_value)
		endif
		if var_name eq 'hist_nbins' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			hist_nbins = fix(var_value)
		endif
		if var_name eq 'lambda_vac' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			lambda_vac = float(var_value)
		endif
		if var_name eq 'nd_water' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			nd_water = float(var_value)
		endif
		if var_name eq 'nd_oil' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			nd_oil = float(var_value)
		endif
		if var_name eq 'wind_range' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			wind_range = float(var_value)
		endif
		if var_name eq 'GS_radius' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			GS_radius = float(var_value)
		endif
		if var_name eq 'Fid_Outl_Sz' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			Fid_Outl_Sz = float(var_value)
		endif
		if var_name eq 'AutoDisp_Sel_Fids' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			AutoDisp_Sel_Fids = fix(var_value)
		endif
		if var_name eq 'Disp_Fid_IDs' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			Disp_Fid_IDs = fix(var_value)
		endif
		if var_name eq 'AnchPnts_MaxNum' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			AnchPnts_MaxNum = fix(var_value)
		endif
		if var_name eq 'PkWidth_offset' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			PkWidth_offset = float(var_value)
		endif
		if var_name eq 'hue_scale' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			hue_scale = float(var_value)
		endif
		if var_name eq 'Z_scale_multiplier' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			Z_scale_multiplier = float(var_value)
		endif
		if var_name eq 'vbar_top' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			vbar_top = float(var_value)
		endif
		if var_name eq 'Max_Prob_2DPALM' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			Max_Prob_2DPALM = float(var_value)
		endif

		if var_name eq 'SaveASCII_Filter' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			SaveASCII_Filter = fix(var_value)
		endif
		if var_name eq 'SaveASCII_units' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			SaveASCII_units = fix(var_value)
		endif
		if var_name eq 'SaveASCII_ParamChoice' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			SaveASCII_ParamChoice = fix(var_value)
		endif

		if var_name eq 'ImportASCII_nm_per_pixel' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			ImportASCII_nm_per_pixel = float(var_value)
		endif
		if var_name eq 'ImportASCII_units' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			ImportASCII_units = fix(var_value)
		endif


		if var_name eq 'ellipticity_slopes' then begin
			ellipticity_slopes=fltarr(4)
			var_value_str1 = strmid(var_value_str,strpos(var_value_str,'[')+1)
			var_value_str2 = strmid(var_value_str1,0,strpos(var_value_str1,']'))
				var_value_str0 = strmid(var_value_str2,0,strpos(var_value_str2,','))
				reads, var_value_str0, var_value, FORMAT='(F)' & ellipticity_slopes[0] = float(var_value)
					var_value_str2 = strmid(var_value_str2,strpos(var_value_str2,',')+1)
				var_value_str0 = strmid(var_value_str2,0,strpos(var_value_str2,','))
				reads, var_value_str0, var_value, FORMAT='(F)' & ellipticity_slopes[1] = float(var_value)
					var_value_str2 = strmid(var_value_str2,strpos(var_value_str2,',')+1)
				var_value_str0 = strmid(var_value_str2,0,strpos(var_value_str2,','))
				reads, var_value_str0, var_value, FORMAT='(F)' & ellipticity_slopes[2] = float(var_value)
					var_value_str2 = strmid(var_value_str2,strpos(var_value_str2,',')+1)
				reads, var_value_str2, var_value, FORMAT='(F)' & ellipticity_slopes[3] = float(var_value)
		endif

		if var_name eq 'SaveASCII_ParamList' then begin
			first_br=STRPOS(var_value_str,'[')
			var_val_str1=STRMID(var_value_str,(first_br+1))
			second_br=STRPOS(var_val_str1,']')
			var_val_str2=STRMID(var_val_str1,0,second_br)
			comma_pos=STRPOS(var_val_str2,',')
			cc=0
			while comma_pos gt 0 do begin
				if cc eq 0 then SaveASCII_ParamList =fix(STRMID(var_val_str2,0,second_br)) else SaveASCII_ParamList  = [SaveASCII_ParamList , fix(STRMID(var_val_str2,0,second_br))]
				var_val_str2=STRMID(var_val_str2,(comma_pos+1))
				comma_pos=STRPOS(var_val_str2,',')
				cc++
			endwhile
			SaveASCII_ParamList = [SaveASCII_ParamList , fix(STRMID(var_val_str2,0,second_br))]
		endif

		if var_name eq 'ImportASCII_ParamList' then begin
			first_br=STRPOS(var_value_str,'[')
			var_val_str1=STRMID(var_value_str,(first_br+1))
			second_br=STRPOS(var_val_str1,']')
			var_val_str2=STRMID(var_val_str1,0,second_br)
			comma_pos=STRPOS(var_val_str2,',')
			cc=0
			while comma_pos gt 0 do begin
				if cc eq 0 then ImportASCII_ParamList=fix(STRMID(var_val_str2,0,second_br)) else ImportASCII_ParamList = [ImportASCII_ParamList, fix(STRMID(var_val_str2,0,second_br))]
				var_val_str2=STRMID(var_val_str2,(comma_pos+1))
				comma_pos=STRPOS(var_val_str2,',')
				cc++
			endwhile
			ImportASCII_ParamList = [ImportASCII_ParamList, fix(STRMID(var_val_str2,0,second_br))]
		endif

		if var_name eq 'iPALM_MacroParameters_XY' then begin
			first_br=STRPOS(var_value_str,'[')
			var_val_str1=STRMID(var_value_str,(first_br+1))
			second_br=STRPOS(var_val_str1,']')
			var_val_str2=STRMID(var_val_str1,0,second_br)
			comma_pos=STRPOS(var_val_str2,',')
			cc=0
			while comma_pos gt 0 do begin
				if cc eq 0 then iPALM_MacroParameters_XY=fix(STRMID(var_val_str2,0,second_br)) else ImportASCII_ParamList = [ImportASCII_ParamList, fix(STRMID(var_val_str2,0,second_br))]
				var_val_str2=STRMID(var_val_str2,(comma_pos+1))
				comma_pos=STRPOS(var_val_str2,',')
				cc++
			endwhile
			iPALM_MacroParameters_XY = [iPALM_MacroParameters_XY, fix(STRMID(var_val_str2,0,second_br))]
		endif

		if var_name eq 'iPALM_MacroParameters_R ' then begin
			first_br=STRPOS(var_value_str,'[')
			var_val_str1=STRMID(var_value_str,(first_br+1))
			second_br=STRPOS(var_val_str1,']')
			var_val_str2=STRMID(var_val_str1,0,second_br)
			comma_pos=STRPOS(var_val_str2,',')
			cc=0
			while comma_pos gt 0 do begin
				if cc eq 0 then iPALM_MacroParameters_R =fix(STRMID(var_val_str2,0,second_br)) else ImportASCII_ParamList = [ImportASCII_ParamList, fix(STRMID(var_val_str2,0,second_br))]
				var_val_str2=STRMID(var_val_str2,(comma_pos+1))
				comma_pos=STRPOS(var_val_str2,',')
				cc++
			endwhile
			iPALM_MacroParameters_R  = [iPALM_MacroParameters_R , fix(STRMID(var_val_str2,0,second_br))]
		endif

		if var_name eq 'Astig_MacroParameters ' then begin
			first_br=STRPOS(var_value_str,'[')
			var_val_str1=STRMID(var_value_str,(first_br+1))
			second_br=STRPOS(var_val_str1,']')
			var_val_str2=STRMID(var_val_str1,0,second_br)
			comma_pos=STRPOS(var_val_str2,',')
			cc=0
			while comma_pos gt 0 do begin
				if cc eq 0 then iPALM_MacroParameters_R =fix(STRMID(var_val_str2,0,second_br)) else ImportASCII_ParamList = [ImportASCII_ParamList, fix(STRMID(var_val_str2,0,second_br))]
				var_val_str2=STRMID(var_val_str2,(comma_pos+1))
				comma_pos=STRPOS(var_val_str2,',')
				cc++
			endwhile
			iPALM_MacroParameters_R  = [iPALM_MacroParameters_R , fix(STRMID(var_val_str2,0,second_br))]
		endif

	endelse
	i++
endwhile

IF LMGR(/VM) then begin
	AnchorPnts = dblarr(6,AnchPnts_MaxNum)
	ZPnts = dblarr(3,AnchPnts_MaxNum)
endif
;print, 'ICP: Loading Row Labels:    ',ini_filename
i=0
while (~ EOF(1)) and (i lt CGrpSize) do begin
	c=0b
	cc=32b
	while c ne 10b do begin
		readu,1,c
		cc=[[cc],c]
	endwhile
	line_i=strtrim(cc,2)
	if (strmid(line_i,0,7) ne 'RowLbls') and ((byte(line_i))[0] ne 0b) and ((byte(line_i))[0] ne 13b) then begin
		;print,i,'   ',(byte(line_i))[0],'   ',line_i
		name=strmid(strtrim(line_i),0,strpos(strtrim(line_i),string(13b)))
		if i eq 0 then RowLabels = name else RowLabels = [ RowLabels , name ]
	endif
	i++
endwhile
RowNames=RowLabels[0:(CGrpSize-1)]

start_dir = !VERSION.OS_family eq 'unix' ? linux_start_dir : windows_start_dir
if (strlen((file_search(start_dir ,/test_directory))[0]) gt 0) and (n_elements(CGroupParams) lt 1) then begin
	cd, start_dir
	cd, current=st_dir
	print,'started in directory: ',st_dir
endif

CATCH, /CANCEL
close,1
end
;
;-----------------------------------------------------------------
;
pro LoadThiFitCond,ini_filename,thisfitcond
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w

ini_file_info=FILE_INFO(ini_filename)
if ~(ini_file_info.exists) then return
print, 'LTFC: Loading thisfitcond from the file:    ',ini_filename
close,1
openr, 1, ini_filename
thisfitcond={			$
		f_info:'',				$
		zerodark:360.0,			$
		xsz:256,				$
		ysz:256,				$
		Nframesmax:1500ul,		$
		Frm0:0ul,				$
		FrmN:1499ul,			$
		Thresholdcriteria:10.0,	$
		filetype:0,				$
		LimBotA1:12.6,			$
		LimTopA1:10000.,		$
		LimBotSig:0.5,			$
		LimTopSig:3.5,			$
		LimChiSq:1500.,			$
		Cntpere:15.1,			$
		maxcnt1:126,			$
		maxcnt2:0,				$
		fliphor:0,				$
		flipvert:0,				$
		SigmaSym:0,				$
		MaskSize:5,				$
		GaussSig:1.0,			$
		MaxBlck:512,			$
		LocalizationMethod:0,	$
		SparseOversampling:9,	$
		SparseLambda:1e11,		$
		SparseDelta:1e-5,		$
		SpError:0.3,			$
		SpMaxIter:1e3			}

ini_file_info=FILE_INFO(ini_filename)

CATCH, Error_status
line_i=''
i=0
IF Error_status NE 0 THEN BEGIN
      PRINT, 'LTFC: Incorrect INI format. Could not parse the settings from the line:   ',line_i
      PRINT, byte(line_i);
      window,12,xsize=1000,ysize=100,xpos=50,ypos=250,Title='LoadThiFitCond: Incorrect INI format'
	  msg0= string(ini_filename)
      msg1='LTFC: Cant parse from the line:   ' + line_i + '    i=' + strtrim(i)
      xyouts,0.05,0.8,!ERROR_STATE.MSG,CHARSIZE=1.0,/NORMAL
      xyouts,0.05,0.6,!ERROR_STATE.SYS_MSG,CHARSIZE=1.0,/NORMAL
      xyouts,0.05,0.4,msg0,CHARSIZE=1.0,/NORMAL
	  xyouts,0.05,0.2,msg1,CHARSIZE=1.0,/NORMAL
	  wset,def_w		; set back to main (default) window
	  CATCH, /CANCEL
	  close,1
	  return
ENDIF
while (~ EOF(1)) and (strmid(line_i,0,7) ne 'RowLbls') do begin
	c=0b
	cc=32b
	while c ne 10b do begin
		readu,1,c
		cc=[[cc],c]
	endwhile
	line_i=strtrim(cc,2)
	;print,line_i
	if strmid(line_i,0,11) eq 'thisfitcond' then begin
	IF LMGR(/VM) eq 0 then begin 		; if IDL is not in Virtual Machine mode, then command EXECUTE can be performed
		x=execute(line_i)
		;print,line_i
	endif else begin					; otherwise do the following lengthy and not so elegant checking
		var_name = strtrim(strmid(line_i,0,strpos(line_i,'=')),2)
		var_value_str0 = strtrim(strmid(line_i,strpos(line_i,'=')+1),2)
	    space_pos=strpos(var_value_str0,' ')
	    var_value_str = (space_pos gt 0) ? strtrim(strmid(var_value_str0,0,space_pos),2) : var_value_str0
	    semicolon_pos=strpos(var_value_str0,';')
	    var_value_str = (semicolon_pos gt 0) ? strtrim(strmid(var_value_str0,0,semicolon_pos),2) : var_value_str0
	    semicolon_pos=strpos(var_value_str0,'	')
	    var_value_str = (semicolon_pos gt 0) ? strtrim(strmid(var_value_str0,0,semicolon_pos),2) : var_value_str0
	    semicolon_pos=strpos(var_value_str0,string(13B))
	    var_value_str = (semicolon_pos gt 0) ? strtrim(strmid(var_value_str0,0,semicolon_pos),2) : var_value_str0

		if var_name eq 'thisfitcond.f_info' then thisfitcond.f_info = var_value_str
		if var_name eq 'thisfitcond.zerodark' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.zerodark = float(var_value)
		endif
		if var_name eq 'thisfitcond.xsz' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.xsz = fix(var_value)
		endif
		if var_name eq 'thisfitcond.ysz' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.ysz = fix(var_value)
		endif
		if var_name eq 'thisfitcond.Nframesmax' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.Nframesmax = ulong(var_value)
		endif
		if var_name eq 'thisfitcond.Frm0' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.Nframesmax = ulong(var_value)
		endif
		if var_name eq 'thisfitcond.FrmN' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.FrmN = ulong(var_value)
		endif
		if var_name eq 'thisfitcond.Thresholdcriteria' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.Thresholdcriteria = float(var_value)
		endif
		if var_name eq 'thisfitcond.filetype' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.filetype = fix(var_value)
		endif
		if var_name eq 'thisfitcond.Thresholdcriteria' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.Thresholdcriteria = float(var_value)
		endif
		if var_name eq 'thisfitcond.LimBotA1' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.LimBotA1 = float(var_value)
		endif
		if var_name eq 'thisfitcond.LimTopA1' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.LimTopA1 = float(var_value)
		endif
		if var_name eq 'thisfitcond.LimBotSig' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.LimBotSig = float(var_value)
		endif
		if var_name eq 'thisfitcond.LimTopSig' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.LimTopSig = float(var_value)
		endif
		if var_name eq 'thisfitcond.Cntpere' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.Cntpere = float(var_value)
		endif
		if var_name eq 'thisfitcond.maxcnt1' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.maxcnt1 = fix(var_value)
		endif
		if var_name eq 'thisfitcond.maxcnt2' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.maxcnt2 = fix(var_value)
		endif
		if var_name eq 'thisfitcond.fliphor' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.fliphor = fix(var_value)
		endif
		if var_name eq 'thisfitcond.flipvert' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.flipvert = fix(var_value)
		endif
		if var_name eq 'thisfitcond.SigmaSym' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.SigmaSym = fix(var_value)
		endif
		if var_name eq 'thisfitcond.MaskSize' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.MaskSize = fix(var_value)
		endif
		if var_name eq 'thisfitcond.GaussSig' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.GaussSig = float(var_value)
			endif
		if var_name eq 'thisfitcond.MaxBlck' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.MaxBlck = fix(var_value)
		endif
		if var_name eq 'thisfitcond.LocalizationMethod' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.LocalizationMethod = fix(var_value)
		endif
		if var_name eq 'thisfitcond.SparseOversampling' then begin
			reads, var_value_str, var_value, FORMAT='(I)'
			thisfitcond.SparseOversampling = fix(var_value)
		endif
		if var_name eq 'thisfitcond.SparseLambda' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.SparseLambda = float(var_value)
		endif
		if var_name eq 'thisfitcond.SparseDelta' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.SparseDelta = float(var_value)
		endif
		if var_name eq 'thisfitcond.SpError' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.SpError = float(var_value)
		endif
		if var_name eq 'thisfitcond.SpMaxIter' then begin
			reads, var_value_str, var_value, FORMAT='(F)'
			thisfitcond.SpMaxIter = float(var_value)
		endif
	endelse
	endif
	i++
endwhile

CATCH, /CANCEL
close,1
print,'LTFC: loaded default, thisfitcond= ',thisfitcond
end
;
;-----------------------------------------------------------------
;
pro LoadRowNames
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max
; TransformEngine : 0=Local, 1=Cluster, TransformEngine, grouping_gap,grouping_radius100; TransformEngine : 0=Local, 1=Cluster
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w

ini_file_info=FILE_INFO(ini_filename)
if ~(ini_file_info.exists) then return
print, 'LRN: Loading RowNames from the file:    ',ini_filename
close,1
openr, 1, ini_filename

CATCH, Error_status
line_i=''
i=0
IF Error_status NE 0 THEN BEGIN
      PRINT, 'LRN: Incorrect INI format. Could not parse the settings from the line:   ',line_i
      print, 'Loading Default Values'
      window,11,xsize=1000,ysize=100,xpos=50,ypos=250,Title='LoadRowNames: Incorrect INI format'
	  msg0= string(ini_filename)
      msg1='LRN: Cant parse from the line:   ' + line_i + '    i=' + strtrim(i)
      xyouts,0.05,0.8,!ERROR_STATE.MSG,CHARSIZE=1.0,/NORMAL
      xyouts,0.05,0.6,!ERROR_STATE.SYS_MSG,CHARSIZE=1.0,/NORMAL
      xyouts,0.05,0.4,msg0,CHARSIZE=1.0,/NORMAL
	  xyouts,0.05,0.2,msg1,CHARSIZE=1.0,/NORMAL
	  wset,def_w 	; set back to main (default) window
      CATCH, /CANCEL
	  close,1
	  return
ENDIF
while (~ EOF(1)) and (strmid(line_i,0,7) ne 'RowLbls') do begin
	c=0b
	cc=32b
	while c ne 10b do begin
		readu,1,c
		cc=[[cc],c]
	endwhile
	line_i=strtrim(cc,2)
	;print,'s1  ',line_i
	i++
endwhile
i=0
while (~ EOF(1)) and (i lt CGrpSize) do begin
	c=0b
	cc=32b
	while c ne 10b do begin
		readu,1,c
		cc=[[cc],c]
	endwhile
	line_i=strtrim(cc,2)
	;print,'s2  ',line_i
	if (strmid(line_i,0,7) ne 'RowLbls') and ((byte(line_i))[0] ne 0b) and ((byte(line_i))[0] ne 13b) then begin
		;print,i,'   ',(byte(line_i))[0],'   ',line_i
		name=strmid(strtrim(line_i),0,strpos(strtrim(line_i),string(13b)))
		if i eq 0 then RowLabels = name else RowLabels = [ RowLabels , name ]
	endif
	i++
endwhile
CATCH, /CANCEL
close,1
RowNames=RowLabels[0:(CGrpSize-1)]
end
;
;-----------------------------------------------------------------
;
pro ReloadMainTableColumns, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(rownames) lt CGrpSize then begin
	print,'RMTC: Loading RowNames'
	LoadRowNames
endif

Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
UnwZ_ind=min(where(RowNames eq 'Unwrapped Z'))
UnwZ_Err_ind=min(where(RowNames eq 'Unwrapped Z Error'))
UnwGrZ_ind=min(where(RowNames eq 'Unwrapped Group Z'))
UnwGrZ_Err_ind=min(where(RowNames eq 'Unwrapped Group Z Error'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size = min(where(RowNames eq '24 Group Size'))
Frame_Number_ind = min(where(RowNames eq 'Frame Number'))

WidTableID = Widget_Info(wWidget, find_by_uname='WID_TABLE_0')
widget_control,WidTableID,ROW_LABELS=RowNames,TABLE_YSIZE=CGrpSize
widget_control,WidTableID,COLUMN_WIDTH=[160,85,85,70,70,70],use_table_select = [ -1, 0, 4, (CGrpSize-1) ]

	WidDrXID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_X')
	widget_control,WidDrXID, SET_VALUE=RowNames[0:(CGrpSize-1)], Set_Droplist_Select=9
	WidDrYID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Y')
	widget_control,WidDrYID, SET_VALUE=RowNames[0:(CGrpSize-1)], Set_Droplist_Select=3
	WidDrZID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Z')
	widget_control,WidDrZID, SET_VALUE=RowNames[0:(CGrpSize-1)], Set_Droplist_Select=2


items_sensitivity = (Z_ind gt 0) ? 1 : 0
	Wid_Plot_XgrYgrZgr_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Plot_XgrYgrZgr')
	widget_control,Wid_Plot_XgrYgrZgr_ID, Sensitive=items_sensitivity
	Wid_Plot_XgrZgrYgr_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Plot_XgrZgrYgr')
	widget_control,Wid_Plot_XgrZgrYgr_ID, Sensitive=items_sensitivity
	Wid_Plot_YgrZgrXgr_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Plot_YgrZgrXgr')
	widget_control,Wid_Plot_YgrZgrXgr_ID, Sensitive=items_sensitivity
	Wid_Plot_FrameZX_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Plot_FrameZX')
	widget_control,Wid_Plot_FrameZX_ID, Sensitive=items_sensitivity
	Wid_Plot_XZLbl_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Plot_XZLbl')
	widget_control,Wid_Plot_XZLbl_ID, Sensitive=items_sensitivity
	W_MENU_54_ID = Widget_Info(wWidget, find_by_uname='W_MENU_54')
	widget_control,W_MENU_54_ID, Sensitive=items_sensitivity

items_sensitivity = (UnwZ_ind gt 0) ? 1 : 0
	Wid_Plot_FrameUnwrZX_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Plot_FrameUnwrZX')
	widget_control,Wid_Plot_FrameUnwrZX_ID, Sensitive=items_sensitivity
	Wid_Plot_XgrUnwZgrYgr_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Plot_XgrUnwZgrYgr')
	widget_control,Wid_Plot_XgrUnwZgrYgr_ID, Sensitive=items_sensitivity
	Wid_Plot_YgrUnwZgrXgr_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Plot_YgrUnwZgrXgr')
	widget_control,Wid_Plot_YgrUnwZgrXgr_ID, Sensitive=items_sensitivity
	Wid_Plot_XgrUnwZgrLbl_ID = Widget_Info(wWidget, find_by_uname='WID_BUTTON_Plot_XgrUnwZgrLbl')
	widget_control,Wid_Plot_XgrUnwZgrLbl_ID, Sensitive=items_sensitivity

	W_MENU_40_ID = Widget_Info(wWidget, find_by_uname='W_MENU_40')
	widget_control,W_MENU_40_ID, Sensitive=items_sensitivity
	;W_MENU_38_ID = Widget_Info(wWidget, find_by_uname='W_MENU_38')
	;widget_control,W_MENU_38_ID, Sensitive=items_sensitivity
	W_MENU_iPALM_Macro_ID = Widget_Info(wWidget, find_by_uname='W_MENU_iPALM_Macro')
	widget_control,W_MENU_iPALM_Macro_ID, Sensitive=items_sensitivity
	W_MENU_Transform_SpAnalysis_ID = Widget_Info(wWidget, find_by_uname='W_MENU_Transform_SpAnalysis')
	if n_elements(RowNames) ge 19 then sp_analysis_menu_sens = (RowNames[19] eq 'W0')
	widget_control,W_MENU_Transform_SpAnalysis_ID, Sensitive=sp_analysis_menu_sens

	XZ_swap_menue_ID = Widget_Info(wWidget, find_by_uname='W_MENU_SwapXZ')
	widget_control,XZ_swap_menue_ID,set_button=0, Sensitive=items_sensitivity
	YZ_swap_menue_ID = Widget_Info(wWidget, find_by_uname='W_MENU_SwapYZ')
	widget_control,YZ_swap_menue_ID,set_button=0, Sensitive=items_sensitivity
	Z_unwrap_swap_menue_ID = Widget_Info(wWidget, find_by_uname='W_MENU_65')
	widget_control,Z_unwrap_swap_menue_ID,set_button=0,set_value='Swap Z with Unwrapped Z', Sensitive=items_sensitivity

	W_MENU_17_ID = Widget_Info(wWidget, find_by_uname='W_MENU_17')
	widget_control,W_MENU_17_ID, Sensitive=items_sensitivity
	W_MENU_44_ID = Widget_Info(wWidget, find_by_uname='W_MENU_44')
	widget_control,W_MENU_44_ID, Sensitive=items_sensitivity
	W_MENU_AnalyzePhaseUnwrap_ID = Widget_Info(wWidget, find_by_uname='W_MENU_AnalyzePhaseUnwrap')
	widget_control,W_MENU_AnalyzePhaseUnwrap_ID, Sensitive=items_sensitivity
	W_MENU_66W_MENU_set_Z_scale_multiplier_ID = Widget_Info(wWidget, find_by_uname='W_MENU_66W_MENU_set_Z_scale_multiplier')
	widget_control,W_MENU_66W_MENU_set_Z_scale_multiplier_ID, Sensitive=items_sensitivity
	W_vbar_top_ID = Widget_Info(wWidget, find_by_uname='W_vbar_top')
	widget_control,W_vbar_top_ID, Sensitive=items_sensitivity

end
;
;-----------------------------------------------------------------
;

pro EditPeakSelectorPreferences, Event
	EditPeakSelectorIni,Group_Leader=Event.top
end
;
;-----------------------------------------------------------------
;
; Delete everything below this line for 2D version
;
;
;-----------------------------------------------------------------
;

pro iPALM_Macro, Event
	;Top-level macro for transforming, extracting, reextracting, purging, grouping, Z-extraction of actual iPALM data
	; fiducial transformation file and Z-extraction files must exist.
	Transform_Extract_Reextract_Filter_GetZ, GROUP_LEADER=Event.Top
end
;
;-----------------------------------------------------------------
;
pro ReExtractPeaksMultiLabel, Event			;With Transformed Sum peaks Extracted do constrained fit on matched transformed raw dat peaks
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

ReExtractMultiLabelWid, GROUP_LEADER=Event.Top

if n_elements(CGroupParams) lt 2 then return

CGroupParams[34,*]=atan(sqrt(CGroupParams[27,*]/CGroupParams[28,*]))
if (size(CGroupParams))[0] ne 0 then begin
	ReloadParamlists, Event, [27,28,29,30,31,32,33]
	OnUnZoomButton, Event
endif
end
;
;-----------------------------------------------------------------
;
pro Transform_SpAnalysis, Event			;Transform Spectral Camera Data and perform Spectral Analysis
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
Transform_Save_Process_2D_and_Spectrum, GROUP_LEADER=Event.Top
if (size(CGroupParams))[0] ne 0 then begin
	ReloadParamlists, Event, [18,19,20,21,22,23,24,25,26]
	OnUnZoomButton, Event
endif
end
;
;-----------------------------------------------------------------
;
pro Convert_X_to_wavelength, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
Convert_X_to_Wavelength_WID, GROUP_LEADER=Event.Top
if (size(CGroupParams))[0] ne 0 then begin
	ReloadParamlists, Event, [18,19]
	OnUnZoomButton, Event
endif
end
;
;-----------------------------------------------------------------
;
pro OnPeak_Plot_Spectrum, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
Analyze_Plot_Save_Spectra, GROUP_LEADER=Event.Top
end
;
;-----------------------------------------------------------------
;
pro SaveTransformCoeffs, Event			;save transform coeff for guidestar & fiducial into (-tr.sav) file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
existing_ind=where(((RawFilenames ne '') and (GuideStarDrift[*].present or FiducialCoeff[*].present)),nlabels)
if ((size(existing_ind))[0] eq 0) then return
for jj=0,nlabels-1 do begin
	lbl_ind=existing_ind[jj]
	sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
	pos=max(strsplit(RawFilenames[lbl_ind],sep))
	fpath=strmid(RawFilenames[lbl_ind],0,pos-1)
	if FlipRotate[lbl_ind].present then begin
		ffile=((file_info(AddExtension(RawFilenames[lbl_ind],'.dat'))).exists) ? (strmid(RawFilenames[lbl_ind],pos)+'_FlipRot.sav') : ''
		filename = Dialog_Pickfile(/write,path=fpath,file=ffile,filter=['*.sav'],title='Save Flip/Rotation Transf. for '+ strmid(RawFilenames[lbl_ind],pos)+'.dat into *.sav file')
		if filename ne '' then begin
			FlipRot=FlipRotate[lbl_ind]
			save, FlipRotate, filename=filename
		endif
	endif
	if GuideStarDrift[lbl_ind].present then begin
		ffile=((file_info(AddExtension(RawFilenames[lbl_ind],'.dat'))).exists) ? (strmid(RawFilenames[lbl_ind],pos)+'_GuideStarTr.sav') : ''
		filename = Dialog_Pickfile(/write,path=fpath,file=ffile,filter=['*.sav'],title='Save GuideStar Transf. for '+ strmid(RawFilenames[lbl_ind],pos)+'.dat into *.sav file')
		if filename ne '' then begin
			GStarCoeff=GuideStarDrift[lbl_ind]
			save, GStarCoeff, filename=filename
		endif
	endif
	if FiducialCoeff[lbl_ind].present then begin
		ffile=((file_info(AddExtension(RawFilenames[lbl_ind],'.dat'))).exists) ? (strmid(RawFilenames[lbl_ind],pos)+'_FiducialTr.sav') : ''
		filename = Dialog_Pickfile(/write,path=fpath,file=ffile,filter=['*.sav'],title='Save Fiducial Transf. for '+ strmid(RawFilenames[lbl_ind],pos)+'.dat into *.sav file')
		if filename ne '' then begin
			FidCoeff=FiducialCoeff[lbl_ind]
			save, FidCoeff, filename=filename
		endif
	endif
endfor
return
end
;
;-----------------------------------------------------------------
;
pro TransformRaw_Save_SaveSum_MenuItem, Event		; Transforms and saves the raw data according to already performed GuideStar, Flip/Rotate and Fiducial transformations for the set, also saves the sum of transformed data into a separate file
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
TransformRawSavesaveSumWid,GROUP_LEADER=Event.top
end
;
;-----------------------------------------------------------------
;
pro TransformRaw_Save_UserDefined, Event	;Manual File Select & apply saved transforms to Selected Raw file (-.dat)
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
ApplyTransforms, Event
end
;
;-----------------------------------------------------------------
;
pro ZCoordinateOperations, Event		; Starts Z-Operations Menu Widget
ZoperationsWid, GROUP_LEADER=Event.top
end
;
;-----------------------------------------------------------------
;
pro ZCoordinateOperations_Astig, Event ; Starts Z-Operations Menu Widget for Astigmatic PALM (no interferometry)
Zoperations_Astig_Wid, GROUP_LEADER=Event.top
end
;
;-----------------------------------------------------------------
;
pro Process_Multiple_Palm_Slabs_call, Event ; Starts the macro to process and combine (register to scaffold) multiple PALM slabs
;Process_Multiple_Palm_Slabs, GROUP_LEADER=Event.top
iPALM_Split_ZSlabs_into_Separate_Files, GROUP_LEADER=Event.top
end
;
;-----------------------------------------------------------------
;
pro Polarization_Analysis, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

wind_range = 90.0	; degrees

CGroupParams[34,*] = atan(sqrt(CGroupParams[27,*]/(CGroupParams[28,*]>1e-6)))*57.2958
CGroupParams[40,*] = atan(sqrt(CGroupParams[37,*]/(CGroupParams[38,*]>1e-6)))*57.2958

ReloadParamlists, Event, [34,40]

end
;
;-----------------------------------------------------------------
;
pro Analyze1, Event                                                       ;Empty place holder
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
;cgroupparams[27,*]=(2.*cgroupparams[27,*])>0
if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
cgroupparams[31,*]=cgroupparams[27,*]/total(cgroupparams[27:29,*],1)
cgroupparams[32,*]=cgroupparams[28,*]/total(cgroupparams[27:29,*],1)
cgroupparams[33,*]=cgroupparams[29,*]/total(cgroupparams[27:29,*],1)
ParamLimits[31:33,*]=0.
Zid=widget_info(Event.top, FIND_BY_UNAME='WID_DROPLIST_Z')
Zitem=widget_info(Zid,/DropList_Select)
FilterIt
ThisGroup=where(filter eq 1,cnt)
a=45.*!dtor
b=54.73561*!dtor
R1=[[cos(a),sin(a),0],[-sin(a),cos(a),0],[0,0,1.]]
R2=[[1.,0,0],[0,cos(b),sin(b)],[0,-sin(b),cos(b)]]

v=cgroupparams[31:33,*]
aa=R2#R1#v
plot,aa[0,ThisGroup],aa[1,ThisGroup],xrange=[-0.5,0.5],yrange=[-0.5,0.5],title='polar plot',psym=3
col=bytscl(CGroupParams[Zitem,Thisgroup])
plots,aa[0,ThisGroup],aa[1,ThisGroup],color=col*0.75+64,psym=3,/data

end
;
;-----------------------------------------------------------------
;
pro OnSwapXZ, Event				 ;swap group z and group x and 4x mag on z
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common Zdisplay, Z_scale_multiplier, vbar_top
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

X_ind = min(where(RowNames eq 'X Position'))                            ; CGroupParametersGP[2,*] - Peak X  Position
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))                    ; CGroupParametersGP[16,*] - x - sigma
GrX_ind = min(where(RowNames eq 'Group X Position'))                    ; CGroupParametersGP[19,*] - average x - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))                ; CGroupParametersGP[21,*] - new x - position sigma
Z_ind = min(where(RowNames eq 'Z Position'))                            ; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))                            ; CGroupParametersGP[35,*] - Sigma Z
GrZ_ind = min(where(RowNames eq 'Group Z Position'))                    ; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))                    ; CGroupParametersGP[41,*] - Group Sigma Z
sz=size(CGroupParams)

XZ_swap_menue_ID = Widget_Info(Event.top, find_by_uname='W_MENU_SwapXZ')
XZ_swapped=Widget_Info(XZ_swap_menue_ID,/button_set)
print,'Starting with XZ_swapped=', XZ_swapped

YZ_swap_menue_ID = Widget_Info(Event.top, find_by_uname='W_MENU_SwapYZ')
YZ_swapped=Widget_Info(YZ_swap_menue_ID,/button_set)
if YZ_swapped then OnSwapYZ, Event

nmperpix=nm_per_pixel / Z_scale_multiplier			;z is nm & x is pixels

q=CGroupParams[GrX_ind,*]		;group x
sq=CGroupParams[GrSigX_ind,*]		;sigma group x
plq=ParamLimits[GrX_ind,0:3]
plsq=ParamLimits[GrSigX_ind,0:3]
RowNmGrX = RowNames[GrX_ind]
RowNmGrSigX = RowNames[GrSigX_ind]

if XZ_swapped eq 0  then begin; starting with NOT swapped
	CGroupParams[GrX_ind,*]=CGroupParams[GrZ_ind,*]/nmperpix	;assign group z to group x
	CGroupParams[GrSigX_ind,*]=CGroupParams[GrSigZ_ind,*]/nmperpix	;assign sigma group z to sigma group x
	ParamLimits[GrX_ind,0:3]=ParamLimits[GrZ_ind,0:3]/nmperpix
	ParamLimits[GrSigX_ind,0:3]=ParamLimits[GrSigZ_ind,0:3]/nmperpix
endif else begin
	CGroupParams[GrX_ind,*]=CGroupParams[GrZ_ind,*]*nmperpix	;assign group z to group x
	CGroupParams[GrSigX_ind,*]=CGroupParams[GrSigZ_ind,*]*nmperpix	;assign sigma group z to sigma group x
	ParamLimits[GrX_ind,0:3]=ParamLimits[GrZ_ind,0:3]*nmperpix
	ParamLimits[GrSigX_ind,0:3]=ParamLimits[GrSigZ_ind,0:3]*nmperpix
endelse
RowNames[GrX_ind] = RowNames[GrZ_ind]
RowNames[GrSigX_ind] = RowNames[GrSigZ_ind]

if XZ_swapped eq 0  then begin; starting with NOT swapped
	CGroupParams[GrZ_ind,*]=q*nmperpix	;assign group x to group z
	CGroupParams[GrSigZ_ind,*]=sq*nmperpix	;assign sigma group x to sigma group z
	ParamLimits[GrZ_ind,0:3]=plq*nmperpix
	ParamLimits[GrSigZ_ind,0:3]=plsq*nmperpix
endif else begin
	CGroupParams[GrZ_ind,*]=q/nmperpix	;assign group x to group z
	CGroupParams[GrSigZ_ind,*]=sq/nmperpix	;assign sigma group x to sigma group z
	ParamLimits[GrZ_ind,0:3]=plq/nmperpix
	ParamLimits[GrSigZ_ind,0:3]=plsq/nmperpix
endelse

RowNames[GrZ_ind] = RowNmGrX
RowNames[GrSigZ_ind] = RowNmGrSigX

q=CGroupParams[X_ind,*]			;x
sq=CGroupParams[SigX_ind,*]		;sigma x
plq=ParamLimits[X_ind,0:3]
plsq=ParamLimits[SigX_ind,0:3]
RowNmX = RowNames[X_ind]
RowNmSigX = RowNames[SigX_ind]

if XZ_swapped eq 0  then begin; starting with NOT swapped
	CGroupParams[X_ind,*]=CGroupParams[Z_ind,*]/nmperpix	;assign z to x
	CGroupParams[SigX_ind,*]=CGroupParams[SigZ_ind,*]/nmperpix	;assign sigma z to sigma x
	ParamLimits[X_ind,0:3]=ParamLimits[Z_ind,0:3]/nmperpix
	ParamLimits[SigX_ind,0:3]=ParamLimits[SigZ_ind,0:3]/nmperpix
endif else begin
	CGroupParams[X_ind,*]=CGroupParams[Z_ind,*]*nmperpix	;assign z to x
	CGroupParams[SigX_ind,*]=CGroupParams[SigZ_ind,*]*nmperpix	;assign sigma z to sigma x
	ParamLimits[X_ind,0:3]=ParamLimits[Z_ind,0:3]*nmperpix
	ParamLimits[SigX_ind,0:3]=ParamLimits[SigZ_ind,0:3]*nmperpix
endelse

RowNames[X_ind] = RowNames[Z_ind]
RowNames[SigX_ind] = RowNames[SigZ_ind]

if XZ_swapped eq 0  then begin; starting with NOT swapped
	CGroupParams[Z_ind,*]=q*nmperpix			;assign x to z
	CGroupParams[SigZ_ind,*]=sq*nmperpix			;assign sigma x to sigma z
	ParamLimits[Z_ind,0:3]=plq*nmperpix
	ParamLimits[SigZ_ind,0:3]=plsq*nmperpix
endif else begin
	CGroupParams[Z_ind,*]=q/nmperpix			;assign x to z
	CGroupParams[SigZ_ind,*]=sq/nmperpix			;assign sigma x to sigma z
	ParamLimits[Z_ind,0:3]=plq/nmperpix
	ParamLimits[SigZ_ind,0:3]=plsq/nmperpix
endelse

RowNames[Z_ind] = RowNmX
RowNames[SigZ_ind] = RowNmSigX

wtable = Widget_Info(event.top, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(ParamLimits[0:(CGrpSize-1),0:3]), use_table_select=[0,0,3,(CGrpSize-1)]
widget_control, wtable, /editable,/sensitive
widget_control,wtable,ROW_LABELS=RowNames,TABLE_YSIZE=CGrpSize

WidDrXID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_X')
widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select=9
WidDrYID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_Y')
widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select=3
WidDrZID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_Z')
widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select=2

XZ_swap_menue_ID = Widget_Info(Event.top, find_by_uname='W_MENU_SwapXZ')
XZ_swapped=Widget_Info(XZ_swap_menue_ID,/button_set)
widget_control,XZ_swap_menue_ID,set_button=(1-XZ_swapped)

	if bridge_exists then begin
		print,'Reloading the Bridge Array'
		CATCH, Error_status
		CGroupParams_bridge = SHMVAR(shmName_data)
		CGroupParams_bridge[X_ind,*] = CGroupParams[X_ind,*]
		CGroupParams_bridge[SigX_ind,*] = CGroupParams[SigX_ind,*]
		CGroupParams_bridge[GrX_ind,*] = CGroupParams[GrX_ind,*]
		CGroupParams_bridge[GrSigX_ind,*] = CGroupParams[GrSigX_ind,*]
		CGroupParams_bridge[Z_ind,*] = CGroupParams[Z_ind,*]
		CGroupParams_bridge[SigZ_ind,*] = CGroupParams[SigZ_ind,*]
		CGroupParams_bridge[GrZ_ind,*] = CGroupParams[GrZ_ind,*]
		CGroupParams_bridge[GrSigZ_ind,*] = CGroupParams[GrSigZ_ind,*]
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

end
;-----------------------------------------------------------------
; Activate Button Callback Procedure.
;-----------------------------------------------------------------
pro OnSwapYZ, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common Zdisplay, Z_scale_multiplier, vbar_top
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

X_ind = min(where(RowNames eq 'X Position'))                            ; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))                            ; CGroupParametersGP[3,*] - Peak Y  Position
SigX_ind = min(where(RowNames eq 'Sigma X Pos Full'))                    ; CGroupParametersGP[16,*] - x - sigma
SigY_ind = min(where(RowNames eq 'Sigma Y Pos Full'))                    ; CGroupParametersGP[17,*] - y - sigma
GrX_ind = min(where(RowNames eq 'Group X Position'))                    ; CGroupParametersGP[19,*] - average x - position in the group
GrY_ind = min(where(RowNames eq 'Group Y Position'))                    ; CGroupParametersGP[20,*] - average y - position in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))                ; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))                ; CGroupParametersGP[22,*] - new y - position sigma
Z_ind = min(where(RowNames eq 'Z Position'))                            ; CGroupParametersGP[34,*] - Peak Z Position
SigZ_ind = min(where(RowNames eq 'Sigma Z'))                            ; CGroupParametersGP[35,*] - Sigma Z
GrZ_ind = min(where(RowNames eq 'Group Z Position'))                    ; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))                    ; CGroupParametersGP[41,*] - Group Sigma Z
sz=size(CGroupParams)

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

YZ_swap_menue_ID = Widget_Info(Event.top, find_by_uname='W_MENU_SwapYZ')
YZ_swapped=Widget_Info(YZ_swap_menue_ID,/button_set)
print,'Starting with YZ_swapped=', YZ_swapped

XZ_swap_menue_ID = Widget_Info(Event.top, find_by_uname='W_MENU_SwapXZ')
XZ_swapped=Widget_Info(XZ_swap_menue_ID,/button_set)
if XZ_swapped then OnSwapXZ, Event

print,'Swapping Y and Z axes, Z_scale_multiplier=', Z_scale_multiplier
nmperpix=nm_per_pixel / Z_scale_multiplier			;z is nm & y is pixels

q=CGroupParams[GrY_ind,*]		;group y
sq=CGroupParams[GrSigY_ind,*]		;sigma group y
plq=ParamLimits[GrY_ind,0:3]
plsq=ParamLimits[GrSigY_ind,0:3]
RowNmGrY = RowNames[GrY_ind]
RowNmGrSigY = RowNames[GrSigY_ind]

if YZ_swapped eq 0  then begin; starting with NOT swapped
	CGroupParams[GrY_ind,*]=CGroupParams[GrZ_ind,*]/nmperpix	;assign group z to group y
	CGroupParams[GrSigY_ind,*]=CGroupParams[GrSigZ_ind,*]/nmperpix	;assign sigma group z to sigma group y
	ParamLimits[GrY_ind,0:3]=ParamLimits[GrZ_ind,0:3]/nmperpix
	ParamLimits[GrSigY_ind,0:3]=ParamLimits[GrSigZ_ind,0:3]/nmperpix
endif else begin
	CGroupParams[GrY_ind,*]=CGroupParams[GrZ_ind,*]*nmperpix	;assign group z to group y
	CGroupParams[GrSigY_ind,*]=CGroupParams[GrSigZ_ind,*]*nmperpix	;assign sigma group z to sigma group y
	ParamLimits[GrY_ind,0:3]=ParamLimits[GrZ_ind,0:3]*nmperpix
	ParamLimits[GrSigY_ind,0:3]=ParamLimits[GrSigZ_ind,0:3]*nmperpix
endelse
RowNames[GrY_ind] = RowNames[GrZ_ind]
RowNames[GrSigY_ind] = RowNames[GrSigZ_ind]

if YZ_swapped eq 0  then begin; starting with NOT swapped
	CGroupParams[GrZ_ind,*]=q*nmperpix	;assign group y to group z
	CGroupParams[GrSigZ_ind,*]=sq*nmperpix	;assign sigma group y to sigma group z
	ParamLimits[GrZ_ind,0:3]=plq*nmperpix
	ParamLimits[GrSigZ_ind,0:3]=plsq*nmperpix
endif else begin
	CGroupParams[GrZ_ind,*]=q/nmperpix	;assign group y to group z
	CGroupParams[GrSigZ_ind,*]=sq/nmperpix	;assign sigma group y to sigma group z
	ParamLimits[GrZ_ind,0:3]=plq/nmperpix
	ParamLimits[GrSigZ_ind,0:3]=plsq/nmperpix
endelse
RowNames[GrZ_ind] = RowNmGrY
RowNames[GrSigZ_ind] = RowNmGrSigY

q=CGroupParams[Y_ind,*]			;y
sq=CGroupParams[SigY_ind,*]		;sigma y
plq=ParamLimits[Y_ind,0:3]
plsq=ParamLimits[SigY_ind,0:3]
RowNmY = RowNames[Y_ind]
RowNmSigY = RowNames[SigY_ind]

if YZ_swapped eq 0  then begin; starting with NOT swapped
	CGroupParams[Y_ind,*]=CGroupParams[Z_ind,*]/nmperpix	;assign z to y
	CGroupParams[SigY_ind,*]=CGroupParams[SigZ_ind,*]/nmperpix	;assign sigma z to sigma y
	ParamLimits[Y_ind,0:3]=ParamLimits[Z_ind,0:3]/nmperpix
	ParamLimits[SigY_ind,0:3]=ParamLimits[SigZ_ind,0:3]/nmperpix
endif else begin
	CGroupParams[Y_ind,*]=CGroupParams[Z_ind,*]*nmperpix	;assign z to y
	CGroupParams[SigY_ind,*]=CGroupParams[SigZ_ind,*]*nmperpix	;assign sigma z to sigma y
	ParamLimits[Y_ind,0:3]=ParamLimits[Z_ind,0:3]*nmperpix
	ParamLimits[SigY_ind,0:3]=ParamLimits[SigZ_ind,0:3]*nmperpix
endelse
RowNames[Y_ind] = RowNames[Z_ind]
RowNames[SigY_ind] = RowNames[SigZ_ind]

if YZ_swapped eq 0  then begin; starting with NOT swapped
	CGroupParams[Z_ind,*]=q*nmperpix			;assign y to z
	CGroupParams[SigZ_ind,*]=sq*nmperpix			;assign sigma y to sigma z
	ParamLimits[Z_ind,0:3]=plq*nmperpix
	ParamLimits[SigZ_ind,0:3]=plsq*nmperpix
endif else begin
	CGroupParams[Z_ind,*]=q/nmperpix			;assign y to z
	CGroupParams[SigZ_ind,*]=sq/nmperpix			;assign sigma y to sigma z
	ParamLimits[Z_ind,0:3]=plq/nmperpix
	ParamLimits[SigZ_ind,0:3]=plsq/nmperpix
endelse
RowNames[Z_ind] = RowNmY
RowNames[SigZ_ind] = RowNmSigY

wtable = Widget_Info(Event.top, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(ParamLimits[0:(CGrpSize-1),0:3]), use_table_select=[0,0,3,(CGrpSize-1)]
widget_control, wtable, /editable,/sensitive
widget_control,wtable,ROW_LABELS=RowNames,TABLE_YSIZE=CGrpSize

WidDrXID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_X')
widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select=9
WidDrYID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_Y')
widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select=3
WidDrZID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_Z')
widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select=2


widget_control,YZ_swap_menue_ID,set_button=(1-YZ_swapped)

	if bridge_exists then begin
		print,'Reloading the Bridge Array'
		CATCH, Error_status
		CGroupParams_bridge = SHMVAR(shmName_data)
		CGroupParams_bridge[Y_ind,*] = CGroupParams[Y_ind,*]
		CGroupParams_bridge[SigY_ind,*] = CGroupParams[SigY_ind,*]
		CGroupParams_bridge[GrY_ind,*] = CGroupParams[GrY_ind,*]
		CGroupParams_bridge[GrSigY_ind,*] = CGroupParams[GrSigY_ind,*]
		CGroupParams_bridge[Z_ind,*] = CGroupParams[Z_ind,*]
		CGroupParams_bridge[SigZ_ind,*] = CGroupParams[SigZ_ind,*]
		CGroupParams_bridge[GrZ_ind,*] = CGroupParams[GrZ_ind,*]
		CGroupParams_bridge[GrSigZ_ind,*] = CGroupParams[GrSigZ_ind,*]
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

end
;
;-----------------------------------------------------------------
;
pro Swap_Z_Unwrapped_Z, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2

Z_ind=min(where(RowNames eq 'Z Position'))
SigZ_ind=min(where(RowNames eq 'Sigma Z'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
GrSigZ_ind=min(where(RowNames eq 'Group Sigma Z'))
UnwZ_ind=min(where(RowNames eq 'Unwrapped Z'))
UnwZ_Err_ind=min(where(RowNames eq 'Unwrapped Z Error'))
UnwGrZ_ind=min(where(RowNames eq 'Unwrapped Group Z'))
UnwGrZ_Err_ind=min(where(RowNames eq 'Unwrapped Group Z Error'))
Ell_ind=min(where(RowNames eq 'XY Ellipticity'))
Gr_Ell_ind=min(where(RowNames eq 'XY Group Ellipticity'))
Gr_Size_ind = min(where(RowNames eq '24 Group Size'))
Frame_Number_ind = min(where(RowNames eq 'Frame Number'))
sz=size(CGroupParams)

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

XZ_swap_menue_ID = Widget_Info(Event.top, find_by_uname='W_MENU_SwapXZ')
XZ_swapped=Widget_Info(XZ_swap_menue_ID,/button_set)
if XZ_swapped then OnSwapXZ, Event

YZ_swap_menue_ID = Widget_Info(Event.top, find_by_uname='W_MENU_SwapYZ')
YZ_swapped=Widget_Info(YZ_swap_menue_ID,/button_set)
if YZ_swapped then OnSwapYZ, Event

Z = CGroupParams[Z_ind,*]
GrZ = CGroupParams[GrZ_ind,*]
ZParamL = ParamLimits[Z_ind,0:3]
GrZParamL = ParamLimits[GrZ_ind,0:3]
RowNmZ = RowNames[Z_ind]
RowNmGrZ = RowNames[GrZ_ind]

CGroupParams[Z_ind,*] = CGroupParams[UnwZ_ind,*]
CGroupParams[GrZ_ind,*] = CGroupParams[UnwGrZ_ind,*]
ParamLimits[Z_ind,0:3] = ParamLimits[UnwZ_ind,0:3]
ParamLimits[GrZ_ind,0:3] = ParamLimits[UnwGrZ_ind,0:3]
RowNames[Z_ind] = RowNames[UnwZ_ind]
RowNames[GrZ_ind] = RowNames[UnwGrZ_ind]

CGroupParams[UnwZ_ind,*] = Z
CGroupParams[UnwGrZ_ind,*] = GrZ
ParamLimits[UnwZ_ind,0:3] = ZParamL
ParamLimits[UnwGrZ_ind,0:3] = GrZParamL
RowNames[UnwZ_ind] = RowNmZ
RowNames[UnwGrZ_ind] = RowNmGrZ

wtable = Widget_Info(Event.top, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(ParamLimits[0:(CGrpSize-1),0:3]), use_table_select=[0,0,3,(CGrpSize-1)]
widget_control, wtable, /editable,/sensitive
widget_control,wtable,ROW_LABELS=RowNames,TABLE_YSIZE=CGrpSize

WidDrXID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_X')
widget_control,WidDrXID, SET_VALUE=RowNames, Set_Droplist_Select=9
WidDrYID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_Y')
widget_control,WidDrYID, SET_VALUE=RowNames, Set_Droplist_Select=3
WidDrZID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_Z')
widget_control,WidDrZID, SET_VALUE=RowNames, Set_Droplist_Select=2

button=widget_info(event.id,/button_set)
if button then widget_control,event.id,set_button=0 else widget_control,event.id,set_button=1

	if bridge_exists then begin
		print,'Reloading the Bridge Array'
		CATCH, Error_status
		CGroupParams_bridge = SHMVAR(shmName_data)
		CGroupParams_bridge[Z_ind,*] = CGroupParams[Z_ind,*]
		CGroupParams_bridge[GrZ_ind,*] = CGroupParams[GrZ_ind,*]
		CGroupParams_bridge[UnwZ_ind,*] = CGroupParams[UnwZ_ind,*]
		CGroupParams_bridge[UnwGrZ_ind,*] = CGroupParams[UnwGrZ_ind,*]
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

end
;
;----------------------------------------------------------------
;
pro OnAnalyzePeakDistribution, Event
OnAnalyze2, Dist_Results, 0, Event
end
;
;----------------------------------------------------------------
;
pro OnAnalyzeGroupDistribution, Dist_Results, Event
OnAnalyze2, Dist_Results, 1, Event
end
;
;-----------------------------------------------------------------
;
pro OnAnalyze2, Dist_Results, FilterItem, Event	 ;Analyze the distribution of the selected PEAKS  or GROUPS
; FilterItem=0 - Analyze peak distribution, FilterItem=1 - Analyze group Distribution
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
WidSldFractionHistAnalID = Widget_Info(TopID, find_by_uname='WID_SLIDER_FractionHistAnal')
widget_control,WidSldFractionHistAnalID,get_value=fr_search
fr_srch_str = strtrim(fr_search,2)


if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
if FilterItem then GroupFilterIt else FilterIt
WidLabel0 = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,WidLabel0,get_value=dfilename
sfilename = FilterItem ? AddExtension(dfilename,'_Cluster_Group_Width_Distributions.txt') : AddExtension(dfilename,'_Cluster_Peaks_Width_Distributions.txt')
finfo=file_info(sfilename)
close,1

X_ind = FilterItem ? min(where(RowNames eq 'Group X Position')) : min(where(RowNames eq 'X Position'))
Y_ind = FilterItem ? min(where(RowNames eq 'Group Y Position')) : min(where(RowNames eq 'Y Position'))
Z_ind = FilterItem ? min(where(RowNames eq 'Group Z Position')) : min(where(RowNames eq 'Z Position'))
UnwZ_ind = FilterItem ? min(where(RowNames eq 'Unwrapped Group Z')) : min(where(RowNames eq 'Unwrapped Z'))
Nph_ind = FilterItem ? min(where(RowNames eq 'Group N Photons')) : min(where(RowNames eq '6 N Photons'))
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))                ; CGroupParametersGP[25,*] - Frame Index in the Group

param = [X_ind,Y_ind]
if Z_ind ge 0 then  param = [X_ind,Y_ind,Z_ind]
if UnwZ_ind ge 0 then  param = [X_ind,Y_ind,Z_ind,UnwZ_ind]

if finfo.exists then openu,1,sfilename,/Append else begin
	openw,1,sfilename
	if 	UnwZ_ind ge 0 then begin
		if FilterItem then printf,1,'Gr Nph (mean)	Gr Nph (STD)	# of Groups	X (pixels)	Y (pixels)	Z (nm)	UnwZ (nm)	X-FWHM (nm)	Y-FWHM (nm)	Z-FWHM (nm)	UnwZ-FWHM (nm)'+$
		'	X (STD) (nm)	Y (STD) (nm)	Z (STD) (nm)	UnwZ (STD) (nm)	X'+	fr_srch_str+' (nm)	Y'+fr_srch_str+' (nm)	Z'+fr_srch_str+' (nm)	UnwZ'+fr_srch_str+' (nm)'
		if ~FilterItem then printf,1,'Nph (mean)	Nph (STD)	# of Peaks	X (pixels)	Y (pixels)	Z (nm)	UnwZ (nm)	X-FWHM (nm)	Y-FWHM (nm)	Z-FWHM (nm)	UnwZ-FWHM (nm)'+$
		'	X (STD) (nm)	Y (STD) (nm)	Z (STD) (nm)	UnwZ (STD) (nm)	X'+	fr_srch_str+' (nm)	Y'+fr_srch_str+' (nm)	Z'+fr_srch_str+' (nm)	UnwZ'+fr_srch_str+' (nm)'
	endif else begin
		if Z_ind ge 0 then begin
			if FilterItem then printf,1,'Gr Nph (mean)	Gr Nph (STD)	# of Groups	X (pixels)	Y (pixels)	Z (nm)	X-FWHM (nm)	Y-FWHM (nm)	Z-FWHM (nm)'+$
			'	X (STD) (nm)	Y (STD) (nm)	Z (STD) (nm)	X'+	fr_srch_str+' (nm)	Y'+fr_srch_str+' (nm)	Z'+fr_srch_str+' (nm)'
			if ~FilterItem then printf,1,'Nph (mean)	Nph (STD)	# of Peaks	X (pixels)	Y (pixels)	Z (nm)	X-FWHM (nm)	Y-FWHM (nm)	Z-FWHM (nm)'+$
			'	X (STD) (nm)	Y (STD) (nm)	Z (STD) (nm)	X'+	fr_srch_str+' (nm)	Y'+fr_srch_str+' (nm)	Z'+fr_srch_str+' (nm)'
		endif else begin
			if FilterItem then printf,1,'Gr Nph (mean)	Gr Nph (STD)	# of Groups	X (pixels)	Y (pixels)	X-FWHM (nm)	Y-FWHM (nm)'+$
			'	X (STD) (nm)	Y (STD) (nm)	X'+	fr_srch_str+' (nm)	Y'+fr_srch_str+' (nm)'
			if ~FilterItem then printf,1,'Nph (mean)	Nph (STD)	# of Peaks	X (pixels)	Y (pixels)	X-FWHM (nm)	Y-FWHM (nm)'+$
			'	X (STD) (nm)	Y (STD) (nm)	X'+	fr_srch_str+' (nm)	Y'+fr_srch_str+' (nm)'
		endelse
	endelse
endelse

tk=1.5
nbns=500
xx=fltarr(2*nbns)
histhist=fltarr(2*nbns)
evens=2*indgen(nbns)
odds=evens+1
hist_set = FilterItem ? where(filter*(CGroupParams[GrInd_ind,*] eq 1),cnt) : where(filter,cnt)

if cnt lt 1 then return

;Analize  X-,  Y-, and Z- Peak / Group localizations


FWHM=fltarr(n_elements(param))
CL90=fltarr(n_elements(param))
CR90=fltarr(n_elements(param))

for ip=0,(n_elements(param)-1) do begin
	hist=histogram(CGroupParams[param[ip],hist_set],min=ParamLimits[param[ip],0],max=ParamLimits[param[ip],1],nbins=nbns)
	mn=ParamLimits[param[ip],0]
	mx=ParamLimits[param[ip],1]
	binsize=(mx-mn)/(nbns-1.0)
	x=findgen(nbns)/nbns*ParamLimits[param[ip],3]+ParamLimits[param[ip],0]
	dx=ParamLimits[param[ip],3]/nbns/2
	xx[evens]=x-dx
	xx[odds]=x+dx
	histhist[evens]=hist
	histhist[odds]=hist
	if ip le 1 then begin
		xcoord=nm_per_pixel*(xx-xx[0])
		binsize=binsize*nm_per_pixel
		mn=nm_per_pixel*(mn-xx[0])
		mx=nm_per_pixel*(mx-xx[0])
	endif else xcoord=xx
	histmax=max(histhist,xmax)
	histhalf=mean(histhist[((xmax-2)>0):((xmax+2)<(n_elements(histhist)-1))])*0.5
	nx=n_elements(xcoord)
	xl=xcoord[0]
	xr=xcoord[nx-1]
	rightsearch=1
	leftsearch=1
	;Search for FWHM
	for xi=2,(nx-2) do begin
		if  rightsearch and histhist[nx-xi] lt histhalf and histhist[nx-1-xi] ge histhalf then begin
			rightsearch=0
			xr=xcoord[nx-xi]+(xcoord[nx-1-xi]-xcoord[nx-xi])*$
			(histhalf-histhist[nx-xi])/(histhist[nx-1-xi]-histhist[nx-xi])
		endif
		if  leftsearch and histhist[xi-1] lt histhalf and histhist[xi] ge histhalf then begin
			leftsearch=0
			xl=xcoord[xi-1]+(xcoord[xi]-xcoord[xi-1])*(histhalf-histhist[xi-1])/(histhist[xi]-histhist[xi-1])
		endif
	endfor
	FWHM[ip]=xr-xl
	; Search for width at fraction >= fr_search
	gr_fr=cnt/100.0*fr_search
	xl90=xl
	xr90=xr
	search90=1
	iter=0
	while search90 do begin
		iter+=1
		xl90 = (xl90 - binsize/10.0) > mn
		xr90 = (xr90 + binsize/10.0) < mx
		if ip le 1 then trash = where(((CGroupParams[param[ip],hist_set]-xx[0])*nm_per_pixel ge xl90) and ((CGroupParams[param[ip],hist_set]-xx[0])*nm_per_pixel le xr90),search_cnt) $
		else trash = where((CGroupParams[param[ip],hist_set] ge xl90) and (CGroupParams[param[ip],hist_set] le xr90),search_cnt)
		if  (search_cnt ge gr_fr) or ((xl90 le mn) and (xr90 ge mx)) or (iter ge 300) then search90=0
	endwhile
	CL90[ip]=xl90
	CR90[ip]=xr90

endfor
XPos=mean(CGroupParams[X_ind,hist_set])
YPos=mean(CGroupParams[Y_ind,hist_set])
Xstd=stddev(CGroupParams[X_ind,hist_set])*nm_per_pixel
Ystd=stddev(CGroupParams[Y_ind,hist_set])*nm_per_pixel
Nph_mean=mean(CGroupParams[Nph_ind,hist_set])
Nph_std=stddev(CGroupParams[Nph_ind,hist_set])
X90=CR90[0]-CL90[0]
Y90=CR90[1]-CL90[1]

if Z_ind eq -1 then begin

	if FilterItem then begin
		str1='Gr. Nph mean= '+strtrim(Nph_mean,2)+'   Gr. Nph STD = '+strtrim(Nph_std,2)+'     # of Groups = '+strtrim(cnt,2)
		str2='X = '+strtrim(XPos,2)+' pix        Y = '+strtrim(YPos,2)+' pix'
		str3='X-FWHM = '+strtrim(FWHM[0],2)+' nm    Y-FWHM = '+strtrim(FWHM[1],2)+' nm'
		str4='X @'+fr_srch_str+'% = '+strtrim(X90,2)+' nm    Y @'+fr_srch_str+'% = '+strtrim(Y90,2)+' nm'
		str5='X-std = '+strtrim(Xstd,2)+' nm    Y-std = '+strtrim(Ystd,2)+' nm'
	endif else begin
		str1='Nph mean= '+strtrim(Nph_mean,2)+'   Nph STD = '+strtrim(Nph_std,2)+'     # of Peaks = '+strtrim(cnt,2)
		str2='X = '+strtrim(XPos,2)+' pix        Y = '+strtrim(YPos,2)+' pix'
		str3='X-FWHM = '+strtrim(FWHM[0],2)+' nm    Y-FWHM = '+strtrim(FWHM[1],2)+' nm'
		str4='X @'+fr_srch_str+'% = '+strtrim(X90,2)+' nm    Y @'+fr_srch_str+'% = '+strtrim(Y90,2)+' nm'
		str5='X-std = '+strtrim(Xstd,2)+' nm    Y-std = '+strtrim(Ystd,2)+' nm'
	endelse

	printf,1,strtrim(Nph_mean,2)+'	'+strtrim(Nph_std,2)+'	'+strtrim(cnt,2)+'	'+strtrim(XPos,2)+'	'+strtrim(YPos,2)+$
	'	'+strtrim(FWHM[0],2)+'	'+strtrim(FWHM[1],2)+'	'+strtrim(Xstd,2)+'	'+strtrim(Ystd,2)+$
	'	'+strtrim(X90,2)+'	'+strtrim(Y90,2)
	Dist_Results=[Nph_mean,Nph_std,cnt,XPos,YPos,FWHM,Xstd,Ystd,X90,Y90]

endif else begin

	ZPos=mean(CGroupParams[Z_ind,hist_set])
	Zstd=stddev(CGroupParams[Z_ind,hist_set])
	Z90=CR90[2]-CL90[2]

	if UnwZ_ind eq -1 then begin
		if FilterItem then begin
			str1='Gr. Nph mean= '+strtrim(Nph_mean,2)+'   Gr. Nph STD = '+strtrim(Nph_std,2)+'     # of Groups = '+strtrim(cnt,2)
			str2='X = '+strtrim(XPos,2)+' pix        Y = '+strtrim(YPos,2)+' pix        Z = '+strtrim(ZPos,2)+' nm'
			str3='X-FWHM = '+strtrim(FWHM[0],2)+' nm    Y-FWHM = '+strtrim(FWHM[1],2)+' nm    Z-FWHM = '+strtrim(FWHM[2],2)+' nm'
			str4='X @'+fr_srch_str+'% = '+strtrim(X90,2)+' nm    Y @'+fr_srch_str+'% = '+strtrim(Y90,2)+' nm     Z @'+fr_srch_str+'% = '+strtrim(Z90,2)+' nm'
			str5='X-std = '+strtrim(Xstd,2)+' nm    Y-std = '+strtrim(Ystd,2)+' nm    Z-std = '+strtrim(Zstd,2)+' nm'
		endif else begin
			str1='Nph mean= '+strtrim(Nph_mean,2)+'   Nph STD = '+strtrim(Nph_std,2)+'     # of Peaks = '+strtrim(cnt,2)
			str2='X = '+strtrim(XPos,2)+' pix        Y = '+strtrim(YPos,2)+' pix        Z = '+strtrim(ZPos,2)+' nm'
			str3='X-FWHM = '+strtrim(FWHM[0],2)+' nm    Y-FWHM = '+strtrim(FWHM[1],2)+' nm    Z-FWHM = '+strtrim(FWHM[2],2)+' nm'
			str4='X @'+fr_srch_str+'% = '+strtrim(X90,2)+' nm    Y @'+fr_srch_str+'% = '+strtrim(Y90,2)+' nm     Z @'+fr_srch_str+'% = '+strtrim(Z90,2)+' nm'
			str5='X-std = '+strtrim(Xstd,2)+' nm    Y-std = '+strtrim(Ystd,2)+' nm    Z-std = '+strtrim(Zstd,2)+' nm'
		endelse

		printf,1,strtrim(Nph_mean,2)+'	'+strtrim(Nph_std,2)+ '	'+strtrim(cnt,2)+ $
		'	'+strtrim(XPos,2)+'	'+strtrim(YPos,2)+'	'+strtrim(ZPos,2)+ $
		'	'+strtrim(FWHM[0],2)+'	'+strtrim(FWHM[1],2)+'	'+strtrim(FWHM[2],2)+ $
		'	'+strtrim(Xstd,2)+'	'+strtrim(Ystd,2)+'	'+strtrim(Zstd,2)+ $
		'	'+strtrim(X90,2)+'	'+strtrim(Y90,2)+'	'+strtrim(Z90,2)
		Dist_Results=[Nph_mean,Nph_std,cnt,XPos,YPos,ZPos,FWHM,Xstd,Ystd,Zstd,X90,Y90,Z90]

	endif else begin

		UnwZPos=mean(CGroupParams[UnwZ_ind,hist_set])
		UnwZstd=stddev(CGroupParams[UnwZ_ind,hist_set])
		UnwZ90=CR90[3]-CL90[3]

		if FilterItem then begin
			str1='Gr. Nph mean= '+strtrim(Nph_mean,2)+'   Gr. Nph STD = '+strtrim(Nph_std,2)+'     # of Groups = '+strtrim(cnt,2)
			str2='X = '+strtrim(XPos,2)+' pix        Y = '+strtrim(YPos,2)+' pix        Z = '+strtrim(ZPos,2)+' nm        UnwZ = '+strtrim(UnwZPos,2)+' nm'
			str3='X-FWHM = '+strtrim(FWHM[0],2)+' nm    Y-FWHM = '+strtrim(FWHM[1],2)+' nm    Z-FWHM = '+strtrim(FWHM[2],2)+' nm    UnwZ-FWHM = '+strtrim(FWHM[3],2)+' nm'
			str4='X @'+fr_srch_str+'% = '+strtrim(X90,2)+' nm    Y @'+fr_srch_str+'% = '+strtrim(Y90,2)+' nm     Z @'+fr_srch_str+'% = '+strtrim(Z90,2)+' nm     UnwZ @'+fr_srch_str+'% = '+strtrim(UnwZ90,2)+' nm'
			str5='X-std = '+strtrim(Xstd,2)+' nm    Y-std = '+strtrim(Ystd,2)+' nm    Z-std = '+strtrim(Zstd,2)+' nm    UnwZ-std = '+strtrim(UnwZstd,2)+' nm'
		endif else begin
			str1='Nph mean= '+strtrim(Nph_mean,2)+'   Nph STD = '+strtrim(Nph_std,2)+'     # of Peaks = '+strtrim(cnt,2)
			str2='X = '+strtrim(XPos,2)+' pix        Y = '+strtrim(YPos,2)+' pix        Z = '+strtrim(ZPos,2)+' nm        UnwZ = '+strtrim(UnwZPos,2)+' nm'
			str3='X-FWHM = '+strtrim(FWHM[0],2)+' nm    Y-FWHM = '+strtrim(FWHM[1],2)+' nm    Z-FWHM = '+strtrim(FWHM[2],2)+' nm    UnwZ-FWHM = '+strtrim(FWHM[3],2)+' nm'
			str4='X @'+fr_srch_str+'% = '+strtrim(X90,2)+' nm    Y @'+fr_srch_str+'% = '+strtrim(Y90,2)+' nm     Z @'+fr_srch_str+'% = '+strtrim(Z90,2)+' nm     UnwZ @'+fr_srch_str+'% = '+strtrim(UnwZ90,2)+' nm'
			str5='X-std = '+strtrim(Xstd,2)+' nm    Y-std = '+strtrim(Ystd,2)+' nm    Z-std = '+strtrim(Zstd,2)+' nm    UnwZ-std = '+strtrim(UnwZstd,2)+' nm'
		endelse

		printf,1,strtrim(Nph_mean,2)+'	'+strtrim(Nph_std,2)+'	'+strtrim(cnt,2)+'	'+strtrim(XPos,2)+'	'+strtrim(YPos,2)+'	'+strtrim(ZPos,2)+'	'+strtrim(UnwZPos,2)+$
		'	'+strtrim(FWHM[0],2)+'	'+strtrim(FWHM[1],2)+'	'+strtrim(FWHM[2],2)+'	'+strtrim(FWHM[3],2)+'	'+strtrim(Xstd,2)+'	'+strtrim(Ystd,2)+$
		'	'+strtrim(Zstd,2)+'	'+strtrim(UnwZstd,2)+'	'+strtrim(X90,2)+'	'+strtrim(Y90,2)+'	'+strtrim(Z90,2)+'	'+strtrim(UnwZ90,2)
		Dist_Results=[Nph_mean,Nph_std,cnt,XPos,YPos,ZPos,UnwZPos,FWHM,Xstd,Ystd,Zstd,UnwZstd,X90,Y90,Z90,UnwZ90]

	endelse

endelse

print,str1
print,str2
print,str3
print,str4
print,str5

xyouts,0.02,0.91,str2,/NORMAL,color=200,charsize=1.5
xyouts,0.02,0.89,str1,/NORMAL,color=200,charsize=1.5
xyouts,0.02,0.87,str3,/NORMAL,color=200,charsize=1.5
xyouts,0.02,0.85,str4,/NORMAL,color=200,charsize=1.5
xyouts,0.02,0.83,str5,/NORMAL,color=200,charsize=1.5


close,1

end
;
;-----------------------------------------------------------------
;
pro OnAnalyze3, Event				   ;Perform Analyze2 on multiple peaks
	AnalyzeMultiplePeaks,Group_Leader=Event.top
end
;
;-----------------------------------------------------------------
;
pro On_Analyze_Fiducial_Alignement_2Colors, Event
	Analyze_Fiducial_Alignement_2Colors, Group_Leader=Event.top
end
;
;-----------------------------------------------------------------
;
pro On_AnalyzePhaseUnwrap, Event
	AnalyzePhaseUnwrap, Group_Leader=Event.top
end
;
;-----------------------------------------------------------------
;
pro OnAnalyze4, Event;	*************	Show iPALM PSF Raw and Fits
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir, n_cluster_nodes_max
; TransformEngine : 0=Local, 1=Cluster, TransformEngine, grouping_gap,grouping_radius100; TransformEngine : 0=Local, 1=Cluster
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
WidLabel0 = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,WidLabel0,get_value=dfilename

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number

if n_elements(CGroupParams) le 2 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

erase,0
!p.noerase=1

d=5 & dd=2*d+1
peak_set=where(filter,cnt)
close,1
for k=0,(n_elements(MLRawFilenames)-1) do begin
	ReadThisFitCond, (MLRawFilenames[k]+'.txt'), pth, filen, ini_filename, thisfitcond
	if k eq 0 then ThisFitCondML=thisfitcond else ThisFitCondML=[ThisFitCondML,thisfitcond]
endfor
ReadThisFitCond, (RawFilenames[0]+'.txt'), pth, filen, ini_filename, thisfitcond

Nph_mean=mean(CGroupParams[6,peak_set])
Nph_std = (cnt gt 1) ? stddev(CGroupParams[6,peak_set]) : 0
PeakX=mean(CGroupParams[2,peak_set])
PeakY=mean(CGroupParams[3,peak_set])
str0='X = '+strtrim(PeakX,2)+'   Y = '+strtrim(PeakY,2)+'   Nph mean= '+strtrim(Nph_mean,2)+'     Nph STD = '+strtrim(Nph_std,2)+'     # of Peaks = '+strtrim(cnt,2)
xyouts,0.1,0.98,str0,/NORMAL,color=500,charsize=1.2

xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz
data=intarr(xsz,ysz)
x0=(fix(peakx)-d-1)>0
x1=x0+dd < (xsz-1)
x0=x1-dd+1
y0=(fix(peaky)-d-1)>0
y1=y0+dd < (xsz-1)
y0=y1-dd+1
mag=20.0
xlen=dd
ylen=dd

;sum files
	openr,1,AddExtension(RawFilenames[0],'.dat')
	window,10,xsize=800,ysize=100,xpos=50,ypos=450,Title='Reading Raw Data'
	str_out='Started Reading and Transforming data'
	xyouts,0.1,0.5,str_out,CHARSIZE=2.0,/NORMAL
	region=fltarr(dd,dd)
	for i_peak=0,(cnt-1) do begin
		point_lun,1,2ull*xsz*ysz*CGroupParams[FrNum_ind,peak_set[i_peak]]
		readu,1,dataretall

		region=region+(((float(temporary(data[x0:x1,y0:y1]))-thisfitcond.zerodark)/thisfitcond.cntpere)>0.)
		if (i_peak mod 100) eq 0 then begin
			xyouts,0.1,0.5,str_out,CHARSIZE=2.0,/NORMAL,col=0
			str_out='Sum: Reading Frame  '+strtrim(i_peak,2)+'  of total '+strtrim(cnt,2)
   			xyouts,0.1,0.5,str_out,CHARSIZE=2.0,/NORMAL
   			wait,0.1
   		endif
	endfor
	close,1
	wdelete,10
	wset,def_w	; set back to main (default) window
	clip=region/cnt
	scl=250./max(clip)
	A=fltarr(7)	; [0.0,1.0,1.2,1.2,d,d,0.]
	A[0]=mean(CGroupParams[0,peak_set])			;set A[0] to averaged value of base
	A[1]=mean(CGroupParams[1,peak_set])			;set A[1] to peak amplitude
	A[2]=mean(CGroupParams[4,peak_set])			; X width
	A[3]=mean(CGroupParams[5,peak_set])			; Y width
	A[4]=d
	A[5]=d
	fita = [1,1,1,1,1,1]
	result=gauss2Dfithh(clip,A,sigma=sigma, chisq=chisq, FITA = fita, STATUS = status, ITMAX = 100)	;do 2D fit
	residual=clip-result+A[0]>0.0
	tv,scl*rebin(clip,mag*xlen,mag*ylen,/sample),0,mag*(2*ylen+2)
	tv,scl*rebin(result,mag*xlen,mag*ylen,/sample),0,mag*(ylen+1)
	tv,scl*rebin(residual,mag*xlen,mag*ylen,/sample),0,0
	str1='SigX = '+strtrim(A[2],2)+'     SigY = '+strtrim(A[3],2)
	str2='Amp = '+strtrim(A[1],2)+'     Offset = '+strtrim(A[0],2)
	print,str1
	print,str2
	xyouts,0.005,0.67,'Raw Data: C1+C2+C3',/NORMAL,color=1000,charsize=1.1
	xyouts,0.005,0.44,'2D Gaussian fit',/NORMAL,color=1000,charsize=1.1
	xyouts,0.005,0.20,'Resudual',/NORMAL,color=1000,charsize=1.1
	xyouts,0.005,0.01,str2,/NORMAL,color=1000,charsize=0.9
	xyouts,0.005,0.03,str1,/NORMAL,color=1000,charsize=0.9

	CCD=clip*scl
	CCD_fit=result*scl
	xlen=(size(CCD))[1]
	ylen=(size(CCD))[2]
	xaxis=findgen(xlen)-xlen/2.0
	yaxis=findgen(ylen)-ylen/2.0
	tr=max(CCD,ii_max_CCD)
	i_max_CCD=ii_max_CCD/xlen
	j_max_CCD=ii_max_CCD-(ii_max_CCD/xlen)*xlen
	CCD_X_mag=CCD[*,i_max_CCD]
	;CCD_X_g=GAUSSFIT(xaxis,CCD_X_mag,CCD_X_coeff,CHISQ=CCD_X_CHI,NTERMS=4)
	CCD_X_g=CCD_fit[*,i_max_CCD]
	CCD_Y_mag=CCD[j_max_CCD,*]
	;CCD_Y_g=GAUSSFIT(yaxis,CCD_Y_mag,CCD_Y_coeff,CHISQ=CCD_Y_CHI,NTERMS=4)
	CCD_Y_g=CCD_fit[j_max_CCD,*]
	plot,xaxis,CCD[*,i_max_CCD],xtitle='X,Y coordinate (pix)',psym=6,ticklen=1,YCHARSIZE=1.2,/device,POSITION=[2*mag,(mag*3*ylen+5*mag),mag*xlen,(mag*4*ylen+5*mag)]
	oplot,xaxis,CCD[*,i_max_CCD],psym=6,color=3000
	oplot,xaxis,CCD[*,i_max_CCD],linestyle=1
	oplot,xaxis,CCD_X_g,THICK=2,color=3000
	oplot,yaxis,CCD[j_max_CCD,*],psym=5,color=6000
	oplot,yaxis,CCD[j_max_CCD,*],linestyle=1
	oplot,yaxis,CCD_Y_g,THICK=2,color=6000

order=4
image_moments, clip, order, moments
print,'CCD total'
print,moments
print,''

;C1,2,3 files
for iccd=0,2 do begin
	openr,1,AddExtension(MLRawFilenames[iccd],'.dat')
	window,10,xsize=800,ysize=100,xpos=50,ypos=450,Title='Reading Raw Data'
	str_out='Started Reading and Transforming data'
	xyouts,0.1,0.5,str_out,CHARSIZE=2.0,/NORMAL
	region=fltarr(dd,dd)
	for i_peak=0,(cnt-1) do begin
		point_lun,1,2ull*xsz*ysz*CGroupParams[FrNum_ind,peak_set[i_peak]]
		readu,1,data
		region=region+(((float(temporary(data[x0:x1,y0:y1]))-ThisFitCondML[iccd].zerodark)/ThisFitCondML[iccd].cntpere)>0.)
		if (i_peak mod 100) eq 0 then begin
			xyouts,0.1,0.5,str_out,CHARSIZE=2.0,/NORMAL,col=0
			str_out='CCD'+strtrim((iccd+1),2)+': Reading Frame '+strtrim(i_peak,2)+'   of total '+strtrim(cnt,2)
   			xyouts,0.1,0.5,str_out,CHARSIZE=2.0,/NORMAL
   			wait,0.1
   		endif
	endfor
	close,1
	wdelete,10
	wset,def_w		; set back to main (default) window
	clip=region/cnt
	A=fltarr(7)	; [0.0,1.0,1.2,1.2,d,d,0.]
	A[0]=mean(CGroupParams[0,peak_set])			;set A[0] to averaged value of base
	A[1]=mean(CGroupParams[1,peak_set])			;set A[1] to peak amplitude
	A[2]=mean(CGroupParams[4,peak_set])			; X width
	A[3]=mean(CGroupParams[5,peak_set])			; Y width
	A[4]=d
	A[5]=d
	fita = [1,1,1,1,0,0]
	result = gauss2Dfithh(clip,A,sigma=sigma, chisq=chisq, FITA = fita, STATUS = status, ITMAX = 100)	;do 2D fit
	residual=clip-result+A[0]>0.0
	tv,scl*rebin(clip,mag*xlen,mag*ylen,/sample),mag*((iccd+1)*(xlen+2)),mag*(2*ylen+2)
	tv,scl*rebin(result,mag*xlen,mag*ylen,/sample),mag*((iccd+1)*(xlen+2)),mag*(ylen+1)
	tv,scl*rebin(residual,mag*xlen,mag*ylen,/sample),mag*((iccd+1)*(xlen+2)),0
	str1='SigX = '+strtrim(A[2],2)+'     SigY = '+strtrim(A[3],2)
	str2='Amp = '+strtrim(A[1],2)+'     Offset = '+strtrim(A[0],2)
	xyouts,0.25+iccd*0.26,0.67,'Raw Data: C'+strtrim(iccd+1,2),/NORMAL,color=1000,charsize=1.1
	xyouts,0.25+iccd*0.26,0.44,'2D Gaussian fit',/NORMAL,color=1000,charsize=1.1
	xyouts,0.25+iccd*0.26,0.20,'Resudual',/NORMAL,color=1000,charsize=1.1
	xyouts,0.25+iccd*0.26,0.01,str2,/NORMAL,color=1000,charsize=0.9
	xyouts,0.25+iccd*0.26,0.03,str1,/NORMAL,color=1000,charsize=0.9

	CCD=clip*scl
	CCD_fit=result*scl
	xlen=(size(CCD))[1]
	ylen=(size(CCD))[2]
	xaxis=findgen(xlen)-xlen/2.0
	yaxis=findgen(ylen)-ylen/2.0
	tr=max(CCD,ii_max_CCD)
	i_max_CCD=ii_max_CCD/xlen
	j_max_CCD=ii_max_CCD-(ii_max_CCD/xlen)*xlen
	CCD_X_mag=CCD[*,i_max_CCD]
	;CCD_X_g=GAUSSFIT(xaxis,CCD_X_mag,CCD_X_coeff,CHISQ=CCD_X_CHI,NTERMS=4)
	CCD_X_g=CCD_fit[*,i_max_CCD]
	CCD_Y_mag=CCD[j_max_CCD,*]
	;CCD_Y_g=GAUSSFIT(yaxis,CCD_Y_mag,CCD_Y_coeff,CHISQ=CCD_Y_CHI,NTERMS=4)
	CCD_Y_g=CCD_fit[j_max_CCD,*]
	plot,xaxis,CCD[*,i_max_CCD],xtitle='X,Y coordinate (pix)',psym=6,ticklen=1,YCHARSIZE=1.2,/device,POSITION=[mag*(2+(xlen+1)*(iccd+1)),(mag*3*ylen+5*mag),mag*((xlen+1)*(iccd+1)+xlen),(mag*4*ylen+5*mag)]
	oplot,xaxis,CCD[*,i_max_CCD],psym=6,color=3000
	oplot,xaxis,CCD[*,i_max_CCD],linestyle=1
	oplot,xaxis,CCD_X_g,THICK=2,color=3000
	oplot,yaxis,CCD[j_max_CCD,*],psym=5,color=6000
	oplot,yaxis,CCD[j_max_CCD,*],linestyle=1
	oplot,yaxis,CCD_Y_g,THICK=2,color=6000

order=4
image_moments, clip, order, moments
print,'CCD =',iccd+1
print,moments
print,''

endfor
!p.noerase=0
end
;
;-----------------------------------------------------------------
;
pro Process_SRM, Event
	Process_SRM_Wid, GROUP_LEADER=Event.top
end
;
;-----------------------------------------------------------------
;
pro Calculate_MSD, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier

X_ind = min(where(RowNames eq 'X Position'))                            ; CGroupParametersGP[2,*] - Peak X  Position
Y_ind = min(where(RowNames eq 'Y Position'))                            ; CGroupParametersGP[3,*] - Peak Y  Position
Z_ind = min(where(RowNames eq 'Z Position'))							; CGroupParametersGP[34,*] - Peak Z Position
Nph_ind = min(where(RowNames eq '6 N Photons'))							; CGroupParametersGP[6,*] - Number of Photons in the Peak
FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number

peaks = where(filter,pkcnt)
if pkcnt le 2 then begin
	void = dialog_message('not enough peaks, cnt='+string(pkcnt))
	return      ; if data not loaded return
endif

Pos = [CGroupParams[X_ind,peaks]*nm_per_pixel,CGroupParams[Y_ind,peaks]*nm_per_pixel,CGroupParams[Z_ind,peaks]]
Frame = CGroupParams[FrNum_ind,peaks]
Frame = Frame - Frame[0]
MSD = dblarr(pkcnt)

for i=0,pkcnt-1 do begin
	;Pos_i = Pos[*,0:i]
	indecis=indgen(pkcnt-i)
	MSD[i] = total((Pos[*,indecis+i]-Pos[*,indecis])^2)/(pkcnt-i)
endfor

plot, MSD, xtitle = 'index', ytitle = 'MSD, nm^2', charsize=2
oplot,MSD, psym=2
end
;
;-----------------------------------------------------------------
;
pro Renumber_Group_Peaks, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

Gr_ind = min(where(RowNames eq '18 Grouped Index'))						; CGroupParametersGP[18,*] - group #
GrInd_ind = min(where(RowNames eq 'Frame Index in Grp'))				; CGroupParametersGP[25,*] - Frame Index in the Group

if n_elements(CGroupParams) le 2 then begin
	image=fltarr(wxsz,wysz)
	print,'no points to display'
	return      ; if data not loaded return
endif

oStatusBar = obj_new('PALM_StatusBar', $
        COLOR=[0,0,255], $
        TEXT_COLOR=[255,255,255], $
        TITLE='Renumbering Group Elements...', $
        TOP_LEVEL_BASE=tlb)
fraction_complete_last=0.0D
pr_bar_inc=0.01D

Ngr=max(CGroupParams[Gr_ind,*])

for j=0L,(Ngr-1) do begin
	Gr_indices = where(CGroupParams[Gr_ind,*] eq j, GrCnt)
	if GrCnt ge 1 then begin
		CGroupParams[GrInd_ind,Gr_indices] = findgen(GrCnt)+1
		fraction_complete=FLOAT(j)/FLOAT((Ngr-1.0))
		if	(fraction_complete-fraction_complete_last) gt pr_bar_inc then begin
			fraction_complete_last=fraction_complete
			oStatusBar -> UpdateStatus, fraction_complete
		endif
	endif
endfor
obj_destroy, oStatusBar

end
;
;-----------------------------------------------------------------
;
pro Correct_GroupSigmaXYZ, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
; Cretated 10.03.2017 GS
; This procedure corrects group Sigma X, Y, and Z multiplies values for Groupsize>1 by sqrt(2)
;
Gr_size_ind = min(where(RowNames eq '24 Group Size'))                    ; CGroupParametersGP[24,*] - total peaks in the group
GrSigX_ind = min(where(RowNames eq 'Group Sigma X Pos'))                ; CGroupParametersGP[21,*] - new x - position sigma
GrSigY_ind = min(where(RowNames eq 'Group Sigma Y Pos'))                ; CGroupParametersGP[22,*] - new y - position sigma
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))                    ; CGroupParametersGP[41,*] - Group Sigma Z


ind_to_correct = where(CGroupParams[Gr_size_ind,*] gt 1, cnt)

if cnt gt 0 then begin
	CGroupParams[GrSigX_ind,ind_to_correct]*=sqrt(2.0)
	CGroupParams[GrSigY_ind,ind_to_correct]*=sqrt(2.0)
	CGroupParams[GrSigZ_ind,ind_to_correct]*=sqrt(2.0)
endif

end
;
;-----------------------------------------------------------------
;
pro OnExtractPeaksML, Event
	ExtractMultiLabelWid, GROUP_LEADER=Event.top
end
