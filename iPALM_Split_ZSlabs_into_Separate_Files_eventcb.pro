
;
;-----------------------------------------------------------------
; Empty stub procedure used for autoloading.
;
pro iPALM_Split_ZSlabs_into_Separate_Files_eventcb
end

;
;-----------------------------------------------------------------
;
pro Initialize_SplitZslabs_Wid, wWidget
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope


filename = SavFilenames[max(where(SavFilenames ne ''))]

pos_filename_wind=strpos(filename,'\',/reverse_search,/reverse_offset)
pos_filename_unix=strpos(filename,'/',/reverse_search,/reverse_offset)
pos_filename=max([pos_filename_wind,pos_filename_unix])

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

RawFilename = RawFilenames[max(where(RawFilenames ne ''))]

pos_rawfilename_wind=strpos(RawFilename,'\',/reverse_search,/reverse_offset)
pos_rawfilename_unix=strpos(RawFilename,'/',/reverse_search,/reverse_offset)
pos_rawfilename=max([pos_rawfilename_wind,pos_rawfilename_unix])
ref_file = ''
if (pos_rawfilename gt 0) and (pos_filename gt 0) then begin
	ref_file=strmid(filename,0,pos_filename)+sep+strmid(RawFilename,pos_rawfilename+1,strlen(RawFilename)-pos_rawfilename-1)
endif


FilePrefixWidID = Widget_Info(wWidget, find_by_uname='WID_TEXT_FilePrefix')
widget_control, FilePrefixWidID, SET_VALUE = ref_file
print,'Will use this prefix for ZSlabs:  ', ref_file

end
;
;-----------------------------------------------------------------
;
pro OnPickFilePrefix, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope

FilePrefixWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_FilePrefix')
widget_control, FilePrefixWidID, GET_VALUE = ref_file

filename = Dialog_Pickfile(/write, get_path=fpath, file=ref_file, filter=['*'],title='Select File to use as a Prefix')
if filename eq '' then return
cd,fpath
widget_control, FilePrefixWidID, SET_VALUE = filename
print,'Will use this prefix for ZSlabs:  ', filename

end
;
;-----------------------------------------------------------------
;
pro OnSplitZSlabsOK, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common bridge_stuff, allow_bridge, bridge_exists, n_br_loops, n_br_max, fbr_arr, n_elem_CGP, n_elem_fbr, npk_tot, imin, imax, shmName_data, OS_handle_val1, shmName_filter, OS_handle_val2
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
common calib, aa, wind_range, nmperframe, z_cal_min, z_cal_max, z_unwrap_coeff, ellipticity_slopes, d, wfilename, cal_lookup_data, cal_lookup_zz, GS_anc_fname, GS_radius
common Zdisplay, Z_scale_multiplier, vbar_top
common Offset, PkWidth_offset
common Multy_Z_Slabs, Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[where(names eq 'WID_BASE_0_PeakSelector')]


if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

FilePrefixWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_FilePrefix')
widget_control, FilePrefixWidID, GET_VALUE = ref_file

WidIDNumberOfZSlabs = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Number_of_ZSlabs')
widget_control,WidIDNumberOfZSlabs, get_value = NumberOfZSlabs
if NumberOfZSlabs eq 1 then return

WidIDFramesPerZSlab = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_FramesPerZSlab')
widget_control,WidIDFramesPerZSlab, get_value = FramesPerZSlab

sep = !VERSION.OS_family eq 'unix' ? '/' : '\'

FrNum_ind = min(where(RowNames eq 'Frame Number'))						; CGroupParametersGP[9,*] - frame number
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
GrZ_ind = min(where(RowNames eq 'Group Z Position'))					; CGroupParametersGP[40,*] - Group Z Position
GrSigZ_ind = min(where(RowNames eq 'Group Sigma Z'))					; CGroupParametersGP[41,*] - Group Sigma Z
GrCoh_ind = min(where(RowNames eq '42 Group Coherence'))				; CGroupParametersGP[42,*] - Group Coherence
Gr_Ell_ind = min(where(RowNames eq 'XY Group Ellipticity'))				; CGroupParametersGP[46,*] - Group Ellipticity
UnwGrZ_ind = min(where(RowNames eq 'Unwrapped Group Z'))				; CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
UnwGrZErr_ind = min(where(RowNames eq 'Unwrapped Group Z Error'))		; CGroupParametersGP[48,*] - Group Z Position Error


NFrames = ((size(thisfitcond))[2] eq 8)	?	(thisfitcond.Nframesmax > long64(max(CGroupParams[FrNum_ind,*])+1))	: long64(max(CGroupParams[FrNum_ind,*])+1)
FR=dindgen(NFrames)

CGroupParams_init = CGroupParams

Slab_ID = floor(CGroupParams[FrNum_ind,*] / FramesPerZSlab) mod NumberOfZSlabs


;********* Status Bar Initialization  ******************
oStatusBar = obj_new('PALM_StatusBar', COLOR=[0,0,255], TEXT_COLOR=[255,255,255],   TITLE='Saving separate Z-Slab files ...', TOP_LEVEL_BASE=tlb)
fraction_complete_last=0.0
pr_bar_inc=0.01
; ***********

for j=0,(NumberOfZSlabs-1) do begin
	Slab_inds = where(Slab_ID eq j)
	CGroupParams = CGroupParams_init[*, Slab_inds]

	Slab_filename=AddExtension(ref_file, ('_L'+strtrim(j,2)+'_IDL.sav'))

	; *********** Status Bar Update  **********
    fraction_complete = float(j+1)/float(NumberOfZSlabs)
    if (fraction_complete - fraction_complete_last) ge pr_bar_inc then begin
         oStatusBar -> UpdateStatus, fraction_complete
         fraction_complete_last = fraction_complete
    endif
    ; ***********

	save, CGroupParams, CGrpSize, ParamLimits, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames, MLRawFilenames,$
			GuideStarDrift, FiducialCoeff, FlipRotate, thisfitcond, RowNames, $
			lambda_vac,nd_water, nd_oil, nmperframe, wind_range, aa, z_unwrap_coeff, cal_lookup_data, ellipticity_slopes, $
			nm_per_pixel, wfilename, PkWidth_offset, Z_scale_multiplier, grouping_gap, grouping_radius100, $
			Rundat_Filename, State_Voltages, State_Frames, Transition_Frames, State_Zs, ZvsV_Slope, $
			lab_filenames, sp_cal_file, cal_spectra, sp_d, Max_sp_num, sp_window, cal_frames, sp_dispersion,  sp_offset, filename=Slab_filename

endfor

;********* Status Bar Close ******************
obj_destroy, oStatusBar
; ***********

CGroupParams = CGroupParams_init

widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
