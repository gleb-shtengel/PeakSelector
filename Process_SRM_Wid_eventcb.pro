;
; Empty stub procedure used for autoloading.
;
pro Process_SRM_Wid_eventcb
end
;
;-----------------------------------------------------------------
;
function parse_SRM_file, SRM_filename, camera		;
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster

;SRM_data = {			$
;		f_info:'',				$
;		zerodark:0.,			$
;		xsz:256L,				$
;		ysz:256L,				$
;		Nframesmax:1500ul,		$
;		Frm0:0ul,				$
;		FrmN:1499ul,			$
;		Thresholdcriteria:10.0,	$
;		filetype:0,				$
;		LimBotA1:12.6,			$
;		LimTopA1:10000.,		$
;		LimBotSig:0.5,			$
;		LimTopSig:3.5,			$
;		LimChiSq:1500.,			$
;		Cntpere:0.99,			$
;		maxcnt1:2000,			$
;		maxcnt2:0,				$
;		fliphor:0,				$
;		flipvert:0,				$
;		SigmaSym:1,				$
;		MaskSize:5,				$
;		GaussSig:1.0,			$
;		MaxBlck:2048,			$
;		LocalizationMethod:0,	$
;		SparseOversampling:9,	$
;		SparseLambda:1e11,		$
;		SparseDelta:1e-5,		$
;		SpError:0.3,			$
;		SpMaxIter:1e3			}
SRM_data = thisfitcond

	close,1
	openr,1,SRM_filename
	read_line = ''

	header_match = 0
	while (~header_match) and (~ EOF(1)) do begin
		readf,1,read_line
		header_match = strmatch(read_line, camera)
	endwhile

	header_match = 0
	match_str = 'dark_offset'
	start_match = '<' + match_str + '>'
	while (~header_match) and (~ EOF(1)) do begin
		readf,1,read_line
		header_match = strmatch(read_line,('*<'+match_str+'>*'))
	endwhile
	line = strtrim(strsplit(read_line, start_match,/Extract),2)
	value_read=1.0
	reads,line, value_read
	SRM_data.zerodark = value_read

	header_match = 0
	match_str = 'e_to_p_conversion_factor'
	start_match = '<' + match_str + '>'
	while (~header_match) and (~ EOF(1)) do begin
		readf,1,read_line
		header_match = strmatch(read_line,('*<'+match_str+'>*'))
	endwhile
	line = strtrim(strsplit(read_line, start_match,/Extract),2)
	value_read=1.0
	reads,line, value_read
	SRM_data.Cntpere = value_read

	header_match = 0
	match_str = 'flipX'
	start_match = '<' + match_str + '>'
	while (~header_match) and (~ EOF(1)) do begin
		readf,1,read_line
		header_match = strmatch(read_line,('*<'+match_str+'>*'))
	endwhile
	line = strtrim(strsplit(read_line, start_match,/Extract),2)
	value_read=0
	reads,line, value_read
	SRM_data.fliphor = value_read

	header_match = 0
	match_str = 'width'
	start_match = '<' + match_str + '>'
	while (~header_match) and (~ EOF(1)) do begin
		readf,1,read_line
		header_match = strmatch(read_line,('*<'+match_str+'>*'))
	endwhile
	line = strtrim(strsplit(read_line, start_match,/Extract),2)
	value_read=0
	reads,line, value_read
	SRM_data.xsz = value_read

	header_match = 0
	match_str = 'height'
	start_match = '<' + match_str + '>'
	while (~header_match) and (~ EOF(1)) do begin
		readf,1,read_line
		header_match = strmatch(read_line,('*<'+match_str+'>*'))
	endwhile
	line = strtrim(strsplit(read_line, start_match,/Extract),2)
	value_read=0
	reads,line, value_read
	SRM_data.ysz = value_read

	header_match = 0
	match_str = 'number_images'
	start_match = '<' + match_str + '>'
	while (~header_match) and (~ EOF(1)) do begin
		readf,1,read_line
		header_match = strmatch(read_line,('*<'+match_str+'>*'))
	endwhile
	line = strtrim(strsplit(read_line, start_match,/Extract),2)
	value_read=0L
	reads,line, value_read
	SRM_data.Nframesmax = value_read
	SRM_data.FrmN = SRM_data.Nframesmax - 1

	header_match = 0
	match_str = 'raw_data_filename'
	start_match = '<' + match_str + '>'
	while (~header_match) and (~ EOF(1)) do begin
		readf,1,read_line
		header_match = strmatch(read_line,('*<'+match_str+'>*'))
	endwhile
	line_start = strmid(read_line,(strpos(read_line, start_match)+strlen(start_match)))
	filen = strmid(line_start,0, (strlen(line_start)-strlen(match_str)-7))

	sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
	pth = strmid(SRM_filename,0,(max(strsplit(SRM_filename,sep))-1)) + sep
	SRM_data.f_info = pth + string(13B)+filen

	print,SRM_data

	close,1
	return, SRM_data
end
;
;-----------------------------------------------------------------
;
pro Initialize_Process_SRM, wWidget

end
;
;-----------------------------------------------------------------
;
pro OnCancelReExtract, Event
	widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnPick_SRM_File, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
	sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
	filter_to_read = ['*.srm','*.SRM']
	SRM_filename = Dialog_Pickfile(/read,filter=filter_to_read,title='Pick *.SRM File *.srm, or *.SRM')
	fpath = strmid(SRM_filename,0,(max(strsplit(SRM_filename,sep))-1))
	if SRM_filename ne '' then cd,fpath
	SRM_File_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_SRM_Filename')
	widget_control,SRM_File_ID,SET_VALUE = SRM_filename
end
;
;-----------------------------------------------------------------
;
pro Start_SRM_Processing, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster

	SRM_File_ID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_SRM_Filename')
	widget_control,SRM_File_ID,GET_VALUE = SRM_filename
	if strlen(SRM_filename) eq 0 then begin
		z=dialog_message('SRM Filename is empty or does not exist: '+SRM_filename)
		return
	endif

	print,'processing Cam1 file'
	WID_Make_DAT_Duplicates_ID = Widget_Info(Event.Top, find_by_uname='WID_Make_DAT_Duplicates')
	make_duplicates = widget_info(WID_Make_DAT_Duplicates_ID,/button_set)
	; parse SRM file
	thisfitcond=parse_SRM_file(SRM_filename,'*camera1*')
	print,'CAM1 file: ',filen
	WriteInfoFile
	if (file_info(filen + '.raw')).exists then begin
		if make_duplicates then FILE_COPY,filen+'.raw',filen+'.dat',  /OVERWRITE else FILE_MOVE,filen+'.raw',filen+'.dat', /OVERWRITE
	endif

	print,'processing Cam2 file'
	WID_Make_DAT_Duplicates_ID = Widget_Info(Event.Top, find_by_uname='WID_Make_DAT_Duplicates')
	make_duplicates = widget_info(WID_Make_DAT_Duplicates_ID,/button_set)
	; parse SRM file
	thisfitcond=parse_SRM_file(SRM_filename,'*camera2*')
	print,'CAM1 file: ',filen
	WriteInfoFile
	if (file_info(filen + '.raw')).exists then begin
		if make_duplicates then FILE_COPY,filen+'.raw',filen+'.dat',  /OVERWRITE else FILE_MOVE,filen+'.raw',filen+'.dat', /OVERWRITE
	endif

	print,'processing Cam2 file'
	WID_Make_DAT_Duplicates_ID = Widget_Info(Event.Top, find_by_uname='WID_Make_DAT_Duplicates')
	make_duplicates = widget_info(WID_Make_DAT_Duplicates_ID,/button_set)
	; parse SRM file
	thisfitcond=parse_SRM_file(SRM_filename,'*camera3*')
	print,'CAM1 file: ',filen
	WriteInfoFile
	if (file_info(filen + '.raw')).exists then begin
		if make_duplicates then FILE_COPY,filen+'.raw',filen+'.dat',  /OVERWRITE else FILE_MOVE,filen+'.raw',filen+'.dat', /OVERWRITE
	endif

end
;
;-----------------------------------------------------------------
;
