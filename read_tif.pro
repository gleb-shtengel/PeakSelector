; - Replaced ReadTillNewSpaceSpace() with ReadTillNewSpace() in the version read of TinstaImage
; - Replaced franes_in_chunk with frames_in_chunk in TImage
; - Added number_of_chunks = number_of_chunks < TImageParams.no_images	to make sure that the number of chunks does not
; exceed the number of images
; - replaced i with number_of_chunks in two places in remainder reading inside TImage
; -  modified the TImage procedure so that the remainder frames are always read but only returned as ThisData if requested.
; initially they were not read if the remainedr was not requested, which caused errors reading the rest of the data
; - replaced "data" with "thisdata" in the initial line thisdata=reform(data,xpix,ypix,frames_in_chunk)inside TImage
; - added creation and saving the data into .dat file and .txt file


;
; -------------------------------------
;
Pro Read_tif,pth,tif_filename,config_path, ini_filename, thisfitcond
if (size(thisfitcond))[2] ne 8 then LoadThiFitCond,ini_filename,thisfitcond
test0=file_info(pth+tif_filename)
if test0.EXISTS eq 0 then begin
	z=dialog_message('File does not exist')
	return      ; if data not loaded return
endif

test1=QUERY_TIFF ((pth+tif_filename) , tif_info)

if test1 eq 0 then begin
	z=dialog_message('Incorrect TIF file')
	return      ; if data not loaded return
endif

filen=StripExtension(tif_filename)

ref_filename=(pth+filen+'.txt')
thisfitcond.f_info = filen						; filename wo extension
thisfitcond.xsz = tif_info.dimensions[0]							; x-size (pixels)
thisfitcond.ysz = tif_info.dimensions[1]							; y-size (pixels)
thisfitcond.Nframesmax = tif_info.num_images	; max Number of Frames
thisfitcond.FrmN = tif_info.num_images-1L	; No Last Frame
thisfitcond.filetype = 1						; tif file
		close,2
		openw, 2, ref_filename
		printf, 2, pth
		printf, 2, filen								; filename wo extension
		printf, 2, thisfitcond.zerodark					; Zero Dark Counts
		printf, 2, thisfitcond.xsz						; x-size (pixels)
		printf, 2, thisfitcond.ysz						; y-size (pixels)
		printf, 2, thisfitcond.Nframesmax				; max Number of Frames
		printf, 2, thisfitcond.Frm0						; No First Frame
		printf, 2, thisfitcond.FrmN						; No Last Frame
		printf, 2, thisfitcond.THresholdcriteria		; Threshold Fit Criteria
		printf, 2, thisfitcond.filetype					; File Type
		printf, 2, thisfitcond.LimBotA1					; Min Peak Amplitude
		printf, 2, thisfitcond.LimTopA1					; Max Peak Amplitude
		printf, 2, thisfitcond.LimBotSig				; Min Peak Width
		printf, 2, thisfitcond.LimTopSig				; Max Peak Width
		printf, 2, thisfitcond.LimChiSq					; Limit ChiSq
		printf, 2, thisfitcond.Cntpere					; Counts per electron
		printf, 2, thisfitcond.maxcnt1					; Max peak # in a single frame, Iteration 1
		printf, 2, thisfitcond.maxcnt2					; Max peak # in a single frame, Iteration 2
		printf, 2, thisfitcond.fliphor					; Flip Horisontal
		printf, 2, thisfitcond.flipvert					; Flip Vertical
		printf, 2, thisfitcond.SigmaSym					; SigmaSym
		printf, 2, thisfitcond.MaskSize					; Gaussian Mask (half) size (parameter d)
		printf,	2, thisfitcond.GaussSig
		printf,	2, thisfitcond.MaxBlck
		printf, 2, thisfitcond.LocalizationMethod
		printf, 2, thisfitcond.SparseOversampling
		printf, 2, thisfitcond.SparseLambda
		printf, 2, thisfitcond.SparseDelta
		printf,	2, thisfitcond.SpError
		printf,	2, thisfitcond.SpMaxIter
		close,2
end
;
