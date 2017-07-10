Pro Reassemble_Cluster_Files			;Master program to read data and loop through processing for cluster
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster

data_dir = 'Z:\Cryo_data\ORCA_Data\20170523\PALM4K_Cell14\Run4_488'

cd,data_dir
Filenames = FILE_SEARCH("*.pks" )
nloops=n_elements(Filenames)
;nloops=10

str_search='Iter'
indx = intarr(nloops)

for nlps=0,nloops-1 do begin
	current_file = 	Filenames[nlps]
	iter_pos = strpos(current_file,str_search)
	x=strmid(current_file,iter_pos+strlen(str_search)+1)
	indx[nlps] = fix(strmid(x,0,strpos(x,'_')))
endfor

indx_sorted = sort(indx)

Filenames = (Filenames[indx_sorted])


;file_delete,'temp/npks_det.sav'
xi = 0ul
npks_det = lonarr(nloops)
for nlps=0,nloops-1 do begin			;reassemble little pks files from all the workers into on big one
	restore,Filenames[nlps]
	npks_det[nlps] = n_elements(Apeakparams)
	print, nlps, '    File: ', Filenames[nlps],',   peaks det:',npks_det[nlps]
	wait,0.1
endfor



thefile_no_exten = data_dir
npktot = total(ulong(npks_det), /PRESERVE_TYPE)
print,'Total Peaks Detected: ',npktot
wait,0.2

xi = 0ul
for nlps=0,nloops-1 do begin			;reassemble little pks files from all the workers into on big one
	print,'Concatenating the segment',(nlps+1),'   of ',nloops
	wait,0.1
	current_file = 	Filenames[nlps]
	test1=file_info(current_file)
	if ~test1.exists then stop
	restore,filename = current_file
	if (size(Apeakparams))[2] ne 0 then begin
		if ((size(Apeakparams_tot))[2] eq 0) and (npks_det[nlps] gt 0) then begin
			Apeakparams_tot = replicate(Apeakparams[0],npktot)
			xa = xi + npks_det[nlps]-1uL
			Apeakparams_tot[xi:xa] = Apeakparams
			xi = xa + 1ul
			totdat_tot = totdat
			image_tot = image
			tot_fr = max(Apeakparams.frameindex)+1ul
		endif else begin
			if npks_det[nlps] gt 0 then begin
				Nframes=max(Apeakparams.frameindex)+1ul
				ApeakParams.FrameIndex += tot_fr
				xa = xi + npks_det[nlps]-1uL
				Apeakparams_tot[xi] = Apeakparams	; fast
				xi = xa + 1ul
				tot_fr += Nframes
				totdat_tot=totdat_tot/tot_fr*(tot_fr-Nframes) + totdat/tot_fr*Nframes
				image_tot=image_tot/tot_fr*(tot_fr-Nframes)+image/tot_fr*Nframes
			endif
		endelse
	endif
endfor
if n_elements(Apeakparams_tot) ge 1 then Apeakparams = Apeakparams_tot
totdat=totdat_tot
image=image_tot
saved_pks_filename = data_dir+'\Run4_488_IDL.pks'
save,Apeakparams,image,xsz,ysz,totdat,thefile_no_exten, filename=saved_pks_filename
print,'Wrote result file '+saved_pks_filename


return
end