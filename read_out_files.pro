Function ReadOutFile, out_fname
rstr='hhh'
openr,1,out_fname
while ~ EOF(1) and strmid(rstr,0,5) ne 'Wrote' do begin
	readf,1,rstr,format='(A)'
endwhile
close,1
return, rstr
end

Pro read_out_files, dir
def_wind=!D.window
print,'started Read_Out_Files procedure'
dir=(COMMAND_LINE_ARGS())[0]
print,dir
sh_files=FILE_SEARCH(dir+'*.sh')
if size(sh_files,/N_DIMENSIONS) eq 0 then return
sh_sort=sort(sh_files)
n_sh_files=size(sh_files,/N_ELEMENTS)
yw_size=n_sh_files*90/7<1600
window,9,xpos=0,ypos=50,xsize=900,ysize=yw_size,title='Cluster Processing Progress'

for i=0,n_sh_files-1 do xyouts,0.05,(1-float(i+1)/(n_sh_files+2)),'Started shell: '+sh_files[sh_sort[i]],/normal
xyouts,0.05,0.01,'No OUT files yet',/normal
out_files_completed=0
out_files=FILE_SEARCH(dir+'*.out')
n_out_files=size(out_files,/N_DIMENSIONS)
while size(out_files,/N_DIMENSIONS) eq 0 do out_files=FILE_SEARCH(dir+'*.out')

print,n_sh_files,n_out_files

while (out_files_completed ne 1) or (n_sh_files ne n_out_files) do begin
	out_files=FILE_SEARCH(dir+'*.out')
	n_out_files=n_elements(out_files)
	if n_out_files ge 1 then begin
		out_sort=sort(out_files)
		wset,9
		ERASE
		print,''
		out_files_completed=1
		for i=0,n_out_files-1 do begin
			k=out_sort[i]
			out_file_short=strmid(out_files[k],strpos(out_files[k],'/',/REVERSE_SEARCH)+1)
			status=ReadOutFile(out_files[k])
			xyouts,0.02,(1-float(i+1)/(n_out_files+1)),out_file_short+' :  '+status,/normal
			print,out_file_short+' :  '+status
			out_files_completed=out_files_completed and ((strmid(status,0,5) eq 'Wrote') or (strmid(status,0,30) eq 'Filter returned no valid peaks'))
		endfor
	endif
	wait,5
endwhile

out_files=FILE_SEARCH(dir+'*.out')
n_out_files=n_elements(out_files)
npks_det = ulonarr(n_out_files)
job_prefix = strmid(out_files[0], 0, strpos(out_files[0],'0',/REVERSE_SEARCH))
job_names = strmid(out_files,0,transpose(strpos(out_files,'.out',/REVERSE_SEARCH)))
job_nums = fix(strmid(job_names,strlen(job_prefix)))
sorted_job_nums = sort(job_nums)

for i=0,n_out_files-1 do begin
	k = sorted_job_nums[i]
	out_file_short=strmid(out_files[k],strpos(out_files[k],'/',/REVERSE_SEARCH)+1)
	status=ReadOutFile(out_files[k])
	; status string should have a line with end: Peaks Detected:'+strtrim(npks_det,2)
	search_str = 'Peaks Detected:'
	x = strpos(status,search_str)
	npks_det[i]= x le 0 ? 0 : ulong(strtrim(strmid(status,x+strlen(search_str)),2))
endfor
dir_tm = strmid(dir,0,strpos(dir,'temp_shells',/REVERSE_SEARCH))
save,npks_det,sorted_job_nums,out_files,filename=dir_tm+'/npks_det.sav'
wdelete,9
wset,def_wind
return
end
