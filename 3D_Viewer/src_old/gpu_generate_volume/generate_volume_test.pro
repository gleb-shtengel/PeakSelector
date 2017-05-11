;
;  file:  generate_volume_test.pro
;
;  Reads a peaks file and generates a volume using the gpu_generate_volume
;  DLM.
;
;  RTK, 20-Oct-2008
;  Last update:  21-Oct-2008
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

pro generate_volume_test, fname
  compile_opt idl2

  ;  Ask for a peaks file if none given
  if (n_elements(fname) eq 0) then  $
    fname = dialog_pickfile(TITLE='Select a peaks file...', /MUST_EXIST)
  if (fname eq '') then  $
    return

  ;  Read the peaks file and pull out the necessary information
  openr, u, fname, /GET_LUN
  t = fstat(u)
  fsize = t.size
  free_lun, u

  p = read_binary(fname, DATA_TYPE=4, DATA_DIM=fsize/4)

  zscale = p[0]
  df     = p[1]
  x_low  = p[2]
  x_high = p[3]
  y_low  = p[4]
  y_high = p[5]
  z_low  = p[6]
  z_high = p[7]
  xdim   = long(p[8])
  ydim   = long(p[9])
  zdim   = long(p[10])
  nelem  = xdim*ydim*zdim
  npeaks = long(p[11])
  peaks  = p[12:*]

  ;  Print the file info
  print
  print, "zscale           = ", strtrim(zscale,2)
  print, "df               = ", strtrim(df,2)
  print, "x_low, x_high    = ", string(x_low,x_high,FORMAT='(-F,", ",-F)')
  print, "y_low, y_high    = ", string(y_low,y_high,FORMAT='(-F,", ",-F)')
  print, "z_low, z_high    = ", string(z_low,z_high,FORMAT='(-F,", ",-F)')
  print, "xdim, ydim, zdim = "+strtrim(xdim,2)+", "+strtrim(ydim,2)+", "+strtrim(zdim,2)
  print, "npeaks           = ", strtrim(npeaks,2)

  ;  Load the DLM
  dlm_load, 'gpu_generate_volume'

  ;  Generate the output volume
  volData = fltarr(xdim,ydim,zdim)
  s = systime(1)
  gpu_generate_volume, volData, peaks, zscale, df, 0,  $
    [x_low,x_high], [y_low,y_high], [z_low,z_high]
  e = systime(1)

  ;  Report how long the volume generation took
  print
  print, 'Volume generation took ' + string(e-s,FORMAT='(F10.5)') + ' sec'
  print
  
  ;  Write the volume to disk as a save file
  save, volData, file='volume.sav'
end

