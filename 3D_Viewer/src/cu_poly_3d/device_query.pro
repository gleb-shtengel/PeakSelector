;
;  Query a CUDA GPU
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

pro device_query
  compile_opt idl2
  
  dlm_load, 'gpu_device_query'

  if (gpu_device_count() ne 0) then begin
    for i=0, gpu_device_count()-1 do begin
      name = gpu_device_property('device_name', i)
      globalMem = gpu_device_property('global_memory', i)
      sharedMem = gpu_device_property('shared_memory', i)
      regs = gpu_device_property('registers', i)
      warp = gpu_device_property('warp_size', i)
      pitch = gpu_device_property('memory_pitch', i)
      threadsPerBlock = gpu_device_property('threads_per_block', i)
      tx = gpu_device_property('threads_x', i) 
      ty = gpu_device_property('threads_y', i) 
      tz = gpu_device_property('threads_z', i) 
      gx = gpu_device_property('grid_x', i)
      gy = gpu_device_property('grid_y', i)
      gz = gpu_device_property('grid_z', i)
      const = gpu_device_property('constant_memory', i)
      major = gpu_device_property('major_revision', i)
      minor = gpu_device_property('minor_revision', i)
      clock = gpu_device_property('clock_rate', i)
      talign = gpu_device_property('texture_alignment', i)
      over = gpu_device_property('device_overlap', i)
      mp = gpu_device_property('multiprocessor_count', i)
      
      print
      print, 'Device #', i+1
      print
      print, 'Name                    : ', name
      print, 'Total memory            : ',globalMem
      print, 'Shared memory per block : ',sharedMem
      print, 'Registers per block     : ',regs
      print, 'Warp size               : ',warp
      print, 'Memory pitch            : ',pitch
      print, 'Max Threads Per Block   : ',threadsPerBlock 
      print, 'Thread Dimensions       : ',tx,'x',strtrim(ty,2),'x',strtrim(tz,2)
      print, 'Grid Dimensions         : ',gx,'x',strtrim(gy,2),'x',strtrim(gz,2)
      print, 'Total Constant Memory   : ',const 
      print, 'Major Revision Number   : ',major 
      print, 'Minor Revision Number   : ',minor 
      print, 'Clock Rate (kHz)        : ',clock 
      print, 'Texture Alignment       : ',talign
      print, 'Device Overlap          : ',over
      print, 'Multiprocessor Count    : ',mp
      print
      print
    endfor
  endif
end

