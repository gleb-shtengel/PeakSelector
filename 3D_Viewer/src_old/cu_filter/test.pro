;
;  file: test.pro
;
;  RTK, 09-Mar-2009
;  Last update:  09-Mar-2009
;

pro test
  compile_opt idl2, logical_predicate

  restore, 'group_filterit.sav'
  dlm_load, 'cu_filter_f'
  CGroupParams = float(CGroupParams)

  s = systime(1)
  v0 = GroupFilterIt(CGroupParams, CGrpSize, ParamLimits)
  e = systime(1)
  t0 = e-s
  print, 'IDL runtime = ', e-s
 
  s = systime(1)
  v1 = cu_GroupFilterIt_f(CGroupParams, ParamLimits)
  e = systime(1)
  t1 = e-s
  print, 'Runtime = ', e-s

  help, v0, v1
  print, 'v0 == v1 ?', array_equal(v0,v1)
  print, 'IDL / CUDA = ', t0/t1
  print
  
  s = systime(1)
  v0 = FilterIt(CGroupParams, CGrpSize, ParamLimits)
  e = systime(1)
  t0 = e-s
  print, 'IDL runtime = ', e-s
 
  s = systime(1)
  v1 = cu_FilterIt_f(CGroupParams, ParamLimits)
  e = systime(1)
  t1 = e-s
  print, 'Runtime = ', e-s

  help, v0, v1
  print, 'v0 == v1 ?', array_equal(v0,v1)
  print, 'IDL / CUDA = ', t0/t1
  
  stop
end

