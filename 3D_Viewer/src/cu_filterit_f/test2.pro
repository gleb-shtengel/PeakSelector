;
;  file: test.pro
;
;  RTK, 25-Jun-2009
;  Last update:  25-Jun-2009
;

pro test2
  compile_opt idl2, logical_predicate

  restore, 'group_filterit.sav'
  dlm_load, 'cu_filterit_f'

  s = systime(1)
  v0 = FilterIt(CGroupParams, CGrpSize, ParamLimits)
  e = systime(1)
  t0 = e-s
  print, 'IDL runtime = ', e-s

  FGroupParams = float(CGroupParams)
  index = bytarr(45)
  index[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,26,27,28,29,30,31,32,33,34,35,36]] = 1
  
  s = systime(1)
  v1 = cu_FilterIt_f(FGroupParams, ParamLimits, index)
  e = systime(1)
  t1 = e-s
  print, 'Runtime = ', e-s

  help, v0, v1
  print, 'v0 == v1 ?', array_equal(v0,v1)
  print, 'IDL / CUDA = ', t0/t1
  print
  stop
end

