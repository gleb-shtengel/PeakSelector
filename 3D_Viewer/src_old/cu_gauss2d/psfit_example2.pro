;
;  2D example using PSFit
;
;  RTK, 10-Apr-2007
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;  The function
function fg, x, y, p
  return, p[0]*exp(-(x-p[1])^2/p[2]) + p[3]*exp(-(y-p[4])^2/p[5]) + p[6]
end

;  Show a movie of the fit process
pro psfit_example2

  ;  Original data
  print, 'Generating data...'
  x = -10 + 20*randomu(seed,1000)
  y = -10 + 20*randomu(seed,1000)
  z = 30*exp(-(x-5.5)^2/6.6) + 15*exp(-(y-3.3)^2/8.8) + 15.6
  
  ;  Fit and keep the parameter history
  print, 'Fitting...'
  !EXCEPT = 0
  ob = obj_new('PSFit', X=x, Y=y, Z=z, /TWO, F='fg', N=7, PARTICLES=200, PS_IMAX=250, PS_W=[0.9,0.8], PS_DOMAIN=[-100,100], $
               IMAX=6)
  s = systime(1)
  p = ob->Fit(ERROR=err)
  e = systime(1)
  !EXCEPT = 1
  
  ;  Plot the fit
  ob->GetProperty, USING_DLM=dlm, ITERATIONS_DONE=icount
  if (dlm) then print, 'Using DLM' else print, 'Using class'
  obj_destroy, ob

  print, 'Expected parameters:', [30,5.5,6.6,15,3.3,8.8,15.6]
  print, 'Final parameters:', p
  print, 'Iterations done = ', icount
  print, 'Fitting time = ', e-s
  if (err ne '') then  $
    print, err
end
