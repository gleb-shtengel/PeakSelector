pro filter_test
    compile_opt idl2, logical_predicate

;    dlm_load, 'cu_filterit_f'
    restore, '/groups/hess/hesslab/PalmClusterTest/PeakSelector_V8.5/3D_Viewer/gp.sav'
;    restore, '\\Dm5\hesslab\PalmClusterTest\PeakSelector_V8.5\3D_Viewer\gp.sav'

;	use next 4 lines to further constrain x range  i.e. x min higher, x max lowered
;	ParameterLimits[2,0]=ParameterLimits[2,0]+30.
;	ParameterLimits[19,0]=ParameterLimits[19,0]+30.
;	ParameterLimits[2,1]=ParameterLimits[2,1]-30.
;	ParameterLimits[19,1]=ParameterLimits[19,1]-30.

    a = indgen(45)
    p = [9, a[18:24], 26, a[37:42]]
    index = bytarr(45)
    index[p] = 1

    s=systime(1)
;	tryout the double version
	CGroupParams=double(CGroupParams)
	ParameterLimits=double(ParameterLimits)
	f0 = cu_filterit_d(CGroupParams, ParameterLimits, index)

 ;   f0 = cu_filterit_f(CGroupParams, ParameterLimits, index)
    e=systime(1)
    print, 'CUDA filter time = ', e-s

    s = systime(1)
    CGPsz=size(CGroupParams)
    allind=indgen(45)
    pl0=ParameterLimits[p,0]#replicate(1,CGPsz[2])
    pl1=ParameterLimits[p,1]#replicate(1,CGPsz[2])
    filter0=(CGroupParams[p,*] ge pl0) and (CGroupParams[p,*] le pl1)
    f1=floor(total(filter0,1)/n_elements(p))
    e = systime(1)
    print, 'IDL filter time = ', e-s

    print, 'Filters match = ', array_equal(f0, f1)
    help, f0, f1
	print,total(f0),total(f1)
	print,f0[0:10], f1[0:10]
end
