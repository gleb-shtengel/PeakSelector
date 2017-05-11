function GroupFilterIt, CGroupParams, CGrpSize, ParamLimits
    compile_opt idl2

    sss = systime(1)
    CGPsz=size(CGroupParams)
    allind=indgen(CGrpSize)
    indecis0=(CGrpSize ge 43)  ?  [9,allind[18:24],26,allind[37:42]] : [9,allind[18:24],26]
    check=where(ParamLimits[indecis0,1] ne 0)
    indecis=indecis0[check]
    t0 = systime(1)-sss & s=systime(1)
    pl0=ParamLimits[indecis,0]#replicate(1,CGPsz[2])
    pl1=ParamLimits[indecis,1]#replicate(1,CGPsz[2])
    t1 = systime(1)-s  &  s=systime(1)
    filter0=(CGroupParams[indecis,*] ge pl0) and (CGroupParams[indecis,*] le pl1)
    t2 = systime(1)-s  &  s=systime(1)
    filter=floor(total(filter0,1)/n_elements(indecis))
    t3 = systime(1)-s  &  s=systime(1)
    peakcount=long(total(filter))
    filterG=filter*(CGroupParams[25,*] eq 1)
    peakcountGroups=long(total(filterG))
    t4 = systime(1)-s

    t5 = systime(1)-sss

    print, 't0 = ', t0
    print, 't1 = ', t1
    print, 't2 = ', t2
    print, 't3 = ', t3
    print, 't4 = ', t4

    print, 'Total = ', t5

    return, filter
end

