mode = (strtrim(command_line_args(), 2))[0]

gpuinit, HARDWARE=mode eq '1', IDL=mode eq '0', EMULATOR=mode eq '-1'
mgunit, 'gpu_alltests_uts', /html, filename='unittests.html', $
        npass=npass, ntests=ntests

print, strtrim(npass, 2) + ' out of ' + strtrim(ntests, 2) + ' tests passed'

exit
