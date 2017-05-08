; Creates a new mouse manipulation class
pro create_mode_class, modeName

    newFile = 'palm_'+strlowcase(modeName)+'mode__define.pro'

    if file_test(newFile) then begin
        void = dialog_message(/QUESTION, $
            ['File ('+newFile+') already exists and will be overwritten.', $
             'Would you like to continue?'])
        if strupcase(void) EQ 'NO' then $
            return
    endif

    file = 'mode_class_skeleton.txt'
    nLines = file_lines(file)
    code = strarr(nLines)
    openr, lun, file, /GET_LUN
    readf, lun, code
    free_lun, lun
    str = '<ModeName>'
    len = strlen(str)
    lineIndex = strpos(code, str)
    index = where(lineIndex GE 0, count)
    if count EQ 0 then $
        return
    for i = 0, count-1 do begin
        code[index[i]] = strmid(code[index[i]],0,lineIndex[index[i]]) + modeName + $
            strmid(code[index[i]], lineIndex[index[i]]+len)
    endfor

    openw, lun, newFile, /GET_LUN
    printf, lun, transpose(temporary(code))
    free_lun, lun

end