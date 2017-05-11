;--------------------------------------------------------------
;+
;  Add the path of the source for the current routine to the path.
;-
;--------------------------------------------------------------
pro addthispath
    compile_opt StrictArr
    ON_ERROR, 2
;
; Where does this routine reside?  We're going to
; build our !path based on this location as the root.
;
    expanded = EXPAND_PATH('+' + SourceRoot())
;
; Do we already have this expanded path listing in the
; current !PATH?
;
    inPath = STRPOS(!PATH, expanded)
    case inpath of
        0 : begin
;
; Yes, and it's at the beginning.  It's safe to return
; from here.
;
            RETURN
            end
        -1 : begin
;
; No.  We need to added our expanded path to the start
; of the search path.
;
            !PATH = expanded + PATH_SEP(/SEARCH_PATH) + !PATH
            end
        else : begin
;
; Yes, it's in the path, but it's not at the beginning.  So we
; want to extract it from where it is and move it to the start.
;
; First we need to escape the "\" characters in the expanded path.
;
            e = STRJOIN(STRTOK(expanded, '\', /EXTRACT), '\\')
;
; Next divide !PATH using the expanded path as our separator.
;
            components = STRTOK(!PATH, e, /EXTRACT, /REGEX, $
                COUNT = count)
            if (count eq 1) then begin
;
; If we only end up with one separated component, the expanded path
; was either at the beginning or the end.  In either case, push it
; to the beginning.
;
                !PATH = expanded + $
                    PATH_SEP(/SEARCH_PATH) + components[0]
            endif else begin
;
; If we end up with multiple components, then the expanded path
; was in the middle of other path elements.  Stitch the other
; elements back together and put the expanded path at the start.
;
                !PATH = expanded + PATH_SEP(/SEARCH_PATH) + $
                    STRJOIN(components, '')
            endelse
;
; As a result of all this appending, we may end up with some
; duplicated semicolons.  They don't do any harm, but why have
; them around?  This part strips the duplicates and reappends
; the components using a single semicolon.
;
            components = STRTOK(!PATH, ';;', /EXTRACT, /REGEX, $
                COUNT = count)
            if (count gt 1) then begin
                !PATH = STRJOIN(components, ';')
            endif
            end
    endcase
end
