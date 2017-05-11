;+
; This procedure returns the directory path associated with
; the routine calling this function.  This is useful for
; building applications that need to bootstrap themselves.
;
; @Keyword
;   Base_Name {out}{optional}{type=string}
;       Set this keyword to a named variable to retrieve the
;       base file name of the routine's source.
;
; @Examples <pre>
;   Create a file abc.pro with the contents and run it.
;     PRO ABC
;     PRINT, SourceRoot()
;     END
;   The printed output will be the full path to the
;   directory in which abc.pro was created. </pre>
;
; @Returns
;   The return value is the root directory path to
;   the calling routine's source file or SAVE file.
;
; @History
;   11/02/2004  JLP, RSI - Original version <br>
;   07/07/2004  JLP, RSI - Added Base_Name keyword
;-
Function SourceRoot, Base_Name = BaseName
Compile_Opt StrictArr
On_Error, 2
Help, Calls = Calls
UpperRoutine = (StrTok(Calls[1], ' ', /Extract))[0]
Skip = 0
Catch, ErrorNumber
If (ErrorNumber ne 0) then Begin
    Catch, /Cancel
    ThisRoutine = Routine_Info(UpperRoutine, /Functions, /Source)
    Skip = 1
EndIf
If (Skip eq 0) then Begin
    ThisRoutine = Routine_Info(UpperRoutine, /Source)
EndIf
If (Arg_Present(BaseName)) then Begin
    BaseName = File_BaseName(ThisRoutine.Path)
EndIf
Return, File_DirName(ThisRoutine.Path)
End