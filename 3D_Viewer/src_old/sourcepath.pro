;+
; This procedure returns the directory path associated with
; the routine calling this function.  This is useful for
; building applications that need to bootstrap resource and
; configuration files when the installation directory may not
; be known until run time.  Use this function in conjunction
; with FILEPATH to build platform-independent file path strings
; to your resources. <br>
;
; For example, <pre>
;   b = WIDGET_BUTTON(tlb, /BITMAP, $
;     VALUE=FILEPATH('up.bmp', ROOT = SourcePath(), SUBDIR = ['resource'])</pre>
; This will search for a file named "up.bmp" in the subdirectory named
; "resource" below the directory in which is located the source code
; (or SAVE file) for the routine containing the above statement.
;
; @Keyword
;   Base_Name {out}{optional}{type=string}
;       Set this keyword to a named variable to retrieve the
;       base file name of the routine's source.
; @Keyword
;   Extra {in}{optional}
;       Any extra keywords are passed to the FILE_DIRNAME
;       function, for example /MARK_DIRECTORY.
;
; @Returns
;   The return value is the root directory path to
;   the calling routine's source file or SAVE file.
;
; @Examples <pre>
;   Create a file myapp.pro with the contents and run it.
;     PRO MYAPP
;     PRINT, SourcePath()
;     END
;   The printed output will be the full path to the
;   directory in which abc.pro was created, regardless of
;   IDL's current working directory.</pre>
;
; @Requires
;   IDL 6.2 or later.  See sourceroot.pro for this functionality
;   for earlier version.
;
; @Categories
;   Bootstrap,path,source
;
; @Author
;   Jim Pendleton, RSI Global Services
;
; @History
;   03/18/2005  JLP, RSI - Original version <br>
;   10/10/2005 JLP, RSI - On Ben Tupper's suggestion, added _EXTRA
;
; @Copyright
;   ITT Visual Software Solutions, 2006
;
; @File_Comments
; This procedure returns the directory path associated with
; the routine calling this function.
;-
Function SourcePath, Base_Name = BaseName, _Extra = Extra
Compile_Opt StrictArr
On_Error, 2
Stack = Scope_Traceback(/Structure)
Filename = Stack[N_elements(Stack) - 2].Filename
If (Arg_Present(BaseName)) then Begin
    BaseName = File_BaseName(Filename)
EndIf
Return, File_DirName(Filename, _Extra = Extra)
End