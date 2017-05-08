; docformat = 'rst'

;+
; Results for tests, test cases, and test suites are reported to the test 
; runner. The MGtestCliRunner displays the results in the output log or in a 
; log file.
;-

;+
; Report a test suite has begun.
; 
; :Params:
;    testsuite : in, required, type=string
;       name of test suite
;
; :Keywords:
;    ntestcases : in, required, type=integer
;       number of test suites/cases contained by the test suite
;    ntests : in, required, type=integer
;       number of tests contained in the hierarchy below this test suite
;    level : in, required, type=level
;       level of test suite
;-
pro mgutclirunner::reportTestSuiteStart, testsuite, $
                                         ntestcases=ntestcases, $
                                         ntests=ntests, $
                                         level=level
  compile_opt strictarr

  indent = level eq 0 ? '' : string(bytarr(level * self.indent) + self.space)
  printf, self.logLun, $
          indent + '"' + testsuite $
          + '" test suite starting (' $
          + strtrim(ntestcases, 2) + ' test suite' $
          + (ntestcases eq 1 ? '' : 's') $          
          + '/case' $
          + (ntestcases eq 1 ? '' : 's') $
          + ', ' $
          + strtrim(ntests, 2) + ' test' + (ntests eq 1 ? '' : 's') $
          + ')'
end


;+
; Report the results of a test suite.
;
; :Keywords:
;    npass : in, required, type=integer 
;          number of passing tests contained in the hierarchy below the test 
;          suite
;    nfail : in, required, type=integer
;       number of failing tests contained in the hierarchy below the test 
;       suite
;    level : in, required, type=integer
;       level of test suite
;-
pro mgutclirunner::reportTestSuiteResult, npass=npass, nfail=nfail, level=level
  compile_opt strictarr

  indent = string(bytarr((level + 1L) * self.indent) + self.space)
  printf, self.logLun, $
          indent + 'Results: ' $
          + strtrim(npass, 2) + ' / ' + strtrim(npass + nfail, 2) $
          + ' tests passed'
end


;+
; Report a test case has begun.
; 
; :Params:
;    testcase {in}{required}{type=string} name of test case
;
; :Keywords:
;    ntests : in, required, type=integer
;       number of tests contained in this test case
;    level : in, required, type=level
;       level of test case
;-
pro mgutclirunner::reportTestCaseStart, testcase, ntests=ntests, level=level
  compile_opt strictarr

  indent = string(bytarr(level * self.indent) + self.space)
  printf, self.logLun, $
          indent + '"' + testcase + '" test case starting'$
          + ' (' + strtrim(ntests, 2) + ' test' + (ntests eq 1 ? '' : 's') + ')' 
end


;+
; Report the results of a test case.
;
; :Keywords:
;    npass : in, required, type=integer
;       number of passing tests
;    nfail : in, required, type=integer
;       number of failing tests
;    level : in, required, type=integer
;       level of test case
;-
pro mgutclirunner::reportTestCaseResult, npass=npass, nfail=nfail, level=level
  compile_opt strictarr

  indent = string(bytarr((level + 1L) * self.indent) + self.space)
  printf, self.logLun, $
          indent + 'Results: ' $
          + strtrim(npass, 2) + ' / ' + strtrim(npass + nfail, 2) $
          + ' tests passed'
end


;+
; Report the start of single test.
; 
; :Params:
;    testname : in, required, type=string
;       name of test
;
; :Keywords:
;    level : in, required, type=integer
;       level of test case
;-
pro mgutclirunner::reportTestStart, testname, level=level
  compile_opt strictarr

  indent = string(bytarr((level + 1L) * self.indent) + self.space)
  printf, self.logLun, indent + testname + ': ', format='(A, $)'
end


;+
; Report the result of a single test.
; 
; :Params:
;    msg : in, required, type=string
;       message to display when test fails
; 
; :Keywords:
;    passed : in, required, type=boolean
;       whether the test passed
;-
pro mgutclirunner::reportTestResult, msg, passed=passed
  compile_opt strictarr

  if (passed) then begin
    printf, self.logLun, 'passed'
  endif else begin
    printf, self.logLun, 'failed' + (msg eq '' ? '' : ' "' + msg + '"')
  endelse
end


;+
; Free resources.
;-
pro mgutclirunner::cleanup
  compile_opt strictarr

  if (self.logLun gt 0) then free_lun, self.logLun
  self->mguttestrunner::cleanup
  !quiet = 0
end


;+
; Initialize the test runner.
;
; :Returns: 
;    1 for success, 0 for failure
;
; :Keywords:
;    filename : in, optional, type=string 
;       if present, output is sent to that file, otherwise output is sent to 
;       stdout
;-
function mgutclirunner::init, filename=filename
  compile_opt strictarr

  if (~self->mguttestrunner::init()) then return, 0B

  if (n_elements(filename) gt 0) then begin
    logDir = file_dirname(filename)
    if (~file_test(logDir)) then file_mkdir, logDir
  endif
  
  if (n_elements(filename) gt 0) then begin
    openw, logLun, filename, /get_lun
    self.logLun = logLun
  endif else begin
    self.logLun = -1L
  endelse

  self.indent = 3L
  self.space = (byte(' '))[0]

  !quiet = 1
  return, 1B
end


;+
; Define member variables.
;
; :Fields:
;    logLun 
;       the logical unit number to send output to (-1L by default)
;    indent
;       number of spaces a single indent should be
;    space
;       byte value of the space character
;-
pro mgutclirunner__define
  compile_opt strictarr

  define = { MGutCliRunner, inherits MGutTestRunner, $
             logLun : 0L, $
             indent : 0L, $
             space : 0B $
           }
end
