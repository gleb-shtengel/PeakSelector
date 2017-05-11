; docformat = 'rst'

;+
; Results for tests, test cases, and test suites are reported to the test 
; runner. The MGutHTMLRunner displays the results in the output HTML file.
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
pro mguthtmlrunner::reportTestSuiteStart, testsuite, $
                                           ntestcases=ntestcases, $
                                           ntests=ntests, $
                                           level=level
  compile_opt strictarr

  printf, self.lun, $
          '<ul class="testsuite"><li><span class="suitename">' $
          + testsuite $
          + '</span> test suite starting (' $
          + strtrim(ntestcases, 2) + ' test suite' $
          + (ntestcases eq 1 ? '' : 's') $
          + '/case' $
          + (ntestcases eq 1 ? '' : 's') $
          + ', ' $
          + strtrim(ntests, 2) + ' test' + (ntests eq 1 ? '' : 's') $
          + ')</li>'
end


;+
; Report the results of a test suite.
;
; :Keywords:
;    npass : in, required, type=integer
;       number of passing tests contained in the hierarchy below the test 
;       suite
;    nfail : in, required, type=integer 
;       number of failing tests contained in the hierarchy below the test 
;       suite
;    level : in, required, type=integer
;       level of test suite
;-
pro mguthtmlrunner::reportTestSuiteResult, npass=npass, nfail=nfail, level=level
  compile_opt strictarr

  printf, self.lun, $
          '<span class="results">Results: ' $
          + strtrim(npass, 2) + ' / ' + strtrim(npass + nfail, 2) $
          + ' tests passed</span></ul>'
end


;+
; Report a test case has begun.
; 
; :Params:
;    testcase : in, required, type=string
;       name of test case
;
; :Keywords:
;    ntests : in, required, type=integer
;       number of tests contained in this test case
;    level : in, required, type=level
;       level of test case
;-
pro mguthtmlrunner::reportTestCaseStart, testcase, ntests=ntests, level=level
  compile_opt strictarr

  printf, self.lun, $
          '<ul class="testcase"><li><span class="casename">' + testcase $
          + '</span> test case starting (' + strtrim(ntests, 2) $
          + ' test' + (ntests eq 1 ? '' : 's') + ')</li>' 
  printf, self.lun, '<ol>'
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
pro mguthtmlrunner::reportTestCaseResult, npass=npass, nfail=nfail, level=level
  compile_opt strictarr

  printf, self.lun, '</ol>'
  printf, self.lun, $
          '<span class="results">Results: ' $
          + strtrim(npass, 2) + ' / ' + strtrim(npass + nfail, 2) $
          + ' tests passed</span></ul>'
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
pro mguthtmlrunner::reportTestStart, testname, level=level
  compile_opt strictarr

  printf, self.lun, '<li>' + testname + ': ', format='(A, $)'
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
pro mguthtmlrunner::reportTestResult, msg, passed=passed
  compile_opt strictarr

  result = keyword_set(passed) ? 'passed' : 'failed'
  printf, self.lun, $
    '<span class="' + result + '">' $
    + result $
    + (keyword_set(passed) ? '': (msg eq '' ? '' : ' "' + msg + '"')) $
    + '</span></li>'
end

     
;+
; Free resources.
;-
pro mguthtmlrunner::cleanup
  compile_opt strictarr

  printf, self.lun, '<span class="dateline">Test results from ' + systime() + '</span>'
  printf, self.lun, '</body></html>'
  if (self.lun gt 0) then free_lun, self.lun
  self->mguttestrunner::cleanup
end


;+
; Initialize the test runner.
;
; :Returns: 
;    1 for success, 0 for failure
; 
; :Keywords: 
;    filename : in, optional, type=string
;       if present, output is sent that file, otherwise output is sent to 
;       stdout
;-
function mguthtmlrunner::init, filename=filename
  compile_opt strictarr

  if (~self->mguttestrunner::init()) then return, 0B

  ; make the directory the output file is in if it doesn't exist
  if (n_elements(filename) gt 0) then begin
    dir = file_dirname(filename)
    if (~file_test(dir)) then file_mkdir, dir
  endif
  
  ; setup the LUN for the output
  if (n_elements(filename) gt 0) then begin
    openw, lun, filename, /get_lun
    self.lun = lun
  endif else begin
    self.lun = -1L
  endelse

  printf, self.lun, '<html><head>'
  printf, self.lun, '<title>Test results</title>'
  printf, self.lun, '<style type="text/css" media="all">'
  styleFilename = mg_src_root() + 'style.css'
  styles = strarr(file_lines(styleFilename))
  openr, styleLun, styleFilename, /get_lun
  readf, styleLun, styles
  free_lun, styleLun
  printf, self.lun, transpose(styles)
  printf, self.lun, '</style></head><body>'
  return, 1B
end


;+
; Define member variables.
;
; :Fields:
;    lun 
;       the logical unit number to send output to (-1L by default)
;-
pro mguthtmlrunner__define
  compile_opt strictarr
  
  define = { MGutHTMLRunner, inherits MGutTestRunner, $
             lun: 0L $
           }
end