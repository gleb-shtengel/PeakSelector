; docformat = 'rst'

;+
; Runs unit tests. 
; 
; :Params:
;    tests : in, optional, type=strarr
;       array of test suites and/or test cases
;
; :Keywords:
;    filename : in, optional, type=string
;       name of file to send output to; if not present sends output to the 
;       output log
;    html : in, optional, type=boolean
;       set to indicate HTML output instead of plain text
;    npass : out, optional, type=long
;       number of tests that passed
;    ntests : out, optional, type=long
;       number of tests      
;-
pro mgunit, tests, filename=filename, html=html, npass=npass, ntests=ntests
  compile_opt strictarr

  runnerName = keyword_set(html) ? 'MGutHTMLRunner' : 'MGutCliRunner'
  testRunner = obj_new(runnerName, filename=filename)
  
  npass = 0L
  nfail = 0L
  
  if (n_elements(tests) gt 0) then begin
    testsuite = obj_new('MGutTestSuite', $
                        test_runner=testRunner, $
                        name='All tests')
    testsuite->add, tests
    testsuite->run
    testsuite->getProperty, npass=npass, ntests=ntests
    obj_destroy, testsuite
  endif

  obj_destroy, testRunner
end
