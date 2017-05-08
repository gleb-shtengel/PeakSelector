; docformat = 'rst'

;+
; Results for tests, test cases, and test suites are reported to the test 
; runner. Each subclass of MGutTestRunner displays them in some way. 
; MGutTestRunner itself is abstract and shouldn't be instantiated.  
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
pro mguttestrunner::reportTestSuiteStart, testsuite, $
                                          ntestcases=ntestcases, $
                                          ntests=ntests, $
                                          level=level
  compile_opt strictarr
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
pro mguttestrunner::reportTestSuiteResult, npass=npass, nfail=nfail, level=level
  compile_opt strictarr

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
pro mguttestrunner::reportTestCaseStart, testcase, ntests=ntests, level=level
  compile_opt strictarr

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
pro mguttestrunner::reportTestCaseResult, npass=npass, nfail=nfail, level=level
  compile_opt strictarr

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
pro mguttestrunner::reportTestStart, testname, level=level
  compile_opt strictarr

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
pro mguttestrunner::reportTestResult, msg, passed=passed
  compile_opt strictarr

end


;+
; Free resources.
;-
pro mguttestrunner::cleanup
  compile_opt strictarr

end


;+
; Initialize the test runner.
;
; :Returns: 
;    1 for success, 0 for failure
;-
function mguttestrunner::init
  compile_opt strictarr
  
  return, 1B
end


;+
; Define member variables.
;
; :Fields:
;    dummy 
;       needed because IDL requires at least one field
;-
pro mguttestrunner__define
  compile_opt strictarr

  define = { MGutTestRunner, dummy : 0L }
end
