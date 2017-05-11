; docformat = 'rst'

;+
; Test suites are containers for test cases. Either subclass MGutTestSuite and 
; add test suites/test cases in its init method or create a MGutTestSuite and 
; use the add method to add test suites/cases.
;-

;+
; Recompile the class definition before creating the object to make sure it is
; the newest definition (convenient when making changes to a test).
;-
pro mguttestsuite::_recompile, classname
  compile_opt strictarr

  error = 0L
  catch, error
  if (error ne 0L) then begin
    catch, /cancel
    return
  endif

  resolve_routine, classname + '__define'
end


;+
; Create a new test case or test suite, but check for errors while creating
; it.
;
; :Returns: 
;    obj
;
; :Params:
;    testName : in, required, type=string
;       classname of test case or test suite to create
;
; :Keywords:
;    error : out, optional, type=boolean
;       0 if no error and 1 if an error
;    _extra : in, optional, type=keywords
;       keywords to OBJ_NEW for test cases and test suites
;-
function mguttestsuite::_makeTestCase, testName, error=error, _extra=e
  compile_opt strictarr

  error = 0L
  catch, error
  if (error ne 0L) then begin
    catch, /cancel
    return, obj_new()
  endif

  self->_recompile, testName

  return, obj_new(testName, test_runner=self.testRunner, _strict_extra=e)
end


;+
; Run the contained test suites or test cases.
;-
pro mguttestsuite::run
  compile_opt strictarr

  self->getProperty, name=name, ntestcases=ntestcases, ntests=ntests
  self.testRunner->reportTestSuiteStart, name, $
                                         ntestcases=ntestcases, $
                                         ntests=ntests, $
                                         level=self.level

  ntestcases = self.testcases->count()
  for t = 0L, ntestcases - 1L do begin
    otestcase = self.testcases->get(position=t)
    otestcase->run

    ; accumulate results
    otestcase->getProperty, npass=npass, nfail=nfail
    self.npass += npass
    self.nfail += nfail
  endfor

  self.testRunner->reportTestSuiteResult, npass=self.npass, $
                                          nfail=self.nfail, $
                                          level=self.level
end


;+
; Add a scalar or array of test suites or test cases.
;
; :Params:
;    tests : in, required, type=strarr
;       classnames of test suites or test cases
;
; :Keywords:
;    all : in, optional, type=boolean
;       set to add all the files in the current directory that end in 
;       "_ut__define.pro" (the current directory is defined to be the 
;       directory where the method calls this method is located)
;-
pro mguttestsuite::add, tests, all=all
  compile_opt strictarr

  if (keyword_set(all)) then begin
    ; first, add all the unit tests in the current directory

    ; find matching files
    testFiles = file_search(self.home + '*_ut__define.pro', $
                            /fold_case, count=nTestFiles)

    ; extract just classname from each file
    myTests = file_basename(testFiles)
    for t = 0L, nTestFiles - 1L do begin
      ; "__define.pro" is 12 characters long
      myTests[t] = strmid(myTests[t], 0, strlen(myTests[t]) - 12L)
      self->add, myTests[t]
    endfor

    ; second, for each directory in the current directory, either:
    ;   1) add the *_uts__define.pro found there, or
    ;   2) if no _uts__define.pro found, create one and add /ALL
    testDirs = file_search(self.home + '*', $
                           /test_directory, /mark_directory, $
                           count=nTestDirs)
    for d = 0L, nTestDirs - 1L do begin
      ; if there is a *_uts__define.pro here then add it
      uts = file_search(testDirs[d] + '*_uts__define.pro', $
                        count=nTestSuites)
      if (nTestSuites gt 0) then begin
        suiteName = file_basename(uts[0])
        ; "__define.pro" is 12 characters long
        suiteName = strmid(suiteName, 0, strlen(suiteName) - 12L)
        if (suiteName ne self.name) then self->add, suiteName

        continue   ; finished with this directory
      endif

      ; if there isn't a *_uts__define.pro, but there are unit tests here
      ; then create a new test suite
      uts = file_search(testDirs[d] + '*_ut__define.pro', $
                        count=nTestCases)
      if (nTestCases eq 0) then continue   ; finished with this directory

      otestsuite = obj_new('MGutTestSuite', home=testDirs[d], $
                           name=file_basename(testDirs[d]), $
                           test_runner=self.testRunner)
      otestsuite->add, /all
      self.testcases->add, otestsuite
    endfor
  endif else begin
    for t = 0L, n_elements(tests) - 1L do begin
      ; don't add yourself to yourself
      if (tests[t] eq self.name) then continue

      ; see if test is valid
      otestcase = self->_makeTestCase(tests[t], error=error)
      if (error ne 0L) then begin
        print, 'Test case ' + tests[t] + ' not valid'
        continue
      endif

      ; test case is OK so now set it up
      otestcase->setLevel, self.level + 1L
      self.testcases->add, otestcase
    endfor
  endelse
end


;+
; Get properties of the object.
;
; :Keywords:
;    name : out, optional, type=string
;       name of the object
;    npass : out, optional, type=integer
;       number of passing tests contained in the hierarchy below this object
;    nfail : out, optional, type=integer
;       number of failing tests contained in the hierarchy below this object
;    ntestcases : out, optional, type=integer
;       number of directly contained test suites or test cases
;    ntests : out, optional, type=integer
;       number of tests contained in the hierarchy below this object
;-
pro mguttestsuite::getProperty, name=name, npass=npass, nfail=nfail, $
                                ntestcases=ntestcases, ntests=ntests
  compile_opt strictarr

  name = self.name
  npass = self.npass
  nfail = self.nfail
  if (arg_present(ntestcases)) then ntestcases = self.testcases->count()

  if (arg_present(ntests)) then begin
    ntests = 0L
    for t = 0L, self.testcases->count() - 1L do begin
      otestcase = self.testcases->get(position=t)
      otestcase->getProperty, ntests=nCaseTests
      ntests += nCaseTests
    endfor
  endif

end


;+
; Test suites can contain other test suites or test cases. The level is the
; number of layers down from the top most test suite (level 0).
;
; :Params:
;    level : in, required, type=integer
;       new level of object
;-
pro mguttestsuite::setLevel, level
  compile_opt strictarr

  self.level = level
  for t = 0L, self.testcases->count() - 1L do begin
    testcase = self.testcases->get(position=t)
    testcase->setLevel, level + 1
  endfor
end


;+
; Free resources.
;-
pro mguttestsuite::cleanup
  compile_opt strictarr

  obj_destroy, self.testcases
end


;+
; Initialize test suite.
;
; :Returns:
;    1 for success, 0 for failure
;
; :Keywords:
;    name : in, optional, type=string, default=classname
;       name of the test suite
;    home : in, optional, type=string, default=''
;       location of the root of the test suite
;    test_runner : in, required, type=object
;       subclass of MGtestRunner
;-
function mguttestsuite::init, name=name, home=home, test_runner=testRunner
  compile_opt strictarr

  self.name = n_elements(name) eq 0 ? strlowcase(obj_class(self)) : name

  if (n_elements(home) eq 0) then begin
    ; get directory of caller's source code (subtract 2L to *caller* dir)
    traceback = scope_traceback(/structure)
    callingFrame = traceback[n_elements(traceback) - 2L > 0]
    self.home = file_dirname(callingFrame.filename, /mark_directory)
  endif else begin
    self.home = strmid(home, 0, 1, /reverse_offset) eq path_sep() $
      ? home $
      : home + path_sep()
  endelse

  self.level = 0L

  self.testRunner = testRunner

  self.testcases = obj_new('IDL_Container')

  self.npass = 0L
  self.nfail = 0L

  return, 1B
end


;+
; Define member variables.
;
; :Fields:
;    name 
;       name of the object
;    level 
;       number of layers below the top-most containing test suite
;    testcases
;       IDL_Container holding test suites or test cases
;    testRunner
;       subclass of MGutTestRunner
;    npass 
;       number of passing tests contained in the hierarchy below this test 
;       suite
;    nfail 
;       number of failing tests contained in the hierarchy below this test 
;       suite
;-
pro mguttestsuite__define
  compile_opt strictarr

  define = { MGutTestSuite, $
             name : '', $
             home : '', $
             level : 0L, $
             testcases : obj_new(), $
             testRunner : obj_new(), $
             npass : 0L, $
             nfail : 0L $
           }
end
