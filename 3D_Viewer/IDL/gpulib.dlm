#########################################################################
#
# gpulib.dlm
# 
# Module description file for GPULIB
#
# Copyright (C) 2008 Tech-X Corporation. All rights reserved.
#
# This file is part of GPULib.
#
# This file may be distributed under the terms of the GNU General Public
# License (GPL). This file may be distributed and/or modified under the
# terms of the GNU General Public License version 2 as published by the
# Free Software Foundation.
#
# This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
# WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#
# Licensees holding valid Tech-X commercial licenses may use this file
# in accordance with the Tech-X Commercial License Agreement provided
# with the Software.
#
# See http://gpulib.txcorp.com/ or email sales@txcorp.com for more information.
#
# This work was in part funded by NASA SBIR Phase II Grant #NNG06CA13C.
#
#########################################################################

MODULE gpulib
VERSION 1.0
SOURCE Tech-X Corporation
BUILD_DATE DEC 12 2007
FUNCTION	CUBLASINIT	0	0
FUNCTION	CUBLASSHUTDOWN	0	0
FUNCTION	CUBLASGETERROR	0	0
FUNCTION	CUBLASALLOC	1	3
FUNCTION	CUBLASFREE	1	1
FUNCTION 	CUBLASSETVECTOR 6	6
FUNCTION 	CUBLASGETVECTOR 6	6
FUNCTION 	CUBLASSETMATRIX 7	7
FUNCTION 	CUBLASGETMATRIX 7	7
FUNCTION 	CUBLASISAMAX	3	3
FUNCTION	CUBLASSASUM	3	3
FUNCTION	CUBLASSAXPY	6	6 
FUNCTION	CUBLASSCOPY	5	5
FUNCTION	CUBLASSDOT	5	5
FUNCTION	CUBLASSNRM2	3	3
FUNCTION	CUBLASSROT	7	7
FUNCTION	CUBLASSROTG	4	4
FUNCTION	CUBLASSROTM	6	6
FUNCTION	CUBLASSROTMG	5	5
FUNCTION	CUBLASSCAL	4	4
FUNCTION	CUBLASSSWAP	5	5

FUNCTION	CUBLASCAXPY	6	6 
FUNCTION	CUBLASCCOPY	5	5
FUNCTION	CUBLASCDOTU	5	5
FUNCTION	CUBLASCSCAL	4	4
FUNCTION	CUBLASCSSCAL	4	4
FUNCTION	CUBLASCSWAP	5	5
FUNCTION	CUBLASSCASUM	3	3

FUNCTION 	CUBLASSGBMV   	13	13
FUNCTION	CUBLASSGEMV	11	11
FUNCTION	CUBLASSGER	9	9
FUNCTION	CUBLASSSBMV	11	11
FUNCTION        CUBLASSSPMV	9	9
FUNCTION	CUBLASSSPR	6	6
FUNCTION        CUBLASSSPR2	8	8
FUNCTION        CUBLASSSYMV	10	10
FUNCTION        CUBLASSSYR	7	7
FUNCTION        CUBLASSSYR2	9	9
FUNCTION        CUBLASSTBMV	9	9
FUNCTION   	CUBLASSTBSV	9	9
FUNCTION        CUBLASSTPMV	7	7
FUNCTION        CUBLASSTPSV	7	7
FUNCTION        CUBLASSTRMV	8	8
FUNCTION        CUBLASSTRSV	8	8
FUNCTION	CUBLASSGEMM	13	13
FUNCTION     	CUBLASSSYMM	12	12
FUNCTION    	CUBLASSSYRK	10	10
FUNCTION    	CUBLASSSYR2K	12	12
FUNCTION    	CUBLASSTRMM	11	11
FUNCTION    	CUBLASSTRSM	11	11
FUNCTION    	CUBLASGEMM 	13	13

FUNCTION        CUFFTPLAN1D     4	4
FUNCTION        CUFFTPLAN2D     4	4
FUNCTION        CUFFTPLAN3D     5	5
FUNCTION        CUFFTDESTROY    1	1
FUNCTION        CUFFTEXECC2C    4	4
FUNCTION        CUFFTEXECR2C    3	3
FUNCTION        CUFFTEXECC2R    3	3

FUNCTION        CUDAGETDEVICECOUNT        1    1
FUNCTION        CUDAGETDEVICEPROPERTIES   2    2
FUNCTION        CUDASETDEVICE             1    1
FUNCTION        CUDAGETDEVICE             1    1

FUNCTION        CUDATHREADSYNCHRONIZE     0    0
FUNCTION        CUDATHREADEXIT            0    0 
 
FUNCTION 	CUDAMALLOC		  2    2
FUNCTION        CUDAMALLOCPITCH           4    4
FUNCTION        CUDAFREE                  1    1
FUNCTION        CUDAMALLOCARRAY           4    4
FUNCTION        CUDAFREEARRAY             1    1
FUNCTION        CUDAMALLOCHOST            2    2 
FUNCTION        CUDAFREEHOST              1    1 
FUNCTION        CUDAMEMSET                3    3 
FUNCTION        CUDAMEMSET2D              5    5 
FUNCTION        CUDAMEMCPY                4    4 
FUNCTION        CUDAMEMCPY2D              7    7 

FUNCTION        CUDAGLREGISTERBUFFEROBJECT 1    1 
FUNCTION        CUDAGLMAPBUFFEROBJECT     2    2 
FUNCTION        CUDAGLUNMAPBUFFEROBJECT   1    1 
FUNCTION        CUDAGLUNREGISTERBUFFEROBJECT 1    1 

FUNCTION        GPUSQRTF                  3    3 
FUNCTION        GPUEXPF                   3    3 
FUNCTION        GPUEXP2F                  3    3 
FUNCTION        GPUEXP10F                 3    3 
FUNCTION        GPULOGF                   3    3 
FUNCTION        GPULOG2F                  3    3 
FUNCTION        GPULOG10F                 3    3 
FUNCTION        GPULOG1PF                 3    3 
FUNCTION        GPUSINF                   3    3 
FUNCTION        GPUCOSF                   3    3 
FUNCTION        GPUTANF                   3    3 
FUNCTION        GPUASINF                  3    3 
FUNCTION        GPUACOSF                  3    3 
FUNCTION        GPUATANF                  3    3 
FUNCTION        GPUERFF                   3    3 
FUNCTION        GPULGAMMAF                3    3 
FUNCTION        GPUTGAMMAF                3    3 
FUNCTION        GPULOGBF                  3    3 
FUNCTION        GPUTRUNCF                 3    3 
FUNCTION        GPUROUNDF                 3    3 
FUNCTION        GPURINTF                  3    3 
FUNCTION        GPUNEARBYINTF             3    3 
FUNCTION        GPUCEILF                  3    3 
FUNCTION        GPUFLOORF                 3    3 
FUNCTION        GPULRINTF                 3    3 
FUNCTION        GPULROUNDF                3    3 
FUNCTION        GPUSIGNBITF               3    3 
FUNCTION        GPUISINFF                 3    3 
FUNCTION        GPUISNANF                 3    3 
FUNCTION        GPUISFINITEF              3    3 
FUNCTION        GPUFABSF                  3    3 

FUNCTION        GPUSQRTD                  3    3
FUNCTION        GPUEXPD                   3    3
FUNCTION        GPUEXP2D                  3    3
FUNCTION        GPUEXP10D                 3    3
FUNCTION        GPULOGD                   3    3
FUNCTION        GPULOG2D                  3    3
FUNCTION        GPULOG10D                 3    3
FUNCTION        GPULOG1PD                 3    3
FUNCTION        GPUSIND                   3    3
FUNCTION        GPUCOSD                   3    3
FUNCTION        GPUTAND                   3    3
FUNCTION        GPUASIND                  3    3
FUNCTION        GPUACOSD                  3    3
FUNCTION        GPUATAND                  3    3
FUNCTION        GPUERFD                   3    3
FUNCTION        GPULGAMMAD                3    3
FUNCTION        GPUTGAMMAD                3    3
FUNCTION        GPULOGBD                  3    3
FUNCTION        GPUTRUNCD                 3    3
FUNCTION        GPUROUNDD                 3    3
FUNCTION        GPURINTD                  3    3
FUNCTION        GPUNEARBYINTD             3    3
FUNCTION        GPUCEILD                  3    3
FUNCTION        GPUFLOORD                 3    3
FUNCTION        GPULRINTD                 3    3
FUNCTION        GPULROUNDD                3    3
FUNCTION        GPUSIGNBITD               3    3
FUNCTION        GPUISINFD                 3    3
FUNCTION        GPUISNAND                 3    3
FUNCTION        GPUISFINITED              3    3
FUNCTION        GPUFABSD                  3    3


FUNCTION        GPUSQRTC                  3    3
FUNCTION        GPUEXPC                   3    3
FUNCTION        GPUEXP2C                  3    3
FUNCTION        GPUEXP10C                 3    3
FUNCTION        GPULOGC                   3    3
FUNCTION        GPULOG2C                  3    3
FUNCTION        GPULOG10C                 3    3
FUNCTION        GPULOG1PC                 3    3
FUNCTION        GPUSINC                   3    3
FUNCTION        GPUCOSC                   3    3
FUNCTION        GPUTANC                   3    3
FUNCTION        GPUASINC                  3    3
FUNCTION        GPUACOSC                  3    3
FUNCTION        GPUATANC                  3    3
FUNCTION        GPUERFC                   3    3
FUNCTION        GPULGAMMAC                3    3
FUNCTION        GPUTGAMMAC                3    3
FUNCTION        GPULOGBC                  3    3
FUNCTION        GPUTRUNCC                 3    3
FUNCTION        GPUROUNDC                 3    3
FUNCTION        GPURINTC                  3    3
FUNCTION        GPUNEARBYINTC             3    3
FUNCTION        GPUCEILC                  3    3
FUNCTION        GPUFLOORC                 3    3
FUNCTION        GPULRINTC                 3    3
FUNCTION        GPULROUNDC                3    3
FUNCTION        GPUSIGNBITC               3    3
FUNCTION        GPUISINFC                 3    3
FUNCTION        GPUISNANC                 3    3
FUNCTION        GPUISFINITEC              3    3
FUNCTION        GPUFABSC                  3    3


FUNCTION        GPUSQRTZ                  3    3
FUNCTION        GPUEXPZ                   3    3
FUNCTION        GPUEXP2Z                  3    3
FUNCTION        GPUEXP10Z                 3    3
FUNCTION        GPULOGZ                   3    3
FUNCTION        GPULOG2Z                  3    3
FUNCTION        GPULOG10Z                 3    3
FUNCTION        GPULOG1PZ                 3    3
FUNCTION        GPUSINZ                   3    3
FUNCTION        GPUCOSZ                   3    3
FUNCTION        GPUTANZ                   3    3
FUNCTION        GPUASINZ                  3    3
FUNCTION        GPUACOSZ                  3    3
FUNCTION        GPUATANZ                  3    3
FUNCTION        GPUERFZ                   3    3
FUNCTION        GPULGAMMAZ                3    3
FUNCTION        GPUTGAMMAZ                3    3
FUNCTION        GPULOGBZ                  3    3
FUNCTION        GPUTRUNCZ                 3    3
FUNCTION        GPUROUNDZ                 3    3
FUNCTION        GPURINTZ                  3    3
FUNCTION        GPUNEARBYINTZ             3    3
FUNCTION        GPUCEILZ                  3    3
FUNCTION        GPUFLOORZ                 3    3
FUNCTION        GPULRINTZ                 3    3
FUNCTION        GPULROUNDZ                3    3
FUNCTION        GPUSIGNBITZ               3    3
FUNCTION        GPUISINFZ                 3    3
FUNCTION        GPUISNANZ                 3    3
FUNCTION        GPUISFINITEZ              3    3
FUNCTION        GPUFABSZ                  3    3

FUNCTION        GPUSQRTFAT                7    7 
FUNCTION        GPUEXPFAT                 7    7 
FUNCTION        GPUEXP2FAT                7    7 
FUNCTION        GPUEXP10FAT               7    7 
FUNCTION        GPULOGFAT                 7    7 
FUNCTION        GPULOG2FAT                7    7 
FUNCTION        GPULOG10FAT               7    7 
FUNCTION        GPULOG1PFAT               7    7 
FUNCTION        GPUSINFAT                 7    7 
FUNCTION        GPUCOSFAT                 7    7 
FUNCTION        GPUTANFAT                 7    7 
FUNCTION        GPUASINFAT                7    7 
FUNCTION        GPUACOSFAT                7    7 
FUNCTION        GPUATANFAT                7    7 
FUNCTION        GPUERFFAT                 7    7 
FUNCTION        GPULGAMMAFAT              7    7 
FUNCTION        GPUTGAMMAFAT              7    7 
FUNCTION        GPULOGBFAT                7    7 
FUNCTION        GPUTRUNCFAT               7    7 
FUNCTION        GPUROUNDFAT               7    7 
FUNCTION        GPURINTFAT                7    7 
FUNCTION        GPUNEARBYINTFAT           7    7 
FUNCTION        GPUCEILFAT                7    7 
FUNCTION        GPUFLOORFAT               7    7 
FUNCTION        GPULRINTFAT               7    7 
FUNCTION        GPULROUNDFAT              7    7 
FUNCTION        GPUSIGNBITFAT             7    7 
FUNCTION        GPUISINFFAT               7    7 
FUNCTION        GPUISNANFAT               7    7 
FUNCTION        GPUISFINITEFAT            7    7 
FUNCTION        GPUFABSFAT                7    7 

FUNCTION        GPUSQRTDAT                7    7
FUNCTION        GPUEXPDAT                 7    7
FUNCTION        GPUEXP2DAT                7    7
FUNCTION        GPUEXP10DAT               7    7
FUNCTION        GPULOGDAT                 7    7
FUNCTION        GPULOG2DAT                7    7
FUNCTION        GPULOG10DAT               7    7
FUNCTION        GPULOG1PDAT               7    7
FUNCTION        GPUSINDAT                 7    7
FUNCTION        GPUCOSDAT                 7    7
FUNCTION        GPUTANDAT                 7    7
FUNCTION        GPUASINDAT                7    7
FUNCTION        GPUACOSDAT                7    7
FUNCTION        GPUATANDAT                7    7
FUNCTION        GPUERFDAT                 7    7
FUNCTION        GPULGAMMADAT              7    7
FUNCTION        GPUTGAMMADAT              7    7
FUNCTION        GPULOGBDAT                7    7
FUNCTION        GPUTRUNCDAT               7    7
FUNCTION        GPUROUNDDAT               7    7
FUNCTION        GPURINTDAT                7    7
FUNCTION        GPUNEARBYINTDAT           7    7
FUNCTION        GPUCEILDAT                7    7
FUNCTION        GPUFLOORDAT               7    7
FUNCTION        GPULRINTDAT               7    7
FUNCTION        GPULROUNDDAT              7    7
FUNCTION        GPUSIGNBITDAT             7    7
FUNCTION        GPUISINFDAT               7    7
FUNCTION        GPUISNANDAT               7    7
FUNCTION        GPUISFINITEDAT            7    7
FUNCTION        GPUFABSDAT                7    7

FUNCTION        GPUSQRTCAT                7    7
FUNCTION        GPUEXPCAT                 7    7
FUNCTION        GPUEXP2CAT                7    7
FUNCTION        GPUEXP10CAT               7    7
FUNCTION        GPULOGCAT                 7    7
FUNCTION        GPULOG2CAT                7    7
FUNCTION        GPULOG10CAT               7    7
FUNCTION        GPULOG1PCAT               7    7
FUNCTION        GPUSINCAT                 7    7
FUNCTION        GPUCOSCAT                 7    7
FUNCTION        GPUTANCAT                 7    7
FUNCTION        GPUASINCAT                7    7
FUNCTION        GPUACOSCAT                7    7
FUNCTION        GPUATANCAT                7    7
FUNCTION        GPUERFCAT                 7    7
FUNCTION        GPULGAMMACAT              7    7
FUNCTION        GPUTGAMMACAT              7    7
FUNCTION        GPULOGBCAT                7    7
FUNCTION        GPUTRUNCCAT               7    7
FUNCTION        GPUROUNDCAT               7    7
FUNCTION        GPURINTCAT                7    7
FUNCTION        GPUNEARBYINTCAT           7    7
FUNCTION        GPUCEILCAT                7    7
FUNCTION        GPUFLOORCAT               7    7
FUNCTION        GPULRINTCAT               7    7
FUNCTION        GPULROUNDCAT              7    7
FUNCTION        GPUSIGNBITCAT             7    7
FUNCTION        GPUISINFCAT               7    7
FUNCTION        GPUISNANCAT               7    7
FUNCTION        GPUISFINITECAT            7    7
FUNCTION        GPUFABSCAT                7    7

FUNCTION        GPUSQRTZAT                7    7
FUNCTION        GPUEXPZAT                 7    7
FUNCTION        GPUEXP2ZAT                7    7
FUNCTION        GPUEXP10ZAT               7    7
FUNCTION        GPULOGZAT                 7    7
FUNCTION        GPULOG2ZAT                7    7
FUNCTION        GPULOG10ZAT               7    7
FUNCTION        GPULOG1PZAT               7    7
FUNCTION        GPUSINZAT                 7    7
FUNCTION        GPUCOSZAT                 7    7
FUNCTION        GPUTANZAT                 7    7
FUNCTION        GPUASINZRT                7    7
FUNCTION        GPUACOSZAT                7    7
FUNCTION        GPUATANZAT                7    7
FUNCTION        GPUERFZAT                 7    7
FUNCTION        GPULGAMMAZAT              7    7
FUNCTION        GPUTGAMMAZAT              7    7
FUNCTION        GPULOGBZAT                7    7
FUNCTION        GPUTRUNCZAT               7    7
FUNCTION        GPUROUNDZAT               7    7
FUNCTION        GPURINTZAT                7    7
FUNCTION        GPUNEARBYINTZAT           7    7
FUNCTION        GPUCEILZAT                7    7
FUNCTION        GPUFLOORZAT               7    7
FUNCTION        GPULRINTZAT               7    7
FUNCTION        GPULROUNDZAT              7    7
FUNCTION        GPUSIGNBITZAT             7    7
FUNCTION        GPUISINFZAT               7    7
FUNCTION        GPUISNANZAT               7    7
FUNCTION        GPUISFINITEZAT            7    7
FUNCTION        GPUFABSZAT                7    7

FUNCTION        GPUADDF                   4    4
FUNCTION        GPUSUBF                   4    4
FUNCTION        GPUMULTF                  4    4
FUNCTION        GPUDIVF                   4    4

FUNCTION        GPUADDD                   4    4
FUNCTION        GPUSUBD                   4    4
FUNCTION        GPUMULTD                  4    4
FUNCTION        GPUDIVD                   4    4

FUNCTION        GPUADDC                   4    4
FUNCTION        GPUSUBC                   4    4
FUNCTION        GPUMULTC                  4    4
FUNCTION        GPUDIVC                   4    4

FUNCTION        GPUADDZ                   4    4
FUNCTION        GPUSUBZ                   4    4
FUNCTION        GPUMULTZ                  4    4
FUNCTION        GPUDIVZ                   4    4

FUNCTION        GPULTF                    4    4
FUNCTION        GPUGTF                    4    4
FUNCTION        GPULTEQF                  4    4
FUNCTION        GPUGTEQF                  4    4
FUNCTION        GPUEQF                    4    4
FUNCTION        GPUNEQF                   4    4

FUNCTION        GPULTD                    4    4
FUNCTION        GPUGTD                    4    4
FUNCTION        GPULTEQD                  4    4
FUNCTION        GPUGTEQD                  4    4
FUNCTION        GPUEQD                    4    4
FUNCTION        GPUNEQD                   4    4
 
FUNCTION        GPULTC                    4    4
FUNCTION        GPUGTC                    4    4
FUNCTION        GPULTEQC                  4    4
FUNCTION        GPUGTEQC                  4    4
FUNCTION        GPUEQC                    4    4
FUNCTION        GPUNEQC                   4    4

FUNCTION        GPULTZ                    4    4
FUNCTION        GPUGTZ                    4    4
FUNCTION        GPULTEQZ                  4    4
FUNCTION        GPUGTEQZ                  4    4
FUNCTION        GPUEQZ                    4    4
FUNCTION        GPUNEQZ                   4    4

FUNCTION        GPUADDFAT                 7    7
FUNCTION        GPUSUBFAT                 7    7
FUNCTION        GPUMULTFAT                7    7
FUNCTION        GPUDIVFAT                 7    7

FUNCTION        GPUADDDAT                 7    7
FUNCTION        GPUSUBDAT                 7    7
FUNCTION        GPUMULTDAT                7    7
FUNCTION        GPUDIVDAT                 7    7

FUNCTION        GPUADDCAT                 7    7
FUNCTION        GPUSUBCAT                 7    7
FUNCTION        GPUMULTCAT                7    7
FUNCTION        GPUDIVCAT                 7    7

FUNCTION        GPUADDZAT                 7    7
FUNCTION        GPUSUBZAT                 7    7
FUNCTION        GPUMULTZAT                7    7
FUNCTION        GPUDIVZAT                 7    7

FUNCTION        GPULTFAT                  7    7
FUNCTION        GPUGTFAT                  7    7
FUNCTION        GPULTEQFAT                7    7
FUNCTION        GPUGTEQFAT                7    7
FUNCTION        GPUEQFAT                  7    7
FUNCTION        GPUNEQFAT                 7    7

FUNCTION        GPULTDAT                  7    7
FUNCTION        GPUGTDAT                  7    7
FUNCTION        GPULTEQDAT                7    7
FUNCTION        GPUGTEQDAT                7    7
FUNCTION        GPUEQDAT                  7    7
FUNCTION        GPUNEQDAT                 7    7

FUNCTION        GPULTCAT                  7    7
FUNCTION        GPUGTCAT                  7    7
FUNCTION        GPULTEQCAT                7    7
FUNCTION        GPUGTEQCAT                7    7
FUNCTION        GPUEQCAT                  7    7
FUNCTION        GPUNEQCAT                 7    7

FUNCTION        GPULTZAT                  7    7
FUNCTION        GPUGTZAT                  7    7
FUNCTION        GPULTEQZAT                7    7
FUNCTION        GPUGTEQZAT                7    7
FUNCTION        GPUEQZAT                  7    7
FUNCTION        GPUNEQZAT                 7    7

FUNCTION        GPUFLOATTODOUBLE          3    3
FUNCTION        GPUDOUBLETOFLOAT          3    3 
FUNCTION        GPUFLOATTOCOMPLEXREAL     3    3
FUNCTION        GPUCOMPLEXREALTOFLOAT     3    3
FUNCTION        GPUFLOATTOCOMPLEXIMAG     3    3
FUNCTION        GPUCOMPLEXIMAGTOFLOAT     3    3
FUNCTION        GPUFLOATTODCOMPLEXREAL    3    3
FUNCTION        GPUDCOMPLEXREALTOFLOAT    3    3 
FUNCTION        GPUFLOATTODCOMPLEXIMAG    3    3
FUNCTION        GPUDCOMPLEXIMAGTOFLOAT    3    3 
FUNCTION        GPUDOUBLETOCOMPLEXREAL    3    3
FUNCTION        GPUCOMPLEXREALTODOUBLE    3    3
FUNCTION        GPUDOUBLETOCOMPLEXIMAG    3    3
FUNCTION        GPUCOMPLEXIMAGTODOUBLE    3    3
FUNCTION        GPUDOUBLETODCOMPLEXREAL   3    3
FUNCTIOn        GPUDCOMPLEXREALTODOUBLE   3    3
FUNCTION        GPUDOUBLETODCOMPLEXIMAG   3    3
FUNCTION        GPUDCOMPLEXIMAGTODOUBLE   3    3

FUNCTION        GPUINTERPOLATE1DF         5    5
FUNCTION        GPUINTERPOLATE2DF         7    7
FUNCTION        GPUFINDGENF               2    2
FUNCTION        GPUDINDGEND               2    2
FUNCTION        GPUCINDGENC               2    2
FUNCTION        GPUDCINDGENZ              2    2
FUNCTION        GPUTOTALF                 3    3
FUNCTION        GPUMINF                   4    4
FUNCTION        GPUMAXF                   4    4
FUNCTION        GPUCONGRID1DF             5    5
FUNCTION        GPUCONGRID2DF             7    7
FUNCTION        GPUSUBSCRIPTF             4    4
FUNCTION        GPUSUBSCRIPTLHSF          4    4
FUNCTION        GPUPREFIXSUMF             3    3
FUNCTION        GPUWHEREF                 4    4

FUNCTION        GPUMTF                    2    2
FUNCTION        GPUBOXMULLERF             2    2
FUNCTION        GPUSEEDMTF                1    1

FUNCTION        GPUPOISSON                4    4 
FUNCTION        GPUBRMBREMCROSSF          5    5
FUNCTION        GPUBRMFINNERF             5    5
FUNCTION        GPUGAULEGF                6    6

