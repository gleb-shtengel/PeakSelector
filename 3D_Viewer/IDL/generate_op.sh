#/bin/sh

sed -e"s/BINARY_OP_NAME/Add/" -e"s/BINARY_OP_TEXT/Add two vectors/" -e"s/BINARY_OP/+/" gpuBinaryOp.pro.sed > lib/gpuadd.pro
sed -e"s/BINARY_OP_NAME/Sub/" -e"s/BINARY_OP_TEXT/Subtract two vectors/" -e"s/BINARY_OP/-/" gpuBinaryOp.pro.sed > lib/gpusub.pro
sed -e"s/BINARY_OP_NAME/Mult/" -e"s/BINARY_OP_TEXT/Multiply two vectors/" -e"s/BINARY_OP/*/" gpuBinaryOp.pro.sed > lib/gpumult.pro
sed -e"s/BINARY_OP_NAME/Div/" -e"s/BINARY_OP_TEXT/Divide two vectors/" -e"s/BINARY_OP/\//" gpuBinaryOp.pro.sed > lib/gpudiv.pro

sed -e"s/REL_OP_NAME/Eq/" -e"s/REL_OP_TEXT/Compare two vetors for EQ/" -e"s/REL_OP/EQ/" gpuRelationalOp.pro.sed > lib/gpueq.pro
sed -e"s/REL_OP_NAME/Neq/" -e"s/REL_OP_TEXT/Compare two vetors for NEQ/" -e"s/REL_OP/NE/" gpuRelationalOp.pro.sed > lib/gpuneq.pro
sed -e"s/REL_OP_NAME/Lt/" -e"s/REL_OP_TEXT/Compare two vetors for LT/" -e"s/REL_OP/LT/" gpuRelationalOp.pro.sed > lib/gpult.pro
sed -e"s/REL_OP_NAME/Gt/" -e"s/REL_OP_TEXT/Compare two vetors for GT/" -e"s/REL_OP/GT/" gpuRelationalOp.pro.sed > lib/gpugt.pro
sed -e"s/REL_OP_NAME/LtEq/" -e"s/REL_OP_TEXT/Compare two vetors for LTEQ/" -e"s/REL_OP/LE/" gpuRelationalOp.pro.sed > lib/gpulteq.pro
sed -e"s/REL_OP_NAME/GtEq/" -e"s/REL_OP_TEXT/Compare two vetors for GTEQ/" -e"s/REL_OP/GE/" gpuRelationalOp.pro.sed > lib/gpugteq.pro

sed -e"s/UNARY_OP_NAME/Sqrt/" -e"s/UNARY_OP_TEXT/Compute the sqrt of a vector/" -e"s/UNARY_OP/sqrt/" gpuUnaryOp.pro.sed > lib/gpusqrt.pro
sed -e"s/UNARY_OP_NAME/Exp/" -e"s/UNARY_OP_TEXT/Compute the exp of a vector/" -e"s/UNARY_OP/exp/" gpuUnaryOp.pro.sed > lib/gpuexp.pro
sed -e"s/UNARY_OP_NAME/Log/" -e"s/UNARY_OP_TEXT/Compute the log of a vector/" -e"s/UNARY_OP/alog/" gpuUnaryOp.pro.sed > lib/gpulog.pro
sed -e"s/UNARY_OP_NAME/Log10/" -e"s/UNARY_OP_TEXT/Compute the log10 of a vector/" -e"s/UNARY_OP/alog10/" gpuUnaryOp.pro.sed > lib/gpulog10.pro
sed -e"s/UNARY_OP_NAME/Sin/" -e"s/UNARY_OP_TEXT/Compute the sin of a vector/" -e"s/UNARY_OP/sin/" gpuUnaryOp.pro.sed > lib/gpusin.pro
sed -e"s/UNARY_OP_NAME/Cos/" -e"s/UNARY_OP_TEXT/Compute the cos of a vector/" -e"s/UNARY_OP/cos/" gpuUnaryOp.pro.sed > lib/gpucos.pro
sed -e"s/UNARY_OP_NAME/Tan/" -e"s/UNARY_OP_TEXT/Compute the tan of a vector/" -e"s/UNARY_OP/tan/" gpuUnaryOp.pro.sed > lib/gputan.pro
sed -e"s/UNARY_OP_NAME/Asin/" -e"s/UNARY_OP_TEXT/Compute the asin of a vector/" -e"s/UNARY_OP/asin/" gpuUnaryOp.pro.sed > lib/gpuasin.pro
sed -e"s/UNARY_OP_NAME/Acos/" -e"s/UNARY_OP_TEXT/Compute the acos of a vector/" -e"s/UNARY_OP/acos/" gpuUnaryOp.pro.sed > lib/gpuacos.pro
sed -e"s/UNARY_OP_NAME/Atan/" -e"s/UNARY_OP_TEXT/Compute the atan of a vector/" -e"s/UNARY_OP/atan/" gpuUnaryOp.pro.sed > lib/gpuatan.pro
sed -e"s/UNARY_OP_NAME/Erf/" -e"s/UNARY_OP_TEXT/Compute the erf of a vector/" -e"s/UNARY_OP/erf/" gpuUnaryOp.pro.sed > lib/gpuerf.pro
sed -e"s/UNARY_OP_NAME/Lgamma/" -e"s/UNARY_OP_TEXT/Compute the log gamma function of a vector/" -e"s/UNARY_OP/lngamma/" gpuUnaryOp.pro.sed > lib/gpulgamma.pro
sed -e"s/UNARY_OP_NAME/Tgamma/" -e"s/UNARY_OP_TEXT/Compute the gamma function of a vector/" -e"s/UNARY_OP/gamma/" gpuUnaryOp.pro.sed > lib/gputgamma.pro
sed -e"s/UNARY_OP_NAME/Trunc/" -e"s/UNARY_OP_TEXT/Compute the trunc function of a vector/" -e"s/UNARY_OP/trunc/" gpuUnaryOp.pro.sed > lib/gputrunc.pro
sed -e"s/UNARY_OP_NAME/Round/" -e"s/UNARY_OP_TEXT/Round a vector/" -e"s/UNARY_OP/round/" gpuUnaryOp.pro.sed > lib/gpuround.pro
sed -e"s/UNARY_OP_NAME/Rint/" -e"s/UNARY_OP_TEXT/Rint a vector/" -e"s/UNARY_OP/round/" gpuUnaryOp.pro.sed > lib/gpurint.pro
sed -e"s/UNARY_OP_NAME/Floor/" -e"s/UNARY_OP_TEXT/Return the floor of a vector/" -e"s/UNARY_OP/floor/" gpuUnaryOp.pro.sed > lib/gpufloor.pro
sed -e"s/UNARY_OP_NAME/Ceil/" -e"s/UNARY_OP_TEXT/Return the ceil of a vector/" -e"s/UNARY_OP/ceil/" gpuUnaryOp.pro.sed > lib/gpuceil.pro
sed -e"s/UNARY_OP_NAME/Abs/" -e"s/UNARY_OP_TEXT/Return the absolute value of a vector/" -e"s/UNARY_OP/abs/" gpuUnaryOp.pro.sed > lib/gpuabs.pro


