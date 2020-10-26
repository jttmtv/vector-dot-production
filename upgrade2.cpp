#include <immintrin.h>//Need to include header files
float simd_dot(const float* x, const float* y, const size_t& len) {
	float inner_prod = 0.0f;
	__m256 X, Y; //Declare two variables stored in the 256-bit special register of SSE
	__m256 acc = _mm256_setzero_ps(); //Declare a variable stored in a 256-bit special register of SSE to store the result of X+Y, the initial value is 0
	float sum[8] = { 0 };//Parameters for storing intermediate variables

	size_t i;
	for (i = 0; i + 8 < len; i += 8) {//256-bit special register, which can handle 8 groups of 32-bit variable operations at one time
		X = _mm256_load_ps(x + i); //Load x into X (because 256 bits can store 8 32-bit data, so by default, load 8 consecutive parameters at a time)
		Y = _mm256_load_ps(y + i);//Same as above
		acc = _mm256_add_ps(acc, _mm256_mul_ps(X, Y));//x*y, sum x1*y1 in each round, sum x2*y2, sum x3*y3, sum x4*y4, sum x5*y5, sum x6*y6, sum x7*y7, x8 *y8 sum, the final 8 sums are placed in the 256-bit register of acc.
	}
	_mm256_store_ps(sum, acc); //Load 8 32-bit data in acc into memory
	inner_prod = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];//Dot product sum

	//Just deal with the dot multiplication of elements that are multiples of the first 8. If len is not a multiple of 8, then there is a tail to be processed
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];//Continue to accumulate the tail product
	}
	return inner_prod;
}
