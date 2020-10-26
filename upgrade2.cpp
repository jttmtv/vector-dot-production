float simd_dot(const float* x, const float* y, const size_t& len) {
	float inner_prod = 0.0f;
	__m256 X, Y; //声明两个存放在SSE的256位专用寄存器的变量
	__m256 acc = _mm256_setzero_ps(); // 声明一个存放在SSE的256位专用寄存器的变量，用来存放X+Y的结果，初始值为0
	float sum[8] = { 0 };//存放中间变量的参数

	size_t i;
	for (i = 0; i + 8 < len; i += 8) {//256位专用寄存器，一次性可以处理8组32位变量的运算
		X = _mm256_load_ps(x + i); // 将x加载到X（由于256位可以存放8个32位数据，所以默认一次加载连续的8个参数）
		Y = _mm256_load_ps(y + i);//同上
		acc = _mm256_add_ps(acc, _mm256_mul_ps(X, Y));//x*y，每轮的x1*y1求和，x2*y2求和，x3*y3求和，x4*y4求和，x5*y5求和，x6*y6求和，x7*y7求和，x8*y8求和，最终产生的8个和，放在acc的256位寄存器中。
	}
	_mm256_store_ps(sum, acc); // 将acc中的8个32位的数据加载进内存
	inner_prod = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];//点乘求和

	// 刚才只是处理了前8的倍数个元素的点乘，如果len不是8的倍数，那么还有尾部要处理
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];//继续累加尾部的乘积
	}
	return inner_prod;
}
