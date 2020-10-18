#include <intrin.h> //需包含的头文件

float simd_dot(const float* x, const float* y, const int& len) {
	float inner_prod = 0.0f;
	__m128 X, Y; //声明两个存放在SSE的128位专用寄存器的变量
	__m128 acc = _mm_setzero_ps(); // 声明一个存放在SSE的128位专用寄存器的变量，用来存放X+Y的结果，初始值为0
	float temp[4];//存放中间变量的参数

	long i;
	for (i = 0; i + 4 < len; i += 4) {//128位专用寄存器，一次性可以处理4组32位变量的运算
		X = _mm_loadu_ps(x + i); // 将x加载到X（由于128位可以存放四个32位数据，所以默认一次加载连续的4个参数）
		Y = _mm_loadu_ps(y + i);//同上
		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));//x*y，每轮的x1*y1求和，x2*y2求和，x3*y3求和，x4*y4求和,最终产生的四个和，放在acc的128位寄存器中。
	}
	_mm_storeu_ps(&temp[0], acc); // 将acc中的4个32位的数据加载进内存
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];//点乘求和

	// 刚才只是处理了前4的倍数个元素的点乘，如果len不是4的倍数，那么还有个小尾巴要处理一下
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];//继续累加小尾巴的乘积
	}
	return inner_prod;//大功告成
}