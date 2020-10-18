float vector_dot_product(float* v1, float* v2, int n) {
	float sum1 = 0.0f, sum2 = 0.0f;
	int a, b;
	a = n % 4;
	b = (n - a) / 4;
	for (int i = 0; i < b; i++) {
		for (int j = i * 4; j < i * 4 + 4; j++)
			sum1 += v1[j] * v2[j];
		sum2 += sum1;
		sum1 = 0.0f;
	}
	for (int i = n - 1; i > n - 1 - a; i--)
		sum2 += v1[i] * v2[i];
	return sum2;
}
