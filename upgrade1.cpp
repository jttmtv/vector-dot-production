inline float vector_dot_product(float* v1, float* v2, int n)
{
	float sum = 0.0f;
	int a, b;
	a = n % 8;
	b = (n - a) / 8;
	for (size_t i = 0; i < n; i += 8)
	{
		sum += (v1[i] * v2[i]);
		sum += (v1[i + 1] * v2[i + 1]);
		sum += (v1[i + 2] * v2[i + 2]);
		sum += (v1[i + 3] * v2[i + 3]);
		sum += (v1[i + 4] * v2[i + 4]);
		sum += (v1[i + 5] * v2[i + 5]);
		sum += (v1[i + 6] * v2[i + 6]);
		sum += (v1[i + 7] * v2[i + 7]);
	}
	for (int i = n - 1; i > n - 1 - a; i--)
		sum += v1[i] * v2[i];
	return sum;
}
