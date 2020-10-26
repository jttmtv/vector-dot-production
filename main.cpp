#include <bits/stdc++.h>
using namespace std;

inline size_t Input_int();
inline float vector_dot_product(float*, float*, size_t);

int main()
{
	float res;
	srand((unsigned)time(NULL));
	size_t n;
	n = Input_int();
	float* vec1 = new float[n];
	float* vec2 = new float[n];
	for (size_t i = 0; i < n; i++) {
		vec1[i] = rand() / float(RAND_MAX);
	}
	for (size_t i = 0; i < n; i++) {
		vec2[i] = rand() / float(RAND_MAX);
	}
	clock_t start, end;
	start = clock();
	res = vector_dot_product(vec1, vec2, n);
	end = clock();
	printf("%f\n", res);
	printf("total time = %fms\n", (float)(end - start) * 1000 / CLOCKS_PER_SEC);
	delete[] vec1;
	delete[] vec2;
	return 0;
}

inline float vector_dot_product(float* v1, float* v2, size_t n) {
	float sum = 0.0f;
	for (size_t i = 0; i < n; i++) {
		sum += v1[i] * v2[i];
	}
	return sum;
}

inline size_t Input_int()
{
	string x;
	getline(cin, x);
	while (true)
	{
		bool flag = false;
		for (auto ch : x)
		{
			if (ch < '0' || ch>'9')
			{
				flag = true;
				break;
			}
		}
		cin.clear(); //Clear flow mark
		cin.sync();  //Empty stream
		if (flag)
		{
			cout << "illegal input ,try agin!" << endl;
			getline(cin, x);
		}
		else
			break;
	}
	return stoi(x);
}
