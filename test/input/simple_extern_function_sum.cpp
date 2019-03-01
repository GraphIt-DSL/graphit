#include <vector>

int sum_list(std::vector<int> * v) {
	int sum = 0;
	for (auto const& value: *v)
		sum += value;
	return sum;
}
