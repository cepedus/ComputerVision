#include <iostream>
#include <vector>

using namespace std;

void swap1(int& a, int& b)
{
	int c = a;
	a = b;
	b = c;
}

template <typename T>
void swap2(T& a, T&b)
{
	T c = a;
	a = b;
	b = c;
}

struct Point1
{
	double x, y;
	Point1() {}
	Point1(double a, double b) {
		x = a; y = b;
	}
	double norm2() const {
		return x * x + y * y;
	}
	Point1 operator+(const Point1& b) const
	{
		return Point1(x + b.x, y + b.y);
	}
};
ostream& operator<<(ostream& out, const Point1& p)
{
	out << p.x << ',' << p.y;
	return out;
}

template<typename T>
struct Point2 {
	T x, y;
	Point2() {}
	Point2(T a, T b)
	{
		x = a; y = b;
	}
	T norm2() const
	{
		return x * x + y * y;
	}
	Point2 operator+(const Point2& b) const
	{
		return Point2(x + b.x, y + b.y);
	}
};
template<typename T>
ostream& operator<<(ostream& out, const Point2<T>& p)
{
	out << p.x << ',' << p.y; return out;
}

int main()
{
	int i = 1, j = 2;
	swap1(i, j);
	cout << i << " " << j << endl;
	swap2(i, j);
	cout << i << " " << j << endl;

	int n;
	cout << "Enter size: ";
	cin >> n;
	vector<int> T(n);
	for (int i = 0; i < n; i++)
		T[i] = i;
	T.push_back(123);
	cout << "New size is: " << T.size() << endl;

	for (int& t : T)
		t = t * 2;
	for (int t : T)
		cout << t << ' ';
	cout << endl;

	Point1 p(1, 2);
	cout << "|(" << p << ")|^2=" << p.norm2() << endl;
	cout << p + Point1(3, 4) << endl;

	Point2<double> q(1, 2);
	cout << "|(" << q << ")|^2=" << q.norm2() << endl;
	cout << q + Point2<double>(3, 4) << endl;

	return 0;
}
